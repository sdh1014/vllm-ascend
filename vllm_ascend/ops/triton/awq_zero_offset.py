import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _awq_zero_offset_kernel(
    qzeros,
    output,
    n_elements,
    n_blocks,
    output_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)

    for block_id in range(pid, n_blocks, num_programs):
        element_offsets = block_id * BLOCK_SIZE + offsets
        mask = element_offsets < n_elements
        row = element_offsets // output_size
        output_col = element_offsets - row * output_size
        packed_col = output_col // 8
        packed_index = output_col - packed_col * 8
        source_index = (packed_index // 2) + ((packed_index & 1) * 4)
        shift = source_index * 4
        packed_zero = tl.load(qzeros + row * tl.cdiv(output_size, 8) + packed_col, mask=mask, other=0)
        zero = (packed_zero >> shift) & 15
        offset = 8 - zero
        tl.store(output + element_offsets, offset.to(output.dtype.element_ty), mask=mask)


def awq_zero_offset_triton(qzeros: torch.Tensor, scales: torch.Tensor, output_size: int) -> torch.Tensor:
    if qzeros.dim() != 2 or scales.dim() != 2:
        raise ValueError("AWQ zero offset Triton path expects 2D qzeros and scales.")
    if scales.shape != (qzeros.shape[0], output_size):
        raise ValueError(
            "AWQ zero offset Triton path expects scales shape "
            f"({qzeros.shape[0]}, {output_size}), but got {tuple(scales.shape)}."
        )
    if qzeros.device.type != "npu" or scales.device.type != "npu":
        raise ValueError("AWQ zero offset Triton path expects NPU tensors.")
    if qzeros.dtype != torch.int32:
        raise ValueError(f"AWQ qzeros must be int32, but got {qzeros.dtype}.")

    if not qzeros.is_contiguous():
        qzeros = qzeros.contiguous()

    output = torch.empty(scales.shape, dtype=scales.dtype, device=scales.device)
    n_elements = output.numel()
    block_size = 1024
    n_blocks = triton.cdiv(n_elements, block_size)
    num_programs = min(n_blocks, 256)
    _awq_zero_offset_kernel[(num_programs,)](
        qzeros,
        output,
        n_elements,
        n_blocks,
        output_size=output_size,
        BLOCK_SIZE=block_size,
    )
    return output
