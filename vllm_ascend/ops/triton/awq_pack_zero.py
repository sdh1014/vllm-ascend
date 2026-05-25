import torch
from vllm.triton_utils import tl, triton

DEFAULT_BLOCK_SIZE = 1024


def triton_kernel_launchable(kernel: object | None = None) -> bool:
    if kernel is None:
        kernel = _awq_pack_zero_kernel
    return callable(getattr(kernel, "__getitem__", None))


@triton.jit
def _awq_pack_zero_kernel(
    qweight,
    qzeros,
    packed_weight,
    offset,
    weight_elements,
    offset_elements,
    weight_blocks,
    offset_blocks,
    output_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    element_offsets = tl.arange(0, BLOCK_SIZE)

    for block_id in range(pid, weight_blocks, num_programs):
        offsets = block_id * BLOCK_SIZE + element_offsets
        mask = offsets < weight_elements
        packed = tl.load(qweight + offsets, mask=mask, other=0)

        nibble_0 = ((packed >> 0) & 15) + 8
        nibble_1 = ((packed >> 16) & 15) + 8
        nibble_2 = ((packed >> 4) & 15) + 8
        nibble_3 = ((packed >> 20) & 15) + 8
        nibble_4 = ((packed >> 8) & 15) + 8
        nibble_5 = ((packed >> 24) & 15) + 8
        nibble_6 = ((packed >> 12) & 15) + 8
        nibble_7 = ((packed >> 28) & 15) + 8

        repacked = (
            ((nibble_0 & 15) << 0)
            | ((nibble_1 & 15) << 4)
            | ((nibble_2 & 15) << 8)
            | ((nibble_3 & 15) << 12)
            | ((nibble_4 & 15) << 16)
            | ((nibble_5 & 15) << 20)
            | ((nibble_6 & 15) << 24)
            | ((nibble_7 & 15) << 28)
        )
        tl.store(packed_weight + offsets, repacked, mask=mask)

    for block_id in range(pid, offset_blocks, num_programs):
        offsets = block_id * BLOCK_SIZE + element_offsets
        mask = offsets < offset_elements
        row = offsets // output_size
        output_col = offsets - row * output_size
        packed_col = output_col // 8
        packed_index = output_col - packed_col * 8
        source_index = (packed_index // 2) + ((packed_index & 1) * 4)
        shift = source_index * 4
        packed_zero = tl.load(qzeros + row * tl.cdiv(output_size, 8) + packed_col, mask=mask, other=0)
        zero = (packed_zero >> shift) & 15
        offset_value = 8 - zero
        tl.store(offset + offsets, offset_value.to(offset.dtype.element_ty), mask=mask)


def awq_pack_zero_triton(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    output_size: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    if qweight.dim() != 2 or qzeros.dim() != 2 or scales.dim() != 2:
        raise ValueError("AWQ fused pack-zero Triton path expects 2D tensors.")
    if qweight.device.type != "npu" or qzeros.device.type != "npu" or scales.device.type != "npu":
        raise ValueError("AWQ fused pack-zero Triton path expects NPU tensors.")
    if qweight.dtype != torch.int32 or qzeros.dtype != torch.int32:
        raise ValueError(f"AWQ qweight/qzeros must be int32, but got {qweight.dtype}/{qzeros.dtype}.")
    if scales.shape != (qzeros.shape[0], output_size):
        raise ValueError(
            "AWQ fused pack-zero Triton path expects scales shape "
            f"({qzeros.shape[0]}, {output_size}), but got {tuple(scales.shape)}."
        )
    if qzeros.shape[1] != triton.cdiv(output_size, 8):
        raise ValueError(
            "AWQ fused pack-zero Triton path expects qzeros shape "
            f"(*, {triton.cdiv(output_size, 8)}), but got {tuple(qzeros.shape)}."
        )

    if not triton_kernel_launchable():
        raise RuntimeError("Triton-Ascend fused pack-zero kernel is not launchable.")
    if not qweight.is_contiguous():
        qweight = qweight.contiguous()
    if not qzeros.is_contiguous():
        qzeros = qzeros.contiguous()

    packed_weight = torch.empty_like(qweight)
    offset = torch.empty(scales.shape, dtype=scales.dtype, device=scales.device)
    weight_elements = qweight.numel()
    offset_elements = offset.numel()
    weight_blocks = triton.cdiv(weight_elements, block_size)
    offset_blocks = triton.cdiv(offset_elements, block_size)
    num_programs = min(max(weight_blocks, offset_blocks), 256)
    _awq_pack_zero_kernel[(num_programs,)](
        qweight,
        qzeros,
        packed_weight,
        offset,
        weight_elements,
        offset_elements,
        weight_blocks,
        offset_blocks,
        output_size=output_size,
        BLOCK_SIZE=block_size,
    )
    return packed_weight, offset
