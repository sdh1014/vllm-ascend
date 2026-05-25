import torch
from vllm.triton_utils import tl, triton

MOE_WEIGHT_MASK_TRITON_MIN_ELEMENTS = 65536


def should_use_moe_weight_mask_triton(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
) -> bool:
    # Performance gate only. Re-measure with
    # tools/awq_validation/triton_moe_weight_mask_bench.py when model shape,
    # top_k, EP size, CANN, Triton, or device type changes.
    return (
        topk_weights.numel() >= MOE_WEIGHT_MASK_TRITON_MIN_ELEMENTS
        and topk_weights.device.type == "npu"
        and topk_ids.device.type == "npu"
        and expert_map.device.type == "npu"
    )


@triton.jit
def _mask_topk_weights_by_expert_map_kernel(
    topk_weights,
    topk_ids,
    expert_map,
    output,
    n_elements,
    n_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)

    for block_id in range(pid, n_blocks, num_programs):
        element_offsets = block_id * BLOCK_SIZE + offsets
        mask = element_offsets < n_elements
        expert_ids = tl.load(topk_ids + element_offsets, mask=mask, other=0)
        mapped_expert_ids = tl.load(expert_map + expert_ids, mask=mask, other=-1)
        weights = tl.load(topk_weights + element_offsets, mask=mask, other=0.0)
        masked_weights = tl.where(mapped_expert_ids != -1, weights, 0.0)
        tl.store(output + element_offsets, masked_weights, mask=mask)


def mask_topk_weights_by_expert_map_triton(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
) -> torch.Tensor:
    if topk_weights.dim() != 2 or topk_ids.dim() != 2:
        raise ValueError("MoE weight mask Triton path expects 2D topk tensors.")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "MoE weight mask Triton path expects matching topk shapes, "
            f"but got {tuple(topk_weights.shape)} and {tuple(topk_ids.shape)}."
        )
    if topk_weights.device.type != "npu" or topk_ids.device.type != "npu" or expert_map.device.type != "npu":
        raise ValueError("MoE weight mask Triton path expects NPU tensors.")

    topk_weights = topk_weights.contiguous()
    topk_ids = topk_ids.contiguous()
    expert_map = expert_map.contiguous()
    output = torch.empty_like(topk_weights)

    block_size = 1024
    n_elements = topk_weights.numel()
    n_blocks = triton.cdiv(n_elements, block_size)
    num_programs = min(n_blocks, 256)
    _mask_topk_weights_by_expert_map_kernel[(num_programs,)](
        topk_weights,
        topk_ids,
        expert_map,
        output,
        n_elements,
        n_blocks,
        BLOCK_SIZE=block_size,
    )
    return output
