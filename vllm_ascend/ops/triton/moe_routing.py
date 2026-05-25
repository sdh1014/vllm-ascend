import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _collect_active_expert_order_kernel(
    topk_ids,
    expert_order_workspace,
    expert_tokens,
    flat_size,
    first_expert,
    BLOCK_SIZE: tl.constexpr,
):
    local_expert = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    write_count = 0

    for block_start in range(0, flat_size, BLOCK_SIZE):
        flat_positions = block_start + offsets
        mask = flat_positions < flat_size
        expert_ids = tl.load(topk_ids + flat_positions, mask=mask, other=-1)
        active_mask = mask & ((expert_ids - first_expert) == local_expert)
        active_values = active_mask.to(tl.int32)
        ranks = tl.cumsum(active_values, axis=0) - 1
        active_count = tl.sum(active_values, axis=0)
        write_positions = write_count + ranks
        workspace_offsets = local_expert * flat_size + write_positions
        tl.store(
            expert_order_workspace + workspace_offsets,
            flat_positions.to(tl.int32),
            mask=active_mask,
        )
        write_count += active_count

    tl.store(expert_tokens + local_expert, write_count)


@triton.jit
def _compact_active_expert_order_kernel(
    expert_order_workspace,
    expert_offsets,
    expert_tokens,
    sorted_order,
    expanded_row_idx,
    flat_size,
    BLOCK_SIZE: tl.constexpr,
):
    local_expert = tl.program_id(axis=0)
    block_id = tl.program_id(axis=1)
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    count = tl.load(expert_tokens + local_expert)
    mask = offsets < count
    source_offsets = local_expert * flat_size + offsets
    flat_positions = tl.load(expert_order_workspace + source_offsets, mask=mask, other=0)
    output_offsets = tl.load(expert_offsets + local_expert) + offsets
    tl.store(sorted_order + output_offsets, flat_positions, mask=mask)
    tl.store(expanded_row_idx + flat_positions, output_offsets.to(tl.int32), mask=mask)


@triton.jit
def _gather_sorted_hidden_kernel(
    hidden_states,
    sorted_order,
    sorted_hidden_states,
    num_tiles,
    hidden_blocks,
    top_k,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    for tile_id in range(pid, num_tiles, num_programs):
        row_id = tile_id // hidden_blocks
        block_id = tile_id - row_id * hidden_blocks
        columns = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = columns < hidden_size
        flat_position = tl.load(sorted_order + row_id)
        token_id = flat_position // top_k
        values = tl.load(hidden_states + token_id * hidden_size + columns, mask=mask)
        tl.store(sorted_hidden_states + row_id * hidden_size + columns, values, mask=mask)


def npu_moe_init_routing_active_expert_triton(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    first_expert: int,
    last_expert: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if hidden_states.dim() != 2:
        raise ValueError("Triton active expert routing expects 2D hidden_states.")
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()
    if not topk_ids.is_contiguous():
        topk_ids = topk_ids.contiguous()

    _, top_k = topk_ids.shape
    flat_size = topk_ids.numel()
    hidden_size = hidden_states.shape[-1]
    expert_count = last_expert - first_expert

    expanded_row_idx = torch.zeros((flat_size,), dtype=torch.int32, device=topk_ids.device)
    expert_tokens = torch.empty((expert_count,), dtype=torch.int32, device=topk_ids.device)
    expert_order_workspace = torch.empty((expert_count, flat_size), dtype=torch.int32, device=topk_ids.device)
    block_size = 1024

    _collect_active_expert_order_kernel[(expert_count,)](
        topk_ids,
        expert_order_workspace,
        expert_tokens,
        flat_size,
        first_expert,
        BLOCK_SIZE=block_size,
    )

    selected_count = int(expert_tokens.sum().item())
    if selected_count == 0:
        return (
            torch.empty((0, hidden_size), dtype=hidden_states.dtype, device=hidden_states.device),
            expanded_row_idx,
            expert_tokens,
        )

    expert_offsets = torch.cumsum(expert_tokens, dim=0) - expert_tokens
    sorted_order = torch.empty((selected_count,), dtype=torch.int32, device=topk_ids.device)
    compact_grid = (expert_count, triton.cdiv(flat_size, block_size))
    _compact_active_expert_order_kernel[compact_grid](
        expert_order_workspace,
        expert_offsets,
        expert_tokens,
        sorted_order,
        expanded_row_idx,
        flat_size,
        BLOCK_SIZE=block_size,
    )

    sorted_hidden_states = torch.empty(
        (selected_count, hidden_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    hidden_block_size = 1024
    hidden_blocks = triton.cdiv(hidden_size, hidden_block_size)
    gather_tiles = selected_count * hidden_blocks
    gather_grid = (min(gather_tiles, 256),)
    _gather_sorted_hidden_kernel[gather_grid](
        hidden_states,
        sorted_order,
        sorted_hidden_states,
        gather_tiles,
        hidden_blocks,
        top_k,
        hidden_size=hidden_size,
        BLOCK_SIZE=hidden_block_size,
    )
    return sorted_hidden_states, expanded_row_idx, expert_tokens
