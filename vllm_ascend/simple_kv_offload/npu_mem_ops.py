"""Low-level NPU memory helpers: batched DMA transfers.

Mirrors :mod:`vllm.v1.simple_kv_offload.cuda_mem_ops` but uses the
Ascend ``aclrtMemcpyBatchAsync`` path exposed via
``torch.ops._C_ascend.swap_blocks_batch_indexed`` (see
``csrc/torch_binding.cpp``). Python submits cached tensor descriptors and
block ids; native code expands per-block copy addresses.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch

# Direction codes shared with csrc/torch_binding.cpp::swap_blocks_batch.
DIRECTION_H2D = 0
DIRECTION_D2H = 1


class BatchMemcpyParams(NamedTuple):
    """Pre-computed per-tensor descriptors for batched block copy."""

    src_bases_tensor: torch.Tensor  # [num_sub_tensors] int64
    dst_bases_tensor: torch.Tensor  # [num_sub_tensors] int64
    bpb_tensor: torch.Tensor  # [num_sub_tensors] int64
    num_sub_tensors: int
    direction: int  # DIRECTION_H2D or DIRECTION_D2H


class BlockIdWorkspace:
    """Reusable host-side block-id arrays for one memcpy submission."""

    def __init__(self) -> None:
        self._block_capacity = 0
        self.src_ids = np.empty(0, dtype=np.int64)
        self.dst_ids = np.empty(0, dtype=np.int64)
        self.src_tensor = torch.from_numpy(self.src_ids)
        self.dst_tensor = torch.from_numpy(self.dst_ids)

    def ensure_capacity(self, num_blocks: int) -> None:
        if num_blocks > self._block_capacity:
            self.src_ids = np.empty(num_blocks, dtype=np.int64)
            self.dst_ids = np.empty(num_blocks, dtype=np.int64)
            self.src_tensor = torch.from_numpy(self.src_ids)
            self.dst_tensor = torch.from_numpy(self.dst_ids)
            self._block_capacity = num_blocks

    def tensors(self, num_blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
        if num_blocks == self._block_capacity:
            return self.src_tensor, self.dst_tensor
        return self.src_tensor[:num_blocks], self.dst_tensor[:num_blocks]


def _ordered_tensors(caches: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    """Return values in insertion order (kept as a function for clarity)."""
    return list(caches.values())


def build_params(
    src_caches: dict[str, torch.Tensor],
    dst_caches: dict[str, torch.Tensor],
    direction: int,
) -> BatchMemcpyParams:
    """Build cached pointer/stride descriptors for all sub-tensors.

    Both ``src_caches`` and ``dst_caches`` must have identical keys and a
    matching ``[num_blocks, block_bytes]`` layout (already prepared by
    :class:`SimpleCPUOffloadNPUWorker.register_kv_caches`).
    """
    assert list(src_caches.keys()) == list(dst_caches.keys()), "src/dst cache key order must match"
    src_tensors = _ordered_tensors(src_caches)
    dst_tensors = _ordered_tensors(dst_caches)

    src_bases: list[int] = []
    dst_bases: list[int] = []
    bpb: list[int] = []
    src_num_blocks = src_tensors[0].shape[0]
    dst_num_blocks = dst_tensors[0].shape[0]
    for s, d in zip(src_tensors, dst_tensors):
        assert s.shape[0] == src_num_blocks, "src cache block counts must match"
        assert d.shape[0] == dst_num_blocks, "dst cache block counts must match"
        s_bpb = s.stride(0) * s.element_size()
        d_bpb = d.stride(0) * d.element_size()
        assert s_bpb == d_bpb, f"per-block bytes mismatch src={s_bpb} dst={d_bpb}"
        src_bases.append(s.data_ptr())
        dst_bases.append(d.data_ptr())
        bpb.append(s_bpb)

    src_base_array = np.array(src_bases, dtype=np.int64)
    dst_base_array = np.array(dst_bases, dtype=np.int64)
    bpb_array = np.array(bpb, dtype=np.int64)
    return BatchMemcpyParams(
        src_bases_tensor=torch.from_numpy(src_base_array),
        dst_bases_tensor=torch.from_numpy(dst_base_array),
        bpb_tensor=torch.from_numpy(bpb_array),
        num_sub_tensors=len(src_tensors),
        direction=direction,
    )


def copy_blocks(
    src_block_ids: list[int],
    dst_block_ids: list[int],
    params: BatchMemcpyParams,
    workspace: BlockIdWorkspace | None = None,
) -> None:
    """Issue a batched async DMA on the *current* NPU stream.

    The caller is expected to be inside a ``torch.npu.stream(...)``
    context so the issued copies bind to the dedicated transfer stream.
    """
    n = len(src_block_ids)
    if n == 0:
        return
    assert n == len(dst_block_ids), "src/dst block counts must match"

    if workspace is None:
        workspace = BlockIdWorkspace()
    workspace.ensure_capacity(n)

    src_ids = workspace.src_ids[:n]
    dst_ids = workspace.dst_ids[:n]
    src_ids[:] = src_block_ids
    dst_ids[:] = dst_block_ids
    src_id_tensor, dst_id_tensor = workspace.tensors(n)

    torch.ops._C_ascend.swap_blocks_batch_indexed(
        params.src_bases_tensor,
        params.dst_bases_tensor,
        params.bpb_tensor,
        src_id_tensor,
        dst_id_tensor,
        params.direction,
    )
