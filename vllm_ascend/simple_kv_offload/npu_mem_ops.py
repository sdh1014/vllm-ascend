"""Low-level NPU memory helpers: batched DMA transfers.

Mirrors :mod:`vllm.v1.simple_kv_offload.cuda_mem_ops` but uses the
Ascend ``aclrtMemcpyBatchAsync`` path exposed via
``torch.ops._C_ascend.swap_blocks_batch`` (see
``csrc/torch_binding.cpp``).
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

    src_bases: np.ndarray  # [num_sub_tensors] int64 — data_ptr per tensor
    dst_bases: np.ndarray  # [num_sub_tensors] int64
    bpb: np.ndarray  # [num_sub_tensors] int64 — bytes per block
    src_block_ptrs: np.ndarray  # [num_sub_tensors, src_blocks] int64
    dst_block_ptrs: np.ndarray  # [num_sub_tensors, dst_blocks] int64
    num_sub_tensors: int
    direction: int  # DIRECTION_H2D or DIRECTION_D2H


class BatchMemcpyWorkspace:
    """Reusable host-side arrays for one batched memcpy submission."""

    def __init__(self) -> None:
        self._block_capacity = 0
        self._copy_capacity = 0
        self.src_ids = np.empty(0, dtype=np.int64)
        self.dst_ids = np.empty(0, dtype=np.int64)
        self.src_ptrs = np.empty(0, dtype=np.int64)
        self.dst_ptrs = np.empty(0, dtype=np.int64)
        self.sizes = np.empty(0, dtype=np.int64)
        self.src_tensor = torch.from_numpy(self.src_ptrs)
        self.dst_tensor = torch.from_numpy(self.dst_ptrs)
        self.size_tensor = torch.from_numpy(self.sizes)

    def ensure_capacity(self, num_blocks: int, num_copies: int) -> None:
        if num_blocks > self._block_capacity:
            self.src_ids = np.empty(num_blocks, dtype=np.int64)
            self.dst_ids = np.empty(num_blocks, dtype=np.int64)
            self._block_capacity = num_blocks

        if num_copies > self._copy_capacity:
            self.src_ptrs = np.empty(num_copies, dtype=np.int64)
            self.dst_ptrs = np.empty(num_copies, dtype=np.int64)
            self.sizes = np.empty(num_copies, dtype=np.int64)
            self.src_tensor = torch.from_numpy(self.src_ptrs)
            self.dst_tensor = torch.from_numpy(self.dst_ptrs)
            self.size_tensor = torch.from_numpy(self.sizes)
            self._copy_capacity = num_copies

    def tensors(self, num_copies: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if num_copies == self._copy_capacity:
            return self.src_tensor, self.dst_tensor, self.size_tensor
        return (
            self.src_tensor[:num_copies],
            self.dst_tensor[:num_copies],
            self.size_tensor[:num_copies],
        )


def _ordered_tensors(caches: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    """Return values in insertion order (kept as a function for clarity)."""
    return list(caches.values())


def _block_ptr_table(
    bases: np.ndarray,
    bpb: np.ndarray,
    num_blocks: int,
) -> np.ndarray:
    block_ids = np.arange(num_blocks, dtype=np.int64)
    return bases[:, None] + block_ids[None, :] * bpb[:, None]


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
        src_bases=src_base_array,
        dst_bases=dst_base_array,
        bpb=bpb_array,
        src_block_ptrs=_block_ptr_table(src_base_array, bpb_array, src_num_blocks),
        dst_block_ptrs=_block_ptr_table(dst_base_array, bpb_array, dst_num_blocks),
        num_sub_tensors=len(src_tensors),
        direction=direction,
    )


def copy_blocks(
    src_block_ids: list[int],
    dst_block_ids: list[int],
    params: BatchMemcpyParams,
    workspace: BatchMemcpyWorkspace | None = None,
) -> None:
    """Issue a batched async DMA on the *current* NPU stream.

    The caller is expected to be inside a ``torch.npu.stream(...)``
    context so the issued copies bind to the dedicated transfer stream.
    """
    n = len(src_block_ids)
    if n == 0:
        return
    assert n == len(dst_block_ids), "src/dst block counts must match"

    total_copies = params.num_sub_tensors * n
    if workspace is None:
        workspace = BatchMemcpyWorkspace()
    workspace.ensure_capacity(n, total_copies)

    src_ids = workspace.src_ids[:n]
    dst_ids = workspace.dst_ids[:n]
    src_ids[:] = src_block_ids
    dst_ids[:] = dst_block_ids

    # Layout: (num_sub_tensors, n) flattened, matching swap_blocks_batch.
    shape = (params.num_sub_tensors, n)
    src_matrix = workspace.src_ptrs[:total_copies].reshape(shape)
    dst_matrix = workspace.dst_ptrs[:total_copies].reshape(shape)
    size_matrix = workspace.sizes[:total_copies].reshape(shape)
    bpb_col = params.bpb[:, None]
    np.take(params.src_block_ptrs, src_ids, axis=1, out=src_matrix)
    np.take(params.dst_block_ptrs, dst_ids, axis=1, out=dst_matrix)
    size_matrix[...] = bpb_col

    batch_src, batch_dst, batch_sizes = workspace.tensors(total_copies)

    torch.ops._C_ascend.swap_blocks_batch(batch_src, batch_dst, batch_sizes, params.direction)
