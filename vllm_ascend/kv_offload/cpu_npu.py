from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)


@dataclass
class Transfer:
    job_id: int
    stream: Any
    start_event: Any
    end_event: Any
    num_bytes: int


def compute_sub_block_ptrs(
    block_ids: np.ndarray,
    block_size_factor: int,
    output: np.ndarray,
    tensor: torch.Tensor,
    skip_count: int = 0,
) -> None:
    """
    Compute byte pointers for sub-blocks in a canonical KV tensor.

    CPU offload blocks can be larger than NPU blocks. In that case a CPU block
    contains ``block_size_factor`` NPU-sized sub-blocks, and ``skip_count``
    selects the first sub-block needed for a partially aligned transfer.
    """
    assert skip_count < block_size_factor

    num_sub_blocks = len(output)
    base_ptr = tensor.data_ptr()
    row_stride = tensor.stride(0)

    if block_size_factor == 1:
        output[:] = base_ptr + block_ids[:num_sub_blocks] * row_stride
        return

    assert tensor.shape[1] % block_size_factor == 0
    sub_block_size = tensor.shape[1] // block_size_factor
    sub_offsets = np.arange(block_size_factor, dtype=np.int64) * sub_block_size
    all_ptrs = (
        base_ptr + block_ids.astype(np.int64)[:, np.newaxis] * row_stride
    ) + sub_offsets[np.newaxis, :]
    output[:] = all_ptrs.ravel()[skip_count : skip_count + num_sub_blocks]


class CpuNpuOffloadingHandler(OffloadingHandler):
    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        block_size_factor: int,
        num_cpu_blocks: int,
    ):
        self.block_size_factor = block_size_factor
        self.kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]] = (
            kv_caches.group_data_refs
        )

        # npu streams for npu->cpu and cpu->npu
        self.d2h_stream = torch.npu.Stream()
        self.h2d_stream = torch.npu.Stream()

        # Ordered queue of in-flight transfers per direction
        self._d2h_transfers: deque[Transfer] = deque()
        self._h2d_transfers: deque[Transfer] = deque()

        # Reusable event pool to avoid allocation overhead
        self._event_pool: list[torch.npu.Event] = []

        pin_memory = is_pin_memory_available()

        # allocate cpu tensors
        logger.info("Allocating %d CPU tensors...", len(kv_caches.tensors))
        self.npu_tensors: list[torch.Tensor] = []
        self.cpu_tensors: list[torch.Tensor] = []
        for kv_cache_tensor in kv_caches.tensors:
            npu_page_size_bytes = kv_cache_tensor.page_size_bytes
            npu_tensor = kv_cache_tensor.tensor.view(torch.int8).view(
                (-1, npu_page_size_bytes)
            )
            cpu_page_size_bytes = npu_page_size_bytes * self.block_size_factor

            logger.debug(
                "Allocating CPU tensor of shape (%d, %d)",
                num_cpu_blocks,
                cpu_page_size_bytes,
            )
            cpu_tensor = torch.zeros(
                (num_cpu_blocks, cpu_page_size_bytes),
                dtype=torch.int8,
                device="cpu",
                pin_memory=pin_memory,
            )
            self.npu_tensors.append(npu_tensor)
            self.cpu_tensors.append(cpu_tensor)

    def _get_event(self) -> Any:
        if self._event_pool:
            return self._event_pool.pop()
        return torch.npu.Event(enable_timing=True)

    def _recycle_event(self, event: Any) -> None:
        self._event_pool.append(event)

    def _build_transfer_args(
        self,
        src_spec: BlockIDsLoadStoreSpec,
        dst_spec: BlockIDsLoadStoreSpec,
        src_tensors: list[torch.Tensor],
        dst_tensors: list[torch.Tensor],
        src_block_size_factor: int,
        dst_block_size_factor: int,
        gpu_spec: GPULoadStoreSpec,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        group_sizes = gpu_spec.group_sizes
        block_indices = gpu_spec.block_indices
        assert len(group_sizes) == len(self.kv_cache_groups_data_refs)
        assert len(block_indices) == len(self.kv_cache_groups_data_refs)

        num_copy_ops = 0
        for group_size, group_data_refs in zip(
            group_sizes, self.kv_cache_groups_data_refs
        ):
            num_copy_ops += group_size * len(group_data_refs)

        all_src = np.empty(num_copy_ops, dtype=np.int64)
        all_dst = np.empty(num_copy_ops, dtype=np.int64)
        all_sizes = np.empty(num_copy_ops, dtype=np.int64)

        src_offset = 0
        dst_offset = 0
        op_idx = 0
        num_transfer_bytes = 0

        for group_size, block_idx, group_data_refs in zip(
            group_sizes, block_indices, self.kv_cache_groups_data_refs
        ):
            if group_size == 0:
                continue

            src_logical_blocks_to_skip = block_idx % src_block_size_factor
            dst_logical_blocks_to_skip = block_idx % dst_block_size_factor
            src_logical_blocks_count = group_size + src_logical_blocks_to_skip
            dst_logical_blocks_count = group_size + dst_logical_blocks_to_skip

            src_blocks_count = cdiv(src_logical_blocks_count, src_block_size_factor)
            dst_blocks_count = cdiv(dst_logical_blocks_count, dst_block_size_factor)

            src_end_offset = src_offset + src_blocks_count
            dst_end_offset = dst_offset + dst_blocks_count
            assert src_end_offset <= len(src_blocks)
            assert dst_end_offset <= len(dst_blocks)

            group_src = src_blocks[src_offset:src_end_offset]
            group_dst = dst_blocks[dst_offset:dst_end_offset]

            for data_ref in group_data_refs:
                tensor_idx = data_ref.tensor_idx
                end_idx = op_idx + group_size

                compute_sub_block_ptrs(
                    group_src,
                    src_block_size_factor,
                    all_src[op_idx:end_idx],
                    src_tensors[tensor_idx],
                    skip_count=src_logical_blocks_to_skip,
                )
                compute_sub_block_ptrs(
                    group_dst,
                    dst_block_size_factor,
                    all_dst[op_idx:end_idx],
                    dst_tensors[tensor_idx],
                    skip_count=dst_logical_blocks_to_skip,
                )

                all_sizes[op_idx:end_idx] = data_ref.page_size_bytes
                num_transfer_bytes += group_size * data_ref.page_size_bytes
                op_idx = end_idx

            src_offset = src_end_offset
            dst_offset = dst_end_offset

        assert src_offset == len(src_blocks)
        assert dst_offset == len(dst_blocks)
        assert op_idx == num_copy_ops

        return (
            torch.from_numpy(all_src),
            torch.from_numpy(all_dst),
            torch.from_numpy(all_sizes),
            num_transfer_bytes,
        )

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        src_spec, dst_spec = spec
        if isinstance(src_spec, CPULoadStoreSpec):
            assert isinstance(dst_spec, GPULoadStoreSpec)
            stream = self.h2d_stream
            src_tensors = self.cpu_tensors
            dst_tensors = self.npu_tensors
            src_block_size_factor = self.block_size_factor
            dst_block_size_factor = 1
            gpu_spec = dst_spec
            is_d2h = False
            transfers = self._h2d_transfers
        else:
            assert isinstance(src_spec, GPULoadStoreSpec)
            assert isinstance(dst_spec, CPULoadStoreSpec)
            stream = self.d2h_stream
            src_tensors = self.npu_tensors
            dst_tensors = self.cpu_tensors
            src_block_size_factor = 1
            dst_block_size_factor = self.block_size_factor
            gpu_spec = src_spec
            is_d2h = True
            transfers = self._d2h_transfers

        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)
        batch_src, batch_dst, batch_sizes, num_bytes = self._build_transfer_args(
            src_spec=src_spec,
            dst_spec=dst_spec,
            src_tensors=src_tensors,
            dst_tensors=dst_tensors,
            src_block_size_factor=src_block_size_factor,
            dst_block_size_factor=dst_block_size_factor,
            gpu_spec=gpu_spec,
        )

        start_event = self._get_event()
        end_event = self._get_event()

        if is_d2h:
            # Wait for model computation to finish before reading NPU data
            stream.wait_stream(torch.npu.current_stream())
        if transfers:
            # Ensure this transfer starts only after the previous one completes
            last_transfer = transfers[-1]
            stream.wait_event(last_transfer.end_event)

        with torch.npu.stream(stream):
            start_event.record(stream)
            if batch_sizes.numel() > 0:
                direction = 0 if not is_d2h else 1
                torch.ops._C_ascend.swap_blocks_batch(
                    batch_src, batch_dst, batch_sizes, direction
                )
            end_event.record(stream)

        transfers.append(
            Transfer(
                job_id=job_id,
                stream=stream,
                start_event=start_event,
                end_event=end_event,
                num_bytes=num_bytes,
            )
        )

        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        for transfers, transfer_type in [
            (
                self._d2h_transfers,
                (GPULoadStoreSpec.medium(), CPULoadStoreSpec.medium()),
            ),
            (
                self._h2d_transfers,
                (CPULoadStoreSpec.medium(), GPULoadStoreSpec.medium()),
            ),
        ]:
            while transfers and transfers[0].end_event.query():
                transfer = transfers.popleft()
                transfer_time = (
                    transfer.start_event.elapsed_time(transfer.end_event) * 1e-3
                )
                results.append(
                    TransferResult(
                        job_id=transfer.job_id,
                        success=True,
                        transfer_size=transfer.num_bytes,
                        transfer_time=transfer_time,
                        transfer_type=transfer_type,
                    )
                )
                self._recycle_event(transfer.start_event)
                self._recycle_event(transfer.end_event)
        return results

    def wait(self, job_ids: set[int]) -> None:
        """
        Wait (block) until all specified transfer jobs are completed.
        """
        for transfers in (self._d2h_transfers, self._h2d_transfers):
            for transfer in transfers:
                if transfer.job_id in job_ids:
                    transfer.end_event.synchronize()

    def shutdown(self) -> None:
        for transfers in (self._d2h_transfers, self._h2d_transfers):
            while transfers:
                transfer = transfers.popleft()
                transfer.end_event.synchronize()
        self._event_pool.clear()
        self.npu_tensors.clear()
        self.cpu_tensors.clear()
