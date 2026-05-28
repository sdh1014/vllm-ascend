from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence

import torch
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingManager,
    OffloadingSpec,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

from vllm_ascend.kv_offload.cpu_npu import CpuNpuOffloadingHandler


class NPUOffloadingSpec(OffloadingSpec):
    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise Exception(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config"
            )

        if kv_cache_config.num_blocks > 0:
            total_npu_kv_bytes = sum(t.size for t in kv_cache_config.kv_cache_tensors)
            kv_bytes_per_block = (
                total_npu_kv_bytes // kv_cache_config.num_blocks
            ) * vllm_config.parallel_config.world_size
        else:
            kv_bytes_per_block = 0

        kv_bytes_per_offloaded_block = kv_bytes_per_block * self.block_size_factor
        self.num_cpu_blocks = (
            int(cpu_bytes_to_use) // kv_bytes_per_offloaded_block
            if kv_bytes_per_offloaded_block > 0
            else 0
        )
        self.eviction_policy = self.extra_config.get("eviction_policy", "lru")

        # scheduler-side
        self._manager: OffloadingManager | None = None

        # worker-side
        self._handler: OffloadingHandler | None = None

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None
                and kv_events_config.enable_kv_cache_events
            )
            self._manager = CPUOffloadingManager(
                num_blocks=self.num_cpu_blocks,
                cache_policy=self.eviction_policy,
                enable_events=enable_events,
                store_threshold=int(self.extra_config.get("store_threshold", 0)),
                max_tracker_size=int(
                    self.extra_config.get("max_tracker_size", 64_000)
                ),
            )
        return self._manager

    def get_handlers(
        self,
        kv_caches: CanonicalKVCaches,
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        if not self._handler:
            self._handler = CpuNpuOffloadingHandler(
                kv_caches=kv_caches,
                block_size_factor=self.block_size_factor,
                num_cpu_blocks=self.num_cpu_blocks,
            )

        assert self._handler is not None
        yield GPULoadStoreSpec, CPULoadStoreSpec, self._handler
        yield CPULoadStoreSpec, GPULoadStoreSpec, self._handler

    @staticmethod
    def _as_tensor_sequence(kv_cache: torch.Tensor | Sequence[torch.Tensor]):
        if isinstance(kv_cache, torch.Tensor):
            return (kv_cache,)
        return tuple(kv_cache)

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor | Sequence[torch.Tensor]],
        register_handlers: Callable[[CanonicalKVCaches], None],
    ) -> None:
        num_blocks = self.kv_cache_config.num_blocks
        block_tensors: list[CanonicalKVCacheTensor] = []
        block_data_refs: dict[str, list[CanonicalKVCacheRef]] = defaultdict(list)

        for kv_cache_tensor in self.kv_cache_config.kv_cache_tensors:
            tensor_layer_names = [
                layer_name
                for layer_name in kv_cache_tensor.shared_by
                if layer_name in kv_caches
            ]
            if not tensor_layer_names:
                continue

            first_layer_name = tensor_layer_names[0]
            for tensor in self._as_tensor_sequence(kv_caches[first_layer_name]):
                page_size_bytes = tensor.view(torch.int8).numel() // num_blocks
                block_tensors.append(
                    CanonicalKVCacheTensor(
                        tensor=tensor,
                        page_size_bytes=page_size_bytes,
                    )
                )
                curr_tensor_idx = len(block_tensors) - 1
                for layer_name in tensor_layer_names:
                    block_data_refs[layer_name].append(
                        CanonicalKVCacheRef(
                            tensor_idx=curr_tensor_idx,
                            page_size_bytes=page_size_bytes,
                        )
                    )

        group_data_refs: list[list[CanonicalKVCacheRef]] = []
        for kv_cache_group in self.kv_cache_config.kv_cache_groups:
            group_refs: list[CanonicalKVCacheRef] = []
            for layer_name in kv_cache_group.layer_names:
                group_refs += block_data_refs[layer_name]
            group_data_refs.append(group_refs)

        register_handlers(
            CanonicalKVCaches(
                tensors=block_tensors,
                group_data_refs=group_data_refs,
            )
        )


def _patch_offloading_connector_worker() -> None:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
        OffloadingConnectorWorker,
    )

    if getattr(OffloadingConnectorWorker, "_ascend_native_offload_patched", False):
        return

    original_register_kv_caches = OffloadingConnectorWorker.register_kv_caches

    def register_kv_caches(self, kv_caches):
        spec_register = getattr(self.spec, "register_kv_caches", None)
        if spec_register is not None:
            return spec_register(kv_caches, self._register_handlers)
        return original_register_kv_caches(self, kv_caches)

    OffloadingConnectorWorker.register_kv_caches = register_kv_caches
    OffloadingConnectorWorker._ascend_native_offload_patched = True


_patch_offloading_connector_worker()
