from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

from vllm_ascend.kv_offload.cpu_npu import CpuNpuOffloadingHandler
from vllm_ascend.kv_offload.npu import NPUOffloadingSpec
from vllm_ascend.platform import NPUPlatform


class FakeEvent:
    def record(self, stream):
        return None

    def query(self):
        return True

    def elapsed_time(self, end_event):
        return 1.0

    def synchronize(self):
        return None


class FakeStream:
    def wait_stream(self, stream):
        return None

    def wait_event(self, event):
        return None


class FakeStreamContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeNPU:
    @staticmethod
    def Stream():
        return FakeStream()

    @staticmethod
    def Event(enable_timing=True):
        return FakeEvent()

    @staticmethod
    def current_stream():
        return FakeStream()

    @staticmethod
    def stream(stream):
        return FakeStreamContext()


def make_vllm_config(cpu_bytes_to_use: int):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={"cpu_bytes_to_use": cpu_bytes_to_use}
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            world_size=2,
        ),
        cache_config=SimpleNamespace(block_size=16),
        kv_events_config=None,
    )


def make_kv_cache_config():
    kv_cache_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float16,
    )
    return KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[KVCacheTensor(size=10240, shared_by=["layer.0"])],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=kv_cache_spec)
        ],
    )


def test_platform_sets_ascend_spec_for_native_cli_config():
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="OffloadingConnector",
            kv_connector_extra_config={"cpu_bytes_to_use": 1024},
        )
    )

    NPUPlatform._set_native_kv_offload_spec(vllm_config)

    assert vllm_config.kv_transfer_config.kv_connector_extra_config == {
        "cpu_bytes_to_use": 1024,
        "spec_name": "NPUOffloadingSpec",
        "spec_module_path": "vllm_ascend.kv_offload.npu",
    }


def test_platform_prepares_native_cli_config_before_platform_check():
    class FakeVllmConfig:
        def __init__(self):
            self.cache_config = SimpleNamespace(kv_offloading_size=2)
            self.kv_transfer_config = None

        def _post_init_kv_transfer_config(self):
            self.kv_transfer_config = SimpleNamespace(
                kv_connector="OffloadingConnector",
                kv_connector_extra_config={"cpu_bytes_to_use": 2 << 30},
            )

    vllm_config = FakeVllmConfig()

    NPUPlatform._prepare_native_kv_offload_config(vllm_config)

    assert vllm_config.kv_transfer_config.kv_connector_extra_config == {
        "cpu_bytes_to_use": 2 << 30,
        "spec_name": "NPUOffloadingSpec",
        "spec_module_path": "vllm_ascend.kv_offload.npu",
    }


def test_npu_offloading_spec_uses_cpu_bytes_to_use():
    spec = NPUOffloadingSpec(
        vllm_config=make_vllm_config(cpu_bytes_to_use=1 << 20),
        kv_cache_config=make_kv_cache_config(),
    )

    assert spec.num_cpu_blocks == 512


def test_npu_spec_registers_tuple_kv_cache_as_canonical_cache():
    spec = NPUOffloadingSpec(
        vllm_config=make_vllm_config(cpu_bytes_to_use=1 << 20),
        kv_cache_config=make_kv_cache_config(),
    )
    captured = {}

    k_cache = torch.empty((10, 4), dtype=torch.float16)
    v_cache = torch.empty((10, 4), dtype=torch.float16)

    spec.register_kv_caches(
        {"layer.0": (k_cache, v_cache)},
        lambda canonical_kv_caches: captured.setdefault(
            "canonical", canonical_kv_caches
        ),
    )

    canonical = captured["canonical"]
    assert [tensor.tensor for tensor in canonical.tensors] == [k_cache, v_cache]
    assert [tensor.page_size_bytes for tensor in canonical.tensors] == [8, 8]
    assert canonical.group_data_refs == [
        [
            CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=8),
            CanonicalKVCacheRef(tensor_idx=1, page_size_bytes=8),
        ]
    ]


def test_npu_handler_uses_canonical_kv_cache_groups(monkeypatch):
    captured = {}

    def fake_swap_blocks_batch(src, dst, sizes, direction):
        captured["src"] = src.numpy().copy()
        captured["dst"] = dst.numpy().copy()
        captured["sizes"] = sizes.numpy().copy()
        captured["direction"] = direction

    monkeypatch.setattr(torch, "npu", FakeNPU, raising=False)
    monkeypatch.setattr(
        torch.ops,
        "_C_ascend",
        SimpleNamespace(swap_blocks_batch=fake_swap_blocks_batch),
        raising=False,
    )
    monkeypatch.setattr(
        "vllm_ascend.kv_offload.cpu_npu.is_pin_memory_available",
        lambda: False,
    )

    npu_tensor_0 = torch.empty((64, 4), dtype=torch.int8)
    npu_tensor_1 = torch.empty((64, 6), dtype=torch.int8)
    kv_caches = CanonicalKVCaches(
        tensors=[
            CanonicalKVCacheTensor(tensor=npu_tensor_0, page_size_bytes=4),
            CanonicalKVCacheTensor(tensor=npu_tensor_1, page_size_bytes=6),
        ],
        group_data_refs=[
            [CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=4)],
            [CanonicalKVCacheRef(tensor_idx=1, page_size_bytes=6)],
        ],
    )
    handler = CpuNpuOffloadingHandler(
        kv_caches=kv_caches,
        block_size_factor=2,
        num_cpu_blocks=256,
    )

    gpu_spec = GPULoadStoreSpec(
        block_ids=[10, 11, 20],
        group_sizes=[2, 1],
        block_indices=[1, 0],
    )
    cpu_spec = CPULoadStoreSpec([100, 101, 200])

    assert handler.transfer_async(7, (gpu_spec, cpu_spec))

    expected_src = [
        npu_tensor_0.data_ptr() + 10 * npu_tensor_0.stride(0),
        npu_tensor_0.data_ptr() + 11 * npu_tensor_0.stride(0),
        npu_tensor_1.data_ptr() + 20 * npu_tensor_1.stride(0),
    ]
    cpu_tensor_0, cpu_tensor_1 = handler.cpu_tensors
    expected_dst = [
        cpu_tensor_0.data_ptr() + 100 * cpu_tensor_0.stride(0) + 4,
        cpu_tensor_0.data_ptr() + 101 * cpu_tensor_0.stride(0),
        cpu_tensor_1.data_ptr() + 200 * cpu_tensor_1.stride(0),
    ]

    assert captured["src"].tolist() == expected_src
    assert captured["dst"].tolist() == expected_dst
    assert captured["sizes"].tolist() == [4, 4, 6]
    assert captured["direction"] == 1
    [result] = handler.get_finished()
    assert result.transfer_type == ("GPU", "CPU")


def test_npu_offloading_spec_requires_native_size():
    vllm_config = make_vllm_config(cpu_bytes_to_use=1 << 20)
    vllm_config.kv_transfer_config.kv_connector_extra_config = {}

    with pytest.raises(Exception, match="cpu_bytes_to_use"):
        NPUOffloadingSpec(
            vllm_config=vllm_config,
            kv_cache_config=make_kv_cache_config(),
        )
