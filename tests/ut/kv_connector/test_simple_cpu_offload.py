import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import DeviceConfig, KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import SupportsHMA, supports_hma
from vllm.platforms import current_platform

from vllm_ascend.simple_kv_offload.worker import SimpleCPUOffloadNPUWorker

_ASCEND_REGISTERED_CONNECTORS = [
    "MultiConnector",
    "MooncakeConnectorV1",
    "MooncakeHybridConnector",
    "MooncakeConnectorStoreV1",
    "AscendStoreConnector",
    "MooncakeLayerwiseConnector",
    "UCMConnector",
    "LMCacheAscendConnector",
    "SimpleCPUOffloadConnector",
    "RecomputeCPUOffloadConnector",
]


class _TestConnector:
    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config):
        return False


class _TestHMAConnector(_TestConnector, SupportsHMA):
    def request_finished_all_groups(self, request, block_ids):
        raise NotImplementedError


class _TestNonHMAConnector(_TestConnector):
    pass


@pytest.fixture()
def ascend_connector_registry():
    module = importlib.import_module("vllm_ascend.distributed.kv_transfer")
    old_registry = dict(KVConnectorFactory._registry)

    try:
        for name in _ASCEND_REGISTERED_CONNECTORS:
            KVConnectorFactory._registry.pop(name, None)
        module.register_connector()
        KVConnectorFactory._registry["ExampleConnector"] = lambda: _TestNonHMAConnector
        KVConnectorFactory._registry["OffloadingConnector"] = lambda: _TestHMAConnector
        yield
    finally:
        KVConnectorFactory._registry = old_registry


def test_simple_cpu_offload_connector_registry_override(monkeypatch):
    module = importlib.import_module("vllm_ascend.distributed.kv_transfer")
    factory = module.KVConnectorFactory
    old_registry = dict(factory._registry)

    try:
        factory._registry = {
            "MultiConnector": MagicMock(),
            "SimpleCPUOffloadConnector": MagicMock(),
        }
        module.register_connector()
        connector_cls = factory.get_connector_class_by_name("SimpleCPUOffloadConnector")
        assert connector_cls.__name__ == "AscendSimpleCPUOffloadConnector"
        assert connector_cls.__module__.endswith("simple_cpu_offload_connector")
    finally:
        monkeypatch.setattr(factory, "_registry", old_registry)


def test_ascend_simple_cpu_offload_connector_supports_hma(
    ascend_connector_registry,
):
    connector_cls = KVConnectorFactory.get_connector_class_by_name("SimpleCPUOffloadConnector")

    assert connector_cls.__name__ == "AscendSimpleCPUOffloadConnector"
    assert supports_hma(connector_cls)


@pytest.mark.parametrize(
    "kv_transfer_config,expect_disabled",
    [
        (
            KVTransferConfig(
                kv_connector="SimpleCPUOffloadConnector",
                kv_role="kv_both",
                kv_connector_extra_config={"cpu_bytes_to_use": 1 << 30},
            ),
            False,
        ),
        (
            KVTransferConfig(kv_connector="ExampleConnector", kv_role="kv_both"),
            True,
        ),
        (
            KVTransferConfig(
                kv_connector="MultiConnector",
                kv_role="kv_both",
                kv_connector_extra_config={
                    "connectors": [
                        {
                            "kv_connector": "SimpleCPUOffloadConnector",
                            "kv_role": "kv_both",
                            "kv_connector_extra_config": {
                                "cpu_bytes_to_use": 1 << 30,
                            },
                        },
                        {
                            "kv_connector": "OffloadingConnector",
                            "kv_role": "kv_both",
                            "kv_connector_extra_config": {
                                "cpu_bytes_to_use": 1 << 30,
                            },
                        },
                    ]
                },
            ),
            False,
        ),
        (
            KVTransferConfig(
                kv_connector="MultiConnector",
                kv_role="kv_both",
                kv_connector_extra_config={
                    "connectors": [
                        {
                            "kv_connector": "SimpleCPUOffloadConnector",
                            "kv_role": "kv_both",
                            "kv_connector_extra_config": {
                                "cpu_bytes_to_use": 1 << 30,
                            },
                        },
                        {"kv_connector": "ExampleConnector", "kv_role": "kv_both"},
                    ]
                },
            ),
            True,
        ),
    ],
    ids=["hma_connector", "non_hma_connector", "multi_all_hma", "multi_mixed"],
)
def test_hma_auto_config_with_ascend_simple_cpu_offload(
    ascend_connector_registry,
    monkeypatch,
    kv_transfer_config,
    expect_disabled,
):
    monkeypatch.setattr(
        current_platform,
        "support_hybrid_kv_cache",
        lambda: True,
    )

    vllm_config = VllmConfig(
        device_config=DeviceConfig("cpu"),
        kv_transfer_config=kv_transfer_config,
    )

    assert vllm_config.scheduler_config.disable_hybrid_kv_cache_manager is expect_disabled


def test_build_block_views_ignores_storage_padding():
    tensor = torch.empty((6, 2, 4), dtype=torch.float16)
    views = SimpleCPUOffloadNPUWorker._build_block_views("layer", tensor, num_blocks=4)

    assert list(views) == ["layer"]
    assert views["layer"].shape == (4, 16)


def test_register_kv_caches_keeps_separate_kv_tensors(monkeypatch):
    worker = SimpleCPUOffloadNPUWorker.__new__(SimpleCPUOffloadNPUWorker)
    worker.kv_cache_config = SimpleNamespace(num_blocks=4)
    worker.cpu_capacity_bytes = 4096
    worker._backend = MagicMock()

    monkeypatch.setattr(
        "vllm_ascend.simple_kv_offload.worker.is_pin_memory_available",
        lambda: False,
    )
    monkeypatch.setattr(
        "vllm_ascend.simple_kv_offload.worker.torch.npu",
        SimpleNamespace(Stream=MagicMock(return_value=MagicMock())),
    )

    k_cache = torch.empty((4, 2), dtype=torch.float16)
    v_cache = torch.empty((4, 3), dtype=torch.float16)
    worker.register_kv_caches({"layer": (k_cache, v_cache)})

    assert list(worker.gpu_kv_caches) == ["layer", "layer.1"]
    assert worker.gpu_kv_caches["layer"].shape == (4, 4)
    assert worker.gpu_kv_caches["layer.1"].shape == (4, 6)
    worker._backend.init.assert_called_once()
