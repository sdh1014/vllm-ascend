from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm_ascend.simple_kv_offload.npu_mem_ops import (
    DIRECTION_D2H,
    BatchMemcpyWorkspace,
    build_params,
    copy_blocks,
)
from vllm_ascend.simple_kv_offload.worker import SimpleCPUOffloadNPUWorker


def test_npu_worker_store_waits_for_compute_done(monkeypatch):
    worker = SimpleCPUOffloadNPUWorker.__new__(SimpleCPUOffloadNPUWorker)
    worker._backend = MagicMock()
    worker._connector_metadata = SimpleNamespace(
        load_cpu_blocks=[1],
        load_gpu_blocks=[2],
        load_event=3,
        load_event_to_reqs={},
        store_gpu_blocks=[4],
        store_cpu_blocks=[5],
        store_event=6,
    )
    worker._load_events = []
    worker._store_events = []
    worker._pending_load_event_indices = set()
    worker._pending_store_event_indices = set()
    worker._completed_store_events = {}
    worker._store_compute_done = None
    worker._poll_stream_events = MagicMock(return_value=-1)

    event = MagicMock()
    current_stream = MagicMock()
    monkeypatch.setattr(
        "vllm_ascend.simple_kv_offload.worker.torch.npu.Event",
        MagicMock(return_value=event),
    )
    monkeypatch.setattr(
        "vllm_ascend.simple_kv_offload.worker.torch.npu.current_stream",
        MagicMock(return_value=current_stream),
    )

    assert worker.get_finished(set()) == (None, None)

    event.record.assert_called_once_with(current_stream)
    load_call, store_call = worker._backend.launch_copy.call_args_list
    assert "wait_event" not in load_call.kwargs
    assert store_call.kwargs["wait_event"] is event


def test_npu_worker_recycles_completed_transfer_events():
    worker = SimpleCPUOffloadNPUWorker.__new__(SimpleCPUOffloadNPUWorker)
    worker._backend = MagicMock()
    worker._store_hwm = -1
    worker._load_hwm = -1

    event = MagicMock()
    event.query.return_value = True
    worker._store_events = [(9, event)]
    worker._load_events = []

    assert worker._poll_stream_events(is_store=True) == 9
    assert worker._store_events == []
    assert worker._store_hwm == 9
    worker._backend.recycle_event.assert_called_once_with(event)


def test_copy_blocks_reuses_workspace_tensors(monkeypatch):
    captured: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []

    def fake_swap_blocks_batch(
        src: torch.Tensor,
        dst: torch.Tensor,
        sizes: torch.Tensor,
        direction: int,
    ) -> None:
        captured.append((src, dst, sizes, direction))

    monkeypatch.setattr(
        torch.ops._C_ascend,
        "swap_blocks_batch",
        fake_swap_blocks_batch,
        raising=False,
    )

    src_caches = {f"t{i}": torch.empty((8, 16), dtype=torch.uint8) for i in range(2)}
    dst_caches = {f"t{i}": torch.empty((8, 16), dtype=torch.uint8) for i in range(2)}
    params = build_params(src_caches, dst_caches, DIRECTION_D2H)
    workspace = BatchMemcpyWorkspace()

    copy_blocks([0, 1], [2, 3], params, workspace)
    first_src, first_dst, first_sizes, first_direction = captured[-1]
    copy_blocks([1, 2], [3, 4], params, workspace)
    second_src, second_dst, second_sizes, second_direction = captured[-1]

    assert first_direction == second_direction == DIRECTION_D2H
    assert first_src.data_ptr() == second_src.data_ptr()
    assert first_dst.data_ptr() == second_dst.data_ptr()
    assert first_sizes.data_ptr() == second_sizes.data_ptr()
