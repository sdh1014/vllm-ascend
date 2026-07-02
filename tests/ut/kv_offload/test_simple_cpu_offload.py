from types import SimpleNamespace
from unittest.mock import MagicMock

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
