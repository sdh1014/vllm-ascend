# Upstream Native KV Offload Research

## Source

- vLLM repo: `/Users/songdehao/sdh-lab/code/vllm`
- Checked remote: `origin/main`
- Latest checked commit: `b06813e87207e15b133e903d641e03f237d85b17`
- Local HEAD before update: `87e31455b056c6ce59bf5dcb3c622155431851db`

## Findings

- `--kv-offloading-backend native` is wired through `CacheConfig.kv_offloading_backend`.
- `--kv-offloading-size` activates KV offload. When backend is `native`, upstream creates or updates `KVTransferConfig` with:
  - `kv_connector = "OffloadingConnector"` unless `VLLM_USE_SIMPLE_KV_OFFLOAD` is enabled.
  - `kv_connector_extra_config["cpu_bytes_to_use"] = kv_offloading_size * (1 << 30)`.
  - `kv_role = "kv_both"`.
- Upstream `OffloadingSpec` APIs now live in `vllm.v1.kv_offload.base`, not the older `abstract` / `spec` modules.
- `CPULoadStoreSpec` now lives in `vllm.v1.kv_offload.cpu.common`.
- `GPULoadStoreSpec` is in `vllm.v1.kv_offload.base` and requires `group_sizes` and `block_indices`.
- Worker-side `get_handlers()` now receives `CanonicalKVCaches`.
- `CanonicalKVCaches` contains canonical block tensors plus per-KV-group data refs. Upstream builds it in `vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py`.
- Upstream CPU handler copies by canonical tensor refs and honors per-group sizes and block indices for hybrid or padded page layouts.
- Upstream added `reset_cache()` and `shutdown()` hooks on offload manager/handler paths.

## Impact For vLLM Ascend

- `vllm_ascend/kv_offload/npu.py` imports are stale and must be moved to current upstream modules.
- `NPUOffloadingSpec` should accept `cpu_bytes_to_use` from native CLI path, while preserving explicit `num_cpu_blocks` compatibility only if needed.
- `CpuNpuOffloadingHandler` should consume `CanonicalKVCaches`, not `dict[str, torch.Tensor]`.
- Transfer logic must use `GPULoadStoreSpec.group_sizes` and `block_indices`; the current vectorized logic assumes one contiguous group.
- Tests should cover the config path for `--kv-offloading-backend native` and handler/spec construction without requiring NPU where possible.
