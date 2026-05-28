# Validation

## Local

- `python3 -m compileall -q vllm_ascend/platform.py vllm_ascend/kv_offload vllm_ascend/patch/worker/__init__.py vllm_ascend/utils.py tests/ut/kv_offload/test_native_npu_offload.py tests/e2e/singlecard/test_cpu_offloading.py`: passed.
- `git diff --check -- vllm_ascend/kv_offload/npu.py vllm_ascend/kv_offload/cpu_npu.py vllm_ascend/platform.py vllm_ascend/patch/worker/__init__.py vllm_ascend/utils.py tests/ut/kv_offload/test_native_npu_offload.py tests/e2e/singlecard/test_cpu_offloading.py docs/source/user_guide/feature_guide/kv_cache_cpu_offload.md`: passed.
- `python3 -m pytest -q tests/ut/kv_offload/test_native_npu_offload.py`: blocked locally; this host has no `pytest`, `ruff`, or `torch` in the available Python environments.
- 2026-05-26: Re-ran `python3 -m compileall -q ...`: passed.
- 2026-05-26: Re-ran `git diff --check -- ...`: passed.
- 2026-05-26: Re-ran `python3 -m pytest -q tests/ut/kv_offload/test_native_npu_offload.py`: blocked locally; this host still has no `pytest`.

## NPU

- Host: Huawei Cloud NPU, Ascend 910B2C, driver 25.5.1, CANN/OPP 9.0.0.
- Environment: `/root/vllm-ascend/.venv`, installed with Huawei Cloud PyPI mirror.
- `python` smoke passed:
  - `torch 2.10.0+cpu`
  - `torch_npu 2.10.0`
  - `vllm 0.19.1`
  - `torch.npu.is_available() == True`
  - `NPUOffloadingSpec` import passed.
- `COMPILE_CUSTOM_KERNELS=0 python -m pip install -e . --no-build-isolation -i https://repo.huaweicloud.com/repository/pypi/simple --trusted-host repo.huaweicloud.com`: passed.
- `python -m pytest -q tests/ut/kv_offload/test_native_npu_offload.py`: passed, `4 passed`.
- 2026-05-26: `npu-smi info`: passed; device `910B2C`, health `OK`, driver `25.5.1`.
- 2026-05-26: Synced selected source/test/doc files to `/root/vllm-ascend` without `--delete`, after dry-run showed file-only sync with `--delete` would remove unrelated remote paths.
- 2026-05-26: `cd /root/vllm-ascend && source .venv/bin/activate && python -m compileall -q vllm_ascend/platform.py vllm_ascend/kv_offload vllm_ascend/patch/worker/__init__.py vllm_ascend/utils.py tests/ut/kv_offload/test_native_npu_offload.py tests/e2e/singlecard/test_cpu_offloading.py && python -m pytest -q tests/ut/kv_offload/test_native_npu_offload.py`: passed, `4 passed`.
- 2026-05-26: `python -m ruff check ...`: blocked on remote; `/root/vllm-ascend/.venv` has `pytest` but no `ruff` module or executable.
- The target test now asserts upstream medium names in transfer results: `("GPU", "CPU")` for store completion.
- 2026-05-26: Actual model run attempted with `Qwen/Qwen3-8B`, `max_model_len=2048`, `gpu_memory_utilization=0.5`, `kv_offloading_backend="native"`, and `kv_offloading_size=4`; log saved at `/tmp/native_kv_offload_qwen3_8b.log`.
- 2026-05-26: First model attempt with stdin script failed before engine startup because multiprocessing `spawn` could not reopen `/root/vllm-ascend/<stdin>`.
- 2026-05-26: Retried from `/tmp/native_kv_offload_qwen3_8b.py`; vLLM resolved `Qwen3ForCausalLM`, activated Ascend plugin, set `device_config=npu`, and failed during `NPUModelRunner` initialization on `torch.zeros_like(..., device=npu)`.
- 2026-05-26: Confirmed the failure is a remote NPU/CANN runtime issue, not KV offload logic: `torch.empty((2, 2), device="npu").zero_()` fails after sourcing `/usr/local/Ascend/ascend-toolkit/set_env.sh` with `aclnnInplaceZero failed`, error code `561103`, `Parse dynamic kernel config fail`.
- Full custom kernel build is blocked on this host because current `build_aclnn.sh` fails under CANN 8.5.1 with unsupported generated op soc versions.
- 2026-05-26: Installed official `Ascend-cann-910b-ops_9.0.0_linux-x86_64.run` into `/usr/local/Ascend`; `zeros_like` kernels for `ascend910b` are now present under CANN OPP.
- 2026-05-26: Reinstalled matching Python group: `torch==2.10.0+cpu`, `torch-npu==2.10.0`, `triton-ascend==3.2.1`.
- 2026-05-26: NPU runtime smoke passed after CANN/OPP update: `torch.empty((2, 2), device="npu").zero_()` returned zeros.
- 2026-05-26: Downloaded minimal model from domestic ModelScope source: `Qwen/Qwen2.5-0.5B-Instruct`, cache path `/data/modelscope_cache/Qwen/Qwen2___5-0___5B-Instruct`.
- 2026-05-26: `python -m pytest -q tests/ut/kv_offload/test_native_npu_offload.py`: passed, `6 passed`.
- 2026-05-26: Minimal model native KV offload validation passed with `Qwen/Qwen2.5-0.5B-Instruct`, `max_model_len=512`, `gpu_memory_utilization=0.4`, `kv_offloading_backend="native"`, `kv_offloading_size=2`, `enforce_eager=True`; log saved at `/tmp/native_kv_offload_qwen2_5_0_5b_after_canonical_fix.log`.
- 2026-05-26: Runtime evidence from the passing log: `Creating offloading spec with name: NPUOffloadingSpec`, `Allocating 48 CPU tensors...`, model weights `0.9320 GB`, generated output ended with `native_kv_offload_min_model_ok`.
- 2026-05-26: Larger model native KV offload validation passed with `Qwen/Qwen2.5-1.5B-Instruct`, `max_model_len=1024`, `gpu_memory_utilization=0.5`, `kv_offloading_backend="native"`, `kv_offloading_size=4`, `enforce_eager=True`; log saved at `/tmp/native_kv_offload_qwen2_5_1_5b.log`.
- 2026-05-26: Larger model runtime evidence: `Creating offloading spec with name: NPUOffloadingSpec`, `Allocating 56 CPU tensors...`, checkpoint size `2.88 GiB`, model weights `2.8990 GB`, generated output ended with `native_kv_offload_1_5b_model_ok`.
- 2026-05-26: Native KV offload load-path validation passed with `Qwen/Qwen2.5-0.5B-Instruct`, two sequential requests sharing the same long prefix, `enable_prefix_caching=False`, `kv_offloading_backend="native"`, `kv_offloading_size=2`; log saved at `/tmp/native_kv_offload_load_path.log`.
- 2026-05-26: Load-path runtime evidence: first request printed `OFFLOAD_TRANSFER job_id=0 direction=GPULoadStoreSpec->CPULoadStoreSpec src_blocks=5 dst_blocks=5`; second request printed `OFFLOAD_TRANSFER job_id=1 direction=CPULoadStoreSpec->GPULoadStoreSpec src_blocks=5 dst_blocks=5`; run ended with `native_kv_offload_load_path_script_done`.
- 2026-05-26: Real QA load-path validation passed with `Qwen/Qwen2.5-0.5B-Instruct`; log saved at `/tmp/native_kv_offload_real_qa.log`.
- 2026-05-26: Real QA evidence: first answer started with `Paris`; second request printed `OFFLOAD_TRANSFER job_id=2 direction=CPULoadStoreSpec->GPULoadStoreSpec src_blocks=2 dst_blocks=2`; second answer correctly explained that KV cache offload stores reusable attention KV outside accelerator memory and loads matching cached blocks for later repeated-prefix requests; run ended with `native_kv_offload_real_qa_done`.
- 2026-05-26: Larger real QA load-path validation passed with `Qwen/Qwen2.5-1.5B-Instruct`; log saved at `/tmp/native_kv_offload_real_qa_1_5b.log`.
- 2026-05-26: Larger real QA evidence: first answer started with `Paris`; second request printed `OFFLOAD_TRANSFER job_id=2 direction=CPULoadStoreSpec->GPULoadStoreSpec src_blocks=2 dst_blocks=2`; second answer was `KV cache offload reduces recomputation by storing key and value tensors outside accelerator memory and loading them back when needed.`; run ended with `native_kv_offload_real_qa_1_5b_done`.

```bash
VLLM_USE_MODELSCOPE=True \
HF_HOME=/data/huggingface_home \
TRANSFORMERS_CACHE=/data/huggingface_home/hub \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python /tmp/native_kv_offload_qwen2_5_0_5b_safe.py
```

```bash
VLLM_USE_MODELSCOPE=True \
MODELSCOPE_CACHE=/data/modelscope_cache \
HF_HOME=/data/huggingface_home \
TRANSFORMERS_CACHE=/data/huggingface_home/hub \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python /tmp/native_kv_offload_qwen2_5_1_5b.py
```

```bash
VLLM_USE_MODELSCOPE=True \
MODELSCOPE_CACHE=/data/modelscope_cache \
HF_HOME=/data/huggingface_home \
TRANSFORMERS_CACHE=/data/huggingface_home/hub \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python /tmp/native_kv_offload_load_path.py
```

```bash
VLLM_USE_MODELSCOPE=True \
MODELSCOPE_CACHE=/data/modelscope_cache \
HF_HOME=/data/huggingface_home \
TRANSFORMERS_CACHE=/data/huggingface_home/hub \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python /tmp/native_kv_offload_real_qa.py
```

```bash
VLLM_USE_MODELSCOPE=True \
MODELSCOPE_CACHE=/data/modelscope_cache \
HF_HOME=/data/huggingface_home \
TRANSFORMERS_CACHE=/data/huggingface_home/hub \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python /tmp/native_kv_offload_real_qa_1_5b.py
```
