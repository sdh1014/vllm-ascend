# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
import os
import time

import pytest
from vllm import SamplingParams
from vllm.config import KVTransferConfig

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL_ENV = "VLLM_ASCEND_TEST_SIMPLE_CPU_OFFLOAD_MODEL"
MODEL = os.getenv(MODEL_ENV)
if MODEL is None:
    pytest.skip(
        f"set {MODEL_ENV} to run SimpleCPUOffload e2e tests",
        allow_module_level=True,
    )


def _build_kv_transfer_config(
    cpu_bytes_to_use_per_rank: int,
) -> KVTransferConfig:
    return KVTransferConfig(
        kv_connector="SimpleCPUOffloadConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "cpu_bytes_to_use": 4 << 30,
            "cpu_bytes_to_use_per_rank": cpu_bytes_to_use_per_rank,
        },
    )


@wait_until_npu_memory_free()
def test_simple_cpu_offload_tp2_per_rank_capacity() -> None:
    sampling_params = SamplingParams(max_tokens=1, temperature=0)
    prompt = "hi " * 500 + "Let's count to ten. One, two, three, "

    with VllmRunner(
        MODEL,
        max_model_len=4096,
        tensor_parallel_size=2,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.5,
        enable_prefix_caching=True,
        kv_transfer_config=_build_kv_transfer_config(512 * (1 << 20)),
        enforce_eager=True,
    ) as runner:
        llm = runner.model
        cold_output = llm.generate(prompt, sampling_params, use_tqdm=False)[0]
        expected = cold_output.outputs[0].text

        success = 0
        attempts = 5
        for _ in range(attempts):
            time.sleep(2)
            if not llm.reset_prefix_cache():
                continue
            output = llm.generate(prompt, sampling_params, use_tqdm=False)[0]
            if output.outputs[0].text == expected:
                success += 1

        assert success >= int(0.5 * attempts), (
            f"TP2 CPU-load accuracy too low: {success}/{attempts} matched baseline output {expected!r}"
        )
