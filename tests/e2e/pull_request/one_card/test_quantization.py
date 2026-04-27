#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
import json
import os

from tests.e2e.conftest import VllmRunner
from tests.e2e.model_utils import check_outputs_equal

AWQ_SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
AWQ_SMOKE_PROMPTS = ["vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs."]
AWQ_SMOKE_MAX_TOKENS = 5
AWQ_REFERENCE_GENERATED_TOKEN_IDS = [1084, 374, 6188, 311, 387]
AWQ_REFERENCE_TEXT = (
    "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs. It is designed to be"
)


def _generate_awq_smoke_output():
    with VllmRunner(
        os.getenv("VLLM_ASCEND_AWQ_MODEL", AWQ_SMOKE_MODEL),
        max_model_len=512,
        gpu_memory_utilization=0.7,
        enforce_eager=True,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        quantization="awq",
    ) as vllm_model:
        return vllm_model.generate_greedy(AWQ_SMOKE_PROMPTS, AWQ_SMOKE_MAX_TOKENS)


def _reference_awq_smoke_output():
    token_ids = json.loads(os.getenv("VLLM_ASCEND_AWQ_REFERENCE_GENERATED_TOKEN_IDS", "null"))
    text = os.getenv("VLLM_ASCEND_AWQ_REFERENCE_TEXT", AWQ_REFERENCE_TEXT)
    if token_ids is None:
        token_ids = AWQ_REFERENCE_GENERATED_TOKEN_IDS
    return [(token_ids, text)], True


def test_qwen2_5_awq_greedy_smoke():
    awq_outputs = _generate_awq_smoke_output()

    assert len(awq_outputs) == len(AWQ_SMOKE_PROMPTS)
    assert awq_outputs[0][1].startswith(AWQ_SMOKE_PROMPTS[0])
    assert len(awq_outputs[0][1]) > len(AWQ_SMOKE_PROMPTS[0])


def test_qwen2_5_awq_matches_reference():
    reference_outputs, reference_is_generated_only = _reference_awq_smoke_output()
    awq_outputs = _generate_awq_smoke_output()

    if reference_is_generated_only:
        reference_token_ids, reference_text = reference_outputs[0]
        awq_token_ids, awq_text = awq_outputs[0]
        assert awq_text == reference_text
        assert awq_token_ids[-len(reference_token_ids) :] == reference_token_ids
        return

    check_outputs_equal(
        outputs_0_lst=reference_outputs,
        outputs_1_lst=awq_outputs,
        name_0="reference_outputs",
        name_1="awq_outputs",
    )


# fmt: off
def test_qwen3_w8a8_quant():
    max_tokens = 5
    example_prompts = [
        "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs."
    ]
    vllm_target_outputs = [([
        85, 4086, 44, 374, 264, 1550, 42747, 628, 323, 4938, 72816, 44378, 323,
        13480, 4712, 369, 444, 10994, 82, 13, 1084, 374, 6188, 311, 387
    ], 'vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs. It is designed to be'
                            )]
# fmt: on

    with VllmRunner(
            "vllm-ascend/Qwen3-0.6B-W8A8",
            max_model_len=8192,
            gpu_memory_utilization=0.7,
            cudagraph_capture_sizes=[1, 2, 4, 8],
            quantization="ascend",
    ) as vllm_model:
        vllm_quant_w8a8_outputs = vllm_model.generate_greedy(
            example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=vllm_target_outputs,
        outputs_1_lst=vllm_quant_w8a8_outputs,
        name_0="vllm_target_outputs",
        name_1="vllm_quant_w8a8_outputs",
    )

# fmt: off
def test_qwen3_w8a8_quant_auto_detect():
    """Test that ModelSlim quantization is auto-detected without --quantization.

    Uses the same W8A8 model as test_qwen3_w8a8_quant but omits the
    quantization parameter, verifying that the auto-detection in
    maybe_auto_detect_quantization() picks up quant_model_description.json
    and produces identical results.
    """
    max_tokens = 5
    example_prompts = [
        "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs."
    ]
    vllm_target_outputs = [([
        85, 4086, 44, 374, 264, 1550, 42747, 628, 323, 4938, 72816, 44378, 323,
        13480, 4712, 369, 444, 10994, 82, 13, 1084, 374, 6188, 311, 387
    ], 'vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs. It is designed to be'
                            )]
# fmt: on

    with VllmRunner(
            "vllm-ascend/Qwen3-0.6B-W8A8",
            max_model_len=8192,
            gpu_memory_utilization=0.7,
            cudagraph_capture_sizes=[1, 2, 4, 8],
    ) as vllm_model:
        vllm_quant_auto_detect_outputs = vllm_model.generate_greedy(
            example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=vllm_target_outputs,
        outputs_1_lst=vllm_quant_auto_detect_outputs,
        name_0="vllm_target_outputs",
        name_1="vllm_quant_auto_detect_outputs",
    )


# fmt: off
def test_qwen3_dense_w8a16():
    max_tokens = 5
    example_prompts = [
        "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs."
    ]
    vllm_target_outputs = [([
        85, 4086, 44, 374, 264, 1550, 42747, 628, 323, 4938, 72816, 44378, 323,
        13480, 4712, 369, 444, 10994, 82, 13, 1084, 374, 6188, 311, 387
    ], 'vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs. It is designed to be'
                            )]
# fmt: on

    with VllmRunner(
            "vllm-ascend/Qwen3-0.6B-W8A16",
            max_model_len=8192,
            enforce_eager=False,
            gpu_memory_utilization=0.7,
            quantization="ascend",
    ) as vllm_model:
        vllm_quant_w8a16_outputs = vllm_model.generate_greedy(
            example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=vllm_target_outputs,
        outputs_1_lst=vllm_quant_w8a16_outputs,
        name_0="vllm_target_outputs",
        name_1="vllm_quant_w8a16_outputs",
    )
