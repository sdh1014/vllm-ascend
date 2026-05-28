#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import pytest
from huggingface_hub import hf_hub_download
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, cleanup_dist_env_and_memory, wait_until_npu_memory_free
from tests.e2e.singlecard.utils import PROMPTS_SHORT


GGUF_TEXT_MODELS = [
    (
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "qwen2.5-1.5b-instruct-q6_k.gguf",
    ),
    (
        "Qwen/Qwen3-0.6B",
        "unsloth/Qwen3-0.6B-GGUF",
        "Qwen3-0.6B-BF16.gguf",
    ),
    (
        "microsoft/Phi-3.5-mini-instruct",
        "bartowski/Phi-3.5-mini-instruct-GGUF",
        "Phi-3.5-mini-instruct-IQ4_XS.gguf",
    ),
]


@pytest.mark.parametrize(("tokenizer_name", "gguf_repo", "gguf_filename"), GGUF_TEXT_MODELS)
@wait_until_npu_memory_free()
def test_dense_text_gguf_smoke(tokenizer_name: str, gguf_repo: str, gguf_filename: str):
    sampling_params = SamplingParams(max_tokens=8, temperature=0.0)
    gguf_model = hf_hub_download(gguf_repo, filename=gguf_filename)

    with VllmRunner(
        gguf_model,
        tokenizer_name=tokenizer_name,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=1024,
        enable_chunked_prefill=False,
    ) as runner:
        outputs = runner.generate(PROMPTS_SHORT[:2], sampling_params)
        cleanup_dist_env_and_memory()

    assert len(outputs) == 2
    for token_ids, output_texts in outputs:
        assert token_ids[0]
        assert output_texts[0]
