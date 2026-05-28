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

import os
from typing import Any, NamedTuple
from unittest.mock import patch

import huggingface_hub
import pytest
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from modelscope import snapshot_download as modelscope_snapshot_download  # type: ignore[import-untyped]
from vllm.assets.image import ImageAsset

from tests.e2e.conftest import (
    VllmRunner,
    cleanup_dist_env_and_memory,
    wait_until_npu_memory_free,
)


class GGUFMultiModalModel(NamedTuple):
    tokenizer_model: str
    gguf_repo: str
    gguf_backbone: str
    gguf_mmproj: str
    mm_processor_kwargs: dict[str, Any] | None = None

    @property
    def model_path(self) -> str:
        _hf_hub_download_or_skip(self.gguf_repo, self.gguf_mmproj)
        return _hf_hub_download_or_skip(self.gguf_repo, self.gguf_backbone)

    @property
    def tokenizer_path(self) -> str:
        return modelscope_snapshot_download(
            self.tokenizer_model,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            ignore_file_pattern=["*.safetensors", "*.bin", "*.pt", "*.pth", "*.onnx"],
        )


def _hf_hub_download_or_skip(repo_id: str, filename: str) -> str:
    try:
        return hf_hub_download(repo_id, filename=filename)
    except (HfHubHTTPError, LocalEntryNotFoundError) as exc:
        message = str(exc).lower()
        if "403" in message or "gated" in message:
            pytest.skip(f"GGUF file requires gated Hugging Face access: {repo_id}/{filename}")
        raise


GEMMA3_GGUF_MODELS = [
    GGUFMultiModalModel(
        tokenizer_model="LLM-Research/gemma-3-4b-it",
        gguf_repo="google/gemma-3-4b-it-qat-q4_0-gguf",
        gguf_backbone="gemma-3-4b-it-q4_0.gguf",
        gguf_mmproj="mmproj-model-f16-4B.gguf",
    ),
    GGUFMultiModalModel(
        tokenizer_model="LLM-Research/gemma-3-4b-it",
        gguf_repo="unsloth/gemma-3-4b-it-GGUF",
        gguf_backbone="gemma-3-4b-it-BF16.gguf",
        gguf_mmproj="mmproj-BF16.gguf",
        mm_processor_kwargs={"do_pan_and_scan": True},
    ),
]

GEMMA3_PROMPTS = [
    (
        "<bos><start_of_turn>user\n"
        "<start_of_image>What's the content in the center of the image?"
        "<end_of_turn>\n<start_of_turn>model\n"
    ),
    (
        "<bos><start_of_turn>user\n"
        "<start_of_image>What is the season?"
        "<end_of_turn>\n<start_of_turn>model\n"
    ),
]

GEMMA3_IMAGE_NAMES = ["stop_sign", "cherry_blossom"]


@pytest.mark.parametrize(
    "model",
    GEMMA3_GGUF_MODELS,
    ids=["q4_0_f16_mmproj", "bf16_bf16_mmproj"],
)
@patch.dict(os.environ, {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
@wait_until_npu_memory_free()
def test_gemma3_gguf_multimodal_smoke(model: GGUFMultiModalModel):
    images = [ImageAsset(name).pil_image.convert("RGB") for name in GEMMA3_IMAGE_NAMES]

    with VllmRunner(
        model.model_path,
        tokenizer_name=model.tokenizer_path,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=4096,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs=model.mm_processor_kwargs or {},
        enable_chunked_prefill=False,
    ) as runner:
        outputs = runner.generate_greedy(
            prompts=GEMMA3_PROMPTS,
            images=images,
            max_tokens=8,
        )
        cleanup_dist_env_and_memory()

    assert len(outputs) == len(GEMMA3_PROMPTS)
    for token_ids, output_text in outputs:
        assert token_ids
        assert output_text
