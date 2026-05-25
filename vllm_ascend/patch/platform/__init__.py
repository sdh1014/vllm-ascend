#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

import importlib
import os

import vllm_ascend.patch.platform.patch_camem_allocator  # noqa
import vllm_ascend.patch.platform.patch_distributed  # noqa
import vllm_ascend.patch.platform.patch_kv_cache_interface  # noqa
from vllm_ascend import envs
from vllm_ascend.utils import is_310p


def _try_import_patch(module: str) -> None:
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name is not None and exc.name.startswith("vllm."):
            return
        raise
    except ImportError as exc:
        if "vllm." in str(exc):
            return
        raise


_try_import_patch("vllm_ascend.patch.platform.patch_kv_cache_utils")
_try_import_patch("vllm_ascend.patch.platform.patch_mla_prefill_backend")

if not is_310p():
    _try_import_patch("vllm_ascend.patch.platform.patch_mamba_config")
else:
    _try_import_patch("vllm_ascend.patch.platform.patch_mamba_config_310")
_try_import_patch("vllm_ascend.patch.platform.patch_minimax_m2_config")
_try_import_patch("vllm_ascend.patch.platform.patch_minimax_usage_accounting")
_try_import_patch("vllm_ascend.patch.platform.patch_glm_tool_call_streaming")
_try_import_patch("vllm_ascend.patch.platform.patch_glm47_tool_call_parser")
_try_import_patch("vllm_ascend.patch.platform.patch_anthropic_system_message")
_try_import_patch("vllm_ascend.patch.platform.patch_minimax_m2_tool_call_parser")
_try_import_patch("vllm_ascend.patch.platform.patch_deepseek_v4_tool_call_parser")
_try_import_patch("vllm_ascend.patch.platform.patch_deepseek_v4_thinking")
_try_import_patch("vllm_ascend.patch.platform.patch_torch_accelerator")
_try_import_patch("vllm_ascend.patch.platform.patch_tool_choice_none_content")
_try_import_patch("vllm_ascend.patch.platform.patch_mamba_manager")

if os.getenv("DYNAMIC_EPLB", "false").lower() in ("true", "1") or os.getenv("EXPERT_MAP_RECORD", "false") == "true":
    _try_import_patch("vllm_ascend.patch.platform.patch_multiproc_executor")

_try_import_patch("vllm_ascend.patch.platform.patch_balance_schedule")

if envs.VLLM_ASCEND_APPLY_DSV4_PATCH:
    _try_import_patch("vllm_ascend.patch.platform.patch_kv_cache_coordinator")
    _try_import_patch("vllm_ascend.patch.platform.patch_speculative_config")

_try_import_patch("vllm_ascend.patch.platform.patch_scheduler")
