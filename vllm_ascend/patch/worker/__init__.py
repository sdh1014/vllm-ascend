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
#

import importlib

from vllm.triton_utils import HAS_TRITON

from vllm_ascend.utils import is_310p, vllm_version_is

# The v2 model runner is intentionally NOT made compatible with the v0.22.1
# release. vLLM v0.22.1 and the verified main commit are diverged, and the v2
# worker patches target main-only APIs; rather than maintain a separate v0.22.1
# compatibility path we keep v2 main-only. With v0.22.1 installed this flag is
# False, so none of the patch_v2.* / routed-experts-capture patches below are
# imported and the v2 worker stays dormant (the release uses the v1 runner).
_V2_MODEL_RUNNER_SUPPORTED = not vllm_version_is("0.22.1")


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

if HAS_TRITON:
    import vllm_ascend.patch.worker.patch_triton

    if _V2_MODEL_RUNNER_SUPPORTED:
        _try_import_patch("vllm_ascend.patch.worker.patch_v2.patch_triton")


import vllm_ascend.patch.worker.patch_weight_utils  # noqa
import vllm_ascend.patch.worker.patch_distributed  # noqa
_try_import_patch("vllm_ascend.patch.worker.patch_minimax_m2")
_try_import_patch("vllm_ascend.patch.worker.patch_minimax_m2_linear_attn")
_try_import_patch("vllm_ascend.patch.worker.patch_mamba_utils")
_try_import_patch("vllm_ascend.patch.worker.patch_qwen3_next_mtp")
_try_import_patch("vllm_ascend.patch.worker.patch_deepseek_compressor")

if not is_310p():
    _try_import_patch("vllm_ascend.patch.worker.patch_qwen3_5")
    _try_import_patch("vllm_ascend.patch.worker.patch_gdn_attn")
    _try_import_patch("vllm_ascend.patch.worker.patch_qwen3_dflash")
    _try_import_patch("vllm_ascend.patch.worker.patch_qwen3vl")
else:
    _try_import_patch("vllm_ascend.patch.worker.patch_idex_310")
_try_import_patch("vllm_ascend.patch.worker.patch_rejection_sampler")
_try_import_patch("vllm_ascend.patch.worker.patch_npugraph_ex_triton")
_try_import_patch("vllm_ascend.patch.worker.patch_kimi_k25")
_try_import_patch("vllm_ascend.patch.worker.patch_draft_quarot")
_try_import_patch("vllm_ascend.patch.worker.patch_cudagraph")
_try_import_patch("vllm_ascend.patch.worker.patch_deepseek_mtp")
_try_import_patch("vllm_ascend.patch.worker.patch_gqa_c8")

if _V2_MODEL_RUNNER_SUPPORTED:
    _try_import_patch("vllm_ascend.patch.worker.patch_v2.patch_uva")
    _try_import_patch("vllm_ascend.patch.worker.patch_v2.patch_input_batch")
    _try_import_patch("vllm_ascend.patch.worker.patch_v2.patch_model_state")
    _try_import_patch("vllm_ascend.patch.worker.patch_v2.patch_block_table")
    _try_import_patch("vllm_ascend.patch.worker.patch_v2.patch_attn_utils")
    _try_import_patch("vllm_ascend.patch.worker.patch_routed_experts_capture")
