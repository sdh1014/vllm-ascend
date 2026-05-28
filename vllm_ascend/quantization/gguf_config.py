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

from typing import Any

import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.gguf import (
    GGUFConfig,
    is_layer_skipped_gguf,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)

from vllm_ascend.quantization.methods.gguf import (
    AscendGGUFEmbeddingMethod,
    AscendGGUFLinearMethod,
)
from vllm_ascend.utils import GGUF_QUANTIZATION_METHOD


@register_quantization_config(GGUF_QUANTIZATION_METHOD)
class AscendGGUFConfig(GGUFConfig):
    """Ascend implementation of upstream GGUF quantization semantics."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.quant_description: dict[str, Any] = {}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AscendGGUFConfig":
        return cls()

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            if is_layer_skipped_gguf(
                prefix,
                self.unquantized_modules,
                self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            return AscendGGUFLinearMethod(self)

        if isinstance(layer, VocabParallelEmbedding):
            if is_layer_skipped_gguf(
                prefix,
                self.unquantized_modules,
                self.packed_modules_mapping,
            ):
                return UnquantizedEmbeddingMethod()
            return AscendGGUFEmbeddingMethod(self)

        if isinstance(layer, RoutedExperts):
            raise NotImplementedError("GGUF MoE on Ascend is planned for the MoE TP phase.")

        return None


__all__ = ["AscendGGUFConfig"]
