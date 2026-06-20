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

from typing import Any

import torch
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from vllm_ascend.utils import AWQ_QUANTIZATION_METHOD


@register_quantization_config(AWQ_QUANTIZATION_METHOD)
class AscendAWQConfig(AWQConfig):
    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
        modules_to_not_convert: list[str] | None = None,
        lm_head_quantized: bool = False,
    ) -> None:
        super().__init__(weight_bits, group_size, zero_point, modules_to_not_convert)
        self.quant_description: dict[str, Any] = {}
        self.lm_head_quantized = lm_head_quantized

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AscendAWQConfig":
        quant_config = super().from_config(config)
        quant_config.lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        return quant_config

    @classmethod
    def get_name(cls) -> str:
        return AWQ_QUANTIZATION_METHOD

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
        **_: Any,
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase) or (
            isinstance(layer, ParallelLMHead) and getattr(self, "lm_head_quantized", False)
        ):
            if is_layer_skipped(
                prefix,
                self.modules_to_not_convert,
                self.packed_modules_mapping,
                skip_with_substr=True,
            ):
                return UnquantizedLinearMethod()

            from vllm_ascend.quantization.methods.awq import AscendAWQLinearMethod

            return AscendAWQLinearMethod(self)

        if isinstance(layer, RoutedExperts):
            if is_layer_skipped(
                prefix,
                self.modules_to_not_convert,
                skip_with_substr=True,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)

            from vllm_ascend.quantization.methods.awq import AscendAWQFusedMoEMethod

            return AscendAWQFusedMoEMethod(self, layer.moe_config)

        return None
