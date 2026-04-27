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
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS, register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped

from vllm_ascend.utils import AWQ_QUANTIZATION_METHOD


def _remove_quantization_method() -> None:
    if AWQ_QUANTIZATION_METHOD in QUANTIZATION_METHODS:
        QUANTIZATION_METHODS.remove(AWQ_QUANTIZATION_METHOD)


_remove_quantization_method()


@register_quantization_config(AWQ_QUANTIZATION_METHOD)
class AscendAWQConfig(QuantizationConfig):
    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
        modules_to_not_convert: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.modules_to_not_convert = modules_to_not_convert or []

        if self.weight_bits != 4:
            raise ValueError(
                f"Currently, only 4-bit weight quantization is supported for AWQ, but got {self.weight_bits} bits."
            )
        self.pack_factor = 32 // self.weight_bits

    def __repr__(self) -> str:
        return (
            f"AscendAWQConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, "
            f"zero_point={self.zero_point}, "
            f"modules_to_not_convert={self.modules_to_not_convert})"
        )

    @classmethod
    def get_name(cls) -> str:
        return AWQ_QUANTIZATION_METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return [
            "quant_config.json",
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AscendAWQConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        modules_to_not_convert = config.get("modules_to_not_convert")
        return cls(weight_bits, group_size, zero_point, modules_to_not_convert)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> LinearMethodBase | QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        if is_layer_skipped(
            prefix,
            self.modules_to_not_convert,
            self.packed_modules_mapping,
            skip_with_substr=True,
        ):
            return UnquantizedLinearMethod()
        from vllm_ascend.quantization.methods.awq import AscendAWQLinearMethod

        return AscendAWQLinearMethod(self)
