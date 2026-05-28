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

import gguf
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.gguf import (
    DEQUANT_TYPES,
    GGUFEmbeddingMethod,
    GGUFLinearMethod,
    UNQUANTIZED_TYPES,
)
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm


def _to_weight_type(qweight_type: int) -> gguf.GGMLQuantizationType:
    return gguf.GGMLQuantizationType(int(qweight_type))


def _format_weight_type(qweight_type: int) -> gguf.GGMLQuantizationType | int:
    try:
        return _to_weight_type(qweight_type)
    except ValueError:
        return int(qweight_type)


def _dequantize_gguf_weight(
    qweight: torch.Tensor,
    qweight_type: int,
    output_size: int,
    input_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if qweight_type in UNQUANTIZED_TYPES:
        return qweight.to(dtype=dtype)

    if qweight_type not in DEQUANT_TYPES:
        weight_type = _format_weight_type(qweight_type)
        raise NotImplementedError(f"Unsupported GGUF quantization type: {weight_type}")

    weight_type = _to_weight_type(qweight_type)
    dequantized = gguf.dequantize(qweight.detach().cpu().numpy(), weight_type)
    dense = torch.as_tensor(dequantized, device=qweight.device, dtype=dtype)
    return dense.reshape(output_size, -1)[:, :input_size].contiguous()


def _replace_parameter(layer: torch.nn.Module, name: str, tensor: torch.Tensor) -> None:
    if name in layer._parameters:
        del layer._parameters[name]
    layer.register_parameter(name, Parameter(tensor, requires_grad=False))


def _remove_parameter(layer: torch.nn.Module, name: str) -> None:
    if name in layer._parameters:
        del layer._parameters[name]
    layer.register_parameter(name, None)


class AscendGGUFLinearMethod(GGUFLinearMethod):
    """Ascend GGUF linear method.

    The first Ascend GGUF path materializes packed GGUF weights into dense
    tensors after loading, then reuses the existing Ascend unquantized GEMM.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        qweight_type = layer.qweight_type.weight_type
        if not (qweight_type in UNQUANTIZED_TYPES or qweight_type in DEQUANT_TYPES):
            weight_type = _format_weight_type(qweight_type)
            raise ValueError(f"Unsupported GGUF quantization type {weight_type} in layer {layer}.")

        self._create_padded_weight_param(layer)
        dense_weight = self._materialize_dense_weight(layer)
        _replace_parameter(layer, "weight", dense_weight)
        _remove_parameter(layer, "qweight")
        _remove_parameter(layer, "qweight_type")

    def _materialize_dense_weight(self, layer: torch.nn.Module) -> torch.Tensor:
        qweight = layer.qweight
        output_size, input_size = qweight.tensor_shape
        shard_id = qweight.shard_id

        if not shard_id:
            return _dequantize_gguf_weight(
                qweight,
                layer.qweight_type.weight_type,
                output_size,
                input_size,
                self.params_dtype,
            )

        ordered_shard_ids = ["q", "k", "v"] if "q" in shard_id else shard_id
        dense_shards = []
        for idx in ordered_shard_ids:
            start, end, offset = qweight.shard_offset_map[idx]
            shard_qweight = qweight[start:end, :offset].contiguous()
            dense_shards.append(
                _dequantize_gguf_weight(
                    shard_qweight,
                    layer.qweight_type.shard_weight_type[idx],
                    end - start,
                    input_size,
                    self.params_dtype,
                )
            )
        return torch.cat(dense_shards, dim=0).contiguous()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)


class AscendGGUFEmbeddingMethod(AscendGGUFLinearMethod, GGUFEmbeddingMethod):
    """Ascend GGUF embedding method using dense materialized weights."""

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, layer.weight)


__all__ = [
    "AscendGGUFEmbeddingMethod",
    "AscendGGUFLinearMethod",
]
