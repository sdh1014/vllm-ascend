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
import torch_npu
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.parameter import GroupQuantScaleParameter, PackedvLLMParameter

AWQ_REVERSE_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


def unpack_awq_int32(
    packed_weight: torch.Tensor,
    original_shape: torch.Size,
    num_bits: int = 4,
    packed_dim: int = 1,
) -> torch.Tensor:
    assert packed_weight.dtype == torch.int32, (
        f"Expecting `packed_weight.dtype` is torch.int32 but got {packed_weight.dtype}."
    )
    assert packed_dim in (0, 1), f"Expecting `packed_dim` is 0 or 1 but got {packed_dim}."

    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1
    if packed_dim == 1:
        unpacked = torch.empty(
            packed_weight.shape[0],
            packed_weight.shape[1] * pack_factor,
            device=packed_weight.device,
            dtype=torch.int32,
        )
        for i in range(pack_factor):
            unpacked[:, i::pack_factor] = (packed_weight >> (num_bits * i)) & mask
        unpacked = unpacked.view(packed_weight.shape[0], packed_weight.shape[1], pack_factor)
        unpacked = unpacked[:, :, AWQ_REVERSE_ORDER].reshape(
            packed_weight.shape[0], packed_weight.shape[1] * pack_factor
        )
        return unpacked[:, : original_shape[1]].contiguous()

    unpacked = torch.empty(
        packed_weight.shape[0] * pack_factor,
        packed_weight.shape[1],
        device=packed_weight.device,
        dtype=torch.int32,
    )
    for i in range(pack_factor):
        unpacked[i::pack_factor, :] = (packed_weight >> (num_bits * i)) & mask
    unpacked = unpacked.view(packed_weight.shape[0], pack_factor, packed_weight.shape[1])
    unpacked = unpacked[:, AWQ_REVERSE_ORDER, :].reshape(packed_weight.shape[0] * pack_factor, packed_weight.shape[1])
    return unpacked[: original_shape[0], :].contiguous()


def make_awq_zeros(qzeros: torch.Tensor, output_size: int, num_bits: int = 4) -> torch.Tensor:
    return unpack_awq_int32(
        qzeros,
        torch.Size([qzeros.shape[0], output_size]),
        num_bits=num_bits,
        packed_dim=1,
    )


def convert_awq_to_ascend(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    zero_point: bool,
    num_bits: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pack_factor = 32 // num_bits
    output_size = qweight.shape[1] * pack_factor
    unpacked_weight = unpack_awq_int32(
        qweight,
        torch.Size([qweight.shape[0], output_size]),
        num_bits=num_bits,
        packed_dim=1,
    )

    signed_weight = unpacked_weight - (1 << (num_bits - 1))
    packed_weight = torch_npu.npu_convert_weight_to_int4pack(signed_weight)

    if zero_point:
        zeros = make_awq_zeros(qzeros, output_size, num_bits=num_bits)
        offset = (1 << (num_bits - 1)) - zeros.to(dtype=scales.dtype, device=scales.device)
    else:
        offset = torch.zeros_like(scales)
    return packed_weight, scales.contiguous(), offset.contiguous()


class AscendAWQLinearMethod(LinearMethodBase):
    def __init__(self, quant_config: Any) -> None:
        self.quant_config = quant_config

    def _get_effective_group_size(self, input_size: int) -> int:
        return self.quant_config.group_size if self.quant_config.group_size != -1 else input_size

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        group_size = self._get_effective_group_size(input_size)
        if input_size_per_partition % group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized weight shape. "
                "This can be caused by too large tensor parallel size."
            )

        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized weight shape. "
                "This can be caused by too large tensor parallel size."
            )

        weight_loader = extra_weight_attrs.get("weight_loader")
        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        num_groups = input_size_per_partition // group_size
        qzeros = PackedvLLMParameter(
            data=torch.empty(
                num_groups,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )
        scales = GroupQuantScaleParameter(
            data=torch.empty(num_groups, output_size_per_partition, dtype=params_dtype),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader,
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data, requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)

        weight, scale, offset = convert_awq_to_ascend(
            layer.qweight.data,
            layer.qzeros.data,
            layer.scales.data,
            self.quant_config.zero_point,
            num_bits=self.quant_config.weight_bits,
        )
        layer.register_parameter("weight", torch.nn.Parameter(weight, requires_grad=False))
        layer.register_parameter("weight_scale", torch.nn.Parameter(scale, requires_grad=False))
        layer.register_parameter("weight_offset", torch.nn.Parameter(offset, requires_grad=False))

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch_npu.npu_weight_quant_batchmatmul(
            x=x,
            weight=layer.weight,
            antiquant_scale=layer.weight_scale.to(x.dtype),
            antiquant_offset=layer.weight_offset.to(x.dtype),
            antiquant_group_size=self._get_effective_group_size(x.shape[-1]),
            bias=bias,
        )
