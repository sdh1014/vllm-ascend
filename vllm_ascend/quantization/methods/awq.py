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

from collections.abc import Callable
from dataclasses import dataclass
from math import gcd
from typing import Any

import torch
import torch_npu
from torch.nn import Parameter
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.quantization.awq import AWQLinearMethod
from vllm.model_executor.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from vllm.model_executor.utils import set_weight_attrs

from vllm_ascend.quantization.quant_type import QuantType

AWQ_REVERSE_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)
AWQ_INT4PACK_INNER_K_TILES = 0
AWQ_TRITON_BLOCK_SIZE = 1024
AWQ_PACK_FACTOR = 8

logger = init_logger(__name__)


@dataclass(frozen=True)
class AWQGroupPlan:
    runtime_group_size: int
    repeat_factor: int
    expanded_group_start: int
    local_num_groups: int


def build_awq_group_plan(
    checkpoint_group_size: int,
    global_k: int,
    local_k: int,
    shard_start: int,
    *,
    tp_size: int = 1,
    tp_rank: int = 0,
    runtime_group_size: int | None = None,
) -> AWQGroupPlan:
    if checkpoint_group_size == -1:
        return AWQGroupPlan(
            runtime_group_size=0,
            repeat_factor=1,
            expanded_group_start=0,
            local_num_groups=1,
        )

    if runtime_group_size is None:
        runtime_group_size = gcd(gcd(checkpoint_group_size, local_k), shard_start)
    if (
        global_k % checkpoint_group_size != 0
        or runtime_group_size < 32
        or runtime_group_size % 32 != 0
        or checkpoint_group_size % runtime_group_size != 0
        or local_k % runtime_group_size != 0
        or shard_start % runtime_group_size != 0
    ):
        raise ValueError(
            "Unsupported AWQ TP group layout: "
            f"global_k={global_k}, local_k={local_k}, shard_start={shard_start}, "
            f"checkpoint_group_size={checkpoint_group_size}, tp_size={tp_size}, "
            f"tp_rank={tp_rank}"
        )

    return AWQGroupPlan(
        runtime_group_size=runtime_group_size,
        repeat_factor=checkpoint_group_size // runtime_group_size,
        expanded_group_start=shard_start // runtime_group_size,
        local_num_groups=local_k // runtime_group_size,
    )


def apply_awq_group_plan(
    loaded_weight: torch.Tensor,
    plan: AWQGroupPlan,
    *,
    group_dim: int,
) -> torch.Tensor:
    if loaded_weight.dim() == 0:
        loaded_weight = loaded_weight.reshape(1)
    expanded = loaded_weight.repeat_interleave(plan.repeat_factor, dim=group_dim)
    return expanded.narrow(group_dim, plan.expanded_group_start, plan.local_num_groups).contiguous()


def _awq_group_weight_loader(
    plan: AWQGroupPlan,
    *,
    group_dim: int = 0,
) -> Callable[[torch.nn.Parameter, torch.Tensor], bool | None]:
    def weight_loader(
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        *_args,
        return_success: bool = False,
        **_kwargs,
    ) -> bool | None:
        loaded_weight = apply_awq_group_plan(loaded_weight, plan, group_dim=group_dim)
        if param.data.shape != loaded_weight.shape:
            raise ValueError(
                f"Attempted to load AWQ group weight {tuple(loaded_weight.shape)} "
                f"into parameter {tuple(param.data.shape)}."
            )
        param.data.copy_(loaded_weight)
        return True if return_success else None

    return weight_loader


def _awq_moe_group_weight_loader(
    weight_loader: Callable,
    w13_plan: AWQGroupPlan,
    w2_plan: AWQGroupPlan,
) -> Callable:
    def wrapped_weight_loader(
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        if "qzeros" in weight_name or "scales" in weight_name:
            plan = w13_plan if "w13" in weight_name else w2_plan
            loaded_weight = apply_awq_group_plan(loaded_weight, plan, group_dim=0)
        return weight_loader(
            param,
            loaded_weight,
            weight_name,
            shard_id,
            expert_id,
            return_success=return_success,
        )

    return wrapped_weight_loader


def _set_awq_moe_weight_attrs(
    param: torch.nn.Parameter,
    extra_weight_attrs: dict[str, Any],
) -> None:
    attrs = {
        "is_transposed": True,
        **{k: v for k, v in extra_weight_attrs.items() if k != "weight_loader"},
    }
    set_weight_attrs(param, attrs)


def _set_awq_moe_group_attrs(
    param: torch.nn.Parameter,
    extra_weight_attrs: dict[str, Any],
) -> None:
    attrs = {
        "is_transposed": True,
        "quant_method": FusedMoeWeightScaleSupported.GROUP.value,
        **{k: v for k, v in extra_weight_attrs.items() if k != "weight_loader"},
    }
    set_weight_attrs(param, attrs)


def _validate_awq_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.dim() != 2:
        raise ValueError(f"{name} must be 2D, but got {tuple(tensor.shape)}.")
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must be torch.int32, but got {tensor.dtype}.")


def _triton_unavailable(exc: BaseException) -> bool:
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return True
    if isinstance(exc, RuntimeError):
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "backend is not available",
                "backend unavailable",
                "no available backend",
                "cannot find backend",
                "compiler is not available",
                "compiler unavailable",
                "no available compiler",
            )
        )
    return False


def _has_triton() -> bool:
    try:
        from vllm.triton_utils import HAS_TRITON
    except (ImportError, ModuleNotFoundError):
        return False
    return bool(HAS_TRITON)


def unpack_awq_int32(
    packed_weight: torch.Tensor,
    original_shape: torch.Size,
    *,
    packed_dim: int = 1,
) -> torch.Tensor:
    _validate_awq_tensor("packed_weight", packed_weight)
    if packed_dim not in (0, 1):
        raise ValueError(f"packed_dim must be 0 or 1, but got {packed_dim}.")

    if packed_dim == 1:
        unpacked = torch.empty(
            packed_weight.shape[0],
            packed_weight.shape[1] * AWQ_PACK_FACTOR,
            device=packed_weight.device,
            dtype=torch.int32,
        )
        for index, source_index in enumerate(AWQ_REVERSE_ORDER):
            unpacked[:, index::AWQ_PACK_FACTOR] = (packed_weight >> (4 * source_index)) & 0xF
        return unpacked[:, : original_shape[1]].contiguous()

    unpacked = torch.empty(
        packed_weight.shape[0] * AWQ_PACK_FACTOR,
        packed_weight.shape[1],
        device=packed_weight.device,
        dtype=torch.int32,
    )
    for index, source_index in enumerate(AWQ_REVERSE_ORDER):
        unpacked[index::AWQ_PACK_FACTOR, :] = (packed_weight >> (4 * source_index)) & 0xF
    return unpacked[: original_shape[0], :].contiguous()


def make_awq_zeros(qzeros: torch.Tensor, output_size: int) -> torch.Tensor:
    return unpack_awq_int32(qzeros, torch.Size([qzeros.shape[0], output_size]), packed_dim=1)


def pack_awq_weight_to_ascend_reference(
    qweight: torch.Tensor,
    output_size: int,
    *,
    inner_k_tiles: int = AWQ_INT4PACK_INNER_K_TILES,
) -> torch.Tensor:
    unpacked_weight = unpack_awq_int32(
        qweight,
        torch.Size([qweight.shape[0], output_size]),
        packed_dim=1,
    )
    unpacked_weight = unpacked_weight.sub_(8)
    return torch_npu.npu_convert_weight_to_int4pack(unpacked_weight, inner_k_tiles=inner_k_tiles)


def pack_awq_weight_to_ascend(
    qweight: torch.Tensor,
    output_size: int,
    *,
    inner_k_tiles: int = AWQ_INT4PACK_INNER_K_TILES,
) -> torch.Tensor:
    _validate_awq_tensor("qweight", qweight)
    if (
        inner_k_tiles == AWQ_INT4PACK_INNER_K_TILES
        and qweight.device.type == "npu"
        and _has_triton()
    ):
        try:
            from vllm_ascend.ops.triton.awq_conversion import awq_direct_pack
            return awq_direct_pack(qweight, block_size=AWQ_TRITON_BLOCK_SIZE)
        except (ImportError, ModuleNotFoundError) as exc:
            logger.warning_once("AWQ direct Triton pack unavailable: %s", exc)
        except RuntimeError as exc:
            if not _triton_unavailable(exc):
                raise
            logger.warning_once("AWQ direct Triton pack unavailable: %s", exc)

    return pack_awq_weight_to_ascend_reference(
        qweight,
        output_size,
        inner_k_tiles=inner_k_tiles,
    )


def prepare_awq_scale(scales: torch.Tensor) -> torch.Tensor:
    if scales.dim() != 2:
        raise ValueError(f"scales must be 2D, but got {tuple(scales.shape)}.")
    return scales.contiguous()


def prepare_awq_zero_offset_reference(
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    output_size: int,
    zero_point: bool,
) -> torch.Tensor:
    if not zero_point:
        return torch.zeros_like(scales).contiguous()

    zeros = make_awq_zeros(qzeros, output_size)
    # (q - 8 + 8 - zero) * scale == (q - zero) * scale
    return zeros.to(dtype=scales.dtype, device=scales.device).neg_().add_(8).contiguous()


def prepare_awq_zero_offset(
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    output_size: int,
    zero_point: bool,
) -> torch.Tensor:
    if not zero_point:
        return torch.zeros_like(scales).contiguous()

    _validate_awq_tensor("qzeros", qzeros)
    if qzeros.device.type == "npu" and scales.device.type == "npu" and _has_triton():
        try:
            from vllm_ascend.ops.triton.awq_conversion import awq_zero_offset_triton
            return awq_zero_offset_triton(qzeros, scales, output_size)
        except (ImportError, ModuleNotFoundError) as exc:
            logger.warning_once("AWQ Triton zero offset unavailable: %s", exc)
        except RuntimeError as exc:
            if not _triton_unavailable(exc):
                raise
            logger.warning_once("AWQ Triton zero offset unavailable: %s", exc)

    return prepare_awq_zero_offset_reference(qzeros, scales, output_size, zero_point=True)


def convert_awq_to_ascend_reference(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    zero_point: bool,
    *,
    inner_k_tiles: int = AWQ_INT4PACK_INNER_K_TILES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output_size = qweight.shape[1] * AWQ_PACK_FACTOR
    packed_weight = pack_awq_weight_to_ascend_reference(
        qweight,
        output_size,
        inner_k_tiles=inner_k_tiles,
    )
    scale = prepare_awq_scale(scales)
    offset = prepare_awq_zero_offset_reference(qzeros, scales, output_size, zero_point)
    return packed_weight, scale, offset


def convert_awq_to_ascend(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    zero_point: bool,
    *,
    inner_k_tiles: int = AWQ_INT4PACK_INNER_K_TILES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_awq_tensor("qweight", qweight)
    if zero_point:
        _validate_awq_tensor("qzeros", qzeros)
    if scales.dim() != 2:
        raise ValueError(f"scales must be 2D, but got {tuple(scales.shape)}.")

    output_size = qweight.shape[1] * AWQ_PACK_FACTOR
    if (
        zero_point
        and inner_k_tiles == AWQ_INT4PACK_INNER_K_TILES
        and qweight.device.type == "npu"
        and qzeros.device.type == "npu"
        and scales.device.type == "npu"
        and _has_triton()
    ):
        try:
            from vllm_ascend.ops.triton.awq_conversion import awq_pack_zero_triton

            packed_weight, offset = awq_pack_zero_triton(
                qweight,
                qzeros,
                scales,
                output_size,
                block_size=AWQ_TRITON_BLOCK_SIZE,
            )
            return packed_weight, prepare_awq_scale(scales), offset
        except (ImportError, ModuleNotFoundError) as exc:
            logger.warning_once("AWQ fused Triton pack-zero unavailable: %s", exc)
        except RuntimeError as exc:
            if not _triton_unavailable(exc):
                raise
            logger.warning_once("AWQ fused Triton pack-zero unavailable: %s", exc)

    packed_weight = pack_awq_weight_to_ascend(
        qweight,
        output_size,
        inner_k_tiles=inner_k_tiles,
    )
    scale = prepare_awq_scale(scales)
    offset = prepare_awq_zero_offset(qzeros, scales, output_size, zero_point)
    return packed_weight, scale, offset


def convert_awq_moe_param_to_ascend(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    output_size: int,
    zero_point: bool,
    *,
    inner_k_tiles: int = AWQ_INT4PACK_INNER_K_TILES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    experts, input_size, _ = qweight.shape
    groups_per_expert = scales.shape[1]
    packed_weight, scale, offset = convert_awq_to_ascend(
        qweight.flatten(0, 1),
        qzeros.flatten(0, 1),
        scales.flatten(0, 1),
        zero_point=zero_point,
        inner_k_tiles=inner_k_tiles,
    )
    return (
        packed_weight.view(experts, input_size, -1),
        scale.view(experts, groups_per_expert, output_size),
        offset.view(experts, groups_per_expert, output_size),
    )


def _delete_parameter(layer: torch.nn.Module, name: str) -> None:
    if name in layer._parameters:
        del layer._parameters[name]
    elif hasattr(layer, name):
        delattr(layer, name)


class AscendAWQLinearMethod(AWQLinearMethod):
    def _normalized_group_size(self, input_size: int) -> int:
        if self.quant_config.group_size == -1:
            return input_size
        return self.quant_config.group_size

    def _can_use_base_awq_loader(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        input_size: int,
    ) -> bool:
        group_size = self._normalized_group_size(input_size)
        if input_size_per_partition % group_size != 0:
            return False

        if not isinstance(layer, RowParallelLinear):
            return True

        shard_start = getattr(layer, "tp_rank", 0) * input_size_per_partition
        return shard_start % group_size == 0

    def _get_row_parallel_group_plan(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        input_size: int,
    ) -> AWQGroupPlan:
        tp_rank = getattr(layer, "tp_rank", 0)
        tp_size = getattr(layer, "tp_size", 1)
        shard_start = tp_rank * input_size_per_partition
        return build_awq_group_plan(
            self.quant_config.group_size,
            input_size,
            input_size_per_partition,
            shard_start,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )

    def _set_runtime_group_size(
        self,
        layer: torch.nn.Module,
        plan: AWQGroupPlan | None,
        input_size: int,
    ) -> None:
        if plan is not None:
            layer.awq_runtime_group_size = (
                0 if plan.local_num_groups == 1 else plan.runtime_group_size
            )
            return
        if self.quant_config.group_size == -1 or self.quant_config.group_size == input_size:
            layer.awq_runtime_group_size = 0
        else:
            layer.awq_runtime_group_size = self.quant_config.group_size

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
        if self._can_use_base_awq_loader(layer, input_size_per_partition, input_size):
            self._set_runtime_group_size(layer, None, input_size)
            return super().create_weights(
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                **extra_weight_attrs,
            )

        plan = self._get_row_parallel_group_plan(layer, input_size_per_partition, input_size)
        self._set_runtime_group_size(layer, plan, input_size)
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
        qzeros = PackedvLLMParameter(
            data=torch.empty(
                plan.local_num_groups,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=_awq_group_weight_loader(plan),
        )
        scales = GroupQuantScaleParameter(
            data=torch.empty(plan.local_num_groups, output_size_per_partition, dtype=params_dtype),
            input_dim=0,
            output_dim=1,
            weight_loader=_awq_group_weight_loader(plan),
        )
        layer.register_parameter("qweight", qweight)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight, scale, offset = convert_awq_to_ascend(
            layer.qweight.data,
            layer.qzeros.data,
            layer.scales.data,
            self.quant_config.zero_point,
        )
        layer.register_parameter("weight", Parameter(weight.contiguous(), requires_grad=False))
        layer.register_parameter("weight_scale", Parameter(scale.contiguous(), requires_grad=False))
        layer.register_parameter("weight_offset", Parameter(offset.contiguous(), requires_grad=False))
        for name in ("qweight", "qzeros", "scales"):
            _delete_parameter(layer, name)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch_npu.npu_weight_quant_batchmatmul(
            x=x,
            weight=layer.weight,
            antiquant_scale=layer.weight_scale,
            antiquant_offset=layer.weight_offset,
            antiquant_group_size=getattr(layer, "awq_runtime_group_size", 0),
            bias=bias,
        )


class AscendAWQFusedMoEMethod(FusedMoEMethodBase):
    quant_type: QuantType = QuantType.W4A16

    def __init__(self, quant_config: Any, moe_config: FusedMoEConfig) -> None:
        super().__init__(moe_config)
        self.quant_config = quant_config

    @property
    def supports_eplb(self) -> bool:
        return True

    def _get_moe_tp(self, layer: torch.nn.Module) -> tuple[int, int]:
        moe_config = getattr(layer, "moe_config", None)
        tp_size = getattr(moe_config, "tp_size", getattr(layer, "tp_size", 1))
        tp_rank = getattr(moe_config, "tp_rank", getattr(layer, "tp_rank", 0))
        return int(tp_size), int(tp_rank)

    def _get_moe_runtime_group_size(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        w2_shard_start: int,
    ) -> int:
        checkpoint_group_size = self.quant_config.group_size
        if checkpoint_group_size == -1:
            return 0
        return gcd(
            gcd(gcd(checkpoint_group_size, hidden_size), intermediate_size_per_partition),
            w2_shard_start,
        )

    def _get_moe_group_plans(
        self,
        layer: torch.nn.Module,
        hidden_size: int,
        intermediate_size_per_partition: int,
        intermediate_size_full: int,
    ) -> tuple[AWQGroupPlan, AWQGroupPlan]:
        tp_size, tp_rank = self._get_moe_tp(layer)
        w2_shard_start = tp_rank * intermediate_size_per_partition
        runtime_group_size = self._get_moe_runtime_group_size(
            hidden_size,
            intermediate_size_per_partition,
            w2_shard_start,
        )
        w13_plan = build_awq_group_plan(
            self.quant_config.group_size,
            hidden_size,
            hidden_size,
            0,
            tp_size=tp_size,
            tp_rank=tp_rank,
            runtime_group_size=runtime_group_size,
        )
        w2_plan = build_awq_group_plan(
            self.quant_config.group_size,
            intermediate_size_full,
            intermediate_size_per_partition,
            w2_shard_start,
            tp_size=tp_size,
            tp_rank=tp_rank,
            runtime_group_size=runtime_group_size,
        )
        return w13_plan, w2_plan

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        tp_size, _tp_rank = self._get_moe_tp(layer)
        intermediate_size_full = extra_weight_attrs.pop(
            "intermediate_size_full",
            intermediate_size_per_partition * tp_size,
        )
        w13_plan, w2_plan = self._get_moe_group_plans(
            layer,
            hidden_size,
            intermediate_size_per_partition,
            intermediate_size_full,
        )
        w13_output_size = 2 * intermediate_size_per_partition
        w2_output_size = hidden_size
        weight_loader = extra_weight_attrs.get("weight_loader")
        group_weight_loader = (
            _awq_moe_group_weight_loader(weight_loader, w13_plan, w2_plan)
            if weight_loader is not None
            else None
        )

        w13_qweight = PackedvLLMParameter(
            data=torch.empty(
                num_experts,
                hidden_size,
                w13_output_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=1,
            output_dim=2,
            packed_dim=2,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )
        w2_qweight = PackedvLLMParameter(
            data=torch.empty(
                num_experts,
                intermediate_size_per_partition,
                w2_output_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=1,
            output_dim=2,
            packed_dim=2,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )
        w13_qzeros = PackedvLLMParameter(
            data=torch.empty(
                num_experts,
                w13_plan.local_num_groups,
                w13_output_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=1,
            output_dim=2,
            packed_dim=2,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=group_weight_loader,
        )
        w2_qzeros = PackedvLLMParameter(
            data=torch.empty(
                num_experts,
                w2_plan.local_num_groups,
                w2_output_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=1,
            output_dim=2,
            packed_dim=2,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=group_weight_loader,
        )
        w13_scales = GroupQuantScaleParameter(
            data=torch.empty(num_experts, w13_plan.local_num_groups, w13_output_size, dtype=params_dtype),
            input_dim=1,
            output_dim=2,
            weight_loader=group_weight_loader,
        )
        w2_scales = GroupQuantScaleParameter(
            data=torch.empty(num_experts, w2_plan.local_num_groups, w2_output_size, dtype=params_dtype),
            input_dim=1,
            output_dim=2,
            weight_loader=group_weight_loader,
        )

        layer.register_parameter("w13_qweight", w13_qweight)
        _set_awq_moe_weight_attrs(w13_qweight, extra_weight_attrs)
        layer.register_parameter("w13_qzeros", w13_qzeros)
        _set_awq_moe_group_attrs(w13_qzeros, extra_weight_attrs)
        layer.register_parameter("w13_scales", w13_scales)
        _set_awq_moe_group_attrs(w13_scales, extra_weight_attrs)
        layer.register_parameter("w2_qweight", w2_qweight)
        _set_awq_moe_weight_attrs(w2_qweight, extra_weight_attrs)
        layer.register_parameter("w2_qzeros", w2_qzeros)
        _set_awq_moe_group_attrs(w2_qzeros, extra_weight_attrs)
        set_weight_attrs(w2_qzeros, {"load_full_w2": True})
        layer.register_parameter("w2_scales", w2_scales)
        _set_awq_moe_group_attrs(w2_scales, extra_weight_attrs)
        set_weight_attrs(w2_scales, {"load_full_w2": True})
        layer.awq_runtime_group_size = w2_plan.runtime_group_size

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w13_output_size = layer.w13_qweight.shape[2] * self.quant_config.pack_factor
        w2_output_size = layer.w2_qweight.shape[2] * self.quant_config.pack_factor
        w13_weight, w13_scale, w13_offset = convert_awq_moe_param_to_ascend(
            layer.w13_qweight.data,
            layer.w13_qzeros.data,
            layer.w13_scales.data,
            w13_output_size,
            self.quant_config.zero_point,
        )
        w2_weight, w2_scale, w2_offset = convert_awq_moe_param_to_ascend(
            layer.w2_qweight.data,
            layer.w2_qzeros.data,
            layer.w2_scales.data,
            w2_output_size,
            self.quant_config.zero_point,
        )

        layer.register_parameter("w13_weight_packed", Parameter(w13_weight.contiguous(), requires_grad=False))
        layer.register_parameter("w2_weight_packed", Parameter(w2_weight.contiguous(), requires_grad=False))
        layer.register_parameter("w13_weight_scale", Parameter(w13_scale.contiguous(), requires_grad=False))
        layer.register_parameter("w2_weight_scale", Parameter(w2_scale.contiguous(), requires_grad=False))
        layer.register_parameter("w13_weight_offset", Parameter(w13_offset.contiguous(), requires_grad=False))
        layer.register_parameter("w2_weight_offset", Parameter(w2_offset.contiguous(), requires_grad=False))
        for name in (
            "w13_qweight",
            "w13_qzeros",
            "w13_scales",
            "w2_qweight",
            "w2_qzeros",
            "w2_scales",
        ):
            _delete_parameter(layer, name)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        return None

    @property
    def is_monolithic(self) -> bool:
        return True

    def _apply_ascend_moe(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        from vllm_ascend.quantization.methods.w4a16 import apply_ascend_w4a16_fused_moe

        return apply_ascend_w4a16_fused_moe(
            layer,
            *args,
            quant_type=self.quant_type,
            w13_weight=layer.w13_weight_packed,
            w2_weight=layer.w2_weight_packed,
            w13_weight_scale=layer.w13_weight_scale,
            w2_weight_scale=layer.w2_weight_scale,
            w13_weight_offset=layer.w13_weight_offset,
            w2_weight_offset=layer.w2_weight_offset,
            **kwargs,
        )

    def apply(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        kwargs.setdefault("tid2eid", getattr(layer, "tid2eid", None))
        kwargs.setdefault("dynamic_eplb", getattr(layer, "dynamic_eplb", False))
        return self._apply_ascend_moe(layer, *args, **kwargs)

    def apply_monolithic(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if router_logits is None:
            raise ValueError("AWQ MoE monolithic apply requires router logits.")

        return self._apply_ascend_moe(
            layer,
            x,
            router_logits,
            top_k=layer.top_k,
            renormalize=layer.renormalize,
            use_grouped_topk=layer.use_grouped_topk,
            num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            topk_group=layer.topk_group,
            num_expert_group=layer.num_expert_group,
            custom_routing_function=layer.custom_routing_function,
            scoring_func=layer.scoring_func,
            routed_scaling_factor=layer.routed_scaling_factor,
            e_score_correction_bias=layer.e_score_correction_bias,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            log2phy=getattr(layer, "log2phy", None),
            global_redundant_expert_num=getattr(layer.moe_config, "global_redundant_expert_num", 0),
            mc2_mask=getattr(layer, "mc2_mask", None),
            tid2eid=getattr(layer, "tid2eid", None),
            dynamic_eplb=getattr(layer, "dynamic_eplb", False),
            input_ids=input_ids,
        )
