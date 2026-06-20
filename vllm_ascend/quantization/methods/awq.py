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

import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

import torch
import torch_npu
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase, FusedMoeWeightScaleSupported
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.linear import LinearMethodBase, RowParallelLinear
from vllm.model_executor.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from vllm.model_executor.utils import set_weight_attrs

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input
from vllm_ascend.quantization.quant_type import QuantType

AWQ_REVERSE_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)
AWQ_INT4PACK_INNER_K_TILES = 0
AWQ_TRITON_BLOCK_SIZE = 1024
T = TypeVar("T")
logger = logging.getLogger(__name__)
_AWQ_TRITON_WEIGHT_PACK_WARNINGS: set[str] = set()
_AWQ_TRITON_ZERO_OFFSET_WARNINGS: set[str] = set()
_AWQ_TRITON_PACK_ZERO_WARNINGS: set[str] = set()


def _awq_row_group_weight_loader(
    group_start: int,
    local_num_groups: int,
) -> Callable[[torch.nn.Parameter, torch.Tensor], None]:
    def weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor, *_args, **_kwargs) -> None:
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)
        loaded_weight = loaded_weight.narrow(0, group_start, local_num_groups)
        assert param.data.shape == loaded_weight.shape, (
            f"Attempted to load AWQ group weight ({loaded_weight.size()}) "
            f"into parameter ({param.data.size()})"
        )
        param.data.copy_(loaded_weight)

    return weight_loader


def _set_awq_moe_weight_attrs(param: torch.nn.Parameter, extra_weight_attrs: dict[str, Any]) -> None:
    attrs = {"is_transposed": True, **{k: v for k, v in extra_weight_attrs.items() if k != "weight_loader"}}
    set_weight_attrs(param, attrs)


def _set_awq_moe_group_attrs(param: torch.nn.Parameter, extra_weight_attrs: dict[str, Any]) -> None:
    attrs = {
        "is_transposed": True,
        "quant_method": FusedMoeWeightScaleSupported.GROUP.value,
        **{k: v for k, v in extra_weight_attrs.items() if k != "weight_loader"},
    }
    set_weight_attrs(param, attrs)


def _log_awq_triton_weight_pack_warning_once(message: str) -> None:
    if message in _AWQ_TRITON_WEIGHT_PACK_WARNINGS:
        logger.debug(message)
        return
    _AWQ_TRITON_WEIGHT_PACK_WARNINGS.add(message)
    logger.warning(message)


def _log_awq_triton_zero_offset_warning_once(message: str) -> None:
    if message in _AWQ_TRITON_ZERO_OFFSET_WARNINGS:
        logger.debug(message)
        return
    _AWQ_TRITON_ZERO_OFFSET_WARNINGS.add(message)
    logger.warning(message)


def _log_awq_triton_pack_zero_warning_once(message: str) -> None:
    if message in _AWQ_TRITON_PACK_ZERO_WARNINGS:
        logger.debug(message)
        return
    _AWQ_TRITON_PACK_ZERO_WARNINGS.add(message)
    logger.warning(message)


def _try_awq_triton_weight_pack(qweight: torch.Tensor, block_size: int) -> torch.Tensor | None:
    try:
        from vllm_ascend.ops.triton.awq_direct_pack import awq_direct_pack_candidate, triton_kernel_launchable
    except Exception as exc:
        _log_awq_triton_weight_pack_warning_once(f"AWQ Triton-Ascend weight pack import failed: {exc}")
        return None

    try:
        if not triton_kernel_launchable():
            _log_awq_triton_weight_pack_warning_once(
                "AWQ Triton-Ascend weight pack skipped because the kernel is not launchable."
            )
            return None
        return awq_direct_pack_candidate(qweight, block_size=block_size)
    except Exception as exc:
        _log_awq_triton_weight_pack_warning_once(
            "AWQ Triton-Ascend weight pack failed for "
            f"qweight_shape={list(qweight.shape)} block_size={block_size}: {exc}"
        )
        return None


def _try_awq_triton_zero_offset(qzeros: torch.Tensor, scales: torch.Tensor, output_size: int) -> torch.Tensor | None:
    try:
        from vllm.triton_utils import HAS_TRITON

        if not HAS_TRITON:
            return None
        from vllm_ascend.ops.triton.awq_zero_offset import awq_zero_offset_triton
    except Exception as exc:
        _log_awq_triton_zero_offset_warning_once(f"AWQ Triton-Ascend zero offset import failed: {exc}")
        return None

    try:
        return awq_zero_offset_triton(qzeros, scales, output_size)
    except Exception as exc:
        _log_awq_triton_zero_offset_warning_once(
            "AWQ Triton-Ascend zero offset failed for "
            f"qzeros_shape={list(qzeros.shape)} scales_shape={list(scales.shape)} output_size={output_size}: {exc}"
        )
        return None


def _try_awq_triton_pack_zero(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    output_size: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    try:
        from vllm.triton_utils import HAS_TRITON

        if not HAS_TRITON:
            return None
        from vllm_ascend.ops.triton.awq_pack_zero import awq_pack_zero_triton
    except Exception as exc:
        _log_awq_triton_pack_zero_warning_once(f"AWQ Triton-Ascend fused pack-zero import failed: {exc}")
        return None

    try:
        return awq_pack_zero_triton(qweight, qzeros, scales, output_size, block_size=block_size)
    except Exception as exc:
        _log_awq_triton_pack_zero_warning_once(
            "AWQ Triton-Ascend fused pack-zero failed for "
            f"qweight_shape={list(qweight.shape)} qzeros_shape={list(qzeros.shape)} "
            f"scales_shape={list(scales.shape)} output_size={output_size}: {exc}"
        )
        return None


def _conversion_memory_stats(device: torch.device) -> dict[str, int | None]:
    if device.type != "npu":
        return {
            "allocated_bytes": None,
            "reserved_bytes": None,
            "peak_allocated_bytes": None,
        }

    return {
        "allocated_bytes": int(torch.npu.memory_allocated(str(device))),
        "reserved_bytes": int(torch.npu.memory_reserved(str(device))),
        "peak_allocated_bytes": int(torch.npu.max_memory_allocated(str(device))),
    }


def _conversion_memory_delta(
    before: dict[str, int | None],
    after: dict[str, int | None],
) -> dict[str, int | None]:
    allocated_before = before["allocated_bytes"]
    allocated_after = after["allocated_bytes"]
    peak_after = after["peak_allocated_bytes"]
    return {
        "allocated_bytes": None
        if allocated_before is None or allocated_after is None
        else allocated_after - allocated_before,
        "peak_allocated_bytes": None
        if allocated_before is None or peak_after is None
        else peak_after - allocated_before,
    }


def _run_conversion_stage(
    name: str,
    reference_tensor: torch.Tensor,
    breakdown: list[dict[str, Any]] | None,
    fn: Callable[[], T],
    config: dict[str, Any] | None = None,
) -> T:
    if breakdown is None:
        return fn()

    device = reference_tensor.device
    use_npu_events = device.type == "npu"
    if use_npu_events:
        torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        before = _conversion_memory_stats(device)
        start.record()
    else:
        start = None
        end = None
        before = _conversion_memory_stats(device)

    wall_start = time.perf_counter()
    result = fn()

    npu_event_ms = None
    if use_npu_events:
        assert start is not None and end is not None
        end.record()
        torch.npu.synchronize()
        npu_event_ms = float(start.elapsed_time(end))

    wall_ms = (time.perf_counter() - wall_start) * 1000
    after = _conversion_memory_stats(device)
    stage = {
        "name": name,
        "wall_ms": wall_ms,
        "npu_event_ms": npu_event_ms,
        "memory": {
            "before": before,
            "after": after,
            "delta": _conversion_memory_delta(before, after),
        },
    }
    if config is not None:
        stage["config"] = config
    breakdown.append(stage)
    return result


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
            source_index = AWQ_REVERSE_ORDER[i]
            unpacked[:, i::pack_factor] = (packed_weight >> (num_bits * source_index)) & mask
        return unpacked[:, : original_shape[1]].contiguous()

    unpacked = torch.empty(
        packed_weight.shape[0] * pack_factor,
        packed_weight.shape[1],
        device=packed_weight.device,
        dtype=torch.int32,
    )
    for i in range(pack_factor):
        source_index = AWQ_REVERSE_ORDER[i]
        unpacked[i::pack_factor, :] = (packed_weight >> (num_bits * source_index)) & mask
    return unpacked[: original_shape[0], :].contiguous()


def make_awq_zeros(qzeros: torch.Tensor, output_size: int, num_bits: int = 4) -> torch.Tensor:
    return unpack_awq_int32(
        qzeros,
        torch.Size([qzeros.shape[0], output_size]),
        num_bits=num_bits,
        packed_dim=1,
    )


def pack_awq_weight_to_ascend(
    qweight: torch.Tensor,
    output_size: int,
    num_bits: int = 4,
    *,
    breakdown: list[dict[str, Any]] | None = None,
    inner_k_tiles: int = AWQ_INT4PACK_INNER_K_TILES,
) -> torch.Tensor:
    triton_breakdown = [] if breakdown is not None else None
    triton_weight = _run_conversion_stage(
        "pack_weight",
        qweight,
        triton_breakdown,
        lambda: _try_awq_triton_weight_pack(qweight, AWQ_TRITON_BLOCK_SIZE),
        config={
            "source": "triton_direct_pack",
            "candidate": "awq_direct_pack_candidate",
            "block_size": AWQ_TRITON_BLOCK_SIZE,
        },
    )
    if triton_weight is not None:
        if breakdown is not None and triton_breakdown is not None:
            breakdown.extend(triton_breakdown)
        return triton_weight

    unpacked_weight = _run_conversion_stage(
        "unpack_weight",
        qweight,
        breakdown,
        lambda: unpack_awq_int32(
            qweight,
            torch.Size([qweight.shape[0], output_size]),
            num_bits=num_bits,
            packed_dim=1,
        ),
    )
    _run_conversion_stage(
        "sign_weight",
        unpacked_weight,
        breakdown,
        lambda: unpacked_weight.sub_(1 << (num_bits - 1)),
    )
    return _run_conversion_stage(
        "pack_weight",
        unpacked_weight,
        breakdown,
        lambda: torch_npu.npu_convert_weight_to_int4pack(unpacked_weight, inner_k_tiles=inner_k_tiles),
        config={"inner_k_tiles": inner_k_tiles},
    )


def prepare_awq_scale(
    scales: torch.Tensor,
    *,
    breakdown: list[dict[str, Any]] | None = None,
) -> torch.Tensor:
    return _run_conversion_stage("prepare_scale", scales, breakdown, scales.contiguous)


def prepare_awq_zero_offset(
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    output_size: int,
    zero_point: bool,
    num_bits: int = 4,
    *,
    breakdown: list[dict[str, Any]] | None = None,
) -> torch.Tensor:
    if not zero_point:
        return _run_conversion_stage("prepare_offset", scales, breakdown, lambda: torch.zeros_like(scales))

    if num_bits == 4 and qzeros.device.type == "npu" and scales.device.type == "npu":
        triton_breakdown = [] if breakdown is not None else None
        triton_offset = _run_conversion_stage(
            "prepare_offset",
            qzeros,
            triton_breakdown,
            lambda: _try_awq_triton_zero_offset(qzeros, scales, output_size),
            config={"source": "triton_awq_zero_offset"},
        )
        if triton_offset is not None:
            if breakdown is not None and triton_breakdown is not None:
                breakdown.extend(triton_breakdown)
            return triton_offset

    zeros = _run_conversion_stage(
        "unpack_zeros",
        qzeros,
        breakdown,
        lambda: make_awq_zeros(qzeros, output_size, num_bits=num_bits),
    )
    return _run_conversion_stage(
        "prepare_offset",
        zeros,
        breakdown,
        lambda: zeros.to(dtype=scales.dtype, device=scales.device).neg_().add_(1 << (num_bits - 1)).contiguous(),
    )


def convert_awq_to_ascend(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    zero_point: bool,
    num_bits: int = 4,
    *,
    breakdown: list[dict[str, Any]] | None = None,
    inner_k_tiles: int = AWQ_INT4PACK_INNER_K_TILES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pack_factor = 32 // num_bits
    output_size = qweight.shape[1] * pack_factor
    if (
        zero_point
        and num_bits == 4
        and inner_k_tiles == AWQ_INT4PACK_INNER_K_TILES
        and qweight.device.type == "npu"
        and qzeros.device.type == "npu"
        and scales.device.type == "npu"
    ):
        triton_breakdown = [] if breakdown is not None else None
        fused_output = _run_conversion_stage(
            "pack_weight_zero_offset",
            qweight,
            triton_breakdown,
            lambda: _try_awq_triton_pack_zero(qweight, qzeros, scales, output_size, AWQ_TRITON_BLOCK_SIZE),
            config={
                "source": "triton_awq_pack_zero",
                "block_size": AWQ_TRITON_BLOCK_SIZE,
            },
        )
        if fused_output is not None:
            if breakdown is not None and triton_breakdown is not None:
                breakdown.extend(triton_breakdown)
            packed_weight, offset = fused_output
            scale = prepare_awq_scale(scales, breakdown=breakdown)
            return packed_weight, scale, offset

    packed_weight = pack_awq_weight_to_ascend(
        qweight,
        output_size,
        num_bits=num_bits,
        breakdown=breakdown,
        inner_k_tiles=inner_k_tiles,
    )
    scale = prepare_awq_scale(scales, breakdown=breakdown)
    offset = prepare_awq_zero_offset(
        qzeros,
        scales,
        output_size,
        zero_point,
        num_bits=num_bits,
        breakdown=breakdown,
    )
    return packed_weight, scale, offset


def convert_awq_to_ascend_batch(
    qweights: list[torch.Tensor],
    qzeros: list[torch.Tensor] | None,
    scales: list[torch.Tensor],
    zero_point: bool,
    num_bits: int = 4,
    *,
    breakdown: list[dict[str, Any]] | None = None,
    inner_k_tiles: int = AWQ_INT4PACK_INNER_K_TILES,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    reference_qweight = qweights[0]
    reference_scale = scales[0]
    for qweight, scale in zip(qweights, scales, strict=True):
        if qweight.shape != reference_qweight.shape or qweight.dtype != reference_qweight.dtype:
            raise ValueError("Batched AWQ conversion requires qweight tensors with identical shape and dtype.")
        if qweight.device != reference_qweight.device:
            raise ValueError("Batched AWQ conversion requires qweight tensors on the same device.")
        if scale.shape != reference_scale.shape or scale.dtype != reference_scale.dtype:
            raise ValueError("Batched AWQ conversion requires scale tensors with identical shape and dtype.")
        if scale.device != reference_scale.device:
            raise ValueError("Batched AWQ conversion requires scale tensors on the same device.")

    if zero_point:
        assert qzeros is not None
        reference_qzero = qzeros[0]
        for qzero in qzeros:
            if qzero.shape != reference_qzero.shape or qzero.dtype != reference_qzero.dtype:
                raise ValueError("Batched AWQ conversion requires qzeros tensors with identical shape and dtype.")
            if qzero.device != reference_qzero.device:
                raise ValueError("Batched AWQ conversion requires qzeros tensors on the same device.")

    qweight = torch.cat(qweights, dim=0)
    scale = torch.cat(scales, dim=0)
    qzero = torch.cat(qzeros, dim=0) if zero_point and qzeros is not None else qweight.new_empty(0)

    packed_weight, packed_scale, packed_offset = convert_awq_to_ascend(
        qweight,
        qzero,
        scale,
        zero_point=zero_point,
        num_bits=num_bits,
        breakdown=breakdown,
        inner_k_tiles=inner_k_tiles,
    )

    weight_sizes = [item.shape[0] for item in qweights]
    scale_sizes = [item.shape[0] for item in scales]
    return list(
        zip(
            packed_weight.split(weight_sizes, dim=0),
            packed_scale.split(scale_sizes, dim=0),
            packed_offset.split(scale_sizes, dim=0),
            strict=True,
        )
    )


def convert_awq_moe_param_to_ascend(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    output_size: int,
    zero_point: bool,
    num_bits: int = 4,
    *,
    breakdown: list[dict[str, Any]] | None = None,
    inner_k_tiles: int = AWQ_INT4PACK_INNER_K_TILES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    experts, input_size, _ = qweight.shape
    groups_per_expert = scales.shape[1]
    packed_weight, scale, offset = convert_awq_to_ascend(
        qweight.flatten(0, 1),
        qzeros.flatten(0, 1),
        scales.flatten(0, 1),
        zero_point=zero_point,
        num_bits=num_bits,
        breakdown=breakdown,
        inner_k_tiles=inner_k_tiles,
    )
    return (
        packed_weight.view(experts, input_size, -1),
        scale.view(experts, groups_per_expert, output_size),
        offset.view(experts, groups_per_expert, output_size),
    )


class AscendAWQLinearMethod(LinearMethodBase):
    def __init__(self, quant_config: Any) -> None:
        self.quant_config = quant_config

    def _get_effective_group_size(self, input_size: int) -> int:
        if self.quant_config.group_size == -1 or self.quant_config.group_size == input_size:
            return 0
        return self.quant_config.group_size

    def _get_linear_group_shape(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        input_size: int,
    ) -> tuple[int, int, Callable[[torch.nn.Parameter, torch.Tensor], None] | None]:
        is_row_parallel = isinstance(layer, RowParallelLinear)
        if self.quant_config.group_size == -1:
            group_size = input_size_per_partition if is_row_parallel else input_size
        else:
            group_size = self.quant_config.group_size

        if not is_row_parallel:
            if input_size_per_partition % group_size != 0:
                raise ValueError(
                    "The input size is not aligned with the quantized weight shape. "
                    "This can be caused by too large tensor parallel size."
                )
            return group_size, input_size_per_partition // group_size, None

        if self.quant_config.group_size == -1:
            return group_size, 1, _awq_row_group_weight_loader(0, 1)

        if input_size % group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized weight shape. "
                "This can be caused by too large tensor parallel size."
            )

        input_start = layer.tp_rank * input_size_per_partition
        input_end = input_start + input_size_per_partition
        group_start = input_start // group_size
        group_end = (input_end + group_size - 1) // group_size
        local_num_groups = group_end - group_start
        return group_size, local_num_groups, _awq_row_group_weight_loader(group_start, local_num_groups)

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
        _group_size, num_groups, group_weight_loader = self._get_linear_group_shape(
            layer,
            input_size_per_partition,
            input_size,
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
            weight_loader=group_weight_loader or weight_loader,
        )
        scales = GroupQuantScaleParameter(
            data=torch.empty(num_groups, output_size_per_partition, dtype=params_dtype),
            input_dim=0,
            output_dim=1,
            weight_loader=group_weight_loader or weight_loader,
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


class AscendAWQFusedMoEMethod(FusedMoEMethodBase):
    quant_type: QuantType = QuantType.W4A16

    def __init__(self, quant_config: Any, moe_config: FusedMoEConfig) -> None:
        super().__init__(moe_config)
        self.quant_config = quant_config
        self.dynamic_eplb = get_ascend_config().eplb_config.dynamic_eplb

    def _get_effective_group_size(self, input_size: int) -> int:
        return self.quant_config.group_size if self.quant_config.group_size != -1 else input_size

    def _validate_awq_moe_shape(self, input_size: int, output_size: int, *, name: str) -> int:
        group_size = self._get_effective_group_size(input_size)
        if input_size % group_size != 0:
            raise ValueError(
                f"{name} input size is not aligned with AWQ group size. "
                "This can be caused by too large tensor parallel size."
            )
        if output_size % self.quant_config.pack_factor != 0:
            raise ValueError(
                f"{name} output size is not aligned with AWQ pack factor. "
                "This can be caused by too large tensor parallel size."
            )
        return group_size

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        w13_output_size = 2 * intermediate_size_per_partition
        w2_output_size = hidden_size
        w13_group_size = self._validate_awq_moe_shape(
            hidden_size,
            w13_output_size,
            name="w13",
        )
        w2_group_size = self._validate_awq_moe_shape(
            intermediate_size_per_partition,
            w2_output_size,
            name="w2",
        )
        weight_loader = extra_weight_attrs.get("weight_loader")

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

        w13_num_groups = hidden_size // w13_group_size
        w2_num_groups = intermediate_size_per_partition // w2_group_size
        w13_qzeros = PackedvLLMParameter(
            data=torch.empty(
                num_experts,
                w13_num_groups,
                w13_output_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=1,
            output_dim=2,
            packed_dim=2,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )
        w2_qzeros = PackedvLLMParameter(
            data=torch.empty(
                num_experts,
                w2_num_groups,
                w2_output_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=1,
            output_dim=2,
            packed_dim=2,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )
        w13_scales = GroupQuantScaleParameter(
            data=torch.empty(num_experts, w13_num_groups, w13_output_size, dtype=params_dtype),
            input_dim=1,
            output_dim=2,
            weight_loader=weight_loader,
        )
        w2_scales = GroupQuantScaleParameter(
            data=torch.empty(num_experts, w2_num_groups, w2_output_size, dtype=params_dtype),
            input_dim=1,
            output_dim=2,
            weight_loader=weight_loader,
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
        layer.register_parameter("w2_scales", w2_scales)
        _set_awq_moe_group_attrs(w2_scales, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        for name in ("w13_qweight", "w13_qzeros", "w13_scales", "w2_qweight", "w2_qzeros", "w2_scales"):
            param = getattr(layer, name)
            setattr(layer, name, torch.nn.Parameter(param.data, requires_grad=False))

        w13_output_size = layer.w13_qweight.shape[2] * self.quant_config.pack_factor
        w2_output_size = layer.w2_qweight.shape[2] * self.quant_config.pack_factor
        w13_weight, w13_scale, w13_offset = convert_awq_moe_param_to_ascend(
            layer.w13_qweight.data,
            layer.w13_qzeros.data,
            layer.w13_scales.data,
            w13_output_size,
            self.quant_config.zero_point,
            num_bits=self.quant_config.weight_bits,
        )
        w2_weight, w2_scale, w2_offset = convert_awq_moe_param_to_ascend(
            layer.w2_qweight.data,
            layer.w2_qzeros.data,
            layer.w2_scales.data,
            w2_output_size,
            self.quant_config.zero_point,
            num_bits=self.quant_config.weight_bits,
        )

        layer.register_parameter("w13_weight", torch.nn.Parameter(w13_weight, requires_grad=False))
        layer.register_parameter("w13_weight_scale", torch.nn.Parameter(w13_scale, requires_grad=False))
        layer.register_parameter("w13_weight_offset", torch.nn.Parameter(w13_offset, requires_grad=False))
        layer.register_parameter("w2_weight", torch.nn.Parameter(w2_weight, requires_grad=False))
        layer.register_parameter("w2_weight_scale", torch.nn.Parameter(w2_scale, requires_grad=False))
        layer.register_parameter("w2_weight_offset", torch.nn.Parameter(w2_offset, requires_grad=False))

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        return None

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        num_experts: int = -1,
        global_num_experts: int | None = None,
        expert_map: torch.Tensor | None = None,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor | None = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: Any | None = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        mc2_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if global_num_experts is not None:
            num_experts = global_num_experts
        if num_experts <= 0:
            num_experts = router_logits.shape[1] + global_redundant_expert_num

        assert router_logits.shape[1] == num_experts - global_redundant_expert_num, (
            "Number of global experts mismatch (excluding redundancy)"
        )

        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            num_experts=num_experts,
        )

        return _EXTRA_CTX.moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights.to(x.dtype),
                topk_ids=topk_ids.to(torch.int32),
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                quant_type=self.quant_type,
                dynamic_eplb=self.dynamic_eplb,
                expert_map=expert_map,
                global_redundant_expert_num=global_redundant_expert_num,
                mc2_mask=mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                log2phy=log2phy,
                pertoken_scale=pertoken_scale,
                activation=activation,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                w1_offset=layer.w13_weight_offset,
                w2_offset=layer.w2_weight_offset,
            )
        )
