from unittest.mock import MagicMock, patch

import pytest
import torch

pytest.importorskip("vllm")
pytest.importorskip("torch_npu")

from vllm.model_executor.layers.fused_moe import RoutedExperts, UnquantizedFusedMoEMethod
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from vllm_ascend.quantization.awq_config import AscendAWQConfig
from vllm_ascend.quantization.methods.awq import (
    AscendAWQFusedMoEMethod,
    AscendAWQLinearMethod,
    apply_awq_group_plan,
    build_awq_group_plan,
    convert_awq_to_ascend_reference,
    make_awq_zeros,
    pack_awq_weight_to_ascend_reference,
)

AUTOAWQ_PACK_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


def _pack_int4(values: torch.Tensor) -> torch.Tensor:
    assert values.dim() == 2
    assert values.shape[1] % 8 == 0
    packed = torch.zeros(values.shape[0], values.shape[1] // 8, dtype=torch.int32)
    for logical_index, checkpoint_index in enumerate(AUTOAWQ_PACK_ORDER):
        packed |= (values[:, logical_index::8].to(torch.int32) & 0xF) << (4 * checkpoint_index)
    return packed


def _make_awq_tensors(
    input_size: int,
    output_size: int,
    num_groups: int,
    *,
    scale_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qweight_values = torch.arange(input_size * output_size, dtype=torch.int32).view(input_size, output_size) % 16
    qzero_values = (
        torch.arange(num_groups * output_size, dtype=torch.int32).view(num_groups, output_size).add_(3) % 16
    )
    scales = torch.arange(1, num_groups * output_size + 1, dtype=torch.float32).view(num_groups, output_size)
    return _pack_int4(qweight_values), _pack_int4(qzero_values), scales.to(scale_dtype)


def _mock_int4pack(weight: torch.Tensor, **_kwargs) -> torch.Tensor:
    return torch.zeros(weight.shape[0], weight.shape[1] // 8, dtype=torch.int32, device=weight.device)


def _require_npu_triton() -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is required for AWQ Triton pack tests.")
    try:
        from vllm.triton_utils import HAS_TRITON
    except (ImportError, ModuleNotFoundError):
        pytest.skip("vLLM Triton utilities are required for AWQ Triton pack tests.")
    if not HAS_TRITON:
        pytest.skip("Triton is not available.")


@pytest.mark.parametrize(
    ("zero_point", "input_size", "output_size", "num_groups"),
    [
        (True, 16, 16, 4),
        (True, 33, 24, 3),
        (False, 16, 16, 4),
        (False, 33, 24, 3),
    ],
)
def test_awq_pack_matches_reference(
    zero_point: bool,
    input_size: int,
    output_size: int,
    num_groups: int,
):
    _require_npu_triton()
    from vllm_ascend.ops.triton.awq_direct_pack import awq_direct_pack
    from vllm_ascend.ops.triton.awq_pack_zero import awq_pack_zero_triton

    qweight, qzeros, scales = _make_awq_tensors(input_size, output_size, num_groups)
    qweight = qweight.npu()
    qzeros = qzeros.npu()
    scales = scales.npu()

    if zero_point:
        packed_weight, offset = awq_pack_zero_triton(qweight, qzeros, scales, output_size)
        reference_weight, _reference_scale, reference_offset = convert_awq_to_ascend_reference(
            qweight,
            qzeros,
            scales,
            zero_point=True,
        )
        torch.npu.synchronize()
        torch.testing.assert_close(packed_weight.cpu(), reference_weight.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(offset.cpu(), reference_offset.cpu(), rtol=0, atol=0)
        return

    packed_weight = awq_direct_pack(qweight)
    reference_weight = pack_awq_weight_to_ascend_reference(qweight, output_size)
    torch.npu.synchronize()
    torch.testing.assert_close(packed_weight.cpu(), reference_weight.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize(
    ("global_k", "tp_size", "checkpoint_group_size", "expected_runtime_group_size"),
    [
        (1024, 2, 128, 128),
        (896, 2, 128, 64),
        (896, 2, -1, 0),
    ],
)
def test_awq_group_plan(
    global_k: int,
    tp_size: int,
    checkpoint_group_size: int,
    expected_runtime_group_size: int,
):
    output_size = 16
    local_k = global_k // tp_size
    checkpoint_groups = 1 if checkpoint_group_size == -1 else global_k // checkpoint_group_size
    qzero_values = torch.arange(checkpoint_groups, dtype=torch.int32).view(checkpoint_groups, 1)
    qzero_values = qzero_values.expand(checkpoint_groups, output_size).contiguous()
    qzeros = _pack_int4(qzero_values % 16)
    scales = torch.arange(checkpoint_groups * output_size, dtype=torch.float16).view(checkpoint_groups, output_size)

    for tp_rank in range(tp_size):
        shard_start = tp_rank * local_k
        plan = build_awq_group_plan(
            checkpoint_group_size,
            global_k,
            local_k,
            shard_start,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )

        local_qzeros = apply_awq_group_plan(qzeros, plan, group_dim=0)
        local_scales = apply_awq_group_plan(scales, plan, group_dim=0)
        runtime_group_size = plan.runtime_group_size
        group_stride = local_k if runtime_group_size == 0 else runtime_group_size
        local_group_offsets = torch.arange(0, local_k, group_stride)
        if checkpoint_group_size == -1:
            expected_group_ids = torch.zeros(1, dtype=torch.long)
        else:
            expected_group_ids = (local_group_offsets + shard_start) // checkpoint_group_size

        assert runtime_group_size == expected_runtime_group_size
        torch.testing.assert_close(local_scales, scales[expected_group_ids])
        torch.testing.assert_close(make_awq_zeros(local_qzeros, output_size), qzero_values[expected_group_ids] % 16)

        captured: dict[str, torch.Tensor] = {}

        def capture_loader(
            _param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            _shard_id: str,
            _expert_id: int,
            return_success: bool = False,
        ) -> bool | None:
            captured[weight_name] = loaded_weight
            return True if return_success else None

        moe_config = MagicMock()
        moe_config.tp_size = tp_size
        moe_config.tp_rank = tp_rank
        layer = torch.nn.Module()
        layer.moe_config = moe_config
        config = AscendAWQConfig.from_config(
            {"bits": 4, "group_size": checkpoint_group_size, "zero_point": True}
        )
        method = AscendAWQFusedMoEMethod(config, moe_config)
        method.create_weights(
            layer,
            num_experts=1,
            hidden_size=global_k,
            intermediate_size_per_partition=local_k,
            params_dtype=torch.float16,
            intermediate_size_full=global_k,
            weight_loader=capture_loader,
        )

        assert layer.awq_runtime_group_size == expected_runtime_group_size
        assert layer.w13_scales.shape[1] == (
            1 if expected_runtime_group_size == 0 else global_k // expected_runtime_group_size
        )
        assert layer.w2_scales.shape[1] == (
            1 if expected_runtime_group_size == 0 else local_k // expected_runtime_group_size
        )

        moe_w13_plan = build_awq_group_plan(
            checkpoint_group_size,
            global_k,
            global_k,
            0,
            tp_size=tp_size,
            tp_rank=tp_rank,
            runtime_group_size=expected_runtime_group_size,
        )
        moe_w2_plan = build_awq_group_plan(
            checkpoint_group_size,
            global_k,
            local_k,
            shard_start,
            tp_size=tp_size,
            tp_rank=tp_rank,
            runtime_group_size=expected_runtime_group_size,
        )
        layer.w13_scales.weight_loader(
            layer.w13_scales,
            scales,
            "w13_scales",
            "w1",
            0,
            return_success=True,
        )
        layer.w2_scales.weight_loader(
            layer.w2_scales,
            scales,
            "w2_scales",
            "w2",
            0,
            return_success=True,
        )
        torch.testing.assert_close(
            captured["w13_scales"],
            apply_awq_group_plan(scales, moe_w13_plan, group_dim=0),
        )
        torch.testing.assert_close(
            captured["w2_scales"],
            apply_awq_group_plan(scales, moe_w2_plan, group_dim=0),
        )


def _new_linear_layer() -> LinearBase:
    layer = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    return layer


def _new_routed_experts_layer() -> RoutedExperts:
    layer = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.moe_config = MagicMock()
    return layer


def _new_lm_head_layer() -> ParallelLMHead:
    layer = ParallelLMHead.__new__(ParallelLMHead)
    torch.nn.Module.__init__(layer)
    return layer


def test_awq_config_dispatch():
    config = AscendAWQConfig.from_config(
        {
            "bits": 4,
            "group_size": 128,
            "zero_point": True,
            "modules_to_not_convert": ["skip_linear", "skip_experts"],
        }
    )

    quantized_linear = config.get_quant_method(_new_linear_layer(), "model.layers.0.self_attn.q_proj")
    skipped_linear = config.get_quant_method(_new_linear_layer(), "model.skip_linear")
    quantized_moe = config.get_quant_method(_new_routed_experts_layer(), "model.layers.0.mlp.experts")
    config.lm_head_quantized = True
    quantized_lm_head = config.get_quant_method(_new_lm_head_layer(), "lm_head")
    config.lm_head_quantized = False
    skipped_lm_head = config.get_quant_method(_new_lm_head_layer(), "lm_head")
    with patch.object(UnquantizedFusedMoEMethod, "__init__", return_value=None):
        skipped_moe = config.get_quant_method(_new_routed_experts_layer(), "model.skip_experts")

    assert isinstance(quantized_linear, AscendAWQLinearMethod)
    assert isinstance(quantized_moe, AscendAWQFusedMoEMethod)
    assert isinstance(quantized_lm_head, AscendAWQLinearMethod)
    assert isinstance(skipped_linear, UnquantizedLinearMethod)
    assert not isinstance(skipped_lm_head, AscendAWQLinearMethod)
    assert isinstance(skipped_moe, UnquantizedFusedMoEMethod)


def _make_dense_post_load_case() -> tuple[
    AscendAWQLinearMethod, torch.nn.Module, dict[str, torch.Size], tuple[str, ...]
]:
    config = AscendAWQConfig.from_config({"bits": 4, "group_size": 4, "zero_point": True})
    method = AscendAWQLinearMethod(config)
    layer = torch.nn.Module()
    qweight, qzeros, scales = _make_awq_tensors(4, 16, 1, scale_dtype=torch.float32)
    layer.qweight = torch.nn.Parameter(qweight, requires_grad=False)
    layer.qzeros = torch.nn.Parameter(qzeros, requires_grad=False)
    layer.scales = torch.nn.Parameter(scales, requires_grad=False)
    return (
        method,
        layer,
        {
            "weight": torch.Size([4, 2]),
            "weight_scale": torch.Size([1, 16]),
            "weight_offset": torch.Size([1, 16]),
        },
        ("qweight", "qzeros", "scales"),
    )


def _make_moe_post_load_case() -> tuple[
    AscendAWQFusedMoEMethod, torch.nn.Module, dict[str, torch.Size], tuple[str, ...]
]:
    config = AscendAWQConfig.from_config({"bits": 4, "group_size": 1, "zero_point": True})
    method = AscendAWQFusedMoEMethod(config, MagicMock())
    layer = torch.nn.Module()
    qweight, qzeros, scales = _make_awq_tensors(1, 16, 1, scale_dtype=torch.float32)
    for prefix in ("w13", "w2"):
        setattr(layer, f"{prefix}_qweight", torch.nn.Parameter(qweight.unsqueeze(0), requires_grad=False))
        setattr(layer, f"{prefix}_qzeros", torch.nn.Parameter(qzeros.unsqueeze(0), requires_grad=False))
        setattr(layer, f"{prefix}_scales", torch.nn.Parameter(scales.unsqueeze(0), requires_grad=False))

    return (
        method,
        layer,
        {
            "w13_weight_packed": torch.Size([1, 1, 2]),
            "w2_weight_packed": torch.Size([1, 1, 2]),
            "w13_weight_scale": torch.Size([1, 1, 16]),
            "w2_weight_scale": torch.Size([1, 1, 16]),
            "w13_weight_offset": torch.Size([1, 1, 16]),
            "w2_weight_offset": torch.Size([1, 1, 16]),
        },
        ("w13_qweight", "w13_qzeros", "w13_scales", "w2_qweight", "w2_qzeros", "w2_scales"),
    )


@pytest.mark.parametrize("case_factory", [_make_dense_post_load_case, _make_moe_post_load_case])
def test_awq_post_load_parameters(case_factory):
    method, layer, expected_shapes, checkpoint_names = case_factory()
    with patch("vllm_ascend.quantization.methods.awq.torch_npu.npu_convert_weight_to_int4pack") as mock_pack:
        mock_pack.side_effect = _mock_int4pack
        method.process_weights_after_loading(layer)

    for name, shape in expected_shapes.items():
        param = getattr(layer, name)
        assert isinstance(param, torch.nn.Parameter)
        assert param.shape == shape

    for name in checkpoint_names:
        assert name not in layer._parameters
        assert not hasattr(layer, name)
