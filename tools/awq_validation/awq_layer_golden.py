import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")

import torch
import torch_npu

from vllm_ascend.quantization.awq_config import AscendAWQConfig
from vllm_ascend.quantization.methods.awq import (
    AWQ_REVERSE_ORDER,
    AscendAWQLinearMethod,
    convert_awq_moe_param_to_ascend,
)


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    packed = torch.zeros(values.shape[:-1] + (values.shape[-1] // 8,), dtype=torch.int32)
    for i in range(8):
        packed |= values[..., i::8].to(torch.int32) << (4 * i)
    return packed


def make_quant_values(shape: tuple[int, ...], *, offset: int = 0) -> torch.Tensor:
    values = (torch.arange(int(torch.tensor(shape).prod()), dtype=torch.int32).reshape(shape) + offset) % 16
    return values


def make_zero_values(shape: tuple[int, ...], zero_point: bool, *, offset: int = 0) -> torch.Tensor:
    if not zero_point:
        return torch.zeros(shape, dtype=torch.int32)
    return make_quant_values(shape, offset=offset)


def dequant_awq_weight(
    q_values: torch.Tensor,
    zero_values: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int,
    zero_point: bool,
) -> torch.Tensor:
    q = q_values.reshape(*q_values.shape[:-1], q_values.shape[-1] // 8, 8)[..., AWQ_REVERSE_ORDER].reshape(
        q_values.shape
    )
    if zero_point:
        zeros = zero_values.reshape(*zero_values.shape[:-1], zero_values.shape[-1] // 8, 8)[
            ..., AWQ_REVERSE_ORDER
        ].reshape(zero_values.shape)
        base = q.to(torch.float32) - zeros.to(torch.float32)
    else:
        base = q.to(torch.float32) - 8

    effective_group_size = q.shape[-2] if group_size == -1 else group_size
    group_ids = torch.arange(q.shape[-2]) // effective_group_size
    scale = scales.index_select(-2, group_ids.to(scales.device)).to(torch.float32)
    return base * scale


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype) -> None:
    actual_cpu = actual.detach().cpu().to(torch.float32)
    expected_cpu = expected.detach().cpu().to(torch.float32)
    atol = 0.08 if dtype is torch.float16 else 0.16
    rtol = 0.08 if dtype is torch.float16 else 0.12
    torch.testing.assert_close(actual_cpu, expected_cpu, atol=atol, rtol=rtol)
    max_abs = (actual_cpu - expected_cpu).abs().max().item()
    print(f"PASS,{name},dtype={dtype},max_abs={max_abs:.6f},atol={atol},rtol={rtol}")


def run_linear_case(dtype: torch.dtype, group_size: int, zero_point: bool) -> None:
    input_size = 128 if group_size == 128 else 64
    output_size = 32
    num_groups = 1 if group_size == -1 else input_size // group_size

    q_values = make_quant_values((input_size, output_size), offset=1)
    zero_values = make_zero_values((num_groups, output_size), zero_point, offset=3)
    scales = (torch.arange(num_groups * output_size, dtype=torch.float32).reshape(num_groups, output_size) % 7 + 1)
    scales = scales * 0.003
    x = torch.randn(5, input_size, dtype=torch.float32) * 0.1

    config = AscendAWQConfig.from_config({"bits": 4, "group_size": group_size, "zero_point": zero_point})
    method = AscendAWQLinearMethod(config)
    layer = torch.nn.Module()
    layer.qweight = torch.nn.Parameter(pack_int4(q_values).npu(), requires_grad=False)
    layer.qzeros = torch.nn.Parameter(pack_int4(zero_values).npu(), requires_grad=False)
    layer.scales = torch.nn.Parameter(scales.to(dtype=dtype, device="npu"), requires_grad=False)
    method.process_weights_after_loading(layer)

    actual = method.apply(layer, x.to(dtype=dtype, device="npu"))
    dequant = dequant_awq_weight(q_values, zero_values, scales, group_size=group_size, zero_point=zero_point)
    expected = x @ dequant
    assert_close(f"linear,group_size={group_size},zero_point={zero_point}", actual, expected, dtype)


def run_moe_case(dtype: torch.dtype, group_size: int, zero_point: bool) -> None:
    num_experts = 2
    hidden_size = 128 if group_size == 128 else 64
    intermediate_size = 128 if group_size == 128 else 32
    w13_output_size = intermediate_size * 2
    w2_output_size = hidden_size
    w13_num_groups = 1 if group_size == -1 else hidden_size // group_size
    w2_num_groups = 1 if group_size == -1 else intermediate_size // group_size

    w13_values = make_quant_values((num_experts, hidden_size, w13_output_size), offset=2)
    w13_zero_values = make_zero_values((num_experts, w13_num_groups, w13_output_size), zero_point, offset=4)
    w13_scales = (
        torch.arange(num_experts * w13_num_groups * w13_output_size, dtype=torch.float32).reshape(
            num_experts, w13_num_groups, w13_output_size
        )
        % 5
        + 1
    ) * 0.002

    w2_values = make_quant_values((num_experts, intermediate_size, w2_output_size), offset=6)
    w2_zero_values = make_zero_values((num_experts, w2_num_groups, w2_output_size), zero_point, offset=8)
    w2_scales = (
        torch.arange(num_experts * w2_num_groups * w2_output_size, dtype=torch.float32).reshape(
            num_experts, w2_num_groups, w2_output_size
        )
        % 5
        + 1
    ) * 0.002

    w13_weight, w13_scale, w13_offset = convert_awq_moe_param_to_ascend(
        torch.stack([pack_int4(item) for item in w13_values]).npu(),
        torch.stack([pack_int4(item) for item in w13_zero_values]).npu(),
        w13_scales.to(dtype=dtype, device="npu"),
        w13_output_size,
        zero_point=zero_point,
    )
    w2_weight, w2_scale, w2_offset = convert_awq_moe_param_to_ascend(
        torch.stack([pack_int4(item) for item in w2_values]).npu(),
        torch.stack([pack_int4(item) for item in w2_zero_values]).npu(),
        w2_scales.to(dtype=dtype, device="npu"),
        w2_output_size,
        zero_point=zero_point,
    )

    expert0_tokens = torch.randn(3, hidden_size, dtype=torch.float32) * 0.1
    expert1_tokens = torch.randn(2, hidden_size, dtype=torch.float32) * 0.1
    hidden_states = torch.cat([expert0_tokens, expert1_tokens], dim=0).to(dtype=dtype, device="npu")
    group_list = torch.tensor([3, 2], dtype=torch.int64, device="npu")

    actual_gmm1 = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w13_weight],
        antiquant_scale=[w13_scale],
        antiquant_offset=[w13_offset],
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=group_list,
        output_dtype=dtype,
    )[0]
    actual_act = torch_npu.npu_swiglu(actual_gmm1)
    actual = torch_npu.npu_grouped_matmul(
        x=[actual_act],
        weight=[w2_weight],
        antiquant_scale=[w2_scale],
        antiquant_offset=[w2_offset],
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=group_list,
        output_dtype=dtype,
    )[0]

    w13_dequant = dequant_awq_weight(
        w13_values,
        w13_zero_values,
        w13_scales,
        group_size=group_size,
        zero_point=zero_point,
    )
    w2_dequant = dequant_awq_weight(
        w2_values,
        w2_zero_values,
        w2_scales,
        group_size=group_size,
        zero_point=zero_point,
    )
    ref_gmm1 = torch.cat(
        [
            expert0_tokens @ w13_dequant[0],
            expert1_tokens @ w13_dequant[1],
        ],
        dim=0,
    )
    ref_act = torch_npu.npu_swiglu(ref_gmm1.to(dtype=dtype, device="npu")).cpu().to(torch.float32)
    expected = torch.cat(
        [
            ref_act[:3] @ w2_dequant[0],
            ref_act[3:] @ w2_dequant[1],
        ],
        dim=0,
    )
    assert_close(f"moe,group_size={group_size},zero_point={zero_point}", actual, expected, dtype)


def main() -> None:
    torch.manual_seed(0)
    torch.npu.set_device(0)
    cases = [
        (torch.float16, 128, True),
        (torch.float16, 128, False),
        (torch.float16, -1, True),
        (torch.float16, -1, False),
        (torch.bfloat16, 128, True),
        (torch.bfloat16, 128, False),
        (torch.bfloat16, -1, True),
        (torch.bfloat16, -1, False),
    ]
    for dtype, group_size, zero_point in cases:
        run_linear_case(dtype, group_size, zero_point)
        run_moe_case(dtype, group_size, zero_point)
    print("awq_layer_golden_ok")


if __name__ == "__main__":
    main()
