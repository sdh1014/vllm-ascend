import torch
import torch_npu

from vllm_ascend.quantization.methods.awq import (
    convert_awq_moe_param_to_ascend,
    convert_awq_to_ascend,
)


def _rand_packed_int32(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.randint(0, 2**31 - 1, shape, dtype=torch.int32, device="npu")


def run_linear_smoke() -> None:
    k_size = 256
    n_size = 128
    group_size = 128
    qweight = _rand_packed_int32((k_size, n_size // 8))
    qzeros = _rand_packed_int32((k_size // group_size, n_size // 8))
    scales = torch.rand(k_size // group_size, n_size, dtype=torch.bfloat16, device="npu")
    weight, scale, offset = convert_awq_to_ascend(qweight, qzeros, scales, True)
    x = torch.randn(4, k_size, dtype=torch.bfloat16, device="npu")
    y = torch_npu.npu_weight_quant_batchmatmul(
        x=x,
        weight=weight,
        antiquant_scale=scale.to(x.dtype),
        antiquant_offset=offset.to(x.dtype),
        antiquant_group_size=group_size,
        bias=None,
    )
    torch.npu.synchronize()
    print("linear_ok", tuple(y.shape), y.dtype, bool(torch.isfinite(y).all().item()))


def run_moe_gmm_smoke() -> None:
    num_experts = 2
    hidden_size = 256
    intermediate_size = 256
    group_size = 128
    w13_output_size = intermediate_size * 2
    w2_output_size = hidden_size

    q13 = _rand_packed_int32((num_experts, hidden_size, w13_output_size // 8))
    z13 = _rand_packed_int32((num_experts, hidden_size // group_size, w13_output_size // 8))
    s13 = torch.rand(
        num_experts,
        hidden_size // group_size,
        w13_output_size,
        dtype=torch.bfloat16,
        device="npu",
    )
    q2 = _rand_packed_int32((num_experts, intermediate_size, w2_output_size // 8))
    z2 = _rand_packed_int32((num_experts, intermediate_size // group_size, w2_output_size // 8))
    s2 = torch.rand(
        num_experts,
        intermediate_size // group_size,
        w2_output_size,
        dtype=torch.bfloat16,
        device="npu",
    )

    w13, sc13, off13 = convert_awq_moe_param_to_ascend(q13, z13, s13, w13_output_size, True)
    w2, sc2, off2 = convert_awq_moe_param_to_ascend(q2, z2, s2, w2_output_size, True)
    x = torch.randn(4, hidden_size, dtype=torch.bfloat16, device="npu")
    group_list = torch.tensor([2, 2], dtype=torch.int64, device="npu")
    h = torch_npu.npu_grouped_matmul(
        x=[x],
        weight=[w13],
        antiquant_scale=[sc13],
        antiquant_offset=[off13],
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=group_list,
        output_dtype=torch.bfloat16,
    )[0]
    h = torch_npu.npu_swiglu(h)
    y = torch_npu.npu_grouped_matmul(
        x=[h],
        weight=[w2],
        antiquant_scale=[sc2],
        antiquant_offset=[off2],
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=group_list,
        output_dtype=torch.bfloat16,
    )[0]
    torch.npu.synchronize()
    print("moe_gmm_ok", tuple(y.shape), y.dtype, bool(torch.isfinite(y).all().item()))


def main() -> None:
    torch.npu.set_device(0)
    run_linear_smoke()
    run_moe_gmm_smoke()


if __name__ == "__main__":
    main()
