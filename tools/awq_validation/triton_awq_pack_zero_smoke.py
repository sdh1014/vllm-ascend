import torch
import torch_npu

from vllm_ascend.ops.triton.awq_pack_zero import awq_pack_zero_triton
from vllm_ascend.quantization.methods.awq import AWQ_REVERSE_ORDER, unpack_awq_int32


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    packed = torch.zeros(values.shape[0], values.shape[1] // 8, dtype=torch.int32)
    for i in range(8):
        packed |= values[:, i::8].to(torch.int32) << (4 * i)
    return packed


def main() -> None:
    torch.npu.set_device(0)
    qweight = torch.randint(-(2**31), 2**31 - 1, (128, 64), dtype=torch.int32, device="npu")
    zero_values = torch.arange(4 * 512, dtype=torch.int32).reshape(4, 512) % 16
    qzeros = pack_int4(zero_values).npu()
    scales = torch.ones((4, 512), dtype=torch.bfloat16, device="npu")

    packed_weight, offset = awq_pack_zero_triton(qweight, qzeros, scales, output_size=512)
    unpacked = unpack_awq_int32(qweight, torch.Size([qweight.shape[0], qweight.shape[1] * 8]), packed_dim=1)
    expected_weight = torch_npu.npu_convert_weight_to_int4pack(unpacked.sub(8), inner_k_tiles=0)
    expected_zeros = zero_values.view(4, 64, 8)[:, :, AWQ_REVERSE_ORDER].reshape(4, 512)
    expected_offset = (8 - expected_zeros).to(torch.bfloat16)
    torch.npu.synchronize()

    torch.testing.assert_close(packed_weight.cpu(), expected_weight.cpu())
    torch.testing.assert_close(offset.cpu(), expected_offset)
    print("triton_awq_pack_zero_ok")
    print("packed_weight_shape=", tuple(packed_weight.shape))
    print("offset_shape=", tuple(offset.shape))


if __name__ == "__main__":
    main()
