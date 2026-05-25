import torch
import torch_npu

from vllm_ascend.quantization.methods.awq import AWQ_REVERSE_ORDER, prepare_awq_zero_offset


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    packed = torch.zeros(values.shape[0], values.shape[1] // 8, dtype=torch.int32)
    for i in range(8):
        packed |= values[:, i::8].to(torch.int32) << (4 * i)
    return packed


def main() -> None:
    torch.npu.set_device(0)
    zero_values = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0],
            [5, 4, 3, 2, 1, 0, 7, 6, 0, 2, 4, 6, 1, 3, 5, 7],
        ],
        dtype=torch.int32,
    )
    qzeros = pack_int4(zero_values).npu()
    scales = torch.ones((2, 16), dtype=torch.bfloat16, device="npu")

    offset = prepare_awq_zero_offset(qzeros, scales, output_size=16, zero_point=True)
    expected_zeros = zero_values.view(2, 2, 8)[:, :, AWQ_REVERSE_ORDER].reshape(2, 16)
    expected = (8 - expected_zeros).to(torch.bfloat16)

    torch.testing.assert_close(offset.cpu(), expected)
    print("triton_awq_zero_offset_ok")
    print("offset=", offset.cpu().to(torch.float32).tolist())


if __name__ == "__main__":
    main()
