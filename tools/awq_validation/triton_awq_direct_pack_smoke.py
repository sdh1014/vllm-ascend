import torch
import torch_npu

from vllm_ascend.ops.triton.awq_direct_pack import awq_direct_pack_candidate
from vllm_ascend.quantization.methods.awq import unpack_awq_int32


def main() -> None:
    torch.npu.set_device(0)
    for shape in [(16, 4), (128, 64), (1024, 256)]:
        qweight = torch.randint(-(2**31), 2**31 - 1, shape, dtype=torch.int32, device="npu")
        unpacked = unpack_awq_int32(qweight, torch.Size([shape[0], shape[1] * 8]), packed_dim=1)
        signed = unpacked.sub(8)
        expected = torch_npu.npu_convert_weight_to_int4pack(signed, inner_k_tiles=0)
        packed = awq_direct_pack_candidate(qweight)
        torch.npu.synchronize()
        torch.testing.assert_close(packed.cpu(), expected.cpu())
        print(f"shape={shape} ok")
    print("triton_awq_direct_pack_ok")


if __name__ == "__main__":
    main()
