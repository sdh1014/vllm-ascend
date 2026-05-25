import torch
import torch_npu

from vllm_ascend.ops.triton.moe_weight_mask import mask_topk_weights_by_expert_map_triton


def main() -> None:
    torch.npu.set_device(0)
    topk_weights = torch.tensor(
        [
            [0.25, 0.50, 0.75],
            [1.00, 1.25, 1.50],
            [1.75, 2.00, 2.25],
        ],
        dtype=torch.bfloat16,
        device="npu",
    )
    topk_ids = torch.tensor(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 0],
        ],
        dtype=torch.int32,
        device="npu",
    )
    expert_map = torch.tensor([0, -1, 1, -1, 2, 3, -1, 4], dtype=torch.int32, device="npu")

    actual = mask_topk_weights_by_expert_map_triton(topk_weights, topk_ids, expert_map)
    expected = topk_weights * (expert_map[topk_ids] != -1)

    torch.testing.assert_close(actual.cpu(), expected.cpu())
    print("triton_moe_weight_mask_ok")
    print("masked_weights=", actual.cpu().to(torch.float32).tolist())


if __name__ == "__main__":
    main()
