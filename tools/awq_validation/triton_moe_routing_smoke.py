import torch
import torch_npu

from vllm_ascend.ops.triton.moe_routing import npu_moe_init_routing_active_expert_triton


def main() -> None:
    torch.npu.set_device(0)
    hidden = torch.arange(16, dtype=torch.bfloat16, device="npu").reshape(4, 4)
    topk_ids = torch.tensor(
        [[0, 2], [1, 3], [2, 0], [3, 1]],
        dtype=torch.int32,
        device="npu",
    )

    sorted_hidden, expanded_row_idx, expert_tokens = npu_moe_init_routing_active_expert_triton(
        hidden,
        topk_ids,
        first_expert=0,
        last_expert=2,
    )

    expected_hidden = hidden[torch.tensor([0, 2, 1, 3], device="npu")]
    expected_row_idx = torch.tensor(
        [0, 0, 2, 0, 0, 1, 0, 3],
        dtype=torch.int32,
        device="npu",
    )
    expected_tokens = torch.tensor([2, 2], dtype=torch.int32, device="npu")

    torch.testing.assert_close(sorted_hidden.cpu(), expected_hidden.cpu())
    torch.testing.assert_close(expanded_row_idx.cpu(), expected_row_idx.cpu())
    torch.testing.assert_close(expert_tokens.cpu(), expected_tokens.cpu())
    print("triton_active_expert_routing_ok")
    print("sorted_hidden=", sorted_hidden.cpu().to(torch.float32).tolist())
    print("expanded_row_idx=", expanded_row_idx.cpu().tolist())
    print("expert_tokens=", expert_tokens.cpu().tolist())


if __name__ == "__main__":
    main()
