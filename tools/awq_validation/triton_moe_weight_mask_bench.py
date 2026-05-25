import time

import torch
import torch_npu

from vllm_ascend.ops.triton.moe_weight_mask import (
    MOE_WEIGHT_MASK_TRITON_MIN_ELEMENTS,
    mask_topk_weights_by_expert_map_triton,
)


def _bench(fn, warmup: int = 20, iters: int = 200) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1000 / iters


def _make_expert_map(global_experts: int, local_experts: int) -> torch.Tensor:
    values = [-1] * global_experts
    for expert_id in range(local_experts):
        values[expert_id] = expert_id
    return torch.tensor(values, dtype=torch.int32, device="npu")


def main() -> None:
    torch.npu.set_device(0)
    top_k = 8
    global_experts = 128
    local_experts = 64
    expert_map = _make_expert_map(global_experts, local_experts)

    print(f"default_min_elements={MOE_WEIGHT_MASK_TRITON_MIN_ELEMENTS}")
    print("tokens,torch_ms,triton_ms,speedup")
    for tokens in [128, 512, 2048, 8192, 24576]:
        topk_weights = torch.rand((tokens, top_k), dtype=torch.bfloat16, device="npu")
        topk_ids = torch.randint(
            0,
            global_experts,
            (tokens, top_k),
            dtype=torch.int32,
            device="npu",
        )

        expected = topk_weights * (expert_map[topk_ids] != -1)
        actual = mask_topk_weights_by_expert_map_triton(topk_weights, topk_ids, expert_map)
        torch.testing.assert_close(actual.cpu(), expected.cpu())

        torch_ms = _bench(lambda: topk_weights * (expert_map[topk_ids] != -1))
        triton_ms = _bench(lambda: mask_topk_weights_by_expert_map_triton(topk_weights, topk_ids, expert_map))
        speedup = torch_ms / triton_ms
        print(f"{tokens},{torch_ms:.4f},{triton_ms:.4f},{speedup:.2f}x")


if __name__ == "__main__":
    main()
