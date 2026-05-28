#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import gguf
from gguf import GGMLQuantizationType as WeightType
from huggingface_hub import hf_hub_download
from vllm import LLM, SamplingParams

from vllm_ascend.ops.triton.gguf import prepare_q6_k_metadata, q6_k_matvec


GGUF_MODEL_CASES = {
    "qwen2_5_1_5b_q6_k": (
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "qwen2.5-1.5b-instruct-q6_k.gguf",
    ),
    "qwen3_0_6b_bf16": (
        "Qwen/Qwen3-0.6B",
        "unsloth/Qwen3-0.6B-GGUF",
        "Qwen3-0.6B-BF16.gguf",
    ),
    "phi3_5_mini_iq4_xs": (
        "microsoft/Phi-3.5-mini-instruct",
        "bartowski/Phi-3.5-mini-instruct-GGUF",
        "Phi-3.5-mini-instruct-IQ4_XS.gguf",
    ),
}


@dataclass
class ModelBenchmarkResult:
    case: str
    load_seconds: float
    generate_seconds: float
    prompt_tokens: int
    output_tokens: int
    output_tokens_per_second: float


@dataclass
class MatvecBenchmarkResult:
    mode: str
    batch_size: int
    input_size: int
    output_size: int
    iterations: int
    milliseconds_per_iter: float


def _sync_device() -> None:
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()


def _pack_q6_k(values: np.ndarray) -> np.ndarray:
    values = values.reshape(-1, 256).astype(np.int8)
    blocks = np.zeros((values.shape[0], 210), dtype=np.uint8)
    blocks[:, 192:208] = np.ones((values.shape[0], 16), dtype=np.uint8)
    blocks[:, 208:210] = np.array([1.0], dtype=np.float16).view(np.uint8)

    for block_idx in range(values.shape[0]):
        for k in range(256):
            q = int(values[block_idx, k]) + 32
            ql_byte = k % 64 + (k // 128) * 64
            ql_shift = ((k // 64) % 2) * 4
            qh_byte = k % 32 + (k // 128) * 32
            qh_shift = ((k // 32) % 4) * 2
            blocks[block_idx, ql_byte] |= np.uint8((q & 0x0F) << ql_shift)
            blocks[block_idx, 128 + qh_byte] |= np.uint8(((q >> 4) & 0x03) << qh_shift)
    return blocks


def _benchmark_model(args: argparse.Namespace) -> ModelBenchmarkResult:
    tokenizer, gguf_repo, gguf_filename = GGUF_MODEL_CASES[args.case]
    model_path = hf_hub_download(gguf_repo, filename=gguf_filename)
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    prompts = [
        "Hello, my name is",
        "The capital of France is",
    ][: args.num_prompts]

    load_start = time.perf_counter()
    llm = LLM(
        model=model_path,
        tokenizer=tokenizer,
        dtype=args.dtype,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        enable_chunked_prefill=False,
        trust_remote_code=True,
    )
    _sync_device()
    load_seconds = time.perf_counter() - load_start

    generate_start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    _sync_device()
    generate_seconds = time.perf_counter() - generate_start

    prompt_tokens = sum(len(output.prompt_token_ids) for output in outputs)
    output_tokens = sum(len(item.token_ids) for output in outputs for item in output.outputs)
    return ModelBenchmarkResult(
        case=args.case,
        load_seconds=load_seconds,
        generate_seconds=generate_seconds,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        output_tokens_per_second=output_tokens / generate_seconds,
    )


def _benchmark_q6_k_matvec(args: argparse.Namespace) -> list[MatvecBenchmarkResult]:
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        raise RuntimeError("Q6_K packed matvec benchmark requires NPU.")

    rng = np.random.default_rng(args.seed)
    if args.input_size % 256 != 0:
        raise ValueError("input_size must be divisible by 256 for this benchmark.")

    q_values = rng.integers(-32, 32, size=(args.output_size, args.input_size), dtype=np.int8)
    qweight_np = _pack_q6_k(q_values)
    dense_weight = torch.from_numpy(
        gguf.dequantize(qweight_np, WeightType.Q6_K).reshape(args.output_size, args.input_size)
    )
    qweight = torch.from_numpy(qweight_np).npu()
    scales, d = prepare_q6_k_metadata(qweight)

    x = torch.randn(args.batch_size, args.input_size, dtype=torch.float16, device="npu")
    dense_weight_npu = dense_weight.to(device="npu", dtype=torch.float16)

    results = []
    for mode, fn in (
        (
            "q6_k_packed_triton",
            lambda: q6_k_matvec(
                x,
                qweight,
                scales,
                d,
                args.output_size,
                args.input_size,
                block_n=args.block_n,
                k_sub_block_size=args.k_sub_block_size,
            ),
        ),
        ("dense_torch_matmul", lambda: x.matmul(dense_weight_npu.t())),
        (
            "cpu_dequantize_then_matmul",
            lambda: x.matmul(
                torch.from_numpy(
                    gguf.dequantize(qweight_np, WeightType.Q6_K).reshape(args.output_size, args.input_size)
                )
                .to(device="npu", dtype=torch.float16)
                .t()
            ),
        ),
    ):
        for _ in range(args.warmup):
            fn()
        _sync_device()
        start = time.perf_counter()
        for _ in range(args.iterations):
            fn()
        _sync_device()
        elapsed = time.perf_counter() - start
        results.append(
            MatvecBenchmarkResult(
                mode=mode,
                batch_size=args.batch_size,
                input_size=args.input_size,
                output_size=args.output_size,
                iterations=args.iterations,
                milliseconds_per_iter=elapsed * 1000 / args.iterations,
            )
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark GGUF quantization on Ascend NPU.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    model_parser = subparsers.add_parser("model", help="Benchmark end-to-end GGUF model generation.")
    model_parser.add_argument("--case", choices=GGUF_MODEL_CASES.keys(), default="qwen2_5_1_5b_q6_k")
    model_parser.add_argument("--dtype", default="bfloat16")
    model_parser.add_argument("--max-model-len", type=int, default=1024)
    model_parser.add_argument("--max-tokens", type=int, default=32)
    model_parser.add_argument("--num-prompts", type=int, default=2)

    matvec_parser = subparsers.add_parser("q6-k-matvec", help="Benchmark synthetic Q6_K packed matvec.")
    matvec_parser.add_argument("--batch-size", type=int, default=1)
    matvec_parser.add_argument("--input-size", type=int, default=4096)
    matvec_parser.add_argument("--output-size", type=int, default=4096)
    matvec_parser.add_argument("--warmup", type=int, default=5)
    matvec_parser.add_argument("--iterations", type=int, default=20)
    matvec_parser.add_argument("--seed", type=int, default=0)
    matvec_parser.add_argument("--block-n", type=int, default=32)
    matvec_parser.add_argument("--k-sub-block-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "model":
        result = _benchmark_model(args)
        print(json.dumps(asdict(result), indent=2))
    elif args.mode == "q6-k-matvec":
        results = _benchmark_q6_k_matvec(args)
        print(json.dumps([asdict(result) for result in results], indent=2))
    else:
        raise ValueError(f"Unknown benchmark mode: {args.mode}")


if __name__ == "__main__":
    main()
