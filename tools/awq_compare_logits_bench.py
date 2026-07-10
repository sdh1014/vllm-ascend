#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Collect AWQ logprobs and throughput for one runtime.

The script has two deliberately separate checks:

1. Logprob collection via vLLM offline inference.
   vLLM does not expose raw full logits from the public LLM API. This script
   records returned logprobs, which are logits after log-softmax and optional
   top-k selection. Use --full-vocab-logprobs to request full-vocab logprobs.

2. Throughput collection via `vllm bench throughput`.
   The benchmark uses the built-in random dataset by default, so it does not
   require a ShareGPT dataset file.

Example:

    python tools/awq_compare_logits_bench.py \
        --awq-model /data/models/Qwen2-1.5B-Instruct-AWQ \
        --tensor-parallel-size 1 \
        --dtype float16 \
        --output-dir runs/awq_dense_tp1
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_PROMPTS = [
    "The capital of France is",
    "Explain tensor parallelism in one short paragraph.",
    "2 + 2 =",
]


def _json_or_scalar(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "none" or lowered == "null":
            return None
        return value


def _parse_key_values(items: list[str] | None) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected key=value, got {item!r}")
        key, value = item.split("=", 1)
        parsed[key.replace("-", "_")] = _json_or_scalar(value)
    return parsed


def _parse_optional_json(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("JSON config must decode to an object.")
    return parsed


def _quantization_arg(value: str | None) -> str | None:
    if value is None:
        return None
    if value.lower() in ("", "none", "auto"):
        return None
    return value


def _read_prompts(args: argparse.Namespace) -> list[str]:
    prompts = list(args.prompt or [])
    if args.prompts_file:
        path = Path(args.prompts_file)
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            if path.suffix == ".jsonl":
                item = json.loads(line)
                if isinstance(item, str):
                    prompts.append(item)
                else:
                    prompts.append(str(item["prompt"]))
            else:
                prompts.append(line)
    return prompts or DEFAULT_PROMPTS


def _extract_logprob_map(logprob_map: Any) -> dict[str, float] | None:
    if not logprob_map:
        return None
    result: dict[str, float] = {}
    for token_id, logprob in dict(logprob_map).items():
        result[str(token_id)] = float(getattr(logprob, "logprob", logprob))
    return result


def _records_from_outputs(outputs: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for request_idx, output in enumerate(outputs):
        prompt_positions = []
        for position, item in enumerate(output.prompt_logprobs or []):
            prompt_positions.append(
                {
                    "position": position,
                    "logprobs": _extract_logprob_map(item),
                }
            )

        sample_positions = []
        first_completion = output.outputs[0] if output.outputs else None
        if first_completion is not None:
            for position, item in enumerate(first_completion.logprobs or []):
                sample_positions.append(
                    {
                        "position": position,
                        "logprobs": _extract_logprob_map(item),
                    }
                )

        records.append(
            {
                "request_index": request_idx,
                "prompt": output.prompt,
                "generated_text": first_completion.text
                if first_completion is not None
                else "",
                "prompt_logprobs": prompt_positions,
                "sample_logprobs": sample_positions,
            }
        )
    return records


def _run_logprob_worker(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--quantization")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--logprobs", type=int, default=10)
    parser.add_argument("--prompt-logprobs", type=int, default=10)
    parser.add_argument("--full-vocab-logprobs", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompts-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--additional-config")
    parser.add_argument("--engine-arg", action="append", default=[])
    worker_args = parser.parse_args(argv)

    from vllm import LLM, SamplingParams

    prompts = json.loads(Path(worker_args.prompts_json).read_text(encoding="utf-8"))
    engine_kwargs: dict[str, Any] = {
        "model": worker_args.model,
        "tensor_parallel_size": worker_args.tensor_parallel_size,
        "dtype": worker_args.dtype,
        "gpu_memory_utilization": worker_args.gpu_memory_utilization,
        "trust_remote_code": worker_args.trust_remote_code,
        "enforce_eager": worker_args.enforce_eager,
        "seed": worker_args.seed,
    }
    if worker_args.tokenizer:
        engine_kwargs["tokenizer"] = worker_args.tokenizer
    if worker_args.quantization:
        engine_kwargs["quantization"] = worker_args.quantization
    if worker_args.max_model_len:
        engine_kwargs["max_model_len"] = worker_args.max_model_len
    if worker_args.full_vocab_logprobs:
        engine_kwargs["max_logprobs"] = -1
    additional_config = _parse_optional_json(worker_args.additional_config)
    if additional_config is not None:
        engine_kwargs["additional_config"] = additional_config
    engine_kwargs.update(_parse_key_values(worker_args.engine_arg))

    llm = LLM(**engine_kwargs)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=worker_args.max_tokens,
        logprobs=worker_args.logprobs,
        prompt_logprobs=worker_args.prompt_logprobs,
        seed=worker_args.seed,
    )
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    Path(worker_args.output_json).write_text(
        json.dumps(_records_from_outputs(outputs), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _run_subprocess(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _run_logprob_case(
    label: str,
    model: str,
    quantization: str | None,
    args: argparse.Namespace,
    prompts_json: Path,
    output_dir: Path,
    env: dict[str, str],
) -> Path:
    output_json = output_dir / f"{label}_logprobs.json"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_logprobs_worker",
        "--model",
        model,
        "--dtype",
        args.dtype,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-tokens",
        str(args.sample_output_len),
        "--logprobs",
        str(args.logprobs),
        "--prompt-logprobs",
        str(args.prompt_logprobs),
        "--seed",
        str(args.seed),
        "--prompts-json",
        str(prompts_json),
        "--output-json",
        str(output_json),
    ]
    if args.tokenizer:
        cmd.extend(["--tokenizer", args.tokenizer])
    if quantization:
        cmd.extend(["--quantization", quantization])
    if args.max_model_len:
        cmd.extend(["--max-model-len", str(args.max_model_len)])
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if args.full_vocab_logprobs:
        cmd.append("--full-vocab-logprobs")
    if args.additional_config:
        cmd.extend(["--additional-config", args.additional_config])
    for item in args.engine_arg:
        cmd.extend(["--engine-arg", item])

    _run_subprocess(cmd, Path.cwd(), env)
    return output_json


def _run_bench_case(
    label: str,
    model: str,
    quantization: str | None,
    args: argparse.Namespace,
    output_dir: Path,
    env: dict[str, str],
) -> Path:
    output_json = output_dir / f"{label}_throughput.json"
    vllm_cli = str(Path(sys.executable).with_name("vllm"))
    cmd = [
        vllm_cli,
        "bench",
        "throughput",
        "--model",
        model,
        "--dtype",
        args.dtype,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--dataset-name",
        "random",
        "--random-input-len",
        str(args.bench_input_len),
        "--random-output-len",
        str(args.bench_output_len),
        "--num-prompts",
        str(args.bench_num_prompts),
        "--backend",
        "vllm",
        "--output-json",
        str(output_json),
        "--seed",
        str(args.seed),
    ]
    if args.tokenizer:
        cmd.extend(["--tokenizer", args.tokenizer])
    if quantization:
        cmd.extend(["--quantization", quantization])
    if args.max_model_len:
        cmd.extend(["--max-model-len", str(args.max_model_len)])
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if args.additional_config:
        cmd.extend(["--additional-config", args.additional_config])
    for item in args.bench_arg:
        cmd.extend(shlex.split(item))

    _run_subprocess(cmd, Path.cwd(), env)
    return output_json


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect AWQ model logprobs plus optional throughput."
    )
    parser.add_argument("--awq-model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--awq-quantization", default="awq")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--devices", help="Value for ASCEND_RT_VISIBLE_DEVICES.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--additional-config")
    parser.add_argument("--engine-arg", action="append", default=[])
    parser.add_argument("--bench-arg", action="append", default=[])
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--prompts-file")
    parser.add_argument("--sample-output-len", type=int, default=1)
    parser.add_argument("--logprobs", type=int, default=10)
    parser.add_argument("--prompt-logprobs", type=int, default=10)
    parser.add_argument(
        "--full-vocab-logprobs",
        action="store_true",
        help="Request logprobs=-1/prompt_logprobs=-1 and max_logprobs=-1.",
    )
    parser.add_argument("--bench-input-len", type=int, default=128)
    parser.add_argument("--bench-output-len", type=int, default=128)
    parser.add_argument("--bench-num-prompts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-logprobs", action="store_true")
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument("--output-dir", default="awq_collect_results")
    return parser


def main(argv: list[str]) -> int:
    args = _make_parser().parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.full_vocab_logprobs:
        args.logprobs = -1
        args.prompt_logprobs = -1

    env = os.environ.copy()
    if args.devices:
        env["ASCEND_RT_VISIBLE_DEVICES"] = args.devices

    prompts = _read_prompts(args)
    prompts_json = output_dir / "prompts.json"
    prompts_json.write_text(json.dumps(prompts, indent=2), encoding="utf-8")

    result: dict[str, Any] = {
        "config": {
            "awq_model": args.awq_model,
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.dtype,
            "devices": args.devices,
            "num_prompts_for_logprobs": len(prompts),
            "bench_input_len": args.bench_input_len,
            "bench_output_len": args.bench_output_len,
            "bench_num_prompts": args.bench_num_prompts,
            "full_vocab_logprobs": args.full_vocab_logprobs,
        },
        "artifacts": {"prompts": str(prompts_json)},
    }

    awq_quantization = _quantization_arg(args.awq_quantization)

    if not args.skip_logprobs:
        awq_logprobs = _run_logprob_case(
            "awq",
            args.awq_model,
            awq_quantization,
            args,
            prompts_json,
            output_dir,
            env,
        )
        result["artifacts"]["awq_logprobs"] = str(awq_logprobs)

    if not args.skip_bench:
        awq_bench = _run_bench_case(
            "awq",
            args.awq_model,
            awq_quantization,
            args,
            output_dir,
            env,
        )
        result["artifacts"]["awq_bench"] = str(awq_bench)
        result["bench"] = _load_json(awq_bench)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_logprobs_worker":
        _run_logprob_worker(sys.argv[2:])
        raise SystemExit(0)
    raise SystemExit(main(sys.argv[1:]))
