import os
import time

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0,1")

from vllm import LLM, SamplingParams


DEFAULT_MODEL = "/root/models/Qwen3-Coder-30B-A3B-Instruct-AWQ"
DEFAULT_BASE_PROMPT = (
    "Please write a Python function multiply(a, b), return the product, "
    "and explain its purpose in one Chinese sentence."
)


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _make_prompt(tokenizer, prompt_text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _count_prompt_tokens(tokenizer, prompt: str) -> int:
    return len(tokenizer(prompt, add_special_tokens=False).input_ids)


def _run_case(llm: LLM, name: str, prompt: str, prompt_tokens: int, batch_size: int, max_tokens: int) -> None:
    prompts = [prompt] * batch_size
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        min_tokens=max_tokens,
        ignore_eos=True,
        temperature=0.0,
    )

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start

    input_tokens = prompt_tokens * batch_size
    output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    total_tokens = input_tokens + output_tokens
    print(
        "RESULT,"
        f"case={name},"
        f"batch={batch_size},"
        f"prompt_tokens={prompt_tokens},"
        f"input_tokens={input_tokens},"
        f"output_tokens={output_tokens},"
        f"elapsed_s={elapsed:.4f},"
        f"input_tps={input_tokens / elapsed:.2f},"
        f"output_tps={output_tokens / elapsed:.2f},"
        f"total_tps={total_tokens / elapsed:.2f}"
    )


def main() -> None:
    model = os.environ.get("AWQ_MODEL", DEFAULT_MODEL)
    dtype = os.environ.get("AWQ_DTYPE", "bfloat16")
    tensor_parallel_size = int(os.environ.get("AWQ_TP_SIZE", "2"))
    enable_expert_parallel = os.environ.get("AWQ_ENABLE_EP", "1") == "1"
    max_tokens = int(os.environ.get("AWQ_MAX_TOKENS", "64"))
    short_batches = _parse_int_list(os.environ.get("AWQ_THROUGHPUT_SHORT_BATCHES", "1,4,8,16"))
    long_batches = _parse_int_list(os.environ.get("AWQ_THROUGHPUT_LONG_BATCHES", "1,4"))
    long_repeat = int(os.environ.get("AWQ_THROUGHPUT_LONG_REPEAT", "800"))
    max_batch_size = max(short_batches + long_batches)

    llm = LLM(
        model=model,
        tokenizer=model,
        quantization="awq",
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        enable_expert_parallel=enable_expert_parallel,
        trust_remote_code=True,
        max_model_len=int(os.environ.get("AWQ_MAX_MODEL_LEN", "1024")),
        max_num_seqs=int(os.environ.get("AWQ_MAX_NUM_SEQS", str(max_batch_size))),
        gpu_memory_utilization=float(os.environ.get("AWQ_GPU_MEMORY_UTILIZATION", "0.65")),
        enforce_eager=True,
        enable_chunked_prefill=False,
        async_scheduling=False,
    )
    tokenizer = llm.get_tokenizer()
    short_prompt_text = os.environ.get("AWQ_PROMPT", DEFAULT_BASE_PROMPT)
    long_prompt_text = (
        "Please read the following repeated context and then write a Python function add(a, b), "
        "return the sum, and explain its purpose in one Chinese sentence. Context: "
        + ("token " * long_repeat)
    )
    cases = [
        ("short", _make_prompt(tokenizer, short_prompt_text), short_batches),
        ("long", _make_prompt(tokenizer, long_prompt_text), long_batches),
    ]

    warmup_prompt = cases[0][1]
    llm.generate(
        [warmup_prompt],
        SamplingParams(max_tokens=8, min_tokens=8, ignore_eos=True, temperature=0.0),
    )

    print(
        "CONFIG,"
        f"model={model},"
        f"dtype={dtype},"
        f"tp={tensor_parallel_size},"
        f"ep={enable_expert_parallel},"
        f"max_tokens={max_tokens}"
    )
    for name, prompt, batches in cases:
        prompt_tokens = _count_prompt_tokens(tokenizer, prompt)
        for batch_size in batches:
            _run_case(llm, name, prompt, prompt_tokens, batch_size, max_tokens)


if __name__ == "__main__":
    main()
