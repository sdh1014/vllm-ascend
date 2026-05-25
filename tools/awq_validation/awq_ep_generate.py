import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0,1")

from vllm import LLM, SamplingParams


DEFAULT_MODEL = "/root/models/Qwen3-Coder-30B-A3B-Instruct-AWQ"
DEFAULT_PROMPT = (
    "Please write a Python function multiply(a, b), return the product, "
    "and explain its purpose in one Chinese sentence."
)


def main() -> None:
    model = os.environ.get("AWQ_MODEL", DEFAULT_MODEL)
    prompt_text = os.environ.get("AWQ_PROMPT", DEFAULT_PROMPT)
    dtype = os.environ.get("AWQ_DTYPE", "bfloat16")
    tensor_parallel_size = int(os.environ.get("AWQ_TP_SIZE", "2"))
    enable_expert_parallel = os.environ.get("AWQ_ENABLE_EP", "1") == "1"

    llm = LLM(
        model=model,
        tokenizer=model,
        quantization="awq",
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        enable_expert_parallel=enable_expert_parallel,
        trust_remote_code=True,
        max_model_len=int(os.environ.get("AWQ_MAX_MODEL_LEN", "512")),
        gpu_memory_utilization=float(os.environ.get("AWQ_GPU_MEMORY_UTILIZATION", "0.65")),
        enforce_eager=True,
        enable_chunked_prefill=False,
        async_scheduling=False,
    )
    prompt = llm.get_tokenizer().apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    outputs = llm.generate(
        [prompt],
        SamplingParams(
            max_tokens=int(os.environ.get("AWQ_MAX_TOKENS", "96")),
            temperature=float(os.environ.get("AWQ_TEMPERATURE", "0.0")),
        ),
    )
    print("PROMPT:", prompt_text)
    print("TEXT:", outputs[0].outputs[0].text.strip())


if __name__ == "__main__":
    main()
