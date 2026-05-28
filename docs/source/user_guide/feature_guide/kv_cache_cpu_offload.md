# KV Cache CPU Offload Guide

## Overview

KV Cache CPU Offload enables offloading inactive KV cache blocks from NPU memory to CPU memory, allowing vLLM to handle longer contexts or more concurrent requests when NPU memory is limited. When a prefix cache miss occurs on the NPU but the data exists in CPU memory, the KV cache is asynchronously loaded back to the NPU, reducing recomputation latency.

This feature is built on vLLM's `OffloadingConnector` framework and provides an Ascend NPU-specific implementation (`NPUOffloadingSpec`) that uses dedicated NPU streams for efficient asynchronous data transfers between NPU and CPU.

## Key Concepts

- **CPU Block Pool**: A pre-allocated pool of CPU memory blocks (optionally pinned) used to store offloaded KV cache data.
- **Asynchronous Transfer**: NPU-to-CPU (D2H) and CPU-to-NPU (H2D) transfers are performed on separate NPU streams, overlapping with computation to minimize latency impact.
- **LRU Eviction**: The CPU-side block pool uses an LRU (Least Recently Used) eviction policy to manage limited CPU memory efficiently.

## Usage

### Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    gpu_memory_utilization=0.5,
    kv_offloading_backend="native",
    kv_offloading_size=8,
)

sampling_params = SamplingParams(max_tokens=100, temperature=0.0)
outputs = llm.generate(["Hello, my name is"], sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")
```

### Online Serving

```bash
vllm serve Qwen/Qwen3-0.6B \
    --gpu-memory-utilization 0.5 \
    --kv-offloading-backend native \
    --kv-offloading-size 8
```

## Configuration Parameters

- `kv_offloading_backend`: Set to `"native"`.
- `kv_offloading_size`: CPU memory size in GiB for native KV cache offload.

## How It Works

1. **Normal inference**: KV cache blocks are computed and stored on the NPU as usual.
2. **Eviction to CPU**: When NPU memory is full and new blocks are needed, inactive KV cache blocks are asynchronously copied to CPU memory via a dedicated D2H NPU stream.
3. **Prefix cache hit (CPU)**: When a request shares a prefix with previously computed data, and the prefix cache is not found on NPU but exists in CPU memory, the KV cache blocks are asynchronously loaded back from CPU to NPU via a dedicated H2D NPU stream.
4. **LRU management**: The CPU block pool uses LRU eviction to discard the least recently used blocks when CPU memory is full.

## Optional: KV Cache Events

You can enable KV cache event publishing for monitoring or debugging purposes:

```python
from vllm.config import KVEventsConfig

kv_events_config = KVEventsConfig(
    enable_kv_cache_events=True,
    publisher="zmq",
    endpoint="tcp://*:5555",
    topic="kv_events",
)

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    gpu_memory_utilization=0.5,
    kv_offloading_backend="native",
    kv_offloading_size=8,
    kv_events_config=kv_events_config,
)
```

## Notes

- This feature requires vLLM v1 engine.
- Adjust `kv_offloading_size` based on available CPU memory. Using too much CPU memory may cause out-of-memory errors on the host.
- Pinned (page-locked) memory is used when available for optimal transfer performance.
- The `gpu_memory_utilization` parameter controls how much NPU memory is reserved for KV cache. Lower values leave less NPU memory for KV cache, making offloading more active.
- For production workloads, benchmark with realistic request patterns to find the optimal `kv_offloading_size` setting.
