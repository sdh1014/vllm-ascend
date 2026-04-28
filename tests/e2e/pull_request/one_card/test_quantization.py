#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.
#
import json
from pathlib import Path

import pytest
import torch

from tests.e2e.conftest import VllmRunner
from tests.e2e.model_utils import check_outputs_equal

AWQ_MODEL = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
AWQ_PROMPTS = ["vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs."]
AWQ_MAX_TOKENS = 32
AWQ_REFERENCE_TEXT = (
    "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs. It is designed to be "
    "used in a variety of applications, including natural language processing, computer vision, and speech "
    "recognition. The engine is built on top of the"
)
AWQ_REFERENCE_GENERATED_TOKEN_IDS = [
    1084,
    374,
    6188,
    311,
    387,
    1483,
    304,
    264,
    8045,
    315,
    8357,
    11,
    2670,
    5810,
    4128,
    8692,
    11,
    6366,
    11129,
    11,
    323,
    8806,
    17843,
    13,
    576,
    4712,
    374,
    5798,
    389,
    1909,
    315,
    279,
]
AWQ_REFERENCE_SELECTED_TOKEN_LOGPROBS = [
    -0.8938695788383484,
    -1.3259625434875488,
    -1.645885705947876,
    -0.4036926329135895,
    -1.1428595781326294,
    -2.265812873840332,
    -0.9351692795753479,
    -1.5836435556411743,
    -1.226688265800476,
    -0.0003250309091527015,
    -1.5471279621124268,
    -0.16228832304477692,
    -0.4046784043312073,
    -2.2344865798950195,
    -0.006204391364008188,
    -0.08916322886943817,
    -0.3010861575603485,
    -1.3837802410125732,
    -0.03502972424030304,
    -0.013193332590162754,
    -0.2990036606788635,
    -1.5249570608139038,
    -0.12021300941705704,
    -0.35281381011009216,
    -0.26851609349250793,
    -1.148691177368164,
    -0.6790115833282471,
    -1.3627415895462036,
    -0.3628433048725128,
    -0.2482273280620575,
    -0.0002444683632347733,
    -0.2564292848110199,
]
AWQ_LINEAR_MODULE_NAMES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
AWQ_NUMERIC_LAYER_INDICES = (0, 1)


def _awq_runner():
    return VllmRunner(
        AWQ_MODEL,
        max_model_len=512,
        gpu_memory_utilization=0.7,
        enforce_eager=True,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        quantization="awq",
    )


def _generate_awq_output():
    with _awq_runner() as vllm_model:
        return vllm_model.generate_greedy(AWQ_PROMPTS, AWQ_MAX_TOKENS)


def _generate_awq_logprobs():
    with _awq_runner() as vllm_model:
        return vllm_model.generate_greedy_logprobs(
            AWQ_PROMPTS,
            max_tokens=AWQ_MAX_TOKENS,
            num_logprobs=5,
        )


def _hf_or_local_file(model: str, filename: str) -> Path:
    model_path = Path(model)
    if model_path.exists():
        return model_path / filename

    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=model, filename=filename))


def _awq_weight_map(model: str) -> dict[str, str]:
    model_path = Path(model)
    if model_path.exists():
        index_file = model_path / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file) as f:
                return json.load(f)["weight_map"]
        safetensor_files = sorted(path.name for path in model_path.glob("*.safetensors"))
    else:
        from huggingface_hub import list_repo_files
        from huggingface_hub.errors import EntryNotFoundError

        try:
            index_file = _hf_or_local_file(model, "model.safetensors.index.json")
        except EntryNotFoundError:
            safetensor_files = sorted(
                filename for filename in list_repo_files(model) if filename.endswith(".safetensors")
            )
        else:
            with open(index_file) as f:
                return json.load(f)["weight_map"]

    from safetensors import safe_open

    weight_map = {}
    for filename in safetensor_files:
        with safe_open(_hf_or_local_file(model, filename), framework="pt", device="cpu") as f:
            weight_map.update(dict.fromkeys(f.keys(), filename))
    return weight_map


def _load_awq_tensors(model: str, tensor_names: list[str]) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    weight_map = _awq_weight_map(model)
    tensors_by_file: dict[str, list[str]] = {}
    for tensor_name in tensor_names:
        tensors_by_file.setdefault(weight_map[tensor_name], []).append(tensor_name)

    tensors = {}
    for filename, names in tensors_by_file.items():
        with safe_open(_hf_or_local_file(model, filename), framework="pt", device="cpu") as f:
            for name in names:
                tensors[name] = f.get_tensor(name)
    return tensors


def _awq_layer_prefixes(model: str) -> list[str]:
    weight_map = _awq_weight_map(model)
    module_names = set(AWQ_LINEAR_MODULE_NAMES)
    layer_indices = set(AWQ_NUMERIC_LAYER_INDICES)
    prefixes_by_layer: dict[int, dict[str, str]] = {layer_idx: {} for layer_idx in AWQ_NUMERIC_LAYER_INDICES}

    for tensor_name in weight_map:
        if not tensor_name.endswith(".qweight"):
            continue

        prefix = tensor_name.removesuffix(".qweight")
        parts = prefix.split(".")
        if len(parts) < 5 or parts[:2] != ["model", "layers"]:
            continue

        layer_idx = int(parts[2])
        module_name = parts[-1]
        if layer_idx in layer_indices and module_name in module_names:
            prefixes_by_layer[layer_idx][module_name] = prefix

    return [
        prefixes_by_layer[layer_idx][module_name]
        for layer_idx in AWQ_NUMERIC_LAYER_INDICES
        for module_name in AWQ_LINEAR_MODULE_NAMES
    ]


def _dense_awq_reference(
    x: torch.Tensor, qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    from vllm_ascend.quantization.methods.awq import make_awq_zeros, unpack_awq_int32

    output_size = qweight.shape[1] * 8
    unpacked_weight = unpack_awq_int32(qweight, torch.Size([qweight.shape[0], output_size]))
    zeros = make_awq_zeros(qzeros, output_size)
    group_size = qweight.shape[0] // scales.shape[0]
    group_indices = torch.arange(qweight.shape[0]) // group_size
    dense_weight = (unpacked_weight.float() - zeros[group_indices].float()) * scales[group_indices].float()
    return torch.matmul(x.float(), dense_weight)


def _reference_awq_output():
    token_ids = list(AWQ_REFERENCE_GENERATED_TOKEN_IDS)
    text = AWQ_REFERENCE_TEXT
    return [(token_ids, text)], True


def test_qwen2_5_awq_matches_reference():
    reference_outputs, reference_is_generated_only = _reference_awq_output()
    awq_outputs = _generate_awq_output()

    if reference_is_generated_only:
        reference_token_ids, reference_text = reference_outputs[0]
        awq_token_ids, awq_text = awq_outputs[0]
        assert awq_text == reference_text
        assert awq_token_ids[-len(reference_token_ids) :] == reference_token_ids
        return

    check_outputs_equal(
        outputs_0_lst=reference_outputs,
        outputs_1_lst=awq_outputs,
        name_0="reference_outputs",
        name_1="awq_outputs",
    )


def test_qwen2_5_awq_greedy_logprobs_matches_reference():
    output_ids, output_text, output_logprobs = _generate_awq_logprobs()[0]

    assert output_ids == AWQ_REFERENCE_GENERATED_TOKEN_IDS
    assert AWQ_PROMPTS[0] + output_text == AWQ_REFERENCE_TEXT

    selected_logprobs = torch.tensor(
        [output_logprobs[token_index][token_id].logprob for token_index, token_id in enumerate(output_ids)]
    )
    reference_logprobs = torch.tensor(AWQ_REFERENCE_SELECTED_TOKEN_LOGPROBS)
    torch.testing.assert_close(selected_logprobs, reference_logprobs, rtol=1e-2, atol=2e-2)


# fmt: off
def test_qwen3_w8a8_quant():
    max_tokens = 5
    example_prompts = [
        "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs."
    ]
    vllm_target_outputs = [([
        85, 4086, 44, 374, 264, 1550, 42747, 628, 323, 4938, 72816, 44378, 323,
        13480, 4712, 369, 444, 10994, 82, 13, 1084, 374, 6188, 311, 387
    ], 'vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs. It is designed to be'
                            )]
# fmt: on

    with VllmRunner(
            "vllm-ascend/Qwen3-0.6B-W8A8",
            max_model_len=8192,
            gpu_memory_utilization=0.7,
            cudagraph_capture_sizes=[1, 2, 4, 8],
            quantization="ascend",
    ) as vllm_model:
        vllm_quant_w8a8_outputs = vllm_model.generate_greedy(
            example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=vllm_target_outputs,
        outputs_1_lst=vllm_quant_w8a8_outputs,
        name_0="vllm_target_outputs",
        name_1="vllm_quant_w8a8_outputs",
    )

# fmt: off
def test_qwen3_w8a8_quant_auto_detect():
    """Test that ModelSlim quantization is auto-detected without --quantization.

    Uses the same W8A8 model as test_qwen3_w8a8_quant but omits the
    quantization parameter, verifying that the auto-detection in
    maybe_auto_detect_quantization() picks up quant_model_description.json
    and produces identical results.
    """
    max_tokens = 5
    example_prompts = [
        "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs."
    ]
    vllm_target_outputs = [([
        85, 4086, 44, 374, 264, 1550, 42747, 628, 323, 4938, 72816, 44378, 323,
        13480, 4712, 369, 444, 10994, 82, 13, 1084, 374, 6188, 311, 387
    ], 'vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs. It is designed to be'
                            )]
# fmt: on

    with VllmRunner(
            "vllm-ascend/Qwen3-0.6B-W8A8",
            max_model_len=8192,
            gpu_memory_utilization=0.7,
            cudagraph_capture_sizes=[1, 2, 4, 8],
    ) as vllm_model:
        vllm_quant_auto_detect_outputs = vllm_model.generate_greedy(
            example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=vllm_target_outputs,
        outputs_1_lst=vllm_quant_auto_detect_outputs,
        name_0="vllm_target_outputs",
        name_1="vllm_quant_auto_detect_outputs",
    )


# fmt: off
def test_qwen3_dense_w8a16():
    max_tokens = 5
    example_prompts = [
        "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs."
    ]
    vllm_target_outputs = [([
        85, 4086, 44, 374, 264, 1550, 42747, 628, 323, 4938, 72816, 44378, 323,
        13480, 4712, 369, 444, 10994, 82, 13, 1084, 374, 6188, 311, 387
    ], 'vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs. It is designed to be'
                            )]
# fmt: on

    with VllmRunner(
            "vllm-ascend/Qwen3-0.6B-W8A16",
            max_model_len=8192,
            enforce_eager=False,
            gpu_memory_utilization=0.7,
            quantization="ascend",
    ) as vllm_model:
        vllm_quant_w8a16_outputs = vllm_model.generate_greedy(
            example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=vllm_target_outputs,
        outputs_1_lst=vllm_quant_w8a16_outputs,
        name_0="vllm_target_outputs",
        name_1="vllm_quant_w8a16_outputs",
    )


def test_qwen2_5_awq_layer_numeric_matches_dense_reference():
    torch_npu = pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU is required for AWQ numeric check.")

    from vllm_ascend.quantization.methods.awq import convert_awq_to_ascend

    model = AWQ_MODEL
    prefixes = _awq_layer_prefixes(model)
    assert {prefix.rsplit(".", 1)[-1] for prefix in prefixes} == set(AWQ_LINEAR_MODULE_NAMES)
    assert len({prefix.split(".")[2] for prefix in prefixes}) == len(AWQ_NUMERIC_LAYER_INDICES)

    tensor_names = [f"{prefix}.{name}" for prefix in prefixes for name in ("qweight", "qzeros", "scales")]
    tensors = _load_awq_tensors(model, tensor_names)

    torch.manual_seed(0)
    for prefix in prefixes:
        qweight = tensors[f"{prefix}.qweight"]
        qzeros = tensors[f"{prefix}.qzeros"]
        scales = tensors[f"{prefix}.scales"]
        group_size = qweight.shape[0] // scales.shape[0]
        x = torch.randn(3, qweight.shape[0], dtype=torch.float16) * 0.1

        reference = _dense_awq_reference(x, qweight, qzeros, scales)
        weight, scale, offset = convert_awq_to_ascend(
            qweight.npu(),
            qzeros.npu(),
            scales.npu(),
            zero_point=True,
        )
        actual = torch_npu.npu_weight_quant_batchmatmul(
            x=x.npu(),
            weight=weight,
            antiquant_scale=scale.to(x.dtype),
            antiquant_offset=offset.to(x.dtype),
            antiquant_group_size=group_size,
            bias=None,
        )
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu().float(), reference, rtol=3e-2, atol=2e-1)
