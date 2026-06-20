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
import os

import pytest
import torch

from tests.e2e.conftest import VllmRunner

AWQ_ACTIVATION_DTYPE = torch.float16
AWQ_MOE_MODEL_ENV = "VLLM_ASCEND_AWQ_MOE_MODEL"


def _require_npu():
    torch_npu = pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is required for AWQ device tests.")
    return torch_npu


def _pack_int4(values: torch.Tensor) -> torch.Tensor:
    assert values.dim() == 2
    assert values.shape[1] % 8 == 0
    packed = torch.zeros(values.shape[0], values.shape[1] // 8, dtype=torch.int32)
    for index in range(8):
        packed |= (values[:, index::8].to(torch.int32) & 0xF) << (4 * index)
    return packed


def _dense_awq_reference(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    from vllm_ascend.quantization.methods.awq import make_awq_zeros, unpack_awq_int32

    output_size = qweight.shape[1] * 8
    unpacked_weight = unpack_awq_int32(qweight, torch.Size([qweight.shape[0], output_size]))
    zeros = make_awq_zeros(qzeros, output_size)
    group_size = qweight.shape[0] // scales.shape[0]
    group_indices = torch.arange(qweight.shape[0]) // group_size
    dense_weight = (unpacked_weight.float() - zeros[group_indices].float()) * scales[group_indices].float()
    return torch.matmul(x.float(), dense_weight)


def test_dense_awq_npu_matmul_matches_reference():
    torch_npu = _require_npu()
    from vllm_ascend.quantization.methods.awq import convert_awq_to_ascend

    input_size = 32
    output_size = 16
    group_size = 8
    num_groups = input_size // group_size
    qweight_values = torch.arange(input_size * output_size, dtype=torch.int32).view(input_size, output_size) % 16
    qzero_values = (
        torch.arange(num_groups * output_size, dtype=torch.int32).view(num_groups, output_size).add_(5) % 16
    )
    qweight = _pack_int4(qweight_values)
    qzeros = _pack_int4(qzero_values)
    scales = torch.arange(1, num_groups * output_size + 1, dtype=torch.float32).view(num_groups, output_size)
    scales = (scales / 128).to(AWQ_ACTIVATION_DTYPE)
    x = torch.randn(3, input_size, dtype=AWQ_ACTIVATION_DTYPE) * 0.05

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
        antiquant_scale=scale,
        antiquant_offset=offset,
        antiquant_group_size=group_size,
        bias=None,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu().float(), reference, rtol=3e-2, atol=2e-1)


def test_awq_moe_model_short_generation_smoke():
    _require_npu()
    model = os.getenv(AWQ_MOE_MODEL_ENV)
    if not model:
        pytest.skip(f"Set {AWQ_MOE_MODEL_ENV} to run the AWQ MoE smoke test.")

    with VllmRunner(
        model,
        max_model_len=512,
        gpu_memory_utilization=0.7,
        quantization="awq",
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(["Explain AWQ in one short sentence."], max_tokens=8)

    assert len(outputs) == 1
    token_ids, text = outputs[0]
    assert 0 < len(token_ids) <= 8
    assert isinstance(text, str)
