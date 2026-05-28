#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import torch
from vllm.triton_utils import tl, triton

GGUF_QK_K = 256
GGUF_Q6_K_TYPE_SIZE = 210
GGUF_Q6_K_QL_SIZE = 128
GGUF_Q6_K_QH_OFFSET = 128
GGUF_Q6_K_SCALES_OFFSET = 192
GGUF_Q6_K_D_OFFSET = 208
GGUF_Q6_K_SCALE_GROUP_SIZE = 16


@triton.jit
def _q6_k_matvec_kernel(
    x_ptr,
    qweight_ptr,
    scales_ptr,
    d_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    blocks_per_row: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qe: tl.constexpr,
    stride_outm: tl.constexpr,
    stride_outn: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_SUB_BLOCKS: tl.constexpr,
):
    m = tl.program_id(0)
    k_program = tl.program_id(2)
    k_block = k_program // K_SUB_BLOCKS
    sub_block = k_program - k_block * K_SUB_BLOCKS
    n_offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    k_lanes = tl.arange(0, BLOCK_K)
    local_k_offsets = sub_block * BLOCK_K + k_lanes

    block_ids = n_offsets[:, None] * blocks_per_row + k_block
    k_offsets = k_block * 256 + local_k_offsets
    mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)

    ql_byte_offsets = local_k_offsets % 64 + (local_k_offsets // 128) * 64
    ql_shift = ((local_k_offsets // 64) % 2) * 4
    qh_byte_offsets = local_k_offsets % 32 + (local_k_offsets // 128) * 32
    qh_shift = ((local_k_offsets // 32) % 4) * 2
    scale_offsets = local_k_offsets // 16

    ql = tl.load(
        qweight_ptr + block_ids * stride_qb + ql_byte_offsets[None, :] * stride_qe,
        mask=mask,
        other=0,
    )
    qh = tl.load(
        qweight_ptr
        + block_ids * stride_qb
        + (128 + qh_byte_offsets[None, :]) * stride_qe,
        mask=mask,
        other=0,
    )
    scales = tl.load(
        scales_ptr + block_ids * 16 + scale_offsets[None, :],
        mask=mask,
        other=0,
    ).to(tl.float32)
    d = tl.load(d_ptr + block_ids, mask=n_offsets[:, None] < N, other=0.0).to(tl.float32)

    ql_values = (ql >> ql_shift) & 0x0F
    qh_values = (qh >> qh_shift) & 0x03
    q_values = (ql_values | (qh_values << 4)).to(tl.int32) - 32
    weight_values = q_values.to(tl.float32) * scales * d

    x_values = tl.load(
        x_ptr + m * stride_xm + k_offsets * stride_xk,
        mask=k_offsets < K,
        other=0.0,
    ).to(tl.float32)
    acc = tl.sum(weight_values * x_values[None, :], axis=1)

    tl.atomic_add(
        output_ptr + m * stride_outm + n_offsets * stride_outn,
        acc,
        mask=n_offsets < N,
    )


def prepare_q6_k_metadata(qweight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract small per-block metadata needed by the Q6_K packed matvec prototype."""
    if qweight.dtype != torch.uint8:
        raise TypeError(f"Q6_K qweight must be torch.uint8, got {qweight.dtype}.")
    if qweight.dim() != 2 or qweight.shape[1] < GGUF_Q6_K_TYPE_SIZE:
        raise ValueError(
            f"Q6_K qweight must have shape [num_blocks, >= {GGUF_Q6_K_TYPE_SIZE}], got {tuple(qweight.shape)}."
        )

    scales = qweight[:, GGUF_Q6_K_SCALES_OFFSET:GGUF_Q6_K_D_OFFSET].contiguous().view(torch.int8)
    d = qweight[:, GGUF_Q6_K_D_OFFSET:GGUF_Q6_K_TYPE_SIZE].contiguous().view(torch.float16).reshape(-1)
    return scales, d.to(torch.float32)


def q6_k_matvec(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    d: torch.Tensor,
    output_size: int,
    input_size: int,
    bias: torch.Tensor | None = None,
    block_n: int = 32,
    k_sub_block_size: int = GGUF_QK_K,
) -> torch.Tensor:
    """Compute ``x @ dequantize(qweight).T`` for GGUF Q6_K packed weights.

    This is a prototype decode-path kernel. It supports 2D or flattened
    higher-rank inputs and keeps the packed GGUF weight on NPU.
    """
    if qweight.dtype != torch.uint8:
        raise TypeError(f"Q6_K qweight must be torch.uint8, got {qweight.dtype}.")
    if x.shape[-1] != input_size:
        raise ValueError(f"Input last dimension {x.shape[-1]} does not match input_size {input_size}.")
    if input_size <= 0 or output_size <= 0:
        raise ValueError("input_size and output_size must be positive.")
    if block_n <= 0:
        raise ValueError("block_n must be positive.")
    if GGUF_QK_K % k_sub_block_size != 0:
        raise ValueError(f"k_sub_block_size must divide {GGUF_QK_K}.")
    max_block_n = 32 if k_sub_block_size == GGUF_QK_K else 64
    if block_n > max_block_n:
        raise ValueError(f"block_n must be <= {max_block_n} to fit the Ascend Triton UB limit.")

    blocks_per_row = triton.cdiv(input_size, GGUF_QK_K)
    k_sub_blocks = GGUF_QK_K // k_sub_block_size
    expected_blocks = output_size * blocks_per_row
    if qweight.shape[0] < expected_blocks:
        raise ValueError(f"Q6_K qweight has {qweight.shape[0]} blocks, expected at least {expected_blocks}.")

    original_shape = x.shape[:-1]
    x_2d = x.reshape(-1, input_size).contiguous()
    output = torch.zeros((x_2d.shape[0], output_size), dtype=torch.float32, device=x.device)

    output_blocks = triton.cdiv(output_size, block_n)
    total_programs = x_2d.shape[0] * output_blocks * blocks_per_row * k_sub_blocks
    if total_programs > 65535:
        raise ValueError(
            f"Q6_K matvec grid has {total_programs} programs, exceeding the Ascend limit 65535. "
            "Use a larger block_n or a smaller batch/output size."
        )

    grid = (x_2d.shape[0], output_blocks, blocks_per_row * k_sub_blocks)
    _q6_k_matvec_kernel[grid](
        x_2d,
        qweight,
        scales,
        d,
        output,
        x_2d.shape[0],
        output_size,
        input_size,
        blocks_per_row,
        x_2d.stride(0),
        x_2d.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_N=block_n,
        BLOCK_K=k_sub_block_size,
        K_SUB_BLOCKS=k_sub_blocks,
    )

    if bias is not None:
        output = output + bias
    return output.to(x.dtype).reshape(*original_shape, output_size)


__all__ = ["prepare_q6_k_metadata", "q6_k_matvec"]
