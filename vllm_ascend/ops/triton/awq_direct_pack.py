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

"""AWQ direct-pack Triton-Ascend candidate."""

import torch
from vllm.triton_utils import tl, triton  # type: ignore[import-not-found]

DEFAULT_BLOCK_SIZE = 1024


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _direct_pack_grid(n_elements: int, block_size: int) -> tuple[int]:
    return (_ceil_div(n_elements, block_size),)


def triton_kernel_launchable(kernel: object | None = None) -> bool:
    if kernel is None:
        kernel = _awq_direct_pack_candidate_kernel
    return callable(getattr(kernel, "__getitem__", None))


@triton.jit
def _awq_direct_pack_candidate_kernel(
    qweight,
    output,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    packed = tl.load(qweight + offsets, mask=mask, other=0)

    nibble_0 = ((packed >> 0) & 15) + 8
    nibble_1 = ((packed >> 16) & 15) + 8
    nibble_2 = ((packed >> 4) & 15) + 8
    nibble_3 = ((packed >> 20) & 15) + 8
    nibble_4 = ((packed >> 8) & 15) + 8
    nibble_5 = ((packed >> 24) & 15) + 8
    nibble_6 = ((packed >> 12) & 15) + 8
    nibble_7 = ((packed >> 28) & 15) + 8

    repacked = (
        ((nibble_0 & 15) << 0)
        | ((nibble_1 & 15) << 4)
        | ((nibble_2 & 15) << 8)
        | ((nibble_3 & 15) << 12)
        | ((nibble_4 & 15) << 16)
        | ((nibble_5 & 15) << 20)
        | ((nibble_6 & 15) << 24)
        | ((nibble_7 & 15) << 28)
    )

    tl.store(output + offsets, repacked, mask=mask)


def awq_direct_pack_candidate(
    qweight: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    assert qweight.dtype == torch.int32
    if not triton_kernel_launchable():
        raise RuntimeError("Triton-Ascend direct-pack kernel is not launchable.")
    qweight = qweight.contiguous()
    output = torch.empty_like(qweight)
    grid = _direct_pack_grid(qweight.numel(), block_size)
    _awq_direct_pack_candidate_kernel[grid](
        qweight,
        output,
        qweight.numel(),
        BLOCK_SIZE=block_size,
    )
    return output
