import importlib.util

import gguf
import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType as WeightType

from vllm_ascend.ops.triton.gguf import prepare_q6_k_metadata, q6_k_matvec


def _npu_available() -> bool:
    return importlib.util.find_spec("torch_npu") is not None and torch.npu.is_available()


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


@pytest.mark.skipif(not _npu_available(), reason="requires NPU")
def test_q6_k_matvec_matches_gguf_dequantize():
    rng = np.random.default_rng(0)
    input_size = 256
    output_size = 4
    batch_size = 3
    q_values = rng.integers(-32, 32, size=(output_size, input_size), dtype=np.int8)
    x_cpu = torch.randn(batch_size, input_size, dtype=torch.float16)

    qweight_np = _pack_q6_k(q_values)
    dense_weight_np = gguf.dequantize(qweight_np, WeightType.Q6_K).reshape(output_size, input_size)

    x = x_cpu.npu()
    qweight = torch.from_numpy(qweight_np).npu()
    scales, d = prepare_q6_k_metadata(qweight)

    output = q6_k_matvec(x, qweight, scales, d, output_size, input_size)
    expected = x_cpu.float().matmul(torch.from_numpy(dense_weight_np).float().t()).to(torch.float16)

    torch.testing.assert_close(output.cpu(), expected, atol=2e-2, rtol=2e-2)
