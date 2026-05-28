from unittest.mock import patch

import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType as WeightType

from vllm_ascend.quantization.gguf_config import AscendGGUFConfig
from vllm_ascend.quantization.methods.gguf import (
    AscendGGUFEmbeddingMethod,
    AscendGGUFLinearMethod,
)


def _make_quantized_layer(qweight_type: WeightType | int = WeightType.Q4_0):
    layer = torch.nn.Module()
    qweight = torch.nn.Parameter(torch.zeros((2, 2), dtype=torch.uint8), requires_grad=False)
    qweight.tensor_shape = (2, 3)
    qweight.shard_id = []
    qweight.shard_id_map = {}
    qweight.data_container = []
    layer.register_parameter("qweight", qweight)

    qweight_type_param = torch.nn.Parameter(torch.empty(1, dtype=torch.uint8), requires_grad=False)
    qweight_type_param.weight_type = int(qweight_type)
    qweight_type_param.shard_weight_type = {}
    layer.register_parameter("qweight_type", qweight_type_param)
    return layer


def test_linear_process_weights_materializes_dense_weight():
    layer = _make_quantized_layer()
    method = AscendGGUFLinearMethod(AscendGGUFConfig())
    method.params_dtype = torch.float16
    dequantized = np.arange(6, dtype=np.float32).reshape(2, 3)

    with patch("vllm_ascend.quantization.methods.gguf.gguf.dequantize", return_value=dequantized):
        method.process_weights_after_loading(layer)

    assert layer.weight.shape == (2, 3)
    assert layer.weight.dtype == torch.float16
    torch.testing.assert_close(layer.weight.cpu(), torch.tensor(dequantized, dtype=torch.float16))
    assert layer.qweight is None
    assert layer.qweight_type is None


def test_embedding_process_weights_materializes_dense_weight():
    layer = _make_quantized_layer()
    method = AscendGGUFEmbeddingMethod(AscendGGUFConfig())
    method.params_dtype = torch.bfloat16
    dequantized = np.arange(6, dtype=np.float32).reshape(2, 3)

    with patch("vllm_ascend.quantization.methods.gguf.gguf.dequantize", return_value=dequantized):
        method.process_weights_after_loading(layer)

    output = method.embedding(layer, torch.tensor([0, 1]))

    assert output.shape == (2, 3)
    torch.testing.assert_close(output.cpu(), torch.tensor(dequantized, dtype=torch.bfloat16))


def test_process_weights_rejects_unsupported_quant_type():
    layer = _make_quantized_layer(255)
    method = AscendGGUFLinearMethod(AscendGGUFConfig())
    method.params_dtype = torch.float16

    with pytest.raises(ValueError, match="Unsupported GGUF quantization type"):
        method.process_weights_after_loading(layer)
