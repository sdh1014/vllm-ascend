from unittest.mock import MagicMock

import pytest
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)

from vllm_ascend.quantization.gguf_config import AscendGGUFConfig
from vllm_ascend.quantization.methods.gguf import (
    AscendGGUFEmbeddingMethod,
    AscendGGUFLinearMethod,
)


class DummyLinear(LinearBase):
    pass


class DummyEmbedding(VocabParallelEmbedding):
    pass


class DummyRoutedExperts(RoutedExperts):
    pass


def test_get_quant_method_returns_ascend_linear_method():
    config = AscendGGUFConfig()
    layer = object.__new__(DummyLinear)

    method = config.get_quant_method(layer, "model.layers.0.mlp.down_proj")

    assert isinstance(method, AscendGGUFLinearMethod)


def test_config_exposes_empty_quant_description_for_ascend_norm():
    config = AscendGGUFConfig()

    assert config.quant_description == {}


def test_get_quant_method_skips_unquantized_linear():
    config = AscendGGUFConfig(unquantized_modules=["model.layers.0.mlp.down_proj"])
    layer = object.__new__(DummyLinear)

    method = config.get_quant_method(layer, "model.layers.0.mlp.down_proj")

    assert isinstance(method, UnquantizedLinearMethod)


def test_get_quant_method_returns_ascend_embedding_method():
    config = AscendGGUFConfig()
    layer = object.__new__(DummyEmbedding)

    method = config.get_quant_method(layer, "model.embed_tokens")

    assert isinstance(method, AscendGGUFEmbeddingMethod)


def test_get_quant_method_skips_unquantized_embedding():
    config = AscendGGUFConfig(unquantized_modules=["model.embed_tokens"])
    layer = object.__new__(DummyEmbedding)

    method = config.get_quant_method(layer, "model.embed_tokens")

    assert isinstance(method, UnquantizedEmbeddingMethod)


def test_get_quant_method_rejects_moe_for_first_phase():
    config = AscendGGUFConfig()
    layer = object.__new__(DummyRoutedExperts)

    with pytest.raises(NotImplementedError, match="MoE TP phase"):
        config.get_quant_method(layer, "model.layers.0.mlp.experts")


def test_get_quant_method_ignores_unknown_layer():
    config = AscendGGUFConfig()

    assert config.get_quant_method(MagicMock(), "model.norm") is None
