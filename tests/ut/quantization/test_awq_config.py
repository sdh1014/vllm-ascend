from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

from tests.ut.base import TestBase
from vllm_ascend.quantization.awq_config import AscendAWQConfig
from vllm_ascend.quantization.methods.awq import AscendAWQLinearMethod
from vllm_ascend.utils import AWQ_QUANTIZATION_METHOD


class TestAscendAWQConfig(TestBase):
    def test_get_name(self):
        self.assertEqual(AscendAWQConfig.get_name(), AWQ_QUANTIZATION_METHOD)

    def test_replaces_upstream_awq_registration(self):
        self.assertEqual(QUANTIZATION_METHODS.count(AWQ_QUANTIZATION_METHOD), 1)

    def test_get_config_filenames(self):
        self.assertEqual(AscendAWQConfig.get_config_filenames(), ["quant_config.json", "quantize_config.json"])

    def test_from_config_uses_awq_aliases(self):
        config = AscendAWQConfig.from_config(
            {
                "w_bit": 4,
                "q_group_size": 128,
                "zero_point": True,
                "modules_to_not_convert": ["lm_head"],
            }
        )

        self.assertEqual(config.weight_bits, 4)
        self.assertEqual(config.group_size, 128)
        self.assertTrue(config.zero_point)
        self.assertEqual(config.modules_to_not_convert, ["lm_head"])
        self.assertEqual(config.pack_factor, 8)

        config = AscendAWQConfig.from_config({"bits": 4, "group_size": -1, "zero_point": False})
        self.assertEqual(config.group_size, -1)
        self.assertFalse(config.zero_point)

    def test_weight_bits_must_be_int4(self):
        with self.assertRaises(ValueError):
            AscendAWQConfig.from_config({"bits": 8, "group_size": 128, "zero_point": True})

    @patch("vllm_ascend.quantization.awq_config.is_layer_skipped")
    def test_get_quant_method_for_linear(self, mock_is_layer_skipped):
        config = AscendAWQConfig.from_config({"bits": 4, "group_size": 128, "zero_point": True})
        layer = MagicMock(spec=LinearBase)

        mock_is_layer_skipped.return_value = False
        method = config.get_quant_method(layer, "model.layers.0.self_attn.q_proj")
        self.assertIsInstance(method, AscendAWQLinearMethod)

        mock_is_layer_skipped.return_value = True
        method = config.get_quant_method(layer, "lm_head")
        self.assertIsInstance(method, UnquantizedLinearMethod)

    def test_get_quant_method_ignores_non_linear(self):
        config = AscendAWQConfig.from_config({"bits": 4, "group_size": 128, "zero_point": True})
        self.assertIsNone(config.get_quant_method(torch.nn.Module(), "attn"))
