from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

from tests.ut.base import TestBase
from vllm_ascend.ops.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod
from vllm_ascend.quantization.awq_config import AscendAWQConfig
from vllm_ascend.quantization.methods.awq import AscendAWQFusedMoEMethod, AscendAWQLinearMethod
from vllm_ascend.utils import AWQ_QUANTIZATION_METHOD


class TestAscendAWQConfig(TestBase):
    def test_get_name(self):
        self.assertEqual(AscendAWQConfig.get_name(), AWQ_QUANTIZATION_METHOD)

    def test_replaces_upstream_awq_registration(self):
        self.assertEqual(QUANTIZATION_METHODS.count(AWQ_QUANTIZATION_METHOD), 1)

    def test_get_config_filenames(self):
        self.assertEqual(AscendAWQConfig.get_config_filenames(), ["quant_config.json", "quantize_config.json"])

    def test_get_supported_act_dtypes_includes_fp16_and_bf16(self):
        self.assertEqual(AscendAWQConfig.get_supported_act_dtypes(), [torch.half, torch.bfloat16])

    def test_from_config_uses_awq_aliases(self):
        config = AscendAWQConfig.from_config(
            {
                "w_bit": 4,
                "q_group_size": 128,
                "zero_point": True,
                "modules_to_not_convert": ["lm_head"],
                "version": "GEMM",
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

    def test_apply_vllm_mapper_maps_modules_to_not_convert(self):
        config = AscendAWQConfig.from_config(
            {
                "bits": 4,
                "group_size": 128,
                "zero_point": True,
                "modules_to_not_convert": ["language_model.lm_head"],
            }
        )
        mapper = MagicMock()
        mapper.apply_list.return_value = ["lm_head"]

        config.apply_vllm_mapper(mapper)

        self.assertEqual(config.modules_to_not_convert, ["lm_head"])
        mapper.apply_list.assert_called_once_with(["language_model.lm_head"])

    def test_apply_vllm_mapper_ignores_empty_modules_to_not_convert(self):
        config = AscendAWQConfig.from_config({"bits": 4, "group_size": 128, "zero_point": True})
        mapper = MagicMock()

        config.apply_vllm_mapper(mapper)

        self.assertEqual(config.modules_to_not_convert, [])
        mapper.apply_list.assert_not_called()

    @patch("vllm_ascend.quantization.awq_config.get_safetensors_params_metadata")
    def test_maybe_update_config_infers_unquantized_modules_from_metadata(self, mock_metadata):
        mock_metadata.return_value = {
            "model.layers.0.self_attn.q_proj.qweight": {"dtype": "I32"},
            "model.layers.0.self_attn.q_proj.qzeros": {"dtype": "I32"},
            "model.layers.0.self_attn.q_proj.scales": {"dtype": "F16"},
            "model.layers.0.self_attn.k_proj.weight": {"dtype": "F16"},
            "lm_head.weight": {"dtype": "F16"},
        }
        config = AscendAWQConfig.from_config({"bits": 4, "group_size": 128, "zero_point": True})

        config.maybe_update_config("/model/path", revision="main")

        self.assertCountEqual(config.modules_to_not_convert, ["model.layers.0.self_attn.k_proj", "lm_head"])
        mock_metadata.assert_called_once_with("/model/path", revision="main")

    @patch("vllm_ascend.quantization.awq_config.get_safetensors_params_metadata")
    def test_metadata_derived_mixed_checkpoint_methods(self, mock_metadata):
        mock_metadata.return_value = {
            "model.layers.0.self_attn.q_proj.qweight": {"dtype": "I32"},
            "model.layers.0.self_attn.q_proj.qzeros": {"dtype": "I32"},
            "model.layers.0.self_attn.q_proj.scales": {"dtype": "F16"},
            "lm_head.weight": {"dtype": "F16"},
        }
        config = AscendAWQConfig.from_config({"bits": 4, "group_size": 128, "zero_point": True})
        layer = MagicMock(spec=LinearBase)

        config.maybe_update_config("/model/path")

        self.assertIsInstance(config.get_quant_method(layer, "model.layers.0.self_attn.q_proj"), AscendAWQLinearMethod)
        self.assertIsInstance(config.get_quant_method(layer, "lm_head"), UnquantizedLinearMethod)

    @patch("vllm_ascend.quantization.methods.awq.get_ascend_config")
    @patch("vllm_ascend.quantization.awq_config.get_safetensors_params_metadata")
    def test_metadata_derived_mixed_checkpoint_keeps_quantized_moe(self, mock_metadata, mock_get_ascend_config):
        mock_get_ascend_config.return_value.eplb_config.dynamic_eplb = False
        mock_metadata.return_value = {
            "model.layers.0.mlp.experts.0.gate_proj.qweight": {"dtype": "I32"},
            "model.layers.0.mlp.experts.0.gate_proj.qzeros": {"dtype": "I32"},
            "model.layers.0.mlp.experts.0.gate_proj.scales": {"dtype": "F16"},
            "lm_head.weight": {"dtype": "F16"},
        }
        config = AscendAWQConfig.from_config({"bits": 4, "group_size": 128, "zero_point": True})
        moe_layer = MagicMock(spec=FusedMoE)
        moe_layer.moe_config = MagicMock()
        linear_layer = MagicMock(spec=LinearBase)

        config.maybe_update_config("/model/path")

        self.assertIsInstance(config.get_quant_method(moe_layer, "model.layers.0.mlp.experts"), AscendAWQFusedMoEMethod)
        self.assertIsInstance(config.get_quant_method(linear_layer, "lm_head"), UnquantizedLinearMethod)

    @patch("vllm_ascend.quantization.awq_config.get_safetensors_params_metadata")
    def test_maybe_update_config_keeps_explicit_modules_to_not_convert(self, mock_metadata):
        config = AscendAWQConfig.from_config(
            {
                "bits": 4,
                "group_size": 128,
                "zero_point": True,
                "modules_to_not_convert": ["lm_head"],
            }
        )

        config.maybe_update_config("/model/path")

        self.assertEqual(config.modules_to_not_convert, ["lm_head"])
        mock_metadata.assert_not_called()

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

    @patch("vllm_ascend.quantization.methods.awq.get_ascend_config")
    @patch("vllm_ascend.quantization.awq_config.is_layer_skipped")
    def test_get_quant_method_for_fused_moe(self, mock_is_layer_skipped, mock_get_ascend_config):
        mock_get_ascend_config.return_value.eplb_config.dynamic_eplb = False
        config = AscendAWQConfig.from_config({"bits": 4, "group_size": 128, "zero_point": True})
        layer = MagicMock(spec=FusedMoE)
        layer.moe_config = MagicMock()

        mock_is_layer_skipped.return_value = False
        method = config.get_quant_method(layer, "model.layers.0.mlp.experts")

        self.assertIsInstance(method, AscendAWQFusedMoEMethod)

    @patch("vllm_ascend.ops.fused_moe.fused_moe.AscendUnquantizedFusedMoEMethod.__init__")
    @patch("vllm_ascend.quantization.awq_config.is_layer_skipped")
    def test_get_quant_method_for_skipped_fused_moe(self, mock_is_layer_skipped, mock_method):
        mock_method.return_value = None
        config = AscendAWQConfig.from_config(
            {
                "bits": 4,
                "group_size": 128,
                "zero_point": True,
                "modules_to_not_convert": ["model.layers.0.mlp.experts"],
            }
        )
        layer = MagicMock(spec=FusedMoE)
        layer.moe_config = MagicMock()

        mock_is_layer_skipped.return_value = True
        method = config.get_quant_method(layer, "model.layers.0.mlp.experts")

        self.assertIsInstance(method, AscendUnquantizedFusedMoEMethod)

    def test_unquantized_lm_head_uses_unquantized_linear_method(self):
        config = AscendAWQConfig.from_config(
            {
                "bits": 4,
                "group_size": 128,
                "zero_point": True,
                "modules_to_not_convert": ["lm_head"],
            }
        )
        layer = MagicMock(spec=LinearBase)

        method = config.get_quant_method(layer, "lm_head")

        self.assertIsInstance(method, UnquantizedLinearMethod)

    @patch("vllm_ascend.quantization.methods.awq.get_ascend_config")
    def test_qwen3_moe_router_gate_skip_does_not_skip_experts(self, mock_get_ascend_config):
        mock_get_ascend_config.return_value.eplb_config.dynamic_eplb = False
        config = AscendAWQConfig.from_config(
            {
                "bits": 4,
                "group_size": 128,
                "zero_point": True,
                "modules_to_not_convert": [".mlp.gate"],
            }
        )
        gate_layer = MagicMock(spec=LinearBase)
        moe_layer = MagicMock(spec=FusedMoE)
        moe_layer.moe_config = MagicMock()

        self.assertIsInstance(
            config.get_quant_method(gate_layer, "model.layers.0.mlp.gate"),
            UnquantizedLinearMethod,
        )
        self.assertIsInstance(
            config.get_quant_method(moe_layer, "model.layers.0.mlp.experts"),
            AscendAWQFusedMoEMethod,
        )

    def test_get_quant_method_ignores_non_linear(self):
        config = AscendAWQConfig.from_config({"bits": 4, "group_size": 128, "zero_point": True})
        self.assertIsNone(config.get_quant_method(torch.nn.Module(), "attn"))
