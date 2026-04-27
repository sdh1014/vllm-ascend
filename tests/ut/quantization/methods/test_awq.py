from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.quantization.awq_config import AscendAWQConfig
from vllm_ascend.quantization.methods.awq import (
    AWQ_REVERSE_ORDER,
    AscendAWQLinearMethod,
    convert_awq_to_ascend,
    make_awq_zeros,
    unpack_awq_int32,
)


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    packed = torch.zeros(values.shape[0], values.shape[1] // 8, dtype=torch.int32)
    for i in range(8):
        packed |= values[:, i::8].to(torch.int32) << (4 * i)
    return packed


class TestAWQConversionHelpers(TestBase):
    def test_unpack_awq_int32_applies_upstream_reverse_order(self):
        values = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32)
        packed = pack_int4(values)

        unpacked = unpack_awq_int32(packed, torch.Size([1, 8]))

        self.assertEqual(unpacked.dtype, torch.int32)
        torch.testing.assert_close(unpacked, values[:, AWQ_REVERSE_ORDER])

    def test_make_awq_zeros_uses_unpacked_checkpoint_values(self):
        zeros = torch.tensor([[0, 7, 14, 1, 2, 3, 4, 5]], dtype=torch.int32)
        qzeros = pack_int4(zeros)

        unpacked = make_awq_zeros(qzeros, output_size=8)

        torch.testing.assert_close(unpacked, zeros[:, AWQ_REVERSE_ORDER])

    @patch("vllm_ascend.quantization.methods.awq.torch_npu.npu_convert_weight_to_int4pack")
    def test_convert_awq_to_ascend_with_zero_point(self, mock_pack):
        qweight_values = torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15]], dtype=torch.int32)
        zero_values = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32)
        scales = torch.arange(1, 9, dtype=torch.float32).view(1, 8)
        mock_pack.side_effect = lambda weight: weight

        packed_weight, scale, offset = convert_awq_to_ascend(
            pack_int4(qweight_values),
            pack_int4(zero_values),
            scales,
            zero_point=True,
        )

        expected_weight = qweight_values[:, AWQ_REVERSE_ORDER]
        expected_zeros = zero_values[:, AWQ_REVERSE_ORDER]
        torch.testing.assert_close(packed_weight, expected_weight - 8)
        torch.testing.assert_close(scale, scales)
        self.assertEqual(offset.shape, scales.shape)
        self.assertEqual(offset.dtype, scales.dtype)
        torch.testing.assert_close(offset, (8 - expected_zeros).to(scales.dtype))

    @patch("vllm_ascend.quantization.methods.awq.torch_npu.npu_convert_weight_to_int4pack")
    def test_convert_awq_to_ascend_preserves_group_major_scale_layout(self, mock_pack):
        qweight_values = torch.tensor(
            [
                [8, 9, 10, 11, 12, 13, 14, 15],
                [7, 6, 5, 4, 3, 2, 1, 0],
            ],
            dtype=torch.int32,
        )
        zero_values = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [7, 6, 5, 4, 3, 2, 1, 0],
            ],
            dtype=torch.int32,
        )
        scales = torch.arange(1, 17, dtype=torch.float32).view(2, 8)
        mock_pack.side_effect = lambda weight: weight

        packed_weight, scale, offset = convert_awq_to_ascend(
            pack_int4(qweight_values),
            pack_int4(zero_values),
            scales,
            zero_point=True,
        )

        expected_weight = qweight_values[:, AWQ_REVERSE_ORDER]
        expected_zeros = zero_values[:, AWQ_REVERSE_ORDER]
        torch.testing.assert_close(packed_weight, expected_weight - 8)
        torch.testing.assert_close(scale, scales)
        torch.testing.assert_close(offset, (8 - expected_zeros).to(scales.dtype))

    @patch("vllm_ascend.quantization.methods.awq.torch_npu.npu_convert_weight_to_int4pack")
    def test_convert_awq_to_ascend_without_zero_point(self, mock_pack):
        qweight_values = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32)
        zero_values = torch.zeros(1, 8, dtype=torch.int32)
        scales = torch.ones(1, 8, dtype=torch.float32)
        mock_pack.side_effect = lambda weight: weight

        packed_weight, scale, offset = convert_awq_to_ascend(
            pack_int4(qweight_values),
            pack_int4(zero_values),
            scales,
            zero_point=False,
        )

        torch.testing.assert_close(
            packed_weight,
            qweight_values[:, AWQ_REVERSE_ORDER] - 8,
        )
        torch.testing.assert_close(scale, scales)
        torch.testing.assert_close(offset, torch.zeros_like(scales))


class TestAscendAWQLinearMethod(TestBase):
    @patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=1)
    @patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=0)
    def test_create_weights_uses_upstream_checkpoint_names_and_shapes(self, *_):
        config = AscendAWQConfig.from_config({"bits": 4, "group_size": 4, "zero_point": True})
        method = AscendAWQLinearMethod(config)
        layer = torch.nn.Module()
        weight_loader = MagicMock()

        method.create_weights(
            layer,
            input_size_per_partition=8,
            output_partition_sizes=[16],
            input_size=8,
            output_size=16,
            params_dtype=torch.float16,
            weight_loader=weight_loader,
        )

        self.assertEqual(layer.qweight.shape, torch.Size([8, 2]))
        self.assertEqual(layer.qweight.dtype, torch.int32)
        self.assertEqual(layer.qzeros.shape, torch.Size([2, 2]))
        self.assertEqual(layer.qzeros.dtype, torch.int32)
        self.assertEqual(layer.scales.shape, torch.Size([2, 16]))
        self.assertEqual(layer.scales.dtype, torch.float16)
        self.assertEqual(layer.qweight.packed_dim, 1)
        self.assertEqual(layer.qweight.packed_factor, 8)

    @patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=1)
    @patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=0)
    def test_create_weights_group_size_minus_one(self, *_):
        config = AscendAWQConfig.from_config({"bits": 4, "group_size": -1, "zero_point": True})
        method = AscendAWQLinearMethod(config)
        layer = torch.nn.Module()

        method.create_weights(
            layer,
            input_size_per_partition=8,
            output_partition_sizes=[16],
            input_size=8,
            output_size=16,
            params_dtype=torch.float16,
        )

        self.assertEqual(layer.qzeros.shape, torch.Size([1, 2]))
        self.assertEqual(layer.scales.shape, torch.Size([1, 16]))

    @patch("vllm_ascend.quantization.methods.awq.torch_npu.npu_convert_weight_to_int4pack")
    def test_process_weights_after_loading_registers_ascend_params(self, mock_pack):
        config = AscendAWQConfig.from_config({"bits": 4, "group_size": 4, "zero_point": True})
        method = AscendAWQLinearMethod(config)
        layer = torch.nn.Module()
        qweight_values = torch.tensor(
            [
                [8, 9, 10, 11, 12, 13, 14, 15],
                [7, 6, 5, 4, 3, 2, 1, 0],
                [1, 3, 5, 7, 9, 11, 13, 15],
                [0, 2, 4, 6, 8, 10, 12, 14],
            ],
            dtype=torch.int32,
        )
        zero_values = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32)
        layer.qweight = torch.nn.Parameter(pack_int4(qweight_values), requires_grad=False)
        layer.qzeros = torch.nn.Parameter(pack_int4(zero_values), requires_grad=False)
        layer.scales = torch.nn.Parameter(torch.ones(1, 8, dtype=torch.float32), requires_grad=False)
        mock_pack.side_effect = lambda weight: weight

        method.process_weights_after_loading(layer)

        self.assertEqual(layer.weight.shape, torch.Size([4, 8]))
        self.assertEqual(layer.weight_scale.shape, torch.Size([1, 8]))
        self.assertEqual(layer.weight_offset.shape, torch.Size([1, 8]))
        torch.testing.assert_close(layer.weight, qweight_values[:, AWQ_REVERSE_ORDER] - 8)
        torch.testing.assert_close(
            layer.weight_offset,
            8 - zero_values[:, AWQ_REVERSE_ORDER].to(torch.float32),
        )

    @patch("vllm_ascend.quantization.methods.awq.torch_npu.npu_weight_quant_batchmatmul")
    def test_apply_uses_npu_weight_quant_matmul(self, mock_matmul):
        config = AscendAWQConfig.from_config({"bits": 4, "group_size": 4, "zero_point": True})
        method = AscendAWQLinearMethod(config)
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(torch.zeros(4, 1, dtype=torch.int32), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(torch.ones(1, 8, dtype=torch.float32), requires_grad=False)
        layer.weight_offset = torch.nn.Parameter(torch.zeros(1, 8, dtype=torch.float32), requires_grad=False)
        bias = torch.ones(8, dtype=torch.float16)
        x = torch.ones(2, 4, dtype=torch.float16)
        mock_matmul.return_value = torch.zeros(2, 8, dtype=torch.float16)

        method.apply(layer, x, bias=bias)

        mock_matmul.assert_called_once()
        call_kwargs = mock_matmul.call_args.kwargs
        self.assertIs(call_kwargs["x"], x)
        self.assertIs(call_kwargs["weight"], layer.weight)
        torch.testing.assert_close(call_kwargs["antiquant_scale"], layer.weight_scale.to(x.dtype))
        torch.testing.assert_close(call_kwargs["antiquant_offset"], layer.weight_offset.to(x.dtype))
        self.assertEqual(call_kwargs["antiquant_group_size"], 4)
        self.assertIs(call_kwargs["bias"], bias)

    @patch("vllm_ascend.quantization.methods.awq.torch_npu.npu_weight_quant_batchmatmul")
    def test_apply_group_size_minus_one_uses_input_size(self, mock_matmul):
        config = AscendAWQConfig.from_config({"bits": 4, "group_size": -1, "zero_point": True})
        method = AscendAWQLinearMethod(config)
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(torch.zeros(8, 1, dtype=torch.int32), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(torch.ones(1, 8, dtype=torch.float32), requires_grad=False)
        layer.weight_offset = torch.nn.Parameter(torch.zeros(1, 8, dtype=torch.float32), requires_grad=False)
        x = torch.ones(2, 8, dtype=torch.float16)
        mock_matmul.return_value = torch.zeros(2, 8, dtype=torch.float16)

        method.apply(layer, x)

        self.assertEqual(mock_matmul.call_args.kwargs["antiquant_group_size"], 8)
