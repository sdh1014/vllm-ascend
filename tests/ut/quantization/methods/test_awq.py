from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoeWeightScaleSupported
from vllm.model_executor.layers.linear import RowParallelLinear

import vllm_ascend.quantization.methods.awq as awq_module
from tests.ut.base import TestBase
from vllm_ascend.quantization.awq_config import AscendAWQConfig
from vllm_ascend.quantization.methods.awq import (
    AWQ_INT4PACK_INNER_K_TILES,
    AWQ_REVERSE_ORDER,
    AscendAWQFusedMoEMethod,
    AscendAWQLinearMethod,
    convert_awq_to_ascend,
    convert_awq_to_ascend_batch,
    convert_awq_moe_param_to_ascend,
    make_awq_zeros,
    pack_awq_weight_to_ascend,
    prepare_awq_scale,
    prepare_awq_zero_offset,
    unpack_awq_int32,
)
from vllm_ascend.quantization.quant_type import QuantType


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    packed = torch.zeros(values.shape[0], values.shape[1] // 8, dtype=torch.int32)
    for i in range(8):
        packed |= values[:, i::8].to(torch.int32) << (4 * i)
    return packed


def pack_int4_dim0(values: torch.Tensor) -> torch.Tensor:
    packed = torch.zeros(values.shape[0] // 8, values.shape[1], dtype=torch.int32)
    for i in range(8):
        packed |= values[i::8, :].to(torch.int32) << (4 * i)
    return packed


def mock_int4pack(weight: torch.Tensor, **_: int) -> torch.Tensor:
    return weight


class TestAWQConversionHelpers(TestBase):
    def test_unpack_awq_int32_applies_upstream_reverse_order(self):
        values = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32)
        packed = pack_int4(values)

        unpacked = unpack_awq_int32(packed, torch.Size([1, 8]))

        self.assertEqual(unpacked.dtype, torch.int32)
        torch.testing.assert_close(unpacked, values[:, AWQ_REVERSE_ORDER])

    def test_unpack_awq_int32_packed_dim_1_matches_upstream_reverse_order(self):
        values = torch.arange(32, dtype=torch.int32).view(2, 16) % 16
        packed = pack_int4(values)

        unpacked = unpack_awq_int32(packed, torch.Size([2, 14]), packed_dim=1)

        expected = values.view(2, 2, 8)[:, :, AWQ_REVERSE_ORDER].reshape(2, 16)[:, :14]
        torch.testing.assert_close(unpacked, expected)

    def test_unpack_awq_int32_packed_dim_0_matches_upstream_reverse_order(self):
        values = torch.arange(48, dtype=torch.int32).view(16, 3) % 16
        packed = pack_int4_dim0(values)

        unpacked = unpack_awq_int32(packed, torch.Size([14, 3]), packed_dim=0)

        expected = values.view(2, 8, 3)[:, AWQ_REVERSE_ORDER, :].reshape(16, 3)[:14, :]
        torch.testing.assert_close(unpacked, expected)

    def test_make_awq_zeros_uses_unpacked_checkpoint_values(self):
        zeros = torch.tensor([[0, 7, 14, 1, 2, 3, 4, 5]], dtype=torch.int32)
        qzeros = pack_int4(zeros)

        unpacked = make_awq_zeros(qzeros, output_size=8)

        torch.testing.assert_close(unpacked, zeros[:, AWQ_REVERSE_ORDER])

    @patch("vllm_ascend.quantization.methods.awq.torch_npu.npu_convert_weight_to_int4pack")
    def test_pack_awq_weight_to_ascend_subtracts_8_and_passes_inner_k_tiles(self, mock_pack):
        qweight_values = torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15]], dtype=torch.int32)
        qweight = pack_int4(qweight_values)
        breakdown = []
        mock_pack.side_effect = mock_int4pack

        packed_weight = pack_awq_weight_to_ascend(
            qweight,
            output_size=8,
            breakdown=breakdown,
            inner_k_tiles=3,
        )

        expected_signed_weight = qweight_values[:, AWQ_REVERSE_ORDER] - 8
        torch.testing.assert_close(mock_pack.call_args.args[0], expected_signed_weight)
        self.assertEqual(mock_pack.call_args.kwargs["inner_k_tiles"], 3)
        torch.testing.assert_close(packed_weight, expected_signed_weight)
        self.assertEqual([stage["name"] for stage in breakdown], ["unpack_weight", "sign_weight", "pack_weight"])
        self.assertEqual(breakdown[2]["config"], {"inner_k_tiles": 3})

    @patch("vllm_ascend.quantization.methods.awq.torch_npu.npu_convert_weight_to_int4pack")
    def test_pack_awq_weight_to_ascend_uses_triton_weight_pack_by_default(self, mock_pack):
        qweight = pack_int4(torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15]], dtype=torch.int32))
        triton_weight = torch.full_like(qweight, 123)
        fake_module = MagicMock()
        fake_module.triton_kernel_launchable.return_value = True
        fake_module.awq_direct_pack_candidate.return_value = triton_weight
        mock_pack.side_effect = AssertionError("torch_npu int4pack must not be used by default Triton weight pack")

        with (
            patch.dict("os.environ", {}, clear=True),
            patch.dict("sys.modules", {"vllm_ascend.ops.triton.awq_direct_pack": fake_module}),
        ):
            packed_weight = pack_awq_weight_to_ascend(qweight, output_size=8)

        torch.testing.assert_close(packed_weight, triton_weight)
        fake_module.awq_direct_pack_candidate.assert_called_once_with(qweight, block_size=1024)
        mock_pack.assert_not_called()

    @patch("vllm_ascend.quantization.methods.awq.torch_npu.npu_convert_weight_to_int4pack")
    def test_pack_awq_weight_to_ascend_triton_weight_pack_enabled_uses_candidate(self, mock_pack):
        qweight = pack_int4(torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15]], dtype=torch.int32))
        triton_weight = torch.full_like(qweight, 123)
        breakdown = []
        fake_module = MagicMock()
        fake_module.triton_kernel_launchable.return_value = True
        fake_module.awq_direct_pack_candidate.return_value = triton_weight
        mock_pack.side_effect = AssertionError("torch_npu int4pack must not be used by Triton weight pack")
        awq_module._AWQ_TRITON_WEIGHT_PACK_WARNINGS.clear()

        with (
            patch.dict(
                "os.environ",
                {"VLLM_ASCEND_AWQ_TRITON_PREWARM_BLOCK_SIZE": "auto"},
                clear=False,
            ),
            patch.dict("sys.modules", {"vllm_ascend.ops.triton.awq_direct_pack": fake_module}),
        ):
            packed_weight = pack_awq_weight_to_ascend(qweight, output_size=8, breakdown=breakdown)

        torch.testing.assert_close(packed_weight, triton_weight)
        fake_module.awq_direct_pack_candidate.assert_called_once_with(qweight, block_size=1024)
        mock_pack.assert_not_called()
        self.assertEqual([stage["name"] for stage in breakdown], ["pack_weight"])
        self.assertEqual(
            breakdown[0]["config"],
            {
                "source": "triton_direct_pack",
                "candidate": "awq_direct_pack_candidate",
                "block_size_source": "VLLM_ASCEND_AWQ_TRITON_PREWARM_BLOCK_SIZE",
                "block_size": 1024,
            },
        )

    @patch("vllm_ascend.quantization.methods.awq.torch_npu.npu_convert_weight_to_int4pack")
    def test_pack_awq_weight_to_ascend_triton_weight_pack_uses_explicit_block_size(self, mock_pack):
        qweight = pack_int4(torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15]], dtype=torch.int32))
        triton_weight = torch.full_like(qweight, 123)
        fake_module = MagicMock()
        fake_module.triton_kernel_launchable.return_value = True
        fake_module.awq_direct_pack_candidate.return_value = triton_weight
        mock_pack.side_effect = AssertionError("torch_npu int4pack must not be used by Triton weight pack")
        awq_module._AWQ_TRITON_WEIGHT_PACK_WARNINGS.clear()

        with (
            patch.dict(
                "os.environ",
                {"VLLM_ASCEND_AWQ_TRITON_PREWARM_BLOCK_SIZE": "2048"},
                clear=False,
            ),
            patch.dict("sys.modules", {"vllm_ascend.ops.triton.awq_direct_pack": fake_module}),
        ):
            packed_weight = pack_awq_weight_to_ascend(qweight, output_size=8)

        torch.testing.assert_close(packed_weight, triton_weight)
        fake_module.awq_direct_pack_candidate.assert_called_once_with(qweight, block_size=2048)
        mock_pack.assert_not_called()

    @patch("vllm_ascend.quantization.methods.awq.torch_npu.npu_convert_weight_to_int4pack")
    def test_pack_awq_weight_to_ascend_triton_weight_pack_failure_falls_back(self, mock_pack):
        qweight_values = torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15]], dtype=torch.int32)
        qweight = pack_int4(qweight_values)
        fake_module = MagicMock()
        fake_module.triton_kernel_launchable.return_value = True
        fake_module.awq_direct_pack_candidate.side_effect = RuntimeError("launch failed")
        mock_pack.side_effect = mock_int4pack
        breakdown = []
        awq_module._AWQ_TRITON_WEIGHT_PACK_WARNINGS.clear()

        with (
            patch.dict(
                "os.environ",
                {"VLLM_ASCEND_AWQ_TRITON_PREWARM_BLOCK_SIZE": "auto"},
                clear=False,
            ),
            patch.dict("sys.modules", {"vllm_ascend.ops.triton.awq_direct_pack": fake_module}),
        ):
            packed_weight = pack_awq_weight_to_ascend(qweight, output_size=8, breakdown=breakdown)

        torch.testing.assert_close(packed_weight, qweight_values[:, AWQ_REVERSE_ORDER] - 8)
        self.assertEqual(mock_pack.call_count, 1)
        self.assertEqual([stage["name"] for stage in breakdown], ["unpack_weight", "sign_weight", "pack_weight"])
        self.assertEqual(breakdown[2]["config"], {"inner_k_tiles": 0})

    @patch("vllm_ascend.quantization.methods.awq.torch_npu.npu_convert_weight_to_int4pack")
    def test_pack_awq_weight_to_ascend_triton_weight_pack_bad_block_size_falls_back(self, mock_pack):
        qweight_values = torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15]], dtype=torch.int32)
        qweight = pack_int4(qweight_values)
        fake_module = MagicMock()
        fake_module.triton_kernel_launchable.side_effect = AssertionError(
            "Triton path must not be used when block-size env parsing fails"
        )
        mock_pack.side_effect = mock_int4pack
        awq_module._AWQ_TRITON_WEIGHT_PACK_WARNINGS.clear()

        with (
            patch.dict("os.environ", {"VLLM_ASCEND_AWQ_TRITON_PREWARM_BLOCK_SIZE": "not-an-int"}, clear=False),
            patch.dict("sys.modules", {"vllm_ascend.ops.triton.awq_direct_pack": fake_module}),
        ):
            packed_weight = pack_awq_weight_to_ascend(qweight, output_size=8)

        torch.testing.assert_close(packed_weight, qweight_values[:, AWQ_REVERSE_ORDER] - 8)
        self.assertEqual(mock_pack.call_count, 1)
        fake_module.triton_kernel_launchable.assert_not_called()

    def test_prepare_awq_scale_returns_contiguous_tensor(self):
        scales = torch.arange(16, dtype=torch.float32).view(2, 8)[:, ::2]
        self.assertFalse(scales.is_contiguous())

        prepared = prepare_awq_scale(scales)

        torch.testing.assert_close(prepared, scales)
        self.assertTrue(prepared.is_contiguous())

    def test_prepare_awq_zero_offset_uses_8_minus_unpacked_zeros(self):
        zero_values = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [7, 6, 5, 4, 3, 2, 1, 0],
            ],
            dtype=torch.int32,
        )
        qzeros = pack_int4(zero_values)
        scales = torch.arange(16, dtype=torch.float32).view(2, 8)[:, ::2]

        offset = prepare_awq_zero_offset(qzeros, scales, output_size=4, zero_point=True)

        expected_zeros = zero_values[:, AWQ_REVERSE_ORDER][:, :4]
        self.assertEqual(offset.shape, scales.shape)
        self.assertEqual(offset.dtype, scales.dtype)
        self.assertEqual(offset.device, scales.device)
        self.assertTrue(offset.is_contiguous())
        torch.testing.assert_close(offset, (8 - expected_zeros).to(scales.dtype))

    def test_prepare_awq_zero_offset_without_zero_point_returns_zeros_like_scales(self):
        qzeros = pack_int4(torch.zeros(1, 8, dtype=torch.int32))
        scales = torch.arange(8, dtype=torch.float32).view(1, 8)

        offset = prepare_awq_zero_offset(qzeros, scales, output_size=8, zero_point=False)

        self.assertEqual(offset.shape, scales.shape)
        self.assertEqual(offset.dtype, scales.dtype)
        self.assertEqual(offset.device, scales.device)
        torch.testing.assert_close(offset, torch.zeros_like(scales))

    @patch("vllm_ascend.quantization.methods.awq.torch_npu.npu_convert_weight_to_int4pack")
    def test_convert_awq_to_ascend_with_zero_point(self, mock_pack):
        qweight_values = torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15]], dtype=torch.int32)
        zero_values = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32)
        scales = torch.arange(1, 9, dtype=torch.float32).view(1, 8)
        mock_pack.side_effect = mock_int4pack

        packed_weight, scale, offset = convert_awq_to_ascend(
            pack_int4(qweight_values),
            pack_int4(zero_values),
            scales,
            zero_point=True,
        )

        self.assertEqual(mock_pack.call_args.kwargs["inner_k_tiles"], AWQ_INT4PACK_INNER_K_TILES)
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
        mock_pack.side_effect = mock_int4pack

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
        mock_pack.side_effect = mock_int4pack

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

    @patch("vllm_ascend.quantization.methods.awq.torch_npu.npu_convert_weight_to_int4pack")
    def test_convert_awq_to_ascend_triton_weight_pack_preserves_scale_and_zero(self, mock_pack):
        qweight = pack_int4(torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15]], dtype=torch.int32))
        zero_values = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32)
        qzeros = pack_int4(zero_values)
        scales = torch.arange(1, 9, dtype=torch.float32).view(1, 8)
        triton_weight = torch.full_like(qweight, 123)
        fake_module = MagicMock()
        fake_module.triton_kernel_launchable.return_value = True
        fake_module.awq_direct_pack_candidate.return_value = triton_weight
        mock_pack.side_effect = AssertionError("torch_npu int4pack must not be used by Triton weight pack")
        awq_module._AWQ_TRITON_WEIGHT_PACK_WARNINGS.clear()

        with (
            patch.dict(
                "os.environ",
                {"VLLM_ASCEND_AWQ_TRITON_PREWARM_BLOCK_SIZE": "auto"},
                clear=False,
            ),
            patch.dict("sys.modules", {"vllm_ascend.ops.triton.awq_direct_pack": fake_module}),
        ):
            packed_weight, scale, offset = convert_awq_to_ascend(qweight, qzeros, scales, zero_point=True)

        torch.testing.assert_close(packed_weight, triton_weight)
        torch.testing.assert_close(scale, scales)
        self.assertTrue(scale.is_contiguous())
        torch.testing.assert_close(offset, (8 - zero_values[:, AWQ_REVERSE_ORDER]).to(scales.dtype))
        self.assertTrue(offset.is_contiguous())
        mock_pack.assert_not_called()

    @patch("vllm_ascend.quantization.methods.awq.torch_npu.npu_convert_weight_to_int4pack")
    def test_convert_awq_to_ascend_passes_explicit_inner_k_tiles(self, mock_pack):
        qweight_values = torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15]], dtype=torch.int32)
        zero_values = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32)
        scales = torch.arange(1, 9, dtype=torch.float32).view(1, 8)
        mock_pack.side_effect = mock_int4pack

        convert_awq_to_ascend(
            pack_int4(qweight_values),
            pack_int4(zero_values),
            scales,
            zero_point=True,
            inner_k_tiles=1,
        )

        self.assertEqual(mock_pack.call_args.kwargs["inner_k_tiles"], 1)

    @patch("vllm_ascend.quantization.methods.awq.torch_npu.npu_convert_weight_to_int4pack")
    def test_convert_awq_to_ascend_does_not_mutate_checkpoint_tensors(self, mock_pack):
        qweight = pack_int4(torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15]], dtype=torch.int32))
        qzeros = pack_int4(torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32))
        scales = torch.arange(1, 9, dtype=torch.float32).view(1, 8)
        qweight_before = qweight.clone()
        qzeros_before = qzeros.clone()
        scales_before = scales.clone()
        mock_pack.side_effect = mock_int4pack

        convert_awq_to_ascend(qweight, qzeros, scales, zero_point=True)

        torch.testing.assert_close(qweight, qweight_before)
        torch.testing.assert_close(qzeros, qzeros_before)
        torch.testing.assert_close(scales, scales_before)

    @patch("vllm_ascend.quantization.methods.awq.torch_npu.npu_convert_weight_to_int4pack")
    def test_convert_awq_to_ascend_breakdown_records_stage_structure(self, mock_pack):
        qweight = pack_int4(torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15]], dtype=torch.int32))
        qzeros = pack_int4(torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32))
        scales = torch.arange(1, 9, dtype=torch.float32).view(1, 8)
        breakdown = []
        mock_pack.side_effect = mock_int4pack

        convert_awq_to_ascend(qweight, qzeros, scales, zero_point=True, breakdown=breakdown)

        self.assertEqual(
            [stage["name"] for stage in breakdown],
            ["unpack_weight", "sign_weight", "pack_weight", "prepare_scale", "unpack_zeros", "prepare_offset"],
        )
        for stage in breakdown:
            self.assertIsInstance(stage["wall_ms"], float)
            self.assertIsNone(stage["npu_event_ms"])
            self.assertEqual(
                set(stage["memory"]),
                {"before", "after", "delta"},
            )
        self.assertEqual(breakdown[2]["config"], {"inner_k_tiles": AWQ_INT4PACK_INNER_K_TILES})

    @patch("vllm_ascend.quantization.methods.awq.torch_npu.npu_convert_weight_to_int4pack")
    def test_convert_awq_to_ascend_batch_packs_concatenated_weights_once(self, mock_pack):
        qweight_values = [
            torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15]], dtype=torch.int32),
            torch.tensor([[7, 6, 5, 4, 3, 2, 1, 0]], dtype=torch.int32),
        ]
        zero_values = [
            torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32),
            torch.tensor([[7, 6, 5, 4, 3, 2, 1, 0]], dtype=torch.int32),
        ]
        scales = [
            torch.arange(1, 9, dtype=torch.float32).view(1, 8),
            torch.arange(9, 17, dtype=torch.float32).view(1, 8),
        ]
        mock_pack.side_effect = mock_int4pack

        outputs = convert_awq_to_ascend_batch(
            [pack_int4(values) for values in qweight_values],
            [pack_int4(values) for values in zero_values],
            scales,
            zero_point=True,
        )

        self.assertEqual(mock_pack.call_count, 1)
        self.assertEqual(mock_pack.call_args.args[0].shape, torch.Size([2, 8]))
        self.assertEqual(mock_pack.call_args.kwargs["inner_k_tiles"], AWQ_INT4PACK_INNER_K_TILES)
        self.assertEqual(len(outputs), 2)
        for output, qweight, qzero, scale in zip(outputs, qweight_values, zero_values, scales, strict=True):
            packed_weight, packed_scale, offset = output
            torch.testing.assert_close(packed_weight, qweight[:, AWQ_REVERSE_ORDER] - 8)
            torch.testing.assert_close(packed_scale, scale)
            torch.testing.assert_close(offset, (8 - qzero[:, AWQ_REVERSE_ORDER]).to(scale.dtype))

    @patch("vllm_ascend.quantization.methods.awq.torch_npu.npu_convert_weight_to_int4pack")
    def test_convert_awq_to_ascend_batch_without_zero_point(self, mock_pack):
        qweight_values = [
            torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32),
            torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15]], dtype=torch.int32),
        ]
        scales = [torch.ones(1, 8, dtype=torch.float32), torch.full((1, 8), 2.0)]
        mock_pack.side_effect = mock_int4pack

        outputs = convert_awq_to_ascend_batch(
            [pack_int4(values) for values in qweight_values],
            None,
            scales,
            zero_point=False,
        )

        self.assertEqual(mock_pack.call_count, 1)
        for output, qweight, scale in zip(outputs, qweight_values, scales, strict=True):
            packed_weight, packed_scale, offset = output
            torch.testing.assert_close(packed_weight, qweight[:, AWQ_REVERSE_ORDER] - 8)
            torch.testing.assert_close(packed_scale, scale)
            torch.testing.assert_close(offset, torch.zeros_like(scale))

    def test_convert_awq_to_ascend_batch_rejects_mismatched_shapes(self):
        qweights = [
            pack_int4(torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15]], dtype=torch.int32)),
            pack_int4(
                torch.tensor(
                    [
                        [7, 6, 5, 4, 3, 2, 1, 0],
                        [0, 1, 2, 3, 4, 5, 6, 7],
                    ],
                    dtype=torch.int32,
                )
            ),
        ]
        qzeros = [pack_int4(torch.zeros(1, 8, dtype=torch.int32)) for _ in qweights]
        scales = [torch.ones(1, 8, dtype=torch.float32), torch.ones(2, 8, dtype=torch.float32)]

        with self.assertRaisesRegex(ValueError, "identical shape and dtype"):
            convert_awq_to_ascend_batch(qweights, qzeros, scales, zero_point=True)


def make_awq_moe_method(group_size=4, zero_point=True):
    config = AscendAWQConfig.from_config({"bits": 4, "group_size": group_size, "zero_point": zero_point})
    with patch("vllm_ascend.quantization.methods.awq.get_ascend_config") as mock_get_ascend_config:
        mock_get_ascend_config.return_value.eplb_config.dynamic_eplb = False
        return AscendAWQFusedMoEMethod(config, MagicMock())


def make_vllm_fused_moe_loader(tp_rank: int):
    layer = FusedMoE.__new__(FusedMoE)
    torch.nn.Module.__init__(layer)
    layer.quant_config = MagicMock()
    layer.quant_config.get_name.return_value = "awq"
    layer.quant_method = MagicMock()
    layer.expert_map_manager = MagicMock()
    layer.expert_map_manager.map_global_to_local.side_effect = lambda expert_id: expert_id
    layer._expert_map = None
    layer.global_num_experts = 1
    layer.moe_parallel_config = MagicMock()
    layer.moe_parallel_config.tp_rank = tp_rank
    layer.moe_parallel_config.tp_size = 2
    layer.moe_config = MagicMock()
    layer.moe_config.is_act_and_mul = True
    return layer


class TestAscendAWQFusedMoEMethod(TestBase):
    @patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=1)
    @patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=0)
    def test_create_weights_uses_awq_moe_checkpoint_names_and_shapes(self, *_):
        method = make_awq_moe_method(group_size=4)
        layer = torch.nn.Module()
        weight_loader = MagicMock()

        method.create_weights(
            layer,
            num_experts=2,
            hidden_size=16,
            intermediate_size_per_partition=8,
            params_dtype=torch.float16,
            weight_loader=weight_loader,
        )

        self.assertEqual(layer.w13_qweight.shape, torch.Size([2, 16, 2]))
        self.assertEqual(layer.w13_qweight.dtype, torch.int32)
        self.assertEqual(layer.w13_qzeros.shape, torch.Size([2, 4, 2]))
        self.assertEqual(layer.w13_scales.shape, torch.Size([2, 4, 16]))
        self.assertEqual(layer.w2_qweight.shape, torch.Size([2, 8, 2]))
        self.assertEqual(layer.w2_qzeros.shape, torch.Size([2, 2, 2]))
        self.assertEqual(layer.w2_scales.shape, torch.Size([2, 2, 16]))
        self.assertEqual(layer.w13_qweight.packed_dim, 2)
        self.assertEqual(layer.w13_qweight.packed_factor, 8)
        self.assertEqual(layer.w2_qzeros.packed_dim, 2)
        self.assertEqual(layer.w2_qzeros.packed_factor, 8)
        self.assertTrue(layer.w13_qweight.is_transposed)
        self.assertTrue(layer.w2_qweight.is_transposed)
        self.assertEqual(layer.w13_qzeros.quant_method, FusedMoeWeightScaleSupported.GROUP.value)
        self.assertEqual(layer.w13_scales.quant_method, FusedMoeWeightScaleSupported.GROUP.value)
        self.assertEqual(layer.w2_qzeros.quant_method, FusedMoeWeightScaleSupported.GROUP.value)
        self.assertEqual(layer.w2_scales.quant_method, FusedMoeWeightScaleSupported.GROUP.value)

    @patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=1)
    @patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=0)
    def test_vllm_fused_moe_loader_slices_awq_moe_tp_rank(self, *_):
        method = make_awq_moe_method(group_size=4)
        layer = torch.nn.Module()
        moe_loader = make_vllm_fused_moe_loader(tp_rank=1)
        self.assertEqual(moe_loader.tp_rank, 1)

        method.create_weights(
            layer,
            num_experts=1,
            hidden_size=8,
            intermediate_size_per_partition=8,
            params_dtype=torch.float32,
            weight_loader=moe_loader.weight_loader,
        )

        full_w1_qweight = torch.arange(16, dtype=torch.int32).view(8, 2)
        full_w3_qweight = torch.arange(100, 116, dtype=torch.int32).view(8, 2)
        full_w2_qweight = torch.arange(200, 216, dtype=torch.int32).view(16, 1)
        layer.w13_qweight.weight_loader(layer.w13_qweight, full_w1_qweight, "w13_qweight", "w1", 0)
        layer.w13_qweight.weight_loader(layer.w13_qweight, full_w3_qweight, "w13_qweight", "w3", 0)
        layer.w2_qweight.weight_loader(layer.w2_qweight, full_w2_qweight, "w2_qweight", "w2", 0)

        full_w1_qzeros = torch.arange(4, dtype=torch.int32).view(2, 2)
        full_w3_qzeros = torch.arange(10, 14, dtype=torch.int32).view(2, 2)
        full_w2_qzeros = torch.arange(20, 24, dtype=torch.int32).view(4, 1)
        layer.w13_qzeros.weight_loader(layer.w13_qzeros, full_w1_qzeros, "w13_qzeros", "w1", 0)
        layer.w13_qzeros.weight_loader(layer.w13_qzeros, full_w3_qzeros, "w13_qzeros", "w3", 0)
        layer.w2_qzeros.weight_loader(layer.w2_qzeros, full_w2_qzeros, "w2_qzeros", "w2", 0)

        full_w1_scales = torch.arange(32, dtype=torch.float32).view(2, 16)
        full_w3_scales = torch.arange(100, 132, dtype=torch.float32).view(2, 16)
        full_w2_scales = torch.arange(200, 232, dtype=torch.float32).view(4, 8)
        layer.w13_scales.weight_loader(layer.w13_scales, full_w1_scales, "w13_scales", "w1", 0)
        layer.w13_scales.weight_loader(layer.w13_scales, full_w3_scales, "w13_scales", "w3", 0)
        layer.w2_scales.weight_loader(layer.w2_scales, full_w2_scales, "w2_scales", "w2", 0)

        torch.testing.assert_close(layer.w13_qweight[0, :, 0], full_w1_qweight[:, 1])
        torch.testing.assert_close(layer.w13_qweight[0, :, 1], full_w3_qweight[:, 1])
        torch.testing.assert_close(layer.w2_qweight[0], full_w2_qweight[8:16])
        torch.testing.assert_close(layer.w13_qzeros[0, :, 0], full_w1_qzeros[:, 1])
        torch.testing.assert_close(layer.w13_qzeros[0, :, 1], full_w3_qzeros[:, 1])
        torch.testing.assert_close(layer.w2_qzeros[0], full_w2_qzeros[2:4])
        torch.testing.assert_close(layer.w13_scales[0, :, :8], full_w1_scales[:, 8:16])
        torch.testing.assert_close(layer.w13_scales[0, :, 8:16], full_w3_scales[:, 8:16])
        torch.testing.assert_close(layer.w2_scales[0], full_w2_scales[2:4])

    @patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=1)
    @patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=0)
    def test_create_weights_covers_tp_intermediate_partition(self, *_):
        method = make_awq_moe_method(group_size=4)
        layer = torch.nn.Module()

        method.create_weights(
            layer,
            num_experts=2,
            hidden_size=16,
            intermediate_size_per_partition=4,
            params_dtype=torch.float16,
        )

        self.assertEqual(layer.w13_qweight.shape, torch.Size([2, 16, 1]))
        self.assertEqual(layer.w13_qzeros.shape, torch.Size([2, 4, 1]))
        self.assertEqual(layer.w13_scales.shape, torch.Size([2, 4, 8]))
        self.assertEqual(layer.w2_qweight.shape, torch.Size([2, 4, 2]))
        self.assertEqual(layer.w2_qzeros.shape, torch.Size([2, 1, 2]))
        self.assertEqual(layer.w2_scales.shape, torch.Size([2, 1, 16]))

    @patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=1)
    @patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=0)
    def test_create_weights_group_size_minus_one_uses_each_moe_input_size(self, *_):
        method = make_awq_moe_method(group_size=-1)
        layer = torch.nn.Module()

        method.create_weights(
            layer,
            num_experts=2,
            hidden_size=16,
            intermediate_size_per_partition=8,
            params_dtype=torch.float16,
        )

        self.assertEqual(layer.w13_qzeros.shape, torch.Size([2, 1, 2]))
        self.assertEqual(layer.w13_scales.shape, torch.Size([2, 1, 16]))
        self.assertEqual(layer.w2_qzeros.shape, torch.Size([2, 1, 2]))
        self.assertEqual(layer.w2_scales.shape, torch.Size([2, 1, 16]))

    @patch("vllm_ascend.quantization.methods.awq.torch_npu.npu_convert_weight_to_int4pack")
    def test_convert_awq_moe_param_to_ascend_converts_each_expert_partition(self, mock_pack):
        qweight_values = torch.tensor(
            [
                [[8, 9, 10, 11, 12, 13, 14, 15]],
                [[7, 6, 5, 4, 3, 2, 1, 0]],
            ],
            dtype=torch.int32,
        )
        zero_values = torch.tensor(
            [
                [[0, 1, 2, 3, 4, 5, 6, 7]],
                [[7, 6, 5, 4, 3, 2, 1, 0]],
            ],
            dtype=torch.int32,
        )
        scales = torch.arange(1, 17, dtype=torch.float32).view(2, 1, 8)
        mock_pack.side_effect = mock_int4pack

        weight, scale, offset = convert_awq_moe_param_to_ascend(
            torch.stack([pack_int4(expert) for expert in qweight_values]),
            torch.stack([pack_int4(expert) for expert in zero_values]),
            scales,
            output_size=8,
            zero_point=True,
        )

        self.assertEqual(mock_pack.call_count, 1)
        self.assertEqual(weight.shape, torch.Size([2, 1, 8]))
        self.assertEqual(scale.shape, torch.Size([2, 1, 8]))
        self.assertEqual(offset.shape, torch.Size([2, 1, 8]))
        torch.testing.assert_close(weight, qweight_values[:, :, AWQ_REVERSE_ORDER] - 8)
        torch.testing.assert_close(scale, scales)
        torch.testing.assert_close(offset, (8 - zero_values[:, :, AWQ_REVERSE_ORDER]).to(scales.dtype))

    @patch("vllm_ascend.quantization.methods.awq.torch_npu.npu_convert_weight_to_int4pack")
    def test_process_weights_after_loading_registers_awq_moe_runtime_params(self, mock_pack):
        method = make_awq_moe_method(group_size=1)
        layer = torch.nn.Module()
        qweight_values = torch.tensor([[[8, 9, 10, 11, 12, 13, 14, 15]]], dtype=torch.int32)
        zero_values = torch.tensor([[[0, 1, 2, 3, 4, 5, 6, 7]]], dtype=torch.int32)
        mock_pack.side_effect = mock_int4pack

        layer.w13_qweight = torch.nn.Parameter(torch.stack([pack_int4(qweight_values[0])]), requires_grad=False)
        layer.w13_qzeros = torch.nn.Parameter(torch.stack([pack_int4(zero_values[0])]), requires_grad=False)
        layer.w13_scales = torch.nn.Parameter(torch.ones(1, 1, 8, dtype=torch.float32), requires_grad=False)
        layer.w2_qweight = torch.nn.Parameter(torch.stack([pack_int4(qweight_values[0])]), requires_grad=False)
        layer.w2_qzeros = torch.nn.Parameter(torch.stack([pack_int4(zero_values[0])]), requires_grad=False)
        layer.w2_scales = torch.nn.Parameter(torch.ones(1, 1, 8, dtype=torch.float32), requires_grad=False)

        method.process_weights_after_loading(layer)

        self.assertEqual(layer.w13_weight.shape, torch.Size([1, 1, 8]))
        self.assertEqual(layer.w13_weight_scale.shape, torch.Size([1, 1, 8]))
        self.assertEqual(layer.w13_weight_offset.shape, torch.Size([1, 1, 8]))
        self.assertEqual(layer.w2_weight.shape, torch.Size([1, 1, 8]))
        torch.testing.assert_close(layer.w13_weight, qweight_values[:, :, AWQ_REVERSE_ORDER] - 8)
        torch.testing.assert_close(layer.w2_weight_offset, (8 - zero_values[:, :, AWQ_REVERSE_ORDER]).float())

    @patch("vllm_ascend.quantization.methods.awq._EXTRA_CTX")
    @patch("vllm_ascend.quantization.methods.awq.select_experts")
    def test_apply_uses_awq_moe_runtime_payload(self, mock_select_experts, mock_extra_ctx):
        method = make_awq_moe_method(group_size=4)
        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(torch.zeros(2, 16, 2, dtype=torch.int32), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(torch.zeros(2, 8, 2, dtype=torch.int32), requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(torch.ones(2, 4, 16), requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(torch.ones(2, 2, 16), requires_grad=False)
        layer.w13_weight_offset = torch.nn.Parameter(torch.zeros(2, 4, 16), requires_grad=False)
        layer.w2_weight_offset = torch.nn.Parameter(torch.zeros(2, 2, 16), requires_grad=False)
        x = torch.randn(3, 16, dtype=torch.float32)
        router_logits = torch.randn(3, 2, dtype=torch.float32)
        topk_weights = torch.randn(3, 1, dtype=torch.float32)
        topk_ids = torch.tensor([[0], [1], [0]], dtype=torch.int64)
        pertoken_scale = torch.randn(3, dtype=torch.float32)

        mock_select_experts.return_value = (topk_weights, topk_ids)
        mock_comm = MagicMock()
        mock_comm.fused_experts.return_value = torch.randn(3, 16, dtype=torch.float32)
        mock_extra_ctx.moe_comm_method = mock_comm

        method.apply(
            layer=layer,
            x=x,
            router_logits=router_logits,
            top_k=1,
            renormalize=True,
            num_experts=2,
            routed_scaling_factor=1.5,
            activation="gelu",
            apply_router_weight_on_input=True,
            pertoken_scale=pertoken_scale,
        )

        mock_select_experts.assert_called_once()
        self.assertEqual(mock_select_experts.call_args.kwargs["num_experts"], 2)
        self.assertEqual(mock_select_experts.call_args.kwargs["routed_scaling_factor"], 1.5)
        fused_experts_input = mock_comm.fused_experts.call_args.kwargs["fused_experts_input"]
        self.assertEqual(fused_experts_input.quant.quant_type, QuantType.W4A16)
        self.assertIs(fused_experts_input.weights.w1, layer.w13_weight)
        self.assertIs(fused_experts_input.weights.w2, layer.w2_weight)
        self.assertIs(fused_experts_input.weights.w1_scale, layer.w13_weight_scale)
        self.assertIs(fused_experts_input.weights.w2_scale, layer.w2_weight_scale)
        self.assertIs(fused_experts_input.weights.w1_offset, layer.w13_weight_offset)
        self.assertIs(fused_experts_input.weights.w2_offset, layer.w2_weight_offset)
        self.assertEqual(fused_experts_input.activation, "gelu")
        self.assertTrue(fused_experts_input.routing.apply_router_weight_on_input)
        self.assertIs(fused_experts_input.routing.pertoken_scale, pertoken_scale)


class TestAscendAWQLinearMethod(TestBase):
    def make_row_parallel_layer(self, tp_rank: int):
        layer = RowParallelLinear.__new__(RowParallelLinear)
        torch.nn.Module.__init__(layer)
        layer.tp_rank = tp_rank
        return layer

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

    def test_create_weights_row_parallel_uses_overlapping_awq_groups(self):
        config = AscendAWQConfig.from_config({"bits": 4, "group_size": 128, "zero_point": True})
        method = AscendAWQLinearMethod(config)

        for tp_rank in (0, 1):
            with (
                self.subTest(tp_rank=tp_rank),
                patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=2),
                patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=tp_rank),
            ):
                layer = self.make_row_parallel_layer(tp_rank)

                method.create_weights(
                    layer,
                    input_size_per_partition=448,
                    output_partition_sizes=[16],
                    input_size=896,
                    output_size=16,
                    params_dtype=torch.float16,
                )

            self.assertEqual(layer.qweight.shape, torch.Size([448, 2]))
            self.assertEqual(layer.qzeros.shape, torch.Size([4, 2]))
            self.assertEqual(layer.scales.shape, torch.Size([4, 16]))

    def test_create_weights_row_parallel_group_size_minus_one_uses_local_group(self):
        config = AscendAWQConfig.from_config({"bits": 4, "group_size": -1, "zero_point": True})
        method = AscendAWQLinearMethod(config)

        with (
            patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=2),
            patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=1),
        ):
            layer = self.make_row_parallel_layer(tp_rank=1)
            method.create_weights(
                layer,
                input_size_per_partition=448,
                output_partition_sizes=[16],
                input_size=896,
                output_size=16,
                params_dtype=torch.float16,
            )

        self.assertEqual(layer.qzeros.shape, torch.Size([1, 2]))
        self.assertEqual(layer.scales.shape, torch.Size([1, 16]))

    def test_row_parallel_group_loader_slices_overlapping_qzeros_and_scales(self):
        config = AscendAWQConfig.from_config({"bits": 4, "group_size": 128, "zero_point": True})
        method = AscendAWQLinearMethod(config)
        full_qzeros = torch.arange(14, dtype=torch.int32).view(7, 2)
        full_scales = torch.arange(112, dtype=torch.float16).view(7, 16)

        for tp_rank, expected_group_start in ((0, 0), (1, 3)):
            with (
                self.subTest(tp_rank=tp_rank),
                patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=2),
                patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=tp_rank),
            ):
                layer = self.make_row_parallel_layer(tp_rank)
                method.create_weights(
                    layer,
                    input_size_per_partition=448,
                    output_partition_sizes=[16],
                    input_size=896,
                    output_size=16,
                    params_dtype=torch.float16,
                )
                layer.qzeros.weight_loader(layer.qzeros, full_qzeros)
                layer.scales.weight_loader(layer.scales, full_scales)

            expected_slice = slice(expected_group_start, expected_group_start + 4)
            torch.testing.assert_close(layer.qzeros, full_qzeros[expected_slice])
            torch.testing.assert_close(layer.scales, full_scales[expected_slice])

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
        mock_pack.side_effect = mock_int4pack

        method.process_weights_after_loading(layer)

        self.assertEqual(mock_pack.call_args.kwargs["inner_k_tiles"], AWQ_INT4PACK_INNER_K_TILES)
        self.assertEqual(layer.weight.shape, torch.Size([4, 8]))
        self.assertEqual(layer.weight_scale.shape, torch.Size([1, 8]))
        self.assertEqual(layer.weight_offset.shape, torch.Size([1, 8]))
        torch.testing.assert_close(layer.weight, qweight_values[:, AWQ_REVERSE_ORDER] - 8)
        torch.testing.assert_close(
            layer.weight_offset,
            8 - zero_values[:, AWQ_REVERSE_ORDER].to(torch.float32),
        )

    @patch("vllm_ascend.quantization.methods.awq.torch_npu.npu_convert_weight_to_int4pack")
    def test_process_weights_after_loading_triton_prewarm_is_disabled_by_default(self, mock_pack):
        config = AscendAWQConfig.from_config({"bits": 4, "group_size": 4, "zero_point": True})
        method = AscendAWQLinearMethod(config)
        layer = torch.nn.Module()
        layer.qweight = torch.nn.Parameter(
            pack_int4(torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15]], dtype=torch.int32)),
            requires_grad=False,
        )
        layer.qzeros = torch.nn.Parameter(
            pack_int4(torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32)),
            requires_grad=False,
        )
        layer.scales = torch.nn.Parameter(torch.ones(1, 8, dtype=torch.float32), requires_grad=False)
        mock_pack.side_effect = mock_int4pack

        fake_module = MagicMock()
        fake_module.triton_kernel_launchable.side_effect = AssertionError(
            "Triton path must not be used when prewarm is disabled"
        )
        with (
            patch.dict("os.environ", {"VLLM_ASCEND_AWQ_TRITON_PREWARM": "0"}, clear=False),
            patch.dict("sys.modules", {"vllm_ascend.ops.triton.awq_direct_pack": fake_module}),
        ):
            method.process_weights_after_loading(layer)

        self.assertEqual(mock_pack.call_count, 1)

    @patch("vllm_ascend.quantization.methods.awq.torch_npu.npu_convert_weight_to_int4pack")
    def test_process_weights_after_loading_triton_prewarm_deduplicates_specialization(self, mock_pack):
        config = AscendAWQConfig.from_config({"bits": 4, "group_size": 4, "zero_point": True})
        method = AscendAWQLinearMethod(config)
        calls = []

        def candidate(qweight, block_size=None):
            calls.append((tuple(qweight.shape), block_size))
            return qweight.clone()

        def make_layer():
            layer = torch.nn.Module()
            layer.qweight = torch.nn.Parameter(
                pack_int4(torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15]], dtype=torch.int32)),
                requires_grad=False,
            )
            layer.qzeros = torch.nn.Parameter(
                pack_int4(torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32)),
                requires_grad=False,
            )
            layer.scales = torch.nn.Parameter(torch.ones(1, 8, dtype=torch.float32), requires_grad=False)
            return layer

        fake_module = MagicMock()
        fake_module.triton_kernel_launchable.return_value = True
        fake_module.awq_direct_pack_candidate.side_effect = candidate
        mock_pack.side_effect = mock_int4pack
        awq_module._AWQ_TRITON_PREWARM_ATTEMPTED_KEYS.clear()
        awq_module._AWQ_TRITON_PREWARM_WARNINGS.clear()
        awq_module._AWQ_TRITON_PREWARM_UNAVAILABLE_REASON = None

        with (
            patch.dict(
                "os.environ",
                {
                    "VLLM_ASCEND_AWQ_TRITON_PREWARM": "1",
                    "VLLM_ASCEND_AWQ_TRITON_PREWARM_BLOCK_SIZE": "512",
                },
                clear=False,
            ),
            patch.dict("sys.modules", {"vllm_ascend.ops.triton.awq_direct_pack": fake_module}),
        ):
            method.process_weights_after_loading(make_layer())
            method.process_weights_after_loading(make_layer())

        self.assertEqual(calls, [((1, 1), 512), ((1, 1), 512), ((1, 1), 512)])
        mock_pack.assert_not_called()

    def test_triton_prewarm_auto_block_size_uses_stable_default_for_all_shapes(self):
        calls = []

        def candidate(qweight, block_size=None):
            calls.append((tuple(qweight.shape), block_size))
            return qweight.clone()

        fake_module = MagicMock()
        fake_module.triton_kernel_launchable.return_value = True
        fake_module.awq_direct_pack_candidate.side_effect = candidate
        awq_module._AWQ_TRITON_PREWARM_ATTEMPTED_KEYS.clear()
        awq_module._AWQ_TRITON_PREWARM_WARNINGS.clear()
        awq_module._AWQ_TRITON_PREWARM_UNAVAILABLE_REASON = None

        with (
            patch.dict(
                "os.environ",
                {
                    "VLLM_ASCEND_AWQ_TRITON_PREWARM": "1",
                    "VLLM_ASCEND_AWQ_TRITON_PREWARM_BLOCK_SIZE": "auto",
                },
                clear=False,
            ),
            patch.dict("sys.modules", {"vllm_ascend.ops.triton.awq_direct_pack": fake_module}),
        ):
            awq_module._awq_triton_direct_pack_prewarm(torch.empty(3584, 448, dtype=torch.int32))
            awq_module._awq_triton_direct_pack_prewarm(torch.empty(3584, 64, dtype=torch.int32))
            awq_module._awq_triton_direct_pack_prewarm(torch.empty(3584, 2368, dtype=torch.int32))
            awq_module._awq_triton_direct_pack_prewarm(torch.empty(18944, 448, dtype=torch.int32))

        self.assertEqual(
            calls,
            [
                ((3584, 448), 1024),
                ((3584, 64), 1024),
                ((3584, 2368), 1024),
                ((18944, 448), 1024),
            ],
        )

    def test_triton_prewarm_auto_block_size_uses_1024_for_arbitrary_shape(self):
        calls = []

        def candidate(qweight, block_size=None):
            calls.append((tuple(qweight.shape), block_size))
            return qweight.clone()

        fake_module = MagicMock()
        fake_module.triton_kernel_launchable.return_value = True
        fake_module.awq_direct_pack_candidate.side_effect = candidate
        awq_module._AWQ_TRITON_PREWARM_ATTEMPTED_KEYS.clear()
        awq_module._AWQ_TRITON_PREWARM_WARNINGS.clear()
        awq_module._AWQ_TRITON_PREWARM_UNAVAILABLE_REASON = None

        with (
            patch.dict(
                "os.environ",
                {
                    "VLLM_ASCEND_AWQ_TRITON_PREWARM": "1",
                    "VLLM_ASCEND_AWQ_TRITON_PREWARM_BLOCK_SIZE": "auto",
                },
                clear=False,
            ),
            patch.dict("sys.modules", {"vllm_ascend.ops.triton.awq_direct_pack": fake_module}),
        ):
            awq_module._awq_triton_direct_pack_prewarm(torch.empty(1, 1, dtype=torch.int32))

        self.assertEqual(calls, [((1, 1), 1024)])

    @patch("vllm_ascend.quantization.methods.awq.torch_npu.npu_convert_weight_to_int4pack")
    def test_process_weights_after_loading_triton_prewarm_failure_does_not_break_loading(self, mock_pack):
        config = AscendAWQConfig.from_config({"bits": 4, "group_size": 4, "zero_point": True})
        method = AscendAWQLinearMethod(config)
        layer = torch.nn.Module()
        layer.qweight = torch.nn.Parameter(
            pack_int4(torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15]], dtype=torch.int32)),
            requires_grad=False,
        )
        layer.qzeros = torch.nn.Parameter(
            pack_int4(torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32)),
            requires_grad=False,
        )
        layer.scales = torch.nn.Parameter(torch.ones(1, 8, dtype=torch.float32), requires_grad=False)
        fake_module = MagicMock()
        fake_module.triton_kernel_launchable.return_value = True
        fake_module.awq_direct_pack_candidate.side_effect = RuntimeError("compile failed")
        mock_pack.side_effect = mock_int4pack
        awq_module._AWQ_TRITON_PREWARM_ATTEMPTED_KEYS.clear()
        awq_module._AWQ_TRITON_PREWARM_WARNINGS.clear()
        awq_module._AWQ_TRITON_PREWARM_UNAVAILABLE_REASON = None

        with (
            patch.dict(
                "os.environ",
                {
                    "VLLM_ASCEND_AWQ_TRITON_PREWARM": "1",
                    "VLLM_ASCEND_AWQ_TRITON_PREWARM_BLOCK_SIZE": "auto",
                },
                clear=False,
            ),
            patch.dict("sys.modules", {"vllm_ascend.ops.triton.awq_direct_pack": fake_module}),
        ):
            method.process_weights_after_loading(layer)

        self.assertEqual(mock_pack.call_count, 1)
        self.assertEqual(layer.weight.shape, torch.Size([1, 8]))

    @patch("vllm_ascend.quantization.methods.awq.torch_npu.npu_convert_weight_to_int4pack")
    def test_process_weights_after_loading_triton_prewarm_bad_env_does_not_break_loading(self, mock_pack):
        config = AscendAWQConfig.from_config({"bits": 4, "group_size": 4, "zero_point": True})
        method = AscendAWQLinearMethod(config)
        mock_pack.side_effect = mock_int4pack
        awq_module._AWQ_TRITON_PREWARM_ATTEMPTED_KEYS.clear()
        awq_module._AWQ_TRITON_PREWARM_WARNINGS.clear()
        awq_module._AWQ_TRITON_PREWARM_UNAVAILABLE_REASON = None

        def make_layer():
            layer = torch.nn.Module()
            layer.qweight = torch.nn.Parameter(
                pack_int4(torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15]], dtype=torch.int32)),
                requires_grad=False,
            )
            layer.qzeros = torch.nn.Parameter(
                pack_int4(torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32)),
                requires_grad=False,
            )
            layer.scales = torch.nn.Parameter(torch.ones(1, 8, dtype=torch.float32), requires_grad=False)
            return layer

        fake_module = MagicMock()
        fake_module.triton_kernel_launchable.side_effect = AssertionError(
            "Triton path must not be used when prewarm env parsing fails"
        )
        env_cases = [
            {"VLLM_ASCEND_AWQ_TRITON_PREWARM": "2"},
            {
                "VLLM_ASCEND_AWQ_TRITON_PREWARM": "1",
                "VLLM_ASCEND_AWQ_TRITON_PREWARM_BLOCK_SIZE": "0",
            },
            {
                "VLLM_ASCEND_AWQ_TRITON_PREWARM": "1",
                "VLLM_ASCEND_AWQ_TRITON_PREWARM_BLOCK_SIZE": "not-an-int",
            },
        ]

        for env_overrides in env_cases:
            with (
                self.subTest(env_overrides=env_overrides),
                patch.dict("os.environ", env_overrides, clear=False),
                patch.dict("sys.modules", {"vllm_ascend.ops.triton.awq_direct_pack": fake_module}),
            ):
                layer = make_layer()
                method.process_weights_after_loading(layer)
                self.assertEqual(layer.weight.shape, torch.Size([1, 8]))

        self.assertEqual(mock_pack.call_count, len(env_cases))
        self.assertEqual(fake_module.triton_kernel_launchable.call_count, 1)

    @patch("vllm_ascend.quantization.methods.awq.torch_npu.npu_weight_quant_batchmatmul")
    def test_apply_uses_npu_weight_quant_matmul(self, mock_matmul):
        config = AscendAWQConfig.from_config({"bits": 4, "group_size": 4, "zero_point": True})
        method = AscendAWQLinearMethod(config)
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(torch.zeros(8, 1, dtype=torch.int32), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(torch.ones(2, 8, dtype=torch.float32), requires_grad=False)
        layer.weight_offset = torch.nn.Parameter(torch.zeros(2, 8, dtype=torch.float32), requires_grad=False)
        bias = torch.ones(8, dtype=torch.float16)
        x = torch.ones(2, 8, dtype=torch.float16)
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
    def test_apply_group_size_minus_one_uses_per_tensor_group_size(self, mock_matmul):
        config = AscendAWQConfig.from_config({"bits": 4, "group_size": -1, "zero_point": True})
        method = AscendAWQLinearMethod(config)
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(torch.zeros(8, 1, dtype=torch.int32), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(torch.ones(1, 8, dtype=torch.float32), requires_grad=False)
        layer.weight_offset = torch.nn.Parameter(torch.zeros(1, 8, dtype=torch.float32), requires_grad=False)
        x = torch.ones(2, 8, dtype=torch.float16)
        mock_matmul.return_value = torch.zeros(2, 8, dtype=torch.float16)

        method.apply(layer, x)

        self.assertEqual(mock_matmul.call_args.kwargs["antiquant_group_size"], 0)

    @patch("vllm_ascend.quantization.methods.awq.torch_npu.npu_weight_quant_batchmatmul")
    def test_apply_group_size_equal_input_uses_per_tensor_group_size(self, mock_matmul):
        config = AscendAWQConfig.from_config({"bits": 4, "group_size": 8, "zero_point": True})
        method = AscendAWQLinearMethod(config)
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(torch.zeros(8, 1, dtype=torch.int32), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(torch.ones(1, 8, dtype=torch.float32), requires_grad=False)
        layer.weight_offset = torch.nn.Parameter(torch.zeros(1, 8, dtype=torch.float32), requires_grad=False)
        x = torch.ones(2, 8, dtype=torch.float16)
        mock_matmul.return_value = torch.zeros(2, 8, dtype=torch.float16)

        method.apply(layer, x)

        self.assertEqual(mock_matmul.call_args.kwargs["antiquant_group_size"], 0)
