from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest


FIXTURES = Path(__file__).with_name("fixtures")


@pytest.fixture(autouse=True)
def _restore_decoder_layernorm_backend():
    from trellmlx.decoder_turing_layernorm import (
        DEFAULT_BACKEND,
        configure_decoder_layernorm_backend,
    )

    configure_decoder_layernorm_backend(DEFAULT_BACKEND)
    yield
    configure_decoder_layernorm_backend(DEFAULT_BACKEND)


def _fixture_correction_lut():
    import mlx.core as mx

    with np.load(
        FIXTURES / "source_cuda_shape_decoder_block0_norm_rows.npz",
        allow_pickle=False,
    ) as fixture:
        coordinates = np.asarray(fixture["rsqrt_coordinates"])
        deltas = np.asarray(fixture["rsqrt_deltas"])
    correction = np.zeros((1 << 24,), dtype=np.int8)
    correction[coordinates] = deltas
    return mx.array(correction)


def test_turing_decoder_affine_layernorm_matches_source_cuda_rows():
    import mlx.core as mx

    from trellmlx.decoder_turing_layernorm import (
        turing_layernorm_affine_fp16,
    )

    with np.load(
        FIXTURES / "source_cuda_shape_decoder_block0_norm_rows.npz",
        allow_pickle=False,
    ) as fixture:
        block0_conv = np.asarray(fixture["block0_conv"])
        weight = np.asarray(fixture["weight"])
        bias = np.asarray(fixture["bias"])
        expected = np.asarray(fixture["expected_norm"])
        coordinates = np.asarray(fixture["rsqrt_coordinates"])
        deltas = np.asarray(fixture["rsqrt_deltas"])

    correction = np.zeros((1 << 24,), dtype=np.int8)
    correction[coordinates] = deltas
    actual = turing_layernorm_affine_fp16(
        mx.array(block0_conv),
        mx.array(weight),
        mx.array(bias),
        mx.array(correction),
        eps=1e-6,
    )
    mx.eval(actual)

    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_turing_decoder_backend_requires_attested_lut_and_reports_identity():
    from trellmlx.decoder_turing_layernorm import (
        CUDA_WELFORD_TURING_T4_BACKEND,
        configure_decoder_layernorm_backend,
        decoder_layernorm_backend_identity,
    )

    with pytest.raises(ValueError, match="requires an explicit correction LUT"):
        configure_decoder_layernorm_backend(CUDA_WELFORD_TURING_T4_BACKEND)

    correction = _fixture_correction_lut()
    artifact_sha256 = "d2520d24f5e372fab03ff3b642af724485ad985d03d235c4bb5ef351398998e3"
    configure_decoder_layernorm_backend(
        CUDA_WELFORD_TURING_T4_BACKEND,
        turing_rsqrt_delta_lut=correction,
        turing_rsqrt_lut_artifact_sha256_attested=artifact_sha256,
    )

    identity = decoder_layernorm_backend_identity()
    assert identity["backend"] == CUDA_WELFORD_TURING_T4_BACKEND
    assert identity["cuda_architecture"] == "sm_75"
    assert identity["authenticated_contract"] == {
        "input_dtype": "float16",
        "parameter_dtype": "float16",
        "hidden_width": 1024,
        "affine": True,
    }
    assert identity["turing_rsqrt_lut_artifact_sha256_attested"] == artifact_sha256
    assert identity["turing_rsqrt_lut_content_sha256"] == hashlib.sha256(
        np.asarray(correction).tobytes()
    ).hexdigest()


def test_decoder_layernorm_consumer_dispatches_exact_backend():
    import mlx.core as mx

    from trellmlx.decoder_turing_layernorm import (
        CUDA_WELFORD_TURING_T4_BACKEND,
        configure_decoder_layernorm_backend,
    )
    from trellmlx.modules.norm import LayerNorm32

    with np.load(
        FIXTURES / "source_cuda_shape_decoder_block0_norm_rows.npz",
        allow_pickle=False,
    ) as fixture:
        block0_conv = np.asarray(fixture["block0_conv"])
        weight = np.asarray(fixture["weight"])
        bias = np.asarray(fixture["bias"])
        expected = np.asarray(fixture["expected_norm"])

    configure_decoder_layernorm_backend(
        CUDA_WELFORD_TURING_T4_BACKEND,
        turing_rsqrt_delta_lut=_fixture_correction_lut(),
        turing_rsqrt_lut_artifact_sha256_attested="a" * 64,
    )
    norm = LayerNorm32(1024, affine=True, decoder_layernorm=True)
    norm.weight = mx.array(weight)
    norm.bias = mx.array(bias)
    actual = norm(mx.array(block0_conv))
    mx.eval(actual)

    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_shape_decoder_level_zero_norms_are_exact_route_consumers():
    from trellmlx.models.shape_slat_decoder import (
        SLatDecoder,
        SparseConvNeXtBlock3d,
    )

    decoder = SLatDecoder(
        model_channels=[1024],
        num_blocks=[2],
        use_fp16=True,
    )
    blocks = [
        block
        for block in decoder.blocks[0]
        if isinstance(block, SparseConvNeXtBlock3d)
    ]

    assert len(blocks) == 2
    assert all(block.norm.decoder_layernorm for block in blocks)


def test_shape_decoder_exact_layernorm_enrollment_stops_at_authenticated_width():
    from trellmlx.models.shape_slat_decoder import (
        SLatDecoder,
        SparseConvNeXtBlock3d,
        SparseResBlockC2S3d,
    )

    decoder = SLatDecoder(
        model_channels=[1024, 512],
        num_blocks=[2, 2],
        use_fp16=True,
    )
    level0_blocks = [
        block
        for block in decoder.blocks[0]
        if isinstance(block, SparseConvNeXtBlock3d)
    ]
    level0_upsample = [
        block
        for block in decoder.blocks[0]
        if isinstance(block, SparseResBlockC2S3d)
    ]
    level1_blocks = [
        block
        for block in decoder.blocks[1]
        if isinstance(block, SparseConvNeXtBlock3d)
    ]

    assert len(level0_upsample) == 1
    assert all(block.norm.decoder_layernorm for block in level0_blocks)
    assert level0_upsample[0].norm1.decoder_layernorm is True
    assert all(block.norm.decoder_layernorm is False for block in level1_blocks)


def test_turing_decoder_layernorm_rejects_wrong_contract():
    import mlx.core as mx

    from trellmlx.decoder_turing_layernorm import (
        turing_layernorm_affine_fp16,
    )

    correction = mx.zeros((1 << 24,), dtype=mx.int8)
    with np.testing.assert_raises_regex(ValueError, "float16 input"):
        turing_layernorm_affine_fp16(
            mx.zeros((2, 8), dtype=mx.float32),
            mx.ones((8,), dtype=mx.float16),
            mx.zeros((8,), dtype=mx.float16),
            correction,
        )
    with np.testing.assert_raises_regex(ValueError, "multiple of 4"):
        turing_layernorm_affine_fp16(
            mx.zeros((2, 6), dtype=mx.float16),
            mx.ones((6,), dtype=mx.float16),
            mx.zeros((6,), dtype=mx.float16),
            correction,
        )
    with np.testing.assert_raises_regex(ValueError, "weight shape"):
        turing_layernorm_affine_fp16(
            mx.zeros((2, 8), dtype=mx.float16),
            mx.ones((7,), dtype=mx.float16),
            mx.zeros((8,), dtype=mx.float16),
            correction,
        )
    with np.testing.assert_raises_regex(ValueError, "16777216 entries"):
        turing_layernorm_affine_fp16(
            mx.zeros((2, 8), dtype=mx.float16),
            mx.ones((8,), dtype=mx.float16),
            mx.zeros((8,), dtype=mx.float16),
            mx.zeros((4,), dtype=mx.int8),
        )
