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
    assert identity["authenticated_contracts"] == [
        {
            "input_dtype": "float16",
            "parameter_dtype": "float16",
            "hidden_width": 1024,
            "affine": True,
            "reduction": {
                "threads": 128,
                "warps": 4,
                "vector_width": 4,
                "values_per_thread": 8,
                "accumulator_dtype": "float32",
            },
        },
        {
            "input_dtype": "float16",
            "parameter_dtype": "float16",
            "hidden_width": 512,
            "affine": True,
            "reduction": {
                "threads": 128,
                "warps": 4,
                "vector_width": 4,
                "values_per_thread": 4,
                "accumulator_dtype": "float32",
            },
        },
        {
            "input_dtype": "float16",
            "hidden_width": 512,
            "affine": False,
            "reduction": {
                "threads": 128,
                "warps": 4,
                "vector_width": 4,
                "values_per_thread": 4,
                "accumulator_dtype": "float32",
            },
        },
        {
            "input_dtype": "float16",
            "parameter_dtype": "float16",
            "hidden_width": 256,
            "affine": True,
            "reduction": {
                "threads": 128,
                "warps": 4,
                "vector_width": 4,
                "active_values_per_thread": 4,
                "average_values_per_launched_thread": 2,
                "active_vector_threads": 64,
                "inactive_vector_threads": 64,
                "accumulator_dtype": "float32",
            },
        },
        {
            "input_dtype": "float16",
            "hidden_width": 256,
            "affine": False,
            "reduction": {
                "threads": 128,
                "warps": 4,
                "vector_width": 4,
                "active_values_per_thread": 4,
                "average_values_per_launched_thread": 2,
                "active_vector_threads": 64,
                "inactive_vector_threads": 64,
                "accumulator_dtype": "float32",
            },
        },
        {
            "input_dtype": "float16",
            "parameter_dtype": "float16",
            "hidden_width": 128,
            "affine": True,
            "reduction": {
                "threads": 128,
                "warps": 4,
                "vector_width": 4,
                "active_values_per_thread": 4,
                "average_values_per_launched_thread": 1,
                "active_vector_threads": 32,
                "inactive_vector_threads": 96,
                "accumulator_dtype": "float32",
            },
        },
        {
            "input_dtype": "float16",
            "hidden_width": 128,
            "affine": False,
            "reduction": {
                "threads": 128,
                "warps": 4,
                "vector_width": 4,
                "active_values_per_thread": 4,
                "average_values_per_launched_thread": 1,
                "active_vector_threads": 32,
                "inactive_vector_threads": 96,
                "accumulator_dtype": "float32",
            },
        },
        {
            "input_dtype": "float16",
            "hidden_width": 64,
            "affine": False,
            "reduction": {
                "threads": 128,
                "warps": 4,
                "vector_width": 4,
                "active_values_per_thread": 4,
                "average_values_per_launched_thread": 0.5,
                "active_vector_threads": 16,
                "inactive_vector_threads": 112,
                "accumulator_dtype": "float32",
            },
        },
        {
            "input_dtype": "float32",
            "hidden_width": 64,
            "affine": False,
            "eps": 1e-5,
            "consumer": "shape-decoder-terminal",
            "reduction": {
                "threads": 128,
                "warps": 4,
                "vector_width": 4,
                "active_values_per_thread": 4,
                "average_values_per_launched_thread": 0.5,
                "active_vector_threads": 16,
                "inactive_vector_threads": 112,
                "accumulator_dtype": "float32",
            },
        },
    ]
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


def test_decoder_noaffine_layernorm_consumer_dispatches_exact_backend(monkeypatch):
    import mlx.core as mx

    import trellmlx.decoder_turing_layernorm as decoder_layernorm
    from trellmlx.modules.norm import LayerNorm32

    called = {}

    def fake_noaffine(x, eps):
        called["shape"] = x.shape
        called["eps"] = eps
        return mx.full(x.shape, 7, dtype=x.dtype)

    monkeypatch.setattr(
        decoder_layernorm,
        "layernorm_noaffine",
        fake_noaffine,
        raising=False,
    )
    norm = LayerNorm32(512, affine=False, decoder_layernorm=True)
    actual = norm(mx.zeros((3, 512), dtype=mx.float16))
    mx.eval(actual)

    assert called == {"shape": (3, 512), "eps": 1e-6}
    np.testing.assert_array_equal(np.asarray(actual), np.full((3, 512), 7))


def test_turing_decoder_noaffine_layernorm_matches_identity_affine_schedule():
    import mlx.core as mx

    from trellmlx.decoder_turing_layernorm import (
        turing_layernorm_affine_fp16,
        turing_layernorm_noaffine_fp16,
    )

    values = np.linspace(-9.0, 11.0, 3 * 512, dtype=np.float16).reshape(3, 512)
    correction = mx.zeros((1 << 24,), dtype=mx.int8)
    x = mx.array(values)
    expected = turing_layernorm_affine_fp16(
        x,
        mx.ones((512,), dtype=mx.float16),
        mx.zeros((512,), dtype=mx.float16),
        correction,
    )
    actual = turing_layernorm_noaffine_fp16(x, correction)
    mx.eval(expected, actual)

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_turing_decoder_noaffine_dispatch_accepts_authenticated_width256(
    monkeypatch,
):
    import mlx.core as mx

    import trellmlx.decoder_turing_layernorm as decoder_layernorm

    called = {}

    def fake_turing_noaffine(x, correction, eps):
        called["shape"] = x.shape
        called["correction_shape"] = correction.shape
        called["eps"] = eps
        return mx.full(x.shape, 5, dtype=x.dtype)

    monkeypatch.setattr(
        decoder_layernorm,
        "turing_layernorm_noaffine_fp16",
        fake_turing_noaffine,
    )
    decoder_layernorm.configure_decoder_layernorm_backend(
        decoder_layernorm.CUDA_WELFORD_TURING_T4_BACKEND,
        turing_rsqrt_delta_lut=_fixture_correction_lut(),
        turing_rsqrt_lut_artifact_sha256_attested="a" * 64,
    )

    actual = decoder_layernorm.layernorm_noaffine(
        mx.zeros((3, 256), dtype=mx.float16),
    )
    mx.eval(actual)

    assert called == {
        "shape": (3, 256),
        "correction_shape": (1 << 24,),
        "eps": 1e-6,
    }
    np.testing.assert_array_equal(np.asarray(actual), np.full((3, 256), 5))


def test_turing_decoder_noaffine_dispatch_accepts_authenticated_width128(
    monkeypatch,
):
    import mlx.core as mx

    import trellmlx.decoder_turing_layernorm as decoder_layernorm

    called = {}

    def fake_turing_noaffine(x, correction, eps):
        called["shape"] = x.shape
        called["correction_shape"] = correction.shape
        called["eps"] = eps
        return mx.full(x.shape, 5, dtype=x.dtype)

    monkeypatch.setattr(
        decoder_layernorm,
        "turing_layernorm_noaffine_fp16",
        fake_turing_noaffine,
    )
    decoder_layernorm.configure_decoder_layernorm_backend(
        decoder_layernorm.CUDA_WELFORD_TURING_T4_BACKEND,
        turing_rsqrt_delta_lut=_fixture_correction_lut(),
        turing_rsqrt_lut_artifact_sha256_attested="a" * 64,
    )

    actual = decoder_layernorm.layernorm_noaffine(
        mx.zeros((3, 128), dtype=mx.float16),
    )
    mx.eval(actual)

    assert called == {
        "shape": (3, 128),
        "correction_shape": (1 << 24,),
        "eps": 1e-6,
    }
    np.testing.assert_array_equal(np.asarray(actual), np.full((3, 128), 5))


def test_turing_decoder_noaffine_dispatch_accepts_authenticated_width64(
    monkeypatch,
):
    import mlx.core as mx

    import trellmlx.decoder_turing_layernorm as decoder_layernorm

    called = {}

    def fake_turing_noaffine(x, correction, eps):
        called["shape"] = x.shape
        called["correction_shape"] = correction.shape
        called["eps"] = eps
        return mx.full(x.shape, 5, dtype=x.dtype)

    monkeypatch.setattr(
        decoder_layernorm,
        "turing_layernorm_noaffine_fp16",
        fake_turing_noaffine,
    )
    decoder_layernorm.configure_decoder_layernorm_backend(
        decoder_layernorm.CUDA_WELFORD_TURING_T4_BACKEND,
        turing_rsqrt_delta_lut=_fixture_correction_lut(),
        turing_rsqrt_lut_artifact_sha256_attested="a" * 64,
    )

    actual = decoder_layernorm.layernorm_noaffine(
        mx.zeros((3, 64), dtype=mx.float16),
    )
    mx.eval(actual)

    assert called == {
        "shape": (3, 64),
        "correction_shape": (1 << 24,),
        "eps": 1e-6,
    }
    np.testing.assert_array_equal(np.asarray(actual), np.full((3, 64), 5))


def test_shape_terminal_dispatch_accepts_only_proven_float32_contract(
    monkeypatch,
):
    import mlx.core as mx

    import trellmlx.decoder_turing_layernorm as decoder_layernorm

    called = {}

    def fake_terminal(x, correction, eps):
        called["shape"] = x.shape
        called["dtype"] = x.dtype
        called["correction_shape"] = correction.shape
        called["eps"] = eps
        return mx.full(x.shape, 9, dtype=x.dtype)

    monkeypatch.setattr(
        decoder_layernorm,
        "turing_layernorm_noaffine_fp32_width64",
        fake_terminal,
    )
    decoder_layernorm.configure_decoder_layernorm_backend(
        decoder_layernorm.CUDA_WELFORD_TURING_T4_BACKEND,
        turing_rsqrt_delta_lut=_fixture_correction_lut(),
        turing_rsqrt_lut_artifact_sha256_attested="a" * 64,
    )

    actual = decoder_layernorm.layernorm_noaffine_terminal_shape(
        mx.zeros((3, 64), dtype=mx.float32),
        eps=1e-5,
    )
    mx.eval(actual)

    assert called == {
        "shape": (3, 64),
        "dtype": mx.float32,
        "correction_shape": (1 << 24,),
        "eps": 1e-5,
    }
    np.testing.assert_array_equal(np.asarray(actual), np.full((3, 64), 9))
    with pytest.raises(ValueError, match="float32.*width 64"):
        decoder_layernorm.layernorm_noaffine_terminal_shape(
            mx.zeros((3, 128), dtype=mx.float32),
            eps=1e-5,
        )
    with pytest.raises(ValueError, match="eps"):
        decoder_layernorm.layernorm_noaffine_terminal_shape(
            mx.zeros((3, 64), dtype=mx.float32),
            eps=1e-6,
        )


def test_shape_terminal_helper_routes_shape_only(monkeypatch):
    import mlx.core as mx

    import trellmlx.decoder_turing_layernorm as decoder_layernorm
    from trellmlx.models.shape_slat_decoder import _layernorm_noaffine

    calls = []

    def fake_terminal(x, eps):
        calls.append((x.shape, x.dtype, eps))
        return mx.full(x.shape, 4, dtype=x.dtype)

    monkeypatch.setattr(
        decoder_layernorm,
        "layernorm_noaffine_terminal_shape",
        fake_terminal,
        raising=False,
    )
    values = mx.arange(128, dtype=mx.float32).reshape(2, 64)

    shape_actual = _layernorm_noaffine(
        values,
        decoder_role="shape",
    )
    texture_actual = _layernorm_noaffine(
        values,
        decoder_role="texture",
    )
    texture_fp16_actual = _layernorm_noaffine(
        values.astype(mx.float16),
        decoder_role="texture",
    )
    mx.eval(shape_actual, texture_actual, texture_fp16_actual)

    assert calls == [((2, 64), mx.float32, 1e-5)]
    np.testing.assert_array_equal(
        np.asarray(shape_actual),
        np.full((2, 64), 4, dtype=np.float32),
    )
    assert not np.array_equal(
        np.asarray(texture_actual),
        np.asarray(shape_actual),
    )
    assert texture_fp16_actual.dtype == mx.float16


def test_shape_decoder_fp16_caller_keeps_terminal_output_head_float32(
    monkeypatch,
):
    import mlx.core as mx

    import trellmlx.decoder_turing_layernorm as decoder_layernorm
    from trellmlx.models.shape_slat_decoder import SLatDecoder

    class IdentityProjection:
        def __call__(self, values):
            return values

    class CapturingOutputHead:
        def __init__(self):
            self.input_dtype = None

        def __call__(self, values):
            self.input_dtype = values.dtype
            return values[:, :7]

    def fake_terminal(values, eps):
        assert values.dtype == mx.float32
        assert eps == 1e-5
        return values

    decoder = SLatDecoder(
        out_channels=7,
        latent_channels=64,
        model_channels=[64],
        num_blocks=[0],
        pred_subdiv=True,
        use_fp16=True,
    )
    decoder.from_latent = IdentityProjection()
    decoder.output_layer = CapturingOutputHead()
    decoder._forward_levels = (
        lambda feats, coords, guide_subs=None, return_subs=False: (
            feats,
            coords,
        )
    )
    monkeypatch.setattr(
        decoder_layernorm,
        "layernorm_noaffine_terminal_shape",
        fake_terminal,
    )

    output, output_coords = decoder(
        mx.ones((2, 64), dtype=mx.float16),
        mx.array([[0, 0, 0, 0], [0, 0, 0, 1]], dtype=mx.int32),
    )
    mx.eval(output, output_coords)

    assert decoder.output_layer.input_dtype == mx.float32
    assert output.dtype == mx.float32


def test_turing_decoder_affine_dispatch_accepts_authenticated_width256(
    monkeypatch,
):
    import mlx.core as mx

    import trellmlx.decoder_turing_layernorm as decoder_layernorm

    called = {}

    def fake_turing_affine(x, weight, bias, correction, eps):
        called["shape"] = x.shape
        called["weight_shape"] = weight.shape
        called["bias_shape"] = bias.shape
        called["correction_shape"] = correction.shape
        called["eps"] = eps
        return mx.full(x.shape, 6, dtype=x.dtype)

    monkeypatch.setattr(
        decoder_layernorm,
        "turing_layernorm_affine_fp16",
        fake_turing_affine,
    )
    decoder_layernorm.configure_decoder_layernorm_backend(
        decoder_layernorm.CUDA_WELFORD_TURING_T4_BACKEND,
        turing_rsqrt_delta_lut=_fixture_correction_lut(),
        turing_rsqrt_lut_artifact_sha256_attested="a" * 64,
    )

    actual = decoder_layernorm.layernorm_affine(
        mx.zeros((3, 256), dtype=mx.float16),
        mx.ones((256,), dtype=mx.float16),
        mx.zeros((256,), dtype=mx.float16),
    )
    mx.eval(actual)

    assert called == {
        "shape": (3, 256),
        "weight_shape": (256,),
        "bias_shape": (256,),
        "correction_shape": (1 << 24,),
        "eps": 1e-6,
    }
    np.testing.assert_array_equal(np.asarray(actual), np.full((3, 256), 6))


def test_turing_decoder_affine_dispatch_accepts_authenticated_width128(
    monkeypatch,
):
    import mlx.core as mx

    import trellmlx.decoder_turing_layernorm as decoder_layernorm

    called = {}

    def fake_turing_affine(x, weight, bias, correction, eps):
        called["shape"] = x.shape
        called["weight_shape"] = weight.shape
        called["bias_shape"] = bias.shape
        called["correction_shape"] = correction.shape
        called["eps"] = eps
        return mx.full(x.shape, 6, dtype=x.dtype)

    monkeypatch.setattr(
        decoder_layernorm,
        "turing_layernorm_affine_fp16",
        fake_turing_affine,
    )
    decoder_layernorm.configure_decoder_layernorm_backend(
        decoder_layernorm.CUDA_WELFORD_TURING_T4_BACKEND,
        turing_rsqrt_delta_lut=_fixture_correction_lut(),
        turing_rsqrt_lut_artifact_sha256_attested="a" * 64,
    )

    actual = decoder_layernorm.layernorm_affine(
        mx.zeros((3, 128), dtype=mx.float16),
        mx.ones((128,), dtype=mx.float16),
        mx.zeros((128,), dtype=mx.float16),
    )
    mx.eval(actual)

    assert called == {
        "shape": (3, 128),
        "weight_shape": (128,),
        "bias_shape": (128,),
        "correction_shape": (1 << 24,),
        "eps": 1e-6,
    }
    np.testing.assert_array_equal(np.asarray(actual), np.full((3, 128), 6))


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


def test_shape_decoder_exact_layernorm_enrollment_includes_shape_width128():
    from trellmlx.models.shape_slat_decoder import (
        SLatDecoder,
        SparseConvNeXtBlock3d,
        SparseResBlockC2S3d,
    )

    decoder = SLatDecoder(
        out_channels=7,
        model_channels=[1024, 512, 256, 128, 64],
        num_blocks=[2, 2, 2, 2, 0],
        pred_subdiv=True,
        use_fp16=True,
    )
    texture_decoder = SLatDecoder(
        out_channels=6,
        model_channels=[1024, 512, 256, 128, 64],
        num_blocks=[2, 2, 2, 2, 0],
        pred_subdiv=False,
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
    level1_upsample = [
        block
        for block in decoder.blocks[1]
        if isinstance(block, SparseResBlockC2S3d)
    ]
    level2_blocks = [
        block
        for block in decoder.blocks[2]
        if isinstance(block, SparseConvNeXtBlock3d)
    ]
    level2_upsample = [
        block
        for block in decoder.blocks[2]
        if isinstance(block, SparseResBlockC2S3d)
    ]
    level3_blocks = [
        block
        for block in decoder.blocks[3]
        if isinstance(block, SparseConvNeXtBlock3d)
    ]
    texture_level3_blocks = [
        block
        for block in texture_decoder.blocks[3]
        if isinstance(block, SparseConvNeXtBlock3d)
    ]
    level3_upsample = [
        block
        for block in decoder.blocks[3]
        if isinstance(block, SparseResBlockC2S3d)
    ]
    texture_level1_upsample = [
        block
        for block in texture_decoder.blocks[1]
        if isinstance(block, SparseResBlockC2S3d)
    ]
    texture_level2_upsample = [
        block
        for block in texture_decoder.blocks[2]
        if isinstance(block, SparseResBlockC2S3d)
    ]
    texture_level3_upsample = [
        block
        for block in texture_decoder.blocks[3]
        if isinstance(block, SparseResBlockC2S3d)
    ]

    assert len(level0_upsample) == 1
    assert len(level1_upsample) == 1
    assert all(block.norm.decoder_layernorm for block in level0_blocks)
    assert level0_upsample[0].norm1.decoder_layernorm is True
    assert level0_upsample[0].norm2.decoder_layernorm is True
    assert all(block.norm.decoder_layernorm for block in level1_blocks)
    assert level1_upsample[0].norm1.decoder_layernorm is True
    assert level1_upsample[0].norm2.decoder_layernorm is True
    assert all(block.norm.decoder_layernorm is True for block in level2_blocks)
    assert len(level2_upsample) == 1
    assert level2_upsample[0].norm2.decoder_layernorm is True
    assert all(block.norm.decoder_layernorm is True for block in level3_blocks)
    assert len(level3_upsample) == 1
    assert level3_upsample[0].norm1.decoder_layernorm is True
    assert level3_upsample[0].norm2.decoder_layernorm is True
    assert all(
        block.norm.decoder_layernorm is False for block in texture_level3_blocks
    )
    assert len(texture_level1_upsample) == 1
    assert texture_level1_upsample[0].norm2.decoder_layernorm is False
    assert len(texture_level2_upsample) == 1
    assert texture_level2_upsample[0].norm2.decoder_layernorm is False
    assert len(texture_level3_upsample) == 1
    assert texture_level3_upsample[0].norm1.decoder_layernorm is False
    assert texture_level3_upsample[0].norm2.decoder_layernorm is False


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
