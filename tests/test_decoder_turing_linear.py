from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest


def test_decoder_turing_linear_routes_convnext_mlp(monkeypatch):
    import trellmlx.models.shape_slat_decoder as decoder_module

    monkeypatch.setenv("TRELLIS2MLX_DECODER_LINEAR_BACKEND", "turing_fda")
    monkeypatch.setenv("TRELLIS2MLX_SPARSE_CONV_MATMUL_BACKEND", "native")
    calls = []

    def fake_turing_linear(value, weight, bias):
        calls.append((value.shape, weight.shape, bias.shape))
        return mx.zeros((value.shape[0], weight.shape[1]), dtype=mx.float16)

    monkeypatch.setattr(
        decoder_module,
        "turing_fda_linear",
        fake_turing_linear,
    )
    block = decoder_module.SparseConvNeXtBlock3d(8)
    block.set_dtype(mx.float16)
    feats = mx.ones((1, 8), dtype=mx.float16)
    indices = mx.array(np.array([0], dtype=np.int32))
    output = block(feats, (indices, indices, indices))
    mx.eval(output)

    assert calls == [
        ((1, 8), (8, 32), (32,)),
        ((1, 32), (32, 8), (8,)),
    ]


def test_level0_trace_uses_same_turing_linear_route_as_forward(monkeypatch):
    import trellmlx.models.shape_slat_decoder as decoder_module
    from trellmlx.decoder_level0_trace import capture_mlx_decoder_level0_trace

    monkeypatch.setenv("TRELLIS2MLX_DECODER_LINEAR_BACKEND", "turing_fda")
    monkeypatch.setenv("TRELLIS2MLX_SPARSE_CONV_MATMUL_BACKEND", "native")

    def fake_turing_linear(value, weight, bias):
        return mx.zeros((value.shape[0], weight.shape[1]), dtype=mx.float16)

    monkeypatch.setattr(
        decoder_module,
        "turing_fda_linear",
        fake_turing_linear,
    )
    decoder = decoder_module.SLatDecoder(
        out_channels=7,
        latent_channels=4,
        model_channels=[8, 4],
        num_blocks=[4, 0],
        pred_subdiv=True,
        use_fp16=True,
    )
    feats = mx.arange(12, dtype=mx.float32).reshape(3, 4) / 11
    coords = mx.array(
        [[0, 1, 2, 3], [0, 1, 2, 4], [0, 2, 2, 3]],
        dtype=mx.int32,
    )

    arrays = capture_mlx_decoder_level0_trace(decoder, feats, coords)

    assert arrays["block0_mlp_fc1"].shape == (3, 32)
    assert arrays["block0_mlp_fc2"].shape == (3, 8)


def test_decoder_native_linear_does_not_call_turing_fda(monkeypatch):
    import trellmlx.models.shape_slat_decoder as decoder_module

    monkeypatch.setenv("TRELLIS2MLX_DECODER_LINEAR_BACKEND", "native")

    def forbidden(*_args):
        raise AssertionError("native decoder linear route called Turing FDA")

    monkeypatch.setattr(decoder_module, "turing_fda_linear", forbidden)
    linear = decoder_module.nn.Linear(8, 4)
    output = decoder_module._decoder_linear(
        linear,
        mx.ones((2, 8), dtype=mx.float16),
    )
    mx.eval(output)

    assert output.shape == (2, 4)


def test_decoder_linear_rejects_unknown_backend(monkeypatch):
    import trellmlx.models.shape_slat_decoder as decoder_module

    monkeypatch.setenv(
        "TRELLIS2MLX_DECODER_LINEAR_BACKEND",
        "silent-fallback",
    )
    linear = decoder_module.nn.Linear(8, 4)

    with pytest.raises(ValueError, match="decoder linear backend"):
        decoder_module._decoder_linear(
            linear,
            mx.ones((2, 8), dtype=mx.float16),
        )
