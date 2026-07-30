from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest


def _singleton_convolution(monkeypatch, backend: str):
    from trellmlx.modules import sparse_conv

    monkeypatch.setenv("TRELLIS2MLX_SPARSE_CONV_MATMUL_BACKEND", backend)
    convolution = sparse_conv.SparseConv3d(8, 3, kernel_size=1)
    convolution.weight = mx.ones((3, 1, 1, 1, 8), dtype=mx.float16)
    convolution.bias = mx.zeros((3,), dtype=mx.float16)
    feats = mx.ones((1, 8), dtype=mx.float16)
    indices = mx.array(np.array([0], dtype=np.int32))
    return sparse_conv, convolution, feats, (indices, indices, indices)


def test_sparse_conv_turing_backend_routes_each_kernel_position(
    monkeypatch,
):
    sparse_conv, convolution, feats, neighbor_map = _singleton_convolution(
        monkeypatch,
        "turing_fda",
    )
    calls = []

    def fake_turing_fda(left, right):
        calls.append((left.shape, right.shape))
        return mx.full((left.shape[0], right.shape[1]), 4.0, dtype=mx.float32)

    monkeypatch.setattr(sparse_conv, "turing_fda_matmul", fake_turing_fda)
    output = convolution(feats, neighbor_map)
    mx.eval(output)

    assert calls == [((1, 8), (8, 3))]
    np.testing.assert_array_equal(
        np.asarray(output),
        np.full((1, 3), 4.0, dtype=np.float16),
    )


def test_sparse_conv_native_backend_does_not_call_turing_fda(monkeypatch):
    sparse_conv, convolution, feats, neighbor_map = _singleton_convolution(
        monkeypatch,
        "native",
    )

    def forbidden(*_args):
        raise AssertionError("native route must not call Turing FDA")

    monkeypatch.setattr(sparse_conv, "turing_fda_matmul", forbidden)
    output = convolution(feats, neighbor_map)
    mx.eval(output)

    np.testing.assert_array_equal(
        np.asarray(output),
        np.full((1, 3), 8.0, dtype=np.float16),
    )


def test_sparse_conv_rejects_unknown_matmul_backend(monkeypatch):
    _, convolution, feats, neighbor_map = _singleton_convolution(
        monkeypatch,
        "silent-fallback",
    )

    with pytest.raises(ValueError, match="sparse-convolution matmul backend"):
        convolution(feats, neighbor_map)
