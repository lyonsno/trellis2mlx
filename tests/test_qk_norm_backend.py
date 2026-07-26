from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest


FIXTURE = Path(__file__).parent / "fixtures" / "block0_qk_norm_cuda_row.npz"


def _run_fixture(monkeypatch, backend=None):
    from trellmlx.modules.attention import MultiHeadRMSNorm

    if backend is None:
        monkeypatch.delenv("TRELLIS2MLX_QK_NORM_BACKEND", raising=False)
    else:
        monkeypatch.setenv("TRELLIS2MLX_QK_NORM_BACKEND", backend)
    with np.load(FIXTURE, allow_pickle=False) as witness:
        norm = MultiHeadRMSNorm(head_dim=128, num_heads=1)
        norm.gamma = mx.array(witness["gamma"][None, :])
        x = mx.array(witness["x"][None, None, :]).astype(mx.bfloat16)
        out = norm(x)
        mx.eval(out)
        return (
            np.asarray(out.astype(mx.float32))[0, 0],
            witness["expected"],
            witness["ordinary_mlx"],
        )


def test_default_qk_norm_matches_authenticated_source_cuda_row(monkeypatch):
    actual, expected, ordinary_mlx = _run_fixture(monkeypatch)

    assert np.count_nonzero(ordinary_mlx - expected) == 1
    np.testing.assert_array_equal(actual, expected)


def test_cuda_warp32_qk_norm_matches_authenticated_source_cuda_row(monkeypatch):
    actual, expected, _ = _run_fixture(monkeypatch, "source-cuda-warp32")

    np.testing.assert_array_equal(actual, expected)


def test_mlx_sum_qk_norm_preserves_the_previous_route(monkeypatch):
    actual, _, ordinary_mlx = _run_fixture(monkeypatch, "mlx-sum")

    np.testing.assert_array_equal(actual, ordinary_mlx)


def test_qk_norm_rejects_an_unknown_backend(monkeypatch):
    from trellmlx.modules.attention import MultiHeadRMSNorm

    monkeypatch.setenv("TRELLIS2MLX_QK_NORM_BACKEND", "silent-fallback")
    norm = MultiHeadRMSNorm(head_dim=128, num_heads=1)
    x = mx.zeros((1, 1, 128), dtype=mx.bfloat16)

    with pytest.raises(ValueError, match="TRELLIS2MLX_QK_NORM_BACKEND"):
        norm(x)


def test_cuda_warp32_qk_norm_clamps_zero_norm_like_source(monkeypatch):
    from trellmlx.modules.attention import MultiHeadRMSNorm

    monkeypatch.setenv(
        "TRELLIS2MLX_QK_NORM_BACKEND", "source-cuda-warp32"
    )
    norm = MultiHeadRMSNorm(head_dim=128, num_heads=1)
    x = mx.zeros((1, 1, 128), dtype=mx.bfloat16)

    out = norm(x)
    mx.eval(out)
    actual = np.asarray(out.astype(mx.float32))

    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual, np.zeros_like(actual))
