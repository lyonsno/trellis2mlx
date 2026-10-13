import mlx.core as mx
import numpy as np
import pytest

from trellmlx.modules import attention


def test_fast_attention_query_chunk_preserves_all_queries_and_mask(monkeypatch):
    q = mx.array(np.arange(1 * 2 * 11 * 16, dtype=np.float32).reshape(1, 2, 11, 16) / 100)
    k = q + 0.1
    v = q - 0.2
    mask_np = np.zeros((1, 1, 11, 11), dtype=np.float32)
    mask_np[:, :, :, 0] = -1000
    mask = mx.array(mask_np)
    original = mx.fast.scaled_dot_product_attention
    expected = original(q, k, v, scale=0.25, mask=mask)
    mx.eval(expected)
    dispatches = []

    def bounded_dispatch(query, keys, values, *, scale, mask=None):
        dispatches.append((query.shape[2], None if mask is None else mask.shape[2]))
        if query.shape[2] > 4:
            raise AssertionError("oversize fast-attention query dispatch")
        return original(query, keys, values, scale=scale, mask=mask)

    monkeypatch.setenv("TRELLIS2MLX_FAST_ATTENTION_QUERY_CHUNK_SIZE", "4")
    monkeypatch.setattr(attention.mx.fast, "scaled_dot_product_attention", bounded_dispatch)
    actual = attention.scaled_dot_product_attention_for_backend(
        q, k, v, mask,
        backend="fast",
        softmax_backend="mlx-softmax",
        value_backend="mlx-matmul",
    )
    mx.eval(actual)

    assert dispatches == [(4, 4), (4, 4), (3, 3)]
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)


def test_fast_attention_query_chunk_rejects_invalid_setting(monkeypatch):
    q = mx.zeros((1, 1, 3, 8), dtype=mx.float32)
    monkeypatch.setenv("TRELLIS2MLX_FAST_ATTENTION_QUERY_CHUNK_SIZE", "garbage")
    with pytest.raises(ValueError, match="FAST_ATTENTION_QUERY_CHUNK_SIZE"):
        attention.scaled_dot_product_attention_for_backend(
            q, q, q,
            backend="fast",
            softmax_backend="mlx-softmax",
            value_backend="mlx-matmul",
        )


def test_fast_attention_query_chunk_preserves_broadcast_mask(monkeypatch):
    q = mx.ones((1, 1, 9, 16), dtype=mx.float32)
    mask = mx.array([[[[-1000] + [0] * 8]]], dtype=mx.float32)
    original = mx.fast.scaled_dot_product_attention
    expected = original(q, q, q, scale=0.25, mask=mask)
    mx.eval(expected)
    masks = []

    def bounded_dispatch(query, keys, values, *, scale, mask=None):
        masks.append(mask.shape if mask is not None else None)
        if query.shape[2] > 4:
            raise AssertionError("oversize fast-attention query dispatch")
        return original(query, keys, values, scale=scale, mask=mask)

    monkeypatch.setenv("TRELLIS2MLX_FAST_ATTENTION_QUERY_CHUNK_SIZE", "4")
    monkeypatch.setattr(attention.mx.fast, "scaled_dot_product_attention", bounded_dispatch)
    actual = attention.scaled_dot_product_attention_for_backend(
        q, q, q, mask,
        backend="fast",
        softmax_backend="mlx-softmax",
        value_backend="mlx-matmul",
    )
    mx.eval(actual)

    assert masks == [mask.shape] * 3
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)


def test_shape_flow_route_records_effective_query_chunk(monkeypatch):
    from generate import _shape_flow_attention_route_from_env

    monkeypatch.setenv("TRELLIS2MLX_ATTENTION_BACKEND", "fast")
    monkeypatch.setenv("TRELLIS2MLX_FAST_ATTENTION_QUERY_CHUNK_SIZE", "2048")
    route = _shape_flow_attention_route_from_env()

    assert route["shape_flow_fast_attention_query_chunk_size_effective"] == 2048
