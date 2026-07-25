import numpy as np
import pytest

import mlx.core as mx


@pytest.fixture(autouse=True)
def _restore_shape_flow_layernorm_backend():
    from trellmlx.shape_flow_layernorm import (
        DEFAULT_BACKEND,
        configure_shape_flow_layernorm_backend,
    )

    configure_shape_flow_layernorm_backend(DEFAULT_BACKEND)
    yield
    configure_shape_flow_layernorm_backend(DEFAULT_BACKEND)


def test_default_shape_flow_layernorm_preserves_existing_two_pass_contract():
    from trellmlx.shape_flow_layernorm import (
        DEFAULT_BACKEND,
        get_shape_flow_layernorm_backend,
        layernorm_noaffine,
    )

    mx.random.seed(35)
    x = mx.random.normal((4, 1536), dtype=mx.float32).astype(mx.bfloat16)
    out = layernorm_noaffine(x)
    xf = x.astype(mx.float32)
    mean = mx.mean(xf, axis=-1, keepdims=True)
    var = mx.mean((xf - mean) * (xf - mean), axis=-1, keepdims=True)
    expected = ((xf - mean) * mx.rsqrt(var + 1e-6)).astype(mx.bfloat16)
    mx.eval(out, expected)

    assert get_shape_flow_layernorm_backend() == DEFAULT_BACKEND
    assert out.dtype == mx.bfloat16
    assert mx.array_equal(out, expected).item()


def test_shape_flow_layernorm_backend_rejects_unknown_route():
    from trellmlx.shape_flow_layernorm import configure_shape_flow_layernorm_backend

    with pytest.raises(ValueError, match="unsupported shape-flow LayerNorm backend"):
        configure_shape_flow_layernorm_backend("silent-fallback")


@pytest.mark.parametrize(
    ("shape", "dtype", "message"),
    [
        ((2, 1536), mx.float32, "requires bfloat16 input"),
        ((2, 512), mx.bfloat16, "requires hidden width 1536"),
    ],
)
def test_cuda_welford_shape_flow_layernorm_fails_outside_authenticated_contract(
    shape, dtype, message
):
    from trellmlx.shape_flow_layernorm import (
        CUDA_WELFORD_METAL_BACKEND,
        configure_shape_flow_layernorm_backend,
        layernorm_noaffine,
    )

    configure_shape_flow_layernorm_backend(CUDA_WELFORD_METAL_BACKEND)
    x = mx.zeros(shape, dtype=dtype)

    with pytest.raises(ValueError, match=message):
        layernorm_noaffine(x)


def test_cuda_welford_shape_flow_layernorm_executes_authenticated_geometry():
    from trellmlx.shape_flow_layernorm import (
        CUDA_WELFORD_METAL_BACKEND,
        configure_shape_flow_layernorm_backend,
        layernorm_noaffine,
    )

    configure_shape_flow_layernorm_backend(CUDA_WELFORD_METAL_BACKEND)
    values = np.linspace(-3.0, 5.0, 2 * 1536, dtype=np.float32).reshape(2, 1536)
    x = mx.array(values).astype(mx.bfloat16)
    out = layernorm_noaffine(x, eps=1e-5)
    mx.eval(out)
    out_np = np.asarray(out.astype(mx.float32))

    assert out.shape == x.shape
    assert out.dtype == mx.bfloat16
    assert np.isfinite(out_np).all()
    np.testing.assert_allclose(
        np.mean(out_np, axis=-1),
        np.zeros((2,), dtype=np.float32),
        rtol=0.0,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        np.mean(out_np * out_np, axis=-1),
        np.ones((2,), dtype=np.float32),
        rtol=0.0,
        atol=4e-3,
    )


def test_cuda_welford_backend_identity_names_residual_instead_of_claiming_cuda_exactness():
    from trellmlx.shape_flow_layernorm import (
        CUDA_WELFORD_METAL_BACKEND,
        shape_flow_layernorm_backend_identity,
    )

    identity = shape_flow_layernorm_backend_identity(CUDA_WELFORD_METAL_BACKEND)

    assert identity["backend"] == CUDA_WELFORD_METAL_BACKEND
    assert identity["cuda_source_tag"] == "pytorch-v2.10.0"
    assert identity["reduction"]["threads"] == 128
    assert identity["reduction"]["warps"] == 4
    assert identity["reduction"]["vector_width"] == 4
    assert identity["cuda_rsqrt_bit_exact"] is False
    assert identity["measured_residual"]["native_partial_rstd_mismatches"] == 903
    assert identity["measured_residual"]["native_partial_rows"] == 4096


def test_slat_flow_uses_shape_specific_layernorm_without_rebinding_sparse_flow():
    import trellmlx.models.slat_flow as slat_flow
    import trellmlx.models.sparse_structure_flow as sparse_flow
    from trellmlx.shape_flow_layernorm import layernorm_noaffine

    assert slat_flow._shape_flow_layernorm_noaffine is layernorm_noaffine
    assert slat_flow._layernorm_noaffine is sparse_flow._layernorm_noaffine


def test_cuda_welford_shape_model_normalizes_bfloat16_and_returns_sampler_dtype(
    monkeypatch,
):
    import trellmlx.models.slat_flow as slat_flow
    from trellmlx.shape_flow_layernorm import (
        CUDA_WELFORD_METAL_BACKEND,
        configure_shape_flow_layernorm_backend,
    )

    configure_shape_flow_layernorm_backend(CUDA_WELFORD_METAL_BACKEND)
    model = slat_flow.SLatFlowModel(
        in_channels=2,
        out_channels=2,
        model_channels=1536,
        num_heads=12,
        num_blocks=0,
        mlp_hidden=4,
        context_channels=4,
        shape_flow_layernorm=True,
    )
    monkeypatch.setattr(slat_flow, "_infer_compute_dtype", lambda _model: mx.bfloat16)
    monkeypatch.setattr(
        slat_flow,
        "_source_shared_modulation",
        lambda *_args, **_kwargs: mx.zeros((1, 6 * 1536), dtype=mx.bfloat16),
    )
    seen_dtypes = []
    original_layernorm = slat_flow._shape_flow_layernorm_noaffine

    def capture_layernorm(x, eps):
        seen_dtypes.append(x.dtype)
        return original_layernorm(x, eps=eps)

    monkeypatch.setattr(slat_flow, "_shape_flow_layernorm_noaffine", capture_layernorm)
    x = mx.zeros((1, 2), dtype=mx.float32)
    out = model(
        x,
        mx.zeros((1,), dtype=mx.float32),
        mx.zeros((1, 1, 4), dtype=mx.float32),
    )
    mx.eval(out)

    assert seen_dtypes == [mx.bfloat16]
    assert out.dtype == mx.float32


def test_default_shape_model_preserves_float32_final_layernorm_order(monkeypatch):
    import trellmlx.models.slat_flow as slat_flow

    model = slat_flow.SLatFlowModel.for_shape(
        in_channels=2,
        out_channels=2,
        model_channels=12,
        num_heads=3,
        num_blocks=0,
        mlp_hidden=16,
        context_channels=4,
    )
    monkeypatch.setattr(
        slat_flow,
        "_source_shared_modulation",
        lambda *_args, **_kwargs: mx.zeros((1, 6 * 12), dtype=mx.float32),
    )
    seen_dtypes = []
    original_layernorm = slat_flow._shape_flow_layernorm_noaffine

    def capture_layernorm(x, eps):
        seen_dtypes.append(x.dtype)
        return original_layernorm(x, eps=eps)

    monkeypatch.setattr(slat_flow, "_shape_flow_layernorm_noaffine", capture_layernorm)
    out = model(
        mx.zeros((1, 2), dtype=mx.float32),
        mx.zeros((1,), dtype=mx.float32),
        mx.zeros((1, 1, 4), dtype=mx.float32),
    )
    mx.eval(out)

    assert seen_dtypes == [mx.float32]
    assert out.dtype == mx.float32


def test_slat_flow_role_constructors_isolate_shape_backend_from_texture():
    from trellmlx.models.slat_flow import SLatFlowModel

    shared = {
        "model_channels": 12,
        "num_heads": 3,
        "num_blocks": 1,
        "mlp_hidden": 16,
        "context_channels": 4,
    }
    shape = SLatFlowModel.for_shape(**shared)
    texture = SLatFlowModel.for_texture(**shared)

    assert shape.shape_flow_layernorm is True
    assert all(block.shape_flow_layernorm is True for block in shape.blocks)
    assert texture.shape_flow_layernorm is False
    assert all(block.shape_flow_layernorm is False for block in texture.blocks)


def test_texture_role_keeps_default_layernorm_under_shape_experiment(monkeypatch):
    import trellmlx.models.slat_flow as slat_flow
    from trellmlx.shape_flow_layernorm import (
        CUDA_WELFORD_METAL_BACKEND,
        configure_shape_flow_layernorm_backend,
    )

    configure_shape_flow_layernorm_backend(CUDA_WELFORD_METAL_BACKEND)
    model = slat_flow.SLatFlowModel.for_texture(
        model_channels=12,
        num_heads=3,
        num_blocks=0,
        mlp_hidden=16,
        context_channels=4,
    )
    monkeypatch.setattr(
        slat_flow,
        "_source_shared_modulation",
        lambda *_args, **_kwargs: mx.zeros((1, 6 * 12), dtype=mx.float32),
    )
    x = mx.zeros((1, 64), dtype=mx.float32)
    out = model(
        x,
        mx.zeros((1,), dtype=mx.float32),
        mx.zeros((1, 1, 4), dtype=mx.float32),
    )
    mx.eval(out)

    assert out.shape == (1, 32)
    assert out.dtype == mx.float32
