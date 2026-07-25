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

    assert slat_flow._layernorm_noaffine is layernorm_noaffine
    assert sparse_flow._layernorm_noaffine is not layernorm_noaffine
