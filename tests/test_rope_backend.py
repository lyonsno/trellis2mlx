import mlx.core as mx
import numpy as np
import pytest
from types import SimpleNamespace


@pytest.fixture(autouse=True)
def _restore_rope_backend():
    from trellmlx.modules.rope import MLX_REAL_BACKEND, configure_rope_backend

    configure_rope_backend(MLX_REAL_BACKEND)
    yield
    configure_rope_backend(MLX_REAL_BACKEND)


def test_rope_backend_rejects_unknown_route():
    from trellmlx.modules.rope import configure_rope_backend

    with pytest.raises(ValueError, match="unsupported RoPE backend"):
        configure_rope_backend("silent-fallback")


def test_turing_rope_backend_requires_lut_and_canonical_digest():
    from trellmlx.modules.rope import (
        CUDA_POLAR_TURING_T4_BACKEND,
        configure_rope_backend,
    )

    with pytest.raises(ValueError, match="requires an explicit phase LUT"):
        configure_rope_backend(CUDA_POLAR_TURING_T4_BACKEND)

    phase_lut = mx.zeros((64, 21, 2), dtype=mx.float32)
    with pytest.raises(ValueError, match="lowercase hexadecimal LUT SHA256"):
        configure_rope_backend(
            CUDA_POLAR_TURING_T4_BACKEND,
            turing_phase_lut=phase_lut,
            turing_phase_lut_sha256="A" * 64,
        )


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((63, 21, 2), mx.float32),
        ((64, 20, 2), mx.float32),
        ((64, 21, 2), mx.float16),
    ],
)
def test_turing_rope_backend_rejects_malformed_lut(shape, dtype):
    from trellmlx.modules.rope import (
        CUDA_POLAR_TURING_T4_BACKEND,
        configure_rope_backend,
    )

    with pytest.raises(ValueError, match=r"float32\[64,21,2\]"):
        configure_rope_backend(
            CUDA_POLAR_TURING_T4_BACKEND,
            turing_phase_lut=mx.zeros(shape, dtype=dtype),
            turing_phase_lut_sha256="a" * 64,
        )


def test_source_complex_arithmetic_reproduces_t4_boundary_rounding():
    from trellmlx.modules.rope import (
        MLX_REAL_BACKEND,
        SOURCE_COMPLEX_BACKEND,
        apply_rope,
        configure_rope_backend,
    )

    x = mx.array([[[0.11669921875, -0.017578125]]]).astype(mx.bfloat16)
    phases = mx.array(
        [[[0.9887404441833496, 0.1496405452489853]]],
        dtype=mx.float32,
    )
    expected = mx.array(
        [[[0.1181640625, 8.249282836914062e-05]]],
        dtype=mx.float32,
    )

    configure_rope_backend(MLX_REAL_BACKEND)
    old = apply_rope(x, phases).astype(mx.float32)
    configure_rope_backend(SOURCE_COMPLEX_BACKEND)
    source = apply_rope(x, phases).astype(mx.float32)
    mx.eval(old, source)

    assert not mx.array_equal(old, expected).item()
    assert mx.array_equal(source, expected).item()


def test_default_sparse_phase_builder_preserves_existing_formula():
    from trellmlx.modules.rope import build_sparse_rope_phases

    coords = mx.array([[0, 1, 2], [31, 17, 5]], dtype=mx.int32)
    phases = build_sparse_rope_phases(coords, head_dim=128)
    freq_dim = 128 // 2 // 3
    frequencies = np.arange(freq_dim, dtype=np.float32) / freq_dim
    frequencies = 1.0 / (10000.0 ** frequencies)
    frequencies_mx = mx.array(frequencies)
    coordinates = coords.astype(mx.float32)
    angles = mx.concatenate(
        [
            coordinates[:, dimension : dimension + 1]
            * frequencies_mx[None, :]
            for dimension in range(3)
        ],
        axis=-1,
    )
    angles = mx.concatenate(
        [angles, mx.zeros((angles.shape[0], 1), dtype=mx.float32)],
        axis=-1,
    )
    expected = mx.stack(
        [mx.cos(angles), mx.sin(angles)],
        axis=-1,
    )
    mx.eval(phases, expected)

    assert mx.array_equal(phases, expected).item()


def test_turing_phase_lut_is_gathered_by_spatial_axis_and_frequency():
    from trellmlx.modules.rope import (
        CUDA_POLAR_TURING_T4_BACKEND,
        build_sparse_rope_phases,
        configure_rope_backend,
    )

    table = np.zeros((64, 21, 2), dtype=np.float32)
    for coordinate in range(64):
        for frequency in range(21):
            table[coordinate, frequency] = (
                coordinate + frequency / 100.0,
                -coordinate - frequency / 100.0,
            )
    configure_rope_backend(
        CUDA_POLAR_TURING_T4_BACKEND,
        turing_phase_lut=mx.array(table),
        turing_phase_lut_sha256="b" * 64,
    )

    phases = build_sparse_rope_phases(
        mx.array([[2, 3, 4]], dtype=mx.int32),
        head_dim=128,
    )
    mx.eval(phases)
    actual = np.array(phases)

    assert actual.shape == (1, 64, 2)
    np.testing.assert_array_equal(actual[0, :21], table[2])
    np.testing.assert_array_equal(actual[0, 21:42], table[3])
    np.testing.assert_array_equal(actual[0, 42:63], table[4])
    np.testing.assert_array_equal(actual[0, 63], np.array([1.0, 0.0]))


def test_turing_phase_lut_rejects_out_of_domain_coordinates():
    from trellmlx.modules.rope import (
        CUDA_POLAR_TURING_T4_BACKEND,
        build_sparse_rope_phases,
        configure_rope_backend,
    )

    configure_rope_backend(
        CUDA_POLAR_TURING_T4_BACKEND,
        turing_phase_lut=mx.zeros((64, 21, 2), dtype=mx.float32),
        turing_phase_lut_sha256="c" * 64,
    )

    with pytest.raises(ValueError, match="integer coordinates in 0..63"):
        build_sparse_rope_phases(
            mx.array([[0, 64, 1]], dtype=mx.int32),
            head_dim=128,
        )


def test_shape_flow_model_consumes_configured_rope_phase_backend():
    from trellmlx.models.slat_flow import SLatFlowModel
    from trellmlx.modules.rope import (
        CUDA_POLAR_TURING_T4_BACKEND,
        configure_rope_backend,
    )

    table = np.zeros((64, 21, 2), dtype=np.float32)
    table[..., 0] = np.arange(64, dtype=np.float32)[:, None]
    table[..., 1] = np.arange(21, dtype=np.float32)[None, :]
    configure_rope_backend(
        CUDA_POLAR_TURING_T4_BACKEND,
        turing_phase_lut=mx.array(table),
        turing_phase_lut_sha256="e" * 64,
    )

    phases = SLatFlowModel._coords_to_rope_phases(
        SimpleNamespace(head_dim=128),
        mx.array([[5, 7, 11]], dtype=mx.int32),
    )
    mx.eval(phases)
    actual = np.array(phases)

    np.testing.assert_array_equal(actual[0, :21], table[5])
    np.testing.assert_array_equal(actual[0, 21:42], table[7])
    np.testing.assert_array_equal(actual[0, 42:63], table[11])
    np.testing.assert_array_equal(actual[0, 63], np.array([1.0, 0.0]))


def test_dense_rope_phases_consume_turing_phase_table():
    from trellmlx.modules.rope import (
        CUDA_POLAR_TURING_T4_BACKEND,
        build_rope_phases,
        configure_rope_backend,
    )

    table = np.zeros((64, 21, 2), dtype=np.float32)
    table[..., 0] = np.arange(64, dtype=np.float32)[:, None]
    table[..., 1] = np.arange(21, dtype=np.float32)[None, :]
    configure_rope_backend(
        CUDA_POLAR_TURING_T4_BACKEND,
        turing_phase_lut=mx.array(table),
        turing_phase_lut_sha256="f" * 64,
    )

    phases = build_rope_phases(resolution=2, head_dim=128)
    mx.eval(phases)
    actual = np.array(phases)

    np.testing.assert_array_equal(actual[0, :21], table[0])
    np.testing.assert_array_equal(actual[-1, :21], table[1])
    np.testing.assert_array_equal(actual[-1, 21:42], table[1])
    np.testing.assert_array_equal(actual[-1, 42:63], table[1])
    np.testing.assert_array_equal(actual[-1, 63], np.array([1.0, 0.0]))


def test_turing_rope_backend_identity_binds_phase_table():
    from trellmlx.modules.rope import (
        CUDA_POLAR_TURING_T4_BACKEND,
        configure_rope_backend,
        rope_backend_identity,
    )

    configure_rope_backend(
        CUDA_POLAR_TURING_T4_BACKEND,
        turing_phase_lut=mx.zeros((64, 21, 2), dtype=mx.float32),
        turing_phase_lut_sha256="d" * 64,
    )

    assert rope_backend_identity() == {
        "backend": CUDA_POLAR_TURING_T4_BACKEND,
        "phase_algorithm": "torch-polar-turing-t4-float32-lut",
        "rotation_algorithm": "mlx-complex64-multiply",
        "experimental": True,
        "cuda_device_anchor": "Tesla T4",
        "cuda_source_tag": "pytorch-v2.10.0",
        "phase_lut_sha256": "d" * 64,
        "phase_lut_shape": [64, 21, 2],
    }
