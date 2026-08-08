from pathlib import Path
import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _restore_rope_backends():
    from trellmlx.modules.rope import MLX_REAL_BACKEND, configure_rope_backend
    from trellmlx.sparse_flow_rope import (
        DEFAULT_BACKEND,
        configure_sparse_flow_rope_backend,
    )

    configure_rope_backend(MLX_REAL_BACKEND)
    configure_sparse_flow_rope_backend(DEFAULT_BACKEND)
    yield
    configure_rope_backend(MLX_REAL_BACKEND)
    configure_sparse_flow_rope_backend(DEFAULT_BACKEND)


def test_source_cpu_sparse_rope_requires_bound_phase_lut():
    from trellmlx.sparse_flow_rope import (
        SOURCE_CPU_POLAR_TORCH_2_10_BACKEND,
        configure_sparse_flow_rope_backend,
    )

    with pytest.raises(ValueError, match="requires an explicit phase LUT"):
        configure_sparse_flow_rope_backend(
            SOURCE_CPU_POLAR_TORCH_2_10_BACKEND
        )


def test_sparse_flow_phase_route_is_independent_from_shape_flow_route():
    from trellmlx.modules.rope import (
        CUDA_POLAR_TURING_T4_BACKEND,
        configure_rope_backend,
    )
    from trellmlx.sparse_flow_rope import (
        SOURCE_CPU_POLAR_TORCH_2_10_BACKEND,
        build_sparse_flow_rope_phases,
        configure_sparse_flow_rope_backend,
    )

    shape_table = np.zeros((64, 21, 2), dtype=np.float32)
    sparse_table = np.zeros((64, 21, 2), dtype=np.float32)
    shape_table[..., 0] = -1.0
    sparse_table[..., 0] = np.arange(64, dtype=np.float32)[:, None]
    sparse_table[..., 1] = np.arange(21, dtype=np.float32)[None, :]
    configure_rope_backend(
        CUDA_POLAR_TURING_T4_BACKEND,
        turing_phase_lut=mx.array(shape_table),
        turing_phase_lut_sha256="a" * 64,
    )
    configure_sparse_flow_rope_backend(
        SOURCE_CPU_POLAR_TORCH_2_10_BACKEND,
        phase_lut=mx.array(sparse_table),
        phase_lut_artifact_sha256_attested="b" * 64,
    )

    phases = build_sparse_flow_rope_phases(2, 128)
    mx.eval(phases)
    actual = np.array(phases)

    np.testing.assert_array_equal(actual[0, :21], sparse_table[0])
    np.testing.assert_array_equal(actual[-1, :21], sparse_table[1])
    np.testing.assert_array_equal(actual[-1, 21:42], sparse_table[1])
    np.testing.assert_array_equal(actual[-1, 42:63], sparse_table[1])
    np.testing.assert_array_equal(actual[-1, 63], np.array([1.0, 0.0]))


def test_source_cpu_sparse_rope_uses_complex_rotation_independent_of_global():
    from trellmlx.modules.rope import MLX_REAL_BACKEND, configure_rope_backend
    from trellmlx.sparse_flow_rope import (
        SOURCE_CPU_POLAR_TORCH_2_10_BACKEND,
        apply_sparse_flow_rope,
        configure_sparse_flow_rope_backend,
    )

    table = np.zeros((64, 21, 2), dtype=np.float32)
    configure_rope_backend(MLX_REAL_BACKEND)
    configure_sparse_flow_rope_backend(
        SOURCE_CPU_POLAR_TORCH_2_10_BACKEND,
        phase_lut=mx.array(table),
        phase_lut_artifact_sha256_attested="c" * 64,
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

    actual = apply_sparse_flow_rope(x, phases).astype(mx.float32)
    mx.eval(actual)

    assert mx.array_equal(actual, expected).item()


def test_generate_exposes_distinct_sparse_flow_rope_selector():
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parents[1] / "generate.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--sparse-flow-rope-backend" in result.stdout
    assert "--sparse-flow-rope-phase-lut" in result.stdout

