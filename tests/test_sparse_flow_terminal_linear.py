from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np


PARTITION_BOUNDS = (0, 308, 616, 924, 1232, 1536)


def _round_to_bfloat16_float32(value: np.ndarray) -> np.ndarray:
    bits = np.asarray(value, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def _fma_float32(left, right, accumulator) -> np.ndarray:
    return np.asarray(
        np.asarray(left, dtype=np.float64)
        * np.asarray(right, dtype=np.float64)
        + np.asarray(accumulator, dtype=np.float64),
        dtype=np.float32,
    )


def _source_partitioned_reference(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    output = np.broadcast_to(bias, (x.shape[0], weight.shape[0])).copy()
    for start, stop in zip(PARTITION_BOUNDS[:-1], PARTITION_BOUNDS[1:]):
        partial = np.zeros_like(output)
        for reduction_index in range(start, stop):
            partial = _fma_float32(
                x[:, reduction_index, None],
                weight[:, reduction_index][None, :],
                partial,
            )
        output = np.asarray(output + partial, dtype=np.float32)
    return output


def test_sparse_terminal_linear_reproduces_authenticated_t4_schedule():
    import mlx.core as mx

    from trellmlx.models.sparse_structure_flow import (
        _source_cuda_t4_sparse_terminal_linear,
    )

    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 1536), dtype=np.float32)
    weight = _round_to_bfloat16_float32(
        rng.standard_normal((8, 1536), dtype=np.float32) * 0.03
    )
    bias = _round_to_bfloat16_float32(
        rng.standard_normal(8, dtype=np.float32) * 0.03
    )
    expected = _source_partitioned_reference(x, weight, bias)

    assert hashlib.sha256(expected.tobytes()).hexdigest() == (
        "1f2bf48f4a02960acb4d5e969f53325c3408009ff351fc42241e7aef6dd1e703"
    )
    linear = SimpleNamespace(weight=mx.array(weight), bias=mx.array(bias))
    actual = _source_cuda_t4_sparse_terminal_linear(mx.array(x), linear)
    mx.eval(actual)

    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint32),
        expected.view(np.uint32),
    )


def test_sparse_terminal_linear_identity_binds_exact_geometry():
    import pytest

    from trellmlx.models.sparse_structure_flow import (
        SOURCE_CUDA_T4_SPARSE_TERMINAL_LINEAR_BACKEND,
        sparse_flow_terminal_linear_backend_identity,
    )

    native = sparse_flow_terminal_linear_backend_identity(
        4096,
        input_width=1536,
        output_width=8,
        has_bias=True,
    )
    exact = sparse_flow_terminal_linear_backend_identity(
        4096,
        input_width=1536,
        output_width=8,
        has_bias=True,
        backend=SOURCE_CUDA_T4_SPARSE_TERMINAL_LINEAR_BACKEND,
    )

    assert native["backend"] == "mlx-native-linear"
    assert exact["backend"] == "source-cuda-t4-volta-sgemm-32x128-tn-metal"
    assert exact["authenticated_contract"]["partition_bounds"] == list(
        PARTITION_BOUNDS
    )
    assert exact["authenticated_contract"]["rows"] == 4096
    assert exact["authenticated_contract"]["output_width"] == 8
    with pytest.raises(ValueError, match="authenticated geometry"):
        sparse_flow_terminal_linear_backend_identity(
            4095,
            input_width=1536,
            output_width=8,
            has_bias=True,
            backend=SOURCE_CUDA_T4_SPARSE_TERMINAL_LINEAR_BACKEND,
        )


def test_sparse_terminal_linear_dispatch_requires_explicit_source_backend(monkeypatch):
    import trellmlx.models.sparse_structure_flow as sparse_flow

    exact = SimpleNamespace(ndim=2, shape=(4096, 1536))
    class LinearStub:
        weight = SimpleNamespace(ndim=2, shape=(8, 1536))
        bias = SimpleNamespace(shape=(8,))

        def __call__(self, value):
            return value

    linear = LinearStub()
    sentinel = object()
    monkeypatch.setattr(
        sparse_flow,
        "_source_cuda_t4_sparse_terminal_linear",
        lambda x, selected_linear: sentinel,
    )

    assert sparse_flow._sparse_terminal_linear(
        exact,
        linear,
        backend="mlx-native-linear",
    ) is exact
    assert sparse_flow._sparse_terminal_linear(
        exact,
        linear,
        backend="source-cuda-t4-volta-sgemm-32x128-tn-metal",
    ) is sentinel


def test_stage_capture_binds_effective_sparse_terminal_linear(tmp_path):
    from scripts.run_mlx_stage_capture import (
        _bind_effective_sparse_flow_terminal_linear_identity,
    )
    from trellmlx.models.sparse_structure_flow import (
        sparse_flow_terminal_linear_backend_identity,
    )

    identity = sparse_flow_terminal_linear_backend_identity(
        4096,
        input_width=1536,
        output_width=8,
        has_bias=True,
        backend="source-cuda-t4-volta-sgemm-32x128-tn-metal",
    )
    route_identity = {
        "route": {"sparse_flow_terminal_linear_identity": identity}
    }
    checkpoint = tmp_path / "sparse_flow_steps.npz"
    np.savez(
        checkpoint,
        sparse_flow_terminal_linear_json=np.asarray(
            __import__("json").dumps(identity, sort_keys=True)
        ),
    )

    effective = _bind_effective_sparse_flow_terminal_linear_identity(
        route_identity,
        checkpoint,
    )

    assert effective == identity
    assert (
        route_identity["route"][
            "sparse_flow_terminal_linear_identity_effective"
        ]
        == identity
    )


def test_stage_capture_rejects_substituted_sparse_terminal_linear(tmp_path):
    from scripts.run_mlx_stage_capture import (
        _bind_effective_sparse_flow_terminal_linear_identity,
    )
    from trellmlx.models.sparse_structure_flow import (
        sparse_flow_terminal_linear_backend_identity,
    )

    identity = sparse_flow_terminal_linear_backend_identity(
        4096,
        input_width=1536,
        output_width=8,
        has_bias=True,
        backend="source-cuda-t4-volta-sgemm-32x128-tn-metal",
    )
    route_identity = {
        "route": {"sparse_flow_terminal_linear_identity": identity}
    }
    checkpoint = tmp_path / "sparse_flow_steps.npz"
    np.savez(
        checkpoint,
        sparse_flow_terminal_linear_json=np.asarray(
            '{"backend":"mlx-native-linear"}'
        ),
    )

    import pytest

    with pytest.raises(ValueError, match="terminal linear identity"):
        _bind_effective_sparse_flow_terminal_linear_identity(
            route_identity,
            checkpoint,
        )
