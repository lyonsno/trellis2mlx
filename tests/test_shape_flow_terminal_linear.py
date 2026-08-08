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


def test_source_cuda_terminal_linear_reproduces_authenticated_partition_schedule():
    import mlx.core as mx

    from trellmlx.models.slat_flow import _source_cuda_t4_terminal_linear

    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 1536), dtype=np.float32)
    weight = _round_to_bfloat16_float32(
        rng.standard_normal((32, 1536), dtype=np.float32) * 0.03
    )
    bias = _round_to_bfloat16_float32(
        rng.standard_normal(32, dtype=np.float32) * 0.03
    )
    expected = _source_partitioned_reference(x, weight, bias)

    assert hashlib.sha256(expected.tobytes()).hexdigest() == (
        "892ec0f95024281a64a54fb94713568e223a789b5795194d066a8d978a249ed6"
    )
    np.testing.assert_array_equal(
        expected.view(np.uint32).ravel()[:8],
        np.array(
            [
                0xBF2B937C,
                0xBF3CE6CC,
                0xBFBC9B4F,
                0x3E93A24B,
                0x3EF2EC74,
                0xBFEB6122,
                0x402E65D4,
                0xBD824430,
            ],
            dtype=np.uint32,
        ),
    )

    linear = SimpleNamespace(weight=mx.array(weight), bias=mx.array(bias))
    actual = _source_cuda_t4_terminal_linear(mx.array(x), linear)
    mx.eval(actual)

    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint32),
        expected.view(np.uint32),
    )


def test_terminal_linear_backend_identity_names_geometry_and_source_evidence():
    from trellmlx.models.slat_flow import (
        shape_flow_terminal_linear_backend_identity,
    )

    exact = shape_flow_terminal_linear_backend_identity(
        6038,
        input_width=1536,
        output_width=32,
        has_bias=True,
        source_cuda_terminal=True,
    )
    exact_cuda_support = shape_flow_terminal_linear_backend_identity(
        6022,
        input_width=1536,
        output_width=32,
        has_bias=True,
        source_cuda_terminal=True,
    )
    fallback = shape_flow_terminal_linear_backend_identity(
        8,
        input_width=1536,
        output_width=32,
        has_bias=True,
        source_cuda_terminal=True,
    )
    wrong_output_width = shape_flow_terminal_linear_backend_identity(
        6038,
        input_width=1536,
        output_width=31,
        has_bias=True,
        source_cuda_terminal=True,
    )
    native = shape_flow_terminal_linear_backend_identity(
        6038,
        input_width=1536,
        output_width=32,
        has_bias=True,
        source_cuda_terminal=False,
    )

    assert exact["backend"] == "source-cuda-t4-volta-sgemm-32x128-tn-metal"
    assert exact_cuda_support["backend"] == (
        "source-cuda-t4-volta-sgemm-32x128-tn-metal"
    )
    assert exact["cuda_source_kernel"] == "volta_sgemm_32x128_tn"
    assert exact["authenticated_contract"]["rows"] == [6022, 6038]
    assert exact["authenticated_contract"]["partition_bounds"] == [
        0,
        308,
        616,
        924,
        1232,
        1536,
    ]
    assert exact["authenticated_contract"]["cuda_prefix_ladder_sha256"] == (
        "b21bad4d52e8202efdeec5a87af4fa9b52edaa7d513bd57fddf393a6f80dd6cc"
    )
    assert exact["authenticated_contract"]["source_recurrence_sha256"] == [
        "5dd57e90fad742e37a345d2e19bf484298577cd5d84336371c8793f587ca947f",
        "ebde6bc1f271813801e44a312da8077d7c46cf5092f7dfee8b0100e48e3d874c",
    ]
    assert fallback["backend"] == "numpy-fp32-blas"
    assert fallback["excluded_row_geometry"] == [6022, 6038]
    assert wrong_output_width["backend"] == "numpy-fp32-blas"
    assert native["backend"] == "mlx-native-linear"


def test_terminal_linear_dispatch_selects_exact_geometry_only(monkeypatch):
    import trellmlx.models.slat_flow as slat_flow

    exact = SimpleNamespace(ndim=2, shape=(6038, 1536))
    exact_cuda_support = SimpleNamespace(ndim=2, shape=(6022, 1536))
    fallback = SimpleNamespace(ndim=2, shape=(8, 1536))
    linear = SimpleNamespace(
        weight=SimpleNamespace(ndim=2, shape=(32, 1536)),
        bias=SimpleNamespace(shape=(32,)),
    )
    sentinel = object()
    monkeypatch.setattr(
        slat_flow,
        "_source_cuda_t4_terminal_linear",
        lambda x, selected_linear: (
            sentinel
            if (x is exact or x is exact_cuda_support) and selected_linear is linear
            else None
        ),
    )

    assert slat_flow._source_cuda_terminal_linear(exact, linear) is sentinel
    assert (
        slat_flow._source_cuda_terminal_linear(exact_cuda_support, linear)
        is sentinel
    )
    assert slat_flow.shape_flow_terminal_linear_backend_identity(
        6038,
        input_width=1536,
        output_width=32,
        has_bias=True,
        source_cuda_terminal=True,
    )[
        "backend"
    ] == "source-cuda-t4-volta-sgemm-32x128-tn-metal"
    assert slat_flow.shape_flow_terminal_linear_backend_identity(
        8,
        input_width=1536,
        output_width=32,
        has_bias=True,
        source_cuda_terminal=True,
    )[
        "backend"
    ] == "numpy-fp32-blas"
    assert fallback.shape == (8, 1536)
