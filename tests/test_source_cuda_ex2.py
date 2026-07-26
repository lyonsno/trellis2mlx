import hashlib

import numpy as np

from trellmlx.source_cuda_ex2 import (
    SOURCE_CUDA_EX2_METAL_HEADER,
    SOURCE_CUDA_EX2_PRIMARY_DIGEST,
    source_cuda_ex2_primary_bits,
    source_cuda_ex2_truncated_square,
)


def test_source_cuda_ex2_specialized_squarer_discards_low_partial_products():
    residual = np.asarray([0, 1, 131071], dtype=np.uint32)

    actual = source_cuda_ex2_truncated_square(residual)
    exact_truncation = (
        residual.astype(np.uint64) * residual.astype(np.uint64)
    ) >> np.uint64(19)

    np.testing.assert_array_equal(
        actual,
        np.asarray([0, 0, 32759], dtype=np.uint32),
    )
    assert int(exact_truncation[-1]) == 32767


def test_source_cuda_ex2_reconstructs_complete_authenticated_primary_census():
    digest = hashlib.sha256()
    chunk_size = 1 << 17

    for start in range(0, 1 << 23, chunk_size):
        coordinates = np.arange(
            start,
            start + chunk_size,
            dtype=np.uint32,
        )
        output_bits = source_cuda_ex2_primary_bits(coordinates)
        assert output_bits.dtype == np.uint32
        assert output_bits.shape == coordinates.shape
        digest.update(output_bits.tobytes())

    assert SOURCE_CUDA_EX2_PRIMARY_DIGEST == (
        "b2a42c4a626469c986e33e43f16f41b"
        "de9d84de94347cd9ceaa1bd68d36bcdf0"
    )
    assert digest.hexdigest() == SOURCE_CUDA_EX2_PRIMARY_DIGEST


def test_source_cuda_ex2_rejects_coordinates_outside_q23_primary_interval():
    for coordinates in (
        np.asarray([-1], dtype=np.int64),
        np.asarray([1 << 23], dtype=np.uint32),
        np.asarray([0.0], dtype=np.float32),
    ):
        try:
            source_cuda_ex2_primary_bits(coordinates)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid primary coordinate was accepted")


def test_source_cuda_expf_guards_saturated_underflow_before_primary_lookup():
    guard = "if (value <= -1.0f) {"
    primary_lookup = "return source_cuda_ex2_primary_coordinate(coordinate) * 0.5f;"

    assert guard in SOURCE_CUDA_EX2_METAL_HEADER
    assert SOURCE_CUDA_EX2_METAL_HEADER.index(guard) < (
        SOURCE_CUDA_EX2_METAL_HEADER.index(primary_lookup)
    )
