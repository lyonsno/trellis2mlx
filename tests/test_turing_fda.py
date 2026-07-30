from __future__ import annotations

from pathlib import Path

import numpy as np


FIXTURES = Path(__file__).with_name("fixtures")


OPERAND_A_BITS = np.array(
    [
        0xBD63,
        0xC156,
        0xC8CE,
        0x41F7,
        0xC35D,
        0x455A,
        0x4220,
        0x427B,
        0x4637,
        0x4085,
        0xC02D,
        0x43E0,
        0xC018,
        0xBD99,
        0xC371,
        0xC4C6,
    ],
    dtype=np.uint16,
)
OPERAND_B_BITS = np.array(
    [
        0x3001,
        0xB654,
        0xB7B2,
        0xADB9,
        0xB5CB,
        0x26CF,
        0x2D85,
        0xB440,
        0xACAF,
        0x3276,
        0x2D58,
        0xB4B3,
        0x35AC,
        0x33B6,
        0x2C2B,
        0x3826,
    ],
    dtype=np.uint16,
)


def _bits(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).view(np.uint32)


def test_reference_reproduces_authenticated_hostile_hmma_stages():
    from trellmlx.turing_fda import turing_fda_reference

    a = OPERAND_A_BITS.view(np.float16)[None, :]
    b = OPERAND_B_BITS.view(np.float16)[:, None]
    stage0 = turing_fda_reference(a[:, :8], b[:8])
    full = turing_fda_reference(a[:, 8:], b[8:], c=stage0)

    assert int(_bits(stage0)[0, 0]) == 0x40C3DACF
    assert int(_bits(full)[0, 0]) == 0x3F815F2E


def test_reference_requires_float16_inputs_and_k_multiple_of_eight():
    from trellmlx.turing_fda import turing_fda_reference

    with np.testing.assert_raises_regex(ValueError, "float16"):
        turing_fda_reference(
            np.ones((1, 8), dtype=np.float32),
            np.ones((8, 1), dtype=np.float16),
        )
    with np.testing.assert_raises_regex(ValueError, "multiple of 8"):
        turing_fda_reference(
            np.ones((1, 7), dtype=np.float16),
            np.ones((7, 1), dtype=np.float16),
        )


def test_metal_kernel_reproduces_reference_bit_exactly():
    import mlx.core as mx

    from trellmlx.turing_fda import turing_fda_matmul, turing_fda_reference

    rng = np.random.default_rng(20260730)
    a = (rng.standard_normal((5, 16)) * 3).astype(np.float16)
    b = (rng.standard_normal((16, 7)) * 0.25).astype(np.float16)
    expected = turing_fda_reference(a, b)

    actual = turing_fda_matmul(mx.array(a), mx.array(b))
    mx.eval(actual)

    np.testing.assert_array_equal(_bits(np.asarray(actual)), _bits(expected))


def test_metal_kernel_is_distinct_from_native_matmul_on_hostile_cell():
    import mlx.core as mx

    from trellmlx.turing_fda import turing_fda_matmul

    a = OPERAND_A_BITS.view(np.float16)[None, :]
    b = OPERAND_B_BITS.view(np.float16)[:, None]
    exact = turing_fda_matmul(mx.array(a), mx.array(b))
    native = mx.matmul(mx.array(a), mx.array(b)).astype(mx.float32)
    mx.eval(exact, native)

    assert int(_bits(np.asarray(exact))[0, 0]) == 0x3F815F2E
    assert int(_bits(np.asarray(native))[0, 0]) != 0x3F815F2E


def test_turing_fda_linear_matches_source_cuda_fc1_rows():
    import mlx.core as mx

    from trellmlx.turing_fda import turing_fda_linear

    with np.load(
        FIXTURES / "source_cuda_shape_decoder_block0_fc1_rows.npz",
        allow_pickle=False,
    ) as fixture:
        layernorm = np.asarray(fixture["block0_norm"])
        weight = np.asarray(fixture["weight"])
        bias = np.asarray(fixture["bias"])
        expected = np.asarray(fixture["expected_fc1"])

    actual = turing_fda_linear(
        mx.array(layernorm),
        mx.array(weight),
        mx.array(bias),
    )
    mx.eval(actual)

    assert actual.dtype == mx.float16
    np.testing.assert_array_equal(np.asarray(actual), expected)
