"""Bit-exact Turing HMMA.1688 FP16/FP32 matrix multiplication."""

from __future__ import annotations

import numpy as np


TURING_FDA_FRACTION_BITS = 24
TURING_FDA_STAGE_WIDTH = 8
TURING_FDA_ZERO_EXPONENT = -132

METAL_HEADER = r"""
inline int turing_fda_exponent(float value, int zero_exponent) {
    uint bits = as_type<uint>(value) & 0x7fffffffu;
    if (bits == 0u) {
        return zero_exponent;
    }
    uint biased = bits >> 23;
    if (biased != 0u) {
        return int(biased) - 127;
    }
    uint fraction = bits & 0x7fffffu;
    int top = -1;
    while (fraction != 0u) {
        fraction >>= 1;
        top += 1;
    }
    return top - 149;
}

inline float turing_fda_rz_from_fixed(int fixed, int common_exponent) {
    if (fixed == 0) {
        return 0.0f;
    }
    uint sign = fixed < 0 ? 0x80000000u : 0u;
    uint magnitude = fixed < 0 ? uint(-fixed) : uint(fixed);
    uint scan = magnitude;
    int top = -1;
    while (scan != 0u) {
        scan >>= 1;
        top += 1;
    }
    int exponent = common_exponent - 24 + top;
    if (exponent > 127) {
        return as_type<float>(sign | 0x7f800000u);
    }
    if (exponent >= -126) {
        uint significand = top > 23
            ? magnitude >> uint(top - 23)
            : magnitude << uint(23 - top);
        uint fraction = significand & 0x7fffffu;
        return as_type<float>(
            sign | (uint(exponent + 127) << 23) | fraction);
    }

    int subnormal_shift = common_exponent + 125;
    uint fraction = subnormal_shift >= 0
        ? magnitude << uint(subnormal_shift)
        : magnitude >> uint(-subnormal_shift);
    return as_type<float>(sign | (fraction & 0x7fffffu));
}

inline float turing_fda_stage(
    const device half* a,
    const device half* b,
    uint row,
    uint column,
    uint reduction_offset,
    uint reduction,
    uint columns,
    float accumulator) {
    float products[8];
    int product_exponents[8];
    int common_exponent = turing_fda_exponent(accumulator, -132);

    for (uint inner = 0; inner < 8; inner += 1) {
        uint k = reduction_offset + inner;
        float left = float(a[row * reduction + k]);
        float right = float(b[k * columns + column]);
        float product = left * right;
        products[inner] = product;
        int exponent = product == 0.0f
            ? -132
            : turing_fda_exponent(left, -132)
                + turing_fda_exponent(right, -132);
        product_exponents[inner] = exponent;
        common_exponent = metal::max(common_exponent, exponent);
    }

    int fixed_sum = int(metal::trunc(
        metal::ldexp(accumulator, 24 - common_exponent)));
    for (uint inner = 0; inner < 8; inner += 1) {
        fixed_sum += int(metal::trunc(
            metal::ldexp(products[inner], 24 - common_exponent)));
    }
    return turing_fda_rz_from_fixed(fixed_sum, common_exponent);
}
"""

METAL_SOURCE = r"""
    uint output_index = thread_position_in_grid.x;
    uint rows = uint(shape[0]);
    uint reduction = uint(shape[1]);
    uint columns = uint(shape[2]);
    uint output_count = rows * columns;
    if (output_index >= output_count) {
        return;
    }
    uint row = output_index / columns;
    uint column = output_index - row * columns;
    float accumulator = 0.0f;
    for (uint k = 0; k < reduction; k += 8) {
        accumulator = turing_fda_stage(
            a,
            b,
            row,
            column,
            k,
            reduction,
            columns,
            accumulator);
    }
    output[output_index] = accumulator;
"""


def _normalize_numpy(
    value: np.ndarray,
    *,
    subnormal_exponent: int,
) -> tuple[np.ndarray, np.ndarray]:
    significand, exponent = np.frexp(np.asarray(value, dtype=np.float64))
    significand *= 2.0
    exponent = exponent.astype(np.int32) - 1
    subnormal = exponent < subnormal_exponent
    significand = np.where(
        subnormal,
        np.ldexp(significand, exponent - subnormal_exponent),
        significand,
    )
    exponent = np.where(subnormal, subnormal_exponent, exponent)
    exponent = np.where(significand == 0.0, subnormal_exponent, exponent)
    return significand, exponent


def _turing_fda_stage_reference(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    significand_a, exponent_a = _normalize_numpy(
        a,
        subnormal_exponent=-14,
    )
    significand_b, exponent_b = _normalize_numpy(
        b,
        subnormal_exponent=-14,
    )
    significand_c, exponent_c = _normalize_numpy(
        c,
        subnormal_exponent=-126,
    )
    significands = np.concatenate(
        (
            significand_a * significand_b,
            significand_c[..., None],
        ),
        axis=-1,
    )
    exponents = np.concatenate(
        (
            exponent_a + exponent_b,
            exponent_c[..., None],
        ),
        axis=-1,
    )
    exponents = np.where(
        significands == 0.0,
        TURING_FDA_ZERO_EXPONENT,
        exponents,
    )
    common_exponent = exponents.max(axis=-1)
    aligned = np.trunc(
        np.ldexp(
            significands,
            exponents
            - common_exponent[..., None]
            + TURING_FDA_FRACTION_BITS,
        )
    ) * 2.0 ** -TURING_FDA_FRACTION_BITS
    fused_sum = aligned.sum(axis=-1, dtype=np.float64)

    normalized, exponent = _normalize_numpy(
        np.ldexp(fused_sum, common_exponent),
        subnormal_exponent=-126,
    )
    normalized = (
        np.trunc(normalized * 2.0**23) * 2.0**-23
    )
    return np.ldexp(normalized, exponent).astype(np.float32)


def _validate_numpy_inputs(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.dtype != np.float16 or b.dtype != np.float16:
        raise ValueError("Turing FDA inputs must have dtype float16")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Turing FDA inputs must have compatible matrix shapes")
    if a.shape[1] == 0 or a.shape[1] % TURING_FDA_STAGE_WIDTH:
        raise ValueError("Turing FDA reduction must be a nonzero multiple of 8")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Turing FDA inputs must be finite")
    expected_c_shape = (a.shape[0], b.shape[1])
    if c is None:
        c_array = np.zeros(expected_c_shape, dtype=np.float32)
    else:
        c_array = np.asarray(c)
        if c_array.dtype != np.float32 or c_array.shape != expected_c_shape:
            raise ValueError(
                "Turing FDA accumulator must be float32 with output shape"
            )
        if not np.all(np.isfinite(c_array)):
            raise ValueError("Turing FDA accumulator must be finite")
    return (
        np.ascontiguousarray(a),
        np.ascontiguousarray(b),
        np.array(c_array, dtype=np.float32, order="C", copy=True),
    )


def turing_fda_reference(
    a: np.ndarray,
    b: np.ndarray,
    *,
    c: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate the source-backed Turing FDA law with NumPy."""

    a, b, result = _validate_numpy_inputs(a, b, c)
    rows, reduction = a.shape
    columns = b.shape[1]
    row_chunk = max(1, min(rows, 64))
    for k in range(0, reduction, TURING_FDA_STAGE_WIDTH):
        for row_start in range(0, rows, row_chunk):
            row_stop = min(rows, row_start + row_chunk)
            left = np.broadcast_to(
                a[row_start:row_stop, None, k : k + 8],
                (row_stop - row_start, columns, 8),
            )
            right = np.broadcast_to(
                b[None, k : k + 8, :].transpose(0, 2, 1),
                (row_stop - row_start, columns, 8),
            )
            result[row_start:row_stop] = _turing_fda_stage_reference(
                left,
                right,
                result[row_start:row_stop],
            )
    return result


def turing_fda_matmul(a, b):
    """Evaluate a bit-exact Turing HMMA.1688 chain with a Metal kernel."""

    import mlx.core as mx

    if a.dtype != mx.float16 or b.dtype != mx.float16:
        raise ValueError("Turing FDA inputs must have dtype float16")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Turing FDA inputs must have compatible matrix shapes")
    rows, reduction = a.shape
    columns = b.shape[1]
    if reduction == 0 or reduction % TURING_FDA_STAGE_WIDTH:
        raise ValueError("Turing FDA reduction must be a nonzero multiple of 8")

    kernel = mx.fast.metal_kernel(
        name="turing_fda_f16_f32_matmul",
        input_names=["a", "b", "shape"],
        output_names=["output"],
        header=METAL_HEADER,
        source=METAL_SOURCE,
        ensure_row_contiguous=True,
    )
    shape = mx.array([rows, reduction, columns], dtype=mx.int32)
    output_count = rows * columns
    grid_size = ((output_count + 255) // 256) * 256
    return kernel(
        inputs=[a, b, shape],
        grid=(grid_size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, columns)],
        output_dtypes=[mx.float32],
    )[0]


def turing_fda_linear(a, weight, bias):
    """Apply the source CUDA FP16 linear product and FP32 bias epilogue."""

    import mlx.core as mx

    if bias.dtype != mx.float16:
        raise ValueError("Turing FDA linear bias must have dtype float16")
    if bias.ndim != 1 or weight.ndim != 2 or bias.shape != (weight.shape[1],):
        raise ValueError(
            "Turing FDA linear bias must match the output width"
        )
    product = turing_fda_matmul(a, weight)
    return (
        product + bias.astype(mx.float32)
    ).astype(mx.float16)
