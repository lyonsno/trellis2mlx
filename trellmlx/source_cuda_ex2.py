"""Bit-exact Turing MUFU.EX2 primary-interval reconstruction."""

from __future__ import annotations

import numpy as np


SOURCE_CUDA_EX2_PRIMARY_DIGEST = (
    "b2a42c4a626469c986e33e43f16f41b"
    "de9d84de94347cd9ceaa1bd68d36bcdf0"
)

SOURCE_CUDA_EX2_C0 = np.asarray(
    [
        33554435, 33919818, 34289182, 34662565, 35040018, 35421577,
        35807292, 36197209, 36591372, 36989825, 37392616, 37799797,
        38211409, 38627504, 39048131, 39473337, 39903173, 40337690,
        40776937, 41220970, 41669836, 42123592, 42582286, 43045977,
        43514717, 43988561, 44467565, 44951786, 45441278, 45936101,
        46436314, 46941971, 47453135, 47969867, 48492225, 49020269,
        49554065, 50093674, 50639159, 51190582, 51748011, 52311510,
        52881146, 53456983, 54039091, 54627538, 55222393, 55823725,
        56431607, 57046106, 57667297, 58295254, 58930048, 59571754,
        60220447, 60876205, 61539104, 62209220, 62886634, 63571424,
        64263672, 64963458, 65670863, 66385972,
    ],
    dtype=np.uint64,
)

SOURCE_CUDA_EX2_C1 = np.asarray(
    [
        22713, 22960, 23210, 23463, 23718, 23977, 24238, 24502,
        24768, 25038, 25311, 25586, 25865, 26147, 26431, 26719,
        27010, 27304, 27602, 27902, 28206, 28513, 28824, 29138,
        29455, 29776, 30100, 30428, 30759, 31094, 31432, 31775,
        32121, 32471, 32824, 33182, 33543, 33908, 34277, 34651,
        35028, 35409, 35795, 36185, 36579, 36977, 37380, 37787,
        38198, 38614, 39035, 39460, 39889, 40324, 40763, 41207,
        41655, 42109, 42568, 43031, 43500, 43973, 44452, 44936,
    ],
    dtype=np.uint64,
)

SOURCE_CUDA_EX2_C2 = np.asarray(
    [
        494, 501, 506, 511, 518, 521, 527, 532,
        541, 546, 551, 559, 564, 568, 577, 583,
        589, 596, 600, 609, 615, 622, 627, 633,
        641, 647, 655, 661, 670, 677, 686, 691,
        699, 705, 715, 721, 730, 739, 748, 753,
        763, 773, 779, 787, 796, 806, 813, 822,
        833, 842, 849, 858, 870, 877, 887, 896,
        909, 917, 925, 938, 946, 959, 969, 980,
    ],
    dtype=np.uint64,
)

SOURCE_CUDA_EX2_BIAS = np.uint64(12228)


def _metal_uint_table(name: str, values: np.ndarray) -> str:
    entries = ", ".join(str(int(value)) for value in values)
    return f"constant uint {name}[64] = {{{entries}}};"


SOURCE_CUDA_EX2_METAL_HEADER = "\n".join(
    [
        _metal_uint_table("source_cuda_ex2_c0", SOURCE_CUDA_EX2_C0),
        _metal_uint_table("source_cuda_ex2_c1", SOURCE_CUDA_EX2_C1),
        _metal_uint_table("source_cuda_ex2_c2", SOURCE_CUDA_EX2_C2),
        r"""
inline uint source_cuda_ex2_truncated_square(uint residual) {
    ulong exact =
        static_cast<ulong>(residual) * static_cast<ulong>(residual);
    uint low = residual & 0x3ffu;
    uint high = residual >> 10;
    ulong omitted =
        static_cast<ulong>(low) * static_cast<ulong>(low);
    for (uint bit = 0; bit < 7; ++bit) {
        uint high_bit = (high >> bit) & 1u;
        uint low_prefix = low & ((1u << (8u - bit)) - 1u);
        omitted += static_cast<ulong>(high_bit)
            * static_cast<ulong>(low_prefix << (bit + 11u));
    }
    return static_cast<uint>(
        (exact >> 19) - (omitted >> 19));
}

inline float source_cuda_ex2_primary_coordinate(uint coordinate) {
    uint segment = coordinate >> 17;
    uint residual = coordinate & 0x1ffffu;
    uint square = source_cuda_ex2_truncated_square(residual);
    ulong accumulator =
        (static_cast<ulong>(source_cuda_ex2_c0[segment]) << 14)
        + 12228ul;
    accumulator += 2ul
        * static_cast<ulong>(source_cuda_ex2_c1[segment])
        * static_cast<ulong>(residual);
    accumulator += 2ul
        * static_cast<ulong>(source_cuda_ex2_c2[segment])
        * static_cast<ulong>(square);
    uint significand = static_cast<uint>(accumulator >> 16);
    return as_type<float>(0x3f000000u + significand);
}

inline float source_cuda_ex2_primary(float value) {
    if (value < 0.0f) {
        if (value <= -1.0f) {
            return 0.0f;
        }
        int signed_coordinate = static_cast<int>(
            metal::floor(value * 8388608.0f));
        if (signed_coordinate == -1) {
            return 1.0f;
        }
        uint coordinate = static_cast<uint>(
            (1 << 23) + signed_coordinate);
        return source_cuda_ex2_primary_coordinate(coordinate) * 0.5f;
    }
    if (value >= 1.0f) {
        return 2.0f;
    }
    uint coordinate = static_cast<uint>(value * 8388608.0f);
    return source_cuda_ex2_primary_coordinate(coordinate);
}

inline uint source_cuda_floor_times_252(float value) {
    if (value <= 0.0f) {
        return 0u;
    }
    if (value >= 1.0f) {
        return 252u;
    }
    uint bits = as_type<uint>(value);
    uint exponent_bits = (bits >> 23) & 0xffu;
    if (exponent_bits == 0u) {
        return 0u;
    }
    ulong significand = static_cast<ulong>(
        (bits & 0x7fffffu) | 0x800000u);
    uint shift = 150u - exponent_bits;
    ulong product = significand * 252ul;
    return shift >= 64u
        ? 0u
        : static_cast<uint>(product >> shift);
}

inline float source_cuda_expf(float value) {
    constexpr float log2e_hi = as_type<float>(1069066811u);
    constexpr float log2e_lo = as_type<float>(849703008u);
    float bucket = metal::saturate(
        metal::fma(value, log2e_hi / 252.0f, 0.5f));
    int exponent = static_cast<int>(
        source_cuda_floor_times_252(bucket)) - 126;
    float reduced = metal::fma(
        value, log2e_hi, -static_cast<float>(exponent));
    reduced = metal::fma(value, log2e_lo, reduced);
    uint scale_bits = static_cast<uint>(exponent + 127) << 23;
    return source_cuda_ex2_primary(reduced)
        * as_type<float>(scale_bits);
}
""",
    ]
)


def _require_primary_coordinates(coordinates: np.ndarray) -> np.ndarray:
    values = np.asarray(coordinates)
    if values.dtype.kind not in "iu":
        raise TypeError("EX2 primary coordinates must use an integer dtype")
    if values.size:
        if values.dtype.kind == "i" and np.any(values < 0):
            raise ValueError("EX2 primary coordinates must be nonnegative")
        if np.any(values.astype(np.uint64) >= np.uint64(1 << 23)):
            raise ValueError("EX2 primary coordinates must be below 2^23")
    return values.astype(np.uint64, copy=False)


def source_cuda_ex2_truncated_square(residual: np.ndarray) -> np.ndarray:
    """Form the 15-bit square after omitted low partial-product carries."""
    values = np.asarray(residual)
    if values.dtype.kind not in "iu":
        raise TypeError("EX2 residuals must use an integer dtype")
    if values.size:
        if values.dtype.kind == "i" and np.any(values < 0):
            raise ValueError("EX2 residuals must be nonnegative")
        if np.any(values.astype(np.uint64) >= np.uint64(1 << 17)):
            raise ValueError("EX2 residuals must be below 2^17")
    values = values.astype(np.uint64, copy=False)

    low = values & np.uint64(0x3FF)
    high = values >> np.uint64(10)
    omitted = low * low
    for bit in range(7):
        high_bit = (high >> np.uint64(bit)) & np.uint64(1)
        low_prefix = low & np.uint64((1 << (8 - bit)) - 1)
        omitted += high_bit * (
            low_prefix << np.uint64(bit + 11)
        )

    exact = (values * values) >> np.uint64(19)
    truncated = exact - (omitted >> np.uint64(19))
    return truncated.astype(np.uint32)


def source_cuda_ex2_primary_bits(coordinates: np.ndarray) -> np.ndarray:
    """Return float32 result bits for Q23 coordinates in the interval [0, 1)."""
    values = _require_primary_coordinates(coordinates)
    segment = values >> np.uint64(17)
    residual = values & np.uint64((1 << 17) - 1)
    square = source_cuda_ex2_truncated_square(residual).astype(np.uint64)

    accumulator = (
        (SOURCE_CUDA_EX2_C0[segment] << np.uint64(14))
        + SOURCE_CUDA_EX2_BIAS
        + np.uint64(2) * SOURCE_CUDA_EX2_C1[segment] * residual
        + np.uint64(2) * SOURCE_CUDA_EX2_C2[segment] * square
    )
    significand = accumulator >> np.uint64(16)
    return (np.uint64(0x3F000000) + significand).astype(np.uint32)
