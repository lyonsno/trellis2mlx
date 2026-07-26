"""Explicit LayerNorm backends for the shape SLat flow.

The experimental backends reproduce the reduction schedule selected by
PyTorch 2.10's vectorized CUDA LayerNorm kernel for aligned BF16 rows of width
1536. One leaves Metal's correctly rounded reciprocal square root visible as a
measured residual; the other applies an explicitly hash-bound Tesla T4
``MUFU.RSQ`` correction table.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx


DEFAULT_BACKEND = "mlx-two-pass"
CUDA_WELFORD_METAL_BACKEND = "cuda-welford-metal"
CUDA_WELFORD_TURING_T4_BACKEND = "cuda-welford-turing-t4"
SUPPORTED_BACKENDS = (
    DEFAULT_BACKEND,
    CUDA_WELFORD_METAL_BACKEND,
    CUDA_WELFORD_TURING_T4_BACKEND,
)
TURING_RSQRT_LUT_SIZE = 1 << 24

_backend = DEFAULT_BACKEND
_turing_rsqrt_delta_lut = None
_turing_rsqrt_lut_sha256 = None
_cuda_welford_kernel = None
_cuda_welford_stats_kernel = None
_cuda_welford_turing_kernel = None
_cuda_welford_turing_stats_kernel = None


def configure_shape_flow_layernorm_backend(
    name: str,
    *,
    turing_rsqrt_delta_lut: mx.array | None = None,
    turing_rsqrt_lut_sha256: str | None = None,
) -> None:
    global _backend, _turing_rsqrt_delta_lut, _turing_rsqrt_lut_sha256
    if name not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"unsupported shape-flow LayerNorm backend {name!r}; "
            f"expected one of {SUPPORTED_BACKENDS}"
        )
    if name == CUDA_WELFORD_TURING_T4_BACKEND:
        if (
            turing_rsqrt_delta_lut is None
            or turing_rsqrt_lut_sha256 is None
        ):
            raise ValueError(
                f"{name} requires an explicit correction LUT and SHA256"
            )
        _validate_turing_rsqrt_lut(turing_rsqrt_delta_lut)
        if (
            len(turing_rsqrt_lut_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in turing_rsqrt_lut_sha256
            )
        ):
            raise ValueError(
                f"{name} requires a lowercase hexadecimal LUT SHA256"
            )
        _turing_rsqrt_delta_lut = turing_rsqrt_delta_lut
        _turing_rsqrt_lut_sha256 = turing_rsqrt_lut_sha256
    else:
        if (
            turing_rsqrt_delta_lut is not None
            or turing_rsqrt_lut_sha256 is not None
        ):
            raise ValueError(
                "Turing rsqrt correction state is only valid for "
                f"{CUDA_WELFORD_TURING_T4_BACKEND}"
            )
        _turing_rsqrt_delta_lut = None
        _turing_rsqrt_lut_sha256 = None
    _backend = name


def get_shape_flow_layernorm_backend() -> str:
    return _backend


def get_shape_flow_turing_rsqrt_lut_sha256() -> str | None:
    return _turing_rsqrt_lut_sha256


def shape_flow_layernorm_backend_identity(name: str | None = None) -> dict[str, Any]:
    backend = _backend if name is None else name
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"unsupported shape-flow LayerNorm backend {backend!r}; "
            f"expected one of {SUPPORTED_BACKENDS}"
        )
    if backend == DEFAULT_BACKEND:
        return {
            "backend": backend,
            "algorithm": "mlx-fp32-two-pass-mean-and-squared-deviation",
            "experimental": False,
        }
    identity = {
        "backend": backend,
        "algorithm": "pytorch-vectorized-cuda-welford-schedule-on-metal",
        "experimental": True,
        "cuda_source_tag": "pytorch-v2.10.0",
        "cuda_source_kernel": "vectorized_layer_norm_kernel",
        "authenticated_contract": {
            "dtype": "bfloat16",
            "hidden_width": 1536,
            "affine": False,
        },
        "reduction": {
            "threads": 128,
            "warps": 4,
            "vector_width": 4,
            "values_per_thread": 12,
            "accumulator_dtype": "float32",
        },
    }
    if backend == CUDA_WELFORD_TURING_T4_BACKEND:
        if _turing_rsqrt_lut_sha256 is None:
            raise ValueError(
                f"{backend} has no configured Turing rsqrt LUT identity"
            )
        return {
            **identity,
            "cuda_architecture": "sm_75",
            "cuda_device_anchor": "Tesla T4",
            "cuda_rsqrt_bit_exact": True,
            "rsqrt": "Turing MUFU.RSQ normalized signed-ULP LUT",
            "turing_rsqrt_lut_sha256": _turing_rsqrt_lut_sha256,
            "turing_rsqrt_lut_entries": TURING_RSQRT_LUT_SIZE,
        }
    return {
        **identity,
        "cuda_rsqrt_bit_exact": False,
        "rsqrt": "metal::precise::rsqrt",
        "measured_residual": {
            "native_partial_rows": 4096,
            "native_partial_mean_mismatches": 0,
            "native_partial_rstd_mismatches": 903,
            "native_partial_output_mismatches": 37,
            "shape_witness_disputed_coordinates": 175,
            "shape_witness_source_coordinates_recovered": 152,
        },
    }


def layernorm_noaffine(x: mx.array, eps: float = 1e-6) -> mx.array:
    if _backend == DEFAULT_BACKEND:
        return _mlx_two_pass_layernorm(x, eps)
    if x.dtype != mx.bfloat16:
        raise ValueError(
            f"{_backend} requires bfloat16 input, got {x.dtype}"
        )
    if not x.shape or x.shape[-1] != 1536:
        width = x.shape[-1] if x.shape else None
        raise ValueError(
            f"{_backend} requires hidden width 1536, got {width}"
        )
    if _backend == CUDA_WELFORD_TURING_T4_BACKEND:
        if _turing_rsqrt_delta_lut is None:
            raise RuntimeError(
                f"{_backend} correction LUT is not configured"
            )
        return _cuda_welford_turing_layernorm(
            x, _turing_rsqrt_delta_lut, eps
        )
    return _cuda_welford_layernorm(x, eps)


def cuda_welford_layernorm_with_stats(
    x: mx.array, eps: float = 1e-6
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Run the authenticated Metal schedule and expose pre/post-rsqrt stats."""
    if x.dtype != mx.bfloat16:
        raise ValueError(
            f"{CUDA_WELFORD_METAL_BACKEND} diagnostics require bfloat16 input, "
            f"got {x.dtype}"
        )
    if not x.shape or x.shape[-1] != 1536:
        width = x.shape[-1] if x.shape else None
        raise ValueError(
            f"{CUDA_WELFORD_METAL_BACKEND} diagnostics require hidden width "
            f"1536, got {width}"
        )
    return _cuda_welford_layernorm_with_stats(x, eps)


def cuda_welford_turing_layernorm_with_stats(
    x: mx.array,
    rsqrt_delta_lut: mx.array,
    eps: float = 1e-6,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Run Welford LayerNorm with an explicit normalized Turing rsqrt LUT."""
    if x.dtype != mx.bfloat16:
        raise ValueError(
            "Turing-rsqrt diagnostics require bfloat16 input, "
            f"got {x.dtype}"
        )
    if not x.shape or x.shape[-1] != 1536:
        width = x.shape[-1] if x.shape else None
        raise ValueError(
            "Turing-rsqrt diagnostics require hidden width 1536, "
            f"got {width}"
        )
    _validate_turing_rsqrt_lut(rsqrt_delta_lut)
    return _cuda_welford_turing_layernorm_with_stats(
        x, rsqrt_delta_lut, eps
    )


def _validate_turing_rsqrt_lut(rsqrt_delta_lut: mx.array) -> None:
    if rsqrt_delta_lut.dtype != mx.int8:
        raise ValueError(
            "Turing-rsqrt correction LUT requires int8, "
            f"got {rsqrt_delta_lut.dtype}"
        )
    if (
        rsqrt_delta_lut.ndim != 1
        or rsqrt_delta_lut.size != TURING_RSQRT_LUT_SIZE
    ):
        raise ValueError(
            "Turing-rsqrt correction LUT requires "
            f"{TURING_RSQRT_LUT_SIZE} entries, "
            f"got shape {rsqrt_delta_lut.shape}"
        )


def _mlx_two_pass_layernorm(x: mx.array, eps: float) -> mx.array:
    if x.dtype in (mx.bfloat16, mx.float16):
        input_dtype = x.dtype
        xf = x.astype(mx.float32)
        mean = mx.mean(xf, axis=-1, keepdims=True)
        var = mx.mean((xf - mean) * (xf - mean), axis=-1, keepdims=True)
        return ((xf - mean) * mx.rsqrt(var + eps)).astype(input_dtype)
    return mx.fast.layer_norm(x, None, None, eps)


def _cuda_welford_layernorm(x: mx.array, eps: float) -> mx.array:
    global _cuda_welford_kernel
    if _cuda_welford_kernel is None:
        _cuda_welford_kernel = _build_cuda_welford_kernel()

    rows = x.size // 1536
    eps_array = mx.array([eps], dtype=mx.float32)
    return _cuda_welford_kernel(
        inputs=[x, eps_array],
        template=[("T", mx.bfloat16)],
        grid=(32, 4, rows),
        threadgroup=(32, 4, 1),
        output_shapes=[x.shape],
        output_dtypes=[mx.bfloat16],
    )[0]


def _cuda_welford_layernorm_with_stats(
    x: mx.array, eps: float
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    global _cuda_welford_stats_kernel
    if _cuda_welford_stats_kernel is None:
        _cuda_welford_stats_kernel = _build_cuda_welford_kernel(
            include_stats=True
        )

    rows = x.size // 1536
    eps_array = mx.array([eps], dtype=mx.float32)
    stats_shape = (*x.shape[:-1], 1)
    out, mean, variance, rstd = _cuda_welford_stats_kernel(
        inputs=[x, eps_array],
        template=[("T", mx.bfloat16)],
        grid=(32, 4, rows),
        threadgroup=(32, 4, 1),
        output_shapes=[x.shape, stats_shape, stats_shape, stats_shape],
        output_dtypes=[
            mx.bfloat16,
            mx.float32,
            mx.float32,
            mx.float32,
        ],
    )
    return out, mean, variance, rstd


def _cuda_welford_turing_layernorm(
    x: mx.array,
    rsqrt_delta_lut: mx.array,
    eps: float,
) -> mx.array:
    global _cuda_welford_turing_kernel
    if _cuda_welford_turing_kernel is None:
        _cuda_welford_turing_kernel = _build_cuda_welford_kernel(
            turing_rsqrt=True
        )

    rows = x.size // 1536
    eps_array = mx.array([eps], dtype=mx.float32)
    return _cuda_welford_turing_kernel(
        inputs=[x, eps_array, rsqrt_delta_lut],
        template=[("T", mx.bfloat16)],
        grid=(32, 4, rows),
        threadgroup=(32, 4, 1),
        output_shapes=[x.shape],
        output_dtypes=[mx.bfloat16],
    )[0]


def _cuda_welford_turing_layernorm_with_stats(
    x: mx.array,
    rsqrt_delta_lut: mx.array,
    eps: float,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    global _cuda_welford_turing_stats_kernel
    if _cuda_welford_turing_stats_kernel is None:
        _cuda_welford_turing_stats_kernel = _build_cuda_welford_kernel(
            include_stats=True,
            turing_rsqrt=True,
        )

    rows = x.size // 1536
    eps_array = mx.array([eps], dtype=mx.float32)
    stats_shape = (*x.shape[:-1], 1)
    out, mean, variance, rstd = _cuda_welford_turing_stats_kernel(
        inputs=[x, eps_array, rsqrt_delta_lut],
        template=[("T", mx.bfloat16)],
        grid=(32, 4, rows),
        threadgroup=(32, 4, 1),
        output_shapes=[x.shape, stats_shape, stats_shape, stats_shape],
        output_dtypes=[
            mx.bfloat16,
            mx.float32,
            mx.float32,
            mx.float32,
        ],
    )
    return out, mean, variance, rstd


def _build_cuda_welford_kernel(
    *,
    include_stats: bool = False,
    turing_rsqrt: bool = False,
):
    header = r"""
        struct WelfordDataLN {
            float mean;
            float sigma2;
            float count;
        };

        inline WelfordDataLN welford_online(
            float value,
            WelfordDataLN current
        ) {
            float delta = value - current.mean;
            float new_count = current.count + 1.0f;
            float new_mean = current.mean + delta * (1.0f / new_count);
            return {
                new_mean,
                current.sigma2 + delta * (value - new_mean),
                new_count
            };
        }

        inline WelfordDataLN welford_combine(
            WelfordDataLN data_b,
            WelfordDataLN data_a
        ) {
            float delta = data_b.mean - data_a.mean;
            float count = data_a.count + data_b.count;
            if (count <= 0.0f) {
                return {0.0f, 0.0f, 0.0f};
            }
            float coefficient = 1.0f / count;
            float n_a = data_a.count * coefficient;
            float n_b = data_b.count * coefficient;
            float mean = n_a * data_a.mean + n_b * data_b.mean;
            float sigma2 = data_a.sigma2 + data_b.sigma2
                + delta * delta * data_a.count * n_b;
            return {mean, sigma2, count};
        }
    """
    source = r"""
        constexpr uint width = 1536;
        constexpr uint vector_width = 4;
        constexpr uint warp_width = 32;
        constexpr uint warp_count = 4;

        uint lane = thread_position_in_threadgroup.x;
        uint warp = thread_position_in_threadgroup.y;
        uint thread_index = lane + warp * warp_width;
        uint row = threadgroup_position_in_grid.z;
        uint row_offset = row * width;

        WelfordDataLN wd = {0.0f, 0.0f, 0.0f};
        for (uint vector_index = thread_index;
             vector_index < width / vector_width;
             vector_index += warp_width * warp_count) {
            uint offset = row_offset + vector_index * vector_width;
            wd = welford_online(static_cast<float>(inp[offset]), wd);
            wd = welford_online(static_cast<float>(inp[offset + 1]), wd);
            wd = welford_online(static_cast<float>(inp[offset + 2]), wd);
            wd = welford_online(static_cast<float>(inp[offset + 3]), wd);
        }

        for (ushort offset = 16; offset > 0; offset >>= 1) {
            WelfordDataLN upper = {
                simd_shuffle_down(wd.mean, offset),
                simd_shuffle_down(wd.sigma2, offset),
                simd_shuffle_down(wd.count, offset)
            };
            wd = welford_combine(wd, upper);
        }

        threadgroup float mean_sigma[4];
        threadgroup float counts[2];
        for (uint offset = 2; offset > 0; offset >>= 1) {
            if (lane == 0 && warp >= offset && warp < 2 * offset) {
                uint slot = warp - offset;
                mean_sigma[2 * slot] = wd.mean;
                mean_sigma[2 * slot + 1] = wd.sigma2;
                counts[slot] = wd.count;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0 && warp < offset) {
                WelfordDataLN upper = {
                    mean_sigma[2 * warp],
                    mean_sigma[2 * warp + 1],
                    counts[warp]
                };
                wd = welford_combine(wd, upper);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0 && warp == 0) {
            mean_sigma[0] = wd.mean;
            mean_sigma[1] = wd.sigma2 / static_cast<float>(width);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float mean = mean_sigma[0];
        __RSQRT__
        __WRITE_STATS__
        for (uint vector_index = thread_index;
             vector_index < width / vector_width;
             vector_index += warp_width * warp_count) {
            uint offset = row_offset + vector_index * vector_width;
            out[offset] = static_cast<T>(
                (static_cast<float>(inp[offset]) - mean) * rstd);
            out[offset + 1] = static_cast<T>(
                (static_cast<float>(inp[offset + 1]) - mean) * rstd);
            out[offset + 2] = static_cast<T>(
                (static_cast<float>(inp[offset + 2]) - mean) * rstd);
            out[offset + 3] = static_cast<T>(
                (static_cast<float>(inp[offset + 3]) - mean) * rstd);
        }
    """
    stats_write = ""
    output_names = ["out"]
    input_names = ["inp", "eps"]
    kernel_name = "shape_flow_cuda_welford_layernorm_bf16_1536"
    rsqrt_source = (
        "float rstd = metal::precise::rsqrt(mean_sigma[1] + eps[0]);"
    )
    if turing_rsqrt:
        input_names.append("rsqrt_delta")
        kernel_name += "_turing_mufu_rsqrt_lut"
        rsqrt_source = """
        float rsqrt_input = mean_sigma[1] + eps[0];
        float rounded_rstd = metal::precise::rsqrt(rsqrt_input);
        uint input_bits = as_type<uint>(rsqrt_input);
        uint biased_exponent = (input_bits >> 23) & 0xffu;
        uint mantissa = input_bits & 0x7fffffu;
        uint coordinate =
            mantissa | (((biased_exponent - 127u) & 1u) << 23);
        int corrected_bits =
            static_cast<int>(as_type<uint>(rounded_rstd))
            + static_cast<int>(rsqrt_delta[coordinate]);
        float rstd = as_type<float>(static_cast<uint>(corrected_bits));
        """
    if include_stats:
        stats_write = """
        if (lane == 0 && warp == 0) {
            mean_out[row] = mean;
            variance_out[row] = mean_sigma[1];
            rstd_out[row] = rstd;
        }
        """
        output_names.extend(["mean_out", "variance_out", "rstd_out"])
        kernel_name += "_with_stats"
    source = source.replace("__RSQRT__", rsqrt_source)
    source = source.replace("__WRITE_STATS__", stats_write)
    return mx.fast.metal_kernel(
        name=kernel_name,
        input_names=input_names,
        output_names=output_names,
        header=header,
        source=source,
    )
