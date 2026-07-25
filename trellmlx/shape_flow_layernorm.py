"""Explicit LayerNorm backends for the shape SLat flow.

The experimental backend reproduces the reduction schedule selected by
PyTorch 2.10's vectorized CUDA LayerNorm kernel for aligned BF16 rows of width
1536. Metal's correctly rounded reciprocal square root is intentionally left
visible as a measured residual from CUDA's approximate ``rsqrtf``.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx


DEFAULT_BACKEND = "mlx-two-pass"
CUDA_WELFORD_METAL_BACKEND = "cuda-welford-metal"
SUPPORTED_BACKENDS = (DEFAULT_BACKEND, CUDA_WELFORD_METAL_BACKEND)

_backend = DEFAULT_BACKEND
_cuda_welford_kernel = None


def configure_shape_flow_layernorm_backend(name: str) -> None:
    global _backend
    if name not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"unsupported shape-flow LayerNorm backend {name!r}; "
            f"expected one of {SUPPORTED_BACKENDS}"
        )
    _backend = name


def get_shape_flow_layernorm_backend() -> str:
    return _backend


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
    return {
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
            f"{CUDA_WELFORD_METAL_BACKEND} requires bfloat16 input, got {x.dtype}"
        )
    if not x.shape or x.shape[-1] != 1536:
        width = x.shape[-1] if x.shape else None
        raise ValueError(
            f"{CUDA_WELFORD_METAL_BACKEND} requires hidden width 1536, got {width}"
        )
    return _cuda_welford_layernorm(x, eps)


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


def _build_cuda_welford_kernel():
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
        float rstd = metal::precise::rsqrt(mean_sigma[1] + eps[0]);
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
    return mx.fast.metal_kernel(
        name="shape_flow_cuda_welford_layernorm_bf16_1536",
        input_names=["inp", "eps"],
        output_names=["out"],
        header=header,
        source=source,
    )
