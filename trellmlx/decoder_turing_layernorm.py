"""Turing-exact FP16 LayerNorm for the shape decoder.

PyTorch 2.10's vectorized CUDA LayerNorm uses 128 threads arranged as four
warps, four values per vector, FP32 Welford state, and CUDA ``rsqrtf``. This
module preserves that reduction and affine evaluation order on Metal while
applying the authenticated Tesla T4 ``MUFU.RSQ`` correction table.
"""

from __future__ import annotations

import hashlib
from typing import Any

import mlx.core as mx
import numpy as np


DEFAULT_BACKEND = "mlx-fast-layer-norm"
CUDA_WELFORD_TURING_T4_BACKEND = "cuda-welford-turing-t4"
SUPPORTED_BACKENDS = (DEFAULT_BACKEND, CUDA_WELFORD_TURING_T4_BACKEND)
TURING_RSQRT_LUT_SIZE = 1 << 24

_backend = DEFAULT_BACKEND
_turing_rsqrt_delta_lut = None
_turing_rsqrt_lut_artifact_sha256_attested = None
_turing_rsqrt_lut_content_sha256 = None
_affine_kernel = None
_affine_stats_kernel = None
_noaffine_kernel = None
_noaffine_stats_kernel = None


def configure_decoder_layernorm_backend(
    name: str,
    *,
    turing_rsqrt_delta_lut: mx.array | None = None,
    turing_rsqrt_lut_artifact_sha256_attested: str | None = None,
) -> None:
    """Select the decoder LayerNorm route and bind its external evidence."""
    global _backend
    global _turing_rsqrt_delta_lut
    global _turing_rsqrt_lut_artifact_sha256_attested
    global _turing_rsqrt_lut_content_sha256
    if name not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"unsupported decoder LayerNorm backend {name!r}; "
            f"expected one of {SUPPORTED_BACKENDS}"
        )
    if name == CUDA_WELFORD_TURING_T4_BACKEND:
        if (
            turing_rsqrt_delta_lut is None
            or turing_rsqrt_lut_artifact_sha256_attested is None
        ):
            raise ValueError(
                f"{name} requires an explicit correction LUT and SHA256"
            )
        _validate_turing_rsqrt_lut(turing_rsqrt_delta_lut)
        digest = turing_rsqrt_lut_artifact_sha256_attested
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(
                f"{name} requires a lowercase hexadecimal attested artifact SHA256"
            )
        _turing_rsqrt_delta_lut = turing_rsqrt_delta_lut
        _turing_rsqrt_lut_artifact_sha256_attested = digest
        payload = np.ascontiguousarray(np.asarray(turing_rsqrt_delta_lut))
        _turing_rsqrt_lut_content_sha256 = hashlib.sha256(
            payload.tobytes()
        ).hexdigest()
    else:
        if (
            turing_rsqrt_delta_lut is not None
            or turing_rsqrt_lut_artifact_sha256_attested is not None
        ):
            raise ValueError(
                "Turing rsqrt correction state is only valid for "
                f"{CUDA_WELFORD_TURING_T4_BACKEND}"
            )
        _turing_rsqrt_delta_lut = None
        _turing_rsqrt_lut_artifact_sha256_attested = None
        _turing_rsqrt_lut_content_sha256 = None
    _backend = name


def get_decoder_layernorm_backend() -> str:
    return _backend


def decoder_layernorm_backend_identity(
    name: str | None = None,
) -> dict[str, Any]:
    backend = _backend if name is None else name
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"unsupported decoder LayerNorm backend {backend!r}; "
            f"expected one of {SUPPORTED_BACKENDS}"
        )
    if backend == DEFAULT_BACKEND:
        return {
            "backend": backend,
            "algorithm": "mlx-fast-layer-norm",
            "experimental": False,
        }
    if (
        _turing_rsqrt_lut_artifact_sha256_attested is None
        or _turing_rsqrt_lut_content_sha256 is None
    ):
        raise ValueError(f"{backend} has no configured Turing rsqrt LUT identity")
    return {
        "backend": backend,
        "algorithm": (
            "pytorch-2.10-vectorized-layernorm-128-thread-welford-"
            "turing-rsqrt-on-metal"
        ),
        "experimental": True,
        "cuda_source_tag": "pytorch-v2.10.0",
        "cuda_source_kernel": "vectorized_layer_norm_kernel",
        "cuda_architecture": "sm_75",
        "cuda_device_anchor": "Tesla T4",
        "cuda_rsqrt_bit_exact_for_configured_lut": True,
        "authenticated_contract": {
            "input_dtype": "float16",
            "parameter_dtype": "float16",
            "hidden_width": 1024,
            "affine": True,
        },
        "authenticated_contracts": [
            {
                "input_dtype": "float16",
                "parameter_dtype": "float16",
                "hidden_width": 1024,
                "affine": True,
                "reduction": {
                    "threads": 128,
                    "warps": 4,
                    "vector_width": 4,
                    "values_per_thread": 8,
                    "accumulator_dtype": "float32",
                },
            },
            {
                "input_dtype": "float16",
                "parameter_dtype": "float16",
                "hidden_width": 512,
                "affine": True,
                "reduction": {
                    "threads": 128,
                    "warps": 4,
                    "vector_width": 4,
                    "values_per_thread": 4,
                    "accumulator_dtype": "float32",
                },
            },
            {
                "input_dtype": "float16",
                "hidden_width": 512,
                "affine": False,
                "reduction": {
                    "threads": 128,
                    "warps": 4,
                    "vector_width": 4,
                    "values_per_thread": 4,
                    "accumulator_dtype": "float32",
                },
            },
        ],
        "reduction": {
            "threads": 128,
            "warps": 4,
            "vector_width": 4,
            "values_per_thread": 8,
            "accumulator_dtype": "float32",
        },
        "rsqrt": "Turing MUFU.RSQ normalized signed-ULP LUT",
        "turing_rsqrt_lut_artifact_sha256_attested": (
            _turing_rsqrt_lut_artifact_sha256_attested
        ),
        "turing_rsqrt_lut_content_sha256": _turing_rsqrt_lut_content_sha256,
        "turing_rsqrt_lut_entries": TURING_RSQRT_LUT_SIZE,
    }


def layernorm_affine(
    x: mx.array,
    weight: mx.array,
    bias: mx.array,
    eps: float = 1e-6,
) -> mx.array:
    """Dispatch an affine decoder LayerNorm through the configured backend."""
    if _backend == DEFAULT_BACKEND:
        return mx.fast.layer_norm(x, weight, bias, eps).astype(x.dtype)
    if x.ndim != 2 or x.shape[1] not in (1024, 512):
        shape = x.shape if x.ndim == 2 else None
        raise ValueError(
            f"{_backend} affine route is authenticated only for "
            f"2D width-1024 or width-512 rows, got {shape}"
        )
    if _turing_rsqrt_delta_lut is None:
        raise RuntimeError(f"{_backend} correction LUT is not configured")
    return turing_layernorm_affine_fp16(
        x,
        weight,
        bias,
        _turing_rsqrt_delta_lut,
        eps,
    )


def layernorm_noaffine(
    x: mx.array,
    eps: float = 1e-6,
) -> mx.array:
    """Dispatch an authenticated non-affine decoder LayerNorm."""
    if _backend == DEFAULT_BACKEND:
        return mx.fast.layer_norm(x, None, None, eps).astype(x.dtype)
    if x.ndim != 2 or x.shape[1] != 512:
        shape = x.shape if x.ndim == 2 else None
        raise ValueError(
            f"{_backend} non-affine route is authenticated only for "
            f"2D width-512 rows, got {shape}"
        )
    if _turing_rsqrt_delta_lut is None:
        raise RuntimeError(f"{_backend} correction LUT is not configured")
    return turing_layernorm_noaffine_fp16(
        x,
        _turing_rsqrt_delta_lut,
        eps,
    )


def _validate_turing_rsqrt_lut(rsqrt_delta_lut: mx.array) -> None:
    if rsqrt_delta_lut.dtype != mx.int8:
        raise ValueError(
            "Turing decoder LayerNorm rsqrt correction requires int8, "
            f"got {rsqrt_delta_lut.dtype}"
        )
    if (
        rsqrt_delta_lut.ndim != 1
        or rsqrt_delta_lut.size != TURING_RSQRT_LUT_SIZE
    ):
        raise ValueError(
            "Turing decoder LayerNorm rsqrt correction requires "
            f"{TURING_RSQRT_LUT_SIZE} entries, got shape {rsqrt_delta_lut.shape}"
        )


def _validate_input_contract(
    x: mx.array,
    rsqrt_delta_lut: mx.array,
) -> int:
    if x.dtype != mx.float16:
        raise ValueError(f"Turing decoder LayerNorm requires float16 input, got {x.dtype}")
    if x.ndim != 2:
        raise ValueError(
            f"Turing decoder LayerNorm requires a 2D input, got shape {x.shape}"
        )
    width = x.shape[1]
    if width == 0 or width % 4:
        raise ValueError(
            f"Turing decoder LayerNorm width must be a nonzero multiple of 4, got {width}"
        )
    _validate_turing_rsqrt_lut(rsqrt_delta_lut)
    return width


def _validate_affine_contract(
    x: mx.array,
    weight: mx.array,
    bias: mx.array,
    rsqrt_delta_lut: mx.array,
) -> int:
    width = _validate_input_contract(x, rsqrt_delta_lut)
    for name, parameter in (("weight", weight), ("bias", bias)):
        if parameter.dtype != mx.float16:
            raise ValueError(
                f"Turing decoder LayerNorm requires float16 {name}, "
                f"got {parameter.dtype}"
            )
        if parameter.shape != (width,):
            raise ValueError(
                f"Turing decoder LayerNorm {name} shape must be ({width},), "
                f"got {parameter.shape}"
            )
    return width


def turing_layernorm_affine_fp16(
    x: mx.array,
    weight: mx.array,
    bias: mx.array,
    rsqrt_delta_lut: mx.array,
    eps: float = 1e-6,
) -> mx.array:
    """Apply the PyTorch CUDA FP16 affine LayerNorm arithmetic contract."""
    global _affine_kernel
    width = _validate_affine_contract(x, weight, bias, rsqrt_delta_lut)
    if _affine_kernel is None:
        _affine_kernel = _build_affine_kernel(include_stats=False)
    return _run_affine_kernel(
        _affine_kernel,
        x,
        weight,
        bias,
        rsqrt_delta_lut,
        width,
        eps,
        include_stats=False,
    )[0]


def turing_layernorm_affine_fp16_with_stats(
    x: mx.array,
    weight: mx.array,
    bias: mx.array,
    rsqrt_delta_lut: mx.array,
    eps: float = 1e-6,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Apply the exact route and expose mean, pre-rsqrt variance, and rstd."""
    global _affine_stats_kernel
    width = _validate_affine_contract(x, weight, bias, rsqrt_delta_lut)
    if _affine_stats_kernel is None:
        _affine_stats_kernel = _build_affine_kernel(include_stats=True)
    return tuple(
        _run_affine_kernel(
            _affine_stats_kernel,
            x,
            weight,
            bias,
            rsqrt_delta_lut,
            width,
            eps,
            include_stats=True,
        )
    )


def turing_layernorm_noaffine_fp16(
    x: mx.array,
    rsqrt_delta_lut: mx.array,
    eps: float = 1e-6,
) -> mx.array:
    """Apply the PyTorch CUDA FP16 non-affine LayerNorm arithmetic contract."""
    global _noaffine_kernel
    width = _validate_input_contract(x, rsqrt_delta_lut)
    if _noaffine_kernel is None:
        _noaffine_kernel = _build_noaffine_kernel(include_stats=False)
    return _run_noaffine_kernel(
        _noaffine_kernel,
        x,
        rsqrt_delta_lut,
        width,
        eps,
        include_stats=False,
    )[0]


def turing_layernorm_noaffine_fp16_with_stats(
    x: mx.array,
    rsqrt_delta_lut: mx.array,
    eps: float = 1e-6,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Apply the non-affine exact route and expose its reduction statistics."""
    global _noaffine_stats_kernel
    width = _validate_input_contract(x, rsqrt_delta_lut)
    if _noaffine_stats_kernel is None:
        _noaffine_stats_kernel = _build_noaffine_kernel(include_stats=True)
    return tuple(
        _run_noaffine_kernel(
            _noaffine_stats_kernel,
            x,
            rsqrt_delta_lut,
            width,
            eps,
            include_stats=True,
        )
    )


def _run_affine_kernel(
    kernel,
    x: mx.array,
    weight: mx.array,
    bias: mx.array,
    rsqrt_delta_lut: mx.array,
    width: int,
    eps: float,
    *,
    include_stats: bool,
):
    rows = x.shape[0]
    inputs = [
        x,
        mx.array([eps], dtype=mx.float32),
        weight,
        bias,
        rsqrt_delta_lut,
        mx.array([width], dtype=mx.uint32),
    ]
    output_shapes = [x.shape]
    output_dtypes = [mx.float16]
    if include_stats:
        stats_shape = (rows, 1)
        output_shapes.extend((stats_shape, stats_shape, stats_shape))
        output_dtypes.extend((mx.float32, mx.float32, mx.float32))
    return kernel(
        inputs=inputs,
        template=[("T", mx.float16)],
        grid=(32, 4, rows),
        threadgroup=(32, 4, 1),
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
    )


def _run_noaffine_kernel(
    kernel,
    x: mx.array,
    rsqrt_delta_lut: mx.array,
    width: int,
    eps: float,
    *,
    include_stats: bool,
):
    rows = x.shape[0]
    inputs = [
        x,
        mx.array([eps], dtype=mx.float32),
        rsqrt_delta_lut,
        mx.array([width], dtype=mx.uint32),
    ]
    output_shapes = [x.shape]
    output_dtypes = [mx.float16]
    if include_stats:
        stats_shape = (rows, 1)
        output_shapes.extend((stats_shape, stats_shape, stats_shape))
        output_dtypes.extend((mx.float32, mx.float32, mx.float32))
    return kernel(
        inputs=inputs,
        template=[("T", mx.float16)],
        grid=(32, 4, rows),
        threadgroup=(32, 4, 1),
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
    )


def _build_affine_kernel(*, include_stats: bool):
    return _build_kernel(affine=True, include_stats=include_stats)


def _build_noaffine_kernel(*, include_stats: bool):
    return _build_kernel(affine=False, include_stats=include_stats)


def _build_kernel(*, affine: bool, include_stats: bool):
    header = r"""
        struct WelfordDataDecoder {
            float mean;
            float sigma2;
            float count;
        };

        inline WelfordDataDecoder welford_online_decoder(
            float value,
            WelfordDataDecoder current
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

        inline WelfordDataDecoder welford_combine_decoder(
            WelfordDataDecoder data_b,
            WelfordDataDecoder data_a
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
    stats_write = ""
    output_names = ["out"]
    if include_stats:
        stats_write = r"""
        if (lane == 0 && warp == 0) {
            mean_out[row] = mean;
            variance_out[row] = variance;
            rstd_out[row] = rstd;
        }
        """
        output_names.extend(("mean_out", "variance_out", "rstd_out"))
    if affine:
        output_write = r"""
        for (uint vector_index = thread_index;
             vector_index < width_value / vector_width;
             vector_index += warp_width * warp_count) {
            uint channel = vector_index * vector_width;
            uint offset = row_offset + channel;
            float normalized0 =
                rstd * (static_cast<float>(inp[offset]) - mean);
            float normalized1 =
                rstd * (static_cast<float>(inp[offset + 1]) - mean);
            float normalized2 =
                rstd * (static_cast<float>(inp[offset + 2]) - mean);
            float normalized3 =
                rstd * (static_cast<float>(inp[offset + 3]) - mean);
            out[offset] = static_cast<T>(
                metal::fma(static_cast<float>(weight[channel]), normalized0,
                           static_cast<float>(bias[channel])));
            out[offset + 1] = static_cast<T>(
                metal::fma(static_cast<float>(weight[channel + 1]), normalized1,
                           static_cast<float>(bias[channel + 1])));
            out[offset + 2] = static_cast<T>(
                metal::fma(static_cast<float>(weight[channel + 2]), normalized2,
                           static_cast<float>(bias[channel + 2])));
            out[offset + 3] = static_cast<T>(
                metal::fma(static_cast<float>(weight[channel + 3]), normalized3,
                           static_cast<float>(bias[channel + 3])));
        }
        """
    else:
        output_write = r"""
        for (uint vector_index = thread_index;
             vector_index < width_value / vector_width;
             vector_index += warp_width * warp_count) {
            uint offset = row_offset + vector_index * vector_width;
            out[offset] = static_cast<T>(
                rstd * (static_cast<float>(inp[offset]) - mean));
            out[offset + 1] = static_cast<T>(
                rstd * (static_cast<float>(inp[offset + 1]) - mean));
            out[offset + 2] = static_cast<T>(
                rstd * (static_cast<float>(inp[offset + 2]) - mean));
            out[offset + 3] = static_cast<T>(
                rstd * (static_cast<float>(inp[offset + 3]) - mean));
        }
        """
    source = r"""
        constexpr uint vector_width = 4;
        constexpr uint warp_width = 32;
        constexpr uint warp_count = 4;

        uint width_value = width[0];
        uint lane = thread_position_in_threadgroup.x;
        uint warp = thread_position_in_threadgroup.y;
        uint thread_index = lane + warp * warp_width;
        uint row = threadgroup_position_in_grid.z;
        uint row_offset = row * width_value;

        WelfordDataDecoder wd = {0.0f, 0.0f, 0.0f};
        for (uint vector_index = thread_index;
             vector_index < width_value / vector_width;
             vector_index += warp_width * warp_count) {
            uint offset = row_offset + vector_index * vector_width;
            wd = welford_online_decoder(static_cast<float>(inp[offset]), wd);
            wd = welford_online_decoder(static_cast<float>(inp[offset + 1]), wd);
            wd = welford_online_decoder(static_cast<float>(inp[offset + 2]), wd);
            wd = welford_online_decoder(static_cast<float>(inp[offset + 3]), wd);
        }

        for (ushort offset = 16; offset > 0; offset >>= 1) {
            WelfordDataDecoder upper = {
                simd_shuffle_down(wd.mean, offset),
                simd_shuffle_down(wd.sigma2, offset),
                simd_shuffle_down(wd.count, offset)
            };
            wd = welford_combine_decoder(wd, upper);
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
                WelfordDataDecoder upper = {
                    mean_sigma[2 * warp],
                    mean_sigma[2 * warp + 1],
                    counts[warp]
                };
                wd = welford_combine_decoder(wd, upper);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0 && warp == 0) {
            mean_sigma[0] = wd.mean;
            mean_sigma[1] = wd.sigma2 / static_cast<float>(width_value);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float mean = mean_sigma[0];
        float variance = mean_sigma[1];
        float rsqrt_input = variance + eps[0];
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

        __WRITE_STATS__
        __OUTPUT_WRITE__
    """.replace("__WRITE_STATS__", stats_write).replace(
        "__OUTPUT_WRITE__",
        output_write,
    )
    route = "affine" if affine else "noaffine"
    input_names = ["inp", "eps"]
    if affine:
        input_names.extend(("weight", "bias"))
    input_names.extend(("rsqrt_delta", "width"))
    return mx.fast.metal_kernel(
        name=(
            f"decoder_turing_welford_layernorm_f16_{route}_stats"
            if include_stats
            else f"decoder_turing_welford_layernorm_f16_{route}"
        ),
        input_names=input_names,
        output_names=output_names,
        header=header,
        source=source,
    )
