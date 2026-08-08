"""Explicit LayerNorm backends for the sparse-structure flow."""

from __future__ import annotations

import hashlib
from typing import Any

import mlx.core as mx
import numpy as np

from .shape_flow_layernorm import (
    CUDA_WELFORD_METAL_BACKEND,
    CUDA_WELFORD_TURING_T4_BACKEND,
    DEFAULT_BACKEND,
    SUPPORTED_BACKENDS,
    TURING_RSQRT_LUT_SIZE,
    _cuda_welford_affine_layernorm,
    _cuda_welford_layernorm,
    _cuda_welford_layernorm_float32_output,
    _cuda_welford_turing_affine_layernorm,
    _cuda_welford_turing_layernorm,
    _cuda_welford_turing_layernorm_float32_output,
    _mlx_two_pass_layernorm,
)

_backend = DEFAULT_BACKEND
_turing_rsqrt_delta_lut: mx.array | None = None
_turing_rsqrt_lut_artifact_sha256_attested: str | None = None
_turing_rsqrt_lut_content_sha256: str | None = None


def configure_sparse_flow_layernorm_backend(
    name: str,
    *,
    turing_rsqrt_delta_lut: mx.array | None = None,
    turing_rsqrt_lut_artifact_sha256_attested: str | None = None,
) -> None:
    global _backend
    global _turing_rsqrt_delta_lut
    global _turing_rsqrt_lut_artifact_sha256_attested
    global _turing_rsqrt_lut_content_sha256

    if name not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"unsupported sparse-flow LayerNorm backend {name!r}; "
            f"expected one of {SUPPORTED_BACKENDS}"
        )
    if name == CUDA_WELFORD_TURING_T4_BACKEND:
        if (
            turing_rsqrt_delta_lut is None
            or turing_rsqrt_lut_artifact_sha256_attested is None
        ):
            raise ValueError(f"{name} requires an explicit correction LUT and SHA256")
        _validate_turing_lut(turing_rsqrt_delta_lut)
        digest = turing_rsqrt_lut_artifact_sha256_attested
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(
                f"{name} requires a lowercase hexadecimal attested artifact SHA256"
            )
        payload = np.ascontiguousarray(np.asarray(turing_rsqrt_delta_lut))
        _turing_rsqrt_delta_lut = turing_rsqrt_delta_lut
        _turing_rsqrt_lut_artifact_sha256_attested = digest
        _turing_rsqrt_lut_content_sha256 = hashlib.sha256(payload.tobytes()).hexdigest()
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


def get_sparse_flow_layernorm_backend() -> str:
    return _backend


def get_sparse_flow_turing_rsqrt_lut_artifact_sha256_attested() -> str | None:
    return _turing_rsqrt_lut_artifact_sha256_attested


def get_sparse_flow_turing_rsqrt_lut_content_sha256() -> str | None:
    return _turing_rsqrt_lut_content_sha256


def sparse_flow_layernorm_backend_identity() -> dict[str, Any]:
    identity: dict[str, Any] = {
        "backend": _backend,
        "scope": "sparse-structure-flow",
        "input_dtype": "bfloat16",
        "hidden_width": 1536,
    }
    if _backend == DEFAULT_BACKEND:
        return {
            **identity,
            "algorithm": "mlx-fp32-two-pass-mean-and-squared-deviation",
            "experimental": False,
        }
    identity.update(
        {
            "algorithm": "pytorch-vectorized-cuda-welford-schedule-on-metal",
            "experimental": True,
            "cuda_source_tag": "pytorch-v2.10.0",
            "cuda_source_kernel": "vectorized_layer_norm_kernel",
            "reduction": {
                "threads": 128,
                "warps": 4,
                "vector_width": 4,
                "values_per_thread": 12,
                "accumulator_dtype": "float32",
            },
        }
    )
    if _backend == CUDA_WELFORD_TURING_T4_BACKEND:
        if (
            _turing_rsqrt_lut_artifact_sha256_attested is None
            or _turing_rsqrt_lut_content_sha256 is None
        ):
            raise ValueError(f"{_backend} has no configured Turing rsqrt LUT identity")
        identity.update(
            {
                "cuda_architecture": "sm_75",
                "cuda_device_anchor": "Tesla T4",
                "rsqrt": "Turing MUFU.RSQ normalized signed-ULP LUT",
                "turing_rsqrt_lut_artifact_sha256_attested": (
                    _turing_rsqrt_lut_artifact_sha256_attested
                ),
                "turing_rsqrt_lut_content_sha256": _turing_rsqrt_lut_content_sha256,
                "turing_rsqrt_lut_entries": TURING_RSQRT_LUT_SIZE,
            }
        )
    else:
        identity.update(
            {
                "cuda_rsqrt_bit_exact": False,
                "rsqrt": "metal::precise::rsqrt",
            }
        )
    return identity


def layernorm_noaffine(x: mx.array, eps: float = 1e-6) -> mx.array:
    if _backend == DEFAULT_BACKEND:
        return _mlx_two_pass_layernorm(x, eps)
    _validate_input(x, affine=False)
    if _backend == CUDA_WELFORD_TURING_T4_BACKEND:
        if _turing_rsqrt_delta_lut is None:
            raise RuntimeError(f"{_backend} correction LUT is not configured")
        return _cuda_welford_turing_layernorm(x, _turing_rsqrt_delta_lut, eps)
    return _cuda_welford_layernorm(x, eps)


def layernorm_affine(
    x: mx.array,
    weight: mx.array,
    bias: mx.array,
    eps: float = 1e-6,
) -> mx.array:
    if _backend == DEFAULT_BACKEND:
        return mx.fast.layer_norm(x, weight, bias, eps).astype(x.dtype)
    _validate_input(x, affine=True)
    for name, parameter in (("weight", weight), ("bias", bias)):
        if parameter.dtype != mx.float32 or parameter.shape != (1536,):
            raise ValueError(
                f"{_backend} sparse affine LayerNorm requires float32 {name} "
                f"with shape (1536,), got {parameter.dtype} {parameter.shape}"
            )
    if _backend == CUDA_WELFORD_TURING_T4_BACKEND:
        if _turing_rsqrt_delta_lut is None:
            raise RuntimeError(f"{_backend} correction LUT is not configured")
        return _cuda_welford_turing_affine_layernorm(
            x, weight, bias, _turing_rsqrt_delta_lut, eps
        )
    return _cuda_welford_affine_layernorm(x, weight, bias, eps)


def layernorm_noaffine_float32_output(
    x: mx.array, eps: float = 1e-5
) -> mx.array:
    """Normalize a BF16 hidden state after the source's FP32 boundary."""
    if _backend == DEFAULT_BACKEND:
        return mx.fast.layer_norm(x.astype(mx.float32), None, None, eps)
    _validate_input(x, affine=False)
    if _backend == CUDA_WELFORD_TURING_T4_BACKEND:
        if _turing_rsqrt_delta_lut is None:
            raise RuntimeError(f"{_backend} correction LUT is not configured")
        return _cuda_welford_turing_layernorm_float32_output(
            x, _turing_rsqrt_delta_lut, eps
        )
    return _cuda_welford_layernorm_float32_output(x, eps)


def _validate_input(x: mx.array, *, affine: bool) -> None:
    if x.dtype != mx.bfloat16:
        kind = "affine " if affine else ""
        raise ValueError(
            f"{_backend} sparse {kind}LayerNorm requires bfloat16 input, got {x.dtype}"
        )
    width = x.shape[-1] if x.shape else None
    if width != 1536:
        raise ValueError(
            f"{_backend} sparse LayerNorm requires hidden width 1536, got {width}"
        )


def _validate_turing_lut(lut: mx.array) -> None:
    if lut.dtype != mx.int8 or lut.ndim != 1 or lut.size != TURING_RSQRT_LUT_SIZE:
        raise ValueError(
            "Turing-rsqrt correction LUT requires "
            f"{TURING_RSQRT_LUT_SIZE} int8 entries, got {lut.dtype} {lut.shape}"
        )
