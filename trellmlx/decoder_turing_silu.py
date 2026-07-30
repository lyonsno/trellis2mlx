"""Bit-exact Tesla T4 FP16 SiLU replay for the shape decoder."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np


DEFAULT_BACKEND = "mlx-native"
CUDA_TURING_T4_LUT_BACKEND = "cuda-turing-t4-fp16-lut"
SUPPORTED_BACKENDS = (DEFAULT_BACKEND, CUDA_TURING_T4_LUT_BACKEND)
SILU_FP16_LUT_SIZE = 1 << 16

_backend = DEFAULT_BACKEND
_output_lut = None
_output_lut_artifact_path = None
_output_lut_artifact_sha256_attested = None
_output_lut_artifact_sha256_effective = None
_output_lut_content_sha256 = None
_lookup_kernel = None


def configure_decoder_silu_backend(
    name: str,
    *,
    output_lut_artifact_path: str | Path | None = None,
    output_lut_artifact_sha256_attested: str | None = None,
) -> None:
    global _backend
    global _output_lut
    global _output_lut_artifact_path
    global _output_lut_artifact_sha256_attested
    global _output_lut_artifact_sha256_effective
    global _output_lut_content_sha256
    if name not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"unsupported decoder SiLU backend {name!r}; "
            f"expected one of {SUPPORTED_BACKENDS}"
        )
    if name == CUDA_TURING_T4_LUT_BACKEND:
        if (
            output_lut_artifact_path is None
            or output_lut_artifact_sha256_attested is None
        ):
            raise ValueError(
                f"{name} requires an explicit output LUT artifact and SHA256"
            )
        digest = output_lut_artifact_sha256_attested
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(
                f"{name} requires a lowercase hexadecimal attested artifact SHA256"
            )
        artifact_path = Path(output_lut_artifact_path).resolve()
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"Turing decoder SiLU LUT artifact does not exist: {artifact_path}"
            )
        effective_digest = _sha256_file(artifact_path)
        if effective_digest != digest:
            raise ValueError(
                "Turing decoder SiLU artifact SHA256 mismatch: "
                f"attested={digest}, effective={effective_digest}"
            )
        output_lut = mx.array(_load_output_lut_artifact(artifact_path))
        _validate_output_lut(output_lut)
        _output_lut = output_lut
        _output_lut_artifact_path = str(artifact_path)
        _output_lut_artifact_sha256_attested = digest
        _output_lut_artifact_sha256_effective = effective_digest
        payload = np.ascontiguousarray(np.asarray(output_lut))
        _output_lut_content_sha256 = hashlib.sha256(
            payload.tobytes()
        ).hexdigest()
    else:
        if (
            output_lut_artifact_path is not None
            or output_lut_artifact_sha256_attested is not None
        ):
            raise ValueError(
                "Turing SiLU LUT state is only valid for "
                f"{CUDA_TURING_T4_LUT_BACKEND}"
            )
        _output_lut = None
        _output_lut_artifact_path = None
        _output_lut_artifact_sha256_attested = None
        _output_lut_artifact_sha256_effective = None
        _output_lut_content_sha256 = None
    _backend = name


def get_decoder_silu_backend() -> str:
    return _backend


def decoder_silu_backend_identity(name: str | None = None) -> dict[str, Any]:
    backend = _backend if name is None else name
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"unsupported decoder SiLU backend {backend!r}; "
            f"expected one of {SUPPORTED_BACKENDS}"
        )
    if backend == DEFAULT_BACKEND:
        return {
            "backend": backend,
            "algorithm": "mlx-nn-silu",
            "experimental": False,
        }
    if (
        _output_lut_artifact_sha256_attested is None
        or _output_lut_artifact_sha256_effective is None
        or _output_lut_artifact_path is None
        or _output_lut_content_sha256 is None
    ):
        raise ValueError(f"{backend} has no configured output LUT identity")
    return {
        "backend": backend,
        "algorithm": "exhaustive-fp16-bit-pattern-output-lookup",
        "experimental": True,
        "cuda_architecture": "sm_75",
        "cuda_device_anchor": "Tesla T4",
        "cuda_source_operation": "torch.nn.functional.silu",
        "cuda_source_version": "torch-2.10.0+cu128",
        "authenticated_contract": {
            "input_dtype": "float16",
            "output_dtype": "float16",
            "domain": "all-65536-bit-patterns",
        },
        "output_lut_artifact_sha256_attested": (
            _output_lut_artifact_sha256_attested
        ),
        "output_lut_artifact_sha256_effective": (
            _output_lut_artifact_sha256_effective
        ),
        "output_lut_artifact_path": _output_lut_artifact_path,
        "output_lut_content_sha256": _output_lut_content_sha256,
        "output_lut_entries": SILU_FP16_LUT_SIZE,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_output_lut_artifact(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as artifact:
        if set(artifact.files) != {"input_bits", "output_bits"}:
            raise ValueError(
                "Turing decoder SiLU LUT artifact requires exactly "
                "input_bits and output_bits"
            )
        input_bits = np.asarray(artifact["input_bits"])
        output_bits = np.asarray(artifact["output_bits"])
    expected_inputs = np.arange(SILU_FP16_LUT_SIZE, dtype=np.uint16)
    if not np.array_equal(input_bits, expected_inputs):
        raise ValueError(
            "Turing decoder SiLU LUT artifact input_bits are not exhaustive "
            "and ordered"
        )
    if (
        output_bits.dtype != np.uint16
        or output_bits.shape != (SILU_FP16_LUT_SIZE,)
    ):
        raise ValueError(
            "Turing decoder SiLU LUT artifact output_bits must be "
            f"uint16[{SILU_FP16_LUT_SIZE}]"
        )
    return np.ascontiguousarray(output_bits)


def silu(x: mx.array) -> mx.array:
    if _backend == DEFAULT_BACKEND:
        return nn.silu(x)
    if _output_lut is None:
        raise RuntimeError(f"{_backend} output LUT is not configured")
    return turing_silu_fp16(x, _output_lut)


def turing_silu_fp16(x: mx.array, output_lut: mx.array) -> mx.array:
    """Map FP16 values through the exhaustive T4 SiLU output table."""
    global _lookup_kernel
    if x.dtype != mx.float16:
        raise ValueError(f"Turing decoder SiLU requires float16 input, got {x.dtype}")
    _validate_output_lut(output_lut)
    if _lookup_kernel is None:
        _lookup_kernel = mx.fast.metal_kernel(
            name="decoder_turing_silu_f16_lut",
            input_names=["inp", "output_lut", "element_count"],
            output_names=["out"],
            source=r"""
                uint index = thread_position_in_grid.x;
                if (index >= element_count[0]) {
                    return;
                }
                ushort input_bits = as_type<ushort>(inp[index]);
                out[index] = as_type<half>(output_lut[input_bits]);
            """,
            ensure_row_contiguous=True,
        )
    element_count = x.size
    grid_size = ((element_count + 255) // 256) * 256
    return _lookup_kernel(
        inputs=[
            x,
            output_lut,
            mx.array([element_count], dtype=mx.uint32),
        ],
        grid=(grid_size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[mx.float16],
    )[0]


def _validate_output_lut(output_lut: mx.array) -> None:
    if output_lut.dtype != mx.uint16:
        raise ValueError(
            f"Turing decoder SiLU output LUT requires uint16, got {output_lut.dtype}"
        )
    if output_lut.ndim != 1 or output_lut.size != SILU_FP16_LUT_SIZE:
        raise ValueError(
            "Turing decoder SiLU output LUT requires "
            f"{SILU_FP16_LUT_SIZE} entries, got shape {output_lut.shape}"
        )
