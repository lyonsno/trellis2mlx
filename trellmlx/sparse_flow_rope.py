"""Model-specific RoPE routing for the dense sparse-structure flow."""

from __future__ import annotations

import hashlib
from typing import Any

import mlx.core as mx
import numpy as np

from .modules.rope import (
    apply_rope,
    apply_rope_source_complex,
    build_rope_phases,
)


DEFAULT_BACKEND = "inherit"
SOURCE_CPU_POLAR_TORCH_2_10_BACKEND = "source-cpu-polar-torch-2.10"
SUPPORTED_BACKENDS = (
    DEFAULT_BACKEND,
    SOURCE_CPU_POLAR_TORCH_2_10_BACKEND,
)

_backend = DEFAULT_BACKEND
_phase_lut: mx.array | None = None
_phase_lut_artifact_sha256_attested: str | None = None
_phase_lut_content_sha256: str | None = None


def configure_sparse_flow_rope_backend(
    name: str,
    *,
    phase_lut: mx.array | None = None,
    phase_lut_artifact_sha256_attested: str | None = None,
) -> None:
    global _backend
    global _phase_lut
    global _phase_lut_artifact_sha256_attested
    global _phase_lut_content_sha256

    if name not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"unsupported sparse-flow RoPE backend {name!r}; "
            f"expected one of {SUPPORTED_BACKENDS}"
        )
    if name == SOURCE_CPU_POLAR_TORCH_2_10_BACKEND:
        if phase_lut is None or phase_lut_artifact_sha256_attested is None:
            raise ValueError(f"{name} requires an explicit phase LUT and SHA256")
        _validate_phase_lut(phase_lut)
        digest = phase_lut_artifact_sha256_attested
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(
                f"{name} requires a lowercase hexadecimal attested artifact SHA256"
            )
        payload = np.ascontiguousarray(np.asarray(phase_lut))
        _phase_lut = phase_lut
        _phase_lut_artifact_sha256_attested = digest
        _phase_lut_content_sha256 = hashlib.sha256(payload.tobytes()).hexdigest()
    else:
        if phase_lut is not None or phase_lut_artifact_sha256_attested is not None:
            raise ValueError(
                "source CPU phase state is only valid for "
                f"{SOURCE_CPU_POLAR_TORCH_2_10_BACKEND}"
            )
        _phase_lut = None
        _phase_lut_artifact_sha256_attested = None
        _phase_lut_content_sha256 = None
    _backend = name


def get_sparse_flow_rope_backend() -> str:
    return _backend


def get_sparse_flow_rope_phase_lut_artifact_sha256_attested() -> str | None:
    return _phase_lut_artifact_sha256_attested


def get_sparse_flow_rope_phase_lut_content_sha256() -> str | None:
    return _phase_lut_content_sha256


def sparse_flow_rope_backend_identity() -> dict[str, Any]:
    if _backend == DEFAULT_BACKEND:
        return {
            "backend": _backend,
            "scope": "sparse-structure-flow",
            "phase_algorithm": "inherited-global-rope-route",
            "rotation_algorithm": "inherited-global-rope-route",
            "experimental": False,
        }
    if (
        _phase_lut_artifact_sha256_attested is None
        or _phase_lut_content_sha256 is None
    ):
        raise RuntimeError(f"{_backend} has no configured phase LUT identity")
    return {
        "backend": _backend,
        "scope": "sparse-structure-flow",
        "phase_algorithm": "torch-polar-cpu-float32-lut",
        "rotation_algorithm": "mlx-complex64-multiply",
        "source_runtime": "torch-2.10.0+cu128",
        "source_phase_device": "cpu",
        "source_execution_device": "Tesla T4",
        "experimental": True,
        "phase_lut_artifact_sha256_attested": (
            _phase_lut_artifact_sha256_attested
        ),
        "phase_lut_content_sha256": _phase_lut_content_sha256,
        "phase_lut_shape": [64, 21, 2],
    }


def build_sparse_flow_rope_phases(
    resolution: int,
    head_dim: int,
    dim: int = 3,
    rope_freq: tuple = (1.0, 10000.0),
) -> mx.array:
    if _backend == DEFAULT_BACKEND:
        return build_rope_phases(resolution, head_dim, dim, rope_freq)
    if dim != 3 or rope_freq != (1.0, 10000.0):
        raise ValueError(
            f"{_backend} only authenticates 3D RoPE with frequency "
            "parameters (1.0, 10000.0)"
        )
    if head_dim != 128:
        raise ValueError(f"{_backend} requires head dimension 128, got {head_dim}")
    if resolution <= 0 or resolution > 64:
        raise ValueError(f"{_backend} requires resolution in 1..64")
    if _phase_lut is None:
        raise RuntimeError(f"{_backend} phase LUT is not configured")

    coords = np.stack(
        np.meshgrid(
            np.arange(resolution, dtype=np.int32),
            np.arange(resolution, dtype=np.int32),
            np.arange(resolution, dtype=np.int32),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    coords_mx = mx.array(coords)
    spatial = [
        _phase_lut[coords_mx[:, dimension]] for dimension in range(3)
    ]
    phases = mx.concatenate(spatial, axis=1)
    identity = mx.broadcast_to(
        mx.array([1.0, 0.0], dtype=mx.float32),
        (coords.shape[0], head_dim // 2 - phases.shape[1], 2),
    )
    return mx.concatenate([phases, identity], axis=1)


def apply_sparse_flow_rope(x: mx.array, phases: mx.array) -> mx.array:
    if _backend == DEFAULT_BACKEND:
        return apply_rope(x, phases)
    return apply_rope_source_complex(x, phases)


def _validate_phase_lut(phase_lut: mx.array) -> None:
    if phase_lut.dtype != mx.float32 or phase_lut.shape != (64, 21, 2):
        raise ValueError(
            "source CPU RoPE phase LUT must be float32[64,21,2], "
            f"got {phase_lut.dtype}{phase_lut.shape}"
        )
    finite = mx.all(mx.isfinite(phase_lut))
    mx.eval(finite)
    if not finite.item():
        raise ValueError("source CPU RoPE phase LUT contains non-finite values")
