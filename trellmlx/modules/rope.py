"""3D Rotary Position Embedding for TRELLIS.2.

Precomputes rotation phases from a 3D coordinate grid, then applies
them as complex rotations to query/key vectors during attention.
"""

import mlx.core as mx
import numpy as np


MLX_REAL_BACKEND = "mlx-real"
SOURCE_COMPLEX_BACKEND = "source-complex"
CUDA_POLAR_TURING_T4_BACKEND = "cuda-polar-turing-t4"
SUPPORTED_ROPE_BACKENDS = (
    MLX_REAL_BACKEND,
    SOURCE_COMPLEX_BACKEND,
    CUDA_POLAR_TURING_T4_BACKEND,
)

_backend = MLX_REAL_BACKEND
_turing_phase_lut = None
_turing_phase_lut_sha256 = None


def configure_rope_backend(
    name: str,
    *,
    turing_phase_lut: mx.array | None = None,
    turing_phase_lut_sha256: str | None = None,
) -> None:
    global _backend, _turing_phase_lut, _turing_phase_lut_sha256
    if name not in SUPPORTED_ROPE_BACKENDS:
        raise ValueError(
            f"unsupported RoPE backend {name!r}; "
            f"expected one of {SUPPORTED_ROPE_BACKENDS}"
        )
    if name == CUDA_POLAR_TURING_T4_BACKEND:
        if turing_phase_lut is None or turing_phase_lut_sha256 is None:
            raise ValueError(
                f"{name} requires an explicit phase LUT and SHA256"
            )
        _validate_turing_phase_lut(turing_phase_lut)
        if (
            len(turing_phase_lut_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in turing_phase_lut_sha256
            )
        ):
            raise ValueError(
                f"{name} requires a lowercase hexadecimal LUT SHA256"
            )
        _turing_phase_lut = turing_phase_lut
        _turing_phase_lut_sha256 = turing_phase_lut_sha256
    else:
        if turing_phase_lut is not None or turing_phase_lut_sha256 is not None:
            raise ValueError(
                "Turing phase state is only valid for "
                f"{CUDA_POLAR_TURING_T4_BACKEND}"
            )
        _turing_phase_lut = None
        _turing_phase_lut_sha256 = None
    _backend = name


def get_rope_backend() -> str:
    return _backend


def get_turing_phase_lut_sha256() -> str | None:
    return _turing_phase_lut_sha256


def rope_backend_identity() -> dict:
    if _backend == MLX_REAL_BACKEND:
        return {
            "backend": _backend,
            "phase_algorithm": "existing-model-phase-generation",
            "rotation_algorithm": "mlx-separate-real-multiply",
            "experimental": False,
        }
    if _backend == SOURCE_COMPLEX_BACKEND:
        return {
            "backend": _backend,
            "phase_algorithm": "existing-model-phase-generation",
            "rotation_algorithm": "mlx-complex64-multiply",
            "experimental": True,
        }
    if _turing_phase_lut_sha256 is None:
        raise RuntimeError(f"{_backend} has no configured phase LUT identity")
    return {
        "backend": _backend,
        "phase_algorithm": "torch-polar-turing-t4-float32-lut",
        "rotation_algorithm": "mlx-complex64-multiply",
        "experimental": True,
        "cuda_device_anchor": "Tesla T4",
        "cuda_source_tag": "pytorch-v2.10.0",
        "phase_lut_sha256": _turing_phase_lut_sha256,
        "phase_lut_shape": [64, 21, 2],
    }


def _validate_turing_phase_lut(phase_lut: mx.array) -> None:
    if phase_lut.dtype != mx.float32 or phase_lut.shape != (64, 21, 2):
        raise ValueError(
            "Turing RoPE phase LUT must be float32[64,21,2], "
            f"got {phase_lut.dtype}{phase_lut.shape}"
        )
    finite = mx.all(mx.isfinite(phase_lut))
    mx.eval(finite)
    if not finite.item():
        raise ValueError("Turing RoPE phase LUT contains non-finite values")


def build_sparse_rope_phases(
    coords: mx.array,
    *,
    head_dim: int,
) -> mx.array:
    """Build source-ordered 3D phases for sparse spatial coordinates."""
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"sparse RoPE coordinates must have shape [N,3], got {coords.shape}"
        )
    freq_dim = head_dim // 2 // 3
    target = head_dim // 2

    if _backend == CUDA_POLAR_TURING_T4_BACKEND:
        if _turing_phase_lut is None:
            raise RuntimeError(f"{_backend} phase LUT is not configured")
        if head_dim != 128 or freq_dim != 21:
            raise ValueError(
                f"{_backend} requires head dimension 128, got {head_dim}"
            )
        coords_np = np.array(coords)
        if (
            not np.issubdtype(coords_np.dtype, np.integer)
            or np.any(coords_np < 0)
            or np.any(coords_np > 63)
        ):
            raise ValueError(
                f"{_backend} requires integer coordinates in 0..63"
            )
        spatial = [
            _turing_phase_lut[coords[:, dimension]]
            for dimension in range(3)
        ]
        phases = mx.concatenate(spatial, axis=1)
        identity = mx.broadcast_to(
            mx.array([1.0, 0.0], dtype=mx.float32),
            (coords.shape[0], target - phases.shape[1], 2),
        )
        return mx.concatenate([phases, identity], axis=1)

    freqs = np.arange(freq_dim, dtype=np.float32) / freq_dim
    freqs = 1.0 / (10000.0 ** freqs)
    freqs = mx.array(freqs)
    coords_f = coords.astype(mx.float32)
    angles = mx.concatenate(
        [
            coords_f[:, dimension : dimension + 1] * freqs[None, :]
            for dimension in range(3)
        ],
        axis=-1,
    )
    if angles.shape[-1] < target:
        angles = mx.concatenate(
            [
                angles,
                mx.zeros(
                    (angles.shape[0], target - angles.shape[-1]),
                    dtype=mx.float32,
                ),
            ],
            axis=-1,
        )
    return mx.stack([mx.cos(angles), mx.sin(angles)], axis=-1)


def build_rope_phases(
    resolution: int,
    head_dim: int,
    dim: int = 3,
    rope_freq: tuple = (1.0, 10000.0),
) -> mx.array:
    """Precompute RoPE phases for a dense 3D grid.

    Args:
        resolution: Grid resolution (R). Grid is R×R×R.
        head_dim: Attention head dimension.
        rope_freq: (base, scale) for frequency computation.

    Returns:
        Complex rotation phases [R³, head_dim//2] as real pairs [R³, head_dim//2, 2].
    """
    if _backend == CUDA_POLAR_TURING_T4_BACKEND:
        if dim != 3 or rope_freq != (1.0, 10000.0):
            raise ValueError(
                f"{_backend} only authenticates 3D RoPE with frequency "
                "parameters (1.0, 10000.0)"
            )
        coords = np.stack(
            np.meshgrid(
                np.arange(resolution, dtype=np.int32),
                np.arange(resolution, dtype=np.int32),
                np.arange(resolution, dtype=np.int32),
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 3)
        return build_sparse_rope_phases(
            mx.array(coords),
            head_dim=head_dim,
        )

    freq_dim = head_dim // 2 // dim
    freqs = np.arange(freq_dim, dtype=np.float32) / freq_dim
    freqs = rope_freq[0] / (rope_freq[1] ** freqs)

    # Build 3D coordinate meshgrid [R, R, R, 3]
    coords = np.stack(np.meshgrid(
        np.arange(resolution),
        np.arange(resolution),
        np.arange(resolution),
        indexing='ij',
    ), axis=-1).reshape(-1, 3)  # [R³, 3]

    # Compute phases for each spatial dimension
    # For each coordinate dimension, compute outer product with frequencies
    # Then concatenate across dimensions
    all_phases = []
    for d in range(dim):
        angles = np.outer(coords[:, d].astype(np.float32), freqs)  # [R³, freq_dim]
        all_phases.append(angles)
    angles = np.concatenate(all_phases, axis=-1)  # [R³, dim * freq_dim]

    # Pad if needed (head_dim//2 might be larger than dim * freq_dim)
    target_len = head_dim // 2
    if angles.shape[-1] < target_len:
        pad_n = target_len - angles.shape[-1]
        angles = np.concatenate([
            angles,
            np.zeros((angles.shape[0], pad_n), dtype=np.float32)
        ], axis=-1)

    # Store as cos/sin pairs: [R³, head_dim//2, 2]
    cos_phases = np.cos(angles).astype(np.float32)
    sin_phases = np.sin(angles).astype(np.float32)
    phases = np.stack([cos_phases, sin_phases], axis=-1)  # [R³, head_dim//2, 2]

    return mx.array(phases)


def apply_rope(x: mx.array, phases: mx.array) -> mx.array:
    """Apply rotary position embedding to query or key vectors.

    Args:
        x: [T, H, D] where D = head_dim
        phases: [T, D//2, 2] precomputed cos/sin pairs

    Returns:
        [T, H, D] with RoPE applied
    """
    orig_dtype = x.dtype
    T, H, D = x.shape
    half = D // 2

    # Split into pairs for rotation
    x_pairs = x.reshape(T, H, half, 2)  # [T, H, D//2, 2]

    if _backend != MLX_REAL_BACKEND:
        return apply_rope_source_complex(x, phases)

    # phases: [T, D//2, 2] → broadcast to [T, 1, D//2, 2]
    cos_p = phases[:, :, 0:1]  # [T, D//2, 1]
    sin_p = phases[:, :, 1:2]  # [T, D//2, 1]

    # Expand for heads: [T, 1, D//2, 1]
    cos_p = cos_p[:, None, :, :]  # [T, 1, D//2, 1]
    sin_p = sin_p[:, None, :, :]

    x0 = x_pairs[..., 0:1]  # [T, H, D//2, 1]
    x1 = x_pairs[..., 1:2]

    # Complex rotation: (x0 + ix1) * (cos + i*sin) = (x0*cos - x1*sin) + i(x0*sin + x1*cos)
    out0 = x0 * cos_p - x1 * sin_p
    out1 = x0 * sin_p + x1 * cos_p

    out = mx.concatenate([out0, out1], axis=-1)  # [T, H, D//2, 2]
    return out.reshape(T, H, D).astype(orig_dtype)


def apply_rope_source_complex(x: mx.array, phases: mx.array) -> mx.array:
    """Apply the source complex64 multiply independent of global routing."""
    orig_dtype = x.dtype
    T, H, D = x.shape
    x_pairs = x.reshape(T, H, D // 2, 2).astype(mx.float32)
    imaginary = mx.array(1j, dtype=mx.complex64)
    x_complex = (
        x_pairs[..., 0].astype(mx.complex64)
        + imaginary * x_pairs[..., 1].astype(mx.complex64)
    )
    phase_complex = (
        phases[..., 0].astype(mx.complex64)
        + imaginary * phases[..., 1].astype(mx.complex64)
    )
    rotated = x_complex * phase_complex[:, None, :]
    out = mx.stack([mx.real(rotated), mx.imag(rotated)], axis=-1)
    return out.reshape(T, H, D).astype(orig_dtype)
