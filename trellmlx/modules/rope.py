"""3D Rotary Position Embedding for TRELLIS.2.

Precomputes rotation phases from a 3D coordinate grid, then applies
them as complex rotations to query/key vectors during attention.
"""

import mlx.core as mx
import numpy as np


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
    T, H, D = x.shape
    half = D // 2

    # Split into pairs for rotation
    x_pairs = x.reshape(T, H, half, 2)  # [T, H, D//2, 2]

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
    return out.reshape(T, H, D)
