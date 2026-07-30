"""Sparse 3D submanifold convolution for MLX.

Implements the gather-scatter pattern: for each active voxel, gather
features from its 3x3x3 neighborhood (only where other active voxels
exist), multiply by kernel weights, and scatter-add results back.

This is the same algorithm as trellis-mac's conv_none.py but in MLX.
"""

import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from trellmlx.turing_fda import turing_fda_matmul


SPARSE_CONV_MATMUL_BACKEND_ENV = (
    "TRELLIS2MLX_SPARSE_CONV_MATMUL_BACKEND"
)
SPARSE_CONV_MATMUL_BACKENDS = frozenset(("native", "turing_fda"))


def _pack_coords(coords_np: np.ndarray) -> np.ndarray:
    """Pack (batch, z, y, x) into int64 keys."""
    return (
        coords_np[:, 0].astype(np.int64) << 48 |
        coords_np[:, 1].astype(np.int64) << 32 |
        coords_np[:, 2].astype(np.int64) << 16 |
        coords_np[:, 3].astype(np.int64)
    )


# Module-level cache: coords content hash → neighbor map
_neighbor_map_cache: dict[int, tuple] = {}


def build_neighbor_map(coords: mx.array, kernel_size: int = 3) -> tuple:
    """Build source/target index pairs for sparse convolution.

    For each active voxel, find which of its 3³=27 neighbors are also active.
    Returns (src_indices, tgt_indices, kernel_indices) as MLX arrays.

    Results are cached by coords content — safe to call repeatedly with the
    same coords (e.g. across ODE steps or decoder blocks at the same level).

    Args:
        coords: [N, 4] as (batch_idx, z, y, x), int
        kernel_size: convolution kernel size (3)

    Returns:
        src_idx: [E] source voxel indices
        tgt_idx: [E] target voxel indices
        k_idx: [E] kernel position indices (0..26 for 3x3x3)
    """
    coords_np = np.array(coords)

    # Cache key: hash of coords content + shape + kernel_size
    cache_key = hash((coords_np.data.tobytes(), coords_np.shape, kernel_size))
    cached = _neighbor_map_cache.get(cache_key)
    if cached is not None:
        return cached

    result = _build_neighbor_map_impl(coords_np, kernel_size)
    _neighbor_map_cache[cache_key] = result
    return result


def clear_neighbor_map_cache():
    """Clear the neighbor map cache (call between generations)."""
    _neighbor_map_cache.clear()


def _build_neighbor_map_impl(coords_np: np.ndarray, kernel_size: int) -> tuple:
    """Build neighbor map — vectorized inner loop."""
    N = len(coords_np)
    K = kernel_size
    half = K // 2

    # Build spatial hash: packed int64 → voxel index
    packed = _pack_coords(coords_np)
    packed_set = set(packed.tolist())
    coord_lookup = {}
    for i in range(N):
        coord_lookup[packed[i]] = i

    src_parts = []
    tgt_parts = []
    k_parts = []

    for kz in range(K):
        for ky in range(K):
            for kx in range(K):
                oz, oy, ox = kz - half, ky - half, kx - half
                k_idx = kz * K * K + ky * K + kx

                # Vectorized neighbor lookup
                neighbor_packed = (
                    coords_np[:, 0].astype(np.int64) << 48 |
                    (coords_np[:, 1] + oz).astype(np.int64) << 32 |
                    (coords_np[:, 2] + oy).astype(np.int64) << 16 |
                    (coords_np[:, 3] + ox).astype(np.int64)
                )

                # Vectorized set membership check
                mask = np.array([p in packed_set for p in neighbor_packed.tolist()], dtype=bool)
                if not mask.any():
                    continue

                tgt_indices = np.where(mask)[0].astype(np.int32)
                src_indices = np.array(
                    [coord_lookup[neighbor_packed[i]] for i in tgt_indices],
                    dtype=np.int32,
                )

                src_parts.append(src_indices)
                tgt_parts.append(tgt_indices)
                k_parts.append(np.full(len(tgt_indices), k_idx, dtype=np.int32))

    if not src_parts:
        return (
            mx.array(np.zeros(0, dtype=np.int32)),
            mx.array(np.zeros(0, dtype=np.int32)),
            mx.array(np.zeros(0, dtype=np.int32)),
        )

    return (
        mx.array(np.concatenate(src_parts)),
        mx.array(np.concatenate(tgt_parts)),
        mx.array(np.concatenate(k_parts)),
    )


class SparseConv3d(nn.Module):
    """Submanifold sparse 3D convolution.

    Weight layout matches TRELLIS.2 checkpoints: [out_C, kD, kH, kW, in_C]
    (flex_gemm convention, NOT PyTorch's [out_C, in_C, kD, kH, kW]).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        K = kernel_size

        scale = (2.0 / (in_channels * K * K * K)) ** 0.5
        # Weight: [out_C, kD, kH, kW, in_C] — matches checkpoint layout
        self.weight = mx.random.normal((out_channels, K, K, K, in_channels)) * scale
        self.bias = mx.zeros((out_channels,))

    def __call__(self, feats: mx.array, neighbor_map: tuple) -> mx.array:
        """
        Args:
            feats: [N, in_C] features at active voxels
            neighbor_map: (src_idx, tgt_idx, k_idx) from build_neighbor_map

        Returns:
            [N, out_C] convolved features
        """
        src_idx, tgt_idx, k_idx = neighbor_map
        N = feats.shape[0]
        Co = self.out_channels
        Ci = self.in_channels
        K_total = self.kernel_size ** 3
        matmul_backend = os.environ.get(
            SPARSE_CONV_MATMUL_BACKEND_ENV,
            "native",
        ).lower()
        if matmul_backend not in SPARSE_CONV_MATMUL_BACKENDS:
            raise ValueError(
                "sparse-convolution matmul backend must be one of "
                f"{sorted(SPARSE_CONV_MATMUL_BACKENDS)}, "
                f"got {matmul_backend!r}"
            )

        # Reshape weight: [Co, K, K, K, Ci] → [K_total, Ci, Co]
        w = self.weight.reshape(Co, K_total, Ci)
        w = w.transpose(1, 2, 0)  # [K_total, Ci, Co]

        # Process per kernel position to keep memory bounded
        # and avoid scatter_add issues. 27 positions for 3x3x3.
        out = mx.zeros((N, Co), dtype=feats.dtype)

        for k in range(K_total):
            # Find edges for this kernel position (on CPU to avoid Metal events)
            k_mask_np = np.array(k_idx) == k
            if not k_mask_np.any():
                continue
            s_np = np.array(src_idx)[k_mask_np]
            t_np = np.array(tgt_idx)[k_mask_np]

            src_f = feats[mx.array(s_np)]     # [E_k, Ci]
            if matmul_backend == "turing_fda":
                edge_out = turing_fda_matmul(src_f, w[k]).astype(feats.dtype)
            else:
                edge_out = src_f @ w[k]        # [E_k, Co]
            out = out.at[mx.array(t_np)].add(edge_out)

        return out + self.bias
