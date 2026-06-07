"""Sparse 3D submanifold convolution for MLX.

Implements the gather-scatter pattern: for each active voxel, gather
features from its 3x3x3 neighborhood (only where other active voxels
exist), multiply by kernel weights, and scatter-add results back.

This is the same algorithm as trellis-mac's conv_none.py but in MLX.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def build_neighbor_map(coords: mx.array, kernel_size: int = 3) -> tuple:
    """Build source/target index pairs for sparse convolution.

    For each active voxel, find which of its 3³=27 neighbors are also active.
    Returns (src_indices, tgt_indices, kernel_indices) as MLX arrays.

    Uses packed int64 dict lookup (same approach as our mesh extraction).

    Args:
        coords: [N, 4] as (batch_idx, z, y, x), int
        kernel_size: convolution kernel size (3)

    Returns:
        src_idx: [E] source voxel indices
        tgt_idx: [E] target voxel indices
        k_idx: [E] kernel position indices (0..26 for 3x3x3)
    """
    coords_np = np.array(coords)
    N = len(coords_np)
    K = kernel_size
    half = K // 2

    # Build spatial hash: packed int64 → voxel index
    packed = (
        coords_np[:, 0].astype(np.int64) << 48 |
        coords_np[:, 1].astype(np.int64) << 32 |
        coords_np[:, 2].astype(np.int64) << 16 |
        coords_np[:, 3].astype(np.int64)
    )
    coord_to_idx = {}
    for i in range(N):
        coord_to_idx[packed[i]] = i

    src_list = []
    tgt_list = []
    k_list = []

    for kz in range(K):
        for ky in range(K):
            for kx in range(K):
                oz = kz - half
                oy = ky - half
                ox = kx - half
                k_idx = kz * K * K + ky * K + kx

                # Vectorized: compute all neighbor coords at once
                neighbor_packed = (
                    coords_np[:, 0].astype(np.int64) << 48 |
                    (coords_np[:, 1] + oz).astype(np.int64) << 32 |
                    (coords_np[:, 2] + oy).astype(np.int64) << 16 |
                    (coords_np[:, 3] + ox).astype(np.int64)
                )

                for i in range(N):
                    j = coord_to_idx.get(neighbor_packed[i])
                    if j is not None:
                        src_list.append(j)
                        tgt_list.append(i)
                        k_list.append(k_idx)

    return (
        mx.array(np.array(src_list, dtype=np.int32)),
        mx.array(np.array(tgt_list, dtype=np.int32)),
        mx.array(np.array(k_list, dtype=np.int32)),
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

        # Reshape weight: [Co, K, K, K, Ci] → [K_total, Ci, Co]
        w = self.weight.reshape(Co, K_total, Ci)
        w = w.transpose(1, 2, 0)  # [K_total, Ci, Co]

        # Process per kernel position to keep memory bounded
        # and avoid scatter_add issues. 27 positions for 3x3x3.
        out = mx.zeros((N, Co))

        for k in range(K_total):
            # Find edges for this kernel position (on CPU to avoid Metal events)
            k_mask_np = np.array(k_idx) == k
            if not k_mask_np.any():
                continue
            s_np = np.array(src_idx)[k_mask_np]
            t_np = np.array(tgt_idx)[k_mask_np]

            src_f = feats[mx.array(s_np)]     # [E_k, Ci]
            edge_out = src_f @ w[k]            # [E_k, Co]
            out = out.at[mx.array(t_np)].add(edge_out)

        return out + self.bias
