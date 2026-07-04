"""Extract triangle mesh from decoder output.

Takes the ShapeSLatDecoder's 7-channel output and produces vertices + faces
via the flexible dual grid algorithm (same as TRELLIS.2's o_voxel.convert).

Channels:
  [0:3] vertex offsets → sigmoid → scaled to [-margin, 1+margin]
  [3:6] edge intersection flags → threshold > 0
  [6:7] quad split weight → softplus

All computation is numpy (CPU) — no GPU needed for mesh extraction.
"""

import numpy as np


# Static lookup: for each of the 3 edge directions, the 4 corner voxels
# that form a quad when the edge is intersected
_EDGE_NEIGHBOR_OFFSETS = np.array([
    [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],  # edge along x
    [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],  # edge along y
    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],  # edge along z
], dtype=np.int64)

_QUAD_SPLIT_1 = np.array([0, 1, 2, 0, 2, 3], dtype=np.int64)
_QUAD_SPLIT_2 = np.array([0, 1, 3, 3, 1, 2], dtype=np.int64)


def decoder_output_to_mesh(
    feats: np.ndarray,       # [N, 7] decoder output
    coords: np.ndarray,      # [N, 4] as (batch, z, y, x) int
    resolution: int = 256,
    voxel_margin: float = 0.5,
):
    """Convert decoder output to a triangle mesh.

    Args:
        feats: [N, 7] — the 7-channel decoder output
        coords: [N, 4] — voxel coordinates (batch_idx, z, y, x)
        resolution: grid resolution for coordinate scaling
        voxel_margin: margin for vertex offset scaling

    Returns:
        vertices: [V, 3] float32 mesh vertices
        faces: [F, 3] int64 triangle indices
    """
    # Unpack the 7 channels
    vertex_offsets = _sigmoid(feats[:, 0:3])  # [N, 3]
    vertex_offsets = (1 + 2 * voxel_margin) * vertex_offsets - voxel_margin
    intersection_flags = feats[:, 3:6] > 0   # [N, 3] bool
    quad_weights = _softplus(feats[:, 6:7])   # [N, 1]

    # Use spatial coordinates (drop batch dim)
    spatial_coords = coords[:, 1:4].astype(np.int64)  # [N, 3]

    return flexible_dual_grid_to_mesh(
        coords=spatial_coords,
        dual_vertices=vertex_offsets,
        intersected_flag=intersection_flags,
        split_weight=quad_weights,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        grid_size=resolution,
    )


def flexible_dual_grid_to_mesh(
    coords,           # [N, 3] int
    dual_vertices,    # [N, 3] float vertex offsets
    intersected_flag, # [N, 3] bool
    split_weight,     # [N, 1] or None
    aabb,             # [[min_xyz], [max_xyz]]
    grid_size=None,
    voxel_size=None,
):
    """Extract triangle mesh from sparse voxel dual-grid.

    Pure numpy implementation — same algorithm as trellis-mac's
    o_voxel_override_convert.py but without torch dependency.
    """
    coords = np.asarray(coords, dtype=np.int64)
    dual_vertices = np.asarray(dual_vertices, dtype=np.float32)
    intersected_flag = np.asarray(intersected_flag, dtype=bool)

    aabb = np.array(aabb, dtype=np.float32)

    if grid_size is not None:
        if isinstance(grid_size, (int, float)):
            grid_size = np.array([grid_size] * 3, dtype=np.float32)
        else:
            grid_size = np.array(grid_size, dtype=np.float32)
        voxel_size = (aabb[1] - aabb[0]) / grid_size
    elif voxel_size is not None:
        if isinstance(voxel_size, (int, float)):
            voxel_size = np.array([voxel_size] * 3, dtype=np.float32)
        else:
            voxel_size = np.array(voxel_size, dtype=np.float32)

    N = coords.shape[0]

    # Build spatial hash for neighbor lookup
    packed = (coords[:, 0] << 42) | (coords[:, 1] << 21) | coords[:, 2]
    coord_to_idx = {}
    for i in range(N):
        coord_to_idx[packed[i]] = i

    # Find connected voxels for each intersected edge
    # offsets: [3, 4, 3]
    edge_neighbors = coords[:, None, None, :] + _EDGE_NEIGHBOR_OFFSETS[None, :, :, :]
    # edge_neighbors: [N, 3, 4, 3]
    connected = edge_neighbors[intersected_flag]  # [M, 4, 3]
    M = connected.shape[0]

    if M == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64)

    # Look up neighbor indices
    flat = connected.reshape(-1, 3)
    flat_packed = (flat[:, 0] << 42) | (flat[:, 1] << 21) | flat[:, 2]
    MISSING = 0xFFFFFFFF
    indices = np.array([coord_to_idx.get(k, MISSING) for k in flat_packed], dtype=np.int64)
    indices = indices.reshape(M, 4)

    # Filter: only keep quads where all 4 corners exist
    valid = np.all(indices != MISSING, axis=1)
    quad_indices = indices[valid]
    L = quad_indices.shape[0]

    if L == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64)

    # Compute world-space vertex positions
    mesh_vertices = (coords.astype(np.float32) + dual_vertices) * voxel_size + aabb[0]

    # Triangulate quads
    if split_weight is None:
        # Use normal alignment heuristic
        a1 = quad_indices[:, _QUAD_SPLIT_1]
        n0 = np.cross(
            mesh_vertices[a1[:, 1]] - mesh_vertices[a1[:, 0]],
            mesh_vertices[a1[:, 2]] - mesh_vertices[a1[:, 0]],
        )
        n1 = np.cross(
            mesh_vertices[a1[:, 2]] - mesh_vertices[a1[:, 1]],
            mesh_vertices[a1[:, 3]] - mesh_vertices[a1[:, 1]],
        )
        align0 = np.abs(np.sum(n0 * n1, axis=1, keepdims=True))

        a2 = quad_indices[:, _QUAD_SPLIT_2]
        n0 = np.cross(
            mesh_vertices[a2[:, 1]] - mesh_vertices[a2[:, 0]],
            mesh_vertices[a2[:, 2]] - mesh_vertices[a2[:, 0]],
        )
        n1 = np.cross(
            mesh_vertices[a2[:, 2]] - mesh_vertices[a2[:, 1]],
            mesh_vertices[a2[:, 3]] - mesh_vertices[a2[:, 1]],
        )
        align1 = np.abs(np.sum(n0 * n1, axis=1, keepdims=True))

        mesh_triangles = np.where(align0 > align1, a1, a2).reshape(-1, 3)
    else:
        split_weight = np.asarray(split_weight, dtype=np.float32)
        sw = split_weight[quad_indices]  # [L, 4, 1]
        if sw.ndim == 3:
            sw = sw.squeeze(-1)  # [L, 4]
        sw_02 = sw[:, 0] * sw[:, 2]
        sw_13 = sw[:, 1] * sw[:, 3]
        cond = (sw_02 > sw_13)[:, None].repeat(6, axis=1)
        mesh_triangles = np.where(
            cond,
            quad_indices[:, _QUAD_SPLIT_1],
            quad_indices[:, _QUAD_SPLIT_2],
        ).reshape(-1, 3)

    return mesh_vertices, mesh_triangles


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def _softplus(x):
    return np.logaddexp(0, x)
