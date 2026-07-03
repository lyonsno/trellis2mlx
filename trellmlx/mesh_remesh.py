"""Topology-rebuild mesh remeshing helpers."""

from __future__ import annotations

import numpy as np


def _apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    hom = np.concatenate(
        [np.asarray(points, dtype=np.float64), np.ones((len(points), 1), dtype=np.float64)],
        axis=1,
    )
    return (hom @ np.asarray(transform, dtype=np.float64).T)[:, :3]


def voxel_remesh(vertices, faces, *, pitch: float):
    """Rebuild mesh topology through a filled voxel grid and marching cubes.

    ``trimesh.VoxelGrid.marching_cubes`` returns vertices in voxel index space.
    The returned mesh must be transformed back through ``VoxelGrid.transform``
    before texture sampling, GLB export, or coordinate-domain witnesses can use
    it as a replacement for the input mesh.
    """
    if pitch <= 0:
        raise ValueError(f"pitch must be > 0, got {pitch!r}")

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        return vertices.copy(), faces.copy()

    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    voxel_grid = mesh.voxelized(pitch=float(pitch), method="subdivide")
    rebuilt = voxel_grid.fill().marching_cubes

    remesh_vertices = _apply_transform(np.asarray(rebuilt.vertices), voxel_grid.transform)
    remesh_faces = np.asarray(rebuilt.faces, dtype=np.int64)

    return (
        np.asarray(remesh_vertices, dtype=np.float32),
        np.asarray(remesh_faces, dtype=np.int64),
    )
