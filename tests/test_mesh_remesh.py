"""Tests for topology-rebuild remesh helpers."""

import numpy as np
import trimesh

from trellmlx.mesh_remesh import voxel_remesh


def test_voxel_remesh_preserves_world_coordinate_domain():
    mesh = trimesh.creation.box(extents=(0.4, 0.3, 0.2))
    mesh.apply_translation((0.1, -0.05, 0.2))
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    remesh_vertices, remesh_faces = voxel_remesh(vertices, faces, pitch=1.0 / 64.0)

    assert remesh_faces.shape[1] == 3
    assert remesh_vertices.dtype == np.float32
    assert remesh_faces.dtype == np.int64

    original_bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
    remesh_bounds = np.array([remesh_vertices.min(axis=0), remesh_vertices.max(axis=0)])
    margin = 2.0 / 64.0

    assert np.all(remesh_bounds[0] >= original_bounds[0] - margin)
    assert np.all(remesh_bounds[1] <= original_bounds[1] + margin)
    assert np.all(remesh_vertices >= -0.5 - margin)
    assert np.all(remesh_vertices <= 0.5 + margin)
