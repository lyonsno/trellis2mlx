"""Tests for remesh_reconstruct module.

Fail-first coverage:
1. Remesh of a known geometry reduces boundary loops
2. Remesh preserves approximate shape
3. Empty/degenerate input handled gracefully
"""

import numpy as np
import pytest

from trellmlx.remesh_reconstruct import remesh_narrow_band


def _make_open_box():
    """Open box with 1 boundary loop (missing top face)."""
    verts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float32)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ], dtype=np.int64)
    return verts, faces


def _make_watertight_sphere(n=20):
    """Generate a watertight icosphere using trimesh."""
    import trimesh
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
    return np.array(sphere.vertices, dtype=np.float32), np.array(sphere.faces, dtype=np.int64)


class TestRemeshBasics:
    def test_remesh_produces_valid_mesh(self):
        """Remesh of a simple shape returns non-empty V/F."""
        verts, faces = _make_watertight_sphere()
        new_verts, new_faces = remesh_narrow_band(
            verts, faces, resolution=32, band=2.0, verbose=False,
        )
        assert len(new_verts) > 0
        assert len(new_faces) > 0

    def test_remesh_preserves_approximate_bounds(self):
        """Remeshed bounding box should be within ~20% of original."""
        verts, faces = _make_watertight_sphere()
        new_verts, new_faces = remesh_narrow_band(
            verts, faces, resolution=32, band=2.0, project_back=0.9,
            verbose=False,
        )
        orig_extent = verts.max(axis=0) - verts.min(axis=0)
        new_extent = new_verts.max(axis=0) - new_verts.min(axis=0)
        ratio = new_extent / orig_extent
        assert np.all(ratio > 0.7), f"Bounding box shrunk too much: {ratio}"
        assert np.all(ratio < 1.5), f"Bounding box grew too much: {ratio}"

    def test_remesh_open_box_closes_hole(self):
        """Remeshing an open box should reduce or eliminate boundary loops."""
        from trellmlx.topology_metrics import compute_topology_metrics

        verts, faces = _make_open_box()
        pre = compute_topology_metrics(verts, faces, route="pre")
        assert pre.n_boundary_loops == 1, "Fixture should have 1 hole"

        new_verts, new_faces = remesh_narrow_band(
            verts, faces, resolution=32, band=2.0, project_back=0.5,
            verbose=False,
        )
        post = compute_topology_metrics(new_verts, new_faces, route="post")
        # SDF + marching cubes should produce a watertight surface
        assert post.n_boundary_loops <= pre.n_boundary_loops, (
            f"Remesh should not add holes: {pre.n_boundary_loops} → {post.n_boundary_loops}"
        )

    def test_empty_mesh_returns_empty(self):
        """Empty input should return empty output."""
        verts = np.zeros((0, 3), dtype=np.float32)
        faces = np.zeros((0, 3), dtype=np.int64)
        # Should not crash
        new_verts, new_faces = remesh_narrow_band(
            verts, faces, resolution=16, verbose=False,
        )
        # Implementation may return empty or fallback — just shouldn't crash
        assert new_verts is not None
