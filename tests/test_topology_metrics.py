"""Tests for topology metrics harness.

Fail-first coverage:
1. False-closure: missing output must not pretend success
2. False-closure: stale/default config must not replace requested config
3. Known geometry fixture with exact boundary loop count
4. GLB loading round-trip
5. Degenerate face detection
"""

import numpy as np
import pytest
import os
import tempfile

from trellmlx.topology_metrics import (
    TopologyReport,
    compute_topology_metrics,
    load_mesh_from_checkpoint,
    load_mesh_from_glb,
)


# === Geometry fixtures ===

def _make_single_triangle():
    """A single triangle — 3 boundary edges, 1 boundary loop, 1 component."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    return verts, faces


def _make_open_box():
    """An open box (5 quads = 10 triangles, missing one face).
    Has exactly 1 boundary loop around the open face.
    """
    verts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # top
    ], dtype=np.float32)
    faces = np.array([
        # bottom
        [0, 1, 2], [0, 2, 3],
        # front
        [0, 1, 5], [0, 5, 4],
        # right
        [1, 2, 6], [1, 6, 5],
        # back
        [2, 3, 7], [2, 7, 6],
        # left
        [3, 0, 4], [3, 4, 7],
        # top is MISSING — this is the hole
    ], dtype=np.int64)
    return verts, faces


def _make_two_disconnected_triangles():
    """Two separate triangles — 2 components, 6 boundary edges, 2 loops."""
    verts = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [5, 0, 0], [6, 0, 0], [5, 1, 0],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    return verts, faces


def _make_watertight_tetrahedron():
    """A closed tetrahedron — no boundary edges, no boundary loops."""
    verts = np.array([
        [0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0.5, 1],
    ], dtype=np.float32)
    faces = np.array([
        [0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3],
    ], dtype=np.int64)
    return verts, faces


# === Test: False-closure — missing output must not pretend success ===

class TestFalseClosureMissingOutput:
    def test_missing_checkpoint_raises(self):
        """load_mesh_from_checkpoint must raise on missing files, not return empty."""
        with pytest.raises(FileNotFoundError):
            load_mesh_from_checkpoint("/nonexistent/path/to/checkpoints")

    def test_missing_glb_raises(self):
        """load_mesh_from_glb must raise on missing file."""
        with pytest.raises(Exception):
            load_mesh_from_glb("/nonexistent/path/mesh.glb")

    def test_empty_mesh_reports_zero_not_success(self):
        """An empty mesh must report 0 faces/verts, not fake metrics."""
        verts = np.zeros((0, 3), dtype=np.float32)
        faces = np.zeros((0, 3), dtype=np.int64)
        report = compute_topology_metrics(verts, faces, route="empty")
        assert report.n_vertices == 0
        assert report.n_faces == 0
        assert report.n_boundary_edges == 0
        assert report.n_boundary_loops == 0


# === Test: False-closure — config identity ===

class TestFalseClosureConfig:
    def test_report_preserves_requested_route(self):
        """Report must echo the route the caller specified, not a default."""
        verts, faces = _make_single_triangle()
        report = compute_topology_metrics(
            verts, faces, route="my-custom-route",
            config={"target_faces": 350000, "texture_size": 4096},
        )
        assert report.route == "my-custom-route"
        assert report.config["target_faces"] == 350000
        assert report.config["texture_size"] == 4096

    def test_report_route_is_not_silently_defaulted(self):
        """If route is not passed, it must be empty string, not 'default'."""
        verts, faces = _make_single_triangle()
        report = compute_topology_metrics(verts, faces)
        assert report.route == ""
        assert report.config == {}


# === Test: Known geometry fixtures ===

class TestKnownGeometry:
    def test_single_triangle_has_one_boundary_loop(self):
        verts, faces = _make_single_triangle()
        report = compute_topology_metrics(verts, faces, route="fixture")
        assert report.n_boundary_edges == 3
        assert report.n_boundary_loops == 1
        assert report.boundary_loop_sizes == [3]
        assert report.n_connected_components == 1
        assert report.n_non_manifold_edges == 0
        assert report.n_degenerate_faces == 0

    def test_open_box_has_one_boundary_loop(self):
        verts, faces = _make_open_box()
        report = compute_topology_metrics(verts, faces, route="fixture")
        assert report.n_boundary_loops == 1
        assert report.n_boundary_edges == 4  # 4 edges around the open top
        assert report.n_connected_components == 1

    def test_two_disconnected_triangles(self):
        verts, faces = _make_two_disconnected_triangles()
        report = compute_topology_metrics(verts, faces, route="fixture")
        assert report.n_connected_components == 2
        assert report.n_boundary_loops == 2
        assert report.n_boundary_edges == 6

    def test_watertight_tetrahedron_has_no_boundary(self):
        verts, faces = _make_watertight_tetrahedron()
        report = compute_topology_metrics(verts, faces, route="fixture")
        assert report.n_boundary_edges == 0
        assert report.n_boundary_loops == 0
        assert report.n_connected_components == 1


# === Test: Degenerate face detection ===

class TestDegenerateFaces:
    def test_degenerate_collapsed_triangle(self):
        """A triangle with all 3 vertices at the same point is degenerate."""
        verts = np.array([
            [0, 0, 0], [0, 0, 0], [0, 0, 0],
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        report = compute_topology_metrics(verts, faces, route="degen-test")
        assert report.n_degenerate_faces == 1

    def test_collinear_triangle_is_degenerate(self):
        """A triangle with collinear vertices has zero area."""
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [2, 0, 0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        report = compute_topology_metrics(verts, faces, route="degen-test")
        assert report.n_degenerate_faces == 1


# === Test: GLB round-trip ===

class TestGLBRoundTrip:
    def test_glb_load_preserves_topology(self):
        """Export a mesh to GLB, reload, check metrics match."""
        import trimesh

        verts, faces = _make_open_box()
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
            mesh.export(f.name)
            glb_path = f.name

        try:
            loaded_verts, loaded_faces = load_mesh_from_glb(glb_path)
            report = compute_topology_metrics(
                loaded_verts, loaded_faces,
                source_path=glb_path, route="glb-roundtrip",
            )
            assert report.n_boundary_loops == 1
            assert report.n_connected_components == 1
        finally:
            os.unlink(glb_path)


# === Test: Report summary is populated ===

class TestReportSummary:
    def test_summary_includes_route(self):
        verts, faces = _make_single_triangle()
        report = compute_topology_metrics(
            verts, faces, route="test-route",
            source_path="/tmp/test.glb",
        )
        s = report.summary()
        assert "test-route" in s
        assert "/tmp/test.glb" in s
        assert "Boundary loops: 1" in s
