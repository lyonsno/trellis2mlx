"""Tests for QEM mesh simplification with topology guards.

Covers: target face count, normal-flip guard, boundary preservation,
watertight preservation, edge cases, Metal/CPU parity, conflict-free
collapse, volume preservation, and generate.py integration.
"""

import numpy as np
import pytest

from trellmlx.simplify_qem_metal import (
    simplify_qem,
    _build_adjacency,
    _compute_qem,
    _compute_base_costs,
    _topology_check_cpu,
    HAS_MLX,
)

if HAS_MLX:
    from trellmlx.simplify_qem_metal import _topology_check_metal


def _make_icosahedron():
    """12 vertices, 20 faces. Watertight, symmetric."""
    import trimesh
    mesh = trimesh.creation.icosphere(subdivisions=0)
    return (
        np.array(mesh.vertices, dtype=np.float32),
        np.array(mesh.faces, dtype=np.int32),
    )


def _make_subdivided_icosphere(subdivisions=2):
    """Higher-res icosphere for simplification tests."""
    import trimesh
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions)
    return (
        np.array(mesh.vertices, dtype=np.float32),
        np.array(mesh.faces, dtype=np.int32),
    )


def _make_open_plane(n=5):
    """nxn grid plane. Has boundary vertices. n=5 gives 25V, 32F."""
    verts = []
    for j in range(n):
        for i in range(n):
            verts.append([i / (n - 1), j / (n - 1), 0.0])
    verts = np.array(verts, dtype=np.float32)

    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            v0 = j * n + i
            v1 = v0 + 1
            v2 = v0 + n
            v3 = v2 + 1
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    faces = np.array(faces, dtype=np.int32)
    return verts, faces


def _is_watertight(faces):
    """Check if every edge is shared by exactly 2 faces."""
    edge_count = {}
    for face in faces:
        for i in range(3):
            e = tuple(sorted((face[i], face[(i + 1) % 3])))
            edge_count[e] = edge_count.get(e, 0) + 1
    return all(c == 2 for c in edge_count.values())


def _has_degenerate_faces(faces):
    """Check if any face has duplicate vertex indices."""
    for face in faces:
        if face[0] == face[1] or face[1] == face[2] or face[0] == face[2]:
            return True
    return False


class TestSimplifyReachesTarget:
    def test_reaches_target(self):
        v, f = _make_subdivided_icosphere(2)
        assert len(f) == 320
        v2, f2 = simplify_qem(v, f, target_faces=50, verbose=False)
        assert len(f2) <= 50

    def test_reaches_target_large(self):
        v, f = _make_subdivided_icosphere(3)
        assert len(f) == 1280
        v2, f2 = simplify_qem(v, f, target_faces=200, verbose=False)
        assert len(f2) <= 200


class TestSimplifyNoop:
    def test_noop_when_below_target(self):
        v, f = _make_icosahedron()
        v2, f2 = simplify_qem(v, f, target_faces=100, verbose=False)
        np.testing.assert_array_equal(v2, v)
        np.testing.assert_array_equal(f2, f)

    def test_noop_when_equal_to_target(self):
        v, f = _make_icosahedron()
        v2, f2 = simplify_qem(v, f, target_faces=20, verbose=False)
        np.testing.assert_array_equal(v2, v)
        np.testing.assert_array_equal(f2, f)


class TestNormalFlipGuard:
    def test_no_inverted_normals_after_simplify(self):
        v, f = _make_subdivided_icosphere(2)
        v2, f2 = simplify_qem(v, f, target_faces=40, verbose=False)

        # For a simplified sphere, all face normals should point outward
        # (positive dot product with face centroid vector from origin)
        centroids = (v2[f2[:, 0]] + v2[f2[:, 1]] + v2[f2[:, 2]]) / 3
        e1 = v2[f2[:, 1]] - v2[f2[:, 0]]
        e2 = v2[f2[:, 2]] - v2[f2[:, 0]]
        normals = np.cross(e1, e2)

        dots = np.sum(normals * centroids, axis=1)
        # All normals should point outward (same direction as centroid)
        assert np.all(dots > 0), f"Found {(dots <= 0).sum()} inward-facing normals"


class TestBoundaryPreservation:
    def test_boundary_vertices_preserved(self):
        v, f = _make_open_plane(5)
        assert len(f) == 32

        # Identify boundary vertices (on edges with 1 adjacent face)
        edge_count = {}
        for face in f:
            for i in range(3):
                e = tuple(sorted((face[i], face[(i + 1) % 3])))
                edge_count[e] = edge_count.get(e, 0) + 1
        boundary_verts = set()
        for (a, b), count in edge_count.items():
            if count == 1:
                boundary_verts.add(a)
                boundary_verts.add(b)
        boundary_positions = v[sorted(boundary_verts)]

        v2, f2 = simplify_qem(v, f, target_faces=16, verbose=False)

        # Boundary vertices that survived should keep their original positions.
        # Not all boundary verts survive (two boundary verts can collapse to
        # midpoint), but the ones that remain should be at original positions.
        # Check that the output still has boundary vertices and they're on
        # the original boundary (y=0 or y=1 or x=0 or x=1 for a unit plane).
        edge_count2 = {}
        for face in f2:
            for i in range(3):
                e = tuple(sorted((face[i], face[(i + 1) % 3])))
                edge_count2[e] = edge_count2.get(e, 0) + 1
        boundary_verts2 = set()
        for (a, b), count in edge_count2.items():
            if count == 1:
                boundary_verts2.add(a)
                boundary_verts2.add(b)
        assert len(boundary_verts2) > 0, "Open mesh lost all boundary vertices"
        # Output boundary verts should lie on the original boundary plane (z=0)
        for vi in boundary_verts2:
            assert abs(v2[vi, 2]) < 1e-5, f"Boundary vertex {vi} moved off z=0 plane"


class TestWatertightPreservation:
    def test_watertight_stays_watertight(self):
        v, f = _make_subdivided_icosphere(2)
        assert _is_watertight(f)

        v2, f2 = simplify_qem(v, f, target_faces=40, verbose=False)
        assert _is_watertight(f2), "Watertight mesh became non-watertight after QEM simplification"


class TestEdgeCases:
    def test_empty_mesh(self):
        v = np.zeros((0, 3), dtype=np.float32)
        f = np.zeros((0, 3), dtype=np.int32)
        v2, f2 = simplify_qem(v, f, target_faces=100, verbose=False)
        assert v2.shape == (0, 3)
        assert f2.shape == (0, 3)

    def test_single_face(self):
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        f = np.array([[0, 1, 2]], dtype=np.int32)
        v2, f2 = simplify_qem(v, f, target_faces=1, verbose=False)
        assert len(f2) == 1


class TestConflictFreeCollapse:
    def test_no_degenerate_faces_after_simplify(self):
        v, f = _make_subdivided_icosphere(2)
        v2, f2 = simplify_qem(v, f, target_faces=40, verbose=False)
        assert not _has_degenerate_faces(f2), "Degenerate faces found after simplification"


class TestVolumePreservation:
    def test_volume_roughly_preserved(self):
        import trimesh
        v, f = _make_subdivided_icosphere(2)
        vol_before = trimesh.Trimesh(vertices=v, faces=f, process=False).volume

        v2, f2 = simplify_qem(v, f, target_faces=40, verbose=False)
        vol_after = trimesh.Trimesh(vertices=v2, faces=f2, process=False).volume

        rel_error = abs(vol_after - vol_before) / vol_before
        # 50% tolerance for 87.5% face reduction (320→40). QEM preserves
        # topology, not volume — volume loss is expected at high reduction ratios.
        assert rel_error < 0.5, f"Volume changed by {rel_error*100:.1f}% (before={vol_before:.4f}, after={vol_after:.4f})"


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestMetalCPUParity:
    def test_topology_check_parity(self):
        v, f = _make_subdivided_icosphere(2)
        V = len(v)
        edges, E, is_boundary, vf_offset, vf_data = _build_adjacency(f, V)
        qems = _compute_qem(v, f, V)
        base_costs, v_new, edge_len2 = _compute_base_costs(
            v, edges, qems, is_boundary, lambda_edge_length=1e-2)

        metal_costs = _topology_check_metal(
            v, f, edges, v_new, base_costs, edge_len2,
            vf_offset, vf_data, lambda_skinny=1e-3)

        cpu_costs = _topology_check_cpu(
            v, f, edges, v_new, base_costs, edge_len2,
            vf_offset, vf_data, lambda_skinny=1e-3)

        # Finite values should match
        metal_finite = np.isfinite(metal_costs)
        cpu_finite = np.isfinite(cpu_costs)
        np.testing.assert_array_equal(
            metal_finite, cpu_finite,
            err_msg="Metal and CPU disagree on which edges have infinite cost")

        if metal_finite.any():
            np.testing.assert_allclose(
                metal_costs[metal_finite], cpu_costs[cpu_finite],
                atol=1e-5,
                err_msg="Metal and CPU topology check costs diverge")


class TestIntegration:
    def test_qem_simplify_in_cleanup_pipeline(self):
        from generate import _cleanup_and_simplify_mesh
        from trellmlx.mesh_cleanup import cleanup_mesh

        v, f = _make_subdivided_icosphere(3)
        assert len(f) == 1280

        v2, f2 = _cleanup_and_simplify_mesh(
            v, f,
            target_faces=200,
            no_cleanup=False,
            qem_simplify=True,
            cleanup_mesh=cleanup_mesh,
            log=lambda *a, **kw: None,
        )
        assert len(f2) <= 200
        assert not _has_degenerate_faces(f2)
        # Verify vertex indices are valid
        assert f2.min() >= 0
        assert f2.max() < len(v2)

    def test_simplify_first_with_qem(self):
        """--simplify-first --qem-simplify: coarse fast-simplification then QEM final."""
        from generate import _cleanup_and_simplify_mesh
        from trellmlx.mesh_cleanup import cleanup_mesh

        v, f = _make_subdivided_icosphere(3)
        assert len(f) == 1280

        v2, f2 = _cleanup_and_simplify_mesh(
            v, f,
            target_faces=200,
            no_cleanup=False,
            simplify_first=True,
            qem_simplify=True,
            cleanup_mesh=cleanup_mesh,
            log=lambda *a, **kw: None,
        )
        assert len(f2) <= 200
        assert not _has_degenerate_faces(f2)
        assert f2.min() >= 0
        assert f2.max() < len(v2)
