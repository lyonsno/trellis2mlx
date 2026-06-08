"""Tests for mesh cleanup operations.

Tests use synthetic meshes with known defects to verify:
- Duplicate face removal
- Non-manifold edge repair
- Normal/winding unification
- Integration with existing keep_largest_component and fill_small_holes
"""

import numpy as np
import pytest

from trellmlx.mesh_cleanup import (
    cleanup_mesh,
    keep_largest_component,
    fill_small_holes,
    remove_duplicate_faces,
    repair_non_manifold_edges,
    fix_normals,
)


def _make_tetrahedron(offset=None):
    """A simple watertight tetrahedron."""
    verts = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, 1, 0],
        [0.5, 0.5, 1],
    ], dtype=np.float32)
    if offset is not None:
        verts += np.array(offset, dtype=np.float32)
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [1, 2, 3],
        [0, 2, 3],
    ], dtype=np.int64)
    return verts, faces


def _make_open_box():
    """A small box missing the top face — has one rectangular hole.

    Scaled to 0.005 so the hole perimeter (~0.02) is below the default
    max_hole_perimeter=3e-2, matching real decoder mesh scale in [-0.5, 0.5].
    """
    scale = 0.005
    verts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # top
    ], dtype=np.float32) * scale
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


class TestRemoveDuplicateFaces:
    def test_removes_exact_duplicates(self):
        verts, faces = _make_tetrahedron()
        # Add duplicate of face 0
        dup_faces = np.vstack([faces, faces[0:1]])
        assert len(dup_faces) == 5

        cleaned_v, cleaned_f = remove_duplicate_faces(verts, dup_faces)
        assert len(cleaned_f) == 4

    def test_removes_permuted_duplicates(self):
        """Same triangle with vertices in different order is still a duplicate."""
        verts, faces = _make_tetrahedron()
        # [0,1,2] and [1,2,0] are the same triangle
        permuted = np.array([[1, 2, 0]], dtype=np.int64)
        dup_faces = np.vstack([faces, permuted])
        assert len(dup_faces) == 5

        cleaned_v, cleaned_f = remove_duplicate_faces(verts, dup_faces)
        assert len(cleaned_f) == 4

    def test_noop_when_no_duplicates(self):
        verts, faces = _make_tetrahedron()
        cleaned_v, cleaned_f = remove_duplicate_faces(verts, faces)
        assert len(cleaned_f) == len(faces)

    def test_empty_mesh(self):
        verts = np.zeros((0, 3), dtype=np.float32)
        faces = np.zeros((0, 3), dtype=np.int64)
        v, f = remove_duplicate_faces(verts, faces)
        assert len(f) == 0


class TestRepairNonManifoldEdges:
    def test_removes_non_manifold_faces(self):
        """An edge shared by 3+ faces is non-manifold. Repair should reduce to 2."""
        verts, faces = _make_tetrahedron()
        # Add extra face sharing edge (0,1) — now edge (0,1) has 3 adjacent faces
        extra_vert = np.array([[0.5, -1, 0.5]], dtype=np.float32)
        verts = np.vstack([verts, extra_vert])
        extra_face = np.array([[0, 1, 4]], dtype=np.int64)
        faces = np.vstack([faces, extra_face])

        cleaned_v, cleaned_f = repair_non_manifold_edges(verts, faces)
        # The extra face (smallest area on edge 0,1) should be removed
        assert len(cleaned_f) == 4

    def test_noop_on_manifold_mesh(self):
        verts, faces = _make_tetrahedron()
        cleaned_v, cleaned_f = repair_non_manifold_edges(verts, faces)
        assert len(cleaned_f) == len(faces)


class TestFixNormals:
    def test_makes_winding_consistent(self):
        """Flip one face's winding and check fix_normals corrects it."""
        verts, faces = _make_tetrahedron()
        # Flip face 1's winding
        faces[1] = faces[1][::-1]

        fixed_v, fixed_f = fix_normals(verts, faces)
        # All faces should now have consistent winding
        # Check: for each shared edge, the two faces should traverse it in opposite directions
        edge_dirs = {}
        consistent = True
        for fi, face in enumerate(fixed_f):
            for i in range(3):
                e = (face[i], face[(i + 1) % 3])
                e_key = tuple(sorted(e))
                if e_key in edge_dirs:
                    # Consistent winding means shared edges go in opposite directions
                    prev_dir = edge_dirs[e_key]
                    if e == prev_dir:
                        consistent = False
                        break
                else:
                    edge_dirs[e_key] = e
        assert consistent

    def test_noop_on_consistent_mesh(self):
        verts, faces = _make_tetrahedron()
        fixed_v, fixed_f = fix_normals(verts, faces)
        assert len(fixed_f) == len(faces)

    def test_returns_plain_ndarray_not_tracked(self):
        """fix_normals must return plain ndarray, not trimesh TrackedArray."""
        verts, faces = _make_tetrahedron()
        fixed_v, fixed_f = fix_normals(verts, faces)
        assert type(fixed_v) is np.ndarray
        assert type(fixed_f) is np.ndarray


class TestFillSmallHolesPerimeter:
    def test_small_hole_filled_large_hole_skipped(self):
        """Perimeter threshold should fill small holes and skip large ones."""
        # Two open boxes at different scales — same topology, different perimeters.
        # Small box: scale=0.005, hole perimeter ≈ 4*0.005 = 0.02 (below 3e-2)
        # Large box: scale=0.05, hole perimeter ≈ 4*0.05 = 0.2 (above 3e-2)
        for scale, should_fill in [(0.005, True), (0.05, False)]:
            verts = np.array([
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            ], dtype=np.float32) * scale
            faces = np.array([
                [0, 1, 2], [0, 2, 3],  # bottom
                [0, 1, 5], [0, 5, 4],  # front
                [1, 2, 6], [1, 6, 5],  # right
                [2, 3, 7], [2, 7, 6],  # back
                [3, 0, 4], [3, 4, 7],  # left (top missing = hole)
            ], dtype=np.int64)

            orig_faces = len(faces)
            v_out, f_out = fill_small_holes(verts, faces, max_hole_perimeter=3e-2, verbose=False)

            if should_fill:
                assert len(f_out) > orig_faces, (
                    f"scale={scale}: expected hole to be filled, got {len(f_out)} faces (was {orig_faces})")
            else:
                assert len(f_out) == orig_faces, (
                    f"scale={scale}: expected hole to be skipped, got {len(f_out)} faces (was {orig_faces})")

    def test_fan_triangulation_adds_correct_faces(self):
        """Filling a 4-edge hole should add 4 fan triangles + 1 centroid vertex."""
        scale = 0.005  # small enough to be filled
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=np.float32) * scale
        faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ], dtype=np.int64)

        v_out, f_out = fill_small_holes(verts, faces, max_hole_perimeter=3e-2, verbose=False)
        # 4-edge hole → 4 fan triangles, 1 new centroid vertex
        assert len(f_out) == len(faces) + 4
        assert len(v_out) == len(verts) + 1


class TestCleanupMeshIntegration:
    def test_full_cleanup_with_floater_and_duplicate(self):
        """Mesh with a floater, a duplicate face, and a hole."""
        verts, faces = _make_open_box()
        # Add a small disconnected triangle (floater)
        floater_verts = np.array([
            [10, 10, 10], [10.1, 10, 10], [10, 10.1, 10],
        ], dtype=np.float32)
        floater_face = np.array([[8, 9, 10]], dtype=np.int64)
        verts = np.vstack([verts, floater_verts])
        faces = np.vstack([faces, floater_face])
        # Add a duplicate face
        faces = np.vstack([faces, faces[0:1]])

        cleaned_v, cleaned_f = cleanup_mesh(verts, faces, verbose=False)

        # Floater removed (1 face), duplicate removed (1 face), hole filled (adds faces).
        # 12 input → 10 after dedup+floater → 14 after hole fill (4 fan tris for top).
        # The key invariant: no floater vertices remain, no duplicate faces remain.
        assert len(cleaned_f) >= 10  # at least the original non-duplicate faces

        # Verify no floater vertices (all vertices should be within the box's range)
        assert cleaned_v[:, 0].max() <= 0.01  # floater was at x=10

    def test_already_clean_mesh(self):
        verts, faces = _make_tetrahedron()
        cleaned_v, cleaned_f = cleanup_mesh(verts, faces, verbose=False)
        # Tetrahedron is watertight, no floaters, no duplicates
        assert len(cleaned_f) == len(faces)
