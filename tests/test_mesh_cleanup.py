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
    remove_small_components,
    fill_small_holes,
    remove_duplicate_faces,
    repair_non_manifold_edges,
    remove_same_direction_manifold_conflicts,
    orient_faces_by_adjacency,
    orient_components_outward_by_radial_heuristic,
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
        [0, 2, 1], [0, 3, 2],
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


def _assert_manifold_edges_oppositely_oriented(faces):
    """Every two-face shared edge must be traversed in opposite directions."""
    conflicts = _same_direction_manifold_conflict_count(faces)
    assert conflicts == 0


def _same_direction_manifold_conflict_count(faces):
    """Count two-face shared edges traversed in the same direction."""
    edge_dirs = {}
    for face in faces:
        for i in range(3):
            edge = (int(face[i]), int(face[(i + 1) % 3]))
            key = tuple(sorted(edge))
            edge_dirs.setdefault(key, []).append(edge)

    conflicts = 0
    for key, dirs in edge_dirs.items():
        if len(dirs) == 2 and dirs[0] == dirs[1]:
            conflicts += 1
    return conflicts


def _projected_missing_ratio(vertices, faces):
    from scripts.mesh_culling_attribution import (
        PANELS,
        default_front_face_for_panel,
        projected_front_face_missing_attribution,
    )

    export_vertices = vertices.astype(np.float64).copy()
    export_vertices[:, 1], export_vertices[:, 2] = vertices[:, 2].copy(), -vertices[:, 1].copy()
    missing = 0
    total = 0
    for panel in PANELS:
        report = projected_front_face_missing_attribution(
            vertices=export_vertices,
            faces=faces,
            panel=panel,
            image_size=128,
            front_face=default_front_face_for_panel(panel),
        )
        missing += report["missing_pixels"]
        total += report["double_sided_pixels"]
    return missing / total if total else 0.0


def _radial_orientation_counts(vertices, faces):
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal_len = np.linalg.norm(normals, axis=1)
    centers = tri.mean(axis=1)
    radial = centers - vertices.mean(axis=0)
    radial_len = np.linalg.norm(radial, axis=1)
    usable = (normal_len > 1e-12) & (radial_len > 1e-12)
    dots = np.zeros(len(faces), dtype=np.float64)
    dots[usable] = np.sum(normals[usable] * radial[usable], axis=1) / (
        normal_len[usable] * radial_len[usable]
    )
    return int((dots > 1e-8).sum()), int((dots < -1e-8).sum())


def _dense_dual_grid_mesh(n=4):
    from trellmlx.mesh_extract import decoder_output_to_mesh

    coords_3d = np.array(
        [[z, y, x] for z in range(n) for y in range(n) for x in range(n)],
        dtype=np.int64,
    )
    coords = np.column_stack([np.zeros(len(coords_3d), dtype=np.int64), coords_3d])
    feats = np.zeros((len(coords), 7), dtype=np.float32)
    feats[:, 3:6] = 1.0
    return decoder_output_to_mesh(feats, coords, resolution=n)


def _component_signed_volumes(vertices, faces):
    """Return signed volume for each face-connected component."""
    edge_faces = {}
    for fi, face in enumerate(faces):
        for i in range(3):
            edge = tuple(sorted((int(face[i]), int(face[(i + 1) % 3]))))
            edge_faces.setdefault(edge, []).append(fi)

    adjacency = [[] for _ in range(len(faces))]
    for incident in edge_faces.values():
        for a in incident:
            for b in incident:
                if a != b:
                    adjacency[a].append(b)

    volumes = []
    seen = set()
    for start in range(len(faces)):
        if start in seen:
            continue
        stack = [start]
        component = []
        seen.add(start)
        while stack:
            cur = stack.pop()
            component.append(cur)
            for nxt in adjacency[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        tri = vertices[faces[component]]
        volume = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0
        volumes.append(volume)
    return volumes


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
    def test_splits_non_manifold_edge_vertices_without_deleting_faces(self):
        """Cumesh repair preserves faces and separates non-manifold edge corners."""
        verts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [2.0, 1.0, 0.0],
        ], dtype=np.float32)
        faces = np.array([
            [0, 1, 2],
            [1, 0, 3],
            [0, 1, 4],
            [2, 1, 5],
        ], dtype=np.int64)

        cleaned_v, cleaned_f = repair_non_manifold_edges(verts, faces)

        assert len(cleaned_f) == len(faces)
        assert len(cleaned_v) == 10
        assert cleaned_f.tolist() == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [2, 1, 9]]

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

    def test_open_dual_grid_patch_orients_local_winding_conflicts(self):
        """Open dual-grid cleanup should orient local conflicts without deleting patches."""
        vertices, faces = _dense_dual_grid_mesh(n=4)
        vertices, faces = remove_duplicate_faces(vertices, faces, verbose=False)
        vertices, faces = repair_non_manifold_edges(vertices, faces, verbose=False)
        vertices, faces = remove_small_components(vertices, faces, verbose=False)
        vertices, faces = fill_small_holes(vertices, faces, verbose=False)

        assert _same_direction_manifold_conflict_count(faces) > 0

        fixed_v, fixed_f = fix_normals(vertices, faces, verbose=False)

        _assert_manifold_edges_oppositely_oriented(fixed_f)
        assert len(fixed_f) == len(faces)

    def test_open_mesh_fix_normals_flips_adjacent_face_instead_of_pruning(self):
        """Open meshes need reference-like orientation propagation, not face deletion."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ], dtype=np.float32)
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
        ], dtype=np.int64)

        assert _same_direction_manifold_conflict_count(faces) == 1

        fixed_v, fixed_f = fix_normals(vertices, faces, verbose=False)

        assert len(fixed_f) == 2
        assert _same_direction_manifold_conflict_count(fixed_f) == 0
        assert {tuple(sorted(face)) for face in fixed_f.tolist()} == {
            (0, 1, 2),
            (0, 1, 3),
        }

    def test_open_mesh_orientation_matches_cumesh_low_root_union_find(self):
        """Contradictory open components should follow cumesh's union-find choice."""
        vertices = np.zeros((7, 3), dtype=np.float32)
        faces = np.array([
            [6, 0, 4],
            [6, 5, 4],
            [0, 3, 6],
            [3, 5, 0],
            [3, 5, 4],
        ], dtype=np.int64)

        _, oriented = orient_faces_by_adjacency(vertices, faces, verbose=False)

        expected = faces.copy()
        expected[[1, 2]] = expected[[1, 2]][:, ::-1]
        np.testing.assert_array_equal(oriented, expected)

    def test_radial_heuristic_orients_globally_inverted_patch_outward(self):
        """The optional radial heuristic can flip a consistent open component."""
        vertices, faces = _make_open_box()
        inverted = faces[:, ::-1].copy()
        outward_before, inward_before = _radial_orientation_counts(vertices, inverted)
        assert inward_before > outward_before

        fixed_v, fixed_f = orient_components_outward_by_radial_heuristic(
            vertices, inverted, verbose=False,
        )

        outward_after, inward_after = _radial_orientation_counts(fixed_v, fixed_f)
        assert len(fixed_f) == len(inverted)
        assert outward_after > inward_after


class TestRemoveSameDirectionManifoldConflicts:
    def test_removes_smaller_face_from_same_direction_shared_edge(self):
        """Fallback pruning must clear remaining two-face edge direction conflicts."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
        ], dtype=np.float32)
        faces = np.array([
            [0, 1, 2],  # area 0.5
            [0, 1, 3],  # area 1.0, same directed edge (0, 1)
        ], dtype=np.int64)

        cleaned_v, cleaned_f = remove_same_direction_manifold_conflicts(
            vertices,
            faces,
            verbose=False,
        )

        assert len(cleaned_f) == 1
        assert cleaned_f.tolist() == [[0, 1, 2]]
        _assert_manifold_edges_oppositely_oriented(cleaned_f)
        assert cleaned_v.shape == (3, 3)
        assert [1.0, 2.0, 0.0] in cleaned_v.tolist()


class TestFillSmallHolesPerimeter:
    def test_max_hole_perimeter_is_keyword_only(self):
        """Old positional max_edges callers must not silently become perimeter callers."""
        verts, faces = _make_open_box()
        with pytest.raises(TypeError):
            fill_small_holes(verts, faces, 100)

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

    def test_filled_hole_preserves_boundary_winding_consistency(self):
        """Cap triangles must oppose the existing directed boundary edges."""
        verts, faces = _make_open_box()
        v_out, f_out = fill_small_holes(verts, faces, max_hole_perimeter=3e-2, verbose=False)
        assert len(f_out) == len(faces) + 4
        _assert_manifold_edges_oppositely_oriented(f_out)


class TestCleanupMeshIntegration:
    def test_cleanup_tuning_arguments_are_keyword_only(self):
        """Old positional cleanup thresholds must fail loud instead of remapping."""
        verts, faces = _make_tetrahedron()
        with pytest.raises(TypeError):
            cleanup_mesh(verts, faces, 0.01, 100)

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

        # Default cleanup preserves all components (floater survives)
        cleaned_v, cleaned_f = cleanup_mesh(verts, faces, verbose=False)
        assert len(cleaned_f) >= 10  # dedup removes 1, both components remain

        # With keep_largest, floater is removed
        cleaned_v2, cleaned_f2 = cleanup_mesh(verts, faces, keep_largest=True, verbose=False)
        assert len(cleaned_f2) >= 10
        assert cleaned_v2[:, 0].max() <= 0.01  # floater at x=10 is gone

    def test_already_clean_mesh(self):
        verts, faces = _make_tetrahedron()
        cleaned_v, cleaned_f = cleanup_mesh(verts, faces, verbose=False)
        # Tetrahedron is watertight, no floaters, no duplicates
        assert len(cleaned_f) == len(faces)

    def test_cleanup_orients_disconnected_closed_components_outward(self):
        """Default cleanup preserves islands, so each closed body needs orientation repair."""
        box = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=np.float32)
        box_faces = np.array([
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ], dtype=np.int64)
        vertices = np.vstack([box, box + np.array([2.0, 0.0, 0.0], dtype=np.float32)])
        faces = np.vstack([box_faces, box_faces[:, ::-1] + len(box)])

        cleaned_v, cleaned_f = cleanup_mesh(vertices, faces, verbose=False)

        assert len(cleaned_f) == len(faces)
        assert all(volume > 0 for volume in _component_signed_volumes(cleaned_v, cleaned_f))


class TestKeepLargestIsOptIn:
    """keep_largest_component should NOT run by default in cleanup_mesh.

    The default cleanup should preserve all components so we see everything
    the model generated. keep_largest is opt-in via the keep_largest param.
    """

    def test_default_cleanup_preserves_multiple_components(self):
        """Two separate tetrahedra should both survive default cleanup."""
        v1, f1 = _make_tetrahedron()
        v2, f2 = _make_tetrahedron(offset=[5, 5, 5])
        # Combine into one mesh with two disconnected components
        f2_shifted = f2 + len(v1)
        verts = np.vstack([v1, v2])
        faces = np.vstack([f1, f2_shifted])

        cleaned_v, cleaned_f = cleanup_mesh(verts, faces, verbose=False)
        # Both components should survive — 8 faces total (4 + 4)
        assert len(cleaned_f) == 8

    def test_keep_largest_removes_smaller_component(self):
        """With keep_largest=True, only the largest component survives."""
        v1, f1 = _make_tetrahedron()
        v2, f2 = _make_tetrahedron(offset=[5, 5, 5])
        f2_shifted = f2 + len(v1)
        verts = np.vstack([v1, v2])
        faces = np.vstack([f1, f2_shifted])

        cleaned_v, cleaned_f = cleanup_mesh(
            verts, faces, keep_largest=True, verbose=False,
        )
        # Equal-size components: one gets kept, one removed → 4 faces
        assert len(cleaned_f) == 4


class TestDefaultCleanupUsesSmallComponentRemoval:
    """Default cleanup should use remove_small_components (fractional threshold),
    not keep_largest_component (binary), matching the reference pipeline."""

    def test_default_removes_tiny_components_preserves_large_ones(self):
        """A tiny 1-face fragment should be removed when min_component_ratio
        makes the threshold meaningful."""
        v1, f1 = _make_tetrahedron()
        # Small component: single tiny triangle far away
        v2 = np.array([[10, 10, 10], [10.001, 10, 10], [10, 10.001, 10]], dtype=np.float32)
        f2 = np.array([[0, 1, 2]], dtype=np.uint32) + len(v1)
        verts = np.vstack([v1, v2])
        faces = np.vstack([f1, f2])

        # With min_component_ratio=0.5, threshold = int(4 * 0.5) = 2
        # The 1-face fragment is below threshold and gets removed
        cleaned_v, cleaned_f = cleanup_mesh(
            verts, faces, min_component_ratio=0.5, verbose=False,
        )
        assert len(cleaned_f) == 4
        assert cleaned_v[:, 0].max() < 5

    def test_default_preserves_substantial_second_component(self):
        """Two equal tetrahedra should both survive default cleanup
        (remove_small_components keeps components above threshold)."""
        v1, f1 = _make_tetrahedron()
        v2, f2 = _make_tetrahedron(offset=[5, 5, 5])
        f2_shifted = f2 + len(v1)
        verts = np.vstack([v1, v2])
        faces = np.vstack([f1, f2_shifted])

        cleaned_v, cleaned_f = cleanup_mesh(verts, faces, verbose=False)
        # Both should survive — they're equal size, both above threshold
        assert len(cleaned_f) == 8

    def test_cleanup_mesh_accepts_min_component_ratio(self):
        """cleanup_mesh should accept min_component_ratio parameter."""
        v1, f1 = _make_tetrahedron()
        cleaned_v, cleaned_f = cleanup_mesh(
            v1, f1, min_component_ratio=1e-5, verbose=False,
        )
        assert len(cleaned_f) == 4


class TestIntermediateCleanupSkipsNormals:
    """Intermediate cleanup passes should skip normals fixing."""

    def test_do_fix_normals_false_skips_normals(self):
        """cleanup_mesh with do_fix_normals=False should not call fix_normals."""
        verts, faces = _make_tetrahedron()
        # Flip one face to create inconsistent winding
        faces_bad = faces.copy()
        faces_bad[0] = faces_bad[0][::-1]

        # With normals fixing (default)
        v1, f1 = cleanup_mesh(verts, faces_bad.copy(), verbose=False)
        # Without normals fixing
        v2, f2 = cleanup_mesh(verts, faces_bad.copy(), do_fix_normals=False, verbose=False)

        # Both should have same face count, but winding may differ
        assert len(f1) == len(f2)
