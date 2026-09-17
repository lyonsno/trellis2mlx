import numpy as np
import pytest
import trimesh

from trellmlx.exterior_surface_repair import (
    _has_forbidden_bridge_intersection,
    _select_bridge_pairs,
    _triangles_intersect,
    bridge_loop_pair,
    repair_exterior_surface,
)


def _orthogonal_exact_contact_surfaces():
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
            [5, 6, 7],
            [5, 7, 8],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def test_repair_exterior_surface_recovers_intrinsic_inverted_sheet():
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    expected = np.asarray(mesh.faces, dtype=np.int64)
    centroids = vertices[expected].mean(axis=1)
    inverted = centroids[:, 2] > 0.55
    broken = expected.copy()
    broken[inverted] = broken[inverted][:, [0, 2, 1]]

    repaired, report = repair_exterior_surface(vertices, broken)

    np.testing.assert_array_equal(repaired, expected)
    assert report["orientation"]["reversed_faces"] == int(inverted.sum())
    assert report["orientation"]["candidate_count"] == 1
    assert report["bridges"]["accepted_pairs"] == 0


def test_repair_exterior_surface_leaves_consistent_surface_untouched():
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    repaired, report = repair_exterior_surface(vertices, faces)

    np.testing.assert_array_equal(repaired, faces)
    assert report["orientation"]["reversed_faces"] == 0
    assert report["bridges"]["accepted_pairs"] == 0


def test_repair_exterior_surface_refuses_competing_orientation_patches():
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    expected = np.asarray(mesh.faces, dtype=np.int64)
    centroids = vertices[expected].mean(axis=1)
    competing = np.abs(centroids[:, 2]) > 0.55
    broken = expected.copy()
    broken[competing] = broken[competing][:, [0, 2, 1]]

    repaired, report = repair_exterior_surface(vertices, broken)

    np.testing.assert_array_equal(repaired, broken)
    assert report["orientation"]["candidate_count"] == 2
    assert report["orientation"]["selection_status"] == "ambiguous"
    assert report["orientation"]["reversed_faces"] == 0
    assert report["bridges"]["accepted_pairs"] == 0


@pytest.mark.parametrize("confidence", [-0.01, 1.01, np.nan, np.inf, -np.inf])
def test_repair_exterior_surface_rejects_invalid_component_confidence(confidence):
    with pytest.raises(ValueError, match="between 0 and 1"):
        repair_exterior_surface(
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.int64),
            component_orientation_confidence=confidence,
        )


def test_bridge_selector_refuses_orthogonal_exact_contact_surfaces():
    vertices, faces = _orthogonal_exact_contact_surfaces()

    pairs, report = _select_bridge_pairs(
        vertices,
        faces,
        np.arange(4, dtype=np.int64),
    )

    assert pairs == []
    assert report["rejected_incompatible"] == 1


def test_bridge_loop_pair_refuses_when_every_zipper_worsens_topology():
    vertices, faces = _orthogonal_exact_contact_surfaces()

    with pytest.raises(ValueError, match="no topology-safe alignment"):
        bridge_loop_pair(
            vertices,
            faces,
            main_loop=np.asarray([0, 1, 2, 3]),
            detached_loop=np.asarray([5, 6, 7, 8]),
        )


def test_bridge_intersection_guard_detects_nonadjacent_crossing_triangles():
    vertices = np.asarray(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -0.5, -1.0],
            [0.0, -0.5, 1.0],
            [0.0, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    crossing = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)

    assert _has_forbidden_bridge_intersection(
        vertices,
        np.empty((0, 3), dtype=np.int64),
        crossing,
    )


def test_bridge_refuses_weld_that_introduces_same_direction_duplicate_faces():
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)

    with pytest.raises(ValueError, match="no topology-safe alignment"):
        bridge_loop_pair(
            vertices,
            faces,
            main_loop=np.asarray([0, 1, 2]),
            detached_loop=np.asarray([3, 4, 5]),
        )


def test_intersection_guard_rejects_shared_vertex_with_interior_crossing():
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    base = np.asarray([[0, 1, 2]], dtype=np.int64)
    bridge = np.asarray([[0, 3, 4]], dtype=np.int64)

    assert _triangles_intersect(vertices, base[0], bridge[0])
    assert _has_forbidden_bridge_intersection(vertices, base, bridge)


def test_intersection_guard_allows_only_intended_shared_edge_contact():
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )

    assert not _triangles_intersect(
        vertices,
        np.asarray([0, 1, 2]),
        np.asarray([0, 1, 3]),
    )


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_intersection_guard_is_scale_homogeneous(scale):
    vertices = scale * np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    assert not _triangles_intersect(
        vertices,
        np.asarray([0, 1, 2]),
        np.asarray([0, 1, 3]),
    )
    assert _triangles_intersect(
        vertices,
        np.asarray([0, 1, 2]),
        np.asarray([0, 4, 5]),
    )


def test_bridge_refuses_opposite_winding_coincident_contours():
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [3, 5, 4]], dtype=np.int64)

    with pytest.raises(ValueError, match="no topology-safe alignment"):
        bridge_loop_pair(
            vertices,
            faces,
            main_loop=np.asarray([0, 1, 2]),
            detached_loop=np.asarray([3, 4, 5]),
        )


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_bridge_loop_pair_welds_only_local_exact_contacts_without_zero_faces(scale):
    vertices = scale * np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.4, 0.0],
            [1.0, 1.6, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.empty((0, 3), dtype=np.int64)

    repaired, report = bridge_loop_pair(
        vertices,
        faces,
        main_loop=np.asarray([0, 1, 2, 3]),
        detached_loop=np.asarray([4, 5, 6, 7]),
    )

    assert report["welded_vertex_pairs"] == 2
    assert report["new_zero_area_faces"] == 0
    assert report["new_nonmanifold_edges"] == 0
    assert len(repaired) > 0
    assert np.all(
        np.linalg.norm(
            np.cross(
                vertices[repaired][:, 1] - vertices[repaired][:, 0],
                vertices[repaired][:, 2] - vertices[repaired][:, 0],
            ),
            axis=1,
        )
        > 0.0
    )
