import numpy as np
import pytest
import trimesh

from trellmlx.exterior_surface_repair import repair_exterior_surface


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
    assert report["input_mesh"]["faces"] == report["output_mesh"]["faces"]
    assert (
        report["input_mesh"]["topology"]["boundary_edges"]
        == report["output_mesh"]["topology"]["boundary_edges"]
    )
    assert (
        report["input_mesh"]["topology"]["nonmanifold_edges"]
        == report["output_mesh"]["topology"]["nonmanifold_edges"]
    )
    assert (
        report["output_mesh"]["topology"]["same_direction_shared_edges"]
        < report["input_mesh"]["topology"]["same_direction_shared_edges"]
    )


def test_repair_exterior_surface_leaves_consistent_surface_untouched():
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    repaired, report = repair_exterior_surface(vertices, faces)

    np.testing.assert_array_equal(repaired, faces)
    assert report["orientation"]["reversed_faces"] == 0
    assert report["input_mesh"]["topology"] == report["output_mesh"]["topology"]


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
    assert report["input_mesh"]["topology"] == report["output_mesh"]["topology"]


def test_repair_exterior_surface_compares_candidates_across_components():
    small = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    large = trimesh.creation.icosphere(subdivisions=1, radius=2.0)
    small_vertices = np.asarray(small.vertices, dtype=np.float64) + [-4.0, 0.0, 0.0]
    large_vertices = np.asarray(large.vertices, dtype=np.float64) + [4.0, 0.0, 0.0]
    small_faces = np.asarray(small.faces, dtype=np.int64)
    large_faces = np.asarray(large.faces, dtype=np.int64)
    small_inverted = small_vertices[small_faces].mean(axis=1)[:, 2] > 0.55
    large_inverted = large_vertices[large_faces].mean(axis=1)[:, 2] > 1.1
    broken_small = small_faces.copy()
    broken_large = large_faces.copy()
    broken_small[small_inverted] = broken_small[small_inverted][:, [0, 2, 1]]
    broken_large[large_inverted] = broken_large[large_inverted][:, [0, 2, 1]]
    vertices = np.concatenate((small_vertices, large_vertices), axis=0)
    faces = np.concatenate(
        (broken_small, broken_large + len(small_vertices)), axis=0
    )

    repaired, report = repair_exterior_surface(vertices, faces)

    small_count = len(small_faces)
    np.testing.assert_array_equal(repaired[:small_count], broken_small)
    assert np.count_nonzero(
        np.any(repaired[small_count:] != faces[small_count:], axis=1)
    ) == report["orientation"]["reversed_faces"]
    assert report["orientation"]["candidate_count"] == 2
    assert report["orientation"]["selected_area"] > sum(
        row["area"] for row in report["orientation"]["candidates"][1:]
    )


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64])
def test_repair_exterior_surface_preserves_face_dtype(dtype):
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=dtype)

    repaired, _ = repair_exterior_surface(vertices, faces)

    assert repaired.dtype == faces.dtype


@pytest.mark.parametrize("confidence", [-0.01, 1.01, np.nan, np.inf, -np.inf])
def test_repair_exterior_surface_rejects_invalid_component_confidence(confidence):
    with pytest.raises(ValueError, match="between 0 and 1"):
        repair_exterior_surface(
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.int64),
            component_orientation_confidence=confidence,
        )
