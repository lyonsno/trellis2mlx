import numpy as np

from trellmlx.mesh_lineage import (
    approximate_surface_transition,
    attest_uv_mapping,
    exact_face_transition,
    mesh_topology_summary,
)


def _sheet():
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    return vertices, faces


def test_mesh_topology_summary_reports_consistent_sheet_edges():
    vertices, faces = _sheet()
    summary = mesh_topology_summary(vertices, faces)
    assert summary["boundary_edges"] == 4
    assert summary["manifold_edges"] == 1
    assert summary["nonmanifold_edges"] == 0
    assert summary["same_direction_manifold_conflicts"] == 0


def test_exact_face_transition_ignores_reindexing_and_detects_reversal():
    vertices, faces = _sheet()
    permutation = np.array([2, 0, 3, 1])
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(len(permutation))
    reindexed_vertices = vertices[permutation]
    reindexed_faces = inverse[faces].copy()
    reindexed_faces[1] = reindexed_faces[1, [0, 2, 1]]
    report = exact_face_transition(
        vertices, faces, reindexed_vertices, reindexed_faces
    )
    assert report["common_unique_face_keys"] == 2
    assert report["same_orientation"] == 1
    assert report["reversed_orientation"] == 1
    assert report["before_only_unique_face_keys"] == 0
    assert report["after_only_unique_face_keys"] == 0


def test_approximate_surface_transition_is_exact_for_identical_mesh():
    vertices, faces = _sheet()
    report = approximate_surface_transition(vertices, faces, vertices, faces)
    assert report["before_to_after"]["normalized_centroid_distance"]["max"] == 0
    assert report["after_to_before"]["normalized_centroid_distance"]["max"] == 0
    assert report["before_to_after"]["absolute_normal_dot"]["min"] == 1
    assert report["before_to_after"]["opposed_normal_fraction"] == 0


def test_approximate_surface_transition_reports_bounded_sampling():
    vertices, faces = _sheet()
    report = approximate_surface_transition(
        vertices, faces, vertices, faces, max_faces_per_side=1
    )
    assert report["before_faces"] == 2
    assert report["before_sampled_faces"] == 1
    assert report["after_sampled_faces"] == 1


def test_uv_mapping_attestation_accepts_exact_seam_split():
    vertices, faces = _sheet()
    vmapping = np.array([0, 1, 2, 1, 3, 2], dtype=np.uint32)
    uv_vertices = vertices[vmapping]
    uv_faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)
    report = attest_uv_mapping(vertices, faces, uv_vertices, uv_faces, vmapping)
    assert report["geometry_preserved_exactly"] is True


def test_uv_mapping_attestation_rejects_reversed_face():
    vertices, faces = _sheet()
    vmapping = np.array([0, 1, 2, 1, 3, 2], dtype=np.uint32)
    uv_vertices = vertices[vmapping]
    uv_faces = np.array([[0, 1, 2], [3, 5, 4]], dtype=np.uint32)
    report = attest_uv_mapping(vertices, faces, uv_vertices, uv_faces, vmapping)
    assert report["geometry_preserved_exactly"] is False
