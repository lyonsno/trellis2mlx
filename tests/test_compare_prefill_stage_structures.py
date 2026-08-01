import numpy as np
import pytest


def _fixture():
    input_vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        dtype=np.float32,
    )
    input_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    centers = np.array([[0.5, 0.5, 0], [0.25, 0.25, 0]], dtype=np.float32)
    appended_faces = np.array(
        [[0, 1, 4], [1, 2, 4], [2, 3, 5]],
        dtype=np.int32,
    )
    return (
        np.concatenate([input_vertices, centers]),
        np.concatenate([input_faces, appended_faces]),
    )


def test_prefill_structure_comparison_factors_out_center_and_face_order():
    from scripts.compare_prefill_stage_structures import compare_prefill_arrays

    reference_vertices, reference_faces = _fixture()
    candidate_vertices = np.concatenate(
        [reference_vertices[:4], reference_vertices[[5, 4]]]
    )
    candidate_faces = np.concatenate(
        [
            reference_faces[:2],
            np.array([[2, 3, 4], [1, 2, 5], [0, 1, 5]], dtype=np.int32),
        ]
    )

    report = compare_prefill_arrays(
        reference_vertices,
        reference_faces,
        candidate_vertices,
        candidate_faces,
        input_vertex_count=4,
        input_face_count=2,
    )

    assert report["input_prefix"]["vertices_exact"] is True
    assert report["input_prefix"]["faces_exact"] is True
    assert report["appended"]["vertices_ordered_exact"] is False
    assert report["appended"]["faces_ordered_exact"] is False
    assert report["boundary_edges"]["multiset_exact"] is True
    assert report["edge_aligned_centers"]["exact"] is True
    assert report["edge_aligned_centers"]["float32_ulp"]["max"] == 0


def test_prefill_structure_comparison_measures_one_ulp_center_drift():
    from scripts.compare_prefill_stage_structures import compare_prefill_arrays

    reference_vertices, reference_faces = _fixture()
    candidate_vertices = reference_vertices.copy()
    candidate_vertices[4, 0] = np.nextafter(
        candidate_vertices[4, 0],
        np.float32(1.0),
        dtype=np.float32,
    )

    report = compare_prefill_arrays(
        reference_vertices,
        reference_faces,
        candidate_vertices,
        reference_faces,
        input_vertex_count=4,
        input_face_count=2,
    )

    assert report["boundary_edges"]["multiset_exact"] is True
    assert report["edge_aligned_centers"]["exact"] is False
    assert report["edge_aligned_centers"]["float32_ulp"]["max"] == 1
    assert report["edge_aligned_centers"]["float32_ulp"]["nonzero"] == 2


def test_prefill_structure_comparison_detects_boundary_topology_change():
    from scripts.compare_prefill_stage_structures import compare_prefill_arrays

    reference_vertices, reference_faces = _fixture()
    candidate_faces = reference_faces.copy()
    candidate_faces[-1] = [0, 3, 5]

    report = compare_prefill_arrays(
        reference_vertices,
        reference_faces,
        reference_vertices,
        candidate_faces,
        input_vertex_count=4,
        input_face_count=2,
    )

    assert report["boundary_edges"]["multiset_exact"] is False
    assert report["edge_aligned_centers"]["comparable"] is False


def test_prefill_structure_comparison_rejects_changed_input_prefix():
    from scripts.compare_prefill_stage_structures import (
        StructureComparisonError,
        compare_prefill_arrays,
    )

    reference_vertices, reference_faces = _fixture()
    candidate_vertices = reference_vertices.copy()
    candidate_vertices[0, 0] = 1

    with pytest.raises(StructureComparisonError, match="input vertex prefix"):
        compare_prefill_arrays(
            reference_vertices,
            reference_faces,
            candidate_vertices,
            reference_faces,
            input_vertex_count=4,
            input_face_count=2,
        )


def test_cub_vec3f_segmented_sum_uses_balanced_warp_tree():
    from scripts.compare_prefill_stage_structures import cub_vec3f_segmented_sum

    values = np.zeros((4, 3), dtype=np.float32)
    values[:, 0] = [1e20, 1, -1e20, 3]

    reduced = cub_vec3f_segmented_sum(values)

    assert reduced.dtype == np.float32
    np.testing.assert_array_equal(reduced, np.array([0, 0, 0], dtype=np.float32))


def test_cub_vec3f_segmented_sum_combines_warps_in_order():
    from scripts.compare_prefill_stage_structures import cub_vec3f_segmented_sum

    values = np.zeros((65, 3), dtype=np.float32)
    values[0, 0] = np.float32(1e20)
    values[32, 0] = np.float32(-1e20)
    values[64, 0] = np.float32(3)

    reduced = cub_vec3f_segmented_sum(values)

    np.testing.assert_array_equal(reduced, np.array([3, 0, 0], dtype=np.float32))


def test_cub_center_reproduction_uses_original_edges_and_fan_groups():
    from scripts.compare_prefill_stage_structures import reproduce_cub_hole_centers

    input_vertices = np.zeros((65, 3), dtype=np.float32)
    input_vertices[[0, 2, 4, 6], 0] = [
        np.float32(2e20),
        np.float32(2),
        np.float32(-2e20),
        np.float32(6),
    ]
    appended_faces = np.array(
        [[0, 1, 65], [2, 3, 65], [4, 5, 65], [6, 7, 65]],
        dtype=np.int32,
    )

    centers, center_ids, loop_sizes = reproduce_cub_hole_centers(
        input_vertices,
        appended_faces,
        input_vertex_count=65,
    )

    np.testing.assert_array_equal(centers, np.array([[0, 0, 0]], dtype=np.float32))
    np.testing.assert_array_equal(center_ids, np.array([65], dtype=np.int32))
    np.testing.assert_array_equal(loop_sizes, np.array([4], dtype=np.int32))
