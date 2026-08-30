import numpy as np

from scripts.analyze_dual_grid_quad_provenance import (
    _compare_face_rows,
    _triangulation_exonerated,
    analyze_quad_topology,
)


def _analyze(quads):
    quads = np.asarray(quads, dtype=np.int64)
    return analyze_quad_topology(
        quads,
        np.arange(len(quads), dtype=np.int8) % 3,
        np.column_stack(
            (
                np.arange(len(quads)),
                np.zeros(len(quads), dtype=np.int64),
                np.zeros(len(quads), dtype=np.int64),
            )
        ),
        max_witnesses=4,
    )


def test_two_consistently_oriented_quads_are_orientable():
    report = _analyze([[0, 1, 2, 3], [1, 4, 5, 2]])

    assert report["edge_groups"] == 7
    assert report["boundary_edges"] == 6
    assert report["manifold_edges"] == 1
    assert report["same_direction_manifold_edges"] == 0
    assert report["nonmanifold_edges"] == 0
    assert report["orientation"]["contradictory_components"] == 0


def test_same_direction_adjacency_requires_one_quad_flip():
    report = _analyze([[0, 1, 2, 3], [1, 2, 5, 4]])

    assert report["manifold_edges"] == 1
    assert report["same_direction_manifold_edges"] == 1
    assert report["orientation"]["contradictory_components"] == 0


def test_three_incident_quads_make_nonmanifold_edge():
    report = _analyze(
        [[0, 1, 2, 3], [1, 0, 4, 5], [0, 1, 6, 7]]
    )

    assert report["nonmanifold_edges"] == 1
    assert report["witnesses"]["nonmanifold_edges"][0]["vertices"] == [0, 1]
    assert len(report["witnesses"]["nonmanifold_edges"][0]["quad_ids"]) == 3


def test_duplicate_quads_are_reported():
    report = _analyze([[0, 1, 2, 3], [2, 3, 0, 1]])

    assert report["duplicate_quads"]["groups"] == 1
    assert report["duplicate_quads"]["quads"] == 2
    assert report["duplicate_quads"]["witnesses"][0]["quad_ids"] == [0, 1]


def test_face_row_comparison_handles_shape_mismatch_without_broadcasting():
    report = _compare_face_rows(
        np.zeros((2, 3), dtype=np.int32),
        np.zeros((3, 3), dtype=np.int32),
    )

    assert report["shapes_exact"] is False
    assert report["rows_exact"] is False
    assert report["first_mismatching_face_row"] is None


def test_triangulation_exoneration_requires_exact_rows_and_quad_contradiction():
    reconstruction = {
        "raw_faces_row_exact": True,
        "internal_diagonals": {"all_pairs_share_one_diagonal": True},
    }
    topology = {"orientation": {"contradictory_components": 1}}

    assert _triangulation_exonerated(reconstruction, topology) is True
    topology["orientation"]["contradictory_components"] = 0
    assert _triangulation_exonerated(reconstruction, topology) is False
    topology["orientation"]["contradictory_components"] = 1
    reconstruction["raw_faces_row_exact"] = False
    assert _triangulation_exonerated(reconstruction, topology) is False
