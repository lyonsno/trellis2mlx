import json
import sys

import numpy as np

from scripts.analyze_orientation_semantics import (
    _solve_parity_components,
    analyze_face_orientations,
    analyze_orientation_topology,
    main,
)


def test_orientation_analysis_accepts_component_global_flip():
    input_faces = np.array(
        [[0, 1, 2], [2, 1, 3]],
        dtype=np.int32,
    )
    reference_faces = input_faces.copy()
    candidate_faces = input_faces[:, [0, 2, 1]]

    report = analyze_face_orientations(
        input_faces,
        reference_faces,
        candidate_faces,
    )

    assert report["semantic_parity"] is True
    assert report["topology"]["orientable_components"] == 1
    assert report["topology"]["contradictory_components"] == 0
    assert report["reference"]["orientable_edge_violations"] == 0
    assert report["candidate"]["orientable_edge_violations"] == 0
    assert report["comparison"]["orientable_differing_face_rows"] == 2
    assert (
        report["comparison"]["orientable_non_global_choice_components"]
        == 0
    )


def test_orientation_analysis_rejects_partial_component_flip():
    input_faces = np.array(
        [[0, 1, 2], [2, 1, 3]],
        dtype=np.int32,
    )
    reference_faces = input_faces.copy()
    candidate_faces = input_faces.copy()
    candidate_faces[1] = candidate_faces[1, [0, 2, 1]]

    report = analyze_face_orientations(
        input_faces,
        reference_faces,
        candidate_faces,
    )

    assert report["semantic_parity"] is False
    assert report["candidate"]["orientable_edge_violations"] == 1
    assert (
        report["comparison"]["orientable_non_global_choice_components"]
        == 1
    )


def test_parity_solver_marks_contradictory_component():
    result = _solve_parity_components(
        3,
        np.array([0, 1, 2], dtype=np.int32),
        np.array([1, 2, 0], dtype=np.int32),
        np.array([0, 0, 1], dtype=np.uint8),
    )

    assert result["component_count"] == 1
    assert result["contradictory_component_count"] == 1
    assert result["contradictory_face_count"] == 3


def test_orientation_topology_reports_satisfiable_sheet():
    faces = np.array(
        [[0, 1, 2], [2, 1, 3]],
        dtype=np.int32,
    )

    topology = analyze_orientation_topology(faces)

    assert topology["faces"] == 2
    assert topology["manifold_edges"] == 1
    assert topology["orientable_components"] == 1
    assert topology["contradictory_components"] == 0


def test_orientation_analysis_writes_failure_report_for_wrong_hash(
    tmp_path,
    monkeypatch,
):
    input_ply = tmp_path / "input.ply"
    reference_ply = tmp_path / "reference.ply"
    candidate_ply = tmp_path / "candidate.ply"
    output_json = tmp_path / "report.json"
    for path in (input_ply, reference_ply, candidate_ply):
        path.write_bytes(b"not-a-ply")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_orientation_semantics.py",
            "--input-ply",
            str(input_ply),
            "--reference-ply",
            str(reference_ply),
            "--candidate-ply",
            str(candidate_ply),
            "--output-json",
            str(output_json),
            "--expected-input-sha256",
            "0" * 64,
            "--expected-reference-sha256",
            "1" * 64,
            "--expected-candidate-sha256",
            "2" * 64,
        ],
    )

    assert main() == 1
    report = json.loads(output_json.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "identity_validation"
    assert report["last_trustworthy_phase"] == "request_validated"
    assert report["artifacts"]["input"]["status"] == "hash_mismatch"
    assert "SHA256 mismatch" in report["error"]
