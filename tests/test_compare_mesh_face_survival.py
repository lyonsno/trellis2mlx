"""Contracts for authenticated exact source-face survival evidence."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import numpy as np
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

from trellmlx.glb_aabb_crop import write_geometry_glb


SCRIPT = Path(__file__).parents[1] / "scripts" / "compare_mesh_face_survival.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("face_survival_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()
match_exact_source_faces = RUNNER.match_exact_source_faces


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(
    source: Path,
    candidates: list[tuple[str, Path]],
    output_dir: Path,
    report: Path,
    *,
    source_digest: str | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(SCRIPT),
        "--source",
        str(source),
        source_digest or _sha256(source),
    ]
    for label, path in candidates:
        command.extend(("--candidate", label, str(path), _sha256(path)))
    command.extend(("--output-dir", str(output_dir), "--report", str(report)))
    return subprocess.run(command, text=True, capture_output=True, check=False, timeout=10)


def _write_assay_fixtures(source: Path, first: Path, second: Path) -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    source_faces = np.array([[0, 1, 2], [1, 3, 2], [0, 2, 4]], dtype=np.uint32)
    write_geometry_glb(source, vertices, source_faces)
    write_geometry_glb(first, vertices[[4, 2, 0, 1]], np.array([[2, 3, 1]], dtype=np.uint32))
    write_geometry_glb(
        second,
        vertices,
        np.array([[1, 3, 2], [0, 1, 4]], dtype=np.uint32),
    )


def test_matches_reindexed_and_rewound_faces_by_exact_float32_geometry() -> None:
    source_vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    source_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    candidate_vertices = source_vertices[[3, 2, 0, 1]]
    candidate_faces = np.array([[1, 3, 2]], dtype=np.uint32)

    matches = match_exact_source_faces(
        source_vertices, source_faces, candidate_vertices, candidate_faces
    )

    np.testing.assert_array_equal(matches.source_survival_mask, [True, False])
    np.testing.assert_array_equal(matches.candidate_source_face_index, [0])


def test_does_not_match_a_new_triangle_composed_of_existing_source_vertices() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    source_faces = np.array([[0, 1, 2]], dtype=np.uint32)
    candidate_faces = np.array([[0, 1, 3]], dtype=np.uint32)

    matches = match_exact_source_faces(vertices, source_faces, vertices, candidate_faces)

    np.testing.assert_array_equal(matches.source_survival_mask, [False])
    np.testing.assert_array_equal(matches.candidate_source_face_index, [-1])


def test_float32_coordinate_change_is_not_an_exact_match() -> None:
    source_vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    candidate_vertices = source_vertices.copy()
    candidate_vertices[1, 0] = np.nextafter(np.float32(1.0), np.float32(2.0))
    faces = np.array([[0, 1, 2]], dtype=np.uint32)

    matches = match_exact_source_faces(
        source_vertices, faces, candidate_vertices, faces
    )

    np.testing.assert_array_equal(matches.source_survival_mask, [False])
    np.testing.assert_array_equal(matches.candidate_source_face_index, [-1])


def test_duplicate_geometric_faces_are_matched_one_to_one_without_overclaiming() -> None:
    source_vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    source_faces = np.array([[0, 1, 2], [3, 2, 1]], dtype=np.uint32)
    candidate_faces = np.array([[0, 2, 1], [3, 1, 2], [0, 1, 2]], dtype=np.uint32)

    matches = match_exact_source_faces(
        source_vertices, source_faces, source_vertices, candidate_faces
    )

    np.testing.assert_array_equal(matches.source_survival_mask, [True, True])
    np.testing.assert_array_equal(matches.candidate_source_face_index, [0, 1, -1])
    assert matches.source_duplicate_face_keys == 1
    assert matches.candidate_duplicate_face_keys == 1


def test_cli_publishes_uncapped_survival_arrays_and_cross_route_comparison(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.glb"
    first = tmp_path / "first.glb"
    second = tmp_path / "second.glb"
    output_dir = tmp_path / "survival"
    report = tmp_path / "report.json"
    _write_assay_fixtures(source, first, second)

    result = _run(
        source,
        [("first", first), ("second", second)],
        output_dir,
        report,
    )

    assert result.returncode == 0, result.stderr
    data = json.loads(report.read_text())
    assert data["status"] == "completed"
    assert data["route"] == "authenticated-exact-source-face-survival-v1"
    assert data["harness"]["path"] == str(SCRIPT.resolve())
    assert data["harness"]["sha256"] == _sha256(SCRIPT)
    assert data["source"]["faces"] == 3
    assert data["candidates"]["first"]["exact_source_faces"] == 1
    assert data["candidates"]["first"]["generated_or_modified_faces"] == 0
    assert data["candidates"]["second"]["exact_source_faces"] == 1
    assert data["candidates"]["second"]["generated_or_modified_faces"] == 1
    assert data["pairwise"] == [
        {
            "left_label": "first",
            "right_label": "second",
            "intersection": 0,
            "union": 2,
            "jaccard": 0.0,
            "left_only": 1,
            "right_only": 1,
        }
    ]
    assert data["consensus"]["surviving_route_count_histogram"] == {
        "0": 1,
        "1": 2,
        "2": 0,
    }
    np.testing.assert_array_equal(
        np.load(output_dir / "first.source-survival-mask.npy"), [True, False, False]
    )
    np.testing.assert_array_equal(
        np.load(output_dir / "second.source-survival-mask.npy"), [False, True, False]
    )
    np.testing.assert_array_equal(
        np.load(output_dir / "second.candidate-source-face-index.npy"), [1, -1]
    )
    metrics = np.load(output_dir / "source-face-metrics.npz")
    assert set(metrics.files) == {
        "area",
        "centroid",
        "max_edge_length",
        "min_edge_length",
        "triangle_quality",
        "boundary_face_mask",
        "nonmanifold_face_mask",
    }
    assert metrics["area"].shape == (3,)
    assert data["primary_output_status"] == "validated"
    assert all(item["sha256"] for item in data["artifacts"])


def test_digest_mismatch_writes_failure_report_without_primary_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source.glb"
    first = tmp_path / "first.glb"
    second = tmp_path / "second.glb"
    output_dir = tmp_path / "survival"
    report = tmp_path / "report.json"
    _write_assay_fixtures(source, first, second)

    result = _run(
        source,
        [("first", first)],
        output_dir,
        report,
        source_digest="0" * 64,
    )

    assert result.returncode != 0
    data = json.loads(report.read_text())
    assert data["status"] == "failed"
    assert data["failure_phase"] == "authenticate_inputs"
    assert data["primary_output_status"] == "not_started"
    assert not output_dir.exists()


def test_output_collision_fails_without_overwriting_existing_evidence(tmp_path: Path) -> None:
    source = tmp_path / "source.glb"
    first = tmp_path / "first.glb"
    second = tmp_path / "second.glb"
    output_dir = tmp_path / "survival"
    report = tmp_path / "report.json"
    _write_assay_fixtures(source, first, second)
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("custody")

    result = _run(source, [("first", first)], output_dir, report)

    assert result.returncode != 0
    assert marker.read_text() == "custody"
    data = json.loads(report.read_text())
    assert data["failure_phase"] == "validate_paths"
    assert data["primary_output_status"] == "not_started"


def test_report_input_collision_preserves_input_and_reroutes_failure(tmp_path: Path) -> None:
    source = tmp_path / "source.glb"
    first = tmp_path / "first.glb"
    second = tmp_path / "second.glb"
    output_dir = tmp_path / "survival"
    _write_assay_fixtures(source, first, second)
    source_before = _sha256(source)

    result = _run(source, [("first", first)], output_dir, source)

    assert result.returncode != 0
    assert _sha256(source) == source_before
    safe_report = next(tmp_path.glob("source.glb.face-survival-error.*.json"))
    data = json.loads(safe_report.read_text())
    assert data["failure_phase"] == "validate_paths"
    assert data["report"]["requested_path"] == str(source)
    assert data["report"]["effective_path"] == str(safe_report)
    assert data["report"]["rerouted"] is True
    assert data["primary_output_status"] == "not_started"


def test_existing_report_is_preserved_and_failure_is_rerouted(tmp_path: Path) -> None:
    source = tmp_path / "source.glb"
    first = tmp_path / "first.glb"
    second = tmp_path / "second.glb"
    output_dir = tmp_path / "survival"
    report = tmp_path / "report.json"
    _write_assay_fixtures(source, first, second)
    report.write_text("existing evidence")

    result = _run(source, [("first", first)], output_dir, report)

    assert result.returncode != 0
    assert report.read_text() == "existing evidence"
    safe_report = next(tmp_path.glob("report.json.face-survival-error.*.json"))
    data = json.loads(safe_report.read_text())
    assert data["failure_phase"] == "validate_paths"
    assert data["report"]["rerouted"] is True


def test_concurrent_report_capture_is_preserved_and_reservation_reroutes(
    tmp_path: Path, monkeypatch,
) -> None:
    source = tmp_path / "source.glb"
    first = tmp_path / "first.glb"
    second = tmp_path / "second.glb"
    output_dir = tmp_path / "survival"
    report = tmp_path / "report.json"
    _write_assay_fixtures(source, first, second)
    args = SimpleNamespace(
        source=(str(source), _sha256(source)),
        candidate=[("first", str(first), _sha256(first))],
        output_dir=output_dir,
        report=report,
    )
    original_open = os.open
    injected = False

    def capture_before_open(path, flags, mode=0o777):
        nonlocal injected
        if Path(path) == report and not injected:
            injected = True
            report.write_text("competing evidence")
        return original_open(path, flags, mode)

    monkeypatch.setattr(RUNNER.os, "open", capture_before_open)

    custody = RUNNER._reserve_report_custody(args)
    try:
        assert report.read_text() == "competing evidence"
        assert custody.path != report
        assert custody.rerouted is True
        assert custody.path.parent == tmp_path
    finally:
        custody.close()


def test_report_below_output_fails_promptly_outside_output_on_first_run_and_retry(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.glb"
    first = tmp_path / "first.glb"
    second = tmp_path / "second.glb"
    output_dir = tmp_path / "survival"
    requested_report = output_dir / "report.json"
    _write_assay_fixtures(source, first, second)

    first_result = _run(source, [("first", first)], output_dir, requested_report)
    second_result = _run(source, [("first", first)], output_dir, requested_report)

    assert first_result.returncode != 0
    assert second_result.returncode != 0
    assert not output_dir.exists()
    failure_reports = sorted(tmp_path.glob("report.json.face-survival-error.*.json"))
    assert len(failure_reports) == 2
    assert all(json.loads(path.read_text())["failure_phase"] == "validate_paths" for path in failure_reports)
    assert all(json.loads(path.read_text())["report"]["rerouted"] is True for path in failure_reports)


def test_pairwise_records_preserve_all_pairs_for_delimiter_colliding_labels(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.glb"
    first = tmp_path / "first.glb"
    second = tmp_path / "second.glb"
    _write_assay_fixtures(source, first, second)
    candidates = []
    for index, label in enumerate(("a", "b__c", "a__b", "c")):
        path = tmp_path / f"candidate-{index}.glb"
        shutil.copyfile(first, path)
        candidates.append((label, path))
    output_dir = tmp_path / "survival"
    report = tmp_path / "report.json"

    result = _run(source, candidates, output_dir, report)

    assert result.returncode == 0, result.stderr
    pairwise = json.loads(report.read_text())["pairwise"]
    assert len(pairwise) == 6
    identities = {(record["left_label"], record["right_label"]) for record in pairwise}
    assert identities == {
        ("a", "b__c"),
        ("a", "a__b"),
        ("a", "c"),
        ("b__c", "a__b"),
        ("b__c", "c"),
        ("a__b", "c"),
    }
