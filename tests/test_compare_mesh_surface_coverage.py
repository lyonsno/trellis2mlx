"""Contracts for source-to-candidate geometric surface coverage."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from trellmlx.glb_aabb_crop import write_geometry_glb


SCRIPT = Path(__file__).parents[1] / "scripts" / "compare_mesh_surface_coverage.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("surface_coverage_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_survival_report(
    path: Path, source: Path, survival_count: Path, route_count: int
) -> None:
    path.write_text(json.dumps({
        "schema": "trellis2mlx.exact_source_face_survival.v1",
        "route": "authenticated-exact-source-face-survival-v1",
        "status": "completed",
        "primary_output_status": "validated",
        "source": {"path": str(source), "sha256": _sha256(source)},
        "candidates": {f"route-{index}": {} for index in range(route_count)},
        "artifacts": [{"path": str(survival_count), "sha256": _sha256(survival_count)}],
    }))


def _run(
    source: Path,
    candidate: Path,
    survival_count: Path,
    output_dir: Path,
    report: Path,
    *,
    source_digest: str | None = None,
    route_count: int = 5,
) -> subprocess.CompletedProcess[str]:
    survival_report = survival_count.with_suffix(".report.json")
    _write_survival_report(survival_report, source, survival_count, route_count)
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source",
            str(source),
            source_digest or _sha256(source),
            "--candidate",
            "candidate",
            str(candidate),
            _sha256(candidate),
            "--survival-count",
            str(survival_count),
            _sha256(survival_count),
            "--survival-report",
            str(survival_report),
            _sha256(survival_report),
            "--route-count",
            str(route_count),
            "--output-dir",
            str(output_dir),
            "--report",
            str(report),
            "--chunk-faces",
            "1",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )


def _square() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return vertices, faces


def test_opposite_diagonal_retriangulation_preserves_complete_surface_coverage() -> None:
    vertices, source_faces = _square()
    candidate_faces = np.array([[0, 1, 3], [1, 2, 3]], dtype=np.int64)

    result = RUNNER.measure_surface_coverage(
        vertices,
        source_faces,
        vertices,
        candidate_faces,
        distance_ratio=0.01,
        normal_cosine=0.99,
        chunk_faces=1,
    )

    np.testing.assert_allclose(result.coverage_fraction, [1.0, 1.0])
    np.testing.assert_allclose(result.max_normalized_distance, [0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(result.min_normal_agreement, [1.0, 1.0], atol=1e-7)
    assert result.sample_normalized_distance.shape == (2, 3)
    assert result.sample_normal_agreement.shape == (2, 3)
    assert result.sample_candidate_face_index.shape == (2, 3)
    np.testing.assert_allclose(result.sample_normalized_distance, 0.0, atol=1e-7)
    np.testing.assert_allclose(result.sample_normal_agreement, 1.0, atol=1e-7)
    assert np.all(result.sample_candidate_face_index >= 0)


def test_missing_half_plane_is_not_laundered_as_complete_coverage() -> None:
    vertices, source_faces = _square()
    candidate_faces = np.array([[0, 1, 2]], dtype=np.int64)

    result = RUNNER.measure_surface_coverage(
        vertices,
        source_faces,
        vertices,
        candidate_faces,
        distance_ratio=0.01,
        normal_cosine=0.99,
        chunk_faces=2,
    )

    assert result.coverage_fraction[0] == 1.0
    assert result.coverage_fraction[1] < 1.0
    assert result.max_normalized_distance[1] > 0.01


def test_crossing_surface_with_wrong_normal_is_not_coverage() -> None:
    source_vertices = np.array(
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    candidate_vertices = np.array(
        [[0.0, -2.0, -2.0], [0.0, 2.0, -2.0], [0.0, 0.0, 2.0]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    result = RUNNER.measure_surface_coverage(
        source_vertices,
        faces,
        candidate_vertices,
        faces,
        distance_ratio=1.0,
        normal_cosine=0.9,
        chunk_faces=1,
    )

    assert result.min_normal_agreement[0] == 0.0
    assert result.coverage_fraction[0] == 0.0


def test_chunk_size_does_not_change_coverage_arrays() -> None:
    vertices, source_faces = _square()
    candidate_faces = np.array([[0, 1, 3], [1, 2, 3]], dtype=np.int64)

    one = RUNNER.measure_surface_coverage(
        vertices,
        source_faces,
        vertices,
        candidate_faces,
        distance_ratio=0.01,
        normal_cosine=0.99,
        chunk_faces=1,
    )
    all_at_once = RUNNER.measure_surface_coverage(
        vertices,
        source_faces,
        vertices,
        candidate_faces,
        distance_ratio=0.01,
        normal_cosine=0.99,
        chunk_faces=2,
    )

    np.testing.assert_array_equal(one.coverage_fraction, all_at_once.coverage_fraction)
    np.testing.assert_array_equal(one.max_normalized_distance, all_at_once.max_normalized_distance)
    np.testing.assert_array_equal(one.min_normal_agreement, all_at_once.min_normal_agreement)
    np.testing.assert_array_equal(
        one.sample_normalized_distance, all_at_once.sample_normalized_distance
    )
    np.testing.assert_array_equal(one.sample_normal_agreement, all_at_once.sample_normal_agreement)
    np.testing.assert_array_equal(
        one.sample_candidate_face_index, all_at_once.sample_candidate_face_index
    )


def test_degenerate_candidate_primitive_is_excluded_from_surface_coverage() -> None:
    vertices, source_faces = _square()
    candidate_faces = np.array([[0, 0, 1], [0, 1, 3], [1, 2, 3]], dtype=np.int64)

    result = RUNNER.measure_surface_coverage(
        vertices,
        source_faces,
        vertices,
        candidate_faces,
        distance_ratio=0.01,
        normal_cosine=0.99,
        chunk_faces=2,
    )

    assert result.candidate_degenerate_faces == 1
    np.testing.assert_allclose(result.coverage_fraction, [1.0, 1.0])


def test_cli_publishes_authenticated_raw_arrays_and_threshold_matrix(tmp_path: Path) -> None:
    vertices, source_faces = _square()
    candidate_faces = np.array([[0, 1, 3], [1, 2, 3]], dtype=np.uint32)
    source = tmp_path / "source.glb"
    candidate = tmp_path / "candidate.glb"
    survival = tmp_path / "survival.npy"
    output_dir = tmp_path / "coverage"
    report = tmp_path / "report.json"
    write_geometry_glb(source, vertices, source_faces.astype(np.uint32))
    write_geometry_glb(candidate, vertices, candidate_faces)
    np.save(survival, np.array([0, 5], dtype=np.uint16))

    result = _run(source, candidate, survival, output_dir, report)

    assert result.returncode == 0, result.stderr
    data = json.loads(report.read_text())
    assert data["status"] == "completed"
    assert data["primary_output_status"] == "validated"
    assert data["source"]["sha256"] == _sha256(source)
    assert data["candidate"]["sha256"] == _sha256(candidate)
    assert data["survival_count"]["sha256"] == _sha256(survival)
    assert data["survival_report"]["sha256"] == _sha256(survival.with_suffix(".report.json"))
    assert data["survival_count"]["routes"] == 5
    assert data["runtime"]["surface_query_backend"] == "libigl.AABB.squared_distance"
    assert data["effective_config"]["result_cap"] is None
    assert len(data["threshold_matrix"]) == 20
    assert data["threshold_matrix"][0]["classes"]["survives-none"]["source_faces"] == 1
    assert data["threshold_matrix"][0]["classes"]["survives-all"]["source_faces"] == 1
    np.testing.assert_allclose(np.load(output_dir / "sample-normalized-distance.npy"), 0.0)
    np.testing.assert_allclose(np.load(output_dir / "sample-normal-agreement.npy"), 1.0)
    assert np.load(output_dir / "sample-candidate-face-index.npy").shape == (2, 3)
    assert len(data["artifacts"]) == 3


def test_cli_rejects_survival_counts_outside_explicit_route_universe(tmp_path: Path) -> None:
    vertices, faces = _square()
    source = tmp_path / "source.glb"
    candidate = tmp_path / "candidate.glb"
    survival = tmp_path / "survival.npy"
    output_dir = tmp_path / "coverage"
    report = tmp_path / "report.json"
    write_geometry_glb(source, vertices, faces.astype(np.uint32))
    write_geometry_glb(candidate, vertices, faces.astype(np.uint32))
    np.save(survival, np.array([0, 5], dtype=np.uint16))

    result = _run(source, candidate, survival, output_dir, report, route_count=4)

    assert result.returncode == 1
    data = json.loads(report.read_text())
    assert data["failure_phase"] == "load_inputs"
    assert "route count" in data["error"]
    assert not output_dir.exists()


def test_cli_rejects_survival_report_from_another_source(tmp_path: Path) -> None:
    vertices, faces = _square()
    source = tmp_path / "source.glb"
    other_source = tmp_path / "other-source.glb"
    candidate = tmp_path / "candidate.glb"
    survival = tmp_path / "survival.npy"
    survival_report = tmp_path / "survival.report.json"
    output_dir = tmp_path / "coverage"
    report = tmp_path / "report.json"
    write_geometry_glb(source, vertices, faces.astype(np.uint32))
    shifted = vertices.copy()
    shifted[:, 2] = 1.0
    write_geometry_glb(other_source, shifted, faces.astype(np.uint32))
    write_geometry_glb(candidate, vertices, faces.astype(np.uint32))
    np.save(survival, np.array([0, 5], dtype=np.uint16))
    _write_survival_report(survival_report, other_source, survival, 5)

    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--source", str(source), _sha256(source),
            "--candidate", "candidate", str(candidate), _sha256(candidate),
            "--survival-count", str(survival), _sha256(survival),
            "--survival-report", str(survival_report), _sha256(survival_report),
            "--route-count", "5",
            "--output-dir", str(output_dir),
            "--report", str(report),
        ],
        text=True, capture_output=True, check=False, timeout=10,
    )

    assert result.returncode == 1
    data = json.loads(report.read_text())
    assert data["failure_phase"] == "validate_provenance"
    assert "source" in data["error"]
    assert not output_dir.exists()


def test_cli_hash_mismatch_fails_before_primary_output(tmp_path: Path) -> None:
    vertices, faces = _square()
    source = tmp_path / "source.glb"
    candidate = tmp_path / "candidate.glb"
    survival = tmp_path / "survival.npy"
    output_dir = tmp_path / "coverage"
    report = tmp_path / "report.json"
    write_geometry_glb(source, vertices, faces.astype(np.uint32))
    write_geometry_glb(candidate, vertices, faces.astype(np.uint32))
    np.save(survival, np.array([0, 5], dtype=np.uint16))

    result = _run(source, candidate, survival, output_dir, report, source_digest="0" * 64)

    assert result.returncode == 1
    data = json.loads(report.read_text())
    assert data["status"] == "failed"
    assert data["failure_phase"] == "authenticate_inputs"
    assert data["primary_output_status"] == "not_started"
    assert not output_dir.exists()


def test_cli_rejects_existing_output_and_preserves_it(tmp_path: Path) -> None:
    vertices, faces = _square()
    source = tmp_path / "source.glb"
    candidate = tmp_path / "candidate.glb"
    survival = tmp_path / "survival.npy"
    output_dir = tmp_path / "coverage"
    report = tmp_path / "report.json"
    write_geometry_glb(source, vertices, faces.astype(np.uint32))
    write_geometry_glb(candidate, vertices, faces.astype(np.uint32))
    np.save(survival, np.array([0, 5], dtype=np.uint16))
    output_dir.mkdir()
    marker = output_dir / "owned.txt"
    marker.write_text("keep")

    result = _run(source, candidate, survival, output_dir, report)

    assert result.returncode == 1
    assert marker.read_text() == "keep"
    data = json.loads(report.read_text())
    assert data["failure_phase"] == "validate_paths"
    assert data["primary_output_status"] == "not_started"


def test_report_input_collision_preserves_input_and_reroutes_failure(tmp_path: Path) -> None:
    vertices, faces = _square()
    source = tmp_path / "source.glb"
    candidate = tmp_path / "candidate.glb"
    survival = tmp_path / "survival.npy"
    output_dir = tmp_path / "coverage"
    write_geometry_glb(source, vertices, faces.astype(np.uint32))
    write_geometry_glb(candidate, vertices, faces.astype(np.uint32))
    np.save(survival, np.array([0, 5], dtype=np.uint16))
    source_before = _sha256(source)

    result = _run(source, candidate, survival, output_dir, source)

    assert result.returncode == 1
    assert _sha256(source) == source_before
    safe_report = next(tmp_path.glob("source.glb.surface-coverage-error.*.json"))
    data = json.loads(safe_report.read_text())
    assert data["failure_phase"] == "validate_paths"
    assert data["report"]["requested_path"] == str(source)
    assert data["report"]["effective_path"] == str(safe_report)
    assert data["report"]["rerouted"] is True
    assert data["primary_output_status"] == "not_started"


def test_existing_report_is_preserved_and_failure_is_rerouted(tmp_path: Path) -> None:
    vertices, faces = _square()
    source = tmp_path / "source.glb"
    candidate = tmp_path / "candidate.glb"
    survival = tmp_path / "survival.npy"
    output_dir = tmp_path / "coverage"
    report = tmp_path / "report.json"
    write_geometry_glb(source, vertices, faces.astype(np.uint32))
    write_geometry_glb(candidate, vertices, faces.astype(np.uint32))
    np.save(survival, np.array([0, 5], dtype=np.uint16))
    report.write_text("existing evidence")

    result = _run(source, candidate, survival, output_dir, report)

    assert result.returncode == 1
    assert report.read_text() == "existing evidence"
    safe_report = next(tmp_path.glob("report.json.surface-coverage-error.*.json"))
    data = json.loads(safe_report.read_text())
    assert data["failure_phase"] == "validate_paths"
    assert data["report"]["rerouted"] is True


def test_report_below_output_fails_outside_output_on_first_run_and_retry(tmp_path: Path) -> None:
    vertices, faces = _square()
    source = tmp_path / "source.glb"
    candidate = tmp_path / "candidate.glb"
    survival = tmp_path / "survival.npy"
    output_dir = tmp_path / "coverage"
    report = output_dir / "report.json"
    write_geometry_glb(source, vertices, faces.astype(np.uint32))
    write_geometry_glb(candidate, vertices, faces.astype(np.uint32))
    np.save(survival, np.array([0, 5], dtype=np.uint16))

    first = _run(source, candidate, survival, output_dir, report)
    second = _run(source, candidate, survival, output_dir, report)

    assert first.returncode == 1
    assert second.returncode == 1
    assert not output_dir.exists()
    reports = sorted(tmp_path.glob("report.json.surface-coverage-error.*.json"))
    assert len(reports) == 2
    assert all(json.loads(path.read_text())["failure_phase"] == "validate_paths" for path in reports)


def test_post_reservation_report_replacement_publishes_owned_terminal_failure(
    tmp_path: Path, monkeypatch,
) -> None:
    vertices, faces = _square()
    source = tmp_path / "source.glb"
    candidate = tmp_path / "candidate.glb"
    survival = tmp_path / "survival.npy"
    output_dir = tmp_path / "coverage"
    report = tmp_path / "report.json"
    write_geometry_glb(source, vertices, faces.astype(np.uint32))
    write_geometry_glb(candidate, vertices, faces.astype(np.uint32))
    np.save(survival, np.array([0, 5], dtype=np.uint16))
    args = SimpleNamespace(
        source=(str(source), _sha256(source)),
        candidate=("candidate", str(candidate), _sha256(candidate)),
        survival_count=(str(survival), _sha256(survival)),
        survival_report=(
            str(survival.with_suffix(".report.json")),
            "pending",
        ),
        route_count=5,
        output_dir=output_dir,
        report=report,
        chunk_faces=1,
    )
    _write_survival_report(Path(args.survival_report[0]), source, survival, 5)
    args.survival_report = (args.survival_report[0], _sha256(Path(args.survival_report[0])))
    original_write = RUNNER.ReportCustody.write
    replaced = False

    def replace_before_completed_write(custody, payload, *args, **kwargs):
        nonlocal replaced
        if payload["status"] == "completed" and not replaced:
            replaced = True
            report.unlink()
            report.write_text("competing completed evidence")
        return original_write(custody, payload, *args, **kwargs)

    monkeypatch.setattr(RUNNER.ReportCustody, "write", replace_before_completed_write)

    result = RUNNER.run(args)

    assert result == 1
    assert report.read_text() == "competing completed evidence"
    assert output_dir.is_dir()
    owned_reports = sorted(tmp_path.glob("report.json.surface-coverage-invocation.*.json"))
    assert len(owned_reports) == 1
    failure = json.loads(owned_reports[0].read_text())
    assert failure["status"] == "failed"
    assert failure["failure_phase"] == "report_custody"
    assert failure["primary_output_status"] == "validated"
    assert failure["report"]["lost_effective_path"] == str(report)


def test_corrupt_saved_array_fails_validation_without_publishing_output(
    tmp_path: Path, monkeypatch,
) -> None:
    vertices, faces = _square()
    source = tmp_path / "source.glb"
    candidate = tmp_path / "candidate.glb"
    survival = tmp_path / "survival.npy"
    output_dir = tmp_path / "coverage"
    report = tmp_path / "report.json"
    write_geometry_glb(source, vertices, faces.astype(np.uint32))
    write_geometry_glb(candidate, vertices, faces.astype(np.uint32))
    np.save(survival, np.array([0, 5], dtype=np.uint16))
    args = SimpleNamespace(
        source=(str(source), _sha256(source)),
        candidate=("candidate", str(candidate), _sha256(candidate)),
        survival_count=(str(survival), _sha256(survival)),
        survival_report=(
            str(survival.with_suffix(".report.json")),
            "pending",
        ),
        route_count=5,
        output_dir=output_dir,
        report=report,
        chunk_faces=1,
    )
    _write_survival_report(Path(args.survival_report[0]), source, survival, 5)
    args.survival_report = (args.survival_report[0], _sha256(Path(args.survival_report[0])))
    original_save = RUNNER.np.save

    def corrupt_distance(path, array, *args, **kwargs):
        if Path(path).name == "sample-normalized-distance.npy":
            array = array[:, :2]
        return original_save(path, array, *args, **kwargs)

    monkeypatch.setattr(RUNNER.np, "save", corrupt_distance)

    result = RUNNER.run(args)

    assert result == 1
    data = json.loads(report.read_text())
    assert data["failure_phase"] == "validate_outputs"
    assert data["primary_output_status"] == "not_started"
    assert not output_dir.exists()
