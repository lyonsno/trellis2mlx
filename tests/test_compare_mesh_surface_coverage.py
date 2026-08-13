"""Contracts for source-to-candidate geometric surface coverage."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
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
    path: Path,
    source: Path,
    survival_count: Path,
    route_count: int,
    *,
    candidate_label: str,
    candidate_path: Path,
    candidate_digest: str | None = None,
) -> None:
    candidates = {
        candidate_label: {
            "path": str(candidate_path),
            "sha256": candidate_digest or _sha256(candidate_path),
        }
    }
    candidates.update({
        f"filler-{index}": {
            "path": str(candidate_path),
            "sha256": candidate_digest or _sha256(candidate_path),
        }
        for index in range(route_count - 1)
    })
    path.write_text(json.dumps({
        "schema": "trellis2mlx.exact_source_face_survival.v1",
        "route": "authenticated-exact-source-face-survival-v1",
        "status": "completed",
        "primary_output_status": "validated",
        "source": {"path": str(source), "sha256": _sha256(source)},
        "candidates": candidates,
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
    candidate_label: str = "candidate",
    reported_candidate_label: str | None = None,
    reported_candidate: Path | None = None,
    reported_candidate_digest: str | None = None,
) -> subprocess.CompletedProcess[str]:
    survival_report = survival_count.with_suffix(".report.json")
    _write_survival_report(
        survival_report,
        source,
        survival_count,
        route_count,
        candidate_label=reported_candidate_label or candidate_label,
        candidate_path=reported_candidate or candidate,
        candidate_digest=reported_candidate_digest,
    )
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source",
            str(source),
            source_digest or _sha256(source),
            "--candidate",
            candidate_label,
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
    assert data["candidate"]["upstream_identity"] == {
        "label": "candidate",
        "path": str(candidate),
        "sha256": _sha256(candidate),
    }
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
    _write_survival_report(
        survival_report,
        other_source,
        survival,
        5,
        candidate_label="candidate",
        candidate_path=candidate,
    )

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


def test_cli_rejects_foreign_candidate_not_admitted_by_upstream_report(tmp_path: Path) -> None:
    vertices, faces = _square()
    source = tmp_path / "source.glb"
    candidate = tmp_path / "candidate.glb"
    foreign = tmp_path / "foreign.glb"
    survival = tmp_path / "survival.npy"
    output_dir = tmp_path / "coverage"
    report = tmp_path / "report.json"
    write_geometry_glb(source, vertices, faces.astype(np.uint32))
    write_geometry_glb(candidate, vertices, faces.astype(np.uint32))
    shifted = vertices.copy()
    shifted[:, 2] = 0.25
    write_geometry_glb(foreign, shifted, faces.astype(np.uint32))
    np.save(survival, np.array([0, 5], dtype=np.uint16))

    result = _run(
        source,
        foreign,
        survival,
        output_dir,
        report,
        reported_candidate=candidate,
    )

    assert result.returncode == 1
    data = json.loads(report.read_text())
    assert data["failure_phase"] == "validate_provenance"
    assert "candidate" in data["error"]
    assert not output_dir.exists()


def test_cli_rejects_correct_candidate_under_wrong_route_label(tmp_path: Path) -> None:
    vertices, faces = _square()
    source = tmp_path / "source.glb"
    candidate = tmp_path / "candidate.glb"
    survival = tmp_path / "survival.npy"
    output_dir = tmp_path / "coverage"
    report = tmp_path / "report.json"
    write_geometry_glb(source, vertices, faces.astype(np.uint32))
    write_geometry_glb(candidate, vertices, faces.astype(np.uint32))
    np.save(survival, np.array([0, 5], dtype=np.uint16))

    result = _run(
        source,
        candidate,
        survival,
        output_dir,
        report,
        candidate_label="wrong-route",
        reported_candidate_label="candidate",
    )

    assert result.returncode == 1
    assert json.loads(report.read_text())["failure_phase"] == "validate_provenance"
    assert not output_dir.exists()


def test_cli_rejects_correct_label_with_wrong_upstream_digest(tmp_path: Path) -> None:
    vertices, faces = _square()
    source = tmp_path / "source.glb"
    candidate = tmp_path / "candidate.glb"
    survival = tmp_path / "survival.npy"
    output_dir = tmp_path / "coverage"
    report = tmp_path / "report.json"
    write_geometry_glb(source, vertices, faces.astype(np.uint32))
    write_geometry_glb(candidate, vertices, faces.astype(np.uint32))
    np.save(survival, np.array([0, 5], dtype=np.uint16))

    result = _run(
        source,
        candidate,
        survival,
        output_dir,
        report,
        reported_candidate_digest="0" * 64,
    )

    assert result.returncode == 1
    assert json.loads(report.read_text())["failure_phase"] == "validate_provenance"
    assert not output_dir.exists()


def test_cli_rejects_placeholder_candidate_records(tmp_path: Path) -> None:
    vertices, faces = _square()
    source = tmp_path / "source.glb"
    candidate = tmp_path / "candidate.glb"
    survival = tmp_path / "survival.npy"
    survival_report = tmp_path / "survival.report.json"
    output_dir = tmp_path / "coverage"
    report = tmp_path / "report.json"
    write_geometry_glb(source, vertices, faces.astype(np.uint32))
    write_geometry_glb(candidate, vertices, faces.astype(np.uint32))
    np.save(survival, np.array([0, 5], dtype=np.uint16))
    _write_survival_report(
        survival_report,
        source,
        survival,
        5,
        candidate_label="candidate",
        candidate_path=candidate,
    )
    payload = json.loads(survival_report.read_text())
    payload["candidates"] = {f"placeholder-{index}": {} for index in range(5)}
    survival_report.write_text(json.dumps(payload))

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
    assert json.loads(report.read_text())["failure_phase"] == "validate_provenance"
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
    _write_survival_report(
        Path(args.survival_report[0]),
        source,
        survival,
        5,
        candidate_label="candidate",
        candidate_path=candidate,
    )
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
    _write_survival_report(
        Path(args.survival_report[0]),
        source,
        survival,
        5,
        candidate_label="candidate",
        candidate_path=candidate,
    )
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


def _run_args(tmp_path: Path) -> tuple[SimpleNamespace, Path, Path]:
    vertices, faces = _square()
    source = tmp_path / "source.glb"
    candidate = tmp_path / "candidate.glb"
    survival = tmp_path / "survival.npy"
    survival_report = tmp_path / "survival.report.json"
    output_dir = tmp_path / "coverage"
    report = tmp_path / "report.json"
    write_geometry_glb(source, vertices, faces.astype(np.uint32))
    write_geometry_glb(candidate, vertices, faces.astype(np.uint32))
    np.save(survival, np.array([0, 5], dtype=np.uint16))
    _write_survival_report(
        survival_report,
        source,
        survival,
        5,
        candidate_label="candidate",
        candidate_path=candidate,
    )
    return SimpleNamespace(
        source=(str(source), _sha256(source)),
        candidate=("candidate", str(candidate), _sha256(candidate)),
        survival_count=(str(survival), _sha256(survival)),
        survival_report=(str(survival_report), _sha256(survival_report)),
        route_count=5,
        output_dir=output_dir,
        report=report,
        chunk_faces=1,
    ), output_dir, report


def test_competing_empty_destination_survives_exclusive_publication(
    tmp_path: Path, monkeypatch,
) -> None:
    args, output_dir, report = _run_args(tmp_path)
    original_publish = RUNNER._publish_output_directory

    def capture_destination(temporary_dir, destination):
        destination.mkdir()
        return original_publish(temporary_dir, destination)

    monkeypatch.setattr(RUNNER, "_publish_output_directory", capture_destination)

    result = RUNNER.run(args)

    assert result == 1
    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []
    data = json.loads(report.read_text())
    assert data["failure_phase"] == "publish"
    assert data["primary_output_status"] == "not_started"


def test_published_array_replacement_fails_output_custody(tmp_path: Path, monkeypatch) -> None:
    args, output_dir, report = _run_args(tmp_path)
    original_validate = RUNNER._validate_published_outputs

    def replace_array(custody, expected):
        target = output_dir / "sample-normalized-distance.npy"
        target.unlink()
        np.save(target, np.zeros((2, 2), dtype=np.float32))
        return original_validate(custody, expected)

    monkeypatch.setattr(RUNNER, "_validate_published_outputs", replace_array)

    result = RUNNER.run(args)

    assert result == 1
    data = json.loads(report.read_text())
    assert data["failure_phase"] == "output_custody"
    assert data["primary_output_status"] == "published_custody_lost"


def test_published_directory_replacement_fails_output_custody(tmp_path: Path, monkeypatch) -> None:
    args, output_dir, report = _run_args(tmp_path)
    original_validate = RUNNER._validate_published_outputs

    def replace_directory(custody, expected):
        captured = output_dir.with_name("captured")
        output_dir.rename(captured)
        output_dir.mkdir()
        return original_validate(custody, expected)

    monkeypatch.setattr(RUNNER, "_validate_published_outputs", replace_directory)

    result = RUNNER.run(args)

    assert result == 1
    data = json.loads(report.read_text())
    assert data["failure_phase"] == "output_custody"
    assert data["primary_output_status"] == "published_custody_lost"


def test_byte_identical_directory_replacement_before_rebind_fails_custody(
    tmp_path: Path, monkeypatch,
) -> None:
    args, output_dir, report = _run_args(tmp_path)
    original_rebind = RUNNER._rebind_published_outputs

    def replace_before_rebind(custody, destination):
        captured = destination.with_name("captured-original")
        destination.rename(captured)
        shutil.copytree(captured, destination)
        return original_rebind(custody, destination)

    monkeypatch.setattr(RUNNER, "_rebind_published_outputs", replace_before_rebind)

    result = RUNNER.run(args)

    assert result == 1
    data = json.loads(report.read_text())
    assert data["failure_phase"] == "output_custody"
    assert data["primary_output_status"] == "published_custody_lost"


def test_incomplete_directory_replacement_before_rebind_fails_custody(
    tmp_path: Path, monkeypatch,
) -> None:
    args, output_dir, report = _run_args(tmp_path)
    original_rebind = RUNNER._rebind_published_outputs

    def replace_before_rebind(custody, destination):
        destination.rename(destination.with_name("captured-original"))
        destination.mkdir()
        return original_rebind(custody, destination)

    monkeypatch.setattr(RUNNER, "_rebind_published_outputs", replace_before_rebind)

    result = RUNNER.run(args)

    assert result == 1
    data = json.loads(report.read_text())
    assert data["failure_phase"] == "output_custody"
    assert data["primary_output_status"] == "published_custody_lost"


def test_completed_artifact_digests_are_postpublication_validated_bytes(tmp_path: Path) -> None:
    args, output_dir, report = _run_args(tmp_path)

    result = RUNNER.run(args)

    assert result == 0
    data = json.loads(report.read_text())
    assert data["primary_output_status"] == "validated"
    assert data["output_custody"]["directory_identity"]
    for artifact in data["artifacts"]:
        path = Path(artifact["path"])
        assert path.parent == output_dir
        assert artifact["sha256"] == _sha256(path)
        assert artifact["validated_through_published_path"] is True


def test_same_inode_mutation_during_completed_report_loses_output_custody(
    tmp_path: Path, monkeypatch,
) -> None:
    args, output_dir, report = _run_args(tmp_path)
    original_write = RUNNER.ReportCustody.write
    mutated = False

    def mutate_before_completed_write(custody, payload, *write_args, **write_kwargs):
        nonlocal mutated
        if payload["status"] == "completed" and not mutated:
            mutated = True
            target = output_dir / "sample-normalized-distance.npy"
            with target.open("r+b") as stream:
                stream.seek(-1, os.SEEK_END)
                final_byte = stream.read(1)
                stream.seek(-1, os.SEEK_END)
                stream.write(bytes([final_byte[0] ^ 0x01]))
        return original_write(custody, payload, *write_args, **write_kwargs)

    monkeypatch.setattr(RUNNER.ReportCustody, "write", mutate_before_completed_write)

    result = RUNNER.run(args)

    assert result == 1
    data = json.loads(report.read_text())
    assert data["status"] == "failed"
    assert data["failure_phase"] == "output_custody"
    assert data["primary_output_status"] == "published_custody_lost"
