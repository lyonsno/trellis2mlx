"""Contracts for the fast-simplification collapse-prefix assay."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import trimesh


SCRIPT = Path("scripts/run_fast_simplification_prefix_assay.py")


def _load_runner():
    spec = importlib.util.spec_from_file_location("fast_prefix_assay_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_grid(path: Path, side: int = 12) -> None:
    xs, ys = np.meshgrid(
        np.linspace(-1.0, 1.0, side),
        np.linspace(-1.0, 1.0, side),
        indexing="xy",
    )
    vertices = np.column_stack(
        [xs.ravel(), ys.ravel(), 0.08 * np.sin(xs.ravel() * 4) * np.cos(ys.ravel() * 3)]
    ).astype(np.float32)
    faces: list[list[int]] = []
    for y in range(side - 1):
        for x in range(side - 1):
            a = y * side + x
            b = a + 1
            c = a + side
            d = c + 1
            faces.extend(([a, b, d], [a, d, c]))
    trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int32),
        process=False,
    ).export(path, file_type="glb")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _run(
    source: Path,
    output_dir: Path,
    report: Path,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(source),
            "--output-dir",
            str(output_dir),
            "--report",
            str(report),
            "--target-faces",
            "160",
            "80",
            "--repeats",
            "2",
            *extra,
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )


def test_assay_records_repeat_identity_and_cross_target_prefix(tmp_path: Path) -> None:
    source = tmp_path / "grid.glb"
    output_dir = tmp_path / "assay"
    report = tmp_path / "assay.report.json"
    _write_grid(source)

    result = _run(source, output_dir, report)

    assert result.returncode == 0, result.stderr
    data = json.loads(report.read_text())
    assert data["status"] == "completed"
    assert data["route"]["id"] == "fast-simplification-collapse-prefix-v1"
    assert data["route"]["package"] == "fast-simplification"
    assert data["route"]["version"]
    assert data["source"]["sha256"] == _sha256(source)
    assert data["request"]["target_faces"] == [160, 80]
    assert data["request"]["repeats"] == 2
    assert data["effective_config"]["face_limit"] is None
    assert data["repeat_stability"]["all_exact"] is True
    assert data["target_contract"]["all_targets_satisfied"] is True
    assert data["target_contract"]["unsatisfied_targets"] == []
    assert data["prefix_relations"][0]["higher_target"] == 160
    assert data["prefix_relations"][0]["lower_target"] == 80
    assert isinstance(data["prefix_relations"][0]["exact_prefix"], bool)

    for target in data["targets"]:
        assert len(target["runs"]) == 2
        assert target["target_satisfied"] is True
        assert target["runs"][0]["collapse_sha256"] == target["runs"][1]["collapse_sha256"]
        assert target["runs"][0]["mesh_sha256"] == target["runs"][1]["mesh_sha256"]
        for run in target["runs"]:
            assert run["requested_faces"] == target["requested_faces"]
            assert run["target_satisfied"] is (run["achieved_faces"] <= run["requested_faces"])
            assert run["achieved_faces"] <= data["source"]["faces"]
            assert Path(run["mesh_path"]).is_file()
            assert Path(run["collapse_path"]).is_file()


def test_invalid_target_fails_before_outputs_with_durable_report(tmp_path: Path) -> None:
    source = tmp_path / "grid.glb"
    output_dir = tmp_path / "assay"
    report = tmp_path / "assay.report.json"
    _write_grid(source)

    result = _run(source, output_dir, report, "--target-faces", "9999")

    assert result.returncode != 0
    assert report.is_file()
    data = json.loads(report.read_text())
    assert data["status"] == "failed"
    assert data["failure_phase"] == "validate_targets"
    assert data["primary_output_status"] == "not_started"
    assert not list(output_dir.glob("*.glb")) if output_dir.exists() else True


def test_report_source_collision_preserves_source_and_reroutes_failure(tmp_path: Path) -> None:
    source = tmp_path / "grid.glb"
    output_dir = tmp_path / "assay"
    safe_report = tmp_path / "grid.glb.assay-error.json"
    _write_grid(source)
    source_hash = _sha256(source)

    result = _run(source, output_dir, source)

    assert result.returncode != 0
    assert _sha256(source) == source_hash
    assert safe_report.is_file()
    data = json.loads(safe_report.read_text())
    assert data["failure_phase"] == "validate_paths"
    assert data["report"]["requested_path"] == str(source)
    assert data["report"]["effective_path"] == str(safe_report)


def test_report_temporary_source_collision_preserves_source(tmp_path: Path) -> None:
    report = tmp_path / "assay.report.json"
    source = tmp_path / "assay.report.json.tmp"
    output_dir = tmp_path / "assay"
    safe_report = tmp_path / "assay.report.json.assay-error.json"
    _write_grid(source)
    source_hash = _sha256(source)

    result = _run(source, output_dir, report)

    assert result.returncode != 0
    assert _sha256(source) == source_hash
    assert safe_report.is_file()
    data = json.loads(safe_report.read_text())
    assert data["failure_phase"] == "validate_paths"
    assert data["report"]["requested_path"] == str(report)
    assert data["report"]["effective_path"] == str(safe_report)


def test_missing_source_replaces_stale_success_with_failure_report(tmp_path: Path) -> None:
    source = tmp_path / "missing.glb"
    output_dir = tmp_path / "assay"
    report = tmp_path / "assay.report.json"
    report.write_text(json.dumps({"status": "completed"}))

    result = _run(source, output_dir, report)

    assert result.returncode != 0
    data = json.loads(report.read_text())
    assert data["status"] == "failed"
    assert data["failure_phase"] == "load_source"
    assert data["primary_output_status"] == "not_started"


def test_obstructed_legacy_report_temporary_cannot_preserve_stale_success(
    tmp_path: Path,
) -> None:
    source = tmp_path / "grid.glb"
    output_dir = tmp_path / "assay"
    report = tmp_path / "assay.report.json"
    legacy_temporary = tmp_path / "assay.report.json.tmp"
    _write_grid(source)
    report.write_text(json.dumps({"status": "completed", "route": {"id": "stale"}}))
    legacy_temporary.mkdir()

    result = _run(source, output_dir, report)

    assert result.returncode == 0, result.stderr
    data = json.loads(report.read_text())
    assert data["status"] == "completed"
    assert data["route"]["id"] == "fast-simplification-collapse-prefix-v1"
    assert legacy_temporary.is_dir()


def test_failure_after_completed_run_records_partial_cleanup_and_last_run(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "grid.glb"
    output_dir = tmp_path / "assay"
    report = tmp_path / "assay.report.json"
    _write_grid(source)
    runner = _load_runner()
    real_write = runner.write_geometry_glb
    write_count = 0

    def fail_second_mesh(*args, **kwargs):
        nonlocal write_count
        write_count += 1
        if write_count == 2:
            raise RuntimeError("injected second mesh failure")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(runner, "write_geometry_glb", fail_second_mesh)

    returncode = runner.main(
        [
            "--input",
            str(source),
            "--output-dir",
            str(output_dir),
            "--report",
            str(report),
            "--target-faces",
            "160",
            "80",
            "--repeats",
            "2",
        ]
    )

    assert returncode != 0
    data = json.loads(report.read_text())
    assert data["status"] == "failed"
    assert data["failure_phase"] == "simplify"
    assert data["primary_output_status"] == "partial_removed"
    assert data["partial_output_cleanup"]["removed"] is True
    assert data["partial_output_cleanup"]["artifact_count"] == 2
    assert data["last_trustworthy_evidence"]["target"] == 160
    assert data["last_trustworthy_evidence"]["repeat"] == 1
    assert data["last_trustworthy_evidence"]["mesh_sha256"]
    assert data["last_trustworthy_evidence"]["collapse_file_sha256"]
    assert not output_dir.exists()
