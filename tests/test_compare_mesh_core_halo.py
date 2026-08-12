"""Contracts for route-neutral core/halo mesh comparison evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from trellmlx.glb_aabb_crop import write_geometry_glb


SCRIPT = Path("scripts/compare_mesh_core_halo.py")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(path: Path) -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5], [6, 6, 6]], dtype=np.uint32)
    write_geometry_glb(path, vertices, faces)


def _run(mesh: Path, report: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mesh",
            "fixture",
            str(mesh),
            _sha256(mesh),
            "--core-min",
            "-0.1",
            "-0.1",
            "-0.1",
            "--core-max",
            "1.1",
            "1.1",
            "0.1",
            "--chunk-faces",
            "2",
            "--report",
            str(report),
            *extra,
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_reports_authenticated_core_halo_geometry_metrics(tmp_path: Path) -> None:
    mesh = tmp_path / "mesh.glb"
    report = tmp_path / "report.json"
    _write_fixture(mesh)

    result = _run(mesh, report)

    assert result.returncode == 0, result.stderr
    data = json.loads(report.read_text())
    assert data["status"] == "completed"
    assert data["route"] == "authenticated-glb-core-halo-comparison-v1"
    assert data["effective_config"]["classification"] == "triangle-centroid-in-core-aabb"
    assert data["effective_config"]["chunk_faces"] == 2
    measured = data["meshes"]["fixture"]
    assert measured["identity_match"] is True
    assert measured["faces"] == 3
    assert measured["vertices"] == 7
    assert measured["degenerate_faces"] == 1
    assert measured["total_area"] == 2.5
    assert measured["core_centroid"]["faces"] == 2
    assert measured["core_centroid"]["degenerate_faces"] == 1
    assert measured["core_centroid"]["area"] == 0.5
    assert measured["halo_centroid"]["faces"] == 1
    assert measured["halo_centroid"]["area"] == 2.0
    assert data["primary_output_status"] == "validated"


def test_expected_digest_mismatch_fails_without_publishing_metrics(tmp_path: Path) -> None:
    mesh = tmp_path / "mesh.glb"
    report = tmp_path / "report.json"
    _write_fixture(mesh)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mesh",
            "fixture",
            str(mesh),
            "0" * 64,
            "--core-min",
            "-1",
            "-1",
            "-1",
            "--core-max",
            "1",
            "1",
            "1",
            "--report",
            str(report),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    data = json.loads(report.read_text())
    assert data["status"] == "failed"
    assert data["failure_phase"] == "authenticate_inputs"
    assert data["meshes"] is None
    assert data["primary_output_status"] == "not_started"


def test_report_input_collision_preserves_mesh_and_reroutes_failure(tmp_path: Path) -> None:
    mesh = tmp_path / "mesh.glb"
    _write_fixture(mesh)
    before = _sha256(mesh)

    result = _run(mesh, mesh)

    assert result.returncode != 0
    assert _sha256(mesh) == before
    safe_report = tmp_path / "mesh.glb.comparison-error.json"
    data = json.loads(safe_report.read_text())
    assert data["failure_phase"] == "validate_paths"
    assert data["report"]["rerouted"] is True
    assert data["last_trustworthy_evidence"]["inputs_preserved"] is True


def test_duplicate_labels_fail_loud(tmp_path: Path) -> None:
    mesh = tmp_path / "mesh.glb"
    report = tmp_path / "report.json"
    _write_fixture(mesh)
    digest = _sha256(mesh)

    result = _run(mesh, report, "--mesh", "fixture", str(mesh), digest)

    assert result.returncode != 0
    data = json.loads(report.read_text())
    assert data["failure_phase"] == "validate_request"
    assert "duplicate mesh labels" in data["error"]


def test_malformed_glb_leaves_a_failure_report(tmp_path: Path) -> None:
    mesh = tmp_path / "malformed.glb"
    report = tmp_path / "report.json"
    mesh.write_bytes(b"not a GLB")

    result = _run(mesh, report)

    assert result.returncode != 0
    data = json.loads(report.read_text())
    assert data["status"] == "failed"
    assert data["failure_phase"] == "parse_glb"
    assert data["last_trustworthy_evidence"]["authenticated_labels"] == ["fixture"]
    assert data["primary_output_status"] == "not_started"
