"""Contracts for cleanup-stage winding diagnostics."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path("scripts/probe_cleanup_orientation.py")


def _write_mesh(path: Path, vertices, faces):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
    )


def test_probe_writes_stepwise_orientation_report(tmp_path):
    mesh_path = tmp_path / "input.npz"
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
        ],
        dtype=np.int64,
    )
    _write_mesh(mesh_path, vertices, faces)

    output_dir = tmp_path / "probe"
    report_path = output_dir / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mesh",
            str(mesh_path),
            "--output-dir",
            str(output_dir),
            "--report",
            str(report_path),
            "--skip-visible",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text())
    assert report["schema"] == "trellis2mlx.cleanup_orientation_probe.v1"
    assert report["status"] == "ok"
    assert report["input_mesh"] == str(mesh_path)
    assert report["stages"]["input"]["faces"] == 2
    assert report["stages"]["after_adjacency_orient"]["changed_face_rows_from_previous"] == 1
    assert report["stages"]["after_radial_component_orient"]["faces"] == 2
    assert report["stages"]["after_residual_conflict_prune"]["edge_consistency"]["same_direction_conflict_edges"] == 0
    assert (output_dir / "after_adjacency_orient.npz").exists()


def test_probe_writes_failure_report_for_missing_input(tmp_path):
    output_dir = tmp_path / "probe"
    report_path = output_dir / "report.json"
    missing = tmp_path / "missing.npz"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mesh",
            str(missing),
            "--output-dir",
            str(output_dir),
            "--report",
            str(report_path),
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    report = json.loads(report_path.read_text())
    assert report["status"] == "error"
    assert report["phase"] == "load_inputs"
    assert report["last_trustworthy_evidence"]["input_exists"] is False
