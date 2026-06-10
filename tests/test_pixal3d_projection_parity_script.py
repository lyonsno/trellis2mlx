"""Contracts for the Pixal3D projection parity diagnostic."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path("scripts/pixal3d_projection_parity.py")


def test_projection_parity_script_writes_reference_formula_metrics(tmp_path):
    coords_path = tmp_path / "coords.npz"
    np.savez(
        coords_path,
        lr_coords=np.array(
            [
                [0, 1, 2],
                [1, 2, 3],
                [3, 1, 0],
                [2, 3, 1],
            ],
            dtype=np.int32,
        ),
    )
    report_path = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--feature-source",
            "synthetic",
            "--grid-resolution",
            "4",
            "--image-size",
            "64",
            "--patch-size",
            "16",
            "--channels",
            "7",
            "--coords-npz",
            str(coords_path),
            "--coords-key",
            "lr_coords",
            "--coord-count",
            "4",
            "--report",
            str(report_path),
        ],
        cwd=Path.cwd(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text())
    assert report["status"] == "ok"
    assert report["projection"]["mode"] == "native"
    assert report["features"]["source"] == "synthetic"
    assert report["coords"]["selected_count"] == 4
    assert report["metrics"]["selected_rows"]["max_abs"] < 5e-4
    assert report["metrics"]["selected_rows"]["max_abs_over_expected_abs_max"] < 1e-6
    assert report["metrics"]["full_projection"]["max_abs"] < 5e-4
    assert report["coordinate_order_sweep"]["canonical_order"] == "xyz"
    assert "xzy" in report["coordinate_order_sweep"]["permutations"]


def test_projection_parity_script_failure_report_names_missing_coords_key(tmp_path):
    coords_path = tmp_path / "coords.npz"
    np.savez(coords_path, other=np.zeros((1, 3), dtype=np.int32))
    report_path = tmp_path / "missing-key-report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--feature-source",
            "synthetic",
            "--grid-resolution",
            "4",
            "--coords-npz",
            str(coords_path),
            "--coords-key",
            "lr_coords",
            "--report",
            str(report_path),
        ],
        cwd=Path.cwd(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode != 0
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    assert report["failure"]["phase"] == "load_coords"
    assert "lr_coords" in report["failure"]["error"]
