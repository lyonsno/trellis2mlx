"""Contracts for the cumesh postprocess-only probe runner."""

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("scripts/run_cumesh_postprocess_probe.py")


def test_missing_raw_mesh_writes_failure_report_without_primary_output(tmp_path):
    raw_mesh = tmp_path / "missing.npz"
    output_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--raw-mesh",
            str(raw_mesh),
            "--output-dir",
            str(output_dir),
            "--target-faces",
            "10",
            "--device",
            "cpu",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert not (output_dir / "output.glb").exists()
    report_path = output_dir / "postprocess_probe_report.json"
    assert report_path.exists()

    report = json.loads(report_path.read_text())
    assert report["status"] == "error"
    assert report["route"] == "trellis2mlx_cumesh_postprocess_probe"
    assert report["phase"] == "load_inputs"
    assert report["raw_mesh"] == str(raw_mesh)
    assert report["primary_output_status"] == "not_produced"
    assert report["last_trustworthy_evidence"]["raw_mesh_exists"] is False
    assert report["forbidden_to_prove"] == [
        "full_trellis2_parity",
        "texture_bake_parity",
        "production_winding_closure",
        "image_conditioning_or_sampling_equivalence",
    ]
