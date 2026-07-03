"""Contracts for the saved decoder-output extraction probe."""

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("scripts/probe_decoder_extraction.py")


def test_missing_decoder_output_writes_failure_report(tmp_path):
    decoder_output = tmp_path / "missing_decoder_output.npz"
    output_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--decoder-output",
            str(decoder_output),
            "--output-dir",
            str(output_dir),
            "--image-size",
            "128",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    report_path = output_dir / "decoder_extraction_probe_report.json"
    assert report_path.exists()
    assert not (output_dir / "variants").exists()

    report = json.loads(report_path.read_text())
    assert report["status"] == "error"
    assert report["route"] == "trellis2mlx_decoder_extraction_probe"
    assert report["phase"] == "load_inputs"
    assert report["decoder_output"] == str(decoder_output)
    assert report["primary_output_status"] == "not_produced"
    assert report["last_trustworthy_evidence"]["decoder_output_exists"] is False
    assert report["forbidden_to_prove"] == [
        "full_trellis2_parity",
        "microsoft_cuda_parity",
        "production_winding_closure",
        "postprocess_or_texture_bake_behavior",
    ]
