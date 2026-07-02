import importlib
import json
from pathlib import Path
import subprocess


def test_mlx_stage_capture_runner_writes_failed_report_when_child_fails(monkeypatch, tmp_path):
    runner = importlib.import_module("scripts.run_mlx_stage_capture")

    image = tmp_path / "source.png"
    image.write_bytes(b"parser identity only")
    output_dir = tmp_path / "out"

    def fake_run(command, cwd=None, env=None, text=None, capture_output=None):
        assert "--stop-after-stage" in command
        assert "shape_slat" in command
        return subprocess.CompletedProcess(
            command,
            9,
            stdout="child stdout",
            stderr="child stderr",
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    rc = runner.main(
        [
            "--image",
            str(image),
            "--output-dir",
            str(output_dir),
            "--stop-after-stage",
            "shape_slat",
            "--seed",
            "42",
            "--steps",
            "8",
            "--resolution",
            "512",
            "--no-cascade",
            "--target-faces",
            "350000",
            "--texture-size",
            "4096",
        ]
    )

    assert rc == 9
    route = json.loads((output_dir / "route_identity.json").read_text())
    report = json.loads((output_dir / "run_report.json").read_text())
    assert route["route"]["family"] == "trellis2mlx/mlx"
    assert route["route"]["cascade"] is False
    assert route["route"]["resolution"] == 512
    assert route["route"]["steps"] == 8
    assert route["requested_stop"] == "shape_slat"
    assert report["status"] == "failed"
    assert report["failure_phase"] == "generate_subprocess"
    assert report["last_trustworthy_phase"] == "route_identity_written"
    assert report["primary_output_status"] == "missing"


def test_generate_exposes_stage_stop_checkpoints():
    text = (Path(__file__).resolve().parents[1] / "generate.py").read_text()

    assert "--stop-after-stage" in text
    assert 'choices=["conditioning", "sparse_coords", "shape_slat", "decoder_output", "mesh_raw"]' in text
    for stage in ("conditioning", "sparse_coords", "shape_slat", "decoder_output"):
        assert f'"{stage}"' in text
    assert text.count("save_checkpoint(") >= 8
