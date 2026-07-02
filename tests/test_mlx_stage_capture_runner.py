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


def test_mlx_stage_capture_runner_records_and_passes_shared_noise(monkeypatch, tmp_path):
    runner = importlib.import_module("scripts.run_mlx_stage_capture")

    image = tmp_path / "source.png"
    image.write_bytes(b"parser identity only")
    shared_noise = tmp_path / "shared_noise.npz"
    shared_noise.write_bytes(b"stable shared noise fixture")
    output_dir = tmp_path / "out"

    def fake_run(command, cwd=None, env=None, text=None, capture_output=None):
        assert "--shared-noise" in command
        assert str(shared_noise) in command
        checkpoint_dir = Path(command[command.index("--save-checkpoints") + 1])
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "sparse_coords.npz").write_bytes(b"artifact")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    rc = runner.main(
        [
            "--image",
            str(image),
            "--output-dir",
            str(output_dir),
            "--stop-after-stage",
            "sparse_coords",
            "--shared-noise",
            str(shared_noise),
        ]
    )

    assert rc == 0
    route = json.loads((output_dir / "route_identity.json").read_text())
    assert route["route"]["shared_noise_path"] == str(shared_noise)
    assert route["route"]["shared_noise_sha256"] is not None
    assert "--shared-noise" in route["command"]


def test_generate_exposes_stage_stop_checkpoints():
    text = (Path(__file__).resolve().parents[1] / "generate.py").read_text()

    assert "--stop-after-stage" in text
    assert (
        'choices=["conditioning", "sparse_coords", "sparse_flow_step", "sparse_internals", '
        '"shape_slat", "decoder_output", "mesh_raw"]'
    ) in text
    for stage in ("conditioning", "sparse_coords", "sparse_flow_step", "sparse_internals", "shape_slat", "decoder_output"):
        assert f'"{stage}"' in text
    assert text.count("save_checkpoint(") >= 8


def test_generate_exposes_shared_noise_for_sparse_witness():
    text = (Path(__file__).resolve().parents[1] / "generate.py").read_text()

    assert "--shared-noise" in text
    assert 'shared_noise["ss_noise"]' in text


def test_generate_exposes_sparse_internals_checkpoint():
    text = (Path(__file__).resolve().parents[1] / "generate.py").read_text()

    assert '"sparse_internals"' in text
    assert "z_s=np.array(z_s)" in text
    assert "logits=np.array(logits)" in text
    assert "decoded.astype(np.bool_)" in text


def test_generate_exposes_sparse_flow_step_checkpoint():
    text = (Path(__file__).resolve().parents[1] / "generate.py").read_text()

    assert '"sparse_flow_step"' in text
    for key in (
        "noise=np.array(noise)",
        "pred_pos=np.array(step_capture[\"pred_pos\"])",
        "pred_neg=np.array(step_capture[\"pred_neg\"])",
        "pred_cfg=np.array(step_capture[\"pred_cfg\"])",
        "std_ratio=np.array(step_capture[\"std_ratio\"])",
        "pred_final=np.array(step_capture[\"pred_final\"])",
        "sample_next=np.array(step_capture[\"sample_next\"])",
    ):
        assert key in text
