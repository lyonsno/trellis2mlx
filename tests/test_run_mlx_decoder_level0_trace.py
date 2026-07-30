import hashlib
import json

import numpy as np


def test_local_trace_failure_before_primary_writes_durable_phase_report(tmp_path):
    from scripts.run_mlx_decoder_level0_trace import main

    missing_input = tmp_path / "missing-input.npz"
    checkpoint = tmp_path / "decoder.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    output_npz = tmp_path / "trace.npz"
    output_json = tmp_path / "trace.json"

    rc = main(
        [
            "--shape-slat-sample",
            str(missing_input),
            "--expected-shape-slat-sha256",
            "a" * 64,
            "--shape-decoder-checkpoint",
            str(checkpoint),
            "--expected-checkpoint-sha256",
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "--expected-repo-commit",
            "b" * 40,
            "--torso-dtype",
            "fp16",
            "--output-npz",
            str(output_npz),
            "--output-json",
            str(output_json),
        ]
    )

    report = json.loads(output_json.read_text())
    assert rc == 1
    assert report["schema"] == "trellis2mlx.decoder_level0_trace_run.v1"
    assert report["status"] == "failed"
    assert report["failure_phase"] == "input_validation"
    assert report["last_trustworthy_phase"] == "request_received"
    assert report["effective_route"] is None
    assert report["primary"]["status"] == "not_written"
    assert not output_npz.exists()


def test_local_trace_rejects_stale_input_digest_before_model_load(tmp_path):
    from scripts.run_mlx_decoder_level0_trace import main

    sample = tmp_path / "input.npz"
    np.savez(
        sample,
        feats=np.ones((2, 32), dtype=np.float32),
        coords=np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32),
    )
    checkpoint = tmp_path / "decoder.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    output_npz = tmp_path / "trace.npz"
    output_npz.write_bytes(b"stale-primary")
    output_json = tmp_path / "trace.json"

    rc = main(
        [
            "--shape-slat-sample",
            str(sample),
            "--expected-shape-slat-sha256",
            "0" * 64,
            "--shape-decoder-checkpoint",
            str(checkpoint),
            "--expected-checkpoint-sha256",
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "--expected-repo-commit",
            "b" * 40,
            "--torso-dtype",
            "fp16",
            "--output-npz",
            str(output_npz),
            "--output-json",
            str(output_json),
        ]
    )

    report = json.loads(output_json.read_text())
    assert rc == 1
    assert report["failure_phase"] == "input_validation"
    assert "shape SLat digest mismatch" in report["error"]
    assert report["stale_primary_invalidated"] is True
    assert not output_npz.exists()
