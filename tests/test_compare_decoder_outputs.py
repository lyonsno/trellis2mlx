import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_compare_decoder_outputs_writes_route_aware_full_common_report(tmp_path):
    """The decoder comparator is an evidence harness, so it must not sample silently."""
    n_common = 100_001
    common_coords = np.column_stack(
        [
            np.zeros(n_common, dtype=np.int32),
            np.zeros(n_common, dtype=np.int32),
            np.zeros(n_common, dtype=np.int32),
            np.arange(n_common, dtype=np.int32),
        ]
    )
    mac_feats = np.tile(
        np.array([[0.1, 0.2, 0.3, -1.0, 1.0, -1.0, 0.5]], dtype=np.float32),
        (n_common, 1),
    )
    mlx_feats = mac_feats.copy()
    mlx_feats[:, 3] *= -1

    mac_path = tmp_path / "mac_decoder_output.npz"
    mlx_path = tmp_path / "mlx_decoder_output.npz"
    np.savez(mac_path, feats=mac_feats, coords=common_coords)
    np.savez(
        mlx_path,
        feats=np.vstack([mlx_feats, np.zeros((1, 7), dtype=np.float32)]),
        coords=np.vstack(
            [
                common_coords,
                np.array([[0, 9, 9, 9]], dtype=np.int32),
            ]
        ),
    )

    mac_receipt = tmp_path / "mac_receipt.json"
    mlx_receipt = tmp_path / "mlx_receipt.json"
    mac_receipt.write_text(
        json.dumps(
            {
                "job_id": "mac-job",
                "job_type": "trellis2_official_512",
                "status": "done",
                "output_dir": str(tmp_path),
                "exit_code": 0,
                "effective_route": "python run_official_trellis2.py --pipeline-type 512",
                "effective_cwd": "/trellis-mac/TRELLIS.2",
            }
        )
    )
    mlx_receipt.write_text(
        json.dumps(
            {
                "job_id": "mlx-job",
                "job_type": "trellis2mlx_decoder_capture",
                "status": "done",
                "output_dir": str(tmp_path),
                "exit_code": 0,
                "effective_route": "python generate.py --resolution 512 --steps 8 --no-cascade",
                "effective_cwd": "/trellis2mlx",
            }
        )
    )

    output_dir = tmp_path / "comparison"
    _run_compare(mac_path, mlx_path, mac_receipt, mlx_receipt, output_dir, check=True)

    report = json.loads((output_dir / "comparison_report.json").read_text())

    assert report["schema"] == "trellis2mlx.decoder_comparison.v1"
    assert report["artifacts"]["trellis_mac"]["sha256"]
    assert report["artifacts"]["trellis_mlx"]["sha256"]
    assert report["routes"]["trellis_mac"]["job_id"] == "mac-job"
    assert report["routes"]["trellis_mlx"]["job_id"] == "mlx-job"
    assert report["coordinate_comparison"]["common_voxels"] == n_common
    assert report["feature_comparison"]["sample_policy"] == "full_common_set"
    assert report["feature_comparison"]["n_common_compared"] == n_common
    assert report["evidence_use_class"] == "bounded_candidate"
    assert "sampler_seed" in report["unmatched_variables"]


def test_compare_decoder_outputs_uses_intersection_when_same_count_coords_differ(tmp_path):
    mac_path = tmp_path / "mac_decoder_output.npz"
    mlx_path = tmp_path / "mlx_decoder_output.npz"
    np.savez(
        mac_path,
        feats=np.array([[1, 0, 0, 1, 1, 1, 0], [9, 0, 0, 1, 1, 1, 0]], dtype=np.float32),
        coords=np.array([[0, 0, 0, 0], [0, 1, 1, 1]], dtype=np.int32),
    )
    np.savez(
        mlx_path,
        feats=np.array([[1, 0, 0, 1, 1, 1, 0], [9, 0, 0, 1, 1, 1, 0]], dtype=np.float32),
        coords=np.array([[0, 0, 0, 0], [0, 2, 2, 2]], dtype=np.int32),
    )
    mac_receipt = _write_receipt(tmp_path, "mac", mac_path)
    mlx_receipt = _write_receipt(tmp_path, "mlx", mlx_path)
    output_dir = tmp_path / "comparison"

    _run_compare(mac_path, mlx_path, mac_receipt, mlx_receipt, output_dir, check=True)

    report = json.loads((output_dir / "comparison_report.json").read_text())
    assert report["coordinate_comparison"]["mode"] == "set_intersection"
    assert report["coordinate_comparison"]["common_voxels"] == 1
    assert report["feature_comparison"]["sample_policy"] == "full_common_set"
    assert report["feature_comparison"]["n_common_compared"] == 1


def test_compare_decoder_outputs_overwrites_stale_report_on_artifact_failure(tmp_path):
    mac_path = tmp_path / "mac_decoder_output.npz"
    mlx_path = tmp_path / "mlx_decoder_output.npz"
    np.savez(mac_path, coords=np.array([[0, 0, 0, 0]], dtype=np.int32))
    np.savez(
        mlx_path,
        feats=np.ones((1, 7), dtype=np.float32),
        coords=np.array([[0, 0, 0, 0]], dtype=np.int32),
    )
    mac_receipt = _write_receipt(tmp_path, "mac", mac_path)
    mlx_receipt = _write_receipt(tmp_path, "mlx", mlx_path)
    output_dir = tmp_path / "comparison"
    output_dir.mkdir()
    (output_dir / "comparison_report.json").write_text(json.dumps({"status": "stale"}))

    result = _run_compare(mac_path, mlx_path, mac_receipt, mlx_receipt, output_dir)

    assert result.returncode != 0
    report = json.loads((output_dir / "comparison_report.json").read_text())
    assert report["comparison_status"] == "failed"
    assert report["failure_phase"] == "load_artifacts"
    assert report["failure_message"]


def test_compare_decoder_outputs_rejects_failed_receipts(tmp_path):
    mac_path = _write_decoder(tmp_path / "mac_decoder_output.npz")
    mlx_path = _write_decoder(tmp_path / "mlx_decoder_output.npz")
    mac_receipt = _write_receipt(tmp_path, "mac", mac_path, status="failed", exit_code=1)
    mlx_receipt = _write_receipt(tmp_path, "mlx", mlx_path)
    output_dir = tmp_path / "comparison"

    result = _run_compare(mac_path, mlx_path, mac_receipt, mlx_receipt, output_dir)

    assert result.returncode != 0
    report = json.loads((output_dir / "comparison_report.json").read_text())
    assert report["comparison_status"] == "failed"
    assert report["failure_phase"] == "validate_routes"
    assert report["route_validation"]["trellis_mac"]["status"] == "rejected"
    assert report["route_validation"]["trellis_mac"]["evidence_use_class"] == "negative_evidence"
    assert "feature_comparison" not in report


def test_compare_decoder_outputs_rejects_artifact_outside_receipt_output_dir(tmp_path):
    mac_path = _write_decoder(tmp_path / "outside" / "mac_decoder_output.npz")
    mlx_path = _write_decoder(tmp_path / "mlx_decoder_output.npz")
    mac_receipt = _write_receipt(tmp_path, "mac", mac_path, output_dir=tmp_path / "mac-output")
    mlx_receipt = _write_receipt(tmp_path, "mlx", mlx_path)
    output_dir = tmp_path / "comparison"

    result = _run_compare(mac_path, mlx_path, mac_receipt, mlx_receipt, output_dir)

    assert result.returncode != 0
    report = json.loads((output_dir / "comparison_report.json").read_text())
    assert report["comparison_status"] == "failed"
    assert report["failure_phase"] == "validate_routes"
    assert report["route_validation"]["trellis_mac"]["status"] == "rejected"
    assert "artifact_outside_receipt_output_dir" in report["route_validation"]["trellis_mac"]["reasons"]


def test_compare_decoder_outputs_keeps_route_proof_candidate_when_variables_unmatched(tmp_path):
    mac_path = _write_decoder(tmp_path / "mac-out" / "decoder_output.npz")
    mlx_path = _write_decoder(tmp_path / "mlx-out" / "checkpoints" / "decoder_output.npz")
    mac_receipt = _write_receipt(
        tmp_path,
        "mac",
        mac_path,
        route=(
            "python run_official_trellis2.py --output-dir "
            f"{mac_path.parent} --pipeline-type 512 --seed 42 --steps 12"
        ),
        env={"PYTHONPATH": ".", "ATTN_BACKEND": "sdpa"},
    )
    mlx_receipt = _write_receipt(
        tmp_path,
        "mlx",
        mlx_path,
        route=(
            "python generate.py --save-checkpoints "
            f"{mlx_path.parent} --resolution 512 --no-cascade --seed 42 --steps 12"
        ),
        env={"PYTHONPATH": "."},
    )
    output_dir = tmp_path / "comparison"

    _run_compare(mac_path, mlx_path, mac_receipt, mlx_receipt, output_dir, check=True)

    report = json.loads((output_dir / "comparison_report.json").read_text())
    assert report["unmatched_variables"]
    assert report["route_proof_status"] == "candidate"


def test_compare_decoder_outputs_rejects_artifact_not_implied_by_effective_route(tmp_path):
    mac_output = tmp_path / "mac-out"
    mlx_checkpoints = tmp_path / "mlx-out" / "checkpoints"
    implied_mac = mac_output / "decoder_output.npz"
    implied_mlx = mlx_checkpoints / "decoder_output.npz"
    _write_decoder(implied_mac)
    _write_decoder(implied_mlx)
    mac_path = _write_decoder(mac_output / "unbound_decoder_output.npz")
    mlx_path = _write_decoder(tmp_path / "mlx-out" / "not-checkpoints" / "decoder_output.npz")
    mac_receipt = _write_receipt(
        tmp_path,
        "mac",
        mac_path,
        output_dir=mac_output,
        route=f"python run_official_trellis2.py --output-dir {mac_output} --seed 42 --steps 12",
    )
    mlx_receipt = _write_receipt(
        tmp_path,
        "mlx",
        mlx_path,
        output_dir=tmp_path / "mlx-out",
        route=f"python generate.py --save-checkpoints {mlx_checkpoints} --seed 42 --steps 12",
    )
    output_dir = tmp_path / "comparison"

    result = _run_compare(mac_path, mlx_path, mac_receipt, mlx_receipt, output_dir)

    assert result.returncode != 0
    report = json.loads((output_dir / "comparison_report.json").read_text())
    assert report["comparison_status"] == "failed"
    assert report["failure_phase"] == "validate_routes"
    assert "artifact_not_implied_by_effective_route" in report["route_validation"]["trellis_mac"]["reasons"]
    assert "artifact_not_implied_by_effective_route" in report["route_validation"]["trellis_mlx"]["reasons"]


def _write_decoder(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        feats=np.ones((1, 7), dtype=np.float32),
        coords=np.array([[0, 0, 0, 0]], dtype=np.int32),
    )
    return path


def _write_receipt(
    tmp_path,
    name,
    artifact_path,
    *,
    status="done",
    exit_code=0,
    output_dir=None,
    route=None,
    env=None,
):
    output_dir = output_dir or artifact_path.parent
    receipt_path = tmp_path / f"{name}_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "job_id": f"{name}-job",
                "job_type": f"{name}-job-type",
                "status": status,
                "input_path": str(tmp_path / "input.png"),
                "output_dir": str(output_dir),
                "exit_code": exit_code,
                "failure_phase": None if status == "done" else "execution",
                "error_message": None if status == "done" else "boom",
                "ignored_params": None,
                "effective_route": route or f"python {name}.py --seed 42 --steps 12",
                "effective_cwd": f"/{name}",
                "effective_env": env,
            }
        )
    )
    return receipt_path


def _run_compare(mac_path, mlx_path, mac_receipt, mlx_receipt, output_dir, *, check=False):
    repo_root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [
            sys.executable,
            "scripts/compare_decoder_outputs.py",
            "--trellis-mac",
            str(mac_path),
            "--trellis-mlx",
            str(mlx_path),
            "--trellis-mac-receipt",
            str(mac_receipt),
            "--trellis-mlx-receipt",
            str(mlx_receipt),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        check=check,
    )
