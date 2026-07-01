import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_compare_conditioning_writes_route_aware_tensor_report(tmp_path):
    mac_path = tmp_path / "mac-out" / "conditioning.npz"
    mlx_path = tmp_path / "mlx-out" / "checkpoints" / "conditioning.npz"
    _write_conditioning(mac_path, cond=np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32))
    _write_conditioning(mlx_path, cond=np.array([[[1.5, 2.5], [2.0, 6.0]]], dtype=np.float32))
    mac_receipt = _write_receipt(
        tmp_path,
        "mac",
        mac_path,
        route=f"python run_official_trellis2.py --output-dir {mac_path.parent} --pipeline-type 512 --seed 42 --stop-after-conditioning",
    )
    mlx_receipt = _write_receipt(
        tmp_path,
        "mlx",
        mlx_path,
        output_dir=mlx_path.parents[1],
        route=f"python generate.py --save-checkpoints {mlx_path.parent} --resolution 512 --no-cascade --seed 42 --stop-after-conditioning",
    )
    output_dir = tmp_path / "comparison"

    _run_compare(mac_path, mlx_path, mac_receipt, mlx_receipt, output_dir, check=True)

    report = json.loads((output_dir / "conditioning_comparison_report.json").read_text())
    assert report["schema"] == "trellis2mlx.conditioning_comparison.v1"
    assert report["comparison_status"] == "completed"
    assert report["route_proof_status"] == "candidate"
    assert report["evidence_use_class"] == "bounded_candidate"
    assert report["tensor_comparison"]["shape_match"] is True
    assert report["tensor_comparison"]["cond"]["abs_diff"]["mean"] == 1.0
    assert "image_feature_extraction" in report["unmatched_variables"]


def test_compare_conditioning_rejects_failed_receipts(tmp_path):
    mac_path = _write_conditioning(tmp_path / "mac-out" / "conditioning.npz")
    mlx_path = _write_conditioning(tmp_path / "mlx-out" / "checkpoints" / "conditioning.npz")
    mac_receipt = _write_receipt(tmp_path, "mac", mac_path, status="failed", exit_code=1)
    mlx_receipt = _write_receipt(tmp_path, "mlx", mlx_path)
    output_dir = tmp_path / "comparison"

    result = _run_compare(mac_path, mlx_path, mac_receipt, mlx_receipt, output_dir)

    assert result.returncode != 0
    report = json.loads((output_dir / "conditioning_comparison_report.json").read_text())
    assert report["comparison_status"] == "failed"
    assert report["failure_phase"] == "validate_routes"
    assert report["route_validation"]["trellis_mac"]["status"] == "rejected"
    assert "tensor_comparison" not in report


def test_compare_conditioning_rejects_artifact_not_implied_by_effective_route(tmp_path):
    mac_output = tmp_path / "mac-out"
    mlx_checkpoints = tmp_path / "mlx-out" / "checkpoints"
    _write_conditioning(mac_output / "conditioning.npz")
    _write_conditioning(mlx_checkpoints / "conditioning.npz")
    mac_path = _write_conditioning(mac_output / "unbound_conditioning.npz")
    mlx_path = _write_conditioning(tmp_path / "mlx-out" / "not-checkpoints" / "conditioning.npz")
    mac_receipt = _write_receipt(
        tmp_path,
        "mac",
        mac_path,
        output_dir=mac_output,
        route=f"python run_official_trellis2.py --output-dir {mac_output} --seed 42 --stop-after-conditioning",
    )
    mlx_receipt = _write_receipt(
        tmp_path,
        "mlx",
        mlx_path,
        output_dir=tmp_path / "mlx-out",
        route=f"python generate.py --save-checkpoints {mlx_checkpoints} --seed 42 --stop-after-conditioning",
    )
    output_dir = tmp_path / "comparison"

    result = _run_compare(mac_path, mlx_path, mac_receipt, mlx_receipt, output_dir)

    assert result.returncode != 0
    report = json.loads((output_dir / "conditioning_comparison_report.json").read_text())
    assert report["comparison_status"] == "failed"
    assert report["failure_phase"] == "validate_routes"
    assert "artifact_not_implied_by_effective_route" in report["route_validation"]["trellis_mac"]["reasons"]
    assert "artifact_not_implied_by_effective_route" in report["route_validation"]["trellis_mlx"]["reasons"]


def test_compare_conditioning_overwrites_stale_report_on_artifact_failure(tmp_path):
    mac_path = tmp_path / "mac-out" / "conditioning.npz"
    mlx_path = _write_conditioning(tmp_path / "mlx-out" / "checkpoints" / "conditioning.npz")
    mac_path.parent.mkdir(parents=True)
    np.savez(mac_path, cond=np.ones((1, 1, 2), dtype=np.float32))
    mac_receipt = _write_receipt(tmp_path, "mac", mac_path)
    mlx_receipt = _write_receipt(tmp_path, "mlx", mlx_path)
    output_dir = tmp_path / "comparison"
    output_dir.mkdir()
    (output_dir / "conditioning_comparison_report.json").write_text(json.dumps({"status": "stale"}))

    result = _run_compare(mac_path, mlx_path, mac_receipt, mlx_receipt, output_dir)

    assert result.returncode != 0
    report = json.loads((output_dir / "conditioning_comparison_report.json").read_text())
    assert report["comparison_status"] == "failed"
    assert report["failure_phase"] == "load_artifacts"
    assert report["failure_message"]


def _write_conditioning(path, cond=None, neg_cond=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    cond = cond if cond is not None else np.ones((1, 1, 2), dtype=np.float32)
    neg_cond = neg_cond if neg_cond is not None else np.zeros_like(cond)
    np.savez(path, cond=cond, neg_cond=neg_cond)
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
                "effective_route": route or f"python {name}.py --seed 42 --stop-after-conditioning",
                "effective_cwd": f"/{name}",
                "effective_env": {"PYTHONPATH": "."},
            }
        )
    )
    return receipt_path


def _run_compare(mac_path, mlx_path, mac_receipt, mlx_receipt, output_dir, *, check=False):
    repo_root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [
            sys.executable,
            "scripts/compare_conditioning.py",
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
