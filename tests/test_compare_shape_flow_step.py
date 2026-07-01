import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_compare_shape_flow_step_writes_route_aware_prediction_report(tmp_path):
    mac_path = tmp_path / "mac-out" / "shape_flow_step0.npz"
    mlx_path = tmp_path / "mlx-out" / "checkpoints" / "shape_flow_step0.npz"
    _write_step(mac_path)
    _write_step(
        mlx_path,
        pred_v=np.array([[1.5, 2.5], [9.0, 9.0]], dtype=np.float32),
        coords=np.array([[0, 1, 2, 3], [0, 7, 8, 9]], dtype=np.int32),
    )
    mac_receipt = _write_receipt(
        tmp_path,
        "mac",
        mac_path,
        route=f"python run_official_trellis2.py --output-dir {mac_path.parent} --seed 42 --steps 12 --stop-after-shape-flow-step",
    )
    mlx_receipt = _write_receipt(
        tmp_path,
        "mlx",
        mlx_path,
        output_dir=mlx_path.parents[1],
        route=f"python generate.py --save-checkpoints {mlx_path.parent} --seed 42 --steps 12 --no-cascade --stop-after-shape-flow-step",
    )
    output_dir = tmp_path / "comparison"

    _run_compare(mac_path, mlx_path, mac_receipt, mlx_receipt, output_dir, check=True)

    report = json.loads((output_dir / "shape_flow_step_comparison_report.json").read_text())
    assert report["schema"] == "trellis2mlx.shape_flow_step_comparison.v1"
    assert report["route_proof_status"] == "candidate"
    assert report["coordinate_comparison"]["common_voxels"] == 1
    assert report["feature_comparison"]["pred_v_feats"]["n_common_compared"] == 1
    assert report["feature_comparison"]["pred_v_feats"]["channels"]["channel_000"]["abs_diff"]["mean"] == 0.5


def test_compare_shape_flow_step_rejects_failed_receipts(tmp_path):
    mac_path = _write_step(tmp_path / "mac-out" / "shape_flow_step0.npz")
    mlx_path = _write_step(tmp_path / "mlx-out" / "checkpoints" / "shape_flow_step0.npz")
    mac_receipt = _write_receipt(tmp_path, "mac", mac_path, status="failed", exit_code=1)
    mlx_receipt = _write_receipt(tmp_path, "mlx", mlx_path)
    output_dir = tmp_path / "comparison"

    result = _run_compare(mac_path, mlx_path, mac_receipt, mlx_receipt, output_dir)

    assert result.returncode != 0
    report = json.loads((output_dir / "shape_flow_step_comparison_report.json").read_text())
    assert report["comparison_status"] == "failed"
    assert report["failure_phase"] == "validate_routes"
    assert report["route_validation"]["trellis_mac"]["status"] == "rejected"
    assert "feature_comparison" not in report


def test_compare_shape_flow_step_rejects_artifact_not_implied_by_effective_route(tmp_path):
    mac_output = tmp_path / "mac-out"
    mlx_checkpoints = tmp_path / "mlx-out" / "checkpoints"
    _write_step(mac_output / "shape_flow_step0.npz")
    _write_step(mlx_checkpoints / "shape_flow_step0.npz")
    mac_path = _write_step(mac_output / "unbound_shape_flow_step0.npz")
    mlx_path = _write_step(tmp_path / "mlx-out" / "not-checkpoints" / "shape_flow_step0.npz")
    mac_receipt = _write_receipt(
        tmp_path,
        "mac",
        mac_path,
        output_dir=mac_output,
        route=f"python run_official_trellis2.py --output-dir {mac_output} --seed 42 --stop-after-shape-flow-step",
    )
    mlx_receipt = _write_receipt(
        tmp_path,
        "mlx",
        mlx_path,
        output_dir=tmp_path / "mlx-out",
        route=f"python generate.py --save-checkpoints {mlx_checkpoints} --seed 42 --stop-after-shape-flow-step",
    )
    output_dir = tmp_path / "comparison"

    result = _run_compare(mac_path, mlx_path, mac_receipt, mlx_receipt, output_dir)

    assert result.returncode != 0
    report = json.loads((output_dir / "shape_flow_step_comparison_report.json").read_text())
    assert report["comparison_status"] == "failed"
    assert report["failure_phase"] == "validate_routes"
    assert "artifact_not_implied_by_effective_route" in report["route_validation"]["trellis_mac"]["reasons"]
    assert "artifact_not_implied_by_effective_route" in report["route_validation"]["trellis_mlx"]["reasons"]


def _write_step(path, *, sample=None, pred_v=None, pred_x0=None, coords=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    coords = coords if coords is not None else np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32)
    sample = sample if sample is not None else np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    pred_v = pred_v if pred_v is not None else np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    pred_x0 = pred_x0 if pred_x0 is not None else pred_v + 1
    np.savez(path, sample_feats=sample, pred_v_feats=pred_v, pred_x0_feats=pred_x0, coords=coords)
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
                "effective_route": route or f"python {name}.py --seed 42 --stop-after-shape-flow-step",
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
            "scripts/compare_shape_flow_step.py",
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
