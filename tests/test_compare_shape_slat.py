import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_compare_shape_slat_writes_route_aware_common_feature_report(tmp_path):
    mac_path = tmp_path / "mac-out" / "shape_slat.npz"
    mlx_path = tmp_path / "mlx-out" / "checkpoints" / "shape_slat.npz"
    coords_common = np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32)
    _write_slat(
        mac_path,
        feats=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        coords=coords_common,
    )
    _write_slat(
        mlx_path,
        feats=np.array([[1.5, 2.5, 3.5], [9.0, 9.0, 9.0]], dtype=np.float32),
        coords=np.array([[0, 1, 2, 3], [0, 7, 8, 9]], dtype=np.int32),
    )
    mac_receipt = _write_receipt(
        tmp_path,
        "mac",
        mac_path,
        route=f"python run_official_trellis2.py --output-dir {mac_path.parent} --pipeline-type 512 --seed 42 --steps 12 --stop-after-shape-slat",
    )
    mlx_receipt = _write_receipt(
        tmp_path,
        "mlx",
        mlx_path,
        output_dir=mlx_path.parents[1],
        route=f"python generate.py --save-checkpoints {mlx_path.parent} --resolution 512 --no-cascade --seed 42 --steps 12 --stop-after-shape-slat",
    )
    output_dir = tmp_path / "comparison"

    _run_compare(mac_path, mlx_path, mac_receipt, mlx_receipt, output_dir, check=True)

    report = json.loads((output_dir / "shape_slat_comparison_report.json").read_text())
    assert report["schema"] == "trellis2mlx.shape_slat_comparison.v1"
    assert report["route_proof_status"] == "candidate"
    assert report["evidence_use_class"] == "bounded_candidate"
    assert report["coordinate_comparison"]["common_voxels"] == 1
    assert report["coordinate_comparison"]["only_trellis_mac"] == 1
    assert report["coordinate_comparison"]["only_trellis_mlx"] == 1
    assert report["feature_comparison"]["sample_policy"] == "full_common_spatial_set"
    assert report["feature_comparison"]["n_common_compared"] == 1
    assert report["feature_comparison"]["channels"]["channel_000"]["abs_diff"]["mean"] == 0.5
    assert "image_feature_extraction" in report["unmatched_variables"]
    assert "flow_sampler_semantics" in report["unmatched_variables"]


def test_compare_shape_slat_rejects_failed_receipts(tmp_path):
    mac_path = _write_slat(tmp_path / "mac-out" / "shape_slat.npz")
    mlx_path = _write_slat(tmp_path / "mlx-out" / "checkpoints" / "shape_slat.npz")
    mac_receipt = _write_receipt(tmp_path, "mac", mac_path, status="failed", exit_code=1)
    mlx_receipt = _write_receipt(tmp_path, "mlx", mlx_path)
    output_dir = tmp_path / "comparison"

    result = _run_compare(mac_path, mlx_path, mac_receipt, mlx_receipt, output_dir)

    assert result.returncode != 0
    report = json.loads((output_dir / "shape_slat_comparison_report.json").read_text())
    assert report["comparison_status"] == "failed"
    assert report["failure_phase"] == "validate_routes"
    assert report["route_validation"]["trellis_mac"]["status"] == "rejected"
    assert "feature_comparison" not in report


def test_compare_shape_slat_rejects_artifact_not_implied_by_effective_route(tmp_path):
    mac_output = tmp_path / "mac-out"
    mlx_checkpoints = tmp_path / "mlx-out" / "checkpoints"
    _write_slat(mac_output / "shape_slat.npz")
    _write_slat(mlx_checkpoints / "shape_slat.npz")
    mac_path = _write_slat(mac_output / "unbound_shape_slat.npz")
    mlx_path = _write_slat(tmp_path / "mlx-out" / "not-checkpoints" / "shape_slat.npz")
    mac_receipt = _write_receipt(
        tmp_path,
        "mac",
        mac_path,
        output_dir=mac_output,
        route=f"python run_official_trellis2.py --output-dir {mac_output} --seed 42 --steps 12 --stop-after-shape-slat",
    )
    mlx_receipt = _write_receipt(
        tmp_path,
        "mlx",
        mlx_path,
        output_dir=tmp_path / "mlx-out",
        route=f"python generate.py --save-checkpoints {mlx_checkpoints} --seed 42 --steps 12 --stop-after-shape-slat",
    )
    output_dir = tmp_path / "comparison"

    result = _run_compare(mac_path, mlx_path, mac_receipt, mlx_receipt, output_dir)

    assert result.returncode != 0
    report = json.loads((output_dir / "shape_slat_comparison_report.json").read_text())
    assert report["comparison_status"] == "failed"
    assert report["failure_phase"] == "validate_routes"
    assert "artifact_not_implied_by_effective_route" in report["route_validation"]["trellis_mac"]["reasons"]
    assert "artifact_not_implied_by_effective_route" in report["route_validation"]["trellis_mlx"]["reasons"]


def test_compare_shape_slat_overwrites_stale_report_on_artifact_failure(tmp_path):
    mac_path = tmp_path / "mac-out" / "shape_slat.npz"
    mlx_path = _write_slat(tmp_path / "mlx-out" / "checkpoints" / "shape_slat.npz")
    mac_path.parent.mkdir(parents=True)
    np.savez(mac_path, coords=np.array([[0, 0, 0, 0]], dtype=np.int32))
    mac_receipt = _write_receipt(tmp_path, "mac", mac_path)
    mlx_receipt = _write_receipt(tmp_path, "mlx", mlx_path)
    output_dir = tmp_path / "comparison"
    output_dir.mkdir()
    (output_dir / "shape_slat_comparison_report.json").write_text(json.dumps({"status": "stale"}))

    result = _run_compare(mac_path, mlx_path, mac_receipt, mlx_receipt, output_dir)

    assert result.returncode != 0
    report = json.loads((output_dir / "shape_slat_comparison_report.json").read_text())
    assert report["comparison_status"] == "failed"
    assert report["failure_phase"] == "load_artifacts"
    assert report["failure_message"]


def _write_slat(path, feats=None, coords=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    feats = feats if feats is not None else np.ones((1, 3), dtype=np.float32)
    coords = coords if coords is not None else np.array([[0, 0, 0, 0]], dtype=np.int32)
    np.savez(path, feats=feats, coords=coords)
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
                "effective_route": route or f"python {name}.py --seed 42 --steps 12 --stop-after-shape-slat",
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
            "scripts/compare_shape_slat.py",
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
