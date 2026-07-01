import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_compare_sparse_coords_writes_route_aware_set_report(tmp_path):
    mac_path = tmp_path / "mac-out" / "sparse_coords.npz"
    mlx_path = tmp_path / "mlx-out" / "checkpoints" / "sparse_coords.npz"
    _write_sparse(mac_path, np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32))
    _write_sparse(
        mlx_path,
        np.array([[0, 1, 2, 3], [0, 7, 8, 9]], dtype=np.int32),
        include_lr=True,
    )
    mac_receipt = _write_receipt(
        tmp_path,
        "mac",
        mac_path,
        route=f"python run_official_trellis2.py --output-dir {mac_path.parent} --pipeline-type 512 --seed 42 --steps 12 --stop-after-sparse",
    )
    mlx_receipt = _write_receipt(
        tmp_path,
        "mlx",
        mlx_path,
        output_dir=mlx_path.parents[1],
        route=f"python generate.py --save-checkpoints {mlx_path.parent} --resolution 512 --no-cascade --seed 42 --steps 12 --stop-after-sparse",
    )
    output_dir = tmp_path / "comparison"

    _run_compare(mac_path, mlx_path, mac_receipt, mlx_receipt, output_dir, check=True)

    report = json.loads((output_dir / "sparse_coordinate_comparison_report.json").read_text())
    assert report["schema"] == "trellis2mlx.sparse_coordinate_comparison.v1"
    assert report["artifacts"]["trellis_mac"]["sha256"]
    assert report["artifacts"]["trellis_mlx"]["sha256"]
    assert report["routes"]["trellis_mac"]["job_id"] == "mac-job"
    assert report["routes"]["trellis_mlx"]["job_id"] == "mlx-job"
    assert report["route_proof_status"] == "candidate"
    assert report["evidence_use_class"] == "bounded_candidate"
    assert "image_feature_extraction" in report["unmatched_variables"]
    assert "flow_sampler_semantics" in report["unmatched_variables"]
    assert report["coordinate_comparison"]["common_voxels"] == 1
    assert report["coordinate_comparison"]["only_trellis_mac"] == 1
    assert report["coordinate_comparison"]["only_trellis_mlx"] == 1
    assert report["coordinate_comparison"]["jaccard_similarity"] == 1 / 3
    assert report["coordinate_comparison"]["trellis_mac"]["raw_rows"] == 2
    assert report["coordinate_comparison"]["trellis_mlx"]["raw_rows"] == 2


def test_compare_sparse_coords_rejects_failed_receipts(tmp_path):
    mac_path = _write_sparse(tmp_path / "mac-out" / "sparse_coords.npz")
    mlx_path = _write_sparse(tmp_path / "mlx-out" / "checkpoints" / "sparse_coords.npz")
    mac_receipt = _write_receipt(tmp_path, "mac", mac_path, status="failed", exit_code=1)
    mlx_receipt = _write_receipt(tmp_path, "mlx", mlx_path)
    output_dir = tmp_path / "comparison"

    result = _run_compare(mac_path, mlx_path, mac_receipt, mlx_receipt, output_dir)

    assert result.returncode != 0
    report = json.loads((output_dir / "sparse_coordinate_comparison_report.json").read_text())
    assert report["comparison_status"] == "failed"
    assert report["failure_phase"] == "validate_routes"
    assert report["route_validation"]["trellis_mac"]["status"] == "rejected"
    assert "coordinate_comparison" not in report


def test_compare_sparse_coords_rejects_artifact_not_implied_by_effective_route(tmp_path):
    mac_output = tmp_path / "mac-out"
    mlx_checkpoints = tmp_path / "mlx-out" / "checkpoints"
    implied_mac = _write_sparse(mac_output / "sparse_coords.npz")
    implied_mlx = _write_sparse(mlx_checkpoints / "sparse_coords.npz")
    mac_path = _write_sparse(mac_output / "unbound_sparse_coords.npz")
    mlx_path = _write_sparse(tmp_path / "mlx-out" / "not-checkpoints" / "sparse_coords.npz")
    assert implied_mac.exists()
    assert implied_mlx.exists()
    mac_receipt = _write_receipt(
        tmp_path,
        "mac",
        mac_path,
        output_dir=mac_output,
        route=f"python run_official_trellis2.py --output-dir {mac_output} --seed 42 --steps 12 --stop-after-sparse",
    )
    mlx_receipt = _write_receipt(
        tmp_path,
        "mlx",
        mlx_path,
        output_dir=tmp_path / "mlx-out",
        route=f"python generate.py --save-checkpoints {mlx_checkpoints} --seed 42 --steps 12 --stop-after-sparse",
    )
    output_dir = tmp_path / "comparison"

    result = _run_compare(mac_path, mlx_path, mac_receipt, mlx_receipt, output_dir)

    assert result.returncode != 0
    report = json.loads((output_dir / "sparse_coordinate_comparison_report.json").read_text())
    assert report["comparison_status"] == "failed"
    assert report["failure_phase"] == "validate_routes"
    assert "artifact_not_implied_by_effective_route" in report["route_validation"]["trellis_mac"]["reasons"]
    assert "artifact_not_implied_by_effective_route" in report["route_validation"]["trellis_mlx"]["reasons"]


def test_compare_sparse_coords_overwrites_stale_report_on_artifact_failure(tmp_path):
    mac_path = tmp_path / "mac-out" / "sparse_coords.npz"
    mlx_path = _write_sparse(tmp_path / "mlx-out" / "checkpoints" / "sparse_coords.npz")
    mac_path.parent.mkdir(parents=True)
    np.savez(mac_path, not_coords=np.array([[0, 0, 0, 0]], dtype=np.int32))
    mac_receipt = _write_receipt(tmp_path, "mac", mac_path)
    mlx_receipt = _write_receipt(tmp_path, "mlx", mlx_path)
    output_dir = tmp_path / "comparison"
    output_dir.mkdir()
    (output_dir / "sparse_coordinate_comparison_report.json").write_text(json.dumps({"status": "stale"}))

    result = _run_compare(mac_path, mlx_path, mac_receipt, mlx_receipt, output_dir)

    assert result.returncode != 0
    report = json.loads((output_dir / "sparse_coordinate_comparison_report.json").read_text())
    assert report["comparison_status"] == "failed"
    assert report["failure_phase"] == "load_artifacts"
    assert report["failure_message"]


def _write_sparse(path, coords=None, *, include_lr=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    coords = coords if coords is not None else np.array([[0, 0, 0, 0]], dtype=np.int32)
    payload = {"coords": coords}
    if include_lr:
        payload["lr_coords"] = coords[:, 1:4]
    np.savez(path, **payload)
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
                "effective_route": route or f"python {name}.py --seed 42 --steps 12 --stop-after-sparse",
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
            "scripts/compare_sparse_coords.py",
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
