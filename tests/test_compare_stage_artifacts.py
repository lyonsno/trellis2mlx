import json
from pathlib import Path
import subprocess

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_compare_conditioning_reports_shape_and_abs_deltas(tmp_path):
    ref = tmp_path / "ref.npz"
    mlx = tmp_path / "mlx.npz"
    out = tmp_path / "comparison.json"
    np.savez(ref, cond=np.array([[[1.0, 2.0]]], dtype=np.float32), neg_cond=np.zeros((1, 1, 2), dtype=np.float32))
    np.savez(mlx, cond=np.array([[[1.25, 1.5]]], dtype=np.float32), neg_cond=np.zeros((1, 1, 2), dtype=np.float32))

    result = subprocess.run(
        [
            "python",
            "scripts/compare_stage_artifacts.py",
            "--stage",
            "conditioning",
            "--reference",
            str(ref),
            "--candidate",
            str(mlx),
            "--output",
            str(out),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text())
    assert report["stage"] == "conditioning"
    assert report["arrays"]["cond"]["shape_match"] is True
    assert report["arrays"]["cond"]["max_abs_diff"] == 0.5
    assert report["arrays"]["cond"]["mean_abs_diff"] == 0.375
    assert report["arrays"]["neg_cond"]["max_abs_diff"] == 0.0


def test_compare_sparse_reports_coordinate_overlap(tmp_path):
    ref = tmp_path / "ref.npz"
    mlx = tmp_path / "mlx.npz"
    out = tmp_path / "comparison.json"
    np.savez(ref, coords=np.array([[0, 0, 0, 0], [0, 1, 1, 1], [0, 2, 2, 2]], dtype=np.int32))
    np.savez(mlx, coords=np.array([[0, 1, 1, 1], [0, 2, 2, 2], [0, 3, 3, 3]], dtype=np.int32))

    result = subprocess.run(
        [
            "python",
            "scripts/compare_stage_artifacts.py",
            "--stage",
            "sparse_coords",
            "--reference",
            str(ref),
            "--candidate",
            str(mlx),
            "--output",
            str(out),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text())
    assert report["coords"]["reference_count"] == 3
    assert report["coords"]["candidate_count"] == 3
    assert report["coords"]["common_count"] == 2
    assert report["coords"]["jaccard"] == 0.5
    assert report["coords"]["reference_only_count"] == 1
    assert report["coords"]["candidate_only_count"] == 1


def test_compare_shape_slat_aligns_feature_deltas_by_common_coords(tmp_path):
    ref = tmp_path / "ref.npz"
    mlx = tmp_path / "mlx.npz"
    out = tmp_path / "comparison.json"
    ref_coords = np.array([[0, 0, 0, 0], [0, 1, 1, 1]], dtype=np.int32)
    mlx_coords = np.array([[0, 1, 1, 1], [0, 2, 2, 2]], dtype=np.int32)
    np.savez(ref, coords=ref_coords, feats=np.array([[10.0, 10.0], [1.0, 3.0]], dtype=np.float32))
    np.savez(mlx, coords=mlx_coords, feats=np.array([[2.0, 1.0], [20.0, 20.0]], dtype=np.float32))

    result = subprocess.run(
        [
            "python",
            "scripts/compare_stage_artifacts.py",
            "--stage",
            "shape_slat",
            "--reference",
            str(ref),
            "--candidate",
            str(mlx),
            "--output",
            str(out),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text())
    assert report["coords"]["common_count"] == 1
    assert report["features"]["common_shape"] == [1, 2]
    assert report["features"]["max_abs_diff"] == 2.0
    assert report["features"]["mean_abs_diff"] == 1.5
