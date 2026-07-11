import json
import subprocess
import sys

import numpy as np


def _write_coords(path, coords_3d):
    coords_3d = np.asarray(coords_3d, dtype=np.int32)
    coords = np.column_stack([np.zeros(len(coords_3d), dtype=np.int32), coords_3d])
    np.savez(path, coords=coords, coords_3d=coords_3d)


def test_sparse_flow_boundary_direction_score_separates_clean_lost_and_gained_cells(tmp_path):
    source_steps = tmp_path / "source_steps.npz"
    candidate_step = tmp_path / "candidate_step.npz"

    source = np.zeros((1, 1, 2, 2, 2, 2), dtype=np.float32)
    candidate = np.zeros((1, 2, 2, 2, 2), dtype=np.float32)
    candidate[0, 0, 0, 0, 0] = -2.0
    candidate[0, 1, 0, 0, 0] = 0.5
    candidate[0, 0, 1, 1, 1] = 2.0
    candidate[0, 1, 1, 1, 1] = -0.5
    candidate[0, 0, 0, 1, 1] = 0.1
    np.savez(source_steps, pred_final=source)
    np.savez(candidate_step, pred_final=candidate)

    lost_coords = tmp_path / "alpha0_coords.npz"
    gained_coords = tmp_path / "alpha001_coords.npz"
    _write_coords(lost_coords, [[0, 0, 0], [0, 0, 1]])
    _write_coords(gained_coords, [[3, 3, 3], [3, 3, 2]])

    output = tmp_path / "score.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/sparse_flow_boundary_direction_score.py",
            "--source-steps",
            str(source_steps),
            "--candidate-step",
            str(candidate_step),
            "--negative-coords",
            str(lost_coords),
            "--positive-coords",
            str(gained_coords),
            "--output",
            str(output),
            "--step-index",
            "0",
            "--state-resolution",
            "2",
            "--support-resolution",
            "4",
            "--fields",
            "pred_final",
        ],
        cwd=".",
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["schema"] == "trellis2mlx.sparse_flow_boundary_direction_score.v1"
    assert report["labels"]["clean_negative_count"] == 1
    assert report["labels"]["clean_positive_count"] == 1
    assert report["fields"]["pred_final"]["centroid_delta"][0] > 3.0
    assert report["fields"]["pred_final"]["auc_positive_gt_negative"] == 1.0
    assert report["fields"]["pred_final"]["top_channels"][0]["channel"] == 0
