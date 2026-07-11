import json
import subprocess
import sys

import numpy as np


def _write_coords(path, coords_3d):
    coords_3d = np.asarray(coords_3d, dtype=np.int32)
    coords = np.column_stack([np.zeros(len(coords_3d), dtype=np.int32), coords_3d])
    np.savez(path, coords=coords, coords_3d=coords_3d)


def test_sparse_flow_step_residual_probe_projects_delta_onto_support_categories(tmp_path):
    source_steps = tmp_path / "source_steps.npz"
    candidate_step = tmp_path / "candidate_step.npz"

    source_pred = np.zeros((3, 1, 4, 2, 2, 2), dtype=np.float32)
    source_sample = np.zeros_like(source_pred)
    candidate_pred = np.zeros((1, 4, 2, 2, 2), dtype=np.float32)
    candidate_sample = np.zeros_like(candidate_pred)

    candidate_pred[0, 0, 0, 0, 0] = 2.0
    candidate_pred[0, 2, 0, 0, 0] = -1.0
    candidate_pred[0, 1, 1, 1, 1] = 3.0
    candidate_sample[0, 3, 1, 1, 1] = 4.0

    np.savez(source_steps, pred_final=source_pred, sample_next=source_sample)
    np.savez(candidate_step, pred_final=candidate_pred, sample_next=candidate_sample)

    source_coords = tmp_path / "source_coords.npz"
    old_coords = tmp_path / "old_coords.npz"
    current_coords = tmp_path / "current_coords.npz"
    _write_coords(source_coords, [[0, 0, 0], [1, 1, 1], [3, 3, 3]])
    _write_coords(old_coords, [[0, 0, 0], [1, 1, 1]])
    _write_coords(current_coords, [[1, 1, 1], [3, 3, 3]])

    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/sparse_flow_step_residual_probe.py",
            "--source-steps",
            str(source_steps),
            "--candidate-step",
            str(candidate_step),
            "--source-coords",
            str(source_coords),
            "--old-coords",
            str(old_coords),
            "--current-coords",
            str(current_coords),
            "--output",
            str(output),
            "--step-index",
            "2",
            "--state-resolution",
            "2",
            "--support-resolution",
            "4",
            "--top-k",
            "3",
        ],
        cwd=".",
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["schema"] == "trellis2mlx.sparse_flow_step_residual_probe.v1"
    assert report["comparison"]["step_index"] == 2
    assert report["support"]["source_count"] == 3
    assert report["support_projection"]["old_only"]["inside_count"] == 1
    assert report["support_projection"]["current_only"]["inside_count"] == 1
    pred = report["fields"]["pred_final"]
    assert pred["all"]["max_abs"] == 3.0
    assert pred["support_categories"]["old_only"]["mean_delta_l2"] > 2.0
    assert pred["per_channel"]["support_categories"]["old_only"]["signed_rank"][0]["channel"] == 0
    assert pred["top_cells"][0]["state_zyx"] == [1, 1, 1]
    assert "current_only" in pred["top_cells"][0]["support_categories"]
    assert report["fields"]["sample_next"]["support_categories"]["current_only"]["max_abs"] == 4.0
