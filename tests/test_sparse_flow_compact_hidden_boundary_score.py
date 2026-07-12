import json
import subprocess
import sys

import numpy as np


def _write_coords(path, coords_3d):
    coords_3d = np.asarray(coords_3d, dtype=np.int32)
    coords = np.column_stack([np.zeros(len(coords_3d), dtype=np.int32), coords_3d])
    np.savez(path, coords=coords, coords_3d=coords_3d)


def test_compact_hidden_boundary_score_uses_clean_projected_cells_and_delta(tmp_path):
    trace = tmp_path / "trace.npz"
    input_hidden = np.zeros((1, 8, 3), dtype=np.float32)
    after_self = input_hidden.copy()
    after_mlp = input_hidden.copy()

    # Projected clean negative cell: support coords [0,0,0] and [0,0,1] -> state token 0.
    input_hidden[0, 0, 0] = -10.0
    after_self[0, 0, 0] = -9.0
    after_mlp[0, 0, 0] = -8.0
    # Projected clean positive cell: support coords [3,3,3] and [3,3,2] -> state token 7.
    input_hidden[0, 7, 0] = 10.0
    after_self[0, 7, 0] = 12.0
    after_mlp[0, 7, 0] = 16.0
    # Overlapped support cell should be discarded from both classes.
    after_mlp[0, 3, 0] = 1000.0

    np.savez(
        trace,
        pos_block2_input=input_hidden,
        pos_block2_after_self=after_self,
        pos_block2_after_mlp=after_mlp,
    )
    negative_coords = tmp_path / "negative_coords.npz"
    positive_coords = tmp_path / "positive_coords.npz"
    _write_coords(negative_coords, [[0, 0, 0], [0, 0, 1], [0, 2, 2]])
    _write_coords(positive_coords, [[3, 3, 3], [3, 3, 2], [0, 2, 2]])

    output = tmp_path / "score.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/sparse_flow_compact_hidden_boundary_score.py",
            "--trace",
            str(trace),
            "--negative-coords",
            str(negative_coords),
            "--positive-coords",
            str(positive_coords),
            "--output",
            str(output),
            "--block-index",
            "2",
            "--state-resolution",
            "2",
            "--support-resolution",
            "4",
        ],
        cwd=".",
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["schema"] == "trellis2mlx.sparse_flow_compact_hidden_boundary_score.v1"
    assert report["labels"]["clean_negative_count"] == 1
    assert report["labels"]["clean_positive_count"] == 1
    assert report["labels"]["overlap_count"] == 1
    assert report["blocks"]["2"]["raw_stage_scores"]["after_mlp"]["centroid_delta_l2"] == 24.0
    assert report["blocks"]["2"]["raw_stage_scores"]["after_mlp"]["auc_positive_gt_negative"] == 1.0
    assert report["blocks"]["2"]["delta_from_input_scores"]["after_mlp"]["centroid_delta_l2"] == 4.0
    assert report["blocks"]["2"]["delta_from_input_scores"]["after_mlp"]["auc_positive_gt_negative"] == 1.0
