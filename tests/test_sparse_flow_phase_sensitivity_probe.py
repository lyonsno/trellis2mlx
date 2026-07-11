import json
import subprocess
import sys

import numpy as np


def _write_steps(path, *, old_like=False):
    sample_next = np.zeros((1, 1, 3, 4, 4, 4), dtype=np.float32)
    pred_final = np.zeros_like(sample_next)
    if old_like:
        sample_next[0, 0, 0, 1, 2, 0] = 1.0
        sample_next[0, 0, 2, 1, 2, 0] = 0.0
        pred_final[0, 0, 0, 1, 2, 0] = -1.0
        pred_final[0, 0, 2, 1, 2, 0] = 0.0
    else:
        sample_next[0, 0, 0, 1, 2, 0] = 0.0
        sample_next[0, 0, 2, 1, 2, 0] = 1.0
        pred_final[0, 0, 0, 1, 2, 0] = 0.0
        pred_final[0, 0, 2, 1, 2, 0] = -1.0
        sample_next[0, 0, 0, 3, 0, 3] = 0.5
        sample_next[0, 0, 2, 3, 0, 3] = 0.5
        pred_final[0, 0, 0, 3, 0, 3] = -0.5
        pred_final[0, 0, 2, 3, 0, 3] = -0.5
    np.savez(path, sample_next=sample_next, pred_final=pred_final)


def _write_coords(path, coords_3d):
    coords_3d = np.asarray(coords_3d, dtype=np.int32)
    coords = np.column_stack([np.zeros(len(coords_3d), dtype=np.int32), coords_3d])
    np.savez(path, coords=coords, coords_3d=coords_3d)


def test_sparse_flow_phase_sensitivity_probe_reports_gate_phase_and_support_projection(tmp_path):
    old_steps = tmp_path / "old_steps.npz"
    current_steps = tmp_path / "current_steps.npz"
    _write_steps(old_steps, old_like=True)
    _write_steps(current_steps, old_like=False)

    old_coords = tmp_path / "old_coords.npz"
    current_coords = tmp_path / "current_coords.npz"
    _write_coords(old_coords, [[2, 4, 0], [2, 4, 1]])
    _write_coords(current_coords, [[2, 4, 1], [6, 0, 7]])

    output = tmp_path / "phase_report.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/sparse_flow_phase_sensitivity_probe.py",
            "--old-steps",
            str(old_steps),
            "--current-steps",
            str(current_steps),
            "--old-coords",
            str(old_coords),
            "--current-coords",
            str(current_coords),
            "--output",
            str(output),
            "--step-index",
            "0",
            "--state-resolution",
            "4",
            "--support-resolution",
            "8",
            "--x-gate",
            "0:2",
            "--y-gate",
            "2:4",
        ],
        cwd=".",
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["schema"] == "trellis2mlx.sparse_flow_phase_sensitivity_probe.v1"
    assert report["transition"]["step_index"] == 0
    assert report["support_projection"]["old_only"]["count"] == 1
    assert report["support_projection"]["old_only"]["inside_gate_count"] == 1
    assert report["support_projection"]["current_only"]["inside_gate_count"] == 0
    assert report["fields"]["sample_next"]["xy_gate"]["mean_abs_phase_delta_rad"] > 1.5
    assert (
        report["fields"]["sample_next"]["xy_gate"]["mean_active_delta_l2"]
        > report["fields"]["sample_next"]["outside_gate"]["mean_active_delta_l2"]
    )
    assert report["fields"]["pred_final"]["xy_gate"]["mean_abs_phase_delta_rad"] > 1.5
    assert report["fields"]["sample_next"]["per_channel"]["support_categories"]["old_only"]["signed_rank"][0][
        "channel"
    ] in {0, 2}
