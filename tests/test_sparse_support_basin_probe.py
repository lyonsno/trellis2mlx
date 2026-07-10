import json
import subprocess
import sys

import numpy as np


def _write_coords(path, coords_3d):
    coords_3d = np.asarray(coords_3d, dtype=np.int32)
    coords = np.column_stack([np.zeros(len(coords_3d), dtype=np.int32), coords_3d])
    np.savez(path, coords=coords, coords_3d=coords_3d)


def test_sparse_support_basin_probe_reports_margin_and_surface_bands(tmp_path):
    old_coords = tmp_path / "old_coords.npz"
    current_coords = tmp_path / "current_coords.npz"
    _write_coords(old_coords, [[0, 0, 0], [1, 1, 1], [3, 3, 3]])
    _write_coords(current_coords, [[0, 0, 0], [1, 1, 1], [0, 3, 3]])

    old_logits = np.full((1, 1, 4, 4, 4), -2.0, dtype=np.float32)
    current_logits = np.full((1, 1, 4, 4, 4), -2.0, dtype=np.float32)
    old_logits[0, 0, 3, 3, 3] = 4.0
    current_logits[0, 0, 3, 3, 3] = -6.0
    old_logits[0, 0, 0, 3, 3] = -5.0
    current_logits[0, 0, 0, 3, 3] = 7.0
    old_logit_path = tmp_path / "old_logits.npz"
    current_logit_path = tmp_path / "current_logits.npz"
    np.savez(old_logit_path, logits=old_logits)
    np.savez(current_logit_path, logits=current_logits)

    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/sparse_support_basin_probe.py",
            "--old-coords",
            str(old_coords),
            "--current-coords",
            str(current_coords),
            "--old-logits",
            str(old_logit_path),
            "--current-logits",
            str(current_logit_path),
            "--output",
            str(output),
            "--surface-band",
            "1",
        ],
        cwd=".",
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["schema"] == "trellis2mlx.sparse_support_basin_probe.v1"
    assert report["support"]["old_only_count"] == 1
    assert report["support"]["current_only_count"] == 1
    assert report["support"]["jaccard"] == 0.5
    assert report["logit_margins"]["old_only"]["old_logits"]["mean"] == 4.0
    assert report["logit_margins"]["old_only"]["current_logits"]["mean"] == -6.0
    assert report["logit_margins"]["current_only"]["old_logits"]["mean"] == -5.0
    assert report["logit_margins"]["current_only"]["current_logits"]["mean"] == 7.0
    assert report["surface_bands"]["old_only"]["z_high"] == 1
    assert report["surface_bands"]["current_only"]["z_low"] == 1
    assert report["top_cells"]["old_only_by_logit_drop"][0]["coord_zyx"] == [3, 3, 3]
    assert report["top_cells"]["current_only_by_logit_gain"][0]["coord_zyx"] == [0, 3, 3]


def test_sparse_support_basin_probe_reduces_decoder_logits_to_lr_block_max(tmp_path):
    old_coords = tmp_path / "old_coords.npz"
    current_coords = tmp_path / "current_coords.npz"
    _write_coords(old_coords, [[1, 1, 1]])
    _write_coords(current_coords, [[0, 0, 0]])

    old_logits = np.full((1, 1, 4, 4, 4), -8.0, dtype=np.float32)
    current_logits = np.full((1, 1, 4, 4, 4), -8.0, dtype=np.float32)
    old_logits[0, 0, 2, 2, 2] = 3.0
    current_logits[0, 0, 2, 2, 2] = -4.0
    old_logits[0, 0, 1, 1, 1] = -5.0
    current_logits[0, 0, 1, 1, 1] = 6.0
    old_logit_path = tmp_path / "old_logits.npz"
    current_logit_path = tmp_path / "current_logits.npz"
    np.savez(old_logit_path, logits=old_logits)
    np.savez(current_logit_path, logits=current_logits)

    output = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/sparse_support_basin_probe.py",
            "--old-coords",
            str(old_coords),
            "--current-coords",
            str(current_coords),
            "--old-logits",
            str(old_logit_path),
            "--current-logits",
            str(current_logit_path),
            "--output",
            str(output),
            "--logit-grid",
            "block-max",
            "--lr-resolution",
            "2",
        ],
        cwd=".",
        check=True,
    )

    report = json.loads(output.read_text())
    assert report["logit_grid"]["mode"] == "block-max"
    assert report["logit_grid"]["old_logits_shape_zyx"] == [4, 4, 4]
    assert report["logit_grid"]["effective_shape_zyx"] == [2, 2, 2]
    assert report["logit_margins"]["old_only"]["old_logits"]["mean"] == 3.0
    assert report["logit_margins"]["old_only"]["current_logits"]["mean"] == -4.0
    assert report["logit_margins"]["current_only"]["old_logits"]["mean"] == -5.0
    assert report["logit_margins"]["current_only"]["current_logits"]["mean"] == 6.0
