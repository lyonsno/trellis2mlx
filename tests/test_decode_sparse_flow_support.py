import json
import subprocess
import sys

import numpy as np

from scripts.decode_sparse_flow_support import coords_from_logits, load_step_sample


def test_coords_from_logits_uses_block_max_and_writes_batch_column():
    logits = np.full((1, 1, 4, 4, 4), -3.0, dtype=np.float32)
    logits[0, 0, 0, 1, 1] = 2.0
    logits[0, 0, 3, 2, 2] = 5.0
    logits[0, 0, 3, 3, 3] = -0.25

    support = coords_from_logits(logits, lr_resolution=2)

    assert support.coords_3d.tolist() == [[0, 0, 0], [1, 1, 1]]
    assert support.coords.tolist() == [[0, 0, 0, 0], [0, 1, 1, 1]]
    assert support.logits_shape_zyx == (4, 4, 4)
    assert support.effective_logits_shape_zyx == (2, 2, 2)
    assert support.positive_count == 2


def test_load_step_sample_selects_negative_index(tmp_path):
    steps = np.arange(3 * 1 * 8 * 2 * 2 * 2, dtype=np.float32).reshape(3, 1, 8, 2, 2, 2)
    steps_path = tmp_path / "steps.npz"
    np.savez(steps_path, sample_next=steps)

    sample = load_step_sample(steps_path, "sample_next", -1)

    np.testing.assert_array_equal(sample, steps[2])


def test_decode_sparse_flow_support_cli_reduces_provided_logits(tmp_path):
    logits = np.full((1, 1, 4, 4, 4), -2.0, dtype=np.float32)
    logits[0, 0, 2, 2, 2] = 7.0
    logits_path = tmp_path / "decoder_logits.npz"
    np.savez(logits_path, logits=logits)

    output_dir = tmp_path / "decoded"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/decode_sparse_flow_support.py",
            "--logits",
            str(logits_path),
            "--output-dir",
            str(output_dir),
            "--lr-resolution",
            "2",
        ],
        cwd=".",
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    with np.load(output_dir / "sparse_coords.npz") as data:
        assert data["coords_3d"].tolist() == [[1, 1, 1]]
        assert data["coords"].tolist() == [[0, 1, 1, 1]]
    report = json.loads((output_dir / "decode_sparse_flow_support_report.json").read_text())
    assert report["schema"] == "trellis2mlx.decode_sparse_flow_support.v1"
    assert report["support"]["coord_count"] == 1
    assert report["logit_source"]["kind"] == "provided_logits"
    assert report["logit_grid"]["mode"] == "block-max"
