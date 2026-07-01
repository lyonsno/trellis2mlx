import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_compare_decoder_outputs_writes_route_aware_full_common_report(tmp_path):
    """The decoder comparator is an evidence harness, so it must not sample silently."""
    n_common = 100_001
    common_coords = np.column_stack(
        [
            np.zeros(n_common, dtype=np.int32),
            np.zeros(n_common, dtype=np.int32),
            np.zeros(n_common, dtype=np.int32),
            np.arange(n_common, dtype=np.int32),
        ]
    )
    mac_feats = np.tile(
        np.array([[0.1, 0.2, 0.3, -1.0, 1.0, -1.0, 0.5]], dtype=np.float32),
        (n_common, 1),
    )
    mlx_feats = mac_feats.copy()
    mlx_feats[:, 3] *= -1

    mac_path = tmp_path / "mac_decoder_output.npz"
    mlx_path = tmp_path / "mlx_decoder_output.npz"
    np.savez(mac_path, feats=mac_feats, coords=common_coords)
    np.savez(
        mlx_path,
        feats=np.vstack([mlx_feats, np.zeros((1, 7), dtype=np.float32)]),
        coords=np.vstack(
            [
                common_coords,
                np.array([[0, 9, 9, 9]], dtype=np.int32),
            ]
        ),
    )

    mac_receipt = tmp_path / "mac_receipt.json"
    mlx_receipt = tmp_path / "mlx_receipt.json"
    mac_receipt.write_text(
        json.dumps(
            {
                "job_id": "mac-job",
                "job_type": "trellis2_official_512",
                "status": "done",
                "effective_route": "python run_official_trellis2.py --pipeline-type 512",
                "effective_cwd": "/trellis-mac/TRELLIS.2",
            }
        )
    )
    mlx_receipt.write_text(
        json.dumps(
            {
                "job_id": "mlx-job",
                "job_type": "trellis2mlx_decoder_capture",
                "status": "done",
                "effective_route": "python generate.py --resolution 512 --steps 8 --no-cascade",
                "effective_cwd": "/trellis2mlx",
            }
        )
    )

    output_dir = tmp_path / "comparison"
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "scripts/compare_decoder_outputs.py",
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
        check=True,
    )

    report = json.loads((output_dir / "comparison_report.json").read_text())

    assert report["schema"] == "trellis2mlx.decoder_comparison.v1"
    assert report["artifacts"]["trellis_mac"]["sha256"]
    assert report["artifacts"]["trellis_mlx"]["sha256"]
    assert report["routes"]["trellis_mac"]["job_id"] == "mac-job"
    assert report["routes"]["trellis_mlx"]["job_id"] == "mlx-job"
    assert report["coordinate_comparison"]["common_voxels"] == n_common
    assert report["feature_comparison"]["sample_policy"] == "full_common_set"
    assert report["feature_comparison"]["n_common_compared"] == n_common
    assert report["evidence_use_class"] == "bounded_candidate"
    assert "sampler_seed" in report["unmatched_variables"]
