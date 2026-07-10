import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


SCRIPT = Path("scripts/source_shape_block_replay.py")


def _write_trace(path: Path) -> None:
    n_tokens = 3
    channels = 4
    heads = 2
    head_dim = 2
    np.savez_compressed(
        path,
        pos_block1_input=np.zeros((1, n_tokens, channels), dtype=np.float32),
        pos_block1_shift_msa=np.zeros((1, channels), dtype=np.float32),
        pos_block1_scale_msa=np.zeros((1, channels), dtype=np.float32),
        pos_block1_gate_msa=np.zeros((1, channels), dtype=np.float32),
        pos_block1_shift_mlp=np.zeros((1, channels), dtype=np.float32),
        pos_block1_scale_mlp=np.zeros((1, channels), dtype=np.float32),
        pos_block1_gate_mlp=np.zeros((1, channels), dtype=np.float32),
        pos_block1_cross_k_cached_post_norm=np.zeros(
            (1, 1, 5, heads, head_dim), dtype=np.float32
        ),
        pos_block1_cross_v_cached=np.zeros((1, 1, 5, heads, head_dim), dtype=np.float32),
        coords=np.array(
            [
                [0, 1, 2, 3],
                [0, 2, 3, 4],
                [0, 3, 4, 5],
            ],
            dtype=np.int32,
        ),
        trace_block_index=np.array(1, dtype=np.int32),
        shape_flow_trace_step_index=np.array(0, dtype=np.int32),
        t=np.array(1000.0, dtype=np.float32),
    )


def test_route_identity_rejects_non_cpu_effective_device():
    from scripts.source_shape_block_replay import build_route_identity

    with pytest.raises(ValueError, match="CPU"):
        build_route_identity(
            requested_route="source-shape-block-replay",
            effective_device_type="mps",
            source_root=Path("/source"),
            checkpoint=Path("/ckpt.safetensors"),
            trace=Path("/trace.npz"),
            branch="pos",
            block_index=1,
            step_index=0,
        )


def test_trace_payload_extracts_selected_branch_block(tmp_path):
    from scripts.source_shape_block_replay import load_trace_payload

    trace = tmp_path / "trace.npz"
    _write_trace(trace)

    payload = load_trace_payload(trace, branch="pos", block_index=1)

    assert payload["x"].shape == (3, 4)
    assert payload["coords"].shape == (3, 4)
    assert payload["mod"].shape == (6, 4)
    assert payload["cross_k"].shape == (5, 2, 2)
    assert payload["cross_v"].shape == (5, 2, 2)
    assert payload["step_index"] == 0
    assert payload["t"] == 1000.0


def test_cli_writes_failure_report_when_trace_missing(tmp_path):
    output = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--trace",
            str(tmp_path / "missing.npz"),
            "--checkpoint",
            str(tmp_path / "missing.safetensors"),
            "--source-root",
            str(tmp_path / "source"),
            "--output",
            str(output),
            "--branch",
            "pos",
            "--block-index",
            "1",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode != 0
    report = json.loads(output.read_text())
    assert report["schema"] == "trellis2mlx.source_shape_block_replay.v1"
    assert report["status"] == "failed"
    assert report["failure_phase"] == "load_trace"
    assert report["primary_output_status"] == "not_written"
    assert report["route_identity"]["requested_route"] == "source-shape-block-replay"
