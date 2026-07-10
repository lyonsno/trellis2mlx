import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


SCRIPT = Path("scripts/source_sparse_block_replay.py")


def _write_trace(path: Path) -> None:
    n_tokens = 8
    channels = 4
    heads = 2
    head_dim = 2
    np.savez_compressed(
        path,
        pos_block3_input=np.zeros((1, n_tokens, channels), dtype=np.float32),
        pos_block3_shift_msa=np.zeros((1, channels), dtype=np.float32),
        pos_block3_scale_msa=np.zeros((1, channels), dtype=np.float32),
        pos_block3_gate_msa=np.zeros((1, channels), dtype=np.float32),
        pos_block3_shift_mlp=np.zeros((1, channels), dtype=np.float32),
        pos_block3_scale_mlp=np.zeros((1, channels), dtype=np.float32),
        pos_block3_gate_mlp=np.zeros((1, channels), dtype=np.float32),
        pos_block3_cross_k_cached_post_norm=np.zeros(
            (1, 1, 5, heads, head_dim), dtype=np.float32
        ),
        pos_block3_cross_v_cached=np.zeros((1, 1, 5, heads, head_dim), dtype=np.float32),
        pos_block3_norm1=np.zeros((1, n_tokens, channels), dtype=np.float32),
        pos_block3_mlp_gelu=np.zeros((1, n_tokens, channels), dtype=np.float32),
        trace_block_index=np.array(3, dtype=np.int32),
        sparse_flow_trace_step_index=np.array(6, dtype=np.int32),
        sparse_flow_trace_input_mode=np.array("projected_block_input"),
        t=np.array(142.85715, dtype=np.float32),
        steps=np.array(8, dtype=np.int32),
        rescale_t=np.array(5.0, dtype=np.float32),
    )


def test_route_identity_rejects_non_cpu_effective_device():
    from scripts.source_sparse_block_replay import build_route_identity

    with pytest.raises(ValueError, match="CPU"):
        build_route_identity(
            requested_route="source-sparse-block-replay",
            effective_device_type="mps",
            source_root=Path("/source"),
            checkpoint=Path("/ckpt.safetensors"),
            trace=Path("/trace.npz"),
            branch="pos",
            block_index=3,
            step_index=6,
        )


def test_trace_payload_extracts_sparse_branch_block(tmp_path):
    from scripts.source_sparse_block_replay import load_trace_payload

    trace = tmp_path / "trace.npz"
    _write_trace(trace)

    payload = load_trace_payload(trace, branch="pos", block_index=3)

    assert payload["x"].shape == (8, 4)
    assert payload["mod"].shape == (6, 4)
    assert payload["cross_k"].shape == (5, 2, 2)
    assert payload["cross_v"].shape == (5, 2, 2)
    assert payload["resolution"] == 2
    assert payload["step_index"] == 6
    assert payload["trace_input_mode"] == "projected_block_input"
    assert payload["t"] == pytest.approx(142.85715)
    assert "norm1" in payload["captured"]
    assert "mlp_gelu" in payload["captured"]


def test_source_compare_names_cover_sparse_boundary_ops():
    from scripts.source_sparse_block_replay import SOURCE_COMPARE_NAMES

    for name in (
        "norm1",
        "modulated_self_input",
        "self_attn",
        "norm2",
        "cross_q_pre_norm",
        "cross_attention_raw",
        "mlp_fc1",
        "mlp_gelu",
        "after_mlp",
    ):
        assert name in SOURCE_COMPARE_NAMES


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
            "3",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode != 0
    report = json.loads(output.read_text())
    assert report["schema"] == "trellis2mlx.source_sparse_block_replay.v1"
    assert report["status"] == "failed"
    assert report["failure_phase"] == "load_trace"
    assert report["primary_output_status"] == "not_written"
    assert report["route_identity"]["requested_route"] == "source-sparse-block-replay"


def test_cli_reports_load_trace_when_cached_cross_kv_missing(tmp_path):
    trace = tmp_path / "trace-without-cache.npz"
    np.savez_compressed(
        trace,
        pos_block0_input=np.zeros((1, 8, 4), dtype=np.float32),
        pos_block0_shift_msa=np.zeros((1, 4), dtype=np.float32),
        pos_block0_scale_msa=np.zeros((1, 4), dtype=np.float32),
        pos_block0_gate_msa=np.zeros((1, 4), dtype=np.float32),
        pos_block0_shift_mlp=np.zeros((1, 4), dtype=np.float32),
        pos_block0_scale_mlp=np.zeros((1, 4), dtype=np.float32),
        pos_block0_gate_mlp=np.zeros((1, 4), dtype=np.float32),
        trace_block_index=np.array(0, dtype=np.int32),
    )
    output = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--trace",
            str(trace),
            "--checkpoint",
            str(tmp_path / "missing.safetensors"),
            "--source-root",
            str(tmp_path / "source"),
            "--output",
            str(output),
            "--branch",
            "pos",
            "--block-index",
            "0",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode != 0
    report = json.loads(output.read_text())
    assert report["schema"] == "trellis2mlx.source_sparse_block_replay.v1"
    assert report["status"] == "failed"
    assert report["failure_phase"] == "load_trace"
    assert "cross_k_cached_post_norm" in report["error"]
