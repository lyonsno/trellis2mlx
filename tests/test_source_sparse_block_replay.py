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


def test_attention_witness_arrays_include_source_captured_and_projection_weights():
    from scripts.source_sparse_block_replay import build_attention_witness_arrays

    source = {
        "q_post_rope": np.ones((2, 3, 4), dtype=np.float32),
        "k_post_rope": np.ones((2, 3, 4), dtype=np.float32) * 2,
        "v": np.ones((2, 3, 4), dtype=np.float32) * 3,
        "attention_raw": np.ones((2, 12), dtype=np.float32) * 4,
        "self_attn": np.ones((2, 12), dtype=np.float32) * 5,
    }
    captured = {
        "q_post_rope": np.ones((2, 3, 4), dtype=np.float32) * 6,
        "k_post_rope": np.ones((2, 3, 4), dtype=np.float32) * 7,
        "v": np.ones((2, 3, 4), dtype=np.float32) * 8,
        "attention_raw": np.ones((2, 12), dtype=np.float32) * 9,
        "self_attn": np.ones((2, 12), dtype=np.float32) * 10,
    }
    arrays = build_attention_witness_arrays(
        source=source,
        captured=captured,
        to_out_weight=np.eye(12, dtype=np.float32),
        to_out_bias=np.arange(12, dtype=np.float32),
        route_identity={"effective_route": "official-trellis2-source-cpu-selected-sparse-block"},
    )

    assert arrays["source_q_post_rope"].shape == (2, 3, 4)
    assert arrays["captured_q_post_rope"][0, 0, 0] == 6
    assert arrays["source_to_out_weight"].shape == (12, 12)
    assert arrays["source_to_out_bias"].shape == (12,)
    assert arrays["route_identity_json"].shape == ()
    assert "official-trellis2-source-cpu" in str(arrays["route_identity_json"].item())


def test_cross_attention_witness_arrays_include_cached_kv_and_projection_weights():
    from scripts.source_sparse_block_replay import build_cross_attention_witness_arrays

    source = {
        "cross_q_post_norm": np.ones((2, 3, 4), dtype=np.float32),
        "cross_attention_raw": np.ones((2, 12), dtype=np.float32) * 2,
        "cross_attn": np.ones((2, 12), dtype=np.float32) * 3,
    }
    captured = {
        "cross_q_post_norm": np.ones((2, 3, 4), dtype=np.float32) * 4,
        "cross_attention_raw": np.ones((2, 12), dtype=np.float32) * 5,
        "cross_attn": np.ones((2, 12), dtype=np.float32) * 6,
    }
    arrays = build_cross_attention_witness_arrays(
        source=source,
        captured=captured,
        cross_k=np.ones((5, 3, 4), dtype=np.float32) * 7,
        cross_v=np.ones((5, 3, 4), dtype=np.float32) * 8,
        to_out_weight=np.eye(12, dtype=np.float32),
        to_out_bias=np.arange(12, dtype=np.float32),
        route_identity={"effective_route": "official-trellis2-source-cpu-selected-sparse-block"},
    )

    assert arrays["source_cross_q_post_norm"].shape == (2, 3, 4)
    assert arrays["captured_cross_q_post_norm"][0, 0, 0] == 4
    assert arrays["cross_k"].shape == (5, 3, 4)
    assert arrays["cross_v"][0, 0, 0] == 8
    assert arrays["source_to_out_weight"].shape == (12, 12)
    assert arrays["source_to_out_bias"].shape == (12,)
    assert arrays["route_identity_json"].shape == ()


def test_cross_q_witness_arrays_include_norm_q_and_projection_weights():
    from scripts.source_sparse_block_replay import build_cross_q_witness_arrays

    source = {
        "after_self": np.ones((2, 12), dtype=np.float32),
        "norm2": np.ones((2, 12), dtype=np.float32) * 2,
        "cross_q_pre_norm": np.ones((2, 3, 4), dtype=np.float32) * 3,
        "cross_q_post_norm": np.ones((2, 3, 4), dtype=np.float32) * 4,
    }
    captured = {
        "after_self": np.ones((2, 12), dtype=np.float32) * 5,
        "norm2": np.ones((2, 12), dtype=np.float32) * 6,
        "cross_q_pre_norm": np.ones((2, 3, 4), dtype=np.float32) * 7,
        "cross_q_post_norm": np.ones((2, 3, 4), dtype=np.float32) * 8,
    }
    arrays = build_cross_q_witness_arrays(
        source=source,
        captured=captured,
        norm2_weight=np.arange(12, dtype=np.float32),
        norm2_bias=np.arange(12, dtype=np.float32) + 1,
        to_q_weight=np.eye(12, dtype=np.float32),
        to_q_bias=np.arange(12, dtype=np.float32) + 2,
        q_rms_norm_gamma=np.ones((3, 4), dtype=np.float32) * 9,
        route_identity={"effective_route": "official-trellis2-source-cpu-selected-sparse-block"},
    )

    assert arrays["source_after_self"].shape == (2, 12)
    assert arrays["captured_norm2"][0, 0] == 6
    assert arrays["source_cross_q_pre_norm"].shape == (2, 3, 4)
    assert arrays["captured_cross_q_post_norm"][0, 0, 0] == 8
    assert arrays["source_norm2_weight"].shape == (12,)
    assert arrays["source_to_q_weight"].shape == (12, 12)
    assert arrays["source_q_rms_norm_gamma"].shape == (3, 4)
    assert arrays["route_identity_json"].shape == ()


def test_mlp_witness_arrays_include_inputs_outputs_and_weights():
    from scripts.source_sparse_block_replay import build_mlp_witness_arrays

    source = {
        "mlp_input": np.ones((2, 12), dtype=np.float32),
        "mlp_fc1": np.ones((2, 20), dtype=np.float32) * 2,
        "mlp_gelu": np.ones((2, 20), dtype=np.float32) * 3,
        "mlp": np.ones((2, 12), dtype=np.float32) * 4,
        "mlp_gated": np.ones((2, 12), dtype=np.float32) * 5,
        "after_mlp": np.ones((2, 12), dtype=np.float32) * 6,
    }
    captured = {
        "mlp_input": np.ones((2, 12), dtype=np.float32) * 7,
        "mlp_fc1": np.ones((2, 20), dtype=np.float32) * 8,
        "mlp_gelu": np.ones((2, 20), dtype=np.float32) * 9,
        "mlp": np.ones((2, 12), dtype=np.float32) * 10,
        "mlp_gated": np.ones((2, 12), dtype=np.float32) * 11,
        "after_mlp": np.ones((2, 12), dtype=np.float32) * 12,
        "after_cross": np.ones((2, 12), dtype=np.float32) * 13,
        "gate_mlp": np.ones((12,), dtype=np.float32) * 14,
    }

    arrays = build_mlp_witness_arrays(
        source=source,
        captured=captured,
        fc1_weight=np.ones((20, 12), dtype=np.float32),
        fc1_bias=np.ones((20,), dtype=np.float32) * 2,
        fc2_weight=np.ones((12, 20), dtype=np.float32) * 3,
        fc2_bias=np.ones((12,), dtype=np.float32) * 4,
        route_identity={"effective_route": "official-trellis2-source-cpu-selected-sparse-block"},
    )

    assert arrays["captured_mlp_input"][0, 0] == 7
    assert arrays["captured_after_cross"][0, 0] == 13
    assert arrays["captured_gate_mlp"][0] == 14
    assert arrays["source_mlp"][0, 0] == 4
    assert arrays["captured_mlp"][0, 0] == 10
    assert arrays["captured_mlp_fc1"].shape == (2, 20)
    assert arrays["captured_mlp_gelu"][0, 0] == 9
    assert "source_mlp_fc1" not in arrays
    assert "source_mlp_gelu" not in arrays
    assert arrays["source_mlp_fc1_weight"].shape == (20, 12)
    assert arrays["source_mlp_fc1_bias"].shape == (20,)
    assert arrays["source_mlp_fc2_weight"].shape == (12, 20)
    assert arrays["source_mlp_fc2_bias"].shape == (12,)
    assert arrays["route_identity_json"].shape == ()


def test_source_sparse_block_replay_declares_mlp_witness_output_arg():
    source = (Path(__file__).resolve().parents[1] / "scripts" / "source_sparse_block_replay.py").read_text()

    assert "--mlp-witness-output" in source
    assert "write_mlp_witness" in source


def test_cuda_sparse_cross_q_metric_reports_exact_and_delta():
    from scripts.cuda_sparse_cross_q_witness import metric_np

    report = metric_np(
        np.array([[1.0, 2.0]], dtype=np.float32),
        np.array([[1.25, 2.0]], dtype=np.float32),
    )

    assert report["shape"] == [1, 2]
    assert report["exact"] is False
    assert report["nonzero"] == 1
    assert report["mean_abs"] == 0.125
    assert report["max_abs"] == 0.25
    assert report["normalized_singleton_batch"] is False


def test_cuda_sparse_cross_q_metric_aligns_only_extra_singleton_batch():
    from scripts.cuda_sparse_cross_q_witness import metric_np

    report = metric_np(
        np.array([[[1.0, 2.0]]], dtype=np.float32),
        np.array([[1.0, 2.25]], dtype=np.float32),
    )

    assert report["shape"] == [1, 2]
    assert report["normalized_singleton_batch"] is True
    assert report["nonzero"] == 1
    assert report["max_abs"] == 0.25


def test_cuda_sparse_cross_q_rms_norm_matches_source_formula():
    from scripts.cuda_sparse_cross_q_witness import _multihead_rms_norm_np

    x = np.array([[[3.0, 4.0], [5.0, 12.0]]], dtype=np.float32)
    gamma = np.array([[2.0, 0.5], [1.0, 3.0]], dtype=np.float32)

    out = _multihead_rms_norm_np(x, gamma)

    expected = x / np.linalg.norm(x, axis=-1, keepdims=True) * gamma * np.sqrt(2.0)
    np.testing.assert_allclose(out, expected.astype(np.float32), rtol=1e-6, atol=1e-6)


def test_cuda_sparse_cross_q_witness_requires_named_arrays():
    from scripts.cuda_sparse_cross_q_witness import _require

    with pytest.raises(KeyError, match="captured_after_self"):
        _require({}, "captured_after_self")


def test_cuda_sparse_cross_q_witness_schema_and_failure_phase_are_stable():
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "cuda_sparse_cross_q_witness.py"
    ).read_text()

    assert 'SCHEMA = "trellis2mlx.cuda_sparse_cross_q.v1"' in source
    assert '"failure_phase": "cuda_sparse_cross_q"' in source
    assert "cuda_captured_after_self_vs_captured_cross_q_post_norm" in source
    assert "--save-full-output-npz" in source
    assert "full_outputs_omitted" in source


def test_cuda_sparse_attention_metric_reports_exact_and_delta():
    from scripts.cuda_sparse_attention_witness import metric_np

    report = metric_np(
        np.array([[1.0, 2.0]], dtype=np.float32),
        np.array([[1.0, 2.5]], dtype=np.float32),
    )

    assert report["shape"] == [1, 2]
    assert report["exact"] is False
    assert report["nonzero"] == 1
    assert report["mean_abs"] == 0.25
    assert report["max_abs"] == 0.5


def test_cuda_sparse_cross_attention_metric_reports_exact_and_delta():
    from scripts.cuda_sparse_cross_attention_witness import metric_np

    report = metric_np(
        np.array([[1.0, 2.0]], dtype=np.float32),
        np.array([[1.5, 2.0]], dtype=np.float32),
    )

    assert report["shape"] == [1, 2]
    assert report["exact"] is False
    assert report["nonzero"] == 1
    assert report["mean_abs"] == 0.25
    assert report["max_abs"] == 0.5


def test_cuda_sparse_mlp_metric_reports_exact_and_delta():
    from scripts.cuda_sparse_mlp_witness import metric_np

    report = metric_np(
        np.array([[1.0, 2.0]], dtype=np.float32),
        np.array([[1.0, 1.5]], dtype=np.float32),
    )

    assert report["shape"] == [1, 2]
    assert report["exact"] is False
    assert report["nonzero"] == 1
    assert report["mean_abs"] == 0.25
    assert report["max_abs"] == 0.5
    assert report["normalized_singleton_batch"] is False


def test_cuda_sparse_mlp_metric_aligns_only_extra_singleton_batch():
    from scripts.cuda_sparse_mlp_witness import metric_np

    report = metric_np(
        np.array([[[1.0, 2.0]]], dtype=np.float32),
        np.array([[1.0, 1.5]], dtype=np.float32),
    )

    assert report["shape"] == [1, 2]
    assert report["normalized_singleton_batch"] is True
    assert report["nonzero"] == 1
    assert report["max_abs"] == 0.5


def test_cuda_sparse_mlp_witness_requires_named_arrays():
    from scripts.cuda_sparse_mlp_witness import _require

    with pytest.raises(KeyError, match="captured_mlp_input"):
        _require({}, "captured_mlp_input")


def test_cuda_sparse_mlp_witness_schema_and_failure_phase_are_stable():
    source = (Path(__file__).resolve().parents[1] / "scripts" / "cuda_sparse_mlp_witness.py").read_text()

    assert 'SCHEMA = "trellis2mlx.cuda_sparse_mlp.v1"' in source
    assert '"failure_phase": "cuda_sparse_mlp"' in source
    assert "cuda_vs_captured_after_mlp" in source


def test_module_parameter_dtype_prefers_first_parameter_dtype():
    from scripts.source_sparse_block_replay import module_parameter_dtype

    class Param:
        dtype = "float32"

    class Module:
        def parameters(self):
            return iter([Param()])

    assert module_parameter_dtype(Module(), fallback="bfloat16") == "float32"


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
