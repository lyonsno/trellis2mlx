import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


SCRIPT = Path("scripts/layernorm_census.py")


def _write_trace(path: Path, *, mlp_input: np.ndarray) -> None:
    after_cross = np.array(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
                [3.0, 4.0, 5.0, 6.0],
            ]
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        path,
        neg_block2_after_cross=after_cross,
        neg_block2_shift_mlp=np.zeros((1, 4), dtype=np.float32),
        neg_block2_scale_mlp=np.zeros((1, 4), dtype=np.float32),
        neg_block2_mlp_input=mlp_input.astype(np.float32),
        trace_block_index=np.array(2, dtype=np.int32),
        shape_flow_trace_step_index=np.array(0, dtype=np.int32),
        t=np.array(1000.0, dtype=np.float32),
        steps=np.array(8, dtype=np.int32),
    )


def test_layernorm_witness_report_preserves_route_identity_and_channel_signature(tmp_path):
    from trellmlx.layernorm_census import build_layernorm_witness_report

    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    base = np.zeros((1, 3, 4), dtype=np.float32)
    cand = base.copy()
    cand[:, :, 2] = np.float32(0.00390625)
    _write_trace(reference, mlp_input=base)
    _write_trace(candidate, mlp_input=cand)

    report = build_layernorm_witness_report(
        reference_trace_path=reference,
        candidate_trace_path=candidate,
        trace_prefix="neg_block2",
        requested_route="test-layernorm-census",
        reference_route_label="trellis-mac-pytorch-mps",
        candidate_route_label="trellis2mlx-mlx",
    )

    assert report["schema"] == "trellis2mlx.layernorm_witness.v1"
    assert report["status"] == "ok"
    assert report["requested_route"] == "test-layernorm-census"
    assert report["effective_route"] == "local-layernorm-witness-census"
    assert report["trace_identity"]["block_index"] == 2
    assert report["trace_identity"]["step_index"] == 0
    assert report["input_identity"]["after_cross_exact"] is True
    assert report["input_identity"]["shift_mlp_exact"] is True
    assert report["input_identity"]["scale_mlp_exact"] is True
    assert report["mlp_input_delta"]["max_abs_diff"] == 0.00390625
    assert report["channel_signature"]["differing_channel_count"] == 1
    assert report["channel_signature"]["channels"][0]["channel"] == 2
    assert report["channel_signature"]["channels"][0]["token_coverage"] == 3
    assert report["channel_signature"]["channels"][0]["covers_all_tokens"] is True
    assert report["known_routes"]["reference"] == "trellis-mac-pytorch-mps"
    assert report["known_routes"]["candidate"] == "trellis2mlx-mlx"
    json.dumps(report, allow_nan=False)


def test_layernorm_census_ranks_exact_numpy_variant(tmp_path):
    from trellmlx.layernorm_census import build_layernorm_witness_report

    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    after_cross = np.array(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
            ]
        ],
        dtype=np.float32,
    )
    mean = np.mean(after_cross, axis=-1, keepdims=True, dtype=np.float32)
    var = np.mean((after_cross - mean) * (after_cross - mean), axis=-1, keepdims=True, dtype=np.float32)
    expected = (after_cross - mean) / np.sqrt(var + np.float32(1e-6), dtype=np.float32)
    np.savez_compressed(
        reference,
        neg_block2_after_cross=after_cross,
        neg_block2_shift_mlp=np.zeros((1, 4), dtype=np.float32),
        neg_block2_scale_mlp=np.zeros((1, 4), dtype=np.float32),
        neg_block2_mlp_input=expected.astype(np.float32),
        trace_block_index=np.array(2, dtype=np.int32),
        shape_flow_trace_step_index=np.array(0, dtype=np.int32),
    )
    np.savez_compressed(
        candidate,
        neg_block2_after_cross=after_cross,
        neg_block2_shift_mlp=np.zeros((1, 4), dtype=np.float32),
        neg_block2_scale_mlp=np.zeros((1, 4), dtype=np.float32),
        neg_block2_mlp_input=(expected + np.float32(0.25)).astype(np.float32),
        trace_block_index=np.array(2, dtype=np.int32),
        shape_flow_trace_step_index=np.array(0, dtype=np.int32),
    )

    report = build_layernorm_witness_report(
        reference_trace_path=reference,
        candidate_trace_path=candidate,
        trace_prefix="neg_block2",
        requested_route="test-layernorm-census",
    )

    assert report["census"]["variants"][0]["name"] == "numpy_two_pass_fp32"
    assert report["census"]["variants"][0]["max_abs_diff_vs_reference"] == 0.0
    assert report["census"]["variants"][0]["exact_match_reference"] is True
    assert report["census"]["best_reference_match"] == "numpy_two_pass_fp32"
    json.dumps(report, allow_nan=False)


def test_layernorm_census_records_exact_mlx_bfloat16_candidate_match(tmp_path):
    pytest.importorskip("mlx.core")
    import mlx.core as mx

    from trellmlx.layernorm_census import build_layernorm_witness_report
    from trellmlx.models.sparse_structure_flow import _layernorm_noaffine

    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    after_cross = np.array(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
            ]
        ],
        dtype=np.float32,
    )
    x = mx.array(after_cross).astype(mx.bfloat16)
    out = _layernorm_noaffine(x).astype(mx.float32)
    mx.eval(out)
    candidate_mlp = np.array(out)
    reference_mlp = candidate_mlp.copy()
    reference_mlp[:, :, 1] += np.float32(0.00390625)
    np.savez_compressed(
        reference,
        neg_block2_after_cross=after_cross,
        neg_block2_shift_mlp=np.zeros((1, 4), dtype=np.float32),
        neg_block2_scale_mlp=np.zeros((1, 4), dtype=np.float32),
        neg_block2_mlp_input=reference_mlp,
        trace_block_index=np.array(2, dtype=np.int32),
        shape_flow_trace_step_index=np.array(0, dtype=np.int32),
    )
    np.savez_compressed(
        candidate,
        neg_block2_after_cross=after_cross,
        neg_block2_shift_mlp=np.zeros((1, 4), dtype=np.float32),
        neg_block2_scale_mlp=np.zeros((1, 4), dtype=np.float32),
        neg_block2_mlp_input=candidate_mlp,
        trace_block_index=np.array(2, dtype=np.int32),
        shape_flow_trace_step_index=np.array(0, dtype=np.int32),
    )

    report = build_layernorm_witness_report(
        reference_trace_path=reference,
        candidate_trace_path=candidate,
        trace_prefix="neg_block2",
        requested_route="test-layernorm-census",
    )

    variants = {variant["name"]: variant for variant in report["census"]["variants"]}
    assert variants["mlx_trellmlx_noaffine_bfloat16"]["exact_match_candidate"] is True
    assert variants["mlx_trellmlx_noaffine_bfloat16"]["max_abs_diff_vs_candidate"] == 0.0


def test_layernorm_census_cli_writes_failure_report_for_missing_trace(tmp_path):
    output = tmp_path / "failure.json"
    candidate = tmp_path / "candidate.npz"
    _write_trace(candidate, mlp_input=np.zeros((1, 3, 4), dtype=np.float32))

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--reference",
            str(tmp_path / "missing-reference.npz"),
            "--candidate",
            str(candidate),
            "--output",
            str(output),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode != 0
    report = json.loads(output.read_text())
    assert report["schema"] == "trellis2mlx.layernorm_witness.v1"
    assert report["status"] == "failed"
    assert report["failure_phase"] == "load"
    assert report["requested_route"] == "layernorm-witness-census"
    assert "missing-reference.npz" in report["error"]


def test_noaffine_layernorm_boundary_probe_records_rowwise_scale_explanation(tmp_path):
    from trellmlx.layernorm_census import (
        _round_float32_to_bf16,
        build_noaffine_layernorm_boundary_report,
    )

    candidate = tmp_path / "candidate.npz"
    reference = tmp_path / "reference.npz"
    x = np.array(
        [
            [
                [
                    1.2301534e-03,
                    2.9874554e-01,
                    -2.7413785e-01,
                    -8.9059186e-01,
                    -4.5467079e-01,
                    -9.9164653e-01,
                    6.0143601e-02,
                    1.3402152e00,
                ]
            ]
        ],
        dtype=np.float32,
    )
    mean = np.mean(x, axis=-1, keepdims=True, dtype=np.float32)
    centered = x - mean
    var = np.mean(centered * centered, axis=-1, keepdims=True, dtype=np.float32)
    normalized = centered / np.sqrt(var + np.float32(1e-6), dtype=np.float32)
    candidate_norm = _round_float32_to_bf16(normalized)
    reference_norm = _round_float32_to_bf16(normalized * np.float32(1.0005))

    np.savez_compressed(
        candidate,
        pos_block0_input=x,
        pos_block0_norm1=candidate_norm,
    )
    np.savez_compressed(
        reference,
        pos_block0_norm1=reference_norm,
    )

    report = build_noaffine_layernorm_boundary_report(
        reference_trace_path=reference,
        candidate_trace_path=candidate,
        input_key="pos_block0_input",
        reference_norm_key="pos_block0_norm1",
        candidate_norm_key="pos_block0_norm1",
        requested_route="test-noaffine-boundary",
    )

    assert report["schema"] == "trellis2mlx.noaffine_layernorm_boundary_probe.v1"
    assert report["requested_route"] == "test-noaffine-boundary"
    assert report["effective_route"] == "local-noaffine-layernorm-boundary-probe"
    assert report["norm_delta"]["nonzero_count"] == 1
    assert report["coordinate_summary"]["affected_token_count"] == 1
    assert report["coordinate_summary"]["affected_channel_count"] == 1
    assert report["rowwise_perturbation_probe"]["scale"]["affected_token_count"] == 1
    assert report["rowwise_perturbation_probe"]["scale"]["solved_token_count"] == 1
    assert report["rowwise_perturbation_probe"]["bias"]["affected_token_count"] == 1
    json.dumps(report, allow_nan=False)

    output = tmp_path / "boundary-report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--boundary-probe",
            "--reference",
            str(reference),
            "--candidate",
            str(candidate),
            "--output",
            str(output),
            "--input-key",
            "pos_block0_input",
            "--reference-norm-key",
            "pos_block0_norm1",
            "--candidate-norm-key",
            "pos_block0_norm1",
            "--requested-route",
            "test-noaffine-boundary-cli",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0
    cli_report = json.loads(output.read_text())
    assert cli_report["schema"] == "trellis2mlx.noaffine_layernorm_boundary_probe.v1"
    assert cli_report["requested_route"] == "test-noaffine-boundary-cli"
    assert cli_report["status"] == "ok"
