from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import pytest


def test_projection_comparison_records_full_metrics_and_decision_flips():
    from scripts.cuda_decoder_level2_subdiv_cublas_sweep import (
        compare_projection_logits,
    )

    source = np.asarray(
        [[-2.0, -0.25, 0.5, 4.0], [1.0, -1.0, 0.0, 2.0]],
        dtype=np.float16,
    )
    candidate = source.copy()
    candidate[0, 1] = np.float16(0.25)
    candidate[1, 0] = np.float16(1.5)

    result = compare_projection_logits(source, candidate)

    assert result == {
        "exact": False,
        "nonzero_count": 2,
        "mean_abs": 0.125,
        "rms": pytest.approx(0.25),
        "max_abs": 0.5,
        "decision_flip_count": 1,
        "rows_with_decision_flip": 1,
        "source_positive_candidate_nonpositive": 0,
        "source_nonpositive_candidate_positive": 1,
    }


def test_projection_comparison_rejects_partial_wrong_dtype_and_nonfinite():
    from scripts.cuda_decoder_level2_subdiv_cublas_sweep import (
        compare_projection_logits,
    )

    source = np.zeros((2, 8), dtype=np.float16)
    with pytest.raises(ValueError, match="same nonempty two-dimensional shape"):
        compare_projection_logits(source, source[:1])
    with pytest.raises(ValueError, match="float16"):
        compare_projection_logits(source, source.astype(np.float32))
    candidate = source.copy()
    candidate[0, 0] = np.float16(np.nan)
    with pytest.raises(ValueError, match="non-finite"):
        compare_projection_logits(source, candidate)


def test_sweep_main_rejects_wrong_digest_before_torch_and_clears_stale_output(
    monkeypatch,
    tmp_path,
):
    from scripts import cuda_decoder_level2_subdiv_cublas_sweep as sweep

    input_path = tmp_path / "source-trace.npz"
    input_path.write_bytes(b"not-the-requested-source-trace")
    output_path = tmp_path / "sweep.json"
    output_path.write_text('{"status": "done"}\n')
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    assert (
        sweep.main(
            [
                "--source-trace",
                str(input_path),
                "--expected-source-trace-sha256",
                "0" * 64,
                "--output-json",
                str(output_path),
                "--expected-rows",
                "2",
            ]
        )
        == 1
    )

    report = json.loads(output_path.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "input_validation"
    assert report["last_trustworthy_phase"] == "output_path_validated"
    assert report["source_trace"]["requested_sha256"] == "0" * 64
    assert "source trace sha256 mismatch" in report["error"]
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() != "0" * 64
    assert "torch" not in sys.modules


def test_sweep_main_records_cuda_failure_without_claiming_algorithm_results(
    monkeypatch,
    tmp_path,
):
    from scripts import cuda_decoder_level2_subdiv_cublas_sweep as sweep
    from scripts.decoder_level2_subdiv_trace_contract import (
        write_decoder_level2_subdiv_trace_npz,
    )

    rows = 2
    source_path = tmp_path / "source-trace.npz"
    arrays = {
        "level2_child_coords": np.arange(rows * 4, dtype=np.int32).reshape(
            rows,
            4,
        ),
        "level2_block0_output": np.zeros((rows, 256), dtype=np.float16),
        "level2_block7_output": np.zeros((rows, 256), dtype=np.float16),
        "level2_upsample_subdiv_weight": np.zeros(
            (8, 256),
            dtype=np.float16,
        ),
        "level2_upsample_subdiv_bias": np.zeros(8, dtype=np.float16),
        "level2_upsample_subdiv_logits": np.zeros(
            (rows, 8),
            dtype=np.float16,
        ),
    }
    write_decoder_level2_subdiv_trace_npz(source_path, arrays)
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    output_path = tmp_path / "sweep.json"

    def fail_cuda(_arrays):
        raise RuntimeError("synthetic CUDA failure")

    monkeypatch.setattr(sweep, "_run_cuda_sweep", fail_cuda)

    assert (
        sweep.main(
            [
                "--source-trace",
                str(source_path),
                "--expected-source-trace-sha256",
                digest,
                "--output-json",
                str(output_path),
                "--expected-rows",
                str(rows),
            ]
        )
        == 1
    )

    report = json.loads(output_path.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "cuda_execution"
    assert report["last_trustworthy_phase"] == "input_validated"
    assert "algorithm_results" not in report
    assert "synthetic CUDA failure" in report["error"]


def test_sweep_main_rejects_output_alias_without_overwriting_source(tmp_path):
    from scripts import cuda_decoder_level2_subdiv_cublas_sweep as sweep

    source_path = tmp_path / "source-trace.npz"
    original = b"source-trace-must-survive"
    source_path.write_bytes(original)

    assert (
        sweep.main(
            [
                "--source-trace",
                str(source_path),
                "--expected-source-trace-sha256",
                hashlib.sha256(original).hexdigest(),
                "--output-json",
                str(source_path),
            ]
        )
        == 1
    )

    assert source_path.read_bytes() == original


def test_sweep_main_rejects_repeated_algorithm_result_inventory(
    monkeypatch,
    tmp_path,
):
    from scripts import cuda_decoder_level2_subdiv_cublas_sweep as sweep
    from scripts.cuda_decoder_block0_gemm_witness import (
        LEGACY_CUBLAS_EXPLICIT_ALGORITHM_IDS,
    )
    from scripts.decoder_level2_subdiv_trace_contract import (
        write_decoder_level2_subdiv_trace_npz,
    )

    rows = 2
    source_path = tmp_path / "source-trace.npz"
    arrays = {
        "level2_child_coords": np.arange(rows * 4, dtype=np.int32).reshape(
            rows,
            4,
        ),
        "level2_block0_output": np.zeros((rows, 256), dtype=np.float16),
        "level2_block7_output": np.zeros((rows, 256), dtype=np.float16),
        "level2_upsample_subdiv_weight": np.zeros(
            (8, 256),
            dtype=np.float16,
        ),
        "level2_upsample_subdiv_bias": np.zeros(8, dtype=np.float16),
        "level2_upsample_subdiv_logits": np.zeros(
            (rows, 8),
            dtype=np.float16,
        ),
    }
    write_decoder_level2_subdiv_trace_npz(source_path, arrays)
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    output_path = tmp_path / "sweep.json"
    requested = list(LEGACY_CUBLAS_EXPLICIT_ALGORITHM_IDS)
    repeated = [
        {
            "algorithm_id": 0,
            "separate_bias": {"exact": True},
            "fused_beta_bias": {"exact": False},
        }
        for _ in requested
    ]

    monkeypatch.setattr(
        sweep,
        "_run_cuda_sweep",
        lambda _arrays: {
            "runtime": {
                "torch": "2.10.0+cu128",
                "cuda_device": "Tesla T4",
            },
            "source_self_authentication": {
                "torch_f_linear_exact_source": True,
            },
            "matrix": {
                "rows": rows,
                "reduction": 256,
                "channels": 8,
            },
            "algorithm_inventory": {
                "requested_ids": requested,
                "complete": True,
            },
            "algorithm_results": repeated,
            "exact_matches": {
                "separate_bias_algorithm_ids": [0],
                "fused_beta_bias_algorithm_ids": [],
            },
        },
    )

    assert (
        sweep.main(
            [
                "--source-trace",
                str(source_path),
                "--expected-source-trace-sha256",
                digest,
                "--output-json",
                str(output_path),
                "--expected-rows",
                str(rows),
            ]
        )
        == 1
    )
    report = json.loads(output_path.read_text())
    assert report["status"] == "failed"
    assert "algorithm result IDs do not match" in report["error"]


def test_sweep_main_publishes_complete_uncapped_algorithm_inventory(
    monkeypatch,
    tmp_path,
):
    from scripts import cuda_decoder_level2_subdiv_cublas_sweep as sweep
    from scripts.cuda_decoder_block0_gemm_witness import (
        LEGACY_CUBLAS_EXPLICIT_ALGORITHM_IDS,
    )
    from scripts.decoder_level2_subdiv_trace_contract import (
        write_decoder_level2_subdiv_trace_npz,
    )

    rows = 2
    source_path = tmp_path / "source-trace.npz"
    arrays = {
        "level2_child_coords": np.arange(rows * 4, dtype=np.int32).reshape(
            rows,
            4,
        ),
        "level2_block0_output": np.zeros((rows, 256), dtype=np.float16),
        "level2_block7_output": np.zeros((rows, 256), dtype=np.float16),
        "level2_upsample_subdiv_weight": np.zeros(
            (8, 256),
            dtype=np.float16,
        ),
        "level2_upsample_subdiv_bias": np.zeros(8, dtype=np.float16),
        "level2_upsample_subdiv_logits": np.zeros(
            (rows, 8),
            dtype=np.float16,
        ),
    }
    write_decoder_level2_subdiv_trace_npz(source_path, arrays)
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    output_path = tmp_path / "sweep.json"
    requested = list(LEGACY_CUBLAS_EXPLICIT_ALGORITHM_IDS)

    monkeypatch.setattr(
        sweep,
        "_run_cuda_sweep",
        lambda _arrays: {
            "runtime": {
                "torch": "2.10.0+cu128",
                "cuda_device": "Tesla T4",
            },
            "source_self_authentication": {
                "torch_f_linear_exact_source": True,
            },
            "matrix": {
                "rows": rows,
                "reduction": 256,
                "channels": 8,
            },
            "algorithm_inventory": {
                "requested_ids": requested,
                "complete": True,
            },
            "algorithm_results": [
                {
                    "algorithm_id": value,
                    "status": 0,
                    "separate_bias": {"exact": value == 0},
                    "fused_beta_bias": {"exact": value == 1},
                }
                for value in requested
            ],
            "exact_matches": {
                "separate_bias_algorithm_ids": [0],
                "fused_beta_bias_algorithm_ids": [1],
            },
        },
    )

    assert (
        sweep.main(
            [
                "--source-trace",
                str(source_path),
                "--expected-source-trace-sha256",
                digest,
                "--output-json",
                str(output_path),
                "--expected-rows",
                str(rows),
            ]
        )
        == 0
    )

    report = json.loads(output_path.read_text())
    assert report["status"] == "done"
    assert report["schema"] == sweep.SCHEMA
    assert report["last_trustworthy_phase"] == "report_reopened_exact"
    assert report["requested_route"] == {
        "operation": "shape_decoder.level2.upsample.to_subdiv",
        "torch": "2.10.0+cu128",
        "cuda_device": "Tesla T4",
        "projection": "torch.nn.functional.linear",
        "explicit_backend": "legacy_cublasGemmEx",
        "bias_variants": ["separate-fp16-add", "fused-beta-one"],
    }
    assert report["effective_route"]["torch"] == "2.10.0+cu128"
    assert report["effective_route"]["cuda_device"] == "Tesla T4"
    assert report["algorithm_inventory"] == {
        "requested_ids": requested,
        "complete": True,
    }
    assert len(report["algorithm_results"]) == 40
    assert report["exact_matches"] == {
        "separate_bias_algorithm_ids": [0],
        "fused_beta_bias_algorithm_ids": [1],
    }
