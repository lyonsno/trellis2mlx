import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import cuda_sparse_attention_witness as witness


def test_variant_specs_preserve_source_chunk_and_split_backend_precision():
    specs = witness.build_variant_specs(source_chunk_size=4096, manual_chunk_size=128)

    assert specs == [
        witness.AttentionVariant("source_default_chunk4096", "default", 4096, "input"),
        witness.AttentionVariant("default_full", "default", None, "input"),
        witness.AttentionVariant("default_chunk512", "default", 512, "input"),
        witness.AttentionVariant("math_chunk4096", "math", 4096, "input"),
        witness.AttentionVariant("math_chunk512", "math", 512, "input"),
        witness.AttentionVariant("manual_fp32_chunk128", "manual", 128, "float32"),
    ]


def test_load_witness_rejects_reference_with_different_element_count(tmp_path):
    path = tmp_path / "witness.npz"
    q = np.zeros((1, 2, 3, 4), dtype=np.float32)
    np.savez(
        path,
        pos_q=q,
        pos_k=q,
        pos_v=q,
        pos_reference_attention_raw=np.zeros((1, 2, 11), dtype=np.float32),
        pos_source_chunked_attention_raw=np.zeros((1, 2, 12), dtype=np.float32),
        route_identity_json=np.array(json.dumps({"branch": "pos"})),
    )

    try:
        witness.load_witness(path)
    except ValueError as exc:
        assert "reference_attention_raw" in str(exc)
        assert "element count" in str(exc)
    else:
        raise AssertionError("expected mismatched reference attention to fail")


def test_cli_failure_writes_report_before_primary_npz(tmp_path):
    script = Path(witness.__file__)
    report = tmp_path / "report.json"
    output = tmp_path / "result.npz"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--witness",
            str(tmp_path / "missing.npz"),
            "--output-json",
            str(report),
            "--output-npz",
            str(output),
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert report.exists()
    payload = json.loads(report.read_text())
    assert payload["status"] == "failed"
    assert payload["failure_phase"] == "input_validation"
    assert payload["primary_output_status"] == "missing"
    assert not output.exists()


def test_build_stage_selection_preserves_every_residual_and_adds_controls():
    residual = {
        "schema": "trellis2mlx.block0_split_sqrt_residual_rows.v1",
        "witness_sha256": "a" * 64,
        "rows": [
            {"token": 1, "head": 0, "max_abs": 0.25, "nonzero": 1},
            {"token": 7, "head": 1, "max_abs": 0.125, "nonzero": 1},
            {"token": 9, "head": 0, "max_abs": 0.00390625, "nonzero": 2},
        ],
    }

    selection = witness.build_stage_selection(
        residual,
        residual_report_sha256="b" * 64,
        token_count=12,
        head_count=2,
        chunk_size=4,
        control_count=4,
    )

    rows = selection["rows"]
    residual_rows = [row for row in rows if row["kind"] == "residual"]
    controls = [row for row in rows if row["kind"] == "zero_residual_control"]
    assert [(row["token"], row["head"]) for row in residual_rows] == [(1, 0), (7, 1), (9, 0)]
    assert len(controls) == 4
    assert len({(row["token"], row["head"]) for row in rows}) == len(rows)
    assert selection["selection_policy"]["residual_rows_requested"] == "all"
    assert selection["selection_policy"]["residual_rows_selected"] == 3
    assert selection["selection_policy"]["controls_selected"] == 4


def test_group_stage_rows_retains_input_order_and_full_chunk_bounds():
    rows = [
        {"token": 7, "head": 1, "kind": "residual"},
        {"token": 1, "head": 0, "kind": "residual"},
        {"token": 8, "head": 0, "kind": "zero_residual_control"},
    ]

    groups = witness.group_stage_rows(
        rows,
        token_count=10,
        head_count=2,
        chunk_size=4,
    )

    assert groups == [
        {
            "chunk_start": 0,
            "chunk_stop": 4,
            "row_indices": [1],
            "tokens": [1],
            "heads": [0],
        },
        {
            "chunk_start": 4,
            "chunk_stop": 8,
            "row_indices": [0],
            "tokens": [7],
            "heads": [1],
        },
        {
            "chunk_start": 8,
            "chunk_stop": 10,
            "row_indices": [2],
            "tokens": [8],
            "heads": [0],
        },
    ]


@pytest.mark.parametrize(
    "rows, message",
    [
        (
            [
                {"token": 1, "head": 0, "kind": "residual"},
                {"token": 1, "head": 0, "kind": "residual"},
            ],
            "duplicate",
        ),
        ([{"token": 10, "head": 0, "kind": "residual"}], "token"),
        ([{"token": 1, "head": 2, "kind": "residual"}], "head"),
    ],
)
def test_group_stage_rows_rejects_duplicate_and_out_of_range_coordinates(rows, message):
    with pytest.raises(ValueError, match=message):
        witness.group_stage_rows(
            rows,
            token_count=10,
            head_count=2,
            chunk_size=4,
        )


def test_validate_stage_outputs_rejects_selected_row_matmul_disguised_as_full_chunk():
    selection = {
        "rows": [
            {"token": 1, "head": 0, "kind": "residual"},
            {"token": 7, "head": 1, "kind": "residual"},
        ]
    }
    arrays = {
        "row_tokens": np.array([1, 7], dtype=np.int32),
        "row_heads": np.array([0, 1], dtype=np.int32),
        "scores_fp32": np.zeros((2, 10), dtype=np.float32),
        "probs_fp32": np.zeros((2, 10), dtype=np.float32),
        "output_fp32": np.zeros((2, 4), dtype=np.float32),
        "output_bf16_as_fp32": np.zeros((2, 4), dtype=np.float32),
        "source_cuda_bf16_as_fp32": np.zeros((2, 4), dtype=np.float32),
    }
    chunk_receipts = [
        {
            "chunk_start": 0,
            "chunk_stop": 4,
            "computed_query_count": 1,
            "selection_applied_after_full_chunk": True,
        },
        {
            "chunk_start": 4,
            "chunk_stop": 8,
            "computed_query_count": 1,
            "selection_applied_after_full_chunk": True,
        },
    ]

    with pytest.raises(ValueError, match="computed_query_count"):
        witness.validate_stage_outputs(
            selection=selection,
            arrays=arrays,
            chunk_receipts=chunk_receipts,
            token_count=10,
            head_dim=4,
        )


def test_validate_stage_outputs_rejects_joint_materialization_route():
    selection = {
        "rows": [{"token": 1, "head": 0, "kind": "zero_residual_control"}]
    }
    arrays = {
        "row_tokens": np.array([1], dtype=np.int32),
        "row_heads": np.array([0], dtype=np.int32),
        "scores_fp32": np.zeros((1, 4), dtype=np.float32),
        "probs_fp32": np.zeros((1, 4), dtype=np.float32),
        "output_fp32": np.zeros((1, 3), dtype=np.float32),
        "output_bf16_as_fp32": np.zeros((1, 3), dtype=np.float32),
        "source_cuda_bf16_as_fp32": np.zeros((1, 3), dtype=np.float32),
    }

    with pytest.raises(ValueError, match="independent_prefix_replays"):
        witness.validate_stage_outputs(
            selection=selection,
            arrays=arrays,
            chunk_receipts=[
                {
                    "chunk_start": 0,
                    "chunk_stop": 4,
                    "computed_query_count": 4,
                    "selection_applied_after_full_chunk": True,
                    "stage_evaluation_mode": "joint_materialization",
                }
            ],
            token_count=4,
            head_dim=3,
        )


def test_mlx_final_output_must_reproduce_prior_residual_contract():
    selection = {
        "rows": [
            {
                "token": 1,
                "head": 0,
                "kind": "residual",
                "nonzero": 1,
                "max_abs": 0.25,
            },
            {"token": 2, "head": 0, "kind": "zero_residual_control"},
        ]
    }
    arrays = {
        "source_cuda_bf16_as_fp32": np.zeros((2, 3), dtype=np.float32),
        "output_bf16_as_fp32": np.zeros((2, 3), dtype=np.float32),
    }

    with pytest.raises(ValueError, match="residual contract"):
        witness.validate_mlx_final_residual_contract(
            selection=selection,
            arrays=arrays,
        )


def test_stage_source_rows_use_authenticated_source_chunked_anchor():
    arrays = {
        "reference_attention_raw": np.zeros((3, 2, 4), dtype=np.float32),
        "source_chunked_attention_raw": np.ones((3, 2, 4), dtype=np.float32),
    }
    selection = {
        "rows": [
            {"token": 1, "head": 0, "kind": "residual"},
            {"token": 2, "head": 1, "kind": "residual"},
        ]
    }

    selected = witness._stage_source_cuda_rows(arrays, selection)

    assert np.array_equal(selected, np.ones((2, 4), dtype=np.float32))


def test_stage_cli_rejects_stale_selection_before_backend_import(tmp_path):
    q = np.zeros((1, 4, 2, 3), dtype=np.float32)
    witness_path = tmp_path / "witness.npz"
    np.savez(
        witness_path,
        pos_q=q,
        pos_k=q,
        pos_v=q,
        pos_reference_attention_raw=q,
        pos_source_chunked_attention_raw=q,
        route_identity_json=np.array(json.dumps({"branch": "pos"})),
    )
    witness_sha256 = witness._sha256_file(witness_path)
    residual_path = tmp_path / "residual.json"
    residual_path.write_text(
        json.dumps(
            {
                "schema": "trellis2mlx.block0_split_sqrt_residual_rows.v1",
                "witness_sha256": witness_sha256,
                "rows": [{"token": 1, "head": 0, "max_abs": 0.25, "nonzero": 1}],
            }
        )
    )
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "schema": "trellis2mlx.attention_residual_stage_selection.v1",
                "witness_sha256": "0" * 64,
                "branch": "pos",
                "token_count": 4,
                "head_count": 2,
                "head_dim": 3,
                "chunk_size": 2,
                "residual_report_sha256": witness._sha256_file(residual_path),
                "selection_policy": {
                    "residual_rows_requested": "all",
                    "residual_rows_selected": 1,
                    "controls_requested": 0,
                    "controls_selected": 0,
                },
                "rows": [{"token": 1, "head": 0, "kind": "residual"}],
            }
        )
    )
    report = tmp_path / "report.json"
    output = tmp_path / "result.npz"

    result = subprocess.run(
        [
            sys.executable,
            str(Path(witness.__file__)),
            "--witness",
            str(witness_path),
            "--output-json",
            str(report),
            "--output-npz",
            str(output),
            "--residual-stage-capture",
            "--stage-backend",
            "cuda",
            "--selection-json",
            str(selection_path),
            "--residual-report-json",
            str(residual_path),
            "--expected-residual-report-sha256",
            witness._sha256_file(residual_path),
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert report.exists()
    payload = json.loads(report.read_text())
    assert payload["status"] == "failed"
    assert payload["failure_phase"] == "input_validation"
    assert payload["primary_output_status"] == "missing"
    assert "witness_sha256" in payload["error"]
    assert "torch" not in sys.modules
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["omit_row", "falsify_count"])
def test_load_stage_selection_reconstructs_exact_census_selection(tmp_path, mutation):
    q = np.zeros((1, 4, 2, 3), dtype=np.float32)
    witness_path = tmp_path / "witness.npz"
    np.savez(
        witness_path,
        pos_q=q,
        pos_k=q,
        pos_v=q,
        pos_reference_attention_raw=q,
        pos_source_chunked_attention_raw=q,
        route_identity_json=np.array(json.dumps({"branch": "pos"})),
    )
    loaded = witness.load_witness(witness_path)
    residual = {
        "schema": "trellis2mlx.block0_split_sqrt_residual_rows.v1",
        "witness_sha256": loaded["sha256"],
        "rows": [
            {"token": 1, "head": 0, "max_abs": 0.25, "nonzero": 1},
            {"token": 3, "head": 1, "max_abs": 0.125, "nonzero": 1},
        ],
    }
    residual_path = tmp_path / "residual.json"
    residual_path.write_text(json.dumps(residual))
    selection = witness.build_stage_selection(
        residual,
        residual_report_sha256=witness._sha256_file(residual_path),
        token_count=4,
        head_count=2,
        head_dim=3,
        chunk_size=2,
        control_count=2,
    )
    if mutation == "omit_row":
        selection["rows"].pop(0)
    else:
        selection["selection_policy"]["residual_rows_selected"] = 1
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection))

    with pytest.raises(ValueError, match="canonical residual census"):
        witness.load_stage_selection(
            selection_path,
            loaded,
            residual_report_path=residual_path,
            expected_residual_report_sha256=witness._sha256_file(residual_path),
            expected_branch="pos",
            expected_chunk_size=2,
            expected_control_count=2,
        )


def test_load_stage_selection_rejects_substituted_residual_census(tmp_path):
    q = np.zeros((1, 4, 2, 3), dtype=np.float32)
    witness_path = tmp_path / "witness.npz"
    np.savez(
        witness_path,
        pos_q=q,
        pos_k=q,
        pos_v=q,
        pos_reference_attention_raw=q,
        pos_source_chunked_attention_raw=q,
        route_identity_json=np.array(json.dumps({"branch": "pos"})),
    )
    loaded = witness.load_witness(witness_path)
    residual = {
        "schema": "trellis2mlx.block0_split_sqrt_residual_rows.v1",
        "witness_sha256": loaded["sha256"],
        "rows": [{"token": 1, "head": 0, "max_abs": 0.25, "nonzero": 1}],
    }
    residual_path = tmp_path / "residual.json"
    residual_path.write_text(json.dumps(residual))
    expected_residual_sha256 = witness._sha256_file(residual_path)
    selection = witness.build_stage_selection(
        residual,
        residual_report_sha256=expected_residual_sha256,
        token_count=4,
        head_count=2,
        head_dim=3,
        chunk_size=2,
        control_count=1,
    )
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection))
    residual["rows"][0]["max_abs"] = 0.125
    residual_path.write_text(json.dumps(residual))

    with pytest.raises(ValueError, match="residual census SHA256"):
        witness.load_stage_selection(
            selection_path,
            loaded,
            residual_report_path=residual_path,
            expected_residual_report_sha256=expected_residual_sha256,
            expected_branch="pos",
            expected_chunk_size=2,
            expected_control_count=1,
        )


@pytest.mark.parametrize("mutation", ["zero_controls", "chunk_size", "branch"])
def test_load_stage_selection_requires_caller_route_expectations(tmp_path, mutation):
    q = np.zeros((1, 4, 2, 3), dtype=np.float32)
    witness_path = tmp_path / "witness.npz"
    np.savez(
        witness_path,
        pos_q=q,
        pos_k=q,
        pos_v=q,
        pos_reference_attention_raw=q,
        pos_source_chunked_attention_raw=q,
        neg_q=q,
        neg_k=q,
        neg_v=q,
        neg_reference_attention_raw=q,
        neg_source_chunked_attention_raw=q,
        route_identity_json=np.array(json.dumps({"branch": "both"})),
    )
    loaded = witness.load_witness(witness_path)
    residual = {
        "schema": "trellis2mlx.block0_split_sqrt_residual_rows.v1",
        "witness_sha256": loaded["sha256"],
        "rows": [{"token": 1, "head": 0, "max_abs": 0.25, "nonzero": 1}],
    }
    residual_path = tmp_path / "residual.json"
    residual_path.write_text(json.dumps(residual))
    residual_sha256 = witness._sha256_file(residual_path)
    selection = witness.build_stage_selection(
        residual,
        residual_report_sha256=residual_sha256,
        token_count=4,
        head_count=2,
        head_dim=3,
        chunk_size=1 if mutation == "chunk_size" else 2,
        control_count=0 if mutation == "zero_controls" else 2,
        branch="neg" if mutation == "branch" else "pos",
    )
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection))

    with pytest.raises(ValueError, match="caller expectation"):
        witness.load_stage_selection(
            selection_path,
            loaded,
            residual_report_path=residual_path,
            expected_residual_report_sha256=residual_sha256,
            expected_branch="pos",
            expected_chunk_size=2,
            expected_control_count=2,
        )


def test_persisted_cuda_numeric_mismatch_fails_admission():
    selection = {
        "token_count": 4,
        "head_count": 1,
        "head_dim": 2,
        "chunk_size": 4,
        "rows": [{"token": 1, "head": 0, "kind": "residual"}],
    }
    arrays = {
        "row_tokens": np.array([1], dtype=np.int32),
        "row_heads": np.array([0], dtype=np.int32),
        "scores_fp32": np.zeros((1, 4), dtype=np.float32),
        "probs_fp32": np.zeros((1, 4), dtype=np.float32),
        "output_fp32": np.zeros((1, 2), dtype=np.float32),
        "output_bf16_as_fp32": np.array([[1.0, 0.0]], dtype=np.float32),
        "source_cuda_bf16_as_fp32": np.zeros((1, 2), dtype=np.float32),
    }
    receipt = {
        "chunk_start": 0,
        "chunk_stop": 4,
        "computed_query_count": 4,
        "selection_applied_after_full_chunk": True,
        "stage_evaluation_mode": "independent_prefix_replays",
    }

    with pytest.raises(ValueError, match="persisted CUDA"):
        witness.validate_persisted_stage_admission(
            backend="cuda",
            selection=selection,
            arrays=arrays,
            chunk_receipts=[receipt],
            persisted_route={"route": "exact"},
            expected_route={"route": "exact"},
        )


def test_stage_cli_preserves_preexisting_primary_output(tmp_path):
    report = tmp_path / "report.json"
    output = tmp_path / "result.npz"
    output.write_bytes(b"do-not-overwrite")

    result = subprocess.run(
        [
            sys.executable,
            str(Path(witness.__file__)),
            "--witness",
            str(tmp_path / "missing.npz"),
            "--output-json",
            str(report),
            "--output-npz",
            str(output),
            "--residual-stage-capture",
            "--stage-backend",
            "cuda",
            "--selection-json",
            str(tmp_path / "missing-selection.json"),
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert output.read_bytes() == b"do-not-overwrite"
    payload = json.loads(report.read_text())
    assert payload["failure_phase"] == "request_validation"
    assert payload["primary_output_status"] == "preexisting_untrusted_preserved"
