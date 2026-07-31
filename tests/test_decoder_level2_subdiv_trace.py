from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from scripts.decoder_level2_block0_trace_contract import (
    decoder_boundary_hash_entry,
)
from scripts.decoder_level2_subdiv_trace_contract import (
    COMPARISON_SCHEMA,
    TRACE_ARRAY_NAMES,
    compare_decoder_level2_subdiv_traces,
    load_decoder_level2_subdiv_trace,
    validate_decoder_level2_subdiv_trace,
    validate_parent_evidence,
    write_decoder_level2_subdiv_trace_npz,
)
from scripts.decoder_level1_trace_contract import decoder_level1_hash_entry
from trellmlx.decoder_level2_subdiv_trace import project_level2_subdiv


def _arrays(rows: int = 3, *, logit_offset: float = 0.0):
    coords = np.arange(rows * 4, dtype=np.int32).reshape(rows, 4)
    block0 = np.arange(rows * 256, dtype=np.float16).reshape(rows, 256)
    block7 = block0 + np.float16(1)
    weight = np.arange(8 * 256, dtype=np.float16).reshape(8, 256) / 256
    bias = np.arange(8, dtype=np.float16) / 8
    logits = np.arange(rows * 8, dtype=np.float16).reshape(rows, 8)
    logits = logits + np.float16(logit_offset)
    return {
        "level2_child_coords": coords,
        "level2_block0_output": block0,
        "level2_block7_output": block7,
        "level2_upsample_subdiv_weight": weight,
        "level2_upsample_subdiv_bias": bias,
        "level2_upsample_subdiv_logits": logits,
    }


def _block0_comparison(arrays):
    coords = decoder_boundary_hash_entry(
        "level2_child_coords",
        arrays["level2_child_coords"],
    )
    block0 = decoder_boundary_hash_entry(
        "level2_block0_output",
        arrays["level2_block0_output"],
    )
    return {
        "schema": "trellis2mlx.decoder_level2_block0_trace_comparison.v2",
        "status": "done",
        "first_nonexact_boundary": None,
        "child_input_identity": {
            "source": {"level2_child_coords": coords},
            "local": {"level2_child_coords": coords},
        },
        "child_output_identity": {
            "source": block0,
            "local": block0,
        },
        "parent_fork_disposition": {
            "requested": "corrected-child-exact-to-source",
            "effective": "corrected-child-exact-to-source",
            "all_block_boundaries_exact_to_source": True,
        },
    }


def _ledger_comparison(source, local):
    block7 = decoder_level1_hash_entry(
        "level2_block7_output",
        source["level2_block7_output"],
    )
    source_logits = decoder_level1_hash_entry(
        "level2_upsample_subdiv_logits",
        source["level2_upsample_subdiv_logits"],
    )
    local_logits = decoder_level1_hash_entry(
        "level2_upsample_subdiv_logits",
        local["level2_upsample_subdiv_logits"],
    )
    return {
        "schema": "trellis2mlx.decoder_level1_trace_comparison.v1",
        "status": "done",
        "first_nonexact_hash_boundary": "level2_upsample_subdiv_logits",
        "hash_boundaries": [
            {
                **block7,
                "exact": True,
                "source_sha256": block7["sha256"],
                "local_sha256": block7["sha256"],
            },
            {
                "name": source_logits["name"],
                "dtype": source_logits["dtype"],
                "shape": source_logits["shape"],
                "exact": False,
                "source_sha256": source_logits["sha256"],
                "local_sha256": local_logits["sha256"],
            },
        ],
    }


def test_trace_contract_rejects_partial_extra_and_wrong_shapes():
    arrays = _arrays()
    assert tuple(validate_decoder_level2_subdiv_trace(arrays)) == (
        TRACE_ARRAY_NAMES
    )

    partial = dict(arrays)
    partial.pop("level2_upsample_subdiv_bias")
    with pytest.raises(ValueError, match="missing"):
        validate_decoder_level2_subdiv_trace(partial)

    extra = {**arrays, "unbound": np.zeros(1, dtype=np.float16)}
    with pytest.raises(ValueError, match="extra"):
        validate_decoder_level2_subdiv_trace(extra)

    wrong = dict(arrays)
    wrong["level2_upsample_subdiv_weight"] = np.zeros(
        (256, 8),
        dtype=np.float16,
    )
    with pytest.raises(ValueError, match="weight"):
        validate_decoder_level2_subdiv_trace(wrong)


def test_trace_writer_reopens_exact_and_rejects_stale_primary(tmp_path):
    path = tmp_path / "trace.npz"
    validation = write_decoder_level2_subdiv_trace_npz(path, _arrays())
    assert validation["reopened_exact"] is True
    assert validation["rows"] == 3
    loaded = load_decoder_level2_subdiv_trace(path)
    assert all(
        np.array_equal(loaded[name], _arrays()[name])
        for name in TRACE_ARRAY_NAMES
    )

    stale = tmp_path / "stale.npz"
    stale.write_bytes(b"stale")
    wrong = _arrays()
    wrong.pop("level2_upsample_subdiv_logits")
    with pytest.raises(ValueError, match="missing"):
        write_decoder_level2_subdiv_trace_npz(stale, wrong)
    assert not stale.exists()


def test_parent_evidence_rejects_detached_block0_and_false_ledger_fork():
    source = _arrays()
    local = _arrays(logit_offset=1)
    block0 = _block0_comparison(source)
    ledger = _ledger_comparison(source, local)
    parent = {
        "level2_child_coords": source["level2_child_coords"],
        "level2_block0_output": source["level2_block0_output"],
    }
    validated = validate_parent_evidence(parent, block0, ledger)
    assert validated["level2_block7_output"]["exact"] is True

    detached = deepcopy(parent)
    detached["level2_block0_output"][0, 0] += np.float16(1)
    with pytest.raises(ValueError, match="block0 output"):
        validate_parent_evidence(detached, block0, ledger)

    false_fork = deepcopy(ledger)
    false_fork["first_nonexact_hash_boundary"] = "level2_block7_output"
    with pytest.raises(ValueError, match="first nonexact"):
        validate_parent_evidence(parent, block0, false_fork)


def test_historical_projection_requires_both_authenticated_ledger_hashes():
    source = _arrays()
    local = _arrays(logit_offset=1)
    block0 = _block0_comparison(source)
    ledger = _ledger_comparison(source, local)
    result = compare_decoder_level2_subdiv_traces(
        source,
        local,
        block0_comparison=block0,
        ledger_comparison=ledger,
        projection_disposition="historical-turing-fda",
    )
    assert result["schema"] == COMPARISON_SCHEMA
    assert result["status"] == "done"
    assert result["first_nonexact_boundary"] == (
        "level2_upsample_subdiv_logits"
    )
    assert result["projection_disposition"]["effective"] == (
        "historical-turing-fda"
    )

    detached = _arrays(logit_offset=2)
    with pytest.raises(ValueError, match="historical local logits"):
        compare_decoder_level2_subdiv_traces(
            source,
            detached,
            block0_comparison=block0,
            ledger_comparison=ledger,
            projection_disposition="historical-turing-fda",
        )


def test_projection_candidate_keeps_source_and_upstream_anchors_binding():
    source = _arrays()
    historical = _arrays(logit_offset=1)
    candidate = _arrays(logit_offset=2)
    block0 = _block0_comparison(source)
    ledger = _ledger_comparison(source, historical)
    result = compare_decoder_level2_subdiv_traces(
        source,
        candidate,
        block0_comparison=block0,
        ledger_comparison=ledger,
        projection_disposition="projection-candidate",
    )
    assert result["projection_disposition"]["effective"] == (
        "projection-candidate"
    )
    assert result["projection_disposition"]["historical_local_sha256"] != (
        result["projection_disposition"]["current_local_sha256"]
    )

    detached_source = _arrays(logit_offset=3)
    with pytest.raises(ValueError, match="source logits"):
        compare_decoder_level2_subdiv_traces(
            detached_source,
            candidate,
            block0_comparison=block0,
            ledger_comparison=ledger,
            projection_disposition="projection-candidate",
        )

    detached_upstream = _arrays(logit_offset=2)
    detached_upstream["level2_block7_output"] += np.float16(1)
    with pytest.raises(ValueError, match="block7"):
        compare_decoder_level2_subdiv_traces(
            source,
            detached_upstream,
            block0_comparison=block0,
            ledger_comparison=ledger,
            projection_disposition="projection-candidate",
        )


def test_projection_backend_selects_one_explicit_affine_route():
    calls = []

    class Linear:
        weight = np.arange(16, dtype=np.float16).reshape(2, 8)
        bias = np.arange(2, dtype=np.float16)

        def __call__(self, value):
            calls.append(("native", value.shape))
            return value @ self.weight.T + self.bias

    def turing(value, weight, bias):
        calls.append(("turing_fda", value.shape, weight.shape, bias.shape))
        return value @ weight + bias + np.float16(1)

    value = np.arange(24, dtype=np.float16).reshape(3, 8)
    linear = Linear()
    native = project_level2_subdiv(
        linear,
        value,
        "native",
        turing_linear=turing,
    )
    turing_result = project_level2_subdiv(
        linear,
        value,
        "turing_fda",
        turing_linear=turing,
    )
    assert np.array_equal(native, value @ linear.weight.T + linear.bias)
    assert np.array_equal(turing_result, native + np.float16(1))
    assert calls == [
        ("native", (3, 8)),
        ("turing_fda", (3, 8), (8, 2), (2,)),
    ]
    with pytest.raises(ValueError, match="projection backend"):
        project_level2_subdiv(
            linear,
            value,
            "invented",
            turing_linear=turing,
        )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runner_args(tmp_path, *, expected_parent_sha):
    paths = {}
    for name in (
        "parent",
        "block0",
        "ledger",
        "checkpoint",
        "silu",
        "rsqrt",
    ):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode("ascii"))
        paths[name] = path
    return [
        "--parent-block0-trace",
        str(paths["parent"]),
        "--expected-parent-block0-trace-sha256",
        expected_parent_sha,
        "--block0-comparison",
        str(paths["block0"]),
        "--expected-block0-comparison-sha256",
        _sha256(paths["block0"]),
        "--ledger-comparison",
        str(paths["ledger"]),
        "--expected-ledger-comparison-sha256",
        _sha256(paths["ledger"]),
        "--shape-decoder-checkpoint",
        str(paths["checkpoint"]),
        "--expected-checkpoint-sha256",
        _sha256(paths["checkpoint"]),
        "--decoder-silu-lut",
        str(paths["silu"]),
        "--expected-decoder-silu-lut-sha256",
        _sha256(paths["silu"]),
        "--turing-rsqrt-lut",
        str(paths["rsqrt"]),
        "--expected-turing-rsqrt-lut-sha256",
        _sha256(paths["rsqrt"]),
        "--expected-repo-commit",
        "a" * 40,
        "--projection-backend",
        "native",
        "--output-npz",
        str(tmp_path / "output.npz"),
        "--output-json",
        str(tmp_path / "report.json"),
    ]


def test_local_runner_rejects_parent_substitution_before_runtime(tmp_path):
    from scripts.run_mlx_decoder_level2_subdiv_trace import main

    stale = tmp_path / "output.npz"
    stale.write_bytes(b"stale")
    result = main(
        _runner_args(
            tmp_path,
            expected_parent_sha="0" * 64,
        )
    )
    assert result == 1
    assert not stale.exists()
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "parent_authentication"
    assert report["primary"]["status"] == "not_written"
    assert report["stale_primary_invalidated"] is True
    assert report["requested_route"]["projection_backend"] == "native"
    assert "mlx.core" not in report.get("imports_completed", [])


def test_local_runner_rejects_primary_input_collision_with_durable_report(
    tmp_path,
):
    from scripts.run_mlx_decoder_level2_subdiv_trace import main

    args = _runner_args(tmp_path, expected_parent_sha="0" * 64)
    expected_index = (
        args.index("--expected-parent-block0-trace-sha256") + 1
    )
    args[expected_index] = _sha256(tmp_path / "parent.bin")
    output_index = args.index("--output-npz") + 1
    args[output_index] = str(tmp_path / "parent.bin")
    result = main(args)
    assert result == 1
    failure_reports = sorted(tmp_path.glob("report.json*"))
    assert failure_reports
    report = json.loads(failure_reports[-1].read_text())
    assert report["failure_phase"] == "request_validation"
    assert "collides" in report["error"]


def _write_json(path, payload):
    path.write_text(json.dumps(payload))
    return path


def _comparison_cli_fixture(
    tmp_path,
    *,
    projection_disposition="projection-candidate",
    local_backend="native",
):
    source = _arrays()
    historical = _arrays(logit_offset=1)
    local = (
        historical
        if projection_disposition == "historical-turing-fda"
        else _arrays(logit_offset=2)
    )
    source_path = tmp_path / "source.npz"
    local_path = tmp_path / "local.npz"
    write_decoder_level2_subdiv_trace_npz(source_path, source)
    write_decoder_level2_subdiv_trace_npz(local_path, local)
    block0_path = _write_json(
        tmp_path / "block0.json",
        _block0_comparison(source),
    )
    ledger_path = _write_json(
        tmp_path / "ledger.json",
        _ledger_comparison(source, historical),
    )
    source_report = _write_json(
        tmp_path / "source-report.json",
        {
            "schema": "trellis2mlx.source_cuda_shape_slat_grid_decode.v1",
            "status": "done",
            "effective_route": {
                "route": (
                    "official-source-cuda-shape-decoder-"
                    "level2-subdiv-trace"
                ),
                "device_type": "cuda",
                "cuda_device": "Tesla T4",
                "decoder_level2_subdiv_trace": True,
                "projection_backend": "torch-F.linear",
            },
            "decoder_trace_artifacts": [
                {
                    "status": "written",
                    "path": str(source_path),
                    "sha256": _sha256(source_path),
                    "projection_backend": "torch-F.linear",
                }
            ],
        },
    )
    local_report = _write_json(
        tmp_path / "local-report.json",
        {
            "schema": "trellis2mlx.decoder_level2_subdiv_trace.v1",
            "status": "done",
            "effective_route": {
                "route": "mlx-shape-decoder-level2-subdiv-trace",
                "device_type": "metal",
                "decoder_linear_backend": "turing_fda",
                "sparse_conv_matmul_backend": "turing_fda",
                "projection_backend": local_backend,
            },
            "primary": {
                "status": "written",
                "path": str(local_path),
                "sha256": _sha256(local_path),
                "validation": {"reopened_exact": True},
            },
        },
    )
    return [
        "--source",
        str(source_path),
        "--source-report",
        str(source_report),
        "--local",
        str(local_path),
        "--local-report",
        str(local_report),
        "--block0-comparison",
        str(block0_path),
        "--expected-block0-comparison-sha256",
        _sha256(block0_path),
        "--ledger-comparison",
        str(ledger_path),
        "--expected-ledger-comparison-sha256",
        _sha256(ledger_path),
        "--projection-disposition",
        projection_disposition,
        "--output",
        str(tmp_path / "comparison.json"),
    ]


def test_comparator_cli_binds_routes_reports_and_candidate_disposition(
    tmp_path,
):
    from scripts.compare_decoder_level2_subdiv_traces import main

    result = main(_comparison_cli_fixture(tmp_path))
    assert result == 0
    report = json.loads((tmp_path / "comparison.json").read_text())
    assert report["status"] == "done"
    assert report["projection_disposition"]["effective"] == (
        "projection-candidate"
    )
    assert report["artifacts"]["source"]["projection_backend"] == (
        "torch-F.linear"
    )
    assert report["artifacts"]["local"]["projection_backend"] == "native"


def test_comparator_cli_rejects_historical_route_lie_and_preserves_failure(
    tmp_path,
):
    from scripts.compare_decoder_level2_subdiv_traces import main

    result = main(
        _comparison_cli_fixture(
            tmp_path,
            projection_disposition="historical-turing-fda",
            local_backend="native",
        )
    )
    assert result == 1
    report = json.loads((tmp_path / "comparison.json").read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "report_validation"
    assert "projection backend" in report["error"]


def test_comparator_cli_rejects_output_collision_without_mutating_primary(
    tmp_path,
):
    from scripts.compare_decoder_level2_subdiv_traces import main

    args = _comparison_cli_fixture(tmp_path)
    source_path = tmp_path / "source.npz"
    original = source_path.read_bytes()
    args[args.index("--output") + 1] = str(source_path)
    result = main(args)
    assert result == 1
    assert source_path.read_bytes() == original
    failure_reports = sorted(tmp_path.glob("source.npz.failure*.json"))
    assert failure_reports
    report = json.loads(failure_reports[-1].read_text())
    assert report["failure_phase"] == "request_validation"
