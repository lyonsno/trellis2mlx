from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from scripts.decoder_full_hash_ledger_contract import (
    FULL_DECODER_HASH_BOUNDARY_NAMES,
    build_decoder_full_hash_ledger,
    decoder_full_hash_entry,
)


def _full_ledger(*, level3_rows: int = 3, level4_rows: int = 5):
    shapes = {
        "level2_upsample_output": (level3_rows, 128),
        "level3_block0_conv": (level3_rows, 128),
        "level3_block0_norm": (level3_rows, 128),
        "level3_block0_mlp_fc1": (level3_rows, 512),
        "level3_block0_silu": (level3_rows, 512),
        "level3_block0_mlp_fc2": (level3_rows, 128),
        "level3_block0_output": (level3_rows, 128),
        "level3_block1_output": (level3_rows, 128),
        "level3_block2_output": (level3_rows, 128),
        "level3_block3_output": (level3_rows, 128),
        "level3_upsample_subdiv_logits": (level3_rows, 8),
        "level3_upsample_norm1": (level3_rows, 128),
        "level3_upsample_silu1": (level3_rows, 128),
        "level3_upsample_conv1": (level3_rows, 512),
        "level4_child_coords": (level4_rows, 4),
        "level3_upsample_h_c2s": (level4_rows, 64),
        "level3_upsample_skip_c2s": (level4_rows, 16),
        "level3_upsample_skip_repeated": (level4_rows, 64),
        "level3_upsample_norm2": (level4_rows, 64),
        "level3_upsample_silu2": (level4_rows, 64),
        "level3_upsample_conv2": (level4_rows, 64),
        "level3_upsample_output": (level4_rows, 64),
        "decoder_final_layernorm": (level4_rows, 64),
        "decoder_output": (level4_rows, 7),
    }
    arrays = {}
    for index, name in enumerate(FULL_DECODER_HASH_BOUNDARY_NAMES):
        dtype = np.int32 if name == "level4_child_coords" else (
            np.float32
            if name in {"decoder_final_layernorm", "decoder_output"}
            else np.float16
        )
        arrays[name] = np.full(shapes[name], index, dtype=dtype)
    return arrays, build_decoder_full_hash_ledger(arrays)


def _source_report(ledger):
    requested_route = {
        "route": "official-source-cuda-shape-decoder-full-hash-ledger",
        "full_decoder_hash_ledger": True,
        "decoder_level1_trace": True,
        "one_model_load": True,
        "decoder_output_head_backend": "torch-sparse-linear-fp32",
    }
    return {
        "schema": "trellis2mlx.source_cuda_shape_slat_grid_decode.v1",
        "status": "done",
        "requested_route": requested_route,
        "effective_route": {
            **requested_route,
            "device_type": "cuda",
            "cuda_device": "Tesla T4",
        },
        "decoder_trace_artifacts": [
            {
                "path": "decoder-traces/source.npz",
                "sha256": "a" * 64,
                "status": "written",
                "full_decoder_hash_ledger": ledger,
            }
        ],
    }


def test_width64_witness_contract_round_trips_full_values(tmp_path):
    from scripts.decoder_level3_norm2_width64_witness_contract import (
        load_decoder_level3_norm2_width64_witness,
        write_decoder_level3_norm2_width64_witness_npz,
    )

    input_values = np.arange(5 * 64, dtype=np.float16).reshape(5, 64)
    candidate = np.flip(input_values, axis=1).copy()
    output = tmp_path / "witness.npz"

    report = write_decoder_level3_norm2_width64_witness_npz(
        output,
        {
            "level3_upsample_h_c2s": input_values,
            "level3_upsample_norm2_candidate": candidate,
        },
    )
    reopened = load_decoder_level3_norm2_width64_witness(output)

    assert report == {
        "schema": "trellis2mlx.decoder_level3_norm2_width64_witness.v1",
        "rows": 5,
        "channels": 64,
        "array_names": [
            "level3_upsample_h_c2s",
            "level3_upsample_norm2_candidate",
        ],
        "reopened_exact": True,
    }
    np.testing.assert_array_equal(
        reopened["level3_upsample_h_c2s"],
        input_values,
    )
    np.testing.assert_array_equal(
        reopened["level3_upsample_norm2_candidate"],
        candidate,
    )


def test_width64_witness_contract_rejects_wrong_width():
    from scripts.decoder_level3_norm2_width64_witness_contract import (
        validate_decoder_level3_norm2_width64_witness,
    )

    with pytest.raises(ValueError, match=r"\[N, 64\]"):
        validate_decoder_level3_norm2_width64_witness(
            {
                "level3_upsample_h_c2s": np.zeros(
                    (2, 128),
                    dtype=np.float16,
                ),
                "level3_upsample_norm2_candidate": np.zeros(
                    (2, 128),
                    dtype=np.float16,
                ),
            }
        )


def test_source_width64_contract_requires_official_cuda_route():
    from scripts.decoder_level3_norm2_width64_witness_contract import (
        load_source_width64_contract,
    )

    _, ledger = _full_ledger()
    report = _source_report(ledger)
    report["effective_route"]["device_type"] = "cpu"

    with pytest.raises(ValueError, match="official CUDA"):
        load_source_width64_contract(report)


def test_source_width64_contract_rejects_requested_effective_route_disagreement():
    from scripts.decoder_level3_norm2_width64_witness_contract import (
        load_source_width64_contract,
    )

    _, ledger = _full_ledger()
    report = _source_report(ledger)
    report["requested_route"]["full_decoder_hash_ledger"] = False

    with pytest.raises(ValueError, match="requested/effective"):
        load_source_width64_contract(report)


def test_source_width64_contract_rejects_self_consistent_wrong_boundary_ledger():
    from scripts.decoder_level3_norm2_width64_witness_contract import (
        load_source_width64_contract,
    )

    _, ledger = _full_ledger()
    report = _source_report(ledger)

    with pytest.raises(ValueError, match="canonical width-64 boundary"):
        load_source_width64_contract(report)


def test_width64_comparison_requires_exact_ordered_prefix():
    from scripts.decoder_level3_norm2_width64_witness_contract import (
        compare_decoder_level3_norm2_width64_witness,
    )

    arrays, ledger = _full_ledger()
    input_index = FULL_DECODER_HASH_BOUNDARY_NAMES.index(
        "level3_upsample_h_c2s"
    )
    local_prefix = list(ledger["entries"][: input_index + 1])
    local_prefix[4] = {
        **local_prefix[4],
        "sha256": "f" * 64,
    }
    candidate = decoder_full_hash_entry(
        "level3_upsample_norm2",
        arrays["level3_upsample_norm2"],
    )

    with pytest.raises(
        ValueError,
        match="prefix first diverges at level3_block0_silu",
    ):
        compare_decoder_level3_norm2_width64_witness(
            ledger,
            local_prefix,
            candidate,
        )


def test_width64_comparison_accepts_exact_prefix_and_candidate():
    from scripts.decoder_level3_norm2_width64_witness_contract import (
        compare_decoder_level3_norm2_width64_witness,
    )

    arrays, ledger = _full_ledger()
    input_index = FULL_DECODER_HASH_BOUNDARY_NAMES.index(
        "level3_upsample_h_c2s"
    )
    comparison = compare_decoder_level3_norm2_width64_witness(
        ledger,
        ledger["entries"][: input_index + 1],
        decoder_full_hash_entry(
            "level3_upsample_norm2",
            arrays["level3_upsample_norm2"],
        ),
    )

    assert comparison["prefix_exact"] is True
    assert comparison["input_exact"] is True
    assert comparison["candidate_exact"] is True
    assert comparison["candidate"]["source_sha256"] == (
        comparison["candidate"]["local_sha256"]
    )


def test_width64_comparison_reports_candidate_mismatch_without_false_closure():
    from scripts.decoder_level3_norm2_width64_witness_contract import (
        compare_decoder_level3_norm2_width64_witness,
    )

    arrays, ledger = _full_ledger()
    input_index = FULL_DECODER_HASH_BOUNDARY_NAMES.index(
        "level3_upsample_h_c2s"
    )
    changed = arrays["level3_upsample_norm2"].copy()
    changed[0, 0] += np.float16(1)
    comparison = compare_decoder_level3_norm2_width64_witness(
        ledger,
        ledger["entries"][: input_index + 1],
        decoder_full_hash_entry("level3_upsample_norm2", changed),
    )

    assert comparison["status"] == "done"
    assert comparison["prefix_exact"] is True
    assert comparison["candidate_exact"] is False
    assert comparison["candidate"]["exact"] is False
    assert comparison["candidate"]["source_sha256"] != (
        comparison["candidate"]["local_sha256"]
    )


def test_width64_candidate_runs_twice_and_rejects_nondeterminism():
    import mlx.core as mx

    from trellmlx.decoder_level1_trace import (
        capture_level3_norm2_width64_candidate,
    )

    calls = 0

    def candidate(values):
        nonlocal calls
        calls += 1
        return values + mx.array(calls, dtype=mx.float16)

    with pytest.raises(RuntimeError, match="nondeterministic"):
        capture_level3_norm2_width64_candidate(
            mx.zeros((2, 64), dtype=mx.float16),
            candidate,
        )
    assert calls == 2


def test_width64_production_dispatch_enrolls_only_nonaffine_route():
    import mlx.core as mx

    import trellmlx.decoder_turing_layernorm as decoder_layernorm

    decoder_layernorm.configure_decoder_layernorm_backend(
        decoder_layernorm.CUDA_WELFORD_TURING_T4_BACKEND,
        turing_rsqrt_delta_lut=mx.zeros(
            (1 << 24,),
            dtype=mx.int8,
        ),
        turing_rsqrt_lut_artifact_sha256_attested="a" * 64,
    )
    try:
        actual = decoder_layernorm.layernorm_noaffine(
            mx.zeros((2, 64), dtype=mx.float16),
        )
        mx.eval(actual)
        np.testing.assert_array_equal(
            np.asarray(actual),
            np.zeros((2, 64), dtype=np.float16),
        )
        with pytest.raises(ValueError, match="authenticated only"):
            decoder_layernorm.layernorm_affine(
                mx.zeros((2, 64), dtype=mx.float16),
                mx.ones((64,), dtype=mx.float16),
                mx.zeros((64,), dtype=mx.float16),
            )
    finally:
        decoder_layernorm.configure_decoder_layernorm_backend(
            decoder_layernorm.DEFAULT_BACKEND
        )


def test_width64_witness_mode_requires_authenticated_source_report(tmp_path):
    from scripts.run_mlx_decoder_level1_trace import main

    parent = tmp_path / "level0.npz"
    np.savez(
        parent,
        coords=np.array([[0, 1, 2, 3]], dtype=np.int32),
        block3_output=np.ones((1, 1024), dtype=np.float16),
    )
    checkpoint = tmp_path / "decoder.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    silu_lut = tmp_path / "silu.npz"
    silu_lut.write_bytes(b"silu")
    rsqrt_lut = tmp_path / "rsqrt.npz"
    rsqrt_lut.write_bytes(b"rsqrt")
    output_npz = tmp_path / "witness.npz"
    output_npz.write_bytes(b"stale")
    output_json = tmp_path / "witness.json"

    rc = main(
        [
            "--level0-trace",
            str(parent),
            "--expected-level0-trace-sha256",
            hashlib.sha256(parent.read_bytes()).hexdigest(),
            "--shape-decoder-checkpoint",
            str(checkpoint),
            "--expected-checkpoint-sha256",
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "--decoder-silu-lut",
            str(silu_lut),
            "--expected-decoder-silu-lut-sha256",
            hashlib.sha256(silu_lut.read_bytes()).hexdigest(),
            "--turing-rsqrt-lut",
            str(rsqrt_lut),
            "--expected-turing-rsqrt-lut-sha256",
            hashlib.sha256(rsqrt_lut.read_bytes()).hexdigest(),
            "--expected-repo-commit",
            "b" * 40,
            "--level3-norm2-width64-witness",
            "--output-npz",
            str(output_npz),
            "--output-json",
            str(output_json),
        ]
    )

    report = json.loads(output_json.read_text())
    assert rc == 1
    assert report["failure_phase"] == "request_validation"
    assert "source full-ledger report" in report["error"]
    assert report["stale_primary_invalidated"] is True
    assert not output_npz.exists()


def test_width64_witness_mode_rejects_caller_rebound_source_report(tmp_path):
    from scripts.run_mlx_decoder_level1_trace import main

    _, ledger = _full_ledger()
    source_report = tmp_path / "forged-source.json"
    source_report.write_text(json.dumps(_source_report(ledger)))
    parent = tmp_path / "level0.npz"
    np.savez(
        parent,
        coords=np.array([[0, 1, 2, 3]], dtype=np.int32),
        block3_output=np.ones((1, 1024), dtype=np.float16),
    )
    checkpoint = tmp_path / "decoder.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    silu_lut = tmp_path / "silu.npz"
    silu_lut.write_bytes(b"silu")
    rsqrt_lut = tmp_path / "rsqrt.npz"
    rsqrt_lut.write_bytes(b"rsqrt")
    output_npz = tmp_path / "witness.npz"
    output_npz.write_bytes(b"stale")
    output_json = tmp_path / "witness.json"

    rc = main(
        [
            "--level0-trace",
            str(parent),
            "--expected-level0-trace-sha256",
            hashlib.sha256(parent.read_bytes()).hexdigest(),
            "--shape-decoder-checkpoint",
            str(checkpoint),
            "--expected-checkpoint-sha256",
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "--decoder-silu-lut",
            str(silu_lut),
            "--expected-decoder-silu-lut-sha256",
            hashlib.sha256(silu_lut.read_bytes()).hexdigest(),
            "--turing-rsqrt-lut",
            str(rsqrt_lut),
            "--expected-turing-rsqrt-lut-sha256",
            hashlib.sha256(rsqrt_lut.read_bytes()).hexdigest(),
            "--expected-repo-commit",
            "b" * 40,
            "--level3-norm2-width64-witness",
            "--source-full-ledger-report",
            str(source_report),
            "--expected-source-full-ledger-report-sha256",
            hashlib.sha256(source_report.read_bytes()).hexdigest(),
            "--output-npz",
            str(output_npz),
            "--output-json",
            str(output_json),
        ]
    )

    report = json.loads(output_json.read_text())
    assert rc == 1
    assert report["status"] == "failed"
    assert report["failure_phase"] == "request_validation"
    assert "canonical source report SHA256" in report["error"]
    assert report["stale_primary_invalidated"] is True
    assert not output_npz.exists()
