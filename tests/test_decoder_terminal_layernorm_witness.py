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
        dtype = (
            np.int32
            if name == "level4_child_coords"
            else (
                np.float32
                if name in {"decoder_final_layernorm", "decoder_output"}
                else np.float16
            )
        )
        arrays[name] = np.full(shapes[name], index, dtype=dtype)
    return arrays, build_decoder_full_hash_ledger(arrays)


def _source_report(ledger):
    route = {
        "route": "official-source-cuda-shape-decoder-full-hash-ledger",
        "full_decoder_hash_ledger": True,
        "decoder_level1_trace": True,
        "one_model_load": True,
        "decoder_output_head_backend": "torch-sparse-linear-fp32",
    }
    return {
        "schema": "trellis2mlx.source_cuda_shape_slat_grid_decode.v1",
        "status": "done",
        "requested_route": route,
        "effective_route": {
            **route,
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


def test_terminal_witness_round_trips_mixed_dtype_full_values(tmp_path):
    from scripts.decoder_terminal_layernorm_witness_contract import (
        load_decoder_terminal_layernorm_witness,
        write_decoder_terminal_layernorm_witness_npz,
    )

    parent = np.arange(5 * 64, dtype=np.float16).reshape(5, 64)
    candidate = parent.astype(np.float32) / np.float32(7)
    output = tmp_path / "terminal.npz"

    report = write_decoder_terminal_layernorm_witness_npz(
        output,
        {
            "level3_upsample_output": parent,
            "decoder_final_layernorm_candidate": candidate,
        },
    )
    reopened = load_decoder_terminal_layernorm_witness(output)

    assert report == {
        "schema": "trellis2mlx.decoder_terminal_layernorm_witness.v1",
        "rows": 5,
        "channels": 64,
        "array_names": [
            "level3_upsample_output",
            "decoder_final_layernorm_candidate",
        ],
        "reopened_exact": True,
    }
    np.testing.assert_array_equal(reopened["level3_upsample_output"], parent)
    np.testing.assert_array_equal(
        reopened["decoder_final_layernorm_candidate"],
        candidate,
    )


def test_terminal_witness_rejects_candidate_dtype_substitution():
    from scripts.decoder_terminal_layernorm_witness_contract import (
        validate_decoder_terminal_layernorm_witness,
    )

    with pytest.raises(ValueError, match="float32"):
        validate_decoder_terminal_layernorm_witness(
            {
                "level3_upsample_output": np.zeros((2, 64), dtype=np.float16),
                "decoder_final_layernorm_candidate": np.zeros(
                    (2, 64),
                    dtype=np.float16,
                ),
            }
        )


def test_source_terminal_contract_rejects_self_consistent_wrong_ledger():
    from scripts.decoder_terminal_layernorm_witness_contract import (
        load_source_terminal_layernorm_contract,
    )

    _, ledger = _full_ledger()
    with pytest.raises(ValueError, match="canonical terminal boundary"):
        load_source_terminal_layernorm_contract(_source_report(ledger))


def test_terminal_comparison_requires_exact_prefix_through_parent():
    from scripts.decoder_terminal_layernorm_witness_contract import (
        compare_decoder_terminal_layernorm_witness,
    )

    arrays, ledger = _full_ledger()
    parent_index = FULL_DECODER_HASH_BOUNDARY_NAMES.index(
        "level3_upsample_output"
    )
    prefix = list(ledger["entries"][: parent_index + 1])
    prefix[-1] = {**prefix[-1], "sha256": "f" * 64}

    with pytest.raises(
        ValueError,
        match="prefix first diverges at level3_upsample_output",
    ):
        compare_decoder_terminal_layernorm_witness(
            ledger,
            prefix,
            decoder_full_hash_entry(
                "decoder_final_layernorm",
                arrays["decoder_final_layernorm"],
            ),
        )


def test_terminal_comparison_reports_nonexact_candidate_without_closure():
    from scripts.decoder_terminal_layernorm_witness_contract import (
        compare_decoder_terminal_layernorm_witness,
    )

    arrays, ledger = _full_ledger()
    parent_index = FULL_DECODER_HASH_BOUNDARY_NAMES.index(
        "level3_upsample_output"
    )
    changed = arrays["decoder_final_layernorm"].copy()
    changed[0, 0] += np.float32(1)
    comparison = compare_decoder_terminal_layernorm_witness(
        ledger,
        ledger["entries"][: parent_index + 1],
        decoder_full_hash_entry("decoder_final_layernorm", changed),
    )

    assert comparison["status"] == "done"
    assert comparison["prefix_exact"] is True
    assert comparison["parent_exact"] is True
    assert comparison["candidate_exact"] is False
    assert comparison["candidate"]["exact"] is False


def test_terminal_candidate_runs_twice_and_rejects_nondeterminism():
    import mlx.core as mx

    from trellmlx.decoder_level1_trace import (
        capture_decoder_terminal_layernorm_candidate,
    )

    calls = 0

    def candidate(values):
        nonlocal calls
        calls += 1
        return values.astype(mx.float32) + mx.array(calls, dtype=mx.float32)

    with pytest.raises(RuntimeError, match="nondeterministic"):
        capture_decoder_terminal_layernorm_candidate(
            mx.zeros((2, 64), dtype=mx.float16),
            candidate,
        )
    assert calls == 2


def test_turing_terminal_float32_candidate_enforces_exact_contract():
    import mlx.core as mx

    from trellmlx.decoder_turing_layernorm import (
        turing_layernorm_noaffine_fp32_width64,
    )

    correction = mx.zeros((1 << 24,), dtype=mx.int8)
    with pytest.raises(ValueError, match="float32"):
        turing_layernorm_noaffine_fp32_width64(
            mx.zeros((2, 64), dtype=mx.float16),
            correction,
            eps=1e-5,
        )
    with pytest.raises(ValueError, match="width 64"):
        turing_layernorm_noaffine_fp32_width64(
            mx.zeros((2, 128), dtype=mx.float32),
            correction,
            eps=1e-5,
        )

    actual = turing_layernorm_noaffine_fp32_width64(
        mx.zeros((2, 64), dtype=mx.float32),
        correction,
        eps=1e-5,
    )
    mx.eval(actual)
    assert actual.dtype == mx.float32
    np.testing.assert_array_equal(
        np.asarray(actual),
        np.zeros((2, 64), dtype=np.float32),
    )


def test_terminal_witness_mode_requires_authenticated_source_report(tmp_path):
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
            "--terminal-layernorm-witness",
            "--output-npz",
            str(output_npz),
            "--output-json",
            str(output_json),
        ]
    )

    report = json.loads(output_json.read_text())
    assert rc == 1
    assert report["failure_phase"] == "request_validation"
    assert (
        report["requested_route"]["candidate_layernorm_backend"]
        == "direct-cuda-welford-turing-t4-float32-unenrolled"
    )
    assert "source full-ledger report" in report["error"]
    assert report["stale_primary_invalidated"] is True
    assert not output_npz.exists()
