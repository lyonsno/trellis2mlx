from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import pytest


def _arrays(rows: int = 3):
    values = np.arange(rows * 128, dtype=np.float16).reshape(rows, 128)
    return {
        "level3_child_coords": np.arange(
            rows * 4,
            dtype=np.int32,
        ).reshape(rows, 4),
        "level2_upsample_h_c2s": values,
        "level2_upsample_norm2": values + np.float16(1),
    }


def test_norm2_contract_rejects_partial_extra_wrong_shape_and_nonfinite():
    from scripts.decoder_level2_norm2_trace_contract import (
        TRACE_ARRAY_NAMES,
        validate_decoder_level2_norm2_trace,
    )

    arrays = _arrays()
    assert tuple(validate_decoder_level2_norm2_trace(arrays)) == TRACE_ARRAY_NAMES

    partial = dict(arrays)
    partial.pop("level2_upsample_norm2")
    with pytest.raises(ValueError, match="missing"):
        validate_decoder_level2_norm2_trace(partial)
    with pytest.raises(ValueError, match="extra"):
        validate_decoder_level2_norm2_trace({**arrays, "stale": np.zeros(1)})
    wrong = dict(arrays)
    wrong["level2_upsample_h_c2s"] = np.zeros(
        (3, 64),
        dtype=np.float16,
    )
    with pytest.raises(ValueError, match=r"\[N, 128\]"):
        validate_decoder_level2_norm2_trace(wrong)
    nonfinite = dict(arrays)
    nonfinite["level2_upsample_norm2"] = arrays[
        "level2_upsample_norm2"
    ].copy()
    nonfinite["level2_upsample_norm2"][0, 0] = np.float16(np.nan)
    with pytest.raises(ValueError, match="non-finite"):
        validate_decoder_level2_norm2_trace(nonfinite)


def test_norm2_contract_writes_and_reopens_exact(tmp_path):
    from scripts.decoder_level2_norm2_trace_contract import (
        load_decoder_level2_norm2_trace,
        write_decoder_level2_norm2_trace_npz,
    )

    path = tmp_path / "norm2-trace.npz"
    validation = write_decoder_level2_norm2_trace_npz(path, _arrays())
    reopened = load_decoder_level2_norm2_trace(path)

    assert validation["schema"] == (
        "trellis2mlx.decoder_level2_norm2_trace.v1"
    )
    assert validation["rows"] == 3
    assert validation["channels"] == 128
    assert validation["reopened_exact"] is True
    for name, expected in _arrays().items():
        np.testing.assert_array_equal(reopened[name], expected)


def test_norm2_comparison_records_full_metrics():
    from scripts.run_mlx_decoder_level2_norm2_trace import compare_norm2

    source = np.zeros((2, 128), dtype=np.float16)
    candidate = source.copy()
    candidate[0, 0] = np.float16(0.5)

    result = compare_norm2(source, candidate)

    assert result["exact"] is False
    assert result["nonzero_count"] == 1
    assert result["mean_abs"] == pytest.approx(0.5 / 256)
    assert result["rms"] == pytest.approx(0.5 / 16)
    assert result["max_abs"] == 0.5


def test_norm2_replay_rejects_wrong_source_digest_before_mlx(
    monkeypatch,
    tmp_path,
):
    from scripts import run_mlx_decoder_level2_norm2_trace as replay

    source = tmp_path / "source.npz"
    source.write_bytes(b"wrong-source")
    lut = tmp_path / "lut.npz"
    lut.write_bytes(b"wrong-lut")
    output_json = tmp_path / "report.json"
    output_npz = tmp_path / "candidate.npz"
    output_json.write_text('{"status": "done"}\n')
    output_npz.write_bytes(b"stale")
    monkeypatch.delitem(sys.modules, "mlx", raising=False)
    monkeypatch.delitem(sys.modules, "mlx.core", raising=False)

    assert (
        replay.main(
            [
                "--source-trace",
                str(source),
                "--expected-source-trace-sha256",
                "0" * 64,
                "--turing-rsqrt-lut",
                str(lut),
                "--expected-turing-rsqrt-lut-sha256",
                "1" * 64,
                "--output-json",
                str(output_json),
                "--output-npz",
                str(output_npz),
            ]
        )
        == 1
    )

    report = json.loads(output_json.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "input_validation"
    assert "source trace sha256 mismatch" in report["error"]
    assert report["primary_output"]["exists"] is False
    assert not output_npz.exists()
    assert "mlx.core" not in sys.modules


def test_norm2_replay_does_not_delete_colliding_input(tmp_path):
    from scripts import run_mlx_decoder_level2_norm2_trace as replay

    source = tmp_path / "source.npz"
    source.write_bytes(b"protected-source")
    lut = tmp_path / "lut.npz"
    lut.write_bytes(b"protected-lut")
    report = tmp_path / "report.json"

    assert (
        replay.main(
            [
                "--source-trace",
                str(source),
                "--expected-source-trace-sha256",
                hashlib.sha256(source.read_bytes()).hexdigest(),
                "--turing-rsqrt-lut",
                str(lut),
                "--expected-turing-rsqrt-lut-sha256",
                hashlib.sha256(lut.read_bytes()).hexdigest(),
                "--output-json",
                str(report),
                "--output-npz",
                str(source),
            ]
        )
        == 1
    )

    assert source.read_bytes() == b"protected-source"
    failure = json.loads(report.read_text())
    assert failure["failure_phase"] == "input_validation"
    assert failure["primary_output"]["protected_input_collision"] is True


def test_norm2_replay_dual_collision_preserves_both_inputs(tmp_path):
    from scripts import run_mlx_decoder_level2_norm2_trace as replay

    source = tmp_path / "source.npz"
    source_payload = b"protected-source"
    source.write_bytes(source_payload)
    lut = tmp_path / "lut.npz"
    lut_payload = b"protected-lut"
    lut.write_bytes(lut_payload)
    failure_report = lut.with_name(lut.name + ".failure.json")

    assert (
        replay.main(
            [
                "--source-trace",
                str(source),
                "--expected-source-trace-sha256",
                hashlib.sha256(source_payload).hexdigest(),
                "--turing-rsqrt-lut",
                str(lut),
                "--expected-turing-rsqrt-lut-sha256",
                hashlib.sha256(lut_payload).hexdigest(),
                "--output-json",
                str(lut),
                "--output-npz",
                str(source),
            ]
        )
        == 1
    )

    assert source.read_bytes() == source_payload
    assert lut.read_bytes() == lut_payload
    failure = json.loads(failure_report.read_text())
    assert failure["failure_phase"] == "input_validation"
    assert failure["effective_report_path"] == str(failure_report)
    assert failure["primary_output"]["protected_input_collision"] is True


def test_norm2_replay_report_primary_collision_removes_stale_primary(
    tmp_path,
):
    from scripts import run_mlx_decoder_level2_norm2_trace as replay

    source = tmp_path / "source.npz"
    source_payload = b"protected-source"
    source.write_bytes(source_payload)
    lut = tmp_path / "lut.npz"
    lut_payload = b"protected-lut"
    lut.write_bytes(lut_payload)
    primary = tmp_path / "candidate.npz"
    primary.write_bytes(b"stale-candidate")
    failure_report = primary.with_name(primary.name + ".failure.json")

    assert (
        replay.main(
            [
                "--source-trace",
                str(source),
                "--expected-source-trace-sha256",
                hashlib.sha256(source_payload).hexdigest(),
                "--turing-rsqrt-lut",
                str(lut),
                "--expected-turing-rsqrt-lut-sha256",
                hashlib.sha256(lut_payload).hexdigest(),
                "--output-json",
                str(primary),
                "--output-npz",
                str(primary),
            ]
        )
        == 1
    )

    assert not primary.exists()
    assert source.read_bytes() == source_payload
    assert lut.read_bytes() == lut_payload
    failure = json.loads(failure_report.read_text())
    assert failure["failure_phase"] == "input_validation"
    assert failure["effective_report_path"] == str(failure_report)
    assert failure["stale_primary_invalidated"] is True
    assert failure["primary_output"]["exists"] is False


def test_norm2_replay_publishes_exact_candidate_and_route(
    monkeypatch,
    tmp_path,
):
    from scripts import run_mlx_decoder_level2_norm2_trace as replay
    from scripts.decoder_level2_norm2_trace_contract import (
        write_decoder_level2_norm2_trace_npz,
    )

    arrays = _arrays(rows=2)
    source = tmp_path / "source.npz"
    write_decoder_level2_norm2_trace_npz(source, arrays)
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    lut = tmp_path / "lut.npz"
    lut.write_bytes(b"test-lut")
    lut_digest = hashlib.sha256(lut.read_bytes()).hexdigest()
    output_json = tmp_path / "report.json"
    output_npz = tmp_path / "candidate.npz"

    monkeypatch.setattr(
        replay,
        "_load_turing_rsqrt_lut",
        lambda path, digest: (
            np.zeros(1 << 24, dtype=np.int8),
            {
                "path": str(path),
                "sha256": digest,
                "normalized_delta_sha256": "2" * 64,
                "entries": 1 << 24,
                "dtype": "int8",
            },
        ),
    )
    monkeypatch.setattr(
        replay,
        "_run_mlx_candidate",
        lambda source_arrays, _lut, _lut_digest: (
            source_arrays["level2_upsample_norm2"].copy(),
            {
                "backend": "cuda-welford-turing-t4",
                "candidate_contract": {
                    "hidden_width": 128,
                    "affine": False,
                },
            },
        ),
    )

    assert (
        replay.main(
            [
                "--source-trace",
                str(source),
                "--expected-source-trace-sha256",
                source_digest,
                "--turing-rsqrt-lut",
                str(lut),
                "--expected-turing-rsqrt-lut-sha256",
                lut_digest,
                "--output-json",
                str(output_json),
                "--output-npz",
                str(output_npz),
            ]
        )
        == 0
    )

    report = json.loads(output_json.read_text())
    assert report["status"] == "done"
    assert report["comparison"]["exact"] is True
    assert report["effective_route"]["candidate_contract"] == {
        "hidden_width": 128,
        "affine": False,
    }
    assert report["primary_output"]["exists"] is True
    assert report["primary_output"]["reopened_exact"] is True
    with np.load(output_npz, allow_pickle=False) as archive:
        np.testing.assert_array_equal(
            archive["level2_upsample_norm2_candidate"],
            arrays["level2_upsample_norm2"],
        )
