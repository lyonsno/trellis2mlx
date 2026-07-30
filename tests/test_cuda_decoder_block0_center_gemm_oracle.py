import hashlib
import json
import sys

import numpy as np
import pytest


def _witness():
    return {
        "torso_input": np.ones((2, 4), dtype=np.float16),
        "center_weight": np.eye(4, dtype=np.float16),
        "bias": np.asarray([0.0, 0.5, -0.5, 1.0], dtype=np.float16),
        "source_trace_row": np.asarray(
            [1.0, 1.5, 0.5, 2.0],
            dtype=np.float16,
        ),
        "row_index": 1,
    }


def test_center_gemm_oracle_authenticates_selected_source_row():
    from scripts.cuda_decoder_block0_center_gemm_oracle import (
        analyze_full_product,
    )

    witness = _witness()
    product = np.ones((2, 4), dtype=np.float16)

    analysis = analyze_full_product(product=product, witness=witness)

    assert analysis["selected_plus_bias_exact_source"] is True
    assert analysis["matrix_shape"] == [2, 4]
    assert analysis["matrix_dtype"] == "float16"

    product[1, 0] = np.float16(2.0)
    with pytest.raises(ValueError, match="does not reproduce source trace"):
        analyze_full_product(product=product, witness=witness)


def test_center_gemm_oracle_rejects_dtype_substitution():
    from scripts.cuda_decoder_block0_center_gemm_oracle import (
        analyze_full_product,
    )

    with pytest.raises(ValueError, match="dtype float16"):
        analyze_full_product(
            product=np.ones((2, 4), dtype=np.float32),
            witness=_witness(),
        )


def test_center_gemm_oracle_rejects_partial_or_nonfinite_matrix():
    from scripts.cuda_decoder_block0_center_gemm_oracle import (
        analyze_full_product,
    )

    witness = _witness()
    with pytest.raises(ValueError, match="shape"):
        analyze_full_product(
            product=np.ones((1, 4), dtype=np.float16),
            witness=witness,
        )

    product = np.ones((2, 4), dtype=np.float16)
    product[0, 0] = np.float16(np.nan)
    with pytest.raises(ValueError, match="non-finite"):
        analyze_full_product(product=product, witness=witness)


def test_center_gemm_oracle_rejects_wrong_digest_before_torch_and_clears_primary(
    monkeypatch,
    tmp_path,
):
    from scripts import cuda_decoder_block0_center_gemm_oracle as oracle

    input_path = tmp_path / "witness.npz"
    input_path.write_bytes(b"not-the-requested-witness")
    output_json = tmp_path / "result.json"
    output_npz = tmp_path / "result.npz"
    output_npz.write_bytes(b"stale")
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    assert (
        oracle.main(
            [
                "--witness",
                str(input_path),
                "--expected-witness-sha256",
                "0" * 64,
                "--output-json",
                str(output_json),
                "--output-npz",
                str(output_npz),
                "--expected-rows",
                "2",
                "--channels",
                "4",
                "--expected-row",
                "1",
            ]
        )
        == 1
    )

    report = json.loads(output_json.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "input_validation"
    assert report["primary_output"]["exists"] is False
    assert "witness sha256 mismatch" in report["error"]
    assert "torch" not in sys.modules
    assert not output_npz.exists()
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() != "0" * 64


def test_center_gemm_oracle_clears_stale_report_before_hard_interruption(
    monkeypatch,
    tmp_path,
):
    from scripts import cuda_decoder_block0_center_gemm_oracle as oracle

    input_path = tmp_path / "witness.npz"
    input_path.write_bytes(b"input")
    output_json = tmp_path / "result.json"
    output_json.write_text('{"status": "done"}\n')
    output_npz = tmp_path / "result.npz"
    output_npz.write_bytes(b"stale")

    def interrupt(_path):
        raise KeyboardInterrupt

    monkeypatch.setattr(oracle, "sha256_file", interrupt)

    with pytest.raises(KeyboardInterrupt):
        oracle.main(
            [
                "--witness",
                str(input_path),
                "--expected-witness-sha256",
                "0" * 64,
                "--output-json",
                str(output_json),
                "--output-npz",
                str(output_npz),
                "--expected-rows",
                "2",
                "--channels",
                "4",
                "--expected-row",
                "1",
            ]
        )

    assert not output_json.exists()
    assert not output_npz.exists()
