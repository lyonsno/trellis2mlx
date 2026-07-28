import hashlib
import json

import numpy as np
import pytest


def test_flow_conversion_oracle_requires_exact_source_self_authentication():
    from scripts.cuda_flow_conversion_witness import analyze_conversion

    sample = np.asarray([[2.0, -3.0]], dtype=np.float32)
    x0 = np.asarray([[0.5, -0.75]], dtype=np.float32)
    scaled_sample = np.asarray([[1.5, -2.25]], dtype=np.float32)
    numerator = scaled_sample - x0
    pred = np.asarray([[1.0, -1.5]], dtype=np.float32)
    source_pred = pred.copy()
    source_pred[0, 0] = np.nextafter(
        source_pred[0, 0], np.float32(np.inf), dtype=np.float32
    )

    with pytest.raises(
        ValueError, match="recomputed prediction does not reproduce source"
    ):
        analyze_conversion(
            sample=sample,
            x0_after_rescale=x0,
            scaled_sample=scaled_sample,
            numerator=numerator,
            pred_recomputed=pred,
            source_pred_final=source_pred,
        )


def test_flow_conversion_oracle_records_each_eager_intermediate_without_overclaim():
    from scripts.cuda_flow_conversion_witness import analyze_conversion

    sample = np.asarray([[2.0, -3.0]], dtype=np.float32)
    x0 = np.asarray([[0.5, -0.75]], dtype=np.float32)
    scaled_sample = np.asarray([[1.5, -2.25]], dtype=np.float32)
    numerator = np.asarray([[1.0, -1.5]], dtype=np.float32)
    pred = np.asarray([[2.0, -3.0]], dtype=np.float32)

    report = analyze_conversion(
        sample=sample,
        x0_after_rescale=x0,
        scaled_sample=scaled_sample,
        numerator=numerator,
        pred_recomputed=pred,
        source_pred_final=pred,
    )

    assert report["self_authentication"] == {"pred_recomputed_exact": True}
    assert report["intermediate_capture"] == {
        "status": "captured_not_independently_recomputed",
        "arrays": ["scaled_sample", "numerator"],
    }
    assert set(report["array_sha256"]) == {
        "sample",
        "x0_after_rescale",
        "scaled_sample",
        "numerator",
        "pred_recomputed",
        "source_pred_final",
    }


def test_reciprocal_schedule_requires_every_active_row_to_reproduce_source():
    from scripts.cuda_flow_conversion_witness import (
        analyze_reciprocal_schedule,
    )

    step_indices = np.asarray([0, 1, 2], dtype=np.int32)
    coefficient_float64 = np.asarray([1.0, 0.875, 0.75], dtype=np.float64)
    coefficient_float32 = coefficient_float64.astype(np.float32)
    host_reciprocal = (1.0 / coefficient_float64).astype(np.float32)
    native = host_reciprocal.copy()
    pred = np.ones((3, 2, 2), dtype=np.float32)
    source_pred = pred.copy()
    source_pred[2, 1, 1] = np.nextafter(
        source_pred[2, 1, 1], np.float32(np.inf), dtype=np.float32
    )

    with pytest.raises(
        ValueError, match="schedule step 2 does not reproduce source"
    ):
        analyze_reciprocal_schedule(
            step_indices=step_indices,
            coefficient_float64=coefficient_float64,
            coefficient_float32=coefficient_float32,
            native_reciprocals=native,
            host_float64_reciprocals=host_reciprocal,
            pred_recomputed=pred,
            source_pred_final=source_pred,
        )


def test_reciprocal_schedule_records_bits_and_signed_ulp_delta():
    from scripts.cuda_flow_conversion_witness import (
        analyze_reciprocal_schedule,
    )

    step_indices = np.asarray([1, 3], dtype=np.int32)
    coefficient_float64 = np.asarray([0.875, 0.833335], dtype=np.float64)
    coefficient_float32 = coefficient_float64.astype(np.float32)
    host_reciprocal = (1.0 / coefficient_float64).astype(np.float32)
    pred = np.ones((2, 1, 1), dtype=np.float32)

    report = analyze_reciprocal_schedule(
        step_indices=step_indices,
        coefficient_float64=coefficient_float64,
        coefficient_float32=coefficient_float32,
        native_reciprocals=host_reciprocal,
        host_float64_reciprocals=host_reciprocal,
        pred_recomputed=pred,
        source_pred_final=pred,
    )

    assert report["all_active_predictions_exact"] is True
    assert report["all_native_reciprocals_match_host_float64"] is True
    assert report["rows"][0]["coefficient_float32_bits"] == "0x3f600000"
    assert report["rows"][0]["native_reciprocal_bits"] == "0x3f924925"
    assert report["rows"][1]["coefficient_float32_bits"] == "0x3f555571"
    assert report["rows"][1]["native_reciprocal_bits"] == "0x3f999985"
    assert report["rows"][1]["host_float64_reciprocal_bits"] == "0x3f999985"
    assert report["rows"][1]["native_minus_host_float64_ulp"] == 0


def test_reciprocal_schedule_rejects_float32_denominator_reference():
    from scripts.cuda_flow_conversion_witness import (
        analyze_reciprocal_schedule,
    )

    coefficient_float64 = np.asarray([0.833335], dtype=np.float64)
    coefficient_float32 = coefficient_float64.astype(np.float32)
    native = (1.0 / coefficient_float64).astype(np.float32)
    stale_reference = np.divide(
        np.ones_like(coefficient_float32),
        coefficient_float32,
        dtype=np.float32,
    )
    pred = np.ones((1, 1, 1), dtype=np.float32)

    with pytest.raises(
        ValueError, match="host float64 reciprocal reference is stale"
    ):
        analyze_reciprocal_schedule(
            step_indices=np.asarray([3], dtype=np.int32),
            coefficient_float64=coefficient_float64,
            coefficient_float32=coefficient_float32,
            native_reciprocals=native,
            host_float64_reciprocals=stale_reference,
            pred_recomputed=pred,
            source_pred_final=pred,
        )


def test_output_contract_rejects_corrupted_postwrite_direct_prediction():
    from scripts.cuda_flow_conversion_witness import analyze_output_contract

    step_indices = np.asarray([1], dtype=np.int32)
    coefficient_float64 = np.asarray([0.75], dtype=np.float64)
    coefficient_float32 = coefficient_float64.astype(np.float32)
    host_reciprocal = (1.0 / coefficient_float64).astype(np.float32)
    pred = np.ones((1, 1, 1), dtype=np.float32)
    corrupted_direct = pred.copy()
    corrupted_direct[0, 0, 0] = np.nextafter(
        corrupted_direct[0, 0, 0], np.float32(np.inf), dtype=np.float32
    )
    with pytest.raises(
        ValueError, match="direct schedule prediction does not reproduce source"
    ):
        analyze_output_contract(
            source_pred_final=pred,
            step_indices=step_indices,
            coefficient_float64=coefficient_float64,
            coefficient_float32=coefficient_float32,
            pred_direct=corrupted_direct,
            pred_recomputed=pred,
            native_reciprocals=host_reciprocal,
            host_float64_reciprocals=host_reciprocal,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"route_device": "NVIDIA A100-SXM4-40GB"}, "route must be Tesla T4"),
        ({"claimed_tar_sha256": "f" * 64}, "source tar claim mismatch"),
        ({"expected_tar_sha256": "e" * 64}, "source tar sha256 mismatch"),
    ],
)
def test_flow_conversion_rejects_route_or_source_tar_substitution(
    mutation, message, tmp_path
):
    from scripts.cuda_flow_conversion_witness import validate_source_chain

    source_tar = tmp_path / "trellis2-source.tar.gz"
    source_tar.write_bytes(b"official-source")
    expected_digest = hashlib.sha256(source_tar.read_bytes()).hexdigest()
    source_identity = {
        "route": {"cuda_device": "Tesla T4"},
        "source_tar_sha256_claimed": expected_digest,
    }
    source_identity["route"]["cuda_device"] = mutation.get(
        "route_device", source_identity["route"]["cuda_device"]
    )
    source_identity["source_tar_sha256_claimed"] = mutation.get(
        "claimed_tar_sha256",
        source_identity["source_tar_sha256_claimed"],
    )
    expected_digest = mutation.get("expected_tar_sha256", expected_digest)

    with pytest.raises(ValueError, match=message):
        validate_source_chain(
            source_identity=source_identity,
            source_tar_path=source_tar,
            expected_source_tar_sha256=expected_digest,
        )


def test_flow_conversion_failure_removes_stale_primary_and_writes_report(
    monkeypatch, tmp_path
):
    from scripts import cuda_flow_conversion_witness as witness

    source = tmp_path / "source_recurrence.npz"
    source.write_bytes(b"substituted")
    source_tar = tmp_path / "trellis2-source.tar.gz"
    source_tar.write_bytes(b"official-source")
    source_tar_digest = hashlib.sha256(source_tar.read_bytes()).hexdigest()
    output_json = tmp_path / "result.json"
    output_npz = tmp_path / "result.npz"
    output_npz.write_bytes(b"stale")
    monkeypatch.setattr(
        "sys.argv",
        [
            "cuda_flow_conversion_witness.py",
            "--source-recurrence",
            str(source),
            "--expected-source-recurrence-sha256",
            "0" * 64,
            "--source-tar",
            str(source_tar),
            "--expected-source-tar-sha256",
            source_tar_digest,
            "--output-json",
            str(output_json),
            "--output-npz",
            str(output_npz),
        ],
    )

    assert witness.main() == 1

    report = json.loads(output_json.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "input_validation"
    assert report["last_trustworthy_phase"] == "request_validated"
    assert "source recurrence sha256 mismatch" in report["error"]
    assert report["primary_output"]["exists"] is False
    assert not output_npz.exists()
