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
    coefficients = np.asarray([1.0, 0.875, 0.75], dtype=np.float32)
    rounded = np.asarray(1.0 / coefficients, dtype=np.float32)
    native = rounded.copy()
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
            coefficients=coefficients,
            native_reciprocals=native,
            correctly_rounded_reciprocals=rounded,
            pred_recomputed=pred,
            source_pred_final=source_pred,
        )


def test_reciprocal_schedule_records_bits_and_signed_ulp_delta():
    from scripts.cuda_flow_conversion_witness import (
        analyze_reciprocal_schedule,
    )

    step_indices = np.asarray([1, 3], dtype=np.int32)
    coefficients = np.asarray([0.875, 2.0 / 3.0], dtype=np.float32)
    rounded_bits = np.asarray([0x3F924925, 0x3FC00000], dtype=np.uint32)
    native_bits = np.asarray([0x3F924925, 0x3FBFFFFF], dtype=np.uint32)
    pred = np.ones((2, 1, 1), dtype=np.float32)

    report = analyze_reciprocal_schedule(
        step_indices=step_indices,
        coefficients=coefficients,
        native_reciprocals=native_bits.view(np.float32),
        correctly_rounded_reciprocals=rounded_bits.view(np.float32),
        pred_recomputed=pred,
        source_pred_final=pred,
    )

    assert report["all_active_predictions_exact"] is True
    assert report["rows"] == [
        {
            "active_row": 0,
            "step_index": 1,
            "coefficient_bits": "0x3f600000",
            "native_reciprocal_bits": "0x3f924925",
            "correctly_rounded_reciprocal_bits": "0x3f924925",
            "native_minus_correctly_rounded_ulp": 0,
            "prediction_exact": True,
        },
        {
            "active_row": 1,
            "step_index": 3,
            "coefficient_bits": "0x3f2aaaab",
            "native_reciprocal_bits": "0x3fbfffff",
            "correctly_rounded_reciprocal_bits": "0x3fc00000",
            "native_minus_correctly_rounded_ulp": -1,
            "prediction_exact": True,
        },
    ]


def test_normalized_reciprocal_sweep_rejects_partial_domain():
    from scripts.cuda_flow_conversion_witness import (
        analyze_normalized_reciprocal_sweep,
    )

    with pytest.raises(ValueError, match="must cover exactly 8 coordinates"):
        analyze_normalized_reciprocal_sweep(
            reciprocal_ulp_delta=np.zeros((1, 7), dtype=np.int8),
            sweep_exponent_bits=np.asarray([0x3F000000], dtype=np.uint32),
            schedule_coefficients=np.asarray([0.75], dtype=np.float32),
            schedule_native_reciprocals=np.asarray([4.0 / 3.0], dtype=np.float32),
            schedule_correctly_rounded_reciprocals=np.asarray(
                [4.0 / 3.0], dtype=np.float32
            ),
            expected_count=8,
        )


def test_normalized_reciprocal_sweep_cross_authenticates_schedule_coordinates():
    from scripts.cuda_flow_conversion_witness import (
        analyze_normalized_reciprocal_sweep,
    )

    coefficients = np.asarray([0.75, 1.5], dtype=np.float32)
    rounded = np.asarray(1.0 / coefficients, dtype=np.float32)
    native = rounded.copy()
    native_bits = native.view(np.uint32)
    native_bits[0] -= np.uint32(1)
    delta = np.zeros((2, 1 << 23), dtype=np.int8)

    with pytest.raises(
        ValueError, match="sweep does not reproduce schedule native reciprocal"
    ):
        analyze_normalized_reciprocal_sweep(
            reciprocal_ulp_delta=delta,
            sweep_exponent_bits=np.asarray(
                [0x3F000000, 0x3F800000], dtype=np.uint32
            ),
            schedule_coefficients=coefficients,
            schedule_native_reciprocals=native,
            schedule_correctly_rounded_reciprocals=rounded,
        )


def test_normalized_reciprocal_sweep_reports_cross_authenticated_structure():
    from scripts.cuda_flow_conversion_witness import (
        analyze_normalized_reciprocal_sweep,
        reciprocal_coordinates,
    )

    coefficients = np.asarray([0.75, 1.5], dtype=np.float32)
    rounded = np.asarray(1.0 / coefficients, dtype=np.float32)
    native = rounded.copy()
    native.view(np.uint32)[0] -= np.uint32(1)
    delta = np.zeros((2, 1 << 23), dtype=np.int8)
    exponent_bits, mantissa_coordinates = reciprocal_coordinates(coefficients)
    delta[0, mantissa_coordinates[0]] = np.int8(-1)

    report = analyze_normalized_reciprocal_sweep(
        reciprocal_ulp_delta=delta,
        sweep_exponent_bits=np.asarray(
            [0x3F000000, 0x3F800000], dtype=np.uint32
        ),
        schedule_coefficients=coefficients,
        schedule_native_reciprocals=native,
        schedule_correctly_rounded_reciprocals=rounded,
    )

    assert report["domain"] == {
        "input": "positive_normal_float32_values_at_listed_exponents",
        "exponent_bits": ["0x3f000000", "0x3f800000"],
        "mantissa_coordinate_count_per_exponent": 1 << 23,
        "total_coordinate_count": 2 * (1 << 23),
        "complete_per_exponent": True,
    }
    assert report["schedule_cross_authentication"] == {
        "exact": True,
        "coordinate_count": 2,
        "coordinates": [
            {
                "exponent_bits": "0x3f000000",
                "mantissa_coordinate": 4194304,
            },
            {
                "exponent_bits": "0x3f800000",
                "mantissa_coordinate": 4194304,
            },
        ],
    }
    assert exponent_bits.tolist() == [0x3F000000, 0x3F800000]
    assert report["exponents"] == [
        {
            "exponent_bits": "0x3f000000",
            "delta_histogram": {"-1": 1, "0": (1 << 23) - 1},
            "nonzero_count": 1,
            "run_count": 3,
        },
        {
            "exponent_bits": "0x3f800000",
            "delta_histogram": {"0": 1 << 23},
            "nonzero_count": 0,
            "run_count": 1,
        },
    ]


def test_output_contract_rejects_corrupted_postwrite_direct_prediction():
    from scripts.cuda_flow_conversion_witness import analyze_output_contract

    step_indices = np.asarray([1], dtype=np.int32)
    coefficients = np.asarray([0.75], dtype=np.float32)
    rounded = np.asarray(1.0 / coefficients, dtype=np.float32)
    pred = np.ones((1, 1, 1), dtype=np.float32)
    corrupted_direct = pred.copy()
    corrupted_direct[0, 0, 0] = np.nextafter(
        corrupted_direct[0, 0, 0], np.float32(np.inf), dtype=np.float32
    )
    delta = np.zeros((1, 1 << 23), dtype=np.int8)

    with pytest.raises(
        ValueError, match="direct schedule prediction does not reproduce source"
    ):
        analyze_output_contract(
            source_pred_final=pred,
            step_indices=step_indices,
            coefficients=coefficients,
            pred_direct=corrupted_direct,
            pred_recomputed=pred,
            native_reciprocals=rounded,
            correctly_rounded_reciprocals=rounded,
            reciprocal_ulp_delta=delta,
            sweep_exponent_bits=np.asarray([0x3F000000], dtype=np.uint32),
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
