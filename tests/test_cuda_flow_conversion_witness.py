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
