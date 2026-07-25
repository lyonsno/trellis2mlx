import json
from types import SimpleNamespace

import numpy as np
import pytest


def test_variance_oracle_requires_exact_native_self_authentication():
    from scripts.cuda_welford_variance_witness import analyze_oracle

    native_out = np.asarray([[1.0, -1.0]], dtype=np.float32)
    native_mean = np.asarray([[0.25]], dtype=np.float32)
    native_rstd = np.asarray([[1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="custom mean does not reproduce native"):
        analyze_oracle(
            custom_out=native_out,
            custom_mean=np.nextafter(
                native_mean, np.float32(np.inf), dtype=np.float32
            ),
            custom_variance=np.asarray([[1.0]], dtype=np.float32),
            custom_rstd=native_rstd,
            native_out=native_out,
            native_mean=native_mean,
            native_rstd=native_rstd,
            eps=np.float32(0.0),
        )


def test_variance_oracle_censuses_rsqrt_ulp_residual_after_authentication():
    from scripts.cuda_welford_variance_witness import analyze_oracle

    variance = np.asarray([[1.0], [4.0], [16.0]], dtype=np.float32)
    correctly_rounded = np.asarray([[1.0], [0.5], [0.25]], dtype=np.float32)
    native_rstd = correctly_rounded.copy()
    native_rstd[0, 0] = np.nextafter(
        native_rstd[0, 0], np.float32(np.inf), dtype=np.float32
    )
    native_rstd[1, 0] = np.nextafter(
        native_rstd[1, 0], np.float32(-np.inf), dtype=np.float32
    )
    native_out = np.zeros((3, 2), dtype=np.float32)
    native_mean = np.zeros((3, 1), dtype=np.float32)

    report = analyze_oracle(
        custom_out=native_out,
        custom_mean=native_mean,
        custom_variance=variance,
        custom_rstd=native_rstd,
        native_out=native_out,
        native_mean=native_mean,
        native_rstd=native_rstd,
        eps=np.float32(0.0),
    )

    assert report["self_authentication"] == {
        "output_exact": True,
        "mean_exact": True,
        "rstd_exact": True,
    }
    assert report["native_rstd_vs_correctly_rounded"]["nonzero"] == 2
    assert report["native_rstd_vs_correctly_rounded"]["ulp_histogram"] == {
        "-1": 1,
        "0": 1,
        "1": 1,
    }


def test_witness_failure_removes_stale_primary_and_writes_durable_report(
    monkeypatch, tmp_path
):
    from scripts import cuda_welford_variance_witness as witness

    output_json = tmp_path / "result.json"
    output_npz = tmp_path / "result.npz"
    output_npz.write_bytes(b"stale")
    fake_torch = SimpleNamespace(
        __version__="unexpected",
        cuda=SimpleNamespace(
            is_available=lambda: False,
            get_device_name=lambda _index: None,
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setattr(
        "sys.argv",
        [
            "cuda_welford_variance_witness.py",
            "--witness",
            str(tmp_path / "missing.npz"),
            "--output-json",
            str(output_json),
            "--output-npz",
            str(output_npz),
        ],
    )

    assert witness.main() == 1

    report = json.loads(output_json.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "request_validation"
    assert report["last_trustworthy_phase"] == "output_path_validated"
    assert report["primary_output"]["exists"] is False
    assert not output_npz.exists()


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("sha256", "1" * 64, "witness sha256 mismatch"),
        ("input_shape", [1, 8, 1536], "witness input shape mismatch"),
        ("eps_float32_bits", "0x00000000", "witness epsilon mismatch"),
    ],
)
def test_witness_identity_rejects_substituted_digest_shape_or_epsilon(
    field, replacement, message
):
    from scripts.cuda_welford_variance_witness import (
        validate_witness_identity,
    )

    requested = {
        "sha256": "0" * 64,
        "input_shape": [1, 4096, 1536],
        "reference_shape": [1, 4096, 1536],
        "eps_float32_bits": "0x358637bd",
    }
    effective = dict(requested)
    effective[field] = replacement

    with pytest.raises(ValueError, match=message):
        validate_witness_identity(requested, effective)
