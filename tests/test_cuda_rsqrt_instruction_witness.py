import json
from types import SimpleNamespace

import numpy as np
import pytest


def test_rsqrt_probe_authenticates_native_path_and_censuses_variants():
    from scripts.cuda_rsqrt_instruction_witness import analyze_variants

    correctly_rounded = np.asarray([1.0, 0.5, 0.25], dtype=np.float32)
    native = correctly_rounded.copy()
    native[0] = np.nextafter(
        native[0], np.float32(np.inf), dtype=np.float32
    )
    native[1] = np.nextafter(
        native[1], np.float32(-np.inf), dtype=np.float32
    )
    variants = {
        "rsqrtf": native.copy(),
        "inline_ptx_rsqrt_approx": native.copy(),
        "frsqrt_rn": correctly_rounded.copy(),
        "one_over_sqrtf": correctly_rounded.copy(),
    }

    report = analyze_variants(
        native_rstd=native,
        correctly_rounded_rstd=correctly_rounded,
        variants=variants,
    )

    assert report["self_authentication"] == {
        "rsqrtf_exact_native": True,
        "inline_ptx_exact_native": True,
    }
    assert report["variants"]["rsqrtf"]["vs_native"]["nonzero"] == 0
    assert report["variants"]["frsqrt_rn"]["vs_correctly_rounded"][
        "nonzero"
    ] == 0
    assert report["native_vs_correctly_rounded"]["ulp_histogram"] == {
        "-1": 1,
        "0": 1,
        "1": 1,
    }


def test_rsqrt_probe_rejects_a_non_authenticating_ordinary_rsqrtf():
    from scripts.cuda_rsqrt_instruction_witness import analyze_variants

    native = np.asarray([1.0], dtype=np.float32)
    with pytest.raises(
        ValueError, match="rsqrtf does not reproduce authenticated native"
    ):
        analyze_variants(
            native_rstd=native,
            correctly_rounded_rstd=native,
            variants={
                "rsqrtf": np.nextafter(
                    native, np.float32(np.inf), dtype=np.float32
                ),
                "inline_ptx_rsqrt_approx": native,
            },
        )


def test_rsqrt_probe_classifies_ptx_and_turing_sass_evidence():
    from scripts.cuda_rsqrt_instruction_witness import (
        classify_compiler_evidence,
    )

    evidence = classify_compiler_evidence(
        ptx="""
        .visible .entry rsqrt_variants() {
          rsqrt.approx.f32 %f3, %f1;
          rsqrt.approx.ftz.f32 %f2, %f1;
        }
        """,
        sass="""
        Function : rsqrt_variants
          /*0010*/ MUFU.RSQ R2, R4;
        """,
    )

    assert evidence == {
        "ptx_rsqrt_approx_count": 1,
        "ptx_rsqrt_approx_ftz_count": 1,
        "sass_mufu_rsq_count": 1,
    }


def test_rsqrt_probe_rejects_ftz_only_ptx_as_plain_rsqrt_evidence():
    from scripts.cuda_rsqrt_instruction_witness import (
        classify_compiler_evidence,
        validate_compiler_evidence,
    )

    evidence = {
        "classification": classify_compiler_evidence(
            ptx="rsqrt.approx.ftz.f32 %f2, %f1;",
            sass="MUFU.RSQ R2, R4;",
        )
    }

    with pytest.raises(ValueError, match="PTX omits rsqrt.approx.f32"):
        validate_compiler_evidence(evidence)


@pytest.mark.parametrize("expected", [None, "", "A" * 64, "a" * 63])
def test_rsqrt_probe_requires_canonical_requested_witness_digest(expected):
    from scripts.cuda_rsqrt_instruction_witness import (
        requested_witness_identity,
    )

    with pytest.raises(
        ValueError,
        match="expected witness sha256 must be 64 lowercase hexadecimal",
    ):
        requested_witness_identity(expected)


@pytest.mark.parametrize(
    ("classification", "message"),
    [
        (
            {
                "ptx_rsqrt_approx_count": 0,
                "ptx_rsqrt_approx_ftz_count": 0,
                "sass_mufu_rsq_count": 1,
            },
            "PTX omits rsqrt.approx",
        ),
        (
            {
                "ptx_rsqrt_approx_count": 1,
                "ptx_rsqrt_approx_ftz_count": 1,
                "sass_mufu_rsq_count": 0,
            },
            "SASS omits MUFU.RSQ",
        ),
    ],
)
def test_rsqrt_probe_rejects_missing_instruction_evidence(
    classification, message
):
    from scripts.cuda_rsqrt_instruction_witness import (
        validate_compiler_evidence,
    )

    with pytest.raises(ValueError, match=message):
        validate_compiler_evidence({"classification": classification})


def test_rsqrt_normalized_coordinate_covers_one_to_four_domain():
    from scripts.cuda_rsqrt_instruction_witness import (
        normalized_rsqrt_coordinate,
    )

    values = np.asarray(
        [
            1.0,
            np.nextafter(
                np.float32(1.0),
                np.float32(np.inf),
                dtype=np.float32,
            ),
            np.nextafter(
                np.float32(2.0),
                np.float32(-np.inf),
                dtype=np.float32,
            ),
            2.0,
            np.nextafter(
                np.float32(4.0),
                np.float32(-np.inf),
                dtype=np.float32,
            ),
            0.25,
            8.0,
        ],
        dtype=np.float32,
    )

    assert normalized_rsqrt_coordinate(values).tolist() == [
        0,
        1,
        (1 << 23) - 1,
        1 << 23,
        (1 << 24) - 1,
        0,
        1 << 23,
    ]


def test_rsqrt_normalized_sweep_authenticates_witness_coordinates():
    from scripts.cuda_rsqrt_instruction_witness import analyze_normalized_sweep

    sweep = np.asarray([0, -1, 0, 1, 1, 0], dtype=np.int8)
    coordinates = np.asarray([1, 3, 4], dtype=np.uint32)
    witness_delta = np.asarray([-1, 1, 1], dtype=np.int64)

    report = analyze_normalized_sweep(
        normalized_delta=sweep,
        witness_coordinates=coordinates,
        witness_delta=witness_delta,
        expected_count=6,
    )

    assert report == {
        "count": 6,
        "dtype": "int8",
        "minimum_delta": -1,
        "maximum_delta": 1,
        "histogram": {"-1": 1, "0": 3, "1": 2},
        "run_count": 5,
        "witness_exact": True,
    }


def test_rsqrt_normalized_sweep_rejects_witness_disagreement():
    from scripts.cuda_rsqrt_instruction_witness import analyze_normalized_sweep

    with pytest.raises(
        ValueError, match="normalized sweep does not reproduce witness deltas"
    ):
        analyze_normalized_sweep(
            normalized_delta=np.asarray([0, 0], dtype=np.int8),
            witness_coordinates=np.asarray([1], dtype=np.uint32),
            witness_delta=np.asarray([1], dtype=np.int64),
            expected_count=2,
        )


def test_rsqrt_probe_failure_removes_stale_primary_and_writes_report(
    monkeypatch, tmp_path
):
    from scripts import cuda_rsqrt_instruction_witness as witness

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
            "cuda_rsqrt_instruction_witness.py",
            "--witness",
            str(tmp_path / "missing.npz"),
            "--output-json",
            str(output_json),
            "--output-npz",
            str(output_npz),
            "--output-ptx",
            str(tmp_path / "result.ptx"),
            "--output-sass",
            str(tmp_path / "result.sass"),
        ],
    )

    assert witness.main() == 1

    report = json.loads(output_json.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "request_validation"
    assert report["last_trustworthy_phase"] == "output_paths_validated"
    assert report["primary_output"]["exists"] is False
    assert not output_npz.exists()
