from pathlib import Path
import argparse
import json
import sys
import types

import numpy as np
import pytest


def test_sdpa_analysis_requires_default_source_replay():
    from scripts.cuda_sdpa_semantics_witness import analyze_sdpa_results

    expected = np.zeros((2, 2), dtype=np.float32)
    default = expected.copy()
    default[0, 0] = 1.0

    with pytest.raises(ValueError, match="default CUDA SDPA does not replay source"):
        analyze_sdpa_results(
            default_output=default,
            expected_output=expected,
            candidate_outputs={"math": expected.copy()},
        )


def test_sdpa_analysis_identifies_exact_forced_backend():
    from scripts.cuda_sdpa_semantics_witness import analyze_sdpa_results

    expected = np.asarray([[1.0, 2.0]], dtype=np.float32)
    math = expected.copy()
    math[0, 1] = 2.5

    report = analyze_sdpa_results(
        default_output=expected.copy(),
        expected_output=expected,
        candidate_outputs={
            "flash": expected.copy(),
            "math": math,
        },
    )

    assert report["default_self_authentication"]["nonzero"] == 0
    assert report["exact_default_matches"] == ["flash"]
    assert report["candidates"]["math"]["nonzero"] == 1


def test_sdpa_witness_rejects_non_bfloat16_representable_qkv(tmp_path):
    from scripts.cuda_sdpa_semantics_witness import _load_witness

    path = tmp_path / "witness.npz"
    shape = (4096, 12, 128)
    q = np.zeros(shape, dtype=np.float32)
    q[0, 0, 0] = 1.1
    np.savez(
        path,
        q=q,
        k=np.zeros(shape, dtype=np.float32),
        v=np.zeros(shape, dtype=np.float32),
        expected_output=np.zeros(shape, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="exactly BF16-representable"):
        _load_witness(path)


def test_sdpa_witness_rejects_wrong_runtime():
    from scripts.cuda_sdpa_semantics_witness import validate_runtime

    with pytest.raises(ValueError, match="expected CUDA device Tesla T4"):
        validate_runtime(
            torch_version="2.10.0+cu128",
            cuda_available=True,
            cuda_device="Tesla P100",
        )


def test_sdpa_witness_requires_canonical_requested_digest():
    from scripts.cuda_sdpa_semantics_witness import requested_witness_identity

    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        requested_witness_identity("ABC")


def test_sdpa_witness_rejects_missing_arrays(tmp_path):
    from scripts.cuda_sdpa_semantics_witness import _load_witness

    path = tmp_path / "witness.npz"
    np.savez(path, q=np.zeros((1,), dtype=np.float32))

    with pytest.raises(ValueError, match="missing arrays"):
        _load_witness(path)


def test_sdpa_witness_preserves_diagnostic_output_on_self_auth_failure(
    tmp_path, monkeypatch
):
    import scripts.cuda_sdpa_semantics_witness as witness

    monkeypatch.setattr(witness, "EXPECTED_SHAPE", (1, 1, 1))
    input_path = tmp_path / "witness.npz"
    arrays = {
        name: np.zeros((1, 1, 1), dtype=np.float32)
        for name in ("q", "k", "v", "expected_output")
    }
    np.savez(input_path, **arrays)
    output_json = tmp_path / "result.json"
    output_npz = tmp_path / "result.npz"
    args = argparse.Namespace(
        witness=input_path,
        expected_witness_sha256=witness.sha256_file(input_path),
        output_json=output_json,
        output_npz=output_npz,
    )
    fake_torch = types.SimpleNamespace(
        __version__="2.10.0+cu128",
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda _index: "Tesla T4",
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(witness, "_parse_args", lambda _argv=None: args)
    monkeypatch.setattr(
        witness,
        "_run_cuda",
        lambda _torch, _arrays: (
            {
                "default": np.ones((1, 1, 1), dtype=np.float32),
                "flash_attention": np.zeros((1, 1, 1), dtype=np.float32),
            },
            {
                "default": {"status": "done"},
                "flash_attention": {"status": "done"},
            },
        ),
    )

    assert witness.main([]) == 1
    report = json.loads(output_json.read_text())
    assert output_npz.is_file()
    assert report["failure_phase"] == "cuda_self_authentication"
    assert report["primary_output"]["authority"] == "diagnostic-only"
