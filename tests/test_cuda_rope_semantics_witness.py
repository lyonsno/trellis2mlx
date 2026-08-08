import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest


def _write_rope_witness(path, **overrides):
    arrays = {
        "coordinate_values": np.arange(64, dtype=np.int32),
        "frequencies": np.linspace(1.0, 0.01, 21, dtype=np.float32),
        "case_input": np.asarray([[1.0, 2.0]], dtype=np.float32),
        "case_coordinate_index": np.asarray([1], dtype=np.int32),
        "case_frequency_index": np.asarray([2], dtype=np.int32),
        "expected_case_output": np.asarray([[1.0, 2.0]], dtype=np.float32),
    }
    arrays.update(overrides)
    np.savez(path, **arrays)


def test_rope_witness_authenticates_phase_grid_and_boundary_cases():
    from scripts.cuda_rope_semantics_witness import analyze_cuda_results

    phase = np.asarray(
        [
            [[1.0, 0.0], [0.5, 0.25]],
            [[1.0, 0.0], [-0.5, 0.75]],
        ],
        dtype=np.float32,
    )
    expected = np.asarray(
        [[[1.0, 2.0], [3.0, 4.0]], [[-1.0, 0.5], [0.25, -0.75]]],
        dtype=np.float32,
    )

    report = analyze_cuda_results(
        phase_pairs=phase,
        case_output=expected.copy(),
        expected_case_output=expected,
        coordinate_count=2,
        frequency_count=2,
    )

    assert report["phase_pairs"] == {
        "shape": [2, 2, 2],
        "dtype": "float32",
        "finite": True,
    }
    assert report["case_self_authentication"]["nonzero"] == 0


def test_rope_witness_rejects_case_output_that_does_not_replay_source():
    from scripts.cuda_rope_semantics_witness import analyze_cuda_results

    expected = np.zeros((1, 2), dtype=np.float32)
    actual = expected.copy()
    actual[0, 0] = np.nextafter(
        np.float32(0.0), np.float32(np.inf), dtype=np.float32
    )

    with pytest.raises(
        ValueError, match="CUDA RoPE cases do not reproduce source outputs"
    ):
        analyze_cuda_results(
            phase_pairs=np.zeros((2, 2, 2), dtype=np.float32),
            case_output=actual,
            expected_case_output=expected,
            coordinate_count=2,
            frequency_count=2,
        )


@pytest.mark.parametrize("digest", [None, "", "A" * 64, "a" * 63])
def test_rope_witness_requires_canonical_requested_digest(digest):
    from scripts.cuda_rope_semantics_witness import requested_witness_identity

    with pytest.raises(
        ValueError,
        match="expected witness sha256 must be 64 lowercase hexadecimal",
    ):
        requested_witness_identity(digest)


def test_rope_witness_rejects_wrong_runtime_route():
    from scripts.cuda_rope_semantics_witness import validate_runtime

    with pytest.raises(ValueError, match="expected CUDA device Tesla T4"):
        validate_runtime(
            torch_version="2.10.0+cu128",
            cuda_available=True,
            cuda_device="NVIDIA P100",
        )


def test_rope_witness_rejects_non_int32_coordinate_domain(tmp_path):
    from scripts.cuda_rope_semantics_witness import _load_witness

    path = tmp_path / "witness.npz"
    _write_rope_witness(
        path,
        coordinate_values=np.arange(64, dtype=np.float64),
    )

    with pytest.raises(ValueError, match="coordinate_values must be int32"):
        _load_witness(path)


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("case_coordinate_index", np.asarray([1.9], dtype=np.float32)),
        ("case_frequency_index", np.asarray([2.0], dtype=np.float32)),
    ],
)
def test_rope_witness_rejects_noninteger_selector_dtype(
    tmp_path, field, values
):
    from scripts.cuda_rope_semantics_witness import _load_witness

    path = tmp_path / "witness.npz"
    _write_rope_witness(path, **{field: values})

    with pytest.raises(ValueError, match=f"{field} must have integer dtype"):
        _load_witness(path)


def test_rope_witness_rejects_case_input_that_requires_bfloat16_rounding(
    tmp_path,
):
    from scripts.cuda_rope_semantics_witness import _load_witness

    path = tmp_path / "witness.npz"
    _write_rope_witness(
        path,
        case_input=np.asarray([[1.1, 2.0]], dtype=np.float32),
    )

    with pytest.raises(
        ValueError, match="case_input must be exactly BF16-representable"
    ):
        _load_witness(path)


def test_rope_witness_explicit_phase_mode_requires_bound_case_phases(
    tmp_path,
):
    from scripts.cuda_rope_semantics_witness import _load_witness

    path = tmp_path / "witness.npz"
    _write_rope_witness(path)

    with pytest.raises(
        ValueError,
        match="explicit phase mode requires case_phase_pairs",
    ):
        _load_witness(path, phase_mode="explicit")


def test_rope_witness_explicit_phase_mode_validates_case_phase_shape(
    tmp_path,
):
    from scripts.cuda_rope_semantics_witness import _load_witness

    path = tmp_path / "witness.npz"
    _write_rope_witness(
        path,
        case_phase_pairs=np.ones((2, 2), dtype=np.float32),
    )

    with pytest.raises(
        ValueError,
        match="case_phase_pairs must match case_input shape",
    ):
        _load_witness(path, phase_mode="explicit")


def test_rope_witness_accepts_source_runtime_cpu_generated_phases(tmp_path):
    from scripts.cuda_rope_semantics_witness import _load_witness

    path = tmp_path / "witness.npz"
    _write_rope_witness(path)

    arrays = _load_witness(path, phase_mode="cpu-generated")

    assert "case_phase_pairs" not in arrays


def test_rope_witness_result_preserves_precast_cuda_product(tmp_path):
    from scripts.cuda_rope_semantics_witness import _write_result

    output = tmp_path / "result.npz"
    arrays = {
        "coordinate_values": np.arange(64, dtype=np.int32),
        "frequencies": np.linspace(1.0, 0.01, 21, dtype=np.float32),
        "case_coordinate_index": np.asarray([1], dtype=np.int32),
        "case_frequency_index": np.asarray([2], dtype=np.int32),
        "expected_case_output": np.asarray([[1.0, 2.0]], dtype=np.float32),
    }
    precast = np.asarray([[1.001, 2.002]], dtype=np.float32)

    _write_result(
        output,
        phase_pairs=np.zeros((64, 21, 2), dtype=np.float32),
        case_phase_pairs=np.asarray([[0.5, 0.25]], dtype=np.float32),
        case_output=np.asarray([[1.0, 2.0]], dtype=np.float32),
        case_output_float32_precast=precast,
        arrays=arrays,
    )

    with np.load(output, allow_pickle=False) as result:
        assert np.array_equal(result["case_output_float32_precast"], precast)


def test_rope_witness_preserves_diagnostic_output_on_self_auth_failure(
    monkeypatch, tmp_path
):
    from scripts import cuda_rope_semantics_witness as witness

    witness_path = tmp_path / "witness.npz"
    witness_path.write_bytes(b"bound witness")
    digest = hashlib.sha256(witness_path.read_bytes()).hexdigest()
    output_json = tmp_path / "result.json"
    output_npz = tmp_path / "result.npz"
    arrays = {
        "coordinate_values": np.arange(64, dtype=np.int32),
        "frequencies": np.linspace(1.0, 0.01, 21, dtype=np.float32),
        "case_coordinate_index": np.asarray([1], dtype=np.int32),
        "case_frequency_index": np.asarray([2], dtype=np.int32),
        "expected_case_output": np.asarray([[1.0, 2.0]], dtype=np.float32),
    }
    fake_torch = SimpleNamespace(
        __version__="2.10.0+cu128",
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda _index: "Tesla T4",
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setattr(witness, "_load_witness", lambda *_args, **_kwargs: arrays)
    monkeypatch.setattr(
        witness,
        "_run_cuda",
        lambda *_args, **_kwargs: (
            np.zeros((64, 21, 2), dtype=np.float32),
            np.asarray([[0.5, 0.25]], dtype=np.float32),
            np.asarray([[1.001, 2.002]], dtype=np.float32),
            np.asarray([[1.0, 2.0]], dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        witness,
        "analyze_cuda_results",
        lambda **_kwargs: (_ for _ in ()).throw(
            ValueError("CUDA RoPE cases do not reproduce source outputs")
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "cuda_rope_semantics_witness.py",
            "--witness",
            str(witness_path),
            "--expected-witness-sha256",
            digest,
            "--phase-mode",
            "cpu-generated",
            "--output-json",
            str(output_json),
            "--output-npz",
            str(output_npz),
        ],
    )

    assert witness.main() == 1

    report = json.loads(output_json.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "cuda_self_authentication"
    assert report["primary_output"]["exists"] is True
    assert report["primary_output"]["authority"] == "diagnostic-only"
    with np.load(output_npz, allow_pickle=False) as result:
        assert "case_output_float32_precast" in result.files


def test_rope_witness_failure_removes_stale_primary_and_writes_report(
    monkeypatch, tmp_path
):
    from scripts import cuda_rope_semantics_witness as witness

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
            "cuda_rope_semantics_witness.py",
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
    assert report["failure_phase"] == "runtime_validation"
    assert report["last_trustworthy_phase"] == "output_paths_validated"
    assert report["primary_output"]["exists"] is False
    assert not output_npz.exists()
