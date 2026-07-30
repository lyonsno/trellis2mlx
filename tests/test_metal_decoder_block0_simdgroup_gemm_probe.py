from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "metal_decoder_block0_simdgroup_gemm_probe.py"


def _load_module():
    assert SCRIPT.is_file(), "direct Metal simdgroup GEMM witness is missing"
    spec = importlib.util.spec_from_file_location(
        "metal_decoder_block0_simdgroup_gemm_probe",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_inputs(tmp_path: Path):
    module = _load_module()
    witness_path = tmp_path / "witness.npz"
    cuda_path = tmp_path / "cuda.npz"
    weight = np.zeros((1024, 1024), dtype=np.float16)
    window = np.zeros((16, 1024), dtype=np.float16)
    tensor_fp32 = np.zeros((1024,), dtype=np.float32)
    regular_fp32 = np.ones((1024,), dtype=np.float32)
    tensor_fp16 = tensor_fp32.astype(np.float16)
    regular_fp16 = regular_fp32.astype(np.float16)
    np.savez(witness_path, center_weight=weight)
    np.savez(
        cuda_path,
        wmma_input_window=window,
        cublas_tensor_fp16_unbiased_row=tensor_fp16,
        cublas_regular_fp16_unbiased_row=regular_fp16,
        cublas_tensor_fp32_row=tensor_fp32,
        cublas_regular_fp32_row=regular_fp32,
        wmma_fp32_row=tensor_fp32,
    )
    return (
        module,
        witness_path,
        cuda_path,
        module.sha256_file(witness_path),
        module.sha256_file(cuda_path),
    )


def _route(module):
    return {
        "backend": "metal",
        "device": module.EXPECTED_DEVICE,
        "kernel": module.KERNEL_IDENTITY,
        "metal_source_sha256": module.METAL_SOURCE_SHA256,
    }


def test_probe_source_uses_direct_simdgroup_matrix_primitive():
    module = _load_module()
    assert "simdgroup_matrix<half, 8, 8>" in module.METAL_SOURCE
    assert "simdgroup_matrix<float, 8, 8>" in module.METAL_SOURCE
    assert "simdgroup_multiply_accumulate" in module.METAL_SOURCE
    assert "for (uint k = 0; k < 1024; k += 8)" in module.METAL_SOURCE


def test_probe_uses_current_mlx_device_identity_api():
    source = SCRIPT.read_text()
    assert "mx.device_info()" in source
    assert "mx.metal.device_info()" not in source


def test_probe_rejects_wrong_digest_before_backend(tmp_path):
    module, witness_path, cuda_path, _, cuda_sha = _write_inputs(tmp_path)
    called = False

    def backend(*_args):
        nonlocal called
        called = True
        raise AssertionError("backend must not run")

    with pytest.raises(ValueError, match="witness sha256"):
        module.run_probe(
            witness_path=witness_path,
            cuda_result_path=cuda_path,
            expected_witness_sha256="0" * 64,
            expected_cuda_result_sha256=cuda_sha,
            output_json=tmp_path / "result.json",
            output_npz=tmp_path / "result.npz",
            backend=backend,
        )
    assert not called
    assert not (tmp_path / "result.npz").exists()


def test_probe_rejects_json_npz_path_collision_before_backend(tmp_path):
    module, witness_path, cuda_path, witness_sha, cuda_sha = _write_inputs(
        tmp_path
    )
    called = False

    def backend(*_args):
        nonlocal called
        called = True
        raise AssertionError("backend must not run")

    output = tmp_path / "same-output"
    with pytest.raises(ValueError, match="distinct paths"):
        module.run_probe(
            witness_path=witness_path,
            cuda_result_path=cuda_path,
            expected_witness_sha256=witness_sha,
            expected_cuda_result_sha256=cuda_sha,
            output_json=output,
            output_npz=output,
            backend=backend,
        )
    assert not called
    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("device", "not-an-m4-max"),
        ("kernel", "wrong-kernel"),
        ("metal_source_sha256", "0" * 64),
    ],
)
def test_probe_rejects_substituted_metal_route(field, replacement):
    module = _load_module()
    route = _route(module)
    route[field] = replacement
    with pytest.raises(ValueError, match=field):
        module._validate_effective_route(route)


@pytest.mark.parametrize(
    ("archive_name", "key", "replacement", "match"),
    [
        (
            "witness",
            "center_weight",
            np.zeros((1024, 1024), dtype=np.float32),
            "center_weight must have dtype float16",
        ),
        (
            "cuda",
            "wmma_input_window",
            np.zeros((15, 1024), dtype=np.float16),
            r"wmma_input_window must have shape \(16, 1024\)",
        ),
        (
            "cuda",
            "cublas_tensor_fp32_row",
            np.full((1024,), np.nan, dtype=np.float32),
            "cublas_tensor_fp32_row contains non-finite",
        ),
    ],
)
def test_probe_rejects_malformed_exact_inputs(
    tmp_path,
    archive_name,
    key,
    replacement,
    match,
):
    module, witness_path, cuda_path, _, _ = _write_inputs(tmp_path)
    target = witness_path if archive_name == "witness" else cuda_path
    with np.load(target, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays[key] = replacement
    np.savez(target, **arrays)

    with pytest.raises(ValueError, match=match):
        module.load_probe_inputs(
            witness_path,
            cuda_path,
            expected_witness_sha256=module.sha256_file(witness_path),
            expected_cuda_result_sha256=module.sha256_file(cuda_path),
        )


def test_probe_rejects_cuda_anchor_that_does_not_authenticate_wmma(tmp_path):
    module, witness_path, cuda_path, _, _ = _write_inputs(tmp_path)
    with np.load(cuda_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["wmma_fp32_row"] = np.ones((1024,), dtype=np.float32)
    np.savez(cuda_path, **arrays)

    with pytest.raises(
        ValueError,
        match="WMMA FP32 row does not exactly authenticate",
    ):
        module.load_probe_inputs(
            witness_path,
            cuda_path,
            expected_witness_sha256=module.sha256_file(witness_path),
            expected_cuda_result_sha256=module.sha256_file(cuda_path),
        )


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (np.zeros((1024,), dtype=np.float32), "sm75_tensor_exact"),
        (np.ones((1024,), dtype=np.float32), "regular_exact"),
        (np.full((1024,), 0.5, dtype=np.float32), "third_island"),
    ],
)
def test_probe_classifies_tensor_regular_and_third_islands(
    tmp_path,
    row,
    expected,
):
    module, witness_path, cuda_path, witness_sha, cuda_sha = _write_inputs(
        tmp_path
    )
    inputs = module.load_probe_inputs(
        witness_path,
        cuda_path,
        expected_witness_sha256=witness_sha,
        expected_cuda_result_sha256=cuda_sha,
    )
    metal = np.zeros((16, 1024), dtype=np.float32)
    metal[module.SELECTED_WINDOW_ROW] = row
    analysis = module.analyze_metal_output(metal, inputs)
    assert analysis["classification"] == expected
    assert analysis["selected_window_row"] == 13


def test_probe_publishes_complete_result_with_route_identity(tmp_path):
    module, witness_path, cuda_path, witness_sha, cuda_sha = _write_inputs(
        tmp_path
    )
    output_json = tmp_path / "result.json"
    output_npz = tmp_path / "result.npz"

    def backend(window, weight):
        assert window.shape == (16, 1024)
        assert weight.shape == (1024, 1024)
        return np.zeros((16, 1024), dtype=np.float32), _route(module)

    report = module.run_probe(
        witness_path=witness_path,
        cuda_result_path=cuda_path,
        expected_witness_sha256=witness_sha,
        expected_cuda_result_sha256=cuda_sha,
        output_json=output_json,
        output_npz=output_npz,
        backend=backend,
    )
    assert report["status"] == "done"
    assert report["analysis"]["classification"] == "sm75_tensor_exact"
    assert report["effective_route"]["backend"] == "metal"
    assert report["effective_route"]["device"] == module.EXPECTED_DEVICE
    assert (
        report["effective_route"]["metal_source_sha256"]
        == module.METAL_SOURCE_SHA256
    )
    assert report["artifacts"]["witness_sha256"] == witness_sha
    assert report["artifacts"]["cuda_result_sha256"] == cuda_sha
    assert report["artifacts"]["output_npz_sha256"] == module.sha256_file(
        output_npz
    )
    assert json.loads(output_json.read_text()) == report
    with np.load(output_npz, allow_pickle=False) as archive:
        assert archive["metal_fp32_full"].shape == (16, 1024)
        assert archive["metal_fp32_full"].dtype == np.float32
        assert archive["metal_fp16_full"].dtype == np.float16


def test_probe_rejects_wrong_backend_output_without_primary(tmp_path):
    module, witness_path, cuda_path, witness_sha, cuda_sha = _write_inputs(
        tmp_path
    )
    output_json = tmp_path / "result.json"
    output_npz = tmp_path / "result.npz"

    def backend(_window, _weight):
        return np.zeros((16, 1024), dtype=np.float16), {
            "backend": "metal",
            "device": "test-apple-gpu",
            "kernel": "wrong-dtype",
        }

    with pytest.raises(ValueError, match="Metal output must have dtype float32"):
        module.run_probe(
            witness_path=witness_path,
            cuda_result_path=cuda_path,
            expected_witness_sha256=witness_sha,
            expected_cuda_result_sha256=cuda_sha,
            output_json=output_json,
            output_npz=output_npz,
            backend=backend,
        )
    assert not output_npz.exists()
    failure = json.loads(output_json.read_text())
    assert failure["status"] == "failed"
    assert failure["failure_phase"] == "backend_output_validation"


def test_probe_removes_both_outputs_on_postpublication_hard_interrupt(
    tmp_path,
    monkeypatch,
):
    module, witness_path, cuda_path, witness_sha, cuda_sha = _write_inputs(
        tmp_path
    )
    output_json = tmp_path / "result.json"
    output_npz = tmp_path / "result.npz"
    output_json.write_text('{"status":"stale"}')
    np.savez(output_npz, stale=np.array([1]))

    original = module._write_json_atomic

    def interrupt_after_json(path, payload):
        original(path, payload)
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "_write_json_atomic", interrupt_after_json)
    with pytest.raises(KeyboardInterrupt):
        module.run_probe(
            witness_path=witness_path,
            cuda_result_path=cuda_path,
            expected_witness_sha256=witness_sha,
            expected_cuda_result_sha256=cuda_sha,
            output_json=output_json,
            output_npz=output_npz,
            backend=lambda _window, _weight: (
                np.zeros((16, 1024), dtype=np.float32),
                _route(module),
            ),
        )
    assert not output_json.exists()
    assert not output_npz.exists()
