from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cuda_decoder_block0_wmma_prefix_probe.py"


def _load_module():
    assert SCRIPT.is_file(), "SM75 WMMA prefix probe is missing"
    spec = importlib.util.spec_from_file_location(
        "cuda_decoder_block0_wmma_prefix_probe",
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


def _prefixes(module):
    return np.zeros(
        (module.PREFIX_COUNT, module.ROWS, module.TILE_WIDTH),
        dtype=np.float32,
    )


def _route(module):
    return {
        "backend": "cuda",
        "device": "Tesla T4",
        "kernel": module.KERNEL_IDENTITY,
        "effective_compute_capability": "7.5",
        "compiler_evidence": {
            "effective_ptx_architecture": "sm_75",
            "effective_cubin_architecture": "sm_75",
            "ptx_target_symbol": "sm75_wmma_prefix_kernel",
            "sass_target_symbol": "sm75_wmma_prefix_kernel",
            "ptx_wmma_m16n16k16_count": 1,
            "sass_hmma_1688_count": 4,
        },
    }


def test_probe_source_snapshots_every_wmma_k16_prefix():
    module = _load_module()
    assert "wmma::fragment<wmma::accumulator" in module.CUDA_SOURCE
    assert "for (int offset = 0; offset < reduction; offset += 16)" in (
        module.CUDA_SOURCE
    )
    assert "wmma::mma_sync" in module.CUDA_SOURCE
    assert "prefixes + (offset / 16) * 16 * 16" in module.CUDA_SOURCE
    assert "wmma::store_matrix_sync" in module.CUDA_SOURCE


def test_probe_rejects_wrong_digest_before_backend(tmp_path):
    module, witness_path, cuda_path, _, cuda_sha = _write_inputs(tmp_path)
    called = False

    def backend(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("backend must not run")

    with pytest.raises(ValueError, match="witness sha256"):
        module.run_probe(
            witness_path=witness_path,
            cuda_result_path=cuda_path,
            expected_witness_sha256="0" * 64,
            expected_cuda_result_sha256=cuda_sha,
            tile_col=16,
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

    def backend(*_args, **_kwargs):
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
            tile_col=16,
            output_json=output,
            output_npz=output,
            backend=backend,
        )
    assert not called
    assert not output.exists()


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
            "wmma_fp32_row",
            np.full((1024,), np.nan, dtype=np.float32),
            "wmma_fp32_row contains non-finite",
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


def test_probe_requires_authenticated_distinct_cuda_anchors(tmp_path):
    module, witness_path, cuda_path, _, _ = _write_inputs(tmp_path)
    with np.load(cuda_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["wmma_fp32_row"] = np.ones((1024,), dtype=np.float32)
    np.savez(cuda_path, **arrays)

    with pytest.raises(ValueError, match="WMMA FP32 row does not exactly"):
        module.load_probe_inputs(
            witness_path,
            cuda_path,
            expected_witness_sha256=module.sha256_file(witness_path),
            expected_cuda_result_sha256=module.sha256_file(cuda_path),
        )


@pytest.mark.parametrize("tile_col", [-16, 1, 1024])
def test_probe_rejects_unaligned_or_out_of_range_tile(tile_col):
    module = _load_module()
    with pytest.raises(ValueError, match="tile_col"):
        module.validate_tile_col(tile_col)


def test_probe_requires_all_ordered_k_prefixes():
    module = _load_module()
    prefixes = _prefixes(module)
    validated = module.validate_prefixes(prefixes)
    assert validated.shape == (64, 16, 16)

    with pytest.raises(ValueError, match=r"shape \(64, 16, 16\)"):
        module.validate_prefixes(prefixes[:-1])
    with pytest.raises(ValueError, match="dtype float32"):
        module.validate_prefixes(prefixes.astype(np.float16))
    prefixes[12, 3, 4] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        module.validate_prefixes(prefixes)


def test_probe_compiler_evidence_is_target_symbol_and_sm75_scoped():
    module = _load_module()
    target = "_Z25sm75_wmma_prefix_kernelPK6__halfS1_Pfiii"
    ptx = "\n".join(
        [
            ".target sm_75",
            f".visible .entry {target}(",
            ")",
            "{",
            (
                "wmma.mma.sync.aligned.row.row.m16n16k16"
                ".f32.f16.f16.f32"
            ),
            "}",
        ]
    )
    sass = "\n".join(
        [
            "code for sm_75",
            f"Function : {target}",
            "/*03d0*/ HMMA.1688.F32 R8, R12, R20, R8;",
        ]
    )
    evidence = module.classify_compiler_evidence(ptx=ptx, sass=sass)
    assert evidence["effective_ptx_architecture"] == "sm_75"
    assert evidence["effective_cubin_architecture"] == "sm_75"
    assert evidence["ptx_target_symbol"] == target
    assert evidence["sass_target_symbol"] == target

    unrelated = ptx.replace("sm75_wmma_prefix_kernel", "unrelated_kernel")
    with pytest.raises(ValueError, match="target.*prefix-kernel ptx"):
        module.classify_compiler_evidence(ptx=unrelated, sass=sass)

    wrong_arch = sass.replace("code for sm_75", "code for sm_80")
    with pytest.raises(ValueError, match="target.*prefix-kernel sass"):
        module.classify_compiler_evidence(ptx=ptx, sass=wrong_arch)


def test_probe_final_prefix_must_authenticate_admitted_wmma_row(tmp_path):
    module, witness_path, cuda_path, witness_sha, cuda_sha = _write_inputs(
        tmp_path
    )
    inputs = module.load_probe_inputs(
        witness_path,
        cuda_path,
        expected_witness_sha256=witness_sha,
        expected_cuda_result_sha256=cuda_sha,
    )
    prefixes = _prefixes(module)
    prefixes[-1, module.SELECTED_WINDOW_ROW, 4] = 1.0

    with pytest.raises(ValueError, match="final WMMA prefix"):
        module.analyze_prefixes(prefixes, inputs, tile_col=16)


def test_probe_publishes_route_bound_prefix_trace(tmp_path):
    module, witness_path, cuda_path, witness_sha, cuda_sha = _write_inputs(
        tmp_path
    )
    output_json = tmp_path / "result.json"
    output_npz = tmp_path / "result.npz"

    def backend(window, weight, *, tile_col):
        assert window.shape == (16, 1024)
        assert weight.shape == (1024, 1024)
        assert tile_col == 16
        return _prefixes(module), _route(module)

    report = module.run_probe(
        witness_path=witness_path,
        cuda_result_path=cuda_path,
        expected_witness_sha256=witness_sha,
        expected_cuda_result_sha256=cuda_sha,
        tile_col=16,
        output_json=output_json,
        output_npz=output_npz,
        backend=backend,
    )

    assert report["status"] == "done"
    assert report["analysis"]["final_prefix_exact_admitted_wmma_row"] is True
    assert report["analysis"]["k_prefixes"] == list(range(16, 1025, 16))
    assert report["effective_route"]["backend"] == "cuda"
    assert report["effective_route"]["compiler_evidence"][
        "effective_cubin_architecture"
    ] == "sm_75"
    assert report["artifacts"]["witness_sha256"] == witness_sha
    assert report["artifacts"]["cuda_result_sha256"] == cuda_sha
    assert report["artifacts"]["output_npz_sha256"] == module.sha256_file(
        output_npz
    )
    assert json.loads(output_json.read_text()) == report
    with np.load(output_npz, allow_pickle=False) as archive:
        np.testing.assert_array_equal(
            archive["k_prefixes"],
            np.arange(16, 1025, 16, dtype=np.int32),
        )
        assert archive["wmma_prefix_fp32"].shape == (64, 16, 16)


def test_probe_rejects_wrong_effective_route_without_primary(tmp_path):
    module, witness_path, cuda_path, witness_sha, cuda_sha = _write_inputs(
        tmp_path
    )
    output_json = tmp_path / "result.json"
    output_npz = tmp_path / "result.npz"

    def backend(_window, _weight, *, tile_col):
        del tile_col
        route = _route(module)
        route["device"] = "not-a-t4"
        return _prefixes(module), route

    with pytest.raises(ValueError, match="Tesla T4"):
        module.run_probe(
            witness_path=witness_path,
            cuda_result_path=cuda_path,
            expected_witness_sha256=witness_sha,
            expected_cuda_result_sha256=cuda_sha,
            tile_col=16,
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
    output_npz.write_bytes(b"stale")
    original_write = module._write_json_atomic

    def interrupting_write(path, payload):
        original_write(path, payload)
        raise KeyboardInterrupt()

    monkeypatch.setattr(module, "_write_json_atomic", interrupting_write)

    with pytest.raises(KeyboardInterrupt):
        module.run_probe(
            witness_path=witness_path,
            cuda_result_path=cuda_path,
            expected_witness_sha256=witness_sha,
            expected_cuda_result_sha256=cuda_sha,
            tile_col=16,
            output_json=output_json,
            output_npz=output_npz,
            backend=lambda *_args, **_kwargs: (
                _prefixes(module),
                _route(module),
            ),
        )
    assert not output_json.exists()
    assert not output_npz.exists()
