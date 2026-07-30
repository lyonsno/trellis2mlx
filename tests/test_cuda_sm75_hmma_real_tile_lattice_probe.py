from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cuda_sm75_hmma_real_tile_lattice_probe.py"


def _load_module():
    assert SCRIPT.is_file(), "SM75 HMMA real-tile lattice probe is missing"
    spec = importlib.util.spec_from_file_location(
        "cuda_sm75_hmma_real_tile_lattice_probe",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _route(module):
    direct_symbol = "_Z29sm75_hmma_m16n8k8_stage_kernelPK6__halfS1_PKfPfiiii"
    wmma_symbol = "_Z31sm75_wmma_real_lattice_kernelPK6__halfS1_Pfi"
    return {
        "backend": "cuda",
        "device": "Tesla T4",
        "kernel": module.KERNEL_IDENTITY,
        "effective_compute_capability": "7.5",
        "compiler_evidence": {
            "effective_ptx_architecture": "sm_75",
            "effective_cubin_architecture": "sm_75",
            "wmma_effective_ptx_architecture": "sm_75",
            "wmma_effective_cubin_architecture": "sm_75",
            "direct_ptx_target_symbol": direct_symbol,
            "direct_sass_target_symbol": direct_symbol,
            "direct_ptx_m16n8k8_count": 1,
            "direct_sass_hmma_1688_count": 1,
            "wmma_ptx_target_symbol": wmma_symbol,
            "wmma_sass_target_symbol": wmma_symbol,
            "wmma_ptx_m16n16k16_count": 1,
            "wmma_sass_hmma_1688_count": 4,
        },
    }


def _inputs(module):
    base_a = np.arange(256, dtype=np.float16).reshape(16, 16)
    base_b = (np.arange(256, dtype=np.float16) + 1).reshape(16, 16)
    prefix = np.arange(256, dtype=np.float32).reshape(16, 16)
    return {
        "witness_sha256": module.PARENT_WITNESS_SHA256,
        "cuda_result_sha256": module.PARENT_DIRECT_WMMA_SHA256,
        "prefix_result_sha256": module.PARENT_PREFIX_SHA256,
        "base_a": base_a,
        "base_b": base_b,
        "expected_full_wmma": prefix,
    }


def _outputs(case_count, *, exact=True):
    first = np.arange(case_count * 256, dtype=np.float32).reshape(
        case_count,
        16,
        16,
    )
    direct = first + np.float32(1)
    wmma = direct.copy()
    if not exact:
        direct[0, 0, 0] += np.float32(1)
    return first, direct, wmma


def test_source_contains_one_direct_hmma_and_one_wmma_reference():
    module = _load_module()
    assert module.CUDA_SOURCE.count(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
    ) == 1
    assert module.CUDA_SOURCE.count("wmma::mma_sync") == 1
    assert "sm75_hmma_m16n8k8_stage_kernel" in module.CUDA_SOURCE
    assert "sm75_wmma_real_lattice_kernel" in module.CUDA_SOURCE
    assert "stage_zero" in module.CUDA_SOURCE
    assert "stage_one" in module.CUDA_SOURCE


def test_parent_custody_and_real_tile_coordinates_are_pinned():
    module = _load_module()
    assert module.PARENT_WITNESS_SHA256 == (
        "9fb030c521b0489bbdf7e0ee7eed29bd775d3f886894137da7757c2a38e0c105"
    )
    assert module.PARENT_DIRECT_WMMA_SHA256 == (
        "beb81530139d62dcc7f1e8690e0879b3d0ef8653cd6f09ab64baf69bb56206d5"
    )
    assert module.PARENT_PREFIX_SHA256 == (
        "329cd27cf3e90a3db74aa0b66c1a255aa41f1b9e55105416c1ab1552245aea94"
    )
    assert module.TILE_COLUMN == 16


def test_generate_cases_masks_the_real_k_axis_without_substitution():
    module = _load_module()
    base_a = np.arange(256, dtype=np.float16).reshape(16, 16)
    base_b = (np.arange(256, dtype=np.float16) + 1).reshape(16, 16)
    masks = np.array([0, 1, 1 << 15, (1 << 16) - 1], dtype=np.uint16)
    matrix_a, matrix_b = module.generate_cases(base_a, base_b, masks)
    assert matrix_a.shape == (4, 16, 16)
    assert matrix_b.shape == (4, 16, 16)
    assert matrix_a.dtype == np.float16
    assert matrix_b.dtype == np.float16
    assert not np.any(matrix_a[0])
    assert not np.any(matrix_b[0])
    np.testing.assert_array_equal(matrix_a[1, :, 0], base_a[:, 0])
    assert not np.any(matrix_a[1, :, 1:])
    np.testing.assert_array_equal(matrix_b[1, 0, :], base_b[0, :])
    assert not np.any(matrix_b[1, 1:, :])
    np.testing.assert_array_equal(matrix_a[-1], base_a)
    np.testing.assert_array_equal(matrix_b[-1], base_b)


def test_generate_complete_masks_is_uncapped_and_ordered():
    module = _load_module()
    masks = module.complete_masks()
    assert masks.dtype == np.uint16
    assert masks.shape == (65536,)
    np.testing.assert_array_equal(
        masks.astype(np.uint32),
        np.arange(65536, dtype=np.uint32),
    )


def test_validate_outputs_rejects_partial_blank_or_nonfinite_primary():
    module = _load_module()
    valid = np.ones((4, 16, 16), dtype=np.float32)
    module.validate_outputs(valid, case_count=4, label="direct")
    with pytest.raises(ValueError, match="shape"):
        module.validate_outputs(valid[:-1], case_count=4, label="direct")
    with pytest.raises(ValueError, match="blank"):
        module.validate_outputs(np.zeros_like(valid), case_count=4, label="direct")
    invalid = valid.copy()
    invalid[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        module.validate_outputs(invalid, case_count=4, label="direct")


def test_analyze_classifies_exact_composition_without_coercing_mismatch():
    module = _load_module()
    masks = np.array([0, 0xFFFF], dtype=np.uint16)
    first, direct, wmma = _outputs(2)
    expected = wmma[-1].copy()
    exact = module.analyze_outputs(
        masks,
        first,
        direct,
        wmma,
        expected_full_wmma=expected,
    )
    assert exact["classification"] == "register_visible_composition_exact"
    assert exact["direct_vs_wmma"]["nonzero"] == 0
    direct[0, 0, 0] += np.float32(1)
    divergent = module.analyze_outputs(
        masks,
        first,
        direct,
        wmma,
        expected_full_wmma=expected,
    )
    assert divergent["classification"] == "hidden_cross_instruction_state"
    assert divergent["direct_vs_wmma"]["nonzero"] == 1


def test_analyze_rejects_wrong_full_wmma_anchor():
    module = _load_module()
    masks = np.array([0, 0xFFFF], dtype=np.uint16)
    first, direct, wmma = _outputs(2)
    expected = wmma[-1].copy()
    expected[0, 0] += np.float32(1)
    with pytest.raises(ValueError, match="full-mask WMMA"):
        module.analyze_outputs(
            masks,
            first,
            direct,
            wmma,
            expected_full_wmma=expected,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("backend", "cpu", "backend"),
        ("device", "NVIDIA A100", "device"),
        ("kernel", "fallback", "kernel"),
        ("effective_compute_capability", "8.0", "compute"),
    ],
)
def test_route_rejects_fallback_identity(field, value, message):
    module = _load_module()
    route = _route(module)
    route[field] = value
    with pytest.raises(ValueError, match=message):
        module.validate_effective_route(route)


def test_route_rejects_substituted_symbols_or_instruction_counts():
    module = _load_module()
    route = _route(module)
    route["compiler_evidence"]["direct_sass_target_symbol"] += "_other"
    with pytest.raises(ValueError, match="direct.*symbols"):
        module.validate_effective_route(route)
    route = _route(module)
    route["compiler_evidence"]["direct_sass_hmma_1688_count"] = 2
    with pytest.raises(ValueError, match="direct.*HMMA"):
        module.validate_effective_route(route)
    route = _route(module)
    route["compiler_evidence"]["wmma_sass_hmma_1688_count"] = 3
    with pytest.raises(ValueError, match="WMMA.*HMMA"):
        module.validate_effective_route(route)


@pytest.mark.parametrize(
    "field",
    [
        "effective_ptx_architecture",
        "effective_cubin_architecture",
        "wmma_effective_ptx_architecture",
        "wmma_effective_cubin_architecture",
    ],
)
def test_route_rejects_missing_compiler_architecture_identity(field):
    module = _load_module()
    route = _route(module)
    del route["compiler_evidence"][field]
    with pytest.raises(ValueError, match=field):
        module.validate_effective_route(route)


def test_run_probe_rejects_partial_authoritative_lattice_before_backend(
    tmp_path,
):
    module = _load_module()
    inputs = _inputs(module)
    masks = np.array([0, 0xFFFF], dtype=np.uint16)
    backend_called = False

    def backend(*args, **kwargs):
        nonlocal backend_called
        backend_called = True
        return (*_outputs(2), _route(module))

    with pytest.raises(ValueError, match="complete ordered"):
        module.run_probe(
            witness_path=tmp_path / "witness.npz",
            cuda_result_path=tmp_path / "cuda.npz",
            prefix_result_path=tmp_path / "prefix.npz",
            expected_witness_sha256=module.PARENT_WITNESS_SHA256,
            expected_cuda_result_sha256=module.PARENT_DIRECT_WMMA_SHA256,
            expected_prefix_result_sha256=module.PARENT_PREFIX_SHA256,
            output_json=tmp_path / "result.json",
            output_npz=tmp_path / "result.npz",
            masks=masks,
            input_loader=lambda *args, **kwargs: inputs,
            backend=backend,
        )
    assert backend_called is False
    assert not (tmp_path / "result.npz").exists()
    failure = json.loads((tmp_path / "result.json").read_text())
    assert failure["status"] == "failed"
    assert failure["failure_phase"] == "input_validation"


@pytest.mark.parametrize("alias", ["json-witness", "npz-prefix"])
def test_run_probe_rejects_parent_output_collision_without_mutation(
    tmp_path,
    alias,
):
    module = _load_module()
    parent_paths = {
        "witness": tmp_path / "witness.npz",
        "cuda": tmp_path / "cuda.npz",
        "prefix": tmp_path / "prefix.npz",
    }
    parent_bytes = {
        name: f"authenticated {name}".encode()
        for name in parent_paths
    }
    for name, path in parent_paths.items():
        path.write_bytes(parent_bytes[name])
    output_json = tmp_path / "result.json"
    output_npz = tmp_path / "result.npz"
    if alias == "json-witness":
        output_json = parent_paths["witness"]
    else:
        output_npz = parent_paths["prefix"]

    with pytest.raises(ValueError, match="collide"):
        module.run_probe(
            witness_path=parent_paths["witness"],
            cuda_result_path=parent_paths["cuda"],
            prefix_result_path=parent_paths["prefix"],
            expected_witness_sha256=module.PARENT_WITNESS_SHA256,
            expected_cuda_result_sha256=module.PARENT_DIRECT_WMMA_SHA256,
            expected_prefix_result_sha256=module.PARENT_PREFIX_SHA256,
            output_json=output_json,
            output_npz=output_npz,
            masks=np.array([0, 0xFFFF], dtype=np.uint16),
            input_loader=lambda *args, **kwargs: _inputs(module),
            backend=lambda *args, **kwargs: (
                *_outputs(2),
                _route(module),
            ),
        )
    for name, path in parent_paths.items():
        assert path.read_bytes() == parent_bytes[name]


def test_run_probe_publishes_complete_primary_and_effective_route(tmp_path):
    module = _load_module()
    output_json = tmp_path / "result.json"
    output_npz = tmp_path / "result.npz"
    inputs = _inputs(module)
    masks = np.array([0, 0xFFFF], dtype=np.uint16)
    first, direct, wmma = _outputs(2)
    inputs["expected_full_wmma"] = wmma[-1].copy()

    def backend(matrix_a, matrix_b):
        assert matrix_a.shape == (2, 16, 16)
        assert matrix_b.shape == (2, 16, 16)
        return first, direct, wmma, _route(module)

    report = module.run_probe(
        witness_path=tmp_path / "witness.npz",
        cuda_result_path=tmp_path / "cuda.npz",
        prefix_result_path=tmp_path / "prefix.npz",
        expected_witness_sha256=module.PARENT_WITNESS_SHA256,
        expected_cuda_result_sha256=module.PARENT_DIRECT_WMMA_SHA256,
        expected_prefix_result_sha256=module.PARENT_PREFIX_SHA256,
        output_json=output_json,
        output_npz=output_npz,
        masks=masks,
        input_loader=lambda *args, **kwargs: inputs,
        backend=backend,
        authoritative=False,
    )
    assert report["status"] == "test_done"
    assert report["schema"] == module.TEST_SCHEMA
    assert report["analysis"]["classification"] == (
        "register_visible_composition_exact"
    )
    assert report["effective_route"]["device"] == "Tesla T4"
    with np.load(output_npz) as archive:
        assert set(archive.files) == {
            "subset_masks",
            "direct_stage0_fp32",
            "direct_stage1_fp32",
            "wmma_fp32",
        }
        np.testing.assert_array_equal(archive["subset_masks"], masks)


def test_run_probe_writes_failure_phase_and_removes_stale_primary(tmp_path):
    module = _load_module()
    output_json = tmp_path / "result.json"
    output_npz = tmp_path / "result.npz"
    output_json.write_text('{"status":"done"}\n')
    output_npz.write_bytes(b"stale")
    inputs = _inputs(module)
    masks = np.array([0, 0xFFFF], dtype=np.uint16)

    def backend(*args, **kwargs):
        raise RuntimeError("backend exploded")

    with pytest.raises(RuntimeError, match="backend exploded"):
        module.run_probe(
            witness_path=tmp_path / "witness.npz",
            cuda_result_path=tmp_path / "cuda.npz",
            prefix_result_path=tmp_path / "prefix.npz",
            expected_witness_sha256=module.PARENT_WITNESS_SHA256,
            expected_cuda_result_sha256=module.PARENT_DIRECT_WMMA_SHA256,
            expected_prefix_result_sha256=module.PARENT_PREFIX_SHA256,
            output_json=output_json,
            output_npz=output_npz,
            masks=masks,
            input_loader=lambda *args, **kwargs: inputs,
            backend=backend,
            authoritative=False,
        )
    assert not output_npz.exists()
    failure = json.loads(output_json.read_text())
    assert failure["status"] == "failed"
    assert failure["failure_phase"] == "backend_execution"
    assert failure["error_type"] == "RuntimeError"


def test_run_probe_preserves_hard_interrupt_failure_report(tmp_path):
    module = _load_module()
    output_json = tmp_path / "result.json"
    output_npz = tmp_path / "result.npz"
    inputs = _inputs(module)
    masks = np.array([0, 0xFFFF], dtype=np.uint16)

    def backend(*args, **kwargs):
        raise KeyboardInterrupt("stop")

    with pytest.raises(KeyboardInterrupt, match="stop"):
        module.run_probe(
            witness_path=tmp_path / "witness.npz",
            cuda_result_path=tmp_path / "cuda.npz",
            prefix_result_path=tmp_path / "prefix.npz",
            expected_witness_sha256=module.PARENT_WITNESS_SHA256,
            expected_cuda_result_sha256=module.PARENT_DIRECT_WMMA_SHA256,
            expected_prefix_result_sha256=module.PARENT_PREFIX_SHA256,
            output_json=output_json,
            output_npz=output_npz,
            masks=masks,
            input_loader=lambda *args, **kwargs: inputs,
            backend=backend,
            authoritative=False,
        )
    assert not output_npz.exists()
    failure = json.loads(output_json.read_text())
    assert failure["status"] == "failed"
    assert failure["hard_interruption"] is True
    assert failure["failure_phase"] == "backend_execution"
