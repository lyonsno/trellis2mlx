from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cuda_sm75_wmma_subset_lattice_probe.py"


def _load_module():
    assert SCRIPT.is_file(), "SM75 WMMA subset-lattice probe is missing"
    spec = importlib.util.spec_from_file_location(
        "cuda_sm75_wmma_subset_lattice_probe",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _route(module):
    symbol = "_Z30sm75_wmma_subset_lattice_kernelPK6__halfS1_Pffi"
    return {
        "backend": "cuda",
        "device": "Tesla T4",
        "kernel": module.KERNEL_IDENTITY,
        "effective_compute_capability": "7.5",
        "compiler_evidence": {
            "effective_ptx_architecture": "sm_75",
            "effective_cubin_architecture": "sm_75",
            "ptx_target_symbol": symbol,
            "sass_target_symbol": symbol,
            "ptx_wmma_m16n16k16_count": 1,
            "sass_hmma_1688_count": 4,
        },
    }


def _uniform_outputs(representative_bits):
    bits = np.broadcast_to(
        np.asarray(representative_bits, dtype=np.uint32)[:, None, None],
        (65536, 16, 16),
    ).copy()
    return bits.view(np.float32)


def test_probe_source_runs_one_wmma_per_subset():
    module = _load_module()
    assert "blockIdx.x" in module.CUDA_SOURCE
    assert "wmma::fill_fragment(accumulator_fragment, accumulator)" in (
        module.CUDA_SOURCE
    )
    assert module.CUDA_SOURCE.count("wmma::mma_sync") == 1
    assert "wmma::store_matrix_sync" in module.CUDA_SOURCE
    assert "case_index * 16 * 16" in module.CUDA_SOURCE


def test_selected_operands_and_parent_custody_are_pinned():
    module = _load_module()
    assert module.SELECTED_WINDOW_ROW == 13
    assert module.SELECTED_GLOBAL_COLUMN == 16
    assert module.OPERAND_A_BITS == (
        0xBD63,
        0xC156,
        0xC8CE,
        0x41F7,
        0xC35D,
        0x455A,
        0x4220,
        0x427B,
        0x4637,
        0x4085,
        0xC02D,
        0x43E0,
        0xC018,
        0xBD99,
        0xC371,
        0xC4C6,
    )
    assert module.OPERAND_B_BITS == (
        0x3001,
        0xB654,
        0xB7B2,
        0xADB9,
        0xB5CB,
        0x26CF,
        0x2D85,
        0xB440,
        0xACAF,
        0x3276,
        0x2D58,
        0xB4B3,
        0x35AC,
        0x33B6,
        0x2C2B,
        0x3826,
    )
    assert module.PARENT_WITNESS_SHA256 == (
        "9fb030c521b0489bbdf7e0ee7eed29bd775d3f886894137da7757c2a38e0c105"
    )
    assert module.PARENT_DIRECT_WMMA_SHA256 == (
        "beb81530139d62dcc7f1e8690e0879b3d0ef8653cd6f09ab64baf69bb56206d5"
    )
    assert module.PARENT_PREFIX_SHA256 == (
        "329cd27cf3e90a3db74aa0b66c1a255aa41f1b9e55105416c1ab1552245aea94"
    )
    assert module.EXPECTED_FULL_T4_BITS == 0x3F815F2E
    assert module.EXPECTED_FULL_FLAT_BITS == 0x3F815F38


def test_generate_cases_covers_the_complete_boolean_lattice():
    module = _load_module()
    masks, matrix_a, matrix_b = module.generate_cases()
    assert masks.dtype == np.uint16
    assert masks.shape == (65536,)
    np.testing.assert_array_equal(
        masks.astype(np.uint32),
        np.arange(65536, dtype=np.uint32),
    )
    assert matrix_a.shape == (65536, 16, 16)
    assert matrix_b.shape == (65536, 16, 16)
    assert matrix_a.dtype == np.float16
    assert matrix_b.dtype == np.float16
    assert np.count_nonzero(matrix_a[0]) == 0
    assert np.count_nonzero(matrix_b[0]) == 0
    np.testing.assert_array_equal(
        matrix_a[1, :, 0].view(np.uint16),
        np.full(16, module.OPERAND_A_BITS[0], dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        matrix_b[1, 0, :].view(np.uint16),
        np.full(16, module.OPERAND_B_BITS[0], dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        matrix_a[-1, 0].view(np.uint16),
        np.asarray(module.OPERAND_A_BITS, dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        matrix_b[-1, :, 0].view(np.uint16),
        np.asarray(module.OPERAND_B_BITS, dtype=np.uint16),
    )


def test_flat_formal_lattice_is_complete_and_full_anchor_is_pinned():
    module = _load_module()
    masks = np.arange(65536, dtype=np.uint16)
    flat = module.flat_formal_bits_by_subset(masks)
    assert flat.dtype == np.uint32
    assert flat.shape == (65536,)
    assert int(flat[0]) == 0
    assert int(flat[-1]) == module.EXPECTED_FULL_FLAT_BITS
    assert int(flat[-1]) != module.EXPECTED_FULL_T4_BITS


def test_output_validation_requires_complete_finite_fp32_lattice():
    module = _load_module()
    value = np.zeros((65536, 16, 16), dtype=np.float32)
    assert module.validate_outputs(value).shape == (65536, 16, 16)
    with pytest.raises(ValueError, match=r"shape \(65536, 16, 16\)"):
        module.validate_outputs(value[:-1])
    with pytest.raises(ValueError, match="dtype float32"):
        module.validate_outputs(value.astype(np.float16))
    value[9, 3, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        module.validate_outputs(value)


def test_analysis_finds_minimum_higher_order_divergence():
    module = _load_module()
    masks = np.arange(65536, dtype=np.uint16)
    flat = module.flat_formal_bits_by_subset(masks)
    hardware = flat.copy()
    hardware[0b111] ^= np.uint32(1)
    hardware[-1] = module.EXPECTED_FULL_T4_BITS
    analysis = module.analyze_outputs(
        _uniform_outputs(hardware),
        masks,
        flat,
    )
    assert analysis["classification"] == "higher_order_divergence"
    assert analysis["minimum_divergent_cardinality"] == 3
    assert analysis["divergent_subsets"] == 2
    assert analysis["mixed_subsets"] == 0
    assert analysis["unexpected_full_anchor"] is False
    cardinality_three = analysis["by_cardinality"][3]
    assert cardinality_three["subsets"] == 560
    assert cardinality_three["divergent_from_flat"] == 1


def test_analysis_does_not_coerce_mixed_output_cells():
    module = _load_module()
    masks = np.arange(65536, dtype=np.uint16)
    flat = module.flat_formal_bits_by_subset(masks)
    hardware = flat.copy()
    hardware[-1] = module.EXPECTED_FULL_T4_BITS
    outputs = _uniform_outputs(hardware)
    outputs.view(np.uint32)[7, 0, 0] ^= np.uint32(1)
    analysis = module.analyze_outputs(outputs, masks, flat)
    assert analysis["classification"] == "mixed_output"
    assert analysis["mixed_subsets"] == 1
    assert analysis["minimum_divergent_cardinality"] is None


def test_compiler_evidence_is_subset_symbol_and_sm75_scoped():
    module = _load_module()
    target = "_Z30sm75_wmma_subset_lattice_kernelPK6__halfS1_Pffi"
    ptx = "\n".join(
        [
            ".target sm_75",
            f".visible .entry {target}(",
            ")",
            "{",
            "wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32",
            "}",
        ]
    )
    sass = "\n".join(
        [
            "code for sm_75",
            f"Function : {target}",
            "/*03d0*/ HMMA.1688.F32 R8, R12, R20, R8;",
        ]
        * 4
    )
    evidence = module.classify_compiler_evidence(ptx=ptx, sass=sass)
    assert evidence["ptx_target_symbol"] == target
    assert evidence["sass_target_symbol"] == target
    unrelated = ptx.replace(
        "sm75_wmma_subset_lattice_kernel",
        "unrelated_kernel",
    )
    with pytest.raises(ValueError, match="target.*subset-kernel ptx"):
        module.classify_compiler_evidence(ptx=unrelated, sass=sass)


def test_probe_rejects_json_npz_collision_before_backend(tmp_path):
    module = _load_module()
    called = False

    def backend(*_args):
        nonlocal called
        called = True
        raise AssertionError("backend must not run")

    output = tmp_path / "same"
    with pytest.raises(ValueError, match="distinct paths"):
        module.run_probe(
            output_json=output,
            output_npz=output,
            backend=backend,
        )
    assert not called
    assert not output.exists()


def test_probe_publishes_route_bound_complete_lattice(tmp_path):
    module = _load_module()
    output_json = tmp_path / "result.json"
    output_npz = tmp_path / "result.npz"

    def backend(_matrix_a, _matrix_b):
        masks = np.arange(65536, dtype=np.uint16)
        flat = module.flat_formal_bits_by_subset(masks)
        flat[-1] = module.EXPECTED_FULL_T4_BITS
        return _uniform_outputs(flat), _route(module)

    report = module.run_probe(
        output_json=output_json,
        output_npz=output_npz,
        backend=backend,
    )
    assert report["status"] == "done"
    assert report["analysis"]["classification"] == "higher_order_divergence"
    assert report["effective_route"]["backend"] == "cuda"
    assert report["artifacts"]["output_npz_sha256"] == module.sha256_file(
        output_npz
    )
    assert json.loads(output_json.read_text()) == report
    with np.load(output_npz, allow_pickle=False) as archive:
        assert archive.files == [
            "subset_masks",
            "wmma_subset_fp32",
            "flat_formal_bits",
        ]
        assert archive["subset_masks"].shape == (65536,)
        assert archive["wmma_subset_fp32"].shape == (65536, 16, 16)
        assert archive["flat_formal_bits"].shape == (65536,)


def test_probe_rejects_wrong_route_without_primary(tmp_path):
    module = _load_module()
    output_json = tmp_path / "result.json"
    output_npz = tmp_path / "result.npz"

    def backend(_matrix_a, _matrix_b):
        route = _route(module)
        route["device"] = "not-a-t4"
        return np.zeros((65536, 16, 16), dtype=np.float32), route

    with pytest.raises(ValueError, match="Tesla T4"):
        module.run_probe(
            output_json=output_json,
            output_npz=output_npz,
            backend=backend,
        )
    assert not output_npz.exists()
    failure = json.loads(output_json.read_text())
    assert failure["status"] == "failed"
    assert failure["failure_phase"] == "backend_output_validation"


def test_probe_invalidates_stale_outputs_and_reports_backend_failure(tmp_path):
    module = _load_module()
    output_json = tmp_path / "result.json"
    output_npz = tmp_path / "result.npz"
    output_json.write_text('{"status":"stale"}')
    output_npz.write_bytes(b"stale")

    def backend(_matrix_a, _matrix_b):
        raise RuntimeError("backend broke")

    with pytest.raises(RuntimeError, match="backend broke"):
        module.run_probe(
            output_json=output_json,
            output_npz=output_npz,
            backend=backend,
        )
    assert not output_npz.exists()
    failure = json.loads(output_json.read_text())
    assert failure["status"] == "failed"
    assert failure["failure_phase"] == "backend_execution"


def test_probe_removes_both_outputs_on_postpublication_hard_interrupt(
    tmp_path,
    monkeypatch,
):
    module = _load_module()
    output_json = tmp_path / "result.json"
    output_npz = tmp_path / "result.npz"
    original_write = module._write_json_atomic

    def interrupting_write(path, payload):
        original_write(path, payload)
        raise KeyboardInterrupt()

    monkeypatch.setattr(module, "_write_json_atomic", interrupting_write)

    def backend(_matrix_a, _matrix_b):
        masks = np.arange(65536, dtype=np.uint16)
        flat = module.flat_formal_bits_by_subset(masks)
        flat[-1] = module.EXPECTED_FULL_T4_BITS
        return _uniform_outputs(flat), _route(module)

    with pytest.raises(KeyboardInterrupt):
        module.run_probe(
            output_json=output_json,
            output_npz=output_npz,
            backend=backend,
        )
    assert not output_npz.exists()
    failure = json.loads(output_json.read_text())
    assert failure["status"] == "failed"
    assert failure["failure_phase"] == "primary_publication"
    assert failure["error_type"] == "KeyboardInterrupt"
    assert failure["hard_interruption"] is True
