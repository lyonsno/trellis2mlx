from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cuda_sm75_wmma_pairwise_group_probe.py"


def _load_module():
    assert SCRIPT.is_file(), "SM75 WMMA pairwise-group probe is missing"
    spec = importlib.util.spec_from_file_location(
        "cuda_sm75_wmma_pairwise_group_probe",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _route(module):
    return {
        "backend": "cuda",
        "device": "Tesla T4",
        "kernel": module.KERNEL_IDENTITY,
        "effective_compute_capability": "7.5",
        "compiler_evidence": {
            "effective_ptx_architecture": "sm_75",
            "effective_cubin_architecture": "sm_75",
            "ptx_target_symbol": "sm75_wmma_pairwise_group_kernel",
            "sass_target_symbol": "sm75_wmma_pairwise_group_kernel",
            "ptx_wmma_m16n16k16_count": 1,
            "sass_hmma_1688_count": 4,
        },
    }


def _classified_output(module, groups):
    pairs, _, _ = module.generate_cases()
    values = np.empty(
        (module.CASE_COUNT, module.MATRIX_SIZE, module.MATRIX_SIZE),
        dtype=np.uint32,
    )
    group_by_position = {
        position: group_index
        for group_index, group in enumerate(groups)
        for position in group
    }
    for case, (left, right) in enumerate(pairs):
        values[case].fill(
            module.SAME_GROUP_RESULT_BITS
            if group_by_position[int(left)] == group_by_position[int(right)]
            else module.SEPARATE_GROUP_RESULT_BITS
        )
    return values.view(np.float32)


def test_probe_source_runs_one_wmma_case_per_pair():
    module = _load_module()
    assert "blockIdx.x" in module.CUDA_SOURCE
    assert "wmma::fill_fragment(accumulator_fragment, accumulator)" in (
        module.CUDA_SOURCE
    )
    assert "wmma::mma_sync" in module.CUDA_SOURCE
    assert "wmma::store_matrix_sync" in module.CUDA_SOURCE
    assert "case_index * 16 * 16" in module.CUDA_SOURCE


def test_generate_cases_covers_every_unordered_k_pair_once():
    module = _load_module()
    pairs, matrix_a, matrix_b = module.generate_cases()
    assert pairs.shape == (120, 2)
    assert pairs.dtype == np.int32
    assert len({tuple(pair) for pair in pairs.tolist()}) == 120
    assert np.all(pairs[:, 0] < pairs[:, 1])
    assert set(map(tuple, pairs.tolist())) == {
        (left, right)
        for left in range(16)
        for right in range(left + 1, 16)
    }
    assert matrix_a.shape == (120, 16, 16)
    assert matrix_b.shape == (120, 16, 16)
    assert matrix_a.dtype == np.float16
    assert matrix_b.dtype == np.float16

    first_left, first_right = map(int, pairs[0])
    np.testing.assert_array_equal(
        matrix_a[0, :, first_left].view(np.uint16),
        np.full(16, module.A1_BITS, dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        matrix_a[0, :, first_right].view(np.uint16),
        np.full(16, module.A2_BITS, dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        matrix_b[0, first_left, :].view(np.uint16),
        np.full(16, module.B1_BITS, dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        matrix_b[0, first_right, :].view(np.uint16),
        np.full(16, module.B2_BITS, dtype=np.uint16),
    )
    assert np.count_nonzero(matrix_a[0]) == 32
    assert np.count_nonzero(matrix_b[0]) == 32


def test_formal_discriminator_bits_are_pinned():
    module = _load_module()
    assert module.A1_BITS == 0xC1E6
    assert module.B1_BITS == 0xAD61
    assert module.A2_BITS == 0xA3CB
    assert module.B2_BITS == 0x3734
    assert module.ACCUMULATOR_BITS == 0x3B1507BA
    assert module.SAME_GROUP_RESULT_BITS == 0x3E792107
    assert module.SEPARATE_GROUP_RESULT_BITS == 0x3E792108
    assert module.formal_discriminator_check() == {
        "same_group_result_bits": "0x3e792107",
        "separate_group_forward_result_bits": "0x3e792108",
        "separate_group_reverse_result_bits": "0x3e792108",
    }


def test_output_validation_requires_complete_finite_fp32_matrix():
    module = _load_module()
    value = np.zeros((120, 16, 16), dtype=np.float32)
    assert module.validate_outputs(value).shape == (120, 16, 16)
    with pytest.raises(ValueError, match=r"shape \(120, 16, 16\)"):
        module.validate_outputs(value[:-1])
    with pytest.raises(ValueError, match="dtype float32"):
        module.validate_outputs(value.astype(np.float16))
    value[5, 3, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        module.validate_outputs(value)


def test_analysis_recovers_four_equivalence_groups_without_order_claim():
    module = _load_module()
    groups = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15))
    pairs, _, _ = module.generate_cases()
    analysis = module.analyze_outputs(_classified_output(module, groups), pairs)
    assert analysis["classification"] == "four_groups_of_four"
    assert analysis["groups"] == [list(group) for group in groups]
    assert analysis["uniform_same_group_pairs"] == 24
    assert analysis["uniform_separate_group_pairs"] == 96
    assert analysis["unexpected_cells"] == 0
    assert analysis["mixed_pairs"] == 0


def test_analysis_does_not_coerce_unexpected_or_mixed_results():
    module = _load_module()
    groups = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15))
    pairs, _, _ = module.generate_cases()
    output = _classified_output(module, groups)
    output.view(np.uint32)[0, 0, 0] = 0x3F800000
    analysis = module.analyze_outputs(output, pairs)
    assert analysis["classification"] == "unexpected_output"
    assert analysis["unexpected_cells"] == 1
    assert analysis["mixed_pairs"] == 1
    assert analysis["groups"] is None


def test_compiler_evidence_is_pairwise_symbol_and_sm75_scoped():
    module = _load_module()
    target = "_Z33sm75_wmma_pairwise_group_kernelPK6__halfS1_Pfj"
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
    )
    evidence = module.classify_compiler_evidence(ptx=ptx, sass=sass)
    assert evidence["ptx_target_symbol"] == target
    assert evidence["sass_target_symbol"] == target
    unrelated = ptx.replace(
        "sm75_wmma_pairwise_group_kernel",
        "unrelated_kernel",
    )
    with pytest.raises(ValueError, match="target.*pairwise-kernel ptx"):
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


def test_probe_publishes_route_bound_pair_matrix(tmp_path):
    module = _load_module()
    groups = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15))
    output_json = tmp_path / "result.json"
    output_npz = tmp_path / "result.npz"

    def backend(matrix_a, matrix_b):
        assert matrix_a.shape == (120, 16, 16)
        assert matrix_b.shape == (120, 16, 16)
        return _classified_output(module, groups), _route(module)

    report = module.run_probe(
        output_json=output_json,
        output_npz=output_npz,
        backend=backend,
    )
    assert report["status"] == "done"
    assert report["analysis"]["classification"] == "four_groups_of_four"
    assert report["effective_route"]["backend"] == "cuda"
    assert report["artifacts"]["output_npz_sha256"] == module.sha256_file(
        output_npz
    )
    assert json.loads(output_json.read_text()) == report
    with np.load(output_npz, allow_pickle=False) as archive:
        assert archive.files == ["pair_indices", "wmma_pairwise_fp32"]
        assert archive["pair_indices"].shape == (120, 2)
        assert archive["wmma_pairwise_fp32"].shape == (120, 16, 16)


def test_probe_rejects_wrong_route_without_primary(tmp_path):
    module = _load_module()
    output_json = tmp_path / "result.json"
    output_npz = tmp_path / "result.npz"

    def backend(_matrix_a, _matrix_b):
        route = _route(module)
        route["device"] = "not-a-t4"
        return np.zeros((120, 16, 16), dtype=np.float32), route

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


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda route: route["compiler_evidence"].update(
                {
                    "ptx_target_symbol": "unrelated_kernel",
                    "sass_target_symbol": "unrelated_kernel",
                }
            ),
            "target symbol",
        ),
        (
            lambda route: route["compiler_evidence"].update(
                {"ptx_wmma_m16n16k16_count": 2}
            ),
            "exactly one",
        ),
    ],
)
def test_probe_rejects_substituted_or_duplicated_compiler_route(
    tmp_path,
    mutate,
    message,
):
    module = _load_module()
    output_json = tmp_path / "result.json"
    output_npz = tmp_path / "result.npz"

    def backend(_matrix_a, _matrix_b):
        route = _route(module)
        mutate(route)
        return np.zeros((120, 16, 16), dtype=np.float32), route

    with pytest.raises(ValueError, match=message):
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
    groups = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15))
    output_json = tmp_path / "result.json"
    output_npz = tmp_path / "result.npz"
    original_write = module._write_json_atomic

    def interrupting_write(path, payload):
        original_write(path, payload)
        raise KeyboardInterrupt()

    monkeypatch.setattr(module, "_write_json_atomic", interrupting_write)

    def backend(_matrix_a, _matrix_b):
        return _classified_output(module, groups), _route(module)

    with pytest.raises(KeyboardInterrupt):
        module.run_probe(
            output_json=output_json,
            output_npz=output_npz,
            backend=backend,
        )
    assert not output_json.exists()
    assert not output_npz.exists()
