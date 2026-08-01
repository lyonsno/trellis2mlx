import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.source_cuda_cumesh_postprocess_witness import (
    WitnessError,
    sha256_file,
    write_binary_ply,
)
from scripts.source_metal_cuda_qem_cost_witness import (
    compare_float32_arrays,
    run_witness,
)


SOURCE_COMMIT = "1" * 40
PATCH_SHA256 = "2" * 64


def _identity(root: Path) -> dict:
    return {
        "module": "cumesh",
        "module_file": str(root / "cumesh" / "__init__.py"),
        "distribution": "cumesh",
        "distribution_version": "0.0.1",
        "distribution_root": str(root),
        "git_root": str(root),
        "git_commit": SOURCE_COMMIT,
        "git_status_porcelain": "",
        "has_CuMesh": True,
        "has_MtlMesh": True,
        "backend_module": "cumesh.metal_backend",
        "backend_module_file": str(root / "cumesh" / "metal_backend.py"),
    }


def _write_input(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 1]], dtype=np.int32)
    write_binary_ply(path, vertices, faces)
    return vertices, faces


def _cuda_arrays() -> dict[str, np.ndarray]:
    return {
        "vert2face": np.array([1, 0, 0, 1, 1, 0], dtype=np.int32),
        "qems": np.arange(30, dtype=np.float32).reshape(3, 10),
        "edge_collapse_costs": np.array([0.25, np.inf, 0.5], dtype=np.float32),
    }


def _write_cuda_trace(
    tmp_path: Path,
    input_sha256: str,
    *,
    route: str = "release-cumesh-qem-cost-trace-instrumented",
) -> tuple[Path, Path]:
    arrays = _cuda_arrays()
    npz = tmp_path / "cuda_qem_cost.npz"
    report = tmp_path / "cuda_qem_cost.json"
    np.savez(npz, **arrays)
    report.write_text(
        json.dumps(
            {
                "status": "done",
                "primary_output_status": "validated",
                "input_mesh": {"sha256": input_sha256},
                "effective_route": {
                    "geometry_route": route,
                    "cuda_available": True,
                    "cuda_device_name": "Tesla T4",
                    "cumesh_instrumentation": {
                        "schema": "trellis2mlx.cumesh_qem_cost_instrumentation.v1",
                        "patch_sha256": PATCH_SHA256,
                    },
                },
                "output_npz": {
                    "path": str(npz),
                    "sha256": sha256_file(npz),
                },
                "arrays": {
                    name: {
                        "shape": list(array.shape),
                        "dtype": str(array.dtype),
                        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                    }
                    for name, array in arrays.items()
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    return report, npz


def test_compare_float32_arrays_counts_bits_ulps_and_nonfinite_values():
    cuda = np.array([0.0, 1.0, np.inf, np.nan], dtype=np.float32)
    metal = cuda.copy()
    metal[0] = np.nextafter(np.float32(0.0), np.float32(1.0))
    metal[1] = np.nextafter(np.float32(1.0), np.float32(2.0))

    metrics = compare_float32_arrays(cuda, metal, chunk_size=2)

    assert metrics["elements"] == 4
    assert metrics["bit_exact"] is False
    assert metrics["bit_mismatch_count"] == 2
    assert metrics["nonfinite_class_mismatch_count"] == 0
    assert metrics["finite_pair_count"] == 2
    assert metrics["max_ulp"] == 1
    assert metrics["ulp_counts"]["1"] == 2
    assert metrics["first_bit_mismatch_index"] == [0]


def test_witness_injects_cuda_order_compares_native_arrays_and_reopens_npz(
    tmp_path,
):
    source_root = tmp_path / "mtlmesh"
    source_root.mkdir()
    input_ply = tmp_path / "input.ply"
    vertices, faces = _write_input(input_ply)
    cuda_report, cuda_npz = _write_cuda_trace(
        tmp_path,
        sha256_file(input_ply),
    )
    output_npz = tmp_path / "metal_qem_cost.npz"
    report_json = tmp_path / "comparison.json"
    cuda_arrays = _cuda_arrays()
    metal_arrays = {
        "qems": cuda_arrays["qems"].copy(),
        "edge_collapse_costs": cuda_arrays["edge_collapse_costs"].copy(),
    }
    metal_arrays["qems"][1, 4] = np.nextafter(
        metal_arrays["qems"][1, 4],
        np.float32(np.inf),
    )

    def runner(actual_vertices, actual_faces, actual_cuda_arrays, rsqrt_delta):
        assert np.array_equal(actual_vertices, vertices)
        assert np.array_equal(actual_faces, faces)
        assert rsqrt_delta is None
        assert np.array_equal(
            actual_cuda_arrays["vert2face"],
            cuda_arrays["vert2face"],
        )
        return metal_arrays, {
            "injected_readback_exact": True,
            "segment_multisets_exact": True,
            "qem_kernel": "metal-native-rsqrt",
        }

    report = run_witness(
        input_ply=input_ply,
        cuda_report_json=cuda_report,
        cuda_npz=cuda_npz,
        output_npz=output_npz,
        report_json=report_json,
        expected_input_sha256=sha256_file(input_ply),
        expected_cuda_report_sha256=sha256_file(cuda_report),
        expected_cuda_npz_sha256=sha256_file(cuda_npz),
        expected_source_root=source_root,
        expected_source_commit=SOURCE_COMMIT,
        identity_probe=lambda: _identity(source_root),
        runner=runner,
    )

    assert report["status"] == "done"
    assert report["primary_output_status"] == "validated"
    assert report["effective_route"]["cuda_device_name"] == "Tesla T4"
    assert report["injection"]["injected_readback_exact"] is True
    assert report["comparison"]["qems"]["bit_mismatch_count"] == 1
    assert report["comparison"]["qems"]["max_ulp"] == 1
    assert report["comparison"]["edge_collapse_costs"]["bit_exact"] is True
    with np.load(output_npz, allow_pickle=False) as reopened:
        assert set(reopened.files) == {
            "qems",
            "edge_collapse_costs",
            "injected_vert2face",
        }
        assert np.array_equal(
            reopened["qems"],
            metal_arrays["qems"],
            equal_nan=True,
        )
        assert np.array_equal(
            reopened["injected_vert2face"],
            cuda_arrays["vert2face"],
        )
    assert json.loads(report_json.read_text()) == report


def test_witness_rejects_non_cuda_qem_route_before_runner_or_output(tmp_path):
    source_root = tmp_path / "mtlmesh"
    source_root.mkdir()
    input_ply = tmp_path / "input.ply"
    _write_input(input_ply)
    cuda_report, cuda_npz = _write_cuda_trace(
        tmp_path,
        sha256_file(input_ply),
        route="not-the-qem-trace",
    )
    output_npz = tmp_path / "metal_qem_cost.npz"

    with pytest.raises(WitnessError, match="CUDA QEM geometry route"):
        run_witness(
            input_ply=input_ply,
            cuda_report_json=cuda_report,
            cuda_npz=cuda_npz,
            output_npz=output_npz,
            report_json=tmp_path / "comparison.json",
            expected_input_sha256=sha256_file(input_ply),
            expected_cuda_report_sha256=sha256_file(cuda_report),
            expected_cuda_npz_sha256=sha256_file(cuda_npz),
            expected_source_root=source_root,
            expected_source_commit=SOURCE_COMMIT,
            identity_probe=lambda: _identity(source_root),
            runner=lambda *args: pytest.fail("runner should not execute"),
        )

    assert not output_npz.exists()


def test_witness_records_authenticated_turing_rsqrt_route(tmp_path):
    source_root = tmp_path / "mtlmesh"
    source_root.mkdir()
    input_ply = tmp_path / "input.ply"
    _write_input(input_ply)
    cuda_report, cuda_npz = _write_cuda_trace(
        tmp_path,
        sha256_file(input_ply),
    )
    cuda_arrays = _cuda_arrays()
    rsqrt_lut = tmp_path / "cuda_rsqrt_result.npz"
    expected_delta = np.zeros(1 << 24, dtype=np.int8)
    np.savez_compressed(rsqrt_lut, normalized_delta=expected_delta)

    def runner(_vertices, _faces, _cuda_arrays, rsqrt_delta):
        assert np.array_equal(rsqrt_delta, expected_delta)
        return {
            "qems": cuda_arrays["qems"].copy(),
            "edge_collapse_costs": cuda_arrays["edge_collapse_costs"].copy(),
        }, {
            "injected_readback_exact": True,
            "segment_multisets_exact": True,
            "qem_kernel": "turing-rsqrt-lut",
        }

    report = run_witness(
        input_ply=input_ply,
        cuda_report_json=cuda_report,
        cuda_npz=cuda_npz,
        output_npz=tmp_path / "metal_qem_cost.npz",
        report_json=tmp_path / "comparison.json",
        expected_input_sha256=sha256_file(input_ply),
        expected_cuda_report_sha256=sha256_file(cuda_report),
        expected_cuda_npz_sha256=sha256_file(cuda_npz),
        expected_source_root=source_root,
        expected_source_commit=SOURCE_COMMIT,
        rsqrt_lut_npz=rsqrt_lut,
        expected_rsqrt_lut_sha256=sha256_file(rsqrt_lut),
        identity_probe=lambda: _identity(source_root),
        runner=runner,
    )

    assert report["effective_route"]["qem_kernel"] == "turing-rsqrt-lut"
    assert report["rsqrt_lut"]["npz_sha256"] == sha256_file(rsqrt_lut)
    assert report["rsqrt_lut"]["normalized_delta"]["shape"] == [1 << 24]
    assert report["primary_output_status"] == "validated"


def test_witness_rejects_native_kernel_attestation_on_turing_route(tmp_path):
    source_root = tmp_path / "mtlmesh"
    source_root.mkdir()
    input_ply = tmp_path / "input.ply"
    _write_input(input_ply)
    cuda_report, cuda_npz = _write_cuda_trace(
        tmp_path,
        sha256_file(input_ply),
    )
    rsqrt_lut = tmp_path / "cuda_rsqrt_result.npz"
    np.savez_compressed(
        rsqrt_lut,
        normalized_delta=np.zeros(1 << 24, dtype=np.int8),
    )
    output_npz = tmp_path / "metal_qem_cost.npz"

    def fallback_runner(*_args):
        cuda_arrays = _cuda_arrays()
        return {
            "qems": cuda_arrays["qems"].copy(),
            "edge_collapse_costs": cuda_arrays["edge_collapse_costs"].copy(),
        }, {
            "injected_readback_exact": True,
            "segment_multisets_exact": True,
            "qem_kernel": "metal-native-rsqrt",
        }

    with pytest.raises(WitnessError, match="runner QEM kernel route mismatch"):
        run_witness(
            input_ply=input_ply,
            cuda_report_json=cuda_report,
            cuda_npz=cuda_npz,
            output_npz=output_npz,
            report_json=tmp_path / "comparison.json",
            expected_input_sha256=sha256_file(input_ply),
            expected_cuda_report_sha256=sha256_file(cuda_report),
            expected_cuda_npz_sha256=sha256_file(cuda_npz),
            expected_source_root=source_root,
            expected_source_commit=SOURCE_COMMIT,
            rsqrt_lut_npz=rsqrt_lut,
            expected_rsqrt_lut_sha256=sha256_file(rsqrt_lut),
            identity_probe=lambda: _identity(source_root),
            runner=fallback_runner,
        )

    assert not output_npz.exists()


def test_witness_rejects_turing_route_without_authenticated_lut_before_runner(
    tmp_path,
):
    source_root = tmp_path / "mtlmesh"
    source_root.mkdir()
    input_ply = tmp_path / "input.ply"
    _write_input(input_ply)
    cuda_report, cuda_npz = _write_cuda_trace(
        tmp_path,
        sha256_file(input_ply),
    )
    rsqrt_lut = tmp_path / "cuda_rsqrt_result.npz"
    np.savez_compressed(
        rsqrt_lut,
        normalized_delta=np.zeros(1 << 24, dtype=np.int8),
    )

    with pytest.raises(WitnessError, match="both be supplied"):
        run_witness(
            input_ply=input_ply,
            cuda_report_json=cuda_report,
            cuda_npz=cuda_npz,
            output_npz=tmp_path / "metal_qem_cost.npz",
            report_json=tmp_path / "comparison.json",
            expected_input_sha256=sha256_file(input_ply),
            expected_cuda_report_sha256=sha256_file(cuda_report),
            expected_cuda_npz_sha256=sha256_file(cuda_npz),
            expected_source_root=source_root,
            expected_source_commit=SOURCE_COMMIT,
            rsqrt_lut_npz=rsqrt_lut,
            identity_probe=lambda: _identity(source_root),
            runner=lambda *args: pytest.fail("runner should not execute"),
        )
