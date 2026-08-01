import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.source_cuda_cumesh_postprocess_witness import (
    WitnessError,
    read_binary_ply,
    sha256_file,
    write_binary_ply,
)
from scripts.source_metal_cuda_adjacency_simplify_witness import run_witness


SOURCE_COMMIT = "1" * 40


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


def _arrays() -> dict[str, np.ndarray]:
    return {
        "vert2face": np.array([0, 1, 0, 1, 0, 1], dtype=np.int32),
        "vert2face_cnt": np.array([2, 2, 2, 0], dtype=np.int32),
        "vert2face_offset": np.array([0, 2, 4, 6], dtype=np.int32),
        "edges": np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int32),
        "edge2face_cnt": np.array([2, 2, 2], dtype=np.int32),
        "boundaries": np.empty((0,), dtype=np.int32),
        "vert_is_boundary": np.zeros((3,), dtype=np.uint8),
    }


def _write_cuda_structure(
    tmp_path: Path,
    input_sha256: str,
    *,
    route: str = "release-cumesh-simplify-structure",
) -> tuple[Path, Path]:
    arrays = _arrays()
    npz = tmp_path / "cuda_structure.npz"
    report = tmp_path / "cuda_structure.json"
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
                    "edge_readback": (
                        "uint64-little-endian-words-canonicalized-to-min-max"
                    ),
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


def test_witness_authenticates_injection_and_reopens_output(tmp_path):
    source_root = tmp_path / "mtlmesh"
    source_root.mkdir()
    input_ply = tmp_path / "input.ply"
    vertices, faces = _write_input(input_ply)
    cuda_report, cuda_npz = _write_cuda_structure(
        tmp_path,
        sha256_file(input_ply),
    )
    output_ply = tmp_path / "injected.ply"
    report_json = tmp_path / "report.json"
    injected = False

    def runner(actual_vertices, actual_faces, cuda_arrays, target_faces):
        nonlocal injected
        assert np.array_equal(actual_vertices, vertices)
        assert np.array_equal(actual_faces, faces)
        assert np.array_equal(cuda_arrays["vert2face"], _arrays()["vert2face"])
        assert target_faces == 1
        injected = True
        return vertices[:2], np.array([[0, 1, 0]], dtype=np.int32), {
            "first_step_reused_cuda_adjacency": True,
            "step_trace": [{"input_faces": 2, "output_faces": 1}],
        }

    report = run_witness(
        input_ply=input_ply,
        cuda_report_json=cuda_report,
        cuda_npz=cuda_npz,
        output_ply=output_ply,
        report_json=report_json,
        expected_input_sha256=sha256_file(input_ply),
        expected_cuda_report_sha256=sha256_file(cuda_report),
        expected_cuda_npz_sha256=sha256_file(cuda_npz),
        expected_source_root=source_root,
        expected_source_commit=SOURCE_COMMIT,
        target_faces=1,
        identity_probe=lambda: _identity(source_root),
        runner=runner,
    )

    assert injected is True
    assert report["status"] == "done"
    assert report["primary_output_status"] == "validated"
    assert report["effective_route"]["cuda_device_name"] == "Tesla T4"
    assert report["injection"]["first_step_reused_cuda_adjacency"] is True
    assert report["output_mesh"]["sha256"] == sha256_file(output_ply)
    reopened_vertices, reopened_faces = read_binary_ply(output_ply)
    assert np.array_equal(reopened_vertices, vertices[:2])
    assert np.array_equal(reopened_faces, np.array([[0, 1, 0]], dtype=np.int32))
    assert json.loads(report_json.read_text()) == report


def test_witness_rejects_wrong_cuda_route_without_output(tmp_path):
    source_root = tmp_path / "mtlmesh"
    source_root.mkdir()
    input_ply = tmp_path / "input.ply"
    _write_input(input_ply)
    cuda_report, cuda_npz = _write_cuda_structure(
        tmp_path,
        sha256_file(input_ply),
        route="not-cuda",
    )
    output_ply = tmp_path / "injected.ply"
    report_json = tmp_path / "report.json"

    with pytest.raises(WitnessError, match="effective geometry route mismatch"):
        run_witness(
            input_ply=input_ply,
            cuda_report_json=cuda_report,
            cuda_npz=cuda_npz,
            output_ply=output_ply,
            report_json=report_json,
            expected_input_sha256=sha256_file(input_ply),
            expected_cuda_report_sha256=sha256_file(cuda_report),
            expected_cuda_npz_sha256=sha256_file(cuda_npz),
            expected_source_root=source_root,
            expected_source_commit=SOURCE_COMMIT,
            target_faces=1,
            identity_probe=lambda: _identity(source_root),
            runner=lambda *args: pytest.fail("runner should not execute"),
        )

    report = json.loads(report_json.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "cuda_structure_validation"
    assert report["primary_output_status"] == "not_started"
    assert not output_ply.exists()


def test_witness_rejects_cuda_structure_from_other_input(tmp_path):
    source_root = tmp_path / "mtlmesh"
    source_root.mkdir()
    input_ply = tmp_path / "input.ply"
    _write_input(input_ply)
    cuda_report, cuda_npz = _write_cuda_structure(tmp_path, "f" * 64)

    with pytest.raises(WitnessError, match="input SHA256"):
        run_witness(
            input_ply=input_ply,
            cuda_report_json=cuda_report,
            cuda_npz=cuda_npz,
            output_ply=tmp_path / "injected.ply",
            report_json=tmp_path / "report.json",
            expected_input_sha256=sha256_file(input_ply),
            expected_cuda_report_sha256=sha256_file(cuda_report),
            expected_cuda_npz_sha256=sha256_file(cuda_npz),
            expected_source_root=source_root,
            expected_source_commit=SOURCE_COMMIT,
            target_faces=1,
            identity_probe=lambda: _identity(source_root),
            runner=lambda *args: pytest.fail("runner should not execute"),
        )
