import json
from pathlib import Path

import numpy as np
import pytest

from scripts.source_cuda_cumesh_postprocess_witness import (
    WitnessError,
    sha256_file,
    write_binary_ply,
)
from scripts.source_metal_mtlmesh_simplify_structure_witness import run_witness


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


def _write_input(path: Path) -> None:
    write_binary_ply(
        path,
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        np.array([[0, 1, 2], [0, 2, 1]], dtype=np.int32),
    )


def test_structure_witness_records_effective_route_and_reopens_arrays(tmp_path):
    source_root = tmp_path / "mtlmesh"
    source_root.mkdir()
    input_ply = tmp_path / "input.ply"
    output_npz = tmp_path / "trace.npz"
    report_json = tmp_path / "report.json"
    _write_input(input_ply)

    report = run_witness(
        input_ply=input_ply,
        output_npz=output_npz,
        report_json=report_json,
        expected_input_sha256=sha256_file(input_ply),
        expected_source_root=source_root,
        expected_source_commit=SOURCE_COMMIT,
        identity_probe=lambda: _identity(source_root),
        collector=lambda actual_vertices, actual_faces: _arrays(),
    )

    assert report["status"] == "done"
    assert report["primary_output_status"] == "validated"
    assert report["effective_route"]["git_commit"] == SOURCE_COMMIT
    assert report["effective_route"]["input_ply"] == str(input_ply)
    assert report["effective_route"]["geometry_route"] == "metal-mtlmesh-simplify-structure"
    assert report["arrays"]["vert2face"]["shape"] == [6]
    assert report["arrays"]["boundaries"]["shape"] == [0]
    with np.load(output_npz, allow_pickle=False) as reopened:
        assert set(reopened.files) == set(_arrays())
        for name, expected in _arrays().items():
            assert np.array_equal(reopened[name], expected)
    assert json.loads(report_json.read_text()) == report


def test_structure_witness_rejects_substituted_source_commit_before_collection(
    tmp_path,
):
    source_root = tmp_path / "mtlmesh"
    source_root.mkdir()
    input_ply = tmp_path / "input.ply"
    output_npz = tmp_path / "trace.npz"
    report_json = tmp_path / "report.json"
    _write_input(input_ply)
    collected = False

    def collector(vertices, faces):
        nonlocal collected
        collected = True
        return _arrays()

    with pytest.raises(WitnessError, match="source commit mismatch"):
        run_witness(
            input_ply=input_ply,
            output_npz=output_npz,
            report_json=report_json,
            expected_input_sha256=sha256_file(input_ply),
            expected_source_root=source_root,
            expected_source_commit=SOURCE_COMMIT,
            identity_probe=lambda: {
                **_identity(source_root),
                "git_commit": "2" * 40,
            },
            collector=collector,
        )

    report = json.loads(report_json.read_text())
    assert collected is False
    assert report["status"] == "failed"
    assert report["failure_phase"] == "runtime_validation"
    assert report["primary_output_status"] == "not_started"
    assert not output_npz.exists()


def test_structure_witness_rejects_partial_cache_without_primary_output(tmp_path):
    source_root = tmp_path / "mtlmesh"
    source_root.mkdir()
    input_ply = tmp_path / "input.ply"
    output_npz = tmp_path / "trace.npz"
    report_json = tmp_path / "report.json"
    _write_input(input_ply)
    partial = _arrays()
    partial.pop("vert2face")

    with pytest.raises(WitnessError, match="collector array set mismatch"):
        run_witness(
            input_ply=input_ply,
            output_npz=output_npz,
            report_json=report_json,
            expected_input_sha256=sha256_file(input_ply),
            expected_source_root=source_root,
            expected_source_commit=SOURCE_COMMIT,
            identity_probe=lambda: _identity(source_root),
            collector=lambda vertices, faces: partial,
        )

    report = json.loads(report_json.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "structure_collection"
    assert report["primary_output_status"] == "not_started"
    assert not output_npz.exists()
