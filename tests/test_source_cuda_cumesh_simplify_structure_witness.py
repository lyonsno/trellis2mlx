import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.source_cuda_cumesh_postprocess_witness import (
    CUMESH_COMMIT,
    EXPECTED_CUDA_CAPABILITY,
    EXPECTED_CUDA_DEVICE_NAME,
    TRELLIS_COMMIT,
    TRELLIS_POSTPROCESS_SHA256,
    WitnessError,
    sha256_file,
    write_binary_ply,
)
from scripts.source_cuda_cumesh_simplify_structure_witness import run_witness


def _write_input(path: Path) -> None:
    write_binary_ply(
        path,
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        np.array([[0, 1, 2], [0, 2, 1]], dtype=np.int32),
    )


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(
        effective_route={
            "trellis_commit": TRELLIS_COMMIT,
            "trellis_source_clean": True,
            "trellis_postprocess_sha256": TRELLIS_POSTPROCESS_SHA256,
            "cumesh_commit": CUMESH_COMMIT,
            "cumesh_source_clean_before_build": True,
            "cuda_device_name": EXPECTED_CUDA_DEVICE_NAME,
            "cuda_capability": list(EXPECTED_CUDA_CAPABILITY),
            "device_type": "cuda",
        }
    )


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


def test_cuda_structure_witness_binds_t4_route_and_reopens_arrays(tmp_path):
    input_ply = tmp_path / "input.ply"
    output_npz = tmp_path / "trace.npz"
    output_json = tmp_path / "report.json"
    _write_input(input_ply)

    report = run_witness(
        input_ply=input_ply,
        output_npz=output_npz,
        output_json=output_json,
        expected_input_sha256=sha256_file(input_ply),
        work_dir=tmp_path / "runtime",
        runtime_factory=lambda **kwargs: _runtime(),
        collector=lambda runtime, vertices, faces: _arrays(),
    )

    assert report["status"] == "done"
    assert report["primary_output_status"] == "validated"
    assert report["effective_route"]["cumesh_commit"] == CUMESH_COMMIT
    assert report["effective_route"]["cuda_device_name"] == "Tesla T4"
    assert report["effective_route"]["input_sha256"] == sha256_file(input_ply)
    assert report["arrays"]["edges"]["shape"] == [3, 2]
    with np.load(output_npz, allow_pickle=False) as reopened:
        for name, expected in _arrays().items():
            assert np.array_equal(reopened[name], expected)
    assert json.loads(output_json.read_text()) == report


def test_cuda_structure_witness_rejects_non_t4_route_before_collection(tmp_path):
    input_ply = tmp_path / "input.ply"
    output_npz = tmp_path / "trace.npz"
    output_json = tmp_path / "report.json"
    _write_input(input_ply)
    runtime = _runtime()
    runtime.effective_route["cuda_device_name"] = "NVIDIA P100"
    collected = False

    def collector(actual_runtime, vertices, faces):
        nonlocal collected
        collected = True
        return _arrays()

    with pytest.raises(WitnessError, match="required Tesla T4"):
        run_witness(
            input_ply=input_ply,
            output_npz=output_npz,
            output_json=output_json,
            expected_input_sha256=sha256_file(input_ply),
            work_dir=tmp_path / "runtime",
            runtime_factory=lambda **kwargs: runtime,
            collector=collector,
        )

    report = json.loads(output_json.read_text())
    assert collected is False
    assert report["status"] == "failed"
    assert report["failure_phase"] == "runtime_validation"
    assert report["primary_output_status"] == "not_started"
    assert not output_npz.exists()


def test_cuda_structure_entrypoint_imports_from_flat_kaggle_capsule(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    for name in (
        "source_cuda_cumesh_simplify_structure_witness.py",
        "source_cuda_cumesh_postprocess_witness.py",
    ):
        shutil.copy2(repo_root / "scripts" / name, tmp_path / name)

    completed = subprocess.run(
        [
            sys.executable,
            str(tmp_path / "source_cuda_cumesh_simplify_structure_witness.py"),
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--expected-input-sha256" in completed.stdout
