import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.source_cuda_cumesh_first_simplify_step_witness import (
    INSTRUMENTATION_SCHEMA,
    INSTRUMENTED_FILES,
    _default_collector,
    run_witness,
)
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


def _write_input(path: Path) -> None:
    write_binary_ply(
        path,
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        np.array([[0, 1, 2], [0, 2, 1]], dtype=np.int32),
    )


def _runtime(patch_sha256: str) -> SimpleNamespace:
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
            "cumesh_instrumentation": {
                "schema": INSTRUMENTATION_SCHEMA,
                "patch_sha256": patch_sha256,
                "changed_files": list(INSTRUMENTED_FILES),
            },
        }
    )


def _arrays() -> dict[str, np.ndarray]:
    return {
        "vert2face": np.array([0, 1, 0, 1, 0, 1], dtype=np.int32),
        "post_vertices": np.array(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        "post_faces": np.array([[0, 1, 0]], dtype=np.int32),
    }


def test_first_step_collector_reuses_the_adjacency_it_captures():
    class Tensor:
        def __init__(self, array):
            self.array = np.asarray(array)

        def cuda(self):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.array

    class Mesh:
        def __init__(self):
            self.cu_mesh = self
            self.simplify_args = None

        def init(self, vertices, faces):
            pass

        def get_vertex_face_adjacency(self):
            pass

        def read_all_cache(self):
            return {
                "vert2face": Tensor(
                    np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
                )
            }

        def simplify_step(self, *args):
            self.simplify_args = args

        def read(self):
            arrays = _arrays()
            return Tensor(arrays["post_vertices"]), Tensor(arrays["post_faces"])

    mesh = Mesh()
    runtime = SimpleNamespace(
        torch=SimpleNamespace(from_numpy=Tensor),
        cumesh=SimpleNamespace(CuMesh=lambda: mesh),
    )
    vertices = np.zeros((3, 3), dtype=np.float32)
    faces = np.zeros((2, 3), dtype=np.int32)

    _default_collector(runtime, vertices, faces)

    assert mesh.simplify_args[-1] is True


def test_first_step_witness_binds_route_and_reopens_arrays(tmp_path):
    input_ply = tmp_path / "input.ply"
    output_npz = tmp_path / "first-step.npz"
    output_json = tmp_path / "report.json"
    patch = tmp_path / "reuse-adjacency.patch"
    patch.write_text("fixture patch")
    patch_sha256 = sha256_file(patch)
    _write_input(input_ply)

    report = run_witness(
        input_ply=input_ply,
        instrumentation_patch=patch,
        output_npz=output_npz,
        output_json=output_json,
        expected_input_sha256=sha256_file(input_ply),
        expected_patch_sha256=patch_sha256,
        work_dir=tmp_path / "runtime",
        runtime_factory=lambda **kwargs: _runtime(patch_sha256),
        collector=lambda runtime, vertices, faces: _arrays(),
    )

    assert report["status"] == "done"
    assert report["primary_output_status"] == "validated"
    assert report["requested_route"]["target_faces"] == 1
    assert report["effective_route"]["cuda_device_name"] == "Tesla T4"
    assert report["effective_route"]["threshold"] == 1e-8
    assert report["output_mesh"] == {"vertices": 2, "faces": 1}
    with np.load(output_npz, allow_pickle=False) as reopened:
        for name, expected in _arrays().items():
            assert np.array_equal(reopened[name], expected)
    assert json.loads(output_json.read_text()) == report


def test_first_step_witness_rejects_non_t4_before_collection(tmp_path):
    input_ply = tmp_path / "input.ply"
    output_npz = tmp_path / "first-step.npz"
    output_json = tmp_path / "report.json"
    patch = tmp_path / "reuse-adjacency.patch"
    patch.write_text("fixture patch")
    patch_sha256 = sha256_file(patch)
    _write_input(input_ply)
    output_npz.write_bytes(b"stale output that must not survive")
    runtime = _runtime(patch_sha256)
    runtime.effective_route["cuda_device_name"] = "NVIDIA P100"
    collected = False

    def collector(actual_runtime, vertices, faces):
        nonlocal collected
        collected = True
        return _arrays()

    with pytest.raises(WitnessError, match="required Tesla T4"):
        run_witness(
            input_ply=input_ply,
            instrumentation_patch=patch,
            output_npz=output_npz,
            output_json=output_json,
            expected_input_sha256=sha256_file(input_ply),
            expected_patch_sha256=patch_sha256,
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


def test_first_step_entrypoint_imports_from_flat_kaggle_capsule(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    for name in (
        "source_cuda_cumesh_first_simplify_step_witness.py",
        "source_cuda_cumesh_postprocess_witness.py",
    ):
        shutil.copy2(repo_root / "scripts" / name, tmp_path / name)

    completed = subprocess.run(
        [
            sys.executable,
            str(tmp_path / "source_cuda_cumesh_first_simplify_step_witness.py"),
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--expected-input-sha256" in completed.stdout
