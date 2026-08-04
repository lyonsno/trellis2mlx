import importlib.util
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


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "source_cuda_cumesh_orientation_repeats_witness.py"
    )
    assert path.is_file(), "orientation-only CUDA witness entrypoint is missing"
    spec = importlib.util.spec_from_file_location(
        "source_cuda_cumesh_orientation_repeats_witness",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_input(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    write_binary_ply(path, vertices, faces)
    return vertices, faces


def _runtime(*, device: str = EXPECTED_CUDA_DEVICE_NAME):
    return SimpleNamespace(
        effective_route={
            "trellis_commit": TRELLIS_COMMIT,
            "trellis_source_clean": True,
            "trellis_postprocess_sha256": TRELLIS_POSTPROCESS_SHA256,
            "cumesh_commit": CUMESH_COMMIT,
            "cumesh_source_clean_before_build": True,
            "cuda_device_name": device,
            "cuda_capability": list(EXPECTED_CUDA_CAPABILITY),
            "device_type": "cuda",
        }
    )


def _samples(vertices: np.ndarray, faces: np.ndarray):
    return [
        {
            "vertices": vertices.copy(),
            "faces": faces.copy(),
            "elapsed_seconds": 0.01,
        },
        {
            "vertices": vertices.copy(),
            "faces": faces[:, [0, 2, 1]].copy(),
            "elapsed_seconds": 0.02,
        },
        {
            "vertices": vertices.copy(),
            "faces": faces.copy(),
            "elapsed_seconds": 0.03,
        },
    ]


def test_orientation_repeat_witness_preserves_every_assignment(tmp_path):
    module = _module()
    input_ply = tmp_path / "stage11.ply"
    output_npz = tmp_path / "cuda-orientations.npz"
    output_json = tmp_path / "report.json"
    vertices, faces = _write_input(input_ply)

    report = module.run_witness(
        input_ply=input_ply,
        output_npz=output_npz,
        output_json=output_json,
        expected_input_sha256=sha256_file(input_ply),
        repeats=3,
        work_dir=tmp_path / "runtime",
        runtime_factory=lambda **kwargs: _runtime(),
        collector=lambda runtime, actual_vertices, actual_faces, repeats: _samples(
            actual_vertices,
            actual_faces,
        ),
    )

    assert report["status"] == "done"
    assert report["primary_output_status"] == "validated"
    assert report["requested_route"]["repeats"] == 3
    assert report["effective_route"]["repeats"] == 3
    assert report["repeat_count"] == 3
    assert report["pairwise"]["repeat_00_vs_repeat_01"]["reversed"] == 2
    assert report["pairwise"]["repeat_00_vs_repeat_02"]["reversed"] == 0
    with np.load(output_npz, allow_pickle=False) as reopened:
        assert np.array_equal(reopened["vertices"], vertices)
        assert np.array_equal(reopened["input_faces"], faces)
        assert np.array_equal(reopened["repeat_00_faces"], faces)
        assert np.array_equal(reopened["repeat_01_faces"], faces[:, [0, 2, 1]])
        assert np.array_equal(reopened["repeat_02_faces"], faces)
    assert json.loads(output_json.read_text()) == report


def test_orientation_repeat_witness_rejects_non_t4_before_collection(tmp_path):
    module = _module()
    input_ply = tmp_path / "stage11.ply"
    output_npz = tmp_path / "cuda-orientations.npz"
    output_json = tmp_path / "report.json"
    _write_input(input_ply)
    output_npz.write_bytes(b"stale output")
    collected = False

    def collect(*args):
        nonlocal collected
        collected = True
        return []

    with pytest.raises(WitnessError, match="required Tesla T4"):
        module.run_witness(
            input_ply=input_ply,
            output_npz=output_npz,
            output_json=output_json,
            expected_input_sha256=sha256_file(input_ply),
            repeats=3,
            work_dir=tmp_path / "runtime",
            runtime_factory=lambda **kwargs: _runtime(device="NVIDIA P100"),
            collector=collect,
        )

    report = json.loads(output_json.read_text())
    assert collected is False
    assert report["failure_phase"] == "runtime_validation"
    assert report["primary_output_status"] == "not_started"
    assert not output_npz.exists()


def test_orientation_repeat_witness_rejects_geometry_drift(tmp_path):
    module = _module()
    input_ply = tmp_path / "stage11.ply"
    output_npz = tmp_path / "cuda-orientations.npz"
    output_json = tmp_path / "report.json"
    vertices, faces = _write_input(input_ply)
    bad_faces = faces.copy()
    bad_faces[0] = [0, 1, 3]

    with pytest.raises(WitnessError, match="changed triangle membership"):
        module.run_witness(
            input_ply=input_ply,
            output_npz=output_npz,
            output_json=output_json,
            expected_input_sha256=sha256_file(input_ply),
            repeats=1,
            work_dir=tmp_path / "runtime",
            runtime_factory=lambda **kwargs: _runtime(),
            collector=lambda *args: [
                {
                    "vertices": vertices,
                    "faces": bad_faces,
                    "elapsed_seconds": 0.01,
                }
            ],
        )

    report = json.loads(output_json.read_text())
    assert report["failure_phase"] == "collection_validation"
    assert report["primary_output_status"] == "not_started"
    assert not output_npz.exists()


def test_orientation_repeat_witness_rejects_wrong_input_digest_before_runtime(
    tmp_path,
):
    module = _module()
    input_ply = tmp_path / "stage11.ply"
    output_json = tmp_path / "report.json"
    _write_input(input_ply)
    runtime_started = False

    def build_runtime(**kwargs):
        nonlocal runtime_started
        runtime_started = True
        return _runtime()

    with pytest.raises(WitnessError, match="input SHA256 mismatch"):
        module.run_witness(
            input_ply=input_ply,
            output_npz=tmp_path / "out.npz",
            output_json=output_json,
            expected_input_sha256="0" * 64,
            repeats=3,
            work_dir=tmp_path / "runtime",
            runtime_factory=build_runtime,
        )

    assert runtime_started is False
    report = json.loads(output_json.read_text())
    assert report["failure_phase"] == "input_validation"


def test_orientation_repeat_witness_rejects_temporary_output_aliasing_input(
    tmp_path,
):
    module = _module()
    output_npz = tmp_path / "cuda-orientations.npz"
    input_ply = tmp_path / "cuda-orientations.npz.tmp"
    output_json = tmp_path / "report.json"
    _write_input(input_ply)
    input_sha256 = sha256_file(input_ply)
    runtime_started = False

    def build_runtime(**kwargs):
        nonlocal runtime_started
        runtime_started = True
        return _runtime()

    with pytest.raises(WitnessError, match="temporary output aliases protected input"):
        module.run_witness(
            input_ply=input_ply,
            output_npz=output_npz,
            output_json=output_json,
            expected_input_sha256=input_sha256,
            repeats=1,
            work_dir=tmp_path / "runtime",
            runtime_factory=build_runtime,
        )

    assert runtime_started is False
    assert sha256_file(input_ply) == input_sha256
    assert not output_npz.exists()
    report = json.loads(output_json.read_text())
    assert report["failure_phase"] == "request_validation"
    assert report["primary_output_status"] == "not_started"


def test_orientation_repeat_witness_preserves_directly_aliased_input(tmp_path):
    module = _module()
    input_ply = tmp_path / "stage11.ply"
    output_json = tmp_path / "report.json"
    _write_input(input_ply)
    input_sha256 = sha256_file(input_ply)

    with pytest.raises(WitnessError, match="output NPZ aliases protected input"):
        module.run_witness(
            input_ply=input_ply,
            output_npz=input_ply,
            output_json=output_json,
            expected_input_sha256=input_sha256,
            repeats=1,
            work_dir=tmp_path / "runtime",
            runtime_factory=lambda **kwargs: _runtime(),
        )

    assert sha256_file(input_ply) == input_sha256
    report = json.loads(output_json.read_text())
    assert report["failure_phase"] == "request_validation"
    assert report["primary_output_status"] == "not_started"


def test_orientation_repeat_witness_reports_stale_output_directory(tmp_path):
    module = _module()
    input_ply = tmp_path / "stage11.ply"
    output_npz = tmp_path / "cuda-orientations.npz"
    output_json = tmp_path / "report.json"
    _write_input(input_ply)
    output_npz.mkdir()

    with pytest.raises(WitnessError, match="stale output NPZ is not a file"):
        module.run_witness(
            input_ply=input_ply,
            output_npz=output_npz,
            output_json=output_json,
            expected_input_sha256=sha256_file(input_ply),
            repeats=1,
            work_dir=tmp_path / "runtime",
            runtime_factory=lambda **kwargs: _runtime(),
        )

    assert output_npz.is_dir()
    report = json.loads(output_json.read_text())
    assert report["failure_phase"] == "stale_output_cleanup"
    assert report["last_trustworthy_phase"] == "input_validated"
    assert report["primary_output_status"] == "not_started"


def test_orientation_repeat_witness_rejects_report_temp_aliasing_input(
    tmp_path,
):
    module = _module()
    output_json = tmp_path / "report.json"
    input_ply = tmp_path / "report.json.tmp"
    output_npz = tmp_path / "cuda-orientations.npz"
    _write_input(input_ply)
    input_sha256 = sha256_file(input_ply)
    runtime_started = False

    def build_runtime(**kwargs):
        nonlocal runtime_started
        runtime_started = True
        return _runtime()

    with pytest.raises(WitnessError, match="report temporary output aliases protected input"):
        module.run_witness(
            input_ply=input_ply,
            output_npz=output_npz,
            output_json=output_json,
            expected_input_sha256=input_sha256,
            repeats=1,
            work_dir=tmp_path / "runtime",
            runtime_factory=build_runtime,
        )

    assert runtime_started is False
    assert sha256_file(input_ply) == input_sha256
    assert not output_json.exists()
    failure_report = tmp_path / "report.json.failure.json"
    report = json.loads(failure_report.read_text())
    assert report["failure_phase"] == "request_validation"
    assert report["effective_output_json"] == str(failure_report)
    assert report["report_rerouted"] is True


def test_orientation_repeat_witness_preserves_primary_aliased_by_report_temp(
    tmp_path,
):
    module = _module()
    input_ply = tmp_path / "stage11.ply"
    output_json = tmp_path / "report.json"
    output_npz = tmp_path / "report.json.tmp"
    _write_input(input_ply)
    stale_primary = b"caller-owned stale primary"
    output_npz.write_bytes(stale_primary)

    with pytest.raises(WitnessError, match="report temporary output aliases output NPZ"):
        module.run_witness(
            input_ply=input_ply,
            output_npz=output_npz,
            output_json=output_json,
            expected_input_sha256=sha256_file(input_ply),
            repeats=1,
            work_dir=tmp_path / "runtime",
            runtime_factory=lambda **kwargs: _runtime(),
        )

    assert output_npz.read_bytes() == stale_primary
    assert not output_json.exists()
    failure_report = tmp_path / "report.json.failure.json"
    report = json.loads(failure_report.read_text())
    assert report["failure_phase"] == "request_validation"
    assert report["primary_output_status"] == "not_started"


def test_orientation_repeat_witness_preserves_report_aliased_by_npz_temp(
    tmp_path,
):
    module = _module()
    input_ply = tmp_path / "stage11.ply"
    output_npz = tmp_path / "cuda-orientations.npz"
    output_json = tmp_path / "cuda-orientations.npz.tmp"
    _write_input(input_ply)
    caller_report = b"caller-owned requested report"
    output_json.write_bytes(caller_report)
    runtime_started = False

    def build_runtime(**kwargs):
        nonlocal runtime_started
        runtime_started = True
        return _runtime()

    with pytest.raises(WitnessError, match="temporary output aliases requested report"):
        module.run_witness(
            input_ply=input_ply,
            output_npz=output_npz,
            output_json=output_json,
            expected_input_sha256=sha256_file(input_ply),
            repeats=1,
            work_dir=tmp_path / "runtime",
            runtime_factory=build_runtime,
        )

    assert runtime_started is False
    assert output_json.read_bytes() == caller_report
    assert not output_npz.exists()
    failure_report = tmp_path / "cuda-orientations.npz.tmp.failure.json"
    report = json.loads(failure_report.read_text())
    assert report["failure_phase"] == "request_validation"
    assert report["effective_output_json"] == str(failure_report)
    assert report["report_rerouted"] is True


def test_orientation_repeat_entrypoint_imports_from_flat_kaggle_capsule(
    tmp_path,
):
    module = _module()
    repo_root = Path(__file__).resolve().parents[1]
    for name in (
        "source_cuda_cumesh_orientation_repeats_witness.py",
        "source_cuda_cumesh_postprocess_witness.py",
    ):
        shutil.copy2(repo_root / "scripts" / name, tmp_path / name)

    completed = subprocess.run(
        [
            sys.executable,
            str(tmp_path / "source_cuda_cumesh_orientation_repeats_witness.py"),
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--expected-input-sha256" in completed.stdout
    assert "--repeats" in completed.stdout
    assert module.SCHEMA == "trellis2mlx.source_cuda_cumesh_orientation_repeats.v1"
