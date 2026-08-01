import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _identity(module, root):
    return {
        "status": "available",
        "python": "/reference/python",
        "cumesh_file": None,
        "cumesh_path": [str(root / "cumesh")],
        "has_CuMesh": False,
        "metal_backend_file": str(root / "cumesh" / "metal_backend.py"),
        "has_MtlMesh": True,
        "git_root": str(root),
        "git_remote": "https://github.com/lyonsno/mtlmesh.git",
        "git_commit": module.EXPECTED_SOURCE_COMMIT,
        "git_status_porcelain": "",
    }


def _write_input(module, path):
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    module.write_binary_ply(path, vertices, faces)
    return vertices, faces


def _complete_postprocessor(module):
    def postprocess(vertices, faces, target_faces, *, stage_callback, **kwargs):
        current_faces = np.asarray(faces, dtype=np.int32)
        for operation, _ in module.STAGE_SPECS:
            stage_callback(
                operation,
                len(current_faces),
                len(current_faces),
                {},
                np.asarray(vertices, dtype=np.float32),
                current_faces,
            )
        return vertices, current_faces, [{"operation": "complete"}]

    return postprocess


def test_local_metal_witness_records_all_release_stages(tmp_path):
    module = _load_script(
        "source_metal_mtlmesh_postprocess_witness",
        "scripts/source_metal_mtlmesh_postprocess_witness.py",
    )
    input_ply = tmp_path / "input.ply"
    _write_input(module, input_ply)
    source_root = tmp_path / "mtlmesh"

    report = module.run_witness(
        input_ply=input_ply,
        output_dir=tmp_path / "stages",
        report_json=tmp_path / "report.json",
        expected_input_sha256=module.sha256_file(input_ply),
        target_faces=10,
        expected_source_root=source_root,
        identity_probe=lambda: _identity(module, source_root),
        postprocessor=_complete_postprocessor(module),
    )

    assert report["status"] == "done"
    assert report["primary_output_status"] == "validated"
    assert report["effective_route"]["geometry_route"] == "metal-mtlmesh-standard-non-remesh"
    assert [item["operation"] for item in report["stage_artifacts"]] == [
        operation for operation, _ in module.STAGE_SPECS
    ]
    assert all(item["status"] == "validated" for item in report["stage_artifacts"])


def test_local_metal_witness_rejects_wrong_source_commit_with_report(tmp_path):
    module = _load_script(
        "source_metal_mtlmesh_postprocess_witness_wrong_route",
        "scripts/source_metal_mtlmesh_postprocess_witness.py",
    )
    input_ply = tmp_path / "input.ply"
    _write_input(module, input_ply)
    source_root = tmp_path / "mtlmesh"
    identity = _identity(module, source_root)
    identity["git_commit"] = "0" * 40
    report_json = tmp_path / "report.json"

    with pytest.raises(module.WitnessError, match="source commit"):
        module.run_witness(
            input_ply=input_ply,
            output_dir=tmp_path / "stages",
            report_json=report_json,
            expected_input_sha256=module.sha256_file(input_ply),
            target_faces=10,
            expected_source_root=source_root,
            identity_probe=lambda: identity,
            postprocessor=_complete_postprocessor(module),
        )

    report = json.loads(report_json.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "runtime_validation"
    assert report["primary_output_status"] == "not_started"


def test_local_metal_witness_binds_caller_selected_source_commit(tmp_path):
    module = _load_script(
        "source_metal_mtlmesh_postprocess_witness_selected_commit",
        "scripts/source_metal_mtlmesh_postprocess_witness.py",
    )
    input_ply = tmp_path / "input.ply"
    _write_input(module, input_ply)
    source_root = tmp_path / "mtlmesh"
    expected_source_commit = "1" * 40
    identity = _identity(module, source_root)
    identity["git_commit"] = expected_source_commit

    report = module.run_witness(
        input_ply=input_ply,
        output_dir=tmp_path / "stages",
        report_json=tmp_path / "report.json",
        expected_input_sha256=module.sha256_file(input_ply),
        target_faces=10,
        expected_source_root=source_root,
        expected_source_commit=expected_source_commit,
        identity_probe=lambda: identity,
        postprocessor=_complete_postprocessor(module),
    )

    assert (
        report["requested_route"]["expected_source_commit"]
        == expected_source_commit
    )
    assert report["effective_route"]["git_commit"] == expected_source_commit


def test_local_metal_witness_cannot_close_on_partial_stage_set(tmp_path):
    module = _load_script(
        "source_metal_mtlmesh_postprocess_witness_partial",
        "scripts/source_metal_mtlmesh_postprocess_witness.py",
    )
    input_ply = tmp_path / "input.ply"
    vertices, faces = _write_input(module, input_ply)
    source_root = tmp_path / "mtlmesh"

    def partial_postprocessor(*args, stage_callback, **kwargs):
        operation = module.STAGE_SPECS[0][0]
        stage_callback(operation, len(faces), len(faces), {}, vertices, faces)
        return vertices, faces, [{"operation": "partial"}]

    with pytest.raises(module.WitnessError, match="stage set"):
        module.run_witness(
            input_ply=input_ply,
            output_dir=tmp_path / "stages",
            report_json=tmp_path / "report.json",
            expected_input_sha256=module.sha256_file(input_ply),
            target_faces=10,
            expected_source_root=source_root,
            identity_probe=lambda: _identity(module, source_root),
            postprocessor=partial_postprocessor,
        )

    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["primary_output_status"] == "partial"
    assert report["last_trustworthy_phase"] == "stage_validated:prefill_holes"


def test_local_metal_witness_rejects_input_hash_before_runtime(tmp_path):
    module = _load_script(
        "source_metal_mtlmesh_postprocess_witness_input",
        "scripts/source_metal_mtlmesh_postprocess_witness.py",
    )
    input_ply = tmp_path / "input.ply"
    _write_input(module, input_ply)
    runtime_called = False

    def identity_probe():
        nonlocal runtime_called
        runtime_called = True
        return {}

    with pytest.raises(module.WitnessError, match="input SHA256 mismatch"):
        module.run_witness(
            input_ply=input_ply,
            output_dir=tmp_path / "stages",
            report_json=tmp_path / "report.json",
            expected_input_sha256="0" * 64,
            target_faces=10,
            expected_source_root=tmp_path / "mtlmesh",
            identity_probe=identity_probe,
            postprocessor=_complete_postprocessor(module),
        )

    assert runtime_called is False
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["failure_phase"] == "input_validation"
