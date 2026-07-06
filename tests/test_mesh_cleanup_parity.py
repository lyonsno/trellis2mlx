"""Reference mesh cleanup parity route and scalar witness contracts."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


SCRIPT = Path("scripts/mesh_cleanup_parity.py")


def _load_script_module():
    spec = importlib.util.spec_from_file_location("mesh_cleanup_parity_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_cleanup_contract_names_source_order_without_prefill():
    from trellmlx.mesh_cleanup_parity import REFERENCE_CLEANUP_CONTRACT

    assert REFERENCE_CLEANUP_CONTRACT["schema"] == "trellis2mlx.reference_cleanup_contract.v1"
    assert REFERENCE_CLEANUP_CONTRACT["postprocess_source"]["path"].endswith(
        "TRELLIS.2/o-voxel/o_voxel/postprocess.py"
    )
    assert REFERENCE_CLEANUP_CONTRACT["postprocess_source"]["line_range"] == [133, 162]
    assert REFERENCE_CLEANUP_CONTRACT["cumesh_simplify_source"]["line_range"] == [320, 355]
    assert REFERENCE_CLEANUP_CONTRACT["operations"] == [
        "simplify_coarse",
        "cleanup_initial",
        "simplify_final",
        "cleanup_final",
        "unify_face_orientations",
    ]
    assert REFERENCE_CLEANUP_CONTRACT["local_equivalent_operations"] == {
        "unify_face_orientations": "orient_faces_by_adjacency",
    }
    assert "initial_hole_fill" not in REFERENCE_CLEANUP_CONTRACT["operations"]
    assert REFERENCE_CLEANUP_CONTRACT["qem_status"] == "primitive_choice_only_not_reference_equivalent"


def test_mesh_scalars_count_components_boundary_edges_and_orientation_conflicts():
    from trellmlx.mesh_cleanup_parity import compute_mesh_cleanup_scalars

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    same_direction_faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    opposite_direction_faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)

    same_stats = compute_mesh_cleanup_scalars(vertices, same_direction_faces)
    opposite_stats = compute_mesh_cleanup_scalars(vertices, opposite_direction_faces)

    assert same_stats == {
        "vertex_count": 4,
        "face_count": 2,
        "component_count": 1,
        "boundary_edge_count": 4,
        "nonmanifold_edge_count": 0,
        "same_direction_shared_edge_count": 1,
    }
    assert opposite_stats["same_direction_shared_edge_count"] == 0


def test_cleanup_parity_report_records_route_identity_and_jsonable_scalars():
    from trellmlx.mesh_cleanup_parity import build_mesh_cleanup_parity_report

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    input_faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    output_faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)

    report = build_mesh_cleanup_parity_report(
        requested_route="reference-cleanup",
        effective_route="local-reference-cleanup:fast-simplification",
        input_vertices=vertices,
        input_faces=input_faces,
        output_vertices=vertices,
        output_faces=output_faces,
        operation_trace=[
            {"operation": "simplify_coarse", "requested_target_faces": 600_000, "output_faces": 590_000},
            {"operation": "cleanup_initial", "output_faces": 500_000},
        ],
        reference_backend={"status": "unavailable", "reason": "cumesh import failed"},
    )

    assert report["schema"] == "trellis2mlx.mesh_cleanup_parity_report.v1"
    assert report["requested_route"] == "reference-cleanup"
    assert report["effective_route"] == "local-reference-cleanup:fast-simplification"
    assert report["source_contract"]["operations"][0] == "simplify_coarse"
    assert report["reference_backend"] == {"status": "unavailable", "reason": "cumesh import failed"}
    assert report["input_scalars"]["same_direction_shared_edge_count"] == 1
    assert report["output_scalars"]["same_direction_shared_edge_count"] == 0
    assert report["operation_trace"][0]["output_faces"] == 590_000
    json.dumps(report)


def test_cleanup_parity_report_rejects_empty_route_identity():
    from trellmlx.mesh_cleanup_parity import build_mesh_cleanup_parity_report

    vertices = np.zeros((3, 3), dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    with pytest.raises(ValueError, match="requested_route"):
        build_mesh_cleanup_parity_report(
            requested_route="",
            effective_route="local-reference-cleanup",
            input_vertices=vertices,
            input_faces=faces,
            output_vertices=vertices,
            output_faces=faces,
            operation_trace=[],
            reference_backend={"status": "not_requested"},
        )


def test_mesh_cleanup_parity_harness_writes_fixture_report(tmp_path):
    harness = _load_script_module()
    report_path = tmp_path / "cleanup-parity.json"

    exit_code = harness.main([
        "--fixture",
        "two-triangle-sheet",
        "--target-faces",
        "1",
        "--report",
        str(report_path),
    ])

    assert exit_code == 0
    report = json.loads(report_path.read_text())
    assert report["schema"] == "trellis2mlx.mesh_cleanup_parity_report.v1"
    assert report["requested_route"] == "fixture:two-triangle-sheet"
    assert report["effective_route"] == "local-reference-cleanup:fast-simplification"
    assert report["asset"]["route"] == "fixture"
    assert report["asset"]["name"] == "two-triangle-sheet"
    assert report["source_contract"]["operations"][0] == "simplify_coarse"
    assert report["reference_backend"]["status"] in {"available", "unavailable", "not_requested"}
    assert "input_scalars" in report
    assert "output_scalars" in report
    assert isinstance(report["operation_trace"], list)


def test_mesh_cleanup_parity_harness_can_label_qem_probe_route(tmp_path):
    harness = _load_script_module()
    report_path = tmp_path / "cleanup-parity-qem-probe.json"

    exit_code = harness.main([
        "--fixture",
        "two-triangle-sheet",
        "--target-faces",
        "1",
        "--local-simplifier",
        "qem-probe",
        "--report",
        str(report_path),
    ])

    assert exit_code == 0
    report = json.loads(report_path.read_text())
    assert report["effective_route"] == "local-reference-cleanup:qem-probe"
    assert report["settings"]["local_simplifier"] == "qem-probe"
    assert report["source_contract"]["qem_probe_status"] == "probe_only_not_reference_equivalent"
    simplify_entries = [
        entry for entry in report["operation_trace"]
        if entry["operation"] in {"simplify_coarse", "simplify_final"}
    ]
    assert simplify_entries
    assert all("simplifier_step_trace" in entry for entry in simplify_entries)


def test_external_reference_cleanup_code_records_simplifier_step_trace():
    harness = _load_script_module()
    code = harness._external_reference_code()

    assert "simplify_step" in code
    assert "simplifier_step_trace" in code


def test_mesh_cleanup_parity_harness_keeps_fast_simplifier_default_route(tmp_path):
    harness = _load_script_module()
    report_path = tmp_path / "cleanup-parity-fast-default.json"

    exit_code = harness.main([
        "--fixture",
        "two-triangle-sheet",
        "--target-faces",
        "1",
        "--report",
        str(report_path),
    ])

    assert exit_code == 0
    report = json.loads(report_path.read_text())
    assert report["effective_route"] == "local-reference-cleanup:fast-simplification"
    assert report["settings"]["local_simplifier"] == "fast-simplification"


def test_mesh_cleanup_parity_harness_writes_failure_report_for_missing_raw_mesh(tmp_path):
    harness = _load_script_module()
    report_path = tmp_path / "cleanup-parity-failure.json"

    exit_code = harness.main([
        "--raw-mesh",
        str(tmp_path / "missing.npz"),
        "--report",
        str(report_path),
    ])

    assert exit_code == 1
    report = json.loads(report_path.read_text())
    assert report["schema"] == "trellis2mlx.mesh_cleanup_parity_report.v1"
    assert report["status"] == "failed"
    assert report["failure_phase"] == "load_mesh"
    assert report["requested_route"] == "raw-mesh"
    assert report["last_trustworthy_evidence"]["report_written"] is True


def test_mesh_cleanup_parity_script_runs_from_script_path(tmp_path):
    report_path = tmp_path / "cleanup-parity-subprocess.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--fixture",
            "two-triangle-sheet",
            "--target-faces",
            "1",
            "--report",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    report = json.loads(report_path.read_text())
    assert report["status"] == "ok"
    assert report["effective_route"] == "local-reference-cleanup:fast-simplification"


def test_mesh_cleanup_parity_required_reference_backend_fails_loud(tmp_path):
    harness = _load_script_module()
    report_path = tmp_path / "cleanup-parity-reference-failure.json"

    exit_code = harness.main([
        "--fixture",
        "two-triangle-sheet",
        "--target-faces",
        "1",
        "--reference-python",
        sys.executable,
        "--require-reference",
        "--report",
        str(report_path),
    ])

    assert exit_code == 1
    report = json.loads(report_path.read_text())
    assert report["schema"] == "trellis2mlx.mesh_cleanup_parity_report.v1"
    assert report["status"] == "failed"
    assert report["failure_phase"] == "reference_backend"
    assert report["requested_route"] == "fixture:two-triangle-sheet"
    assert report["effective_route"] == "local-reference-cleanup"
    assert "ModuleNotFoundError" in report["reference_backend"]["error"]
    assert ("cumesh" in report["reference_backend"]["error"]) or ("torch" in report["reference_backend"]["error"])


def test_mesh_cleanup_parity_report_can_compare_reference_scalars():
    from trellmlx.mesh_cleanup_parity import build_mesh_cleanup_parity_report

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    input_faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    local_faces = np.array([[0, 1, 2]], dtype=np.int64)
    reference_faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)

    report = build_mesh_cleanup_parity_report(
        requested_route="fixture:two-triangle-sheet",
        effective_route="local-reference-cleanup",
        input_vertices=vertices,
        input_faces=input_faces,
        output_vertices=vertices,
        output_faces=local_faces,
        operation_trace=[],
        reference_backend={"status": "available", "python": "/ref/python"},
        reference_vertices=vertices,
        reference_faces=reference_faces,
        reference_operation_trace=[],
    )

    assert report["reference_scalars"]["face_count"] == 2
    assert report["comparison"]["vertex_count_delta_local_minus_reference"] == 0
    assert report["comparison"]["face_count_delta_local_minus_reference"] == -1
    assert report["comparison"]["same_direction_shared_edge_delta_local_minus_reference"] == 0
