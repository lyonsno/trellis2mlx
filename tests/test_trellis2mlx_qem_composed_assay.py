from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from trellmlx.glb_aabb_crop import open_triangle_glb, write_geometry_glb


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_trellis2mlx_qem_composed_assay.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("qem_composed_assay", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_grid(path: Path, size: int = 8) -> None:
    x, y = np.meshgrid(
        np.linspace(0.0, 1.0, size, dtype=np.float32),
        np.linspace(0.0, 1.0, size, dtype=np.float32),
    )
    vertices = np.stack([x.ravel(), y.ravel(), np.zeros(size * size)], axis=1)
    faces = []
    for row in range(size - 1):
        for column in range(size - 1):
            a = row * size + column
            b = a + 1
            c = a + size
            d = c + 1
            faces.extend([[a, b, d], [a, d, c]])
    write_geometry_glb(path, vertices, np.asarray(faces, dtype=np.uint32))


def _fake_runtime(*, fail_stage: str | None = None, target_satisfied: bool = True):
    calls: list[tuple[str, dict]] = []

    def cleanup(vertices, faces, **kwargs):
        stage = "final_cleanup" if kwargs["do_fix_normals"] else "intermediate_cleanup"
        calls.append((stage, kwargs))
        if fail_stage == stage:
            raise RuntimeError(f"forced {stage} failure")
        return vertices.copy(), faces.copy()

    def simplify(vertices, faces, target_faces, **kwargs):
        calls.append(("local_qem", {"target_faces": target_faces, **kwargs}))
        if fail_stage == "local_qem":
            raise RuntimeError("forced local_qem failure")
        achieved = target_faces if target_satisfied else target_faces + 5
        out_faces = faces[:achieved].copy()
        receipt = {
            "route": "trellis2mlx-sequential-qem-v1",
            "scheduler": "sequential-mutating-conflict-check",
            "source_faces": len(faces),
            "requested_target_faces": target_faces,
            "achieved_faces": len(out_faces),
            "target_satisfied": target_satisfied,
            "termination_reason": (
                "target_reached" if target_satisfied else "max_iterations_reached"
            ),
            "iterations": 7,
            "max_iterations": kwargs["max_iterations"],
            "zero_removal_iterations": 0,
            "final_iteration_removed_faces": 2,
            "initial_threshold": 1e-8,
            "final_executed_threshold": 1e-6,
            "next_scheduled_threshold": 1e-5,
        }
        return vertices.copy(), out_faces, receipt

    identity = {
        "mlx_available": True,
        "mlx_version": "test-mlx",
        "qem_module_path": "/test/simplify_qem_metal.py",
        "qem_module_sha256": "a" * 64,
        "cleanup_module_path": "/test/mesh_cleanup.py",
        "cleanup_module_sha256": "b" * 64,
    }
    return SimpleNamespace(cleanup=cleanup, simplify=simplify, identity=identity), calls


def _invoke(
    runner,
    monkeypatch,
    tmp_path: Path,
    *,
    runtime=None,
    expected_input_sha256: str | None = None,
    expected_lineage_sha256: str | None = None,
    report: Path | None = None,
    output_dir: Path | None = None,
):
    source = tmp_path / "plateau.glb"
    lineage = tmp_path / "fast-prefix-report.json"
    output_dir = output_dir or (tmp_path / "outputs")
    report = report or (tmp_path / "assay.report.json")
    _write_grid(source)
    lineage.write_text('{"status":"completed","route":{"id":"fast"}}\n')
    runtime = runtime or _fake_runtime()[0]
    monkeypatch.setattr(runner, "_load_runtime", lambda: runtime)
    returncode = runner.main(
        [
            "--input", str(source),
            "--expected-input-sha256", expected_input_sha256 or _sha256(source),
            "--output-dir", str(output_dir),
            "--report", str(report),
            "--target-faces", "40",
            "--max-iterations", "23",
            "--continuation-classification", "test-authenticated-parent-consumer",
            "--expected-mlx-version", "test-mlx",
            "--expected-qem-module-sha256", "a" * 64,
            "--expected-cleanup-module-sha256", "b" * 64,
            "--lineage-ref", "fast-prefix-report", str(lineage),
            expected_lineage_sha256 or _sha256(lineage),
        ]
    )
    return returncode, source, lineage, output_dir, report


def test_composed_assay_binds_exact_production_stage_contract(tmp_path, monkeypatch):
    runner = _load_runner()
    runtime, calls = _fake_runtime()

    returncode, source, lineage, output_dir, report = _invoke(
        runner, monkeypatch, tmp_path, runtime=runtime
    )

    assert returncode == 0
    data = json.loads(report.read_text())
    assert data["status"] == "completed"
    assert data["route"]["id"] == "trellis2mlx-cleanup-local-qem-cleanup-v1"
    assert data["route"]["harness_sha256"] == _sha256(SCRIPT)
    assert data["route"]["runtime"] == runtime.identity
    assert data["source"]["sha256"] == _sha256(source)
    assert data["continuation_provenance"]["refs"][0]["sha256"] == _sha256(lineage)
    assert (
        data["continuation_provenance"]["classification"]
        == "test-authenticated-parent-consumer"
    )
    assert [name for name, _ in calls] == [
        "intermediate_cleanup", "local_qem", "final_cleanup"
    ]
    for stage, config in (calls[0], calls[2]):
        assert config == {
            "max_hole_perimeter": 3e-2,
            "keep_largest": False,
            "min_component_area": 1e-5,
            "do_fix_normals": stage == "final_cleanup",
            "verbose": False,
        }
    assert calls[1][1] == {
        "target_faces": 40,
        "lambda_edge_length": 1e-2,
        "lambda_skinny": 1e-3,
        "initial_thresh": 1e-8,
        "max_iterations": 23,
        "return_receipt": True,
        "verbose": True,
    }
    assert data["target_contract"]["qem_target_satisfied"] is True
    assert data["target_contract"]["final_face_budget_satisfied"] is True
    assert data["target_contract"]["termination_reason"] == "target_reached"
    assert data["stages"]["local_qem"]["receipt"]["iterations"] == 7
    assert data["last_trustworthy_evidence"]["stage"] == "final_cleanup"
    assert sorted(path.name for path in output_dir.glob("*.glb")) == [
        "01-intermediate-cleanup.glb",
        "02-local-qem.glb",
        "03-final-cleanup.glb",
    ]


def test_cap_exit_completes_but_does_not_claim_target_satisfaction(tmp_path, monkeypatch):
    runner = _load_runner()
    runtime, _ = _fake_runtime(target_satisfied=False)

    returncode, _, _, _, report = _invoke(runner, monkeypatch, tmp_path, runtime=runtime)

    assert returncode == 0
    data = json.loads(report.read_text())
    assert data["status"] == "completed"
    assert data["target_contract"] == {
        "requested_faces": 40,
        "achieved_faces_after_qem": 45,
        "achieved_faces_after_final_cleanup": 45,
        "qem_target_satisfied": False,
        "final_face_budget_satisfied": False,
        "termination_reason": "max_iterations_reached",
    }
    assert data["claim_ceiling"] == "pipeline-executed-target-unsatisfied"


def test_final_cleanup_budget_growth_is_distinct_from_qem_target_satisfaction(
    tmp_path, monkeypatch
):
    runner = _load_runner()
    runtime, _ = _fake_runtime(target_satisfied=True)
    original_cleanup = runtime.cleanup

    def cleanup_with_final_growth(vertices, faces, **kwargs):
        out_vertices, out_faces = original_cleanup(vertices, faces, **kwargs)
        if kwargs["do_fix_normals"]:
            out_faces = np.concatenate([out_faces, out_faces[:1]], axis=0)
        return out_vertices, out_faces

    runtime.cleanup = cleanup_with_final_growth

    returncode, _, _, _, report = _invoke(
        runner, monkeypatch, tmp_path, runtime=runtime
    )

    assert returncode == 0
    data = json.loads(report.read_text())
    assert data["target_contract"]["qem_target_satisfied"] is True
    assert data["target_contract"]["final_face_budget_satisfied"] is False
    assert data["target_contract"]["achieved_faces_after_final_cleanup"] == 41
    assert data["claim_ceiling"] == "pipeline-executed-target-unsatisfied"


def test_wrong_input_digest_fails_before_runtime_or_outputs(tmp_path, monkeypatch):
    runner = _load_runner()
    monkeypatch.setattr(
        runner,
        "_load_runtime",
        lambda: (_ for _ in ()).throw(AssertionError("runtime loaded too early")),
    )

    returncode, _, _, output_dir, report = _invoke(
        runner, monkeypatch, tmp_path, expected_input_sha256="0" * 64
    )

    assert returncode != 0
    data = json.loads(report.read_text())
    assert data["failure_phase"] == "source_identity"
    assert data["primary_output_status"] == "not_started"
    assert not output_dir.exists()


def test_wrong_lineage_digest_fails_before_runtime_or_outputs(tmp_path, monkeypatch):
    runner = _load_runner()
    monkeypatch.setattr(
        runner,
        "_load_runtime",
        lambda: (_ for _ in ()).throw(AssertionError("runtime loaded too early")),
    )

    returncode, _, _, output_dir, report = _invoke(
        runner, monkeypatch, tmp_path, expected_lineage_sha256="f" * 64
    )

    assert returncode != 0
    data = json.loads(report.read_text())
    assert data["failure_phase"] == "lineage_identity"
    assert data["primary_output_status"] == "not_started"
    assert not output_dir.exists()


def test_report_collision_preserves_input_and_reroutes_failure(tmp_path, monkeypatch):
    runner = _load_runner()
    source = tmp_path / "plateau.glb"
    _write_grid(source)
    source_hash = _sha256(source)
    lineage = tmp_path / "lineage.json"
    lineage.write_text("{}\n")
    output_dir = tmp_path / "outputs"
    runtime, _ = _fake_runtime()
    monkeypatch.setattr(runner, "_load_runtime", lambda: runtime)

    returncode = runner.main(
        [
            "--input", str(source),
            "--expected-input-sha256", source_hash,
            "--output-dir", str(output_dir),
            "--report", str(source),
            "--target-faces", "40",
            "--continuation-classification", "test-authenticated-parent-consumer",
            "--expected-mlx-version", "test-mlx",
            "--expected-qem-module-sha256", "a" * 64,
            "--expected-cleanup-module-sha256", "b" * 64,
            "--lineage-ref", "parent", str(lineage), _sha256(lineage),
        ]
    )

    assert returncode != 0
    assert _sha256(source) == source_hash
    safe_report = tmp_path / "plateau.glb.assay-error.json"
    data = json.loads(safe_report.read_text())
    assert data["failure_phase"] == "validate_paths"
    assert data["report"]["rerouted"] is True


def test_protected_evidence_inside_output_custody_fails_without_reroute_loop(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    source = output_dir / "plateau.glb"
    lineage = output_dir / "lineage.json"
    requested_report = output_dir / "report.json"
    _write_grid(source)
    lineage.write_text("{}\n")
    source_sha256 = _sha256(source)
    lineage_sha256 = _sha256(lineage)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input", str(source),
            "--expected-input-sha256", source_sha256,
            "--output-dir", str(output_dir),
            "--report", str(requested_report),
            "--target-faces", "40",
            "--continuation-classification", "invalid-custody-fixture",
            "--expected-mlx-version", "test-mlx",
            "--expected-qem-module-sha256", "a" * 64,
            "--expected-cleanup-module-sha256", "b" * 64,
            "--lineage-ref", "parent", str(lineage), lineage_sha256,
        ],
        cwd=SCRIPT.parents[1],
        env={**os.environ, "PYTHONPATH": str(SCRIPT.parents[1])},
        capture_output=True,
        text=True,
        timeout=2,
    )

    assert completed.returncode != 0
    assert _sha256(source) == source_sha256
    assert _sha256(lineage) == lineage_sha256
    safe_report = tmp_path / "plateau.glb.assay-error.json"
    data = json.loads(safe_report.read_text())
    assert data["failure_phase"] == "validate_paths"
    assert data["primary_output_status"] == "not_started"
    assert data["report"] == {
        "requested_path": str(requested_report),
        "effective_path": str(safe_report),
        "rerouted": True,
    }
    assert not requested_report.exists()


def test_runtime_identity_mismatch_fails_before_primary_outputs(tmp_path, monkeypatch):
    runner = _load_runner()
    runtime, _ = _fake_runtime()
    runtime.identity["qem_module_sha256"] = "c" * 64

    returncode, _, _, output_dir, report = _invoke(
        runner, monkeypatch, tmp_path, runtime=runtime
    )

    assert returncode != 0
    data = json.loads(report.read_text())
    assert data["failure_phase"] == "runtime_identity"
    assert data["primary_output_status"] == "not_started"
    assert not output_dir.exists()


def test_package_version_falls_back_to_distribution_metadata(monkeypatch):
    runner = _load_runner()
    package_without_version = SimpleNamespace()
    monkeypatch.setattr(runner.importlib_metadata, "version", lambda name: "0.29.3")

    assert runner._package_version("mlx", package_without_version) == "0.29.3"


def test_output_dir_cannot_contain_authenticated_runtime_modules(tmp_path, monkeypatch):
    runner = _load_runner()
    runtime, _ = _fake_runtime()
    output_dir = tmp_path / "runtime-root"
    output_dir.mkdir()
    qem_module = output_dir / "simplify_qem_metal.py"
    cleanup_module = output_dir / "mesh_cleanup.py"
    qem_module.write_text("qem sentinel\n")
    cleanup_module.write_text("cleanup sentinel\n")
    runtime.identity["qem_module_path"] = str(qem_module)
    runtime.identity["cleanup_module_path"] = str(cleanup_module)

    returncode, _, _, _, report = _invoke(
        runner,
        monkeypatch,
        tmp_path,
        runtime=runtime,
        output_dir=output_dir,
    )

    assert returncode != 0
    data = json.loads(report.read_text())
    assert data["failure_phase"] == "runtime_custody"
    assert data["primary_output_status"] == "not_started"
    assert qem_module.read_text() == "qem sentinel\n"
    assert cleanup_module.read_text() == "cleanup sentinel\n"


def test_qem_failure_preserves_completed_cleanup_checkpoint(tmp_path, monkeypatch):
    runner = _load_runner()
    runtime, _ = _fake_runtime(fail_stage="local_qem")

    returncode, _, _, output_dir, report = _invoke(
        runner, monkeypatch, tmp_path, runtime=runtime
    )

    assert returncode != 0
    data = json.loads(report.read_text())
    assert data["failure_phase"] == "local_qem"
    assert data["primary_output_status"] == "partial_preserved"
    assert data["last_trustworthy_evidence"]["stage"] == "intermediate_cleanup"
    checkpoint = output_dir / "01-intermediate-cleanup.glb"
    assert checkpoint.is_file()
    assert _sha256(checkpoint) == data["last_trustworthy_evidence"]["mesh_sha256"]
    assert not (output_dir / "02-local-qem.glb").exists()


def test_final_cleanup_failure_preserves_completed_qem_checkpoint(tmp_path, monkeypatch):
    runner = _load_runner()
    runtime, _ = _fake_runtime(fail_stage="final_cleanup")

    returncode, _, _, output_dir, report = _invoke(
        runner, monkeypatch, tmp_path, runtime=runtime
    )

    assert returncode != 0
    data = json.loads(report.read_text())
    assert data["failure_phase"] == "final_cleanup"
    assert data["primary_output_status"] == "partial_preserved"
    assert data["last_trustworthy_evidence"]["stage"] == "local_qem"
    checkpoint = output_dir / "02-local-qem.glb"
    assert checkpoint.is_file()
    assert _sha256(checkpoint) == data["last_trustworthy_evidence"]["mesh_sha256"]
    assert not (output_dir / "03-final-cleanup.glb").exists()


def test_receipt_route_mismatch_fails_loud_after_qem(tmp_path, monkeypatch):
    runner = _load_runner()
    runtime, _ = _fake_runtime()
    original = runtime.simplify

    def wrong_route(*args, **kwargs):
        vertices, faces, receipt = original(*args, **kwargs)
        receipt["route"] = "wrong-route"
        return vertices, faces, receipt

    runtime.simplify = wrong_route

    returncode, _, _, output_dir, report = _invoke(
        runner, monkeypatch, tmp_path, runtime=runtime
    )

    assert returncode != 0
    data = json.loads(report.read_text())
    assert data["failure_phase"] == "local_qem_receipt"
    assert data["last_trustworthy_evidence"]["stage"] == "intermediate_cleanup"
    assert not (output_dir / "02-local-qem.glb").exists()


def test_each_published_stage_is_a_finite_indexed_triangle_glb(tmp_path, monkeypatch):
    runner = _load_runner()

    returncode, _, _, output_dir, _ = _invoke(runner, monkeypatch, tmp_path)

    assert returncode == 0
    for path in output_dir.glob("*.glb"):
        with open_triangle_glb(path) as mesh:
            assert np.isfinite(mesh.vertices).all()
            assert len(mesh.faces) > 0
            assert int(mesh.faces.max()) < len(mesh.vertices)
