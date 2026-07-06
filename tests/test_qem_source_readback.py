"""QEM source-readback comparison diagnostics."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


SCRIPT = Path("scripts/qem_source_readback_compare.py")


def _load_script_module():
    spec = importlib.util.spec_from_file_location("qem_source_readback_compare_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_mesh():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    return vertices, faces


def test_local_qem_readback_records_costs_props_and_collapsed_edges():
    from trellmlx.qem_source_readback import local_simplify_step_readback

    vertices, faces = _fixture_mesh()
    readback = local_simplify_step_readback(vertices, faces, collapse_thresh=np.float32(1e-8))

    assert readback["schema"] == "trellis2mlx.qem_local_step_readback.v1"
    assert readback["settings"]["lambda_edge_length"] == 1e-2
    assert readback["settings"]["lambda_skinny"] == 1e-3
    assert readback["edges"].dtype == np.int32
    assert readback["costs"].dtype == np.float32
    assert readback["props"].dtype == np.uint64
    assert readback["edges"].shape[1] == 2
    assert readback["costs"].shape == (len(readback["edges"]),)
    assert readback["props"].shape == (len(faces),)
    assert readback["collapse_counts"]["eligible"] >= readback["collapse_counts"]["collapsed_edges"]
    assert isinstance(readback["collapsed_edge_ids"], list)
    assert all(isinstance(edge_id, int) for edge_id in readback["collapsed_edge_ids"])


def test_local_qem_readback_records_source_shaped_metal_backend_identity():
    from trellmlx.simplify_qem_metal import HAS_MLX
    from trellmlx.qem_source_readback import local_simplify_step_readback

    if not HAS_MLX:
        pytest.skip("source-shaped Metal QEM backend requires MLX")

    vertices, faces = _fixture_mesh()
    readback = local_simplify_step_readback(
        vertices,
        faces,
        collapse_thresh=np.float32(1e-8),
        qem_backend="mlx-metal-source",
    )

    assert readback["settings"]["qem_backend"] == "mlx-metal-source"
    assert readback["qems"].shape == (len(vertices), 10)
    assert readback["qems"].dtype == np.float32
    assert readback["costs"].shape == (len(readback["edges"]),)
    assert readback["terms"].shape == (len(readback["edges"]), 7)
    assert readback["terms"].dtype == np.float32
    json.dumps(readback, default=lambda value: value.tolist() if hasattr(value, "tolist") else value)


def test_local_qem_readback_rejects_unknown_qem_backend():
    from trellmlx.qem_source_readback import local_simplify_step_readback

    vertices, faces = _fixture_mesh()

    with pytest.raises(ValueError, match="unknown qem_backend"):
        local_simplify_step_readback(vertices, faces, qem_backend="bogus")


def test_local_qem_readback_rejects_nonfinite_source_shaped_metal_evidence():
    from trellmlx.simplify_qem_metal import HAS_MLX
    from trellmlx.qem_source_readback import local_simplify_step_readback

    if not HAS_MLX:
        pytest.skip("source-shaped Metal QEM backend requires MLX")

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    with pytest.raises(ValueError, match="nonfinite local QEM evidence.*mlx-metal-source"):
        local_simplify_step_readback(vertices, faces, qem_backend="mlx-metal-source")


def test_qem_source_readback_report_compares_source_bits_and_collapse_ids():
    from trellmlx.qem_source_readback import (
        build_qem_source_readback_report,
        local_simplify_step_readback,
    )

    vertices, faces = _fixture_mesh()
    local = local_simplify_step_readback(vertices, faces, collapse_thresh=np.float32(1e-8))
    source = {
        "edges": local["edges"].copy(),
        "costs": local["costs"].copy(),
        "props": local["props"].copy(),
        "qems": local["qems"].copy(),
    }
    if len(source["costs"]):
        source["costs"][0] = np.nextafter(source["costs"][0], np.float32(np.inf), dtype=np.float32)

    report = build_qem_source_readback_report(
        requested_route="qem-source-readback-compare",
        effective_route="local-qem-step-vs-source-readback",
        mesh_path=Path("/tmp/fixture-mesh.npz"),
        source_readback_path=Path("/tmp/source-readback.npz"),
        vertices=vertices,
        faces=faces,
        source=source,
        collapse_thresh=np.float32(1e-8),
    )

    assert report["schema"] == "trellis2mlx.qem_source_readback_compare.v1"
    assert report["status"] == "ok"
    assert report["requested_route"] == "qem-source-readback-compare"
    assert report["effective_route"] == "local-qem-step-vs-source-readback"
    assert report["identity"]["edge_order_exact"] is True
    assert report["cost_summary"]["source_vs_local_bit_exact_edges"] == len(local["costs"]) - 1
    assert "source_collapsed_edge_ids" in report["collapse_identity"]
    assert "local_collapsed_edge_ids" in report["collapse_identity"]
    assert "collapsed_edge_ids_exact" in report["collapse_identity"]
    json.dumps(report)


def test_qem_source_readback_cost_summary_keeps_nonfinite_values_out_of_diffs():
    from trellmlx.qem_source_readback import (
        build_qem_source_readback_report,
        local_simplify_step_readback,
    )

    vertices, faces = _fixture_mesh()
    local = local_simplify_step_readback(vertices, faces, collapse_thresh=np.float32(1e-8))
    source = {
        "edges": local["edges"].copy(),
        "costs": local["costs"].copy(),
        "props": local["props"].copy(),
        "qems": local["qems"].copy(),
    }
    source["costs"][:] = np.inf

    report = build_qem_source_readback_report(
        requested_route="qem-source-readback-compare",
        effective_route="local-qem-step-vs-source-readback",
        mesh_path=Path("/tmp/fixture-mesh.npz"),
        source_readback_path=Path("/tmp/source-readback.npz"),
        vertices=vertices,
        faces=faces,
        source=source,
        collapse_thresh=np.float32(1e-8),
    )

    assert report["cost_summary"]["finite_pair_count"] == 0
    assert np.isfinite(report["cost_summary"]["source_vs_local_max_abs_finite_diff"])
    assert np.isfinite(report["cost_summary"]["source_vs_local_mean_abs_finite_diff"])
    json.dumps(report, allow_nan=False)


def test_qem_source_readback_report_attributes_optional_source_terms():
    from trellmlx.qem_source_readback import (
        build_qem_source_readback_report,
        local_simplify_step_readback,
    )

    vertices, faces = _fixture_mesh()
    local = local_simplify_step_readback(vertices, faces, collapse_thresh=np.float32(1e-8))
    source = {
        "edges": local["edges"].copy(),
        "costs": local["costs"].copy(),
        "props": local["props"].copy(),
        "qems": local["qems"].copy(),
        "terms": local["terms"].copy(),
        "status": local["status"].copy(),
    }
    source["terms"][:, 3] = np.nextafter(source["terms"][:, 3], np.float32(np.inf), dtype=np.float32)
    source["costs"] = (
        source["terms"][:, 3]
        + np.float32(1e-2) * source["terms"][:, 4]
        + source["terms"][:, 6]
    ).astype(np.float32)

    report = build_qem_source_readback_report(
        requested_route="qem-source-readback-compare",
        effective_route="local-qem-step-vs-source-readback",
        mesh_path=Path("/tmp/fixture-mesh.npz"),
        source_readback_path=Path("/tmp/source-readback.npz"),
        vertices=vertices,
        faces=faces,
        source=source,
        collapse_thresh=np.float32(1e-8),
    )

    assert report["term_summary"]["available"] is True
    assert report["term_summary"]["collapse_position"]["bit_exact_entries"] == local["terms"][:, :3].size
    assert report["term_summary"]["qem_cost"]["source_vs_local_bit_exact_edges"] == 0
    assert "base_diff_ge_skinny_diff_edges" in report["term_summary"]["attribution"]
    assert report["term_summary"]["qem_eval_split"]["source_qem_local_eval_vs_source_qem_cost"]["available"] is True
    json.dumps(report, allow_nan=False)


def test_qem_source_distribution_report_preserves_run_ranges_and_local_placement():
    from trellmlx.qem_source_readback import (
        build_qem_source_distribution_report,
        local_simplify_step_readback,
    )

    vertices, faces = _fixture_mesh()
    local = local_simplify_step_readback(vertices, faces, collapse_thresh=np.float32(1e-8))
    source_a = {
        "edges": local["edges"].copy(),
        "costs": local["costs"].copy(),
        "props": local["props"].copy(),
        "qems": local["qems"].copy(),
        "terms": local["terms"].copy(),
    }
    source_b = {
        "edges": local["edges"].copy(),
        "costs": local["costs"].copy(),
        "props": local["props"].copy(),
        "qems": local["qems"].copy(),
        "terms": local["terms"].copy(),
    }
    source_b["costs"][:] = np.float32(1.0)
    source_b["props"][:] = np.uint64(0xFFFFFFFFFFFFFFFF)

    report = build_qem_source_distribution_report(
        requested_route="qem-source-distribution-compare",
        effective_route="local-qem-step-vs-source-distribution",
        mesh_path=Path("/tmp/fixture-mesh.npz"),
        source_readback_paths=[Path("/tmp/source-a.npz"), Path("/tmp/source-b.npz")],
        vertices=vertices,
        faces=faces,
        sources=[source_a, source_b],
        collapse_thresh=np.float32(1e-8),
    )

    assert report["schema"] == "trellis2mlx.qem_source_distribution_compare.v1"
    assert report["status"] == "ok"
    assert report["source_distribution"]["run_count"] == 2
    assert report["source_distribution"]["removed_faces"]["min"] <= local["collapse_counts"]["removed_faces"]
    assert report["source_distribution"]["removed_faces"]["max"] >= local["collapse_counts"]["removed_faces"]
    assert report["local_vs_distribution"]["removed_faces_position"] == "inside"
    assert "source_run_reports" in report
    assert len(report["source_run_reports"]) == 2
    json.dumps(report, allow_nan=False)


def test_qem_source_readback_harness_writes_failure_report_for_missing_source_npz(tmp_path):
    harness = _load_script_module()
    mesh_path = tmp_path / "mesh.npz"
    report_path = tmp_path / "qem-source-readback-failure.json"
    vertices, faces = _fixture_mesh()
    np.savez(mesh_path, vertices=vertices, faces=faces)

    exit_code = harness.main([
        "--mesh",
        str(mesh_path),
        "--source-readback",
        str(tmp_path / "missing-source.npz"),
        "--report",
        str(report_path),
    ])

    assert exit_code == 1
    report = json.loads(report_path.read_text())
    assert report["schema"] == "trellis2mlx.qem_source_readback_compare.v1"
    assert report["status"] == "failed"
    assert report["failure_phase"] == "source_readback"
    assert report["requested_route"] == "qem-source-readback-compare"
    assert report["last_trustworthy_evidence"]["report_written"] is True


def test_qem_source_readback_harness_preserves_distribution_failure_identity(tmp_path):
    harness = _load_script_module()
    mesh_path = tmp_path / "mesh.npz"
    source_a_path = tmp_path / "source-a.npz"
    source_b_path = tmp_path / "missing-source-b.npz"
    report_path = tmp_path / "qem-source-distribution-failure.json"
    vertices, faces = _fixture_mesh()
    np.savez(mesh_path, vertices=vertices, faces=faces)
    np.savez(
        source_a_path,
        edges=np.empty((0, 2), dtype=np.int32),
        costs=np.empty((0,), dtype=np.float32),
        props=np.empty((len(faces),), dtype=np.uint64),
    )

    exit_code = harness.main([
        "--mesh",
        str(mesh_path),
        "--source-readback",
        str(source_a_path),
        "--source-readback",
        str(source_b_path),
        "--local-qem-backend",
        "mlx-metal-source",
        "--report",
        str(report_path),
    ])

    assert exit_code == 1
    report = json.loads(report_path.read_text())
    assert report["schema"] == "trellis2mlx.qem_source_distribution_compare.v1"
    assert report["status"] == "failed"
    assert report["failure_phase"] == "source_readback"
    assert report["requested_route"] == "qem-source-distribution-compare"
    assert report["effective_route"] == "local-qem-step-vs-source-distribution"
    assert report["asset"]["source_readback_path"] == str(source_b_path)
    assert report["asset"]["source_readback_paths"] == [str(source_a_path), str(source_b_path)]
    assert "missing-source-b.npz" in report["error"]


def test_qem_source_readback_harness_preserves_distribution_compare_failure_identity(tmp_path, monkeypatch):
    from trellmlx.qem_source_readback import local_simplify_step_readback

    harness = _load_script_module()
    mesh_path = tmp_path / "mesh.npz"
    source_a_path = tmp_path / "source-a.npz"
    source_b_path = tmp_path / "source-b.npz"
    report_path = tmp_path / "qem-source-distribution-compare-failure.json"
    vertices, faces = _fixture_mesh()
    local = local_simplify_step_readback(vertices, faces, collapse_thresh=np.float32(1e-8))
    np.savez(mesh_path, vertices=vertices, faces=faces)
    for source_path in (source_a_path, source_b_path):
        np.savez(
            source_path,
            edges=local["edges"],
            costs=local["costs"],
            props=local["props"],
            qems=local["qems"],
            terms=local["terms"],
        )

    def fail_distribution_compare(**_kwargs):
        raise ValueError("forced distribution compare failure")

    monkeypatch.setattr(harness, "build_qem_source_distribution_report", fail_distribution_compare)

    exit_code = harness.main([
        "--mesh",
        str(mesh_path),
        "--source-readback",
        str(source_a_path),
        "--source-readback",
        str(source_b_path),
        "--local-qem-backend",
        "mlx-metal-source",
        "--report",
        str(report_path),
    ])

    assert exit_code == 1
    report = json.loads(report_path.read_text())
    assert report["schema"] == "trellis2mlx.qem_source_distribution_compare.v1"
    assert report["status"] == "failed"
    assert report["failure_phase"] == "compare"
    assert report["requested_route"] == "qem-source-distribution-compare"
    assert report["effective_route"] == "local-qem-step-vs-source-distribution"
    assert report["settings"]["qem_backend"] == "mlx-metal-source"
    assert report["asset"]["source_readback_path"] is None
    assert report["asset"]["source_readback_paths"] == [str(source_a_path), str(source_b_path)]
    assert "forced distribution compare failure" in report["error"]


def test_qem_source_readback_script_runs_from_script_path(tmp_path):
    from trellmlx.qem_source_readback import local_simplify_step_readback

    mesh_path = tmp_path / "mesh.npz"
    source_path = tmp_path / "source-readback.npz"
    report_path = tmp_path / "qem-source-readback.json"
    vertices, faces = _fixture_mesh()
    local = local_simplify_step_readback(vertices, faces, collapse_thresh=np.float32(1e-8))
    np.savez(mesh_path, vertices=vertices, faces=faces)
    np.savez(
        source_path,
        edges=local["edges"],
        costs=local["costs"],
        props=local["props"],
        qems=local["qems"],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mesh",
            str(mesh_path),
            "--source-readback",
            str(source_path),
            "--report",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text())
    assert report["status"] == "ok"
    assert report["identity"]["edge_order_exact"] is True
    assert report["collapse_identity"]["collapsed_edge_ids_exact"] is True


def test_qem_source_readback_script_records_source_shaped_metal_backend(tmp_path):
    from trellmlx.simplify_qem_metal import HAS_MLX
    from trellmlx.qem_source_readback import local_simplify_step_readback

    if not HAS_MLX:
        pytest.skip("source-shaped Metal QEM backend requires MLX")

    mesh_path = tmp_path / "mesh.npz"
    source_path = tmp_path / "source-readback.npz"
    report_path = tmp_path / "qem-source-readback-metal.json"
    vertices, faces = _fixture_mesh()
    local = local_simplify_step_readback(
        vertices,
        faces,
        collapse_thresh=np.float32(1e-8),
        qem_backend="mlx-metal-source",
    )
    np.savez(mesh_path, vertices=vertices, faces=faces)
    np.savez(
        source_path,
        edges=local["edges"],
        costs=local["costs"],
        props=local["props"],
        qems=local["qems"],
        terms=local["terms"],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mesh",
            str(mesh_path),
            "--source-readback",
            str(source_path),
            "--local-qem-backend",
            "mlx-metal-source",
            "--report",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text())
    assert report["settings"]["qem_backend"] == "mlx-metal-source"
    assert report["settings"]["local_topology_backend"] == "mlx-metal"


def test_qem_source_readback_script_accepts_multiple_source_readbacks(tmp_path):
    from trellmlx.qem_source_readback import local_simplify_step_readback

    mesh_path = tmp_path / "mesh.npz"
    source_a_path = tmp_path / "source-a.npz"
    source_b_path = tmp_path / "source-b.npz"
    report_path = tmp_path / "qem-source-distribution.json"
    vertices, faces = _fixture_mesh()
    local = local_simplify_step_readback(vertices, faces, collapse_thresh=np.float32(1e-8))
    np.savez(mesh_path, vertices=vertices, faces=faces)
    np.savez(
        source_a_path,
        edges=local["edges"],
        costs=local["costs"],
        props=local["props"],
        qems=local["qems"],
        terms=local["terms"],
    )
    np.savez(
        source_b_path,
        edges=local["edges"],
        costs=local["costs"],
        props=local["props"],
        qems=local["qems"],
        terms=local["terms"],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mesh",
            str(mesh_path),
            "--source-readback",
            str(source_a_path),
            "--source-readback",
            str(source_b_path),
            "--report",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text())
    assert report["schema"] == "trellis2mlx.qem_source_distribution_compare.v1"
    assert report["source_distribution"]["run_count"] == 2
