"""Contracts for the mesh winding/export witness."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh


SCRIPT = Path("scripts/mesh_winding_witness.py")


def _write_mesh_npz(path: Path, vertices, faces, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
        **extra,
    )


def test_analyze_mesh_detects_reversed_face_and_fix_counterfactual():
    from scripts.mesh_winding_witness import analyze_mesh

    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    faces[0] = faces[0][::-1]

    report = analyze_mesh("flipped_box", vertices, faces)

    assert report["orientation"]["inward_faces"] >= 1
    assert report["edge_consistency"]["same_direction_conflict_edges"] >= 1
    assert report["fix_normals_counterfactual"]["changed_faces"] >= 1
    assert (
        report["fix_normals_counterfactual"]["after_orientation"]["inward_faces"]
        < report["orientation"]["inward_faces"]
    )


def test_visible_exterior_orientation_flags_backfacing_open_patch_with_zero_edge_conflicts():
    from scripts.mesh_winding_witness import analyze_mesh

    vertices = np.array(
        [
            [-1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

    report = analyze_mesh("backfacing_open_patch", vertices, faces)

    assert report["is_winding_consistent"] is True
    assert report["edge_consistency"]["same_direction_conflict_edges"] == 0
    assert report["visible_exterior_orientation"]["views"]["+Y"]["visible_pixels"] > 0
    assert (
        report["visible_exterior_orientation"]["views"]["+Y"]["backfacing_visible_ratio"]
        > 0.95
    )
    assert report["visible_exterior_orientation"]["worst_view"] == "+Y"


def test_visible_exterior_orientation_accepts_frontfacing_open_patch():
    from scripts.mesh_winding_witness import analyze_mesh

    vertices = np.array(
        [
            [-1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int64)

    report = analyze_mesh("frontfacing_open_patch", vertices, faces)

    assert report["is_winding_consistent"] is True
    assert report["edge_consistency"]["same_direction_conflict_edges"] == 0
    assert report["visible_exterior_orientation"]["views"]["+Y"]["visible_pixels"] > 0
    assert (
        report["visible_exterior_orientation"]["views"]["+Y"]["backfacing_visible_ratio"]
        < 0.05
    )


def test_checkpoint_report_catches_uv_stage_reversed_source_mapping(tmp_path):
    clean_vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ],
        dtype=np.float32,
    )
    clean_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

    checkpoint_dir = tmp_path / "ckpt"
    _write_mesh_npz(checkpoint_dir / "mesh_clean.npz", clean_vertices, clean_faces)

    uv_vertices = clean_vertices[[0, 1, 2, 0, 3, 2]]
    uv_faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    vmapping = np.array([0, 1, 2, 0, 3, 2], dtype=np.int64)
    _write_mesh_npz(
        checkpoint_dir / "mesh_uv.npz",
        uv_vertices,
        uv_faces,
        vmapping=vmapping,
        uvs=np.zeros((6, 2), dtype=np.float32),
    )

    report_path = tmp_path / "winding.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--report",
            str(report_path),
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text())
    assert report["status"] == "ok"
    assert report["schema"] == "trellis2mlx.mesh_winding_witness.v1"
    assert report["stages"]["mesh_clean"]["faces"] == 2
    assert report["stages"]["mesh_uv"]["edge_consistency"]["boundary_edges"] == 6
    assert report["stages"]["mesh_uv"]["source_face_mapping"] == {
        "source_stage": "mesh_clean",
        "mapped_faces": 2,
        "same_orientation_faces": 1,
        "reversed_orientation_faces": 1,
        "unmatched_faces": 0,
        "ambiguous_faces": 0,
    }


def test_checkpoint_report_loads_postprocess_boundary_stages(tmp_path):
    from scripts.mesh_winding_witness import build_report

    vertices = np.array(
        [
            [-1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

    checkpoint_dir = tmp_path / "ckpt"
    _write_mesh_npz(checkpoint_dir / "mesh_after_cleanup_pass1.npz", vertices, faces)
    _write_mesh_npz(checkpoint_dir / "mesh_after_final_simplify.npz", vertices, faces)
    _write_mesh_npz(checkpoint_dir / "mesh_after_cleanup_final.npz", vertices, faces)

    report = build_report(
        checkpoint_dir=checkpoint_dir,
        glb=None,
        report_path=tmp_path / "winding.json",
    )

    assert "mesh_after_cleanup_pass1" in report["stages"]
    assert "mesh_after_final_simplify" in report["stages"]
    assert "mesh_after_cleanup_final" in report["stages"]
    assert (
        report["stages"]["mesh_after_cleanup_final"]["visible_exterior_orientation"]
        ["views"]["+Y"]["backfacing_visible_ratio"]
        > 0.95
    )


def test_witness_writes_failure_report_when_no_mesh_inputs_exist(tmp_path):
    report_path = tmp_path / "winding.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--checkpoint-dir",
            str(tmp_path / "missing"),
            "--report",
            str(report_path),
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    report = json.loads(report_path.read_text())
    assert report["status"] == "error"
    assert report["phase"] == "load_inputs"
    assert report["last_trustworthy_evidence"]["loaded_stages"] == []


def test_failure_report_preserves_stages_loaded_before_later_checkpoint_failure(tmp_path):
    checkpoint_dir = tmp_path / "ckpt"
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    _write_mesh_npz(
        checkpoint_dir / "mesh_raw.npz",
        np.asarray(box.vertices, dtype=np.float32),
        np.asarray(box.faces, dtype=np.int64),
    )
    checkpoint_dir.mkdir(exist_ok=True)
    np.savez_compressed(
        checkpoint_dir / "mesh_clean.npz",
        vertices=np.asarray(box.vertices, dtype=np.float32),
    )

    report_path = tmp_path / "winding.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--report",
            str(report_path),
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    report = json.loads(report_path.read_text())
    assert report["status"] == "error"
    assert report["phase"] == "load_inputs"
    assert report["last_trustworthy_evidence"]["loaded_stages"] == ["mesh_raw"]
