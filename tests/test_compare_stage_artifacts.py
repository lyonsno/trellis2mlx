import json
from pathlib import Path
import subprocess

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_compare_conditioning_reports_shape_and_abs_deltas(tmp_path):
    ref = tmp_path / "ref.npz"
    mlx = tmp_path / "mlx.npz"
    out = tmp_path / "comparison.json"
    np.savez(ref, cond=np.array([[[1.0, 2.0]]], dtype=np.float32), neg_cond=np.zeros((1, 1, 2), dtype=np.float32))
    np.savez(mlx, cond=np.array([[[1.25, 1.5]]], dtype=np.float32), neg_cond=np.zeros((1, 1, 2), dtype=np.float32))

    result = subprocess.run(
        [
            "python",
            "scripts/compare_stage_artifacts.py",
            "--stage",
            "conditioning",
            "--reference",
            str(ref),
            "--candidate",
            str(mlx),
            "--output",
            str(out),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text())
    assert report["stage"] == "conditioning"
    assert report["arrays"]["cond"]["shape_match"] is True
    assert report["arrays"]["cond"]["max_abs_diff"] == 0.5
    assert report["arrays"]["cond"]["mean_abs_diff"] == 0.375
    assert report["arrays"]["neg_cond"]["max_abs_diff"] == 0.0


def test_compare_sparse_reports_coordinate_overlap(tmp_path):
    ref = tmp_path / "ref.npz"
    mlx = tmp_path / "mlx.npz"
    out = tmp_path / "comparison.json"
    np.savez(ref, coords=np.array([[0, 0, 0, 0], [0, 1, 1, 1], [0, 2, 2, 2]], dtype=np.int32))
    np.savez(mlx, coords=np.array([[0, 1, 1, 1], [0, 2, 2, 2], [0, 3, 3, 3]], dtype=np.int32))

    result = subprocess.run(
        [
            "python",
            "scripts/compare_stage_artifacts.py",
            "--stage",
            "sparse_coords",
            "--reference",
            str(ref),
            "--candidate",
            str(mlx),
            "--output",
            str(out),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text())
    assert report["coords"]["reference_count"] == 3
    assert report["coords"]["candidate_count"] == 3
    assert report["coords"]["common_count"] == 2
    assert report["coords"]["jaccard"] == 0.5
    assert report["coords"]["reference_only_count"] == 1
    assert report["coords"]["candidate_only_count"] == 1


def test_compare_shape_slat_aligns_feature_deltas_by_common_coords(tmp_path):
    ref = tmp_path / "ref.npz"
    mlx = tmp_path / "mlx.npz"
    out = tmp_path / "comparison.json"
    ref_coords = np.array([[0, 0, 0, 0], [0, 1, 1, 1]], dtype=np.int32)
    mlx_coords = np.array([[0, 1, 1, 1], [0, 2, 2, 2]], dtype=np.int32)
    np.savez(ref, coords=ref_coords, feats=np.array([[10.0, 10.0], [1.0, 3.0]], dtype=np.float32))
    np.savez(mlx, coords=mlx_coords, feats=np.array([[2.0, 1.0], [20.0, 20.0]], dtype=np.float32))

    result = subprocess.run(
        [
            "python",
            "scripts/compare_stage_artifacts.py",
            "--stage",
            "shape_slat",
            "--reference",
            str(ref),
            "--candidate",
            str(mlx),
            "--output",
            str(out),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text())
    assert report["coords"]["common_count"] == 1
    assert report["features"]["common_shape"] == [1, 2]
    assert report["features"]["max_abs_diff"] == 2.0
    assert report["features"]["mean_abs_diff"] == 1.5


def test_compare_sparse_internals_reports_array_and_coordinate_deltas(tmp_path):
    ref = tmp_path / "ref.npz"
    mlx = tmp_path / "mlx.npz"
    out = tmp_path / "comparison.json"
    np.savez(
        ref,
        z_s=np.zeros((1, 2, 2, 2, 2), dtype=np.float32),
        logits=np.ones((1, 1, 4, 4, 4), dtype=np.float32),
        decoded=np.zeros((4, 4, 4), dtype=bool),
        decoded_ds=np.zeros((2, 2, 2), dtype=bool),
        coords=np.array([[0, 0, 0, 0]], dtype=np.int32),
    )
    np.savez(
        mlx,
        z_s=np.ones((1, 2, 2, 2, 2), dtype=np.float32),
        logits=np.ones((1, 1, 4, 4, 4), dtype=np.float32) * 3,
        decoded=np.zeros((4, 4, 4), dtype=bool),
        decoded_ds=np.zeros((2, 2, 2), dtype=bool),
        coords=np.array([[0, 0, 0, 0]], dtype=np.int32),
    )

    result = subprocess.run(
        [
            "python",
            "scripts/compare_stage_artifacts.py",
            "--stage",
            "sparse_internals",
            "--reference",
            str(ref),
            "--candidate",
            str(mlx),
            "--output",
            str(out),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text())
    assert report["arrays"]["z_s"]["mean_abs_diff"] == 1.0
    assert report["arrays"]["logits"]["mean_abs_diff"] == 2.0
    assert report["coords"]["jaccard"] == 1.0


def test_compare_sparse_flow_step_reports_sampler_tensor_deltas(tmp_path):
    ref = tmp_path / "ref.npz"
    mlx = tmp_path / "mlx.npz"
    out = tmp_path / "comparison.json"
    base = np.zeros((1, 2, 2, 2, 2), dtype=np.float32)
    ref_payload = {
        "noise": base,
        "pred_pos": base + 1,
        "pred_neg": base + 2,
        "pred_cfg": base + 3,
        "std_ratio": np.array([[[[[1.0]]]]], dtype=np.float32),
        "pred_final": base + 4,
        "sample_next": base + 5,
        "t": np.array(1.0, dtype=np.float32),
        "t_prev": np.array(0.5, dtype=np.float32),
    }
    cand_payload = dict(ref_payload)
    cand_payload["pred_pos"] = base + 1.25
    cand_payload["pred_final"] = base + 5
    np.savez(ref, **ref_payload)
    np.savez(mlx, **cand_payload)

    result = subprocess.run(
        [
            "python",
            "scripts/compare_stage_artifacts.py",
            "--stage",
            "sparse_flow_step",
            "--reference",
            str(ref),
            "--candidate",
            str(mlx),
            "--output",
            str(out),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text())
    assert report["stage"] == "sparse_flow_step"
    assert report["arrays"]["pred_pos"]["mean_abs_diff"] == 0.25
    assert report["arrays"]["pred_final"]["mean_abs_diff"] == 1.0


def test_compare_sparse_flow_block_trace_reports_internal_tensor_deltas(tmp_path):
    ref = tmp_path / "ref.npz"
    mlx = tmp_path / "mlx.npz"
    out = tmp_path / "comparison.json"
    base = np.zeros((1, 4, 3), dtype=np.float32)
    ref_payload = {
        "pos_input_projected": base + 1,
        "pos_block0_norm1": base + 2,
        "pos_block0_gate_mlp": base + 2.5,
        "pos_block0_modulated_self_input": base + 3,
        "pos_block0_q_pre_norm": base + 4,
        "pos_block0_k_pre_norm": base + 5,
        "pos_block0_v": base + 6,
        "pos_block0_q_post_norm": base + 7,
        "pos_block0_k_post_norm": base + 8,
        "pos_block0_q_post_rope": base + 9,
        "pos_block0_k_post_rope": base + 10,
        "pos_block0_attention_raw": base + 11,
        "pos_block0_self_attn": base + 12,
        "pos_block0_after_self": base + 13,
        "pos_block0_cross_attn": base + 14,
        "pos_block0_after_cross": base + 15,
        "pos_block0_mlp": base + 16,
        "pos_block0_after_mlp": base + 17,
        "neg_input_projected": base + 18,
        "neg_block0_norm1": base + 19,
        "neg_block0_gate_mlp": base + 19.5,
        "neg_block0_modulated_self_input": base + 20,
        "neg_block0_q_pre_norm": base + 21,
        "neg_block0_k_pre_norm": base + 22,
        "neg_block0_v": base + 23,
        "neg_block0_q_post_norm": base + 24,
        "neg_block0_k_post_norm": base + 25,
        "neg_block0_q_post_rope": base + 26,
        "neg_block0_k_post_rope": base + 27,
        "neg_block0_attention_raw": base + 28,
        "neg_block0_self_attn": base + 29,
        "neg_block0_after_self": base + 30,
        "neg_block0_cross_attn": base + 31,
        "neg_block0_after_cross": base + 32,
        "neg_block0_mlp": base + 33,
        "neg_block0_after_mlp": base + 34,
    }
    cand_payload = dict(ref_payload)
    cand_payload["pos_block0_gate_mlp"] = base + 2.75
    cand_payload["pos_block0_q_post_rope"] = base + 9.75
    cand_payload["pos_block0_self_attn"] = base + 12.5
    cand_payload["neg_block0_after_mlp"] = base + 36
    np.savez(ref, **ref_payload)
    np.savez(mlx, **cand_payload)

    result = subprocess.run(
        [
            "python",
            "scripts/compare_stage_artifacts.py",
            "--stage",
            "sparse_flow_block_trace",
            "--reference",
            str(ref),
            "--candidate",
            str(mlx),
            "--output",
            str(out),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text())
    assert report["stage"] == "sparse_flow_block_trace"
    assert report["arrays"]["pos_block0_gate_mlp"]["mean_abs_diff"] == 0.25
    assert report["arrays"]["pos_block0_q_post_rope"]["mean_abs_diff"] == 0.75
    assert report["arrays"]["pos_block0_self_attn"]["mean_abs_diff"] == 0.5
    assert report["arrays"]["neg_block0_after_mlp"]["mean_abs_diff"] == 2.0


def test_compare_mesh_stage_reports_reversed_face_orientation(tmp_path):
    ref = tmp_path / "ref.npz"
    mlx = tmp_path / "mlx.npz"
    out = tmp_path / "comparison.json"
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.savez(ref, vertices=vertices, faces=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32))
    np.savez(mlx, vertices=vertices, faces=np.array([[0, 2, 1], [1, 3, 2]], dtype=np.int32))

    result = subprocess.run(
        [
            "python",
            "scripts/compare_stage_artifacts.py",
            "--stage",
            "mesh_raw",
            "--reference",
            str(ref),
            "--candidate",
            str(mlx),
            "--output",
            str(out),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text())
    orientation = report["face_orientation_overlap"]
    assert orientation["common_face_sets"] == 2
    assert orientation["same_orientation_faces"] == 1
    assert orientation["reversed_orientation_faces"] == 1
    assert report["faces"]["exact_row_match"] is False


def test_compare_mesh_stage_reports_same_direction_edge_conflict(tmp_path):
    ref = tmp_path / "ref.npz"
    mlx = tmp_path / "mlx.npz"
    out = tmp_path / "comparison.json"
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    valid_faces = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int32)
    conflict_faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    np.savez(ref, vertices=vertices, faces=valid_faces)
    np.savez(mlx, vertices=vertices, faces=conflict_faces)

    result = subprocess.run(
        [
            "python",
            "scripts/compare_stage_artifacts.py",
            "--stage",
            "mesh_clean",
            "--reference",
            str(ref),
            "--candidate",
            str(mlx),
            "--output",
            str(out),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text())
    assert report["reference_edge_consistency"]["same_direction_conflict_edges"] == 0
    assert report["candidate_edge_consistency"]["same_direction_conflict_edges"] == 1


def test_compare_mesh_uv_reports_uv_and_vmapping_deltas(tmp_path):
    ref = tmp_path / "ref.npz"
    mlx = tmp_path / "mlx.npz"
    out = tmp_path / "comparison.json"
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    np.savez(
        ref,
        vertices=vertices,
        faces=faces,
        uvs=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        vmapping=np.array([0, 1, 2], dtype=np.int32),
    )
    np.savez(
        mlx,
        vertices=vertices,
        faces=faces,
        uvs=np.array([[0.0, 0.0], [1.0, 0.25], [0.0, 1.0]], dtype=np.float32),
        vmapping=np.array([0, 2, 1], dtype=np.int32),
    )

    result = subprocess.run(
        [
            "python",
            "scripts/compare_stage_artifacts.py",
            "--stage",
            "mesh_uv",
            "--reference",
            str(ref),
            "--candidate",
            str(mlx),
            "--output",
            str(out),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text())
    assert report["uvs"]["shape_match"] is True
    assert report["uvs"]["max_abs_diff"] == 0.25
    assert report["vmapping"]["exact_match"] is False
    assert report["vmapping"]["mismatched_entries"] == 2
