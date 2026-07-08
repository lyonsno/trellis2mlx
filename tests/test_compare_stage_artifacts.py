import json

import numpy as np


def test_sparse_coords_comparison_reports_overlap(tmp_path):
    from scripts.compare_stage_artifacts import compare_stage

    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    np.savez(
        reference,
        coords=np.array([[0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9]], dtype=np.int32),
    )
    np.savez(
        candidate,
        coords=np.array([[0, 4, 5, 6], [0, 7, 8, 9], [0, 9, 9, 9]], dtype=np.int32),
    )

    report = compare_stage("sparse_coords", reference, candidate)

    assert report["schema"] == "trellis2mlx.stage_artifact_comparison.v1"
    assert report["coords"]["reference_count"] == 3
    assert report["coords"]["candidate_count"] == 3
    assert report["coords"]["common_count"] == 2
    assert report["coords"]["reference_only_count"] == 1
    assert report["coords"]["candidate_only_count"] == 1
    assert report["coords"]["jaccard"] == 0.5


def test_shape_slat_comparison_aligns_features_by_common_coords(tmp_path):
    from scripts.compare_stage_artifacts import compare_stage

    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    np.savez(
        reference,
        coords=np.array([[0, 1, 1, 1], [0, 2, 2, 2]], dtype=np.int32),
        feats=np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32),
    )
    np.savez(
        candidate,
        coords=np.array([[0, 3, 3, 3], [0, 2, 2, 2]], dtype=np.int32),
        feats=np.array([[30.0, 30.0], [21.0, 18.0]], dtype=np.float32),
    )

    report = compare_stage("shape_slat", reference, candidate)

    assert report["coords"]["common_count"] == 1
    assert report["features"]["common_shape"] == [1, 2]
    assert report["features"]["max_abs_diff"] == 2.0
    assert report["features"]["mean_abs_diff"] == 1.5


def test_compare_stage_cli_writes_json_report(tmp_path):
    from scripts.compare_stage_artifacts import main

    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    output = tmp_path / "report.json"
    np.savez(reference, cond=np.zeros((1, 2, 3), dtype=np.float32), neg_cond=np.zeros((1, 2, 3), dtype=np.float32))
    np.savez(candidate, cond=np.ones((1, 2, 3), dtype=np.float32), neg_cond=np.zeros((1, 2, 3), dtype=np.float32))

    assert main(["--stage", "conditioning", "--reference", str(reference), "--candidate", str(candidate), "--output", str(output)]) == 0

    report = json.loads(output.read_text())
    assert report["stage"] == "conditioning"
    assert report["arrays"]["cond"]["max_abs_diff"] == 1.0
    assert report["arrays"]["neg_cond"]["max_abs_diff"] == 0.0


def test_sparse_flow_block_trace_comparison_reports_dynamic_block_arrays(tmp_path):
    from scripts.compare_stage_artifacts import compare_stage

    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    np.savez(
        reference,
        trace_block_index=np.array(5, dtype=np.int32),
        pos_block5_after_mlp=np.array([[1.0, 2.0]], dtype=np.float32),
        neg_block5_after_mlp=np.array([[3.0, 4.0]], dtype=np.float32),
    )
    np.savez(
        candidate,
        trace_block_index=np.array(5, dtype=np.int32),
        pos_block5_after_mlp=np.array([[1.5, 1.0]], dtype=np.float32),
        neg_block5_after_mlp=np.array([[2.0, 7.0]], dtype=np.float32),
    )

    report = compare_stage("sparse_flow_block_trace", reference, candidate)

    assert report["arrays"]["trace_block_index"]["max_abs_diff"] == 0.0
    assert report["arrays"]["pos_block5_after_mlp"]["max_abs_diff"] == 1.0
    assert report["arrays"]["neg_block5_after_mlp"]["max_abs_diff"] == 3.0


def test_sparse_flow_steps_comparison_reports_step_arrays(tmp_path):
    from scripts.compare_stage_artifacts import compare_stage

    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    np.savez(
        reference,
        sample_in=np.zeros((2, 1, 1), dtype=np.float32),
        pred_final=np.ones((2, 1, 1), dtype=np.float32),
        sample_next=np.array([[[1.0]], [[2.0]]], dtype=np.float32),
        t=np.array([1.0, 0.5], dtype=np.float32),
    )
    np.savez(
        candidate,
        sample_in=np.ones((2, 1, 1), dtype=np.float32),
        pred_final=np.array([[[1.5]], [[0.5]]], dtype=np.float32),
        sample_next=np.array([[[2.0]], [[2.0]]], dtype=np.float32),
        t=np.array([1.0, 0.25], dtype=np.float32),
    )

    report = compare_stage("sparse_flow_steps", reference, candidate)

    assert report["arrays"]["sample_in"]["max_abs_diff"] == 1.0
    assert report["arrays"]["pred_final"]["max_abs_diff"] == 0.5
    assert report["arrays"]["sample_next"]["mean_abs_diff"] == 0.5
    assert report["arrays"]["t"]["max_abs_diff"] == 0.25
