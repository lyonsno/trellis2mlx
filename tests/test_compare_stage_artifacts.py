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


def test_shape_flow_block_trace_comparison_separates_metadata_and_reports_key_coverage(
    tmp_path,
):
    from scripts.compare_stage_artifacts import compare_stage

    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    np.savez(
        reference,
        route_identity_json=np.array('{"route":"source-cuda"}'),
        trace_block_index=np.array(0, dtype=np.int32),
        pos_block0_attention_raw=np.array([[1.0, 2.0]], dtype=np.float32),
        source_only_final=np.array([[3.0]], dtype=np.float32),
    )
    np.savez(
        candidate,
        route_identity_json=np.array('{"route":"mlx-source-cuda-attention"}'),
        trace_block_index=np.array(0, dtype=np.int32),
        pos_block0_attention_raw=np.array([[1.0, 2.5]], dtype=np.float32),
        candidate_only_route=np.array("manual"),
    )

    report = compare_stage("shape_flow_block_trace", reference, candidate)

    assert sorted(report["arrays"]) == [
        "pos_block0_attention_raw",
        "trace_block_index",
    ]
    assert report["arrays"]["pos_block0_attention_raw"]["max_abs_diff"] == 0.5
    assert report["metadata"]["route_identity_json"] == {
        "reference_shape": [],
        "candidate_shape": [],
        "shape_match": True,
        "reference_dtype": "<U23",
        "candidate_dtype": "<U37",
        "dtype_match": False,
        "reference": '{"route":"source-cuda"}',
        "candidate": '{"route":"mlx-source-cuda-attention"}',
        "exact_match": False,
    }
    assert report["keys"] == {
        "common": [
            "pos_block0_attention_raw",
            "route_identity_json",
            "trace_block_index",
        ],
        "reference_only": ["source_only_final"],
        "candidate_only": ["candidate_only_route"],
    }


def test_shape_flow_block_trace_comparison_reports_singleton_route_values(tmp_path):
    from scripts.compare_stage_artifacts import compare_stage

    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    np.savez(
        reference,
        route_identity_json=np.array(['{"route":"source-cuda"}']),
    )
    np.savez(
        candidate,
        route_identity_json=np.array(['{"route":"mlx-source-self"}']),
    )

    report = compare_stage("shape_flow_block_trace", reference, candidate)

    assert report["metadata"]["route_identity_json"]["reference"] == (
        '{"route":"source-cuda"}'
    )
    assert report["metadata"]["route_identity_json"]["candidate"] == (
        '{"route":"mlx-source-self"}'
    )
    assert report["metadata"]["route_identity_json"]["exact_match"] is False


def test_shape_flow_block_trace_metadata_is_json_serializable_for_numpy_temporals(
    tmp_path,
):
    from scripts.compare_stage_artifacts import compare_stage

    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    np.savez(
        reference,
        captured_on=np.array(np.datetime64("2026-07-27")),
        duration=np.array(np.timedelta64(3, "D")),
    )
    np.savez(
        candidate,
        captured_on=np.array(np.datetime64("2026-07-28")),
        duration=np.array(np.timedelta64(4, "D")),
    )

    report = compare_stage("shape_flow_block_trace", reference, candidate)

    json.dumps(report)
    assert report["metadata"]["captured_on"]["reference"] == "2026-07-27"
    assert report["metadata"]["captured_on"]["candidate"] == "2026-07-28"
    assert report["metadata"]["duration"]["reference"] == "3 days"
    assert report["metadata"]["duration"]["candidate"] == "4 days"


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


def test_shape_flow_steps_comparison_aligns_token_rows_by_coordinates(tmp_path):
    from scripts.compare_stage_artifacts import build_parser, compare_stage

    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    ref_coords = np.array([[0, 1, 1, 1], [0, 2, 2, 2]], dtype=np.int32)
    cand_coords = ref_coords[::-1].copy()
    ref_tokens = np.array([[10.0], [20.0]], dtype=np.float32)
    cand_tokens = ref_tokens[::-1].copy()
    ref_steps = np.stack([ref_tokens, ref_tokens + 1.0])
    cand_steps = ref_steps[:, ::-1, :].copy()
    scalar_steps = np.ones((2,), dtype=np.float32)

    def write(path, coords, tokens, stepped):
        np.savez(
            path,
            coords=coords,
            coords_3d=coords[:, 1:],
            noise=tokens,
            sample_feats=tokens,
            sample_in=stepped,
            pred_pos=stepped,
            pred_neg=stepped,
            pred_cfg=stepped,
            x0_pos=stepped,
            x0_cfg=stepped,
            std_pos=scalar_steps,
            std_cfg=scalar_steps,
            ratio_raw=scalar_steps,
            std_ratio=scalar_steps,
            ratio_effective=scalar_steps,
            x0_rescaled=stepped,
            x0_after_rescale=stepped,
            pred_final=stepped,
            pred_v_feats=stepped,
            sample_next=stepped,
            t=np.array([1.0, 0.5], dtype=np.float32),
            t_prev=np.array([0.5, 0.0], dtype=np.float32),
            steps=np.array(2, dtype=np.int32),
            guidance_strength=np.array(7.5, dtype=np.float32),
            guidance_rescale=np.array(0.5, dtype=np.float32),
            guidance_interval=np.array([0.6, 1.0], dtype=np.float32),
            rescale_t=np.array(3.0, dtype=np.float32),
            sigma_min=np.array(1e-5, dtype=np.float32),
            shape_flow_block_injection_json=np.array(""),
        )

    write(reference, ref_coords, ref_tokens, ref_steps)
    write(candidate, cand_coords, cand_tokens, cand_steps)

    args = build_parser().parse_args(
        [
            "--stage",
            "shape_flow_steps",
            "--reference",
            str(reference),
            "--candidate",
            str(candidate),
            "--output",
            str(tmp_path / "report.json"),
        ]
    )
    report = compare_stage(args.stage, reference, candidate)

    assert report["coords"]["common_count"] == 2
    assert report["coords"]["exact_order_match"] is False
    assert report["token_alignment"] == "common-coordinate-order"
    assert report["arrays"]["noise"]["max_abs_diff"] == 0.0
    assert report["arrays"]["sample_in"]["max_abs_diff"] == 0.0
    assert report["arrays"]["pred_final"]["max_abs_diff"] == 0.0
    assert report["metadata"]["shape_flow_block_injection_json"]["exact_match"] is True
