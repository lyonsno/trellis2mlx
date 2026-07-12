import numpy as np
import pytest


def test_perturbed_full_decode_runner_requires_single_alpha():
    from scripts.source_cuda_perturbed_sparse_full_decode import build_parser

    args = build_parser().parse_args(
        [
            "--output-json",
            "out.json",
            "--output-npz",
            "out.npz",
            "--alpha",
            "1.0",
        ]
    )

    assert args.alpha == 1.0
    assert args.start_after_step_index == 2
    assert args.block_injection_trace is None


def test_perturbed_full_decode_runner_accepts_named_block_injection_route():
    from scripts.source_cuda_perturbed_sparse_full_decode import build_parser

    args = build_parser().parse_args(
        [
            "--output-json",
            "out.json",
            "--output-npz",
            "out.npz",
            "--alpha",
            "1.0",
            "--block-injection-trace",
            "mlx_block0_trace.npz",
            "--block-injection-step-index",
            "2",
            "--block-injection-block-index",
            "0",
            "--block-injection-branch",
            "pos",
            "--block-injection-stage",
            "modulated_self_input",
        ]
    )

    assert args.block_injection_trace.name == "mlx_block0_trace.npz"
    assert args.block_injection_step_index == 2
    assert args.block_injection_block_index == 0
    assert args.block_injection_branch == "pos"
    assert args.block_injection_stage == "modulated_self_input"


def test_perturbed_full_decode_requires_pipeline_models_for_512():
    from scripts.source_cuda_perturbed_sparse_full_decode import required_model_names

    assert required_model_names("512") == (
        "sparse_structure_decoder",
        "sparse_structure_flow_model",
        "shape_slat_decoder",
        "shape_slat_flow_model_512",
        "tex_slat_decoder",
        "tex_slat_flow_model_512",
    )


def test_build_single_perturbed_start_records_delta_identity():
    from scripts.source_cuda_perturbed_sparse_full_decode import build_single_perturbed_start

    source_post = np.zeros((1, 2, 2, 2, 2), dtype=np.float32)
    candidate_post = np.ones_like(source_post) * 3.0

    start, delta = build_single_perturbed_start(source_post, candidate_post, alpha=0.25)

    assert start.shape == source_post.shape
    assert np.allclose(delta, 3.0)
    assert np.allclose(start, 0.75)


def test_build_single_perturbed_start_rejects_shape_mismatch():
    from scripts.source_cuda_perturbed_sparse_full_decode import build_single_perturbed_start

    with pytest.raises(ValueError, match="source/candidate post-step shapes differ"):
        build_single_perturbed_start(
            np.zeros((1, 2, 2, 2, 2), dtype=np.float32),
            np.zeros((1, 2, 2, 2, 3), dtype=np.float32),
            alpha=1.0,
        )


def test_sparse_flow_decode_coords_reorder_matches_source_pipeline():
    import torch

    from scripts.source_cuda_perturbed_sparse_full_decode import decoded_mask_to_source_coords

    decoded = torch.zeros((1, 1, 4, 4, 4), dtype=torch.bool)
    decoded[0, 0, 1, 2, 3] = True

    coords = decoded_mask_to_source_coords(decoded, resolution=4)

    assert coords.dtype == torch.int32
    assert coords.tolist() == [[0, 1, 2, 3]]


def test_load_block_injection_defaults_to_branch_block_stage_key(tmp_path):
    from scripts.source_cuda_perturbed_sparse_full_decode import load_block_injection

    trace = tmp_path / "trace.npz"
    np.savez(
        trace,
        pos_block0_modulated_self_input=np.ones((1, 8, 4), dtype=np.float32),
        route_identity_json=np.array('{"effective_route": "mlx-captured-block-trace"}'),
    )

    injection = load_block_injection(
        trace,
        branch="pos",
        step_index=2,
        block_index=0,
        stage="modulated_self_input",
        array_key=None,
    )

    assert injection.array_key == "pos_block0_modulated_self_input"
    assert injection.applies(step_index=2, branch="pos", block_index=0)
    assert not injection.applies(step_index=2, branch="neg", block_index=0)
    assert injection.array.shape == (1, 8, 4)
    assert injection.trace_identity["effective_route"] == "mlx-captured-block-trace"


def test_load_block_injection_records_explicit_array_key_and_both_branch(tmp_path):
    from scripts.source_cuda_perturbed_sparse_full_decode import load_block_injection

    trace = tmp_path / "trace.npz"
    np.savez(trace, custom_after_self=np.zeros((8, 4), dtype=np.float32))

    injection = load_block_injection(
        trace,
        branch="both",
        step_index=3,
        block_index=1,
        stage="after_self",
        array_key="custom_after_self",
    )

    assert injection.array_key == "custom_after_self"
    assert injection.applies(step_index=3, branch="pos", block_index=1)
    assert injection.applies(step_index=3, branch="neg", block_index=1)
    assert not injection.applies(step_index=4, branch="pos", block_index=1)


def test_load_block_injection_rejects_missing_array(tmp_path):
    from scripts.source_cuda_perturbed_sparse_full_decode import load_block_injection

    trace = tmp_path / "trace.npz"
    np.savez(trace, pos_block0_norm1=np.zeros((1, 8, 4), dtype=np.float32))

    with pytest.raises(KeyError, match="pos_block0_modulated_self_input"):
        load_block_injection(
            trace,
            branch="pos",
            step_index=2,
            block_index=0,
            stage="modulated_self_input",
            array_key=None,
        )
