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
