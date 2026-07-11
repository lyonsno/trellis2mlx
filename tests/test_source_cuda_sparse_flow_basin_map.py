import numpy as np


def test_build_perturbed_starts_uses_candidate_minus_source_delta():
    from scripts.source_cuda_sparse_flow_basin_map import build_perturbed_starts

    source_post = np.zeros((1, 2, 2, 2, 2), dtype=np.float32)
    candidate_post = np.ones_like(source_post) * 4.0
    alphas = np.asarray([-0.5, 0.0, 0.25, 1.0], dtype=np.float32)

    starts, delta = build_perturbed_starts(source_post, candidate_post, alphas)

    assert starts.shape == (4, 1, 2, 2, 2, 2)
    assert np.allclose(delta, 4.0)
    assert np.allclose(starts[0], -2.0)
    assert np.allclose(starts[1], 0.0)
    assert np.allclose(starts[2], 1.0)
    assert np.allclose(starts[3], 4.0)


def test_remaining_step_indices_start_after_selected_step():
    from scripts.source_cuda_sparse_flow_basin_map import remaining_step_indices

    assert remaining_step_indices(8, start_after_step_index=2) == [3, 4, 5, 6, 7]


def test_parse_alphas_rejects_empty_input():
    from scripts.source_cuda_sparse_flow_basin_map import parse_alphas

    assert parse_alphas("-0.5,0,0.125,1").tolist() == [-0.5, 0.0, 0.125, 1.0]

    try:
        parse_alphas(" , ")
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("empty alpha list should fail")


def test_alpha_report_records_continuation_elapsed_seconds():
    from scripts.source_cuda_sparse_flow_basin_map import _alpha_report

    final = np.zeros((1, 1, 1, 1, 1), dtype=np.float32)
    report = _alpha_report(
        alpha=0.25,
        elapsed_seconds=1.5,
        final=final,
        source_final=final,
        old_steps=None,
        current_steps=None,
    )

    assert report["alpha"] == 0.25
    assert report["continuation_elapsed_seconds"] == 1.5
    assert report["best_final_anchor"] == "source_cuda"
