import numpy as np


def test_load_sparse_block_injection_both_branch_defaults_to_branch_specific_keys(tmp_path):
    from trellmlx.sparse_block_injection import load_sparse_block_injection

    trace = tmp_path / "source_trace.npz"
    np.savez(
        trace,
        pos_block0_modulated_self_input=np.ones((1, 8, 4), dtype=np.float32),
        neg_block0_modulated_self_input=np.ones((1, 8, 4), dtype=np.float32) * 2,
        route_identity_json=np.array('{"effective_route": "source-cuda-block0-trace"}'),
    )

    injection = load_sparse_block_injection(
        trace,
        branch="both",
        step_index=2,
        block_index=0,
        stage="modulated_self_input",
        array_key=None,
    )

    assert injection.array_key == "pos_block0_modulated_self_input,neg_block0_modulated_self_input"
    assert injection.applies(step_index=2, branch="pos")
    assert injection.applies(step_index=2, branch="neg")
    assert not injection.applies(step_index=3, branch="pos")
    np.testing.assert_allclose(injection.array_for_branch("pos"), 1.0)
    np.testing.assert_allclose(injection.array_for_branch("neg"), 2.0)
    identity = injection.report_identity()
    assert identity["comparison_class"] == "mlx_sparse_flow_with_named_block_tensor_injection"
    assert identity["array_shape_by_branch"] == {"neg": [1, 8, 4], "pos": [1, 8, 4]}
    assert identity["trace_identity"] == {"effective_route": "source-cuda-block0-trace"}


def test_flow_euler_sample_dispatches_sparse_block_injection_by_step_and_branch():
    import mlx.core as mx

    from trellmlx.samplers import flow_euler_sample
    from trellmlx.sparse_block_injection import SparseBlockInjection

    class RecordingModel:
        def __init__(self):
            self.calls = []

        def __call__(
            self,
            sample,
            t,
            cond,
            *,
            sparse_block_injection=None,
            sparse_block_injection_branch=None,
        ):
            self.calls.append(
                {
                    "branch": sparse_block_injection_branch,
                    "has_injection": sparse_block_injection is not None,
                    "t": float(np.array(t)[0]),
                }
            )
            return mx.zeros_like(sample)

    injection = SparseBlockInjection(
        trace_path=None,
        array_key="pos_block0_norm1,neg_block0_norm1",
        branch="both",
        step_index=1,
        block_index=0,
        stage="norm1",
        arrays_by_branch={
            "pos": np.ones((1, 2, 1), dtype=np.float32),
            "neg": np.ones((1, 2, 1), dtype=np.float32) * 2,
        },
        trace_identity={},
    )

    model = RecordingModel()
    flow_euler_sample(
        model,
        mx.zeros((1, 1, 2, 1, 1), dtype=mx.float32),
        mx.zeros((1, 1, 1), dtype=mx.float32),
        mx.zeros((1, 1, 1), dtype=mx.float32),
        steps=3,
        guidance_strength=7.5,
        verbose=False,
        sparse_block_injection=injection,
    )

    assert [call["branch"] for call in model.calls] == ["pos", "neg", "pos", "neg", "pos", "neg"]
    assert [call["has_injection"] for call in model.calls] == [False, False, True, True, False, False]
