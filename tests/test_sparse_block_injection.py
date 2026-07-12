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


def test_load_sparse_block_injection_manifest_dispatches_multiple_sites(tmp_path):
    from trellmlx.sparse_block_injection import load_sparse_block_injection_manifest

    trace0 = tmp_path / "source_block0.npz"
    trace4 = tmp_path / "source_block4.npz"
    np.savez(
        trace0,
        pos_block0_modulated_self_input=np.ones((1, 8, 4), dtype=np.float32),
        route_identity_json=np.array('{"effective_route": "source-cuda-block0"}'),
    )
    np.savez(
        trace4,
        pos_block4_after_self=np.ones((1, 8, 4), dtype=np.float32) * 4,
        route_identity_json=np.array('{"effective_route": "source-cuda-block4"}'),
    )
    manifest = tmp_path / "injections.json"
    manifest.write_text(
        """
        {
          "schema": "trellis2mlx.sparse_block_injection_manifest.v1",
          "sites": [
            {
              "trace_path": "source_block0.npz",
              "branch": "pos",
              "step_index": 2,
              "block_index": 0,
              "stage": "modulated_self_input"
            },
            {
              "trace_path": "source_block4.npz",
              "branch": "pos",
              "step_index": 2,
              "block_index": 4,
              "stage": "after_self"
            }
          ]
        }
        """
    )

    injections = load_sparse_block_injection_manifest(manifest)

    assert injections.applies(step_index=2, branch="pos")
    assert not injections.applies(step_index=2, branch="neg")
    active = injections.active_for_step_branch(step_index=2, branch="pos")
    assert active is not None
    assert active.injection_for_block(0).array_key == "pos_block0_modulated_self_input"
    assert active.injection_for_block(4).array_key == "pos_block4_after_self"
    np.testing.assert_allclose(active.injection_for_block(4).array_for_branch("pos"), 4.0)
    assert active.injection_for_block(5) is None
    identity = injections.report_identity()
    assert identity["comparison_class"] == "mlx_sparse_flow_with_named_block_tensor_injection_set"
    assert identity["manifest_path"] == str(manifest)
    assert [site["block_index"] for site in identity["sites"]] == [0, 4]


def test_flow_euler_sample_dispatches_sparse_block_injection_set_by_step_and_branch():
    import mlx.core as mx

    from trellmlx.samplers import flow_euler_sample
    from trellmlx.sparse_block_injection import SparseBlockInjection, SparseBlockInjectionSet

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
                    "site_count": 0 if sparse_block_injection is None else len(sparse_block_injection.sites),
                    "blocks": [] if sparse_block_injection is None else [
                        site.block_index for site in sparse_block_injection.sites
                    ],
                }
            )
            return mx.zeros_like(sample)

    injections = SparseBlockInjectionSet(
        trace_path=None,
        sites=(
            SparseBlockInjection(
                trace_path=None,
                array_key="pos_block0_norm1",
                branch="pos",
                step_index=1,
                block_index=0,
                stage="norm1",
                arrays_by_branch={"pos": np.ones((1, 2, 1), dtype=np.float32)},
                trace_identity={},
            ),
            SparseBlockInjection(
                trace_path=None,
                array_key="pos_block4_after_self",
                branch="pos",
                step_index=1,
                block_index=4,
                stage="after_self",
                arrays_by_branch={"pos": np.ones((1, 2, 1), dtype=np.float32) * 4},
                trace_identity={},
            ),
        ),
        manifest_identity={},
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
        sparse_block_injection=injections,
    )

    assert [call["branch"] for call in model.calls] == ["pos", "neg", "pos", "neg", "pos", "neg"]
    assert [call["site_count"] for call in model.calls] == [0, 0, 2, 0, 0, 0]
    assert model.calls[2]["blocks"] == [0, 4]
