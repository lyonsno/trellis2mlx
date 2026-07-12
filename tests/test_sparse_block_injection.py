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


def test_load_sparse_layernorm_correction_uses_improved_rows_from_boundary_report(tmp_path):
    from trellmlx.sparse_block_injection import load_sparse_layernorm_correction

    report = tmp_path / "block0_norm1_boundary_probe.json"
    report.write_text(
        """
        {
          "schema": "trellis2mlx.noaffine_layernorm_boundary_probe.v1",
          "status": "ok",
          "rowwise_perturbation_probe": {
            "scale": {
              "tokens": [
                {
                  "batch": 0,
                  "token": 7,
                  "improved": true,
                  "solved": true,
                  "best": {"value": 0.00025}
                },
                {
                  "batch": 0,
                  "token": 9,
                  "improved": false,
                  "solved": false,
                  "best": {"value": -0.0005}
                }
              ]
            },
            "bias": {"tokens": []}
          }
        }
        """
    )

    correction = load_sparse_layernorm_correction(
        report,
        branch="pos",
        step_index=2,
        block_index=0,
        mode="scale",
    )

    assert correction.stage == "norm1_rowwise_scale"
    assert correction.applies(step_index=2, branch="pos")
    assert not correction.applies(step_index=2, branch="neg")
    np.testing.assert_allclose(
        correction.array_for_branch("pos"),
        np.array([[0, 7, 0.00025]], dtype=np.float32),
    )
    identity = correction.report_identity()
    assert identity["comparison_class"] == "mlx_sparse_flow_with_rowwise_layernorm_correction"
    assert identity["report_path"] == str(report)
    assert identity["mode"] == "scale"
    assert identity["row_count_by_branch"] == {"pos": 1}


def test_modulated_block_applies_rowwise_layernorm_correction_before_self_attention():
    import mlx.core as mx

    from trellmlx.models.sparse_structure_flow import ModulatedBlock, _layernorm_noaffine
    from trellmlx.sparse_block_injection import SparseLayerNormCorrection

    class RecordingSelfAttention:
        def __init__(self):
            self.seen = None

        def __call__(self, x, context=None, rope_phases=None, cached_kv=None):
            self.seen = x
            return mx.zeros_like(x)

    block = ModulatedBlock(channels=4, num_heads=1, context_channels=4, mlp_hidden=4)
    recorder = RecordingSelfAttention()
    block.self_attn = recorder
    x = mx.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 1.0, -1.0, -2.0],
            [0.5, -0.25, 0.75, -1.0],
        ],
        dtype=mx.bfloat16,
    )
    correction = SparseLayerNormCorrection(
        report_path=None,
        branch="pos",
        step_index=2,
        block_index=0,
        stage="norm1_rowwise_scale",
        mode="scale",
        arrays_by_branch={"pos": np.array([[0, 1, 0.01]], dtype=np.float32)},
        source_report_identity={},
    )

    block.forward_with_injection(
        x,
        mx.zeros((24,), dtype=mx.bfloat16),
        mx.zeros((1, 1, 4), dtype=mx.bfloat16),
        injection=correction,
        branch="pos",
    )

    base = _layernorm_noaffine(x)
    xf = x.astype(mx.float32)
    mean = mx.mean(xf, axis=-1, keepdims=True)
    var = mx.mean((xf - mean) * (xf - mean), axis=-1, keepdims=True)
    normalized = (xf - mean) * mx.rsqrt(var + 1e-6)
    expected_row = (normalized[1] * mx.array(1.01, dtype=mx.float32)).astype(mx.bfloat16)
    mx.eval(recorder.seen, base, expected_row)

    seen_np = np.array(recorder.seen.astype(mx.float32))
    base_np = np.array(base.astype(mx.float32))
    expected_row_np = np.array(expected_row.astype(mx.float32))
    np.testing.assert_array_equal(seen_np[0], base_np[0])
    np.testing.assert_array_equal(seen_np[2], base_np[2])
    np.testing.assert_array_equal(seen_np[1], expected_row_np)


def test_sparse_flow_trace_block_records_effective_sparse_block_injection():
    import mlx.core as mx

    from trellmlx.models.sparse_structure_flow import SparseStructureFlowModel
    from trellmlx.sparse_block_injection import SparseBlockInjection

    model = SparseStructureFlowModel(
        in_channels=2,
        out_channels=2,
        model_channels=4,
        num_heads=1,
        num_blocks=1,
        mlp_hidden=8,
        context_channels=4,
    )
    injected = np.arange(8 * 4, dtype=np.float32).reshape(1, 8, 4) / 10.0
    injection = SparseBlockInjection(
        trace_path=None,
        array_key="pos_block0_modulated_self_input",
        branch="pos",
        step_index=2,
        block_index=0,
        stage="modulated_self_input",
        arrays_by_branch={"pos": injected},
        trace_identity={},
    )

    trace = model.trace_block(
        mx.zeros((1, 2, 2, 2, 2), dtype=mx.float32),
        mx.array([500.0], dtype=mx.float32),
        mx.zeros((1, 1, 4), dtype=mx.float32),
        block_index=0,
        sparse_block_injection=injection,
        sparse_block_injection_branch="pos",
    )
    mx.eval(trace["block0_modulated_self_input"])

    expected = mx.array(injected.reshape(8, 4), dtype=trace["block0_modulated_self_input"].dtype)
    mx.eval(expected)
    np.testing.assert_allclose(
        np.array(trace["block0_modulated_self_input"].astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
    )


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
