import json

import numpy as np
import pytest


def test_load_shape_attention_raw_injection_flattens_cuda_heads_and_records_route(tmp_path):
    from trellmlx.shape_block_injection import load_shape_block_injection

    trace = tmp_path / "source_cuda_attention.npz"
    pos = np.arange(3 * 2 * 4, dtype=np.float32).reshape(1, 3, 2, 4)
    neg = pos + 100
    np.savez_compressed(
        trace,
        pos_block1_attention_raw=pos,
        neg_block1_attention_raw=neg,
        route_identity_json=np.asarray(
            json.dumps(
                {
                    "effective_route": "microsoft-trellis2-cuda-t4",
                    "effective_device_type": "cuda",
                }
            )
        ),
        trace_block_index=np.asarray([1], dtype=np.int32),
        shape_flow_trace_step_index=np.asarray([0], dtype=np.int32),
    )

    injection = load_shape_block_injection(
        trace,
        branch="both",
        step_index=0,
        block_index=1,
        stage="attention_raw",
    )

    assert injection.applies(step_index=0, branch="pos")
    assert injection.applies(step_index=0, branch="neg")
    assert not injection.applies(step_index=1, branch="pos")
    assert injection.array_for_branch("pos").shape == (1, 3, 8)
    np.testing.assert_array_equal(injection.array_for_branch("pos"), pos.reshape(1, 3, 8))
    identity = injection.report_identity()
    assert identity["comparison_class"] == "mlx_shape_flow_with_source_cuda_attention_raw_injection"
    assert identity["source_array_shape_by_branch"] == {
        "neg": [1, 3, 2, 4],
        "pos": [1, 3, 2, 4],
    }
    assert identity["effective_array_shape_by_branch"] == {
        "neg": [1, 3, 8],
        "pos": [1, 3, 8],
    }
    assert identity["trace_identity"]["effective_route"] == "microsoft-trellis2-cuda-t4"
    assert identity["source_delta_scale"] == 1.0


def test_load_shape_attention_raw_injection_rejects_unidentified_or_wrong_site_trace(tmp_path):
    import pytest

    from trellmlx.shape_block_injection import load_shape_block_injection

    trace = tmp_path / "unidentified.npz"
    np.savez_compressed(
        trace,
        pos_block1_attention_raw=np.zeros((1, 3, 2, 4), dtype=np.float32),
        trace_block_index=np.asarray([1], dtype=np.int32),
        shape_flow_trace_step_index=np.asarray([0], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="route_identity_json"):
        load_shape_block_injection(
            trace, branch="pos", step_index=0, block_index=1, stage="attention_raw"
        )

    np.savez_compressed(
        trace,
        pos_block1_attention_raw=np.zeros((1, 3, 2, 4), dtype=np.float32),
        route_identity_json=np.asarray(
            json.dumps({"effective_route": "source-cuda", "effective_device_type": "cuda"})
        ),
        trace_block_index=np.asarray([1], dtype=np.int32),
        shape_flow_trace_step_index=np.asarray([3], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="step index 3.*requested 0"):
        load_shape_block_injection(
            trace, branch="pos", step_index=0, block_index=1, stage="attention_raw"
        )


def test_load_shape_block_injection_selects_member_of_multi_block_cuda_trace(tmp_path):
    from trellmlx.shape_block_injection import load_shape_block_injection

    trace = tmp_path / "source_cuda_boundaries.npz"
    block23 = np.arange(3 * 4, dtype=np.float32).reshape(1, 3, 4)
    np.savez_compressed(
        trace,
        pos_block23_after_mlp=block23,
        route_identity_json=np.asarray(
            json.dumps(
                {
                    "effective_route": "microsoft-trellis2-cuda-t4",
                    "effective_device_type": "cuda",
                    "shape_flow_trace_block_indices": [19, 23, 27],
                }
            )
        ),
        trace_block_indices=np.asarray([19, 23, 27], dtype=np.int32),
        trace_block_index=np.asarray([19], dtype=np.int32),
        shape_flow_trace_step_index=np.asarray([0], dtype=np.int32),
    )

    injection = load_shape_block_injection(
        trace,
        branch="pos",
        step_index=0,
        block_index=23,
        stage="after_mlp",
    )

    np.testing.assert_array_equal(injection.array_for_branch("pos"), block23)
    assert injection.trace_identity["shape_flow_trace_block_indices"] == [19, 23, 27]


def test_load_shape_block_injection_rejects_multi_block_route_identity_mismatch(tmp_path):
    from trellmlx.shape_block_injection import load_shape_block_injection

    trace = tmp_path / "misidentified_boundaries.npz"
    np.savez_compressed(
        trace,
        pos_block19_after_mlp=np.ones((1, 3, 4), dtype=np.float32),
        route_identity_json=np.asarray(
            json.dumps(
                {
                    "effective_route": "microsoft-trellis2-cuda-t4",
                    "effective_device_type": "cuda",
                    "shape_flow_trace_block_indices": [23],
                }
            )
        ),
        trace_block_indices=np.asarray([19], dtype=np.int32),
        trace_block_index=np.asarray([19], dtype=np.int32),
        shape_flow_trace_step_index=np.asarray([0], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="block indices.*route identity"):
        load_shape_block_injection(
            trace,
            branch="pos",
            step_index=0,
            block_index=19,
            stage="after_mlp",
        )


def test_load_shape_norm1_injection_preserves_hidden_shape_and_records_stage(tmp_path):
    from trellmlx.shape_block_injection import load_shape_block_injection

    trace = tmp_path / "source_cuda_block0.npz"
    pos = np.arange(3 * 4, dtype=np.float32).reshape(1, 3, 4)
    neg = pos + 100
    np.savez_compressed(
        trace,
        pos_block0_norm1=pos,
        neg_block0_norm1=neg,
        route_identity_json=np.asarray(
            json.dumps(
                {
                    "effective_route": "microsoft-trellis2-cuda-t4",
                    "effective_device_type": "cuda",
                }
            )
        ),
        trace_block_index=np.asarray([0], dtype=np.int32),
        shape_flow_trace_step_index=np.asarray([0], dtype=np.int32),
    )

    injection = load_shape_block_injection(
        trace,
        branch="both",
        step_index=0,
        block_index=0,
        stage="norm1",
    )

    np.testing.assert_array_equal(injection.array_for_branch("pos"), pos)
    np.testing.assert_array_equal(injection.array_for_branch("neg"), neg)
    identity = injection.report_identity()
    assert identity["stage"] == "norm1"
    assert identity["comparison_class"] == "mlx_shape_flow_with_source_cuda_block_stage_injection"
    assert identity["effective_array_shape_by_branch"] == {
        "neg": [1, 3, 4],
        "pos": [1, 3, 4],
    }


def test_stage_capture_accepts_shape_norm1_injection_route(tmp_path):
    from scripts.run_mlx_stage_capture import _build_generate_command, build_parser

    image = tmp_path / "input.png"
    image.write_bytes(b"input")
    trace = tmp_path / "source_cuda_block0.npz"
    trace.write_bytes(b"trace")
    args = build_parser().parse_args(
        [
            "--image", str(image),
            "--output-dir", str(tmp_path / "out"),
            "--stop-after-stage", "shape_flow_step",
            "--shape-flow-block-injection-trace", str(trace),
            "--shape-flow-block-injection-block-index", "0",
            "--shape-flow-block-injection-stage", "norm1",
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")

    assert command[command.index("--shape-flow-block-injection-stage") + 1] == "norm1"


def test_load_shape_block_injection_manifest_routes_multiple_source_cuda_sites(tmp_path):
    from trellmlx.shape_block_injection import load_shape_block_injection_manifest

    route_identity = np.asarray(
        json.dumps(
            {
                "effective_route": "microsoft-trellis2-cuda-t4",
                "effective_device_type": "cuda",
            }
        )
    )
    block0 = tmp_path / "block0.npz"
    np.savez_compressed(
        block0,
        pos_block0_norm1=np.ones((1, 3, 4), dtype=np.float32),
        neg_block0_norm1=np.ones((1, 3, 4), dtype=np.float32) * 2,
        route_identity_json=route_identity,
        trace_block_index=np.asarray([0], dtype=np.int32),
        shape_flow_trace_step_index=np.asarray([0], dtype=np.int32),
    )
    block1 = tmp_path / "block1.npz"
    np.savez_compressed(
        block1,
        pos_block1_attention_raw=np.ones((1, 3, 1, 4), dtype=np.float32) * 3,
        neg_block1_attention_raw=np.ones((1, 3, 1, 4), dtype=np.float32) * 4,
        route_identity_json=route_identity,
        trace_block_index=np.asarray([1], dtype=np.int32),
        shape_flow_trace_step_index=np.asarray([0], dtype=np.int32),
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "trellis2mlx.shape_block_injection_manifest.v1",
                "sites": [
                    {
                        "trace_path": "block0.npz",
                        "branch": "both",
                        "step_index": 0,
                        "block_index": 0,
                        "stage": "norm1",
                    },
                    {
                        "trace_path": "block1.npz",
                        "branch": "both",
                        "step_index": 0,
                        "block_index": 1,
                        "stage": "attention_raw",
                        "source_delta_scale": 0.25,
                    },
                ],
            }
        )
    )

    injection_set = load_shape_block_injection_manifest(manifest)
    active = injection_set.active_for_step_branch(step_index=0, branch="pos")

    assert active is not None
    assert active.injection_for_block(0).stage == "norm1"
    assert active.injection_for_block(1).stage == "attention_raw"
    assert active.injection_for_block(1).source_delta_scale == 0.25
    assert injection_set.active_for_step_branch(step_index=1, branch="pos") is None
    identity = injection_set.report_identity()
    assert identity["comparison_class"] == "mlx_shape_flow_with_source_cuda_block_stage_injection_set"
    assert identity["manifest_sha256"]
    assert len(identity["sites"]) == 2


def test_stage_capture_records_and_forwards_shape_injection_manifest(tmp_path):
    from scripts.run_mlx_stage_capture import _build_generate_command, build_parser, build_route_identity

    image = tmp_path / "input.png"
    image.write_bytes(b"input")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"sites": []}')
    args = build_parser().parse_args(
        [
            "--image", str(image),
            "--output-dir", str(tmp_path / "out"),
            "--stop-after-stage", "shape_flow_step",
            "--shape-flow-block-injection-manifest", str(manifest),
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")
    identity = build_route_identity(args, command)["route"]

    assert command[command.index("--shape-flow-block-injection-manifest") + 1] == str(manifest)
    assert identity["shape_flow_block_injection_manifest_path"] == str(manifest)
    assert identity["shape_flow_block_injection_manifest_sha256"]

@pytest.mark.parametrize(
    ("source_delta_scale", "expected_raw"),
    [(0.0, 7.0), (0.5, 4.5), (1.0, 2.0)],
)
def test_modulated_block_injects_attention_raw_before_output_projection(
    source_delta_scale, expected_raw
):
    import mlx.core as mx

    from trellmlx.models.sparse_structure_flow import ModulatedBlock
    from trellmlx.shape_block_injection import ShapeBlockInjection

    class RecordingSelfAttention:
        def __init__(self):
            self.to_out_input = None

        def trace_self_attention(self, x, rope_phases=None, trace_prefix="block0"):
            raw = mx.ones_like(x) * 7
            return mx.ones_like(x) * 99, {f"{trace_prefix}_attention_raw": raw}

        def to_out(self, x):
            self.to_out_input = x
            return x * 3

    class ZeroCrossAttention:
        def __call__(self, x, context=None, cached_kv=None):
            return mx.zeros_like(x)

    class ZeroMLP:
        def __call__(self, x):
            return mx.zeros_like(x)

    block = ModulatedBlock(channels=4, num_heads=1, context_channels=4, mlp_hidden=4)
    recorder = RecordingSelfAttention()
    block.self_attn = recorder
    block.cross_attn = ZeroCrossAttention()
    block.mlp = ZeroMLP()
    injected = np.full((1, 2, 4), 2.0, dtype=np.float32)
    injection = ShapeBlockInjection(
        trace_path=None,
        array_key="pos_block0_attention_raw",
        branch="pos",
        step_index=0,
        block_index=0,
        stage="attention_raw",
        arrays_by_branch={"pos": injected},
        source_shapes_by_branch={"pos": injected.shape},
        trace_identity={},
        source_delta_scale=source_delta_scale,
    )
    mod = np.zeros((24,), dtype=np.float32)
    mod[8:12] = 1.0  # gate_msa

    out = block.forward_with_injection(
        mx.zeros((2, 4), dtype=mx.float32),
        mx.array(mod),
        mx.zeros((1, 1, 4), dtype=mx.float32),
        injection=injection,
        branch="pos",
    )
    mx.eval(out, recorder.to_out_input)

    np.testing.assert_array_equal(
        np.array(recorder.to_out_input), np.full((2, 4), expected_raw)
    )
    np.testing.assert_array_equal(np.array(out), np.full((2, 4), expected_raw * 3))


def test_slat_flow_routes_shape_injection_to_named_block():
    import mlx.core as mx

    from trellmlx.models.slat_flow import SLatFlowModel
    from trellmlx.shape_block_injection import ShapeBlockInjection

    class RecordingBlock:
        def __init__(self, index):
            self.index = index
            self.injected = False

        def __call__(self, x, mod, cond, rope_phases=None, cross_kv_cache=None):
            return x

        def forward_with_injection(
            self, x, mod, cond, *, injection, branch, rope_phases=None, cross_kv_cache=None
        ):
            self.injected = True
            assert injection.block_index == self.index
            assert branch == "neg"
            return x

    model = SLatFlowModel(
        in_channels=2,
        out_channels=2,
        model_channels=4,
        num_heads=1,
        num_blocks=2,
        mlp_hidden=8,
        context_channels=4,
    )
    blocks = [RecordingBlock(0), RecordingBlock(1)]
    model.blocks = blocks
    injection = ShapeBlockInjection(
        trace_path=None,
        array_key="neg_block1_attention_raw",
        branch="neg",
        step_index=0,
        block_index=1,
        stage="attention_raw",
        arrays_by_branch={"neg": np.zeros((1, 2, 4), dtype=np.float32)},
        source_shapes_by_branch={"neg": (1, 2, 1, 4)},
        trace_identity={},
    )

    model(
        mx.zeros((2, 2), dtype=mx.float32),
        mx.array([1000.0], dtype=mx.float32),
        mx.zeros((1, 1, 4), dtype=mx.float32),
        shape_block_injection=injection,
        shape_block_injection_branch="neg",
    )

    assert not blocks[0].injected
    assert blocks[1].injected


def test_flow_sampler_dispatches_shape_injection_only_at_named_step_and_branch():
    import mlx.core as mx

    from trellmlx.samplers import flow_euler_sample
    from trellmlx.shape_block_injection import ShapeBlockInjection

    class RecordingModel:
        def __init__(self):
            self.calls = []

        def __call__(
            self,
            sample,
            t,
            cond,
            *,
            shape_block_injection=None,
            shape_block_injection_branch=None,
        ):
            self.calls.append(
                (shape_block_injection_branch, shape_block_injection is not None)
            )
            return mx.zeros_like(sample)

    injection = ShapeBlockInjection(
        trace_path=None,
        array_key="pos_block1_attention_raw,neg_block1_attention_raw",
        branch="both",
        step_index=1,
        block_index=1,
        stage="attention_raw",
        arrays_by_branch={
            "pos": np.zeros((1, 2, 4), dtype=np.float32),
            "neg": np.zeros((1, 2, 4), dtype=np.float32),
        },
        source_shapes_by_branch={"pos": (1, 2, 1, 4), "neg": (1, 2, 1, 4)},
        trace_identity={},
    )
    model = RecordingModel()

    flow_euler_sample(
        model,
        mx.zeros((2, 4), dtype=mx.float32),
        mx.zeros((1, 1, 4), dtype=mx.float32),
        mx.zeros((1, 1, 4), dtype=mx.float32),
        steps=3,
        guidance_strength=7.5,
        guidance_rescale=0.0,
        verbose=False,
        shape_block_injection=injection,
    )

    assert model.calls == [
        ("pos", False),
        ("neg", False),
        ("pos", True),
        ("neg", True),
        ("pos", False),
        ("neg", False),
    ]


def test_stage_capture_records_and_forwards_shape_injection_route(tmp_path):
    from scripts.run_mlx_stage_capture import _build_generate_command, build_parser, build_route_identity

    image = tmp_path / "input.png"
    image.write_bytes(b"input")
    trace = tmp_path / "source_cuda_attention.npz"
    np.savez_compressed(
        trace,
        pos_block1_attention_raw=np.zeros((1, 3, 2, 4), dtype=np.float32),
        neg_block1_attention_raw=np.zeros((1, 3, 2, 4), dtype=np.float32),
    )
    args = build_parser().parse_args(
        [
            "--image", str(image),
            "--output-dir", str(tmp_path / "out"),
            "--stop-after-stage", "shape_flow_step",
            "--shape-flow-block-injection-trace", str(trace),
            "--shape-flow-block-injection-step-index", "0",
            "--shape-flow-block-injection-block-index", "1",
            "--shape-flow-block-injection-branch", "both",
            "--shape-flow-block-injection-stage", "attention_raw",
            "--shape-flow-block-injection-scale", "0.25",
        ]
    )
    command = _build_generate_command(args, tmp_path / "checkpoints")
    identity = build_route_identity(args, command)

    assert command[command.index("--shape-flow-block-injection-trace") + 1] == str(trace)
    route = identity["route"]
    assert route["shape_flow_block_injection_trace_path"] == str(trace)
    assert route["shape_flow_block_injection_trace_sha256"]
    assert route["shape_flow_block_injection_step_index"] == 0
    assert route["shape_flow_block_injection_block_index"] == 1
    assert route["shape_flow_block_injection_branch"] == "both"
    assert route["shape_flow_block_injection_stage"] == "attention_raw"
    assert route["shape_flow_block_injection_scale"] == 0.25
    assert command[command.index("--shape-flow-block-injection-scale") + 1] == "0.25"
