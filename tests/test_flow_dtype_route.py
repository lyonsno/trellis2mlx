"""Source-parity contracts for TRELLIS.2 flow model dtype routing."""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


class _TinyWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = mx.zeros((2, 2), dtype=mx.float32)


class _TinyBFloat16Weights(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = mx.zeros((2, 2), dtype=mx.bfloat16)


class _RecordingBlock:
    def __init__(self):
        self.calls = []

    def __call__(self, x, mod, cond, rope_phases=None, cross_kv_cache=None):
        self.calls.append(
            {
                "x": x.dtype,
                "mod": mod.dtype,
                "cond": cond.dtype,
                "rope_phases": None if rope_phases is None else rope_phases.dtype,
                "cross_kv_cache": cross_kv_cache,
            }
        )
        return x


def test_weight_loader_preserves_source_bfloat16_tensors(tmp_path):
    """TRELLIS.2 bf16 checkpoints must not be silently demoted when the destination is bf16."""
    from trellmlx.weight_loader import load_weights

    checkpoint = tmp_path / "tiny-bf16.safetensors"
    mx.save_safetensors(
        str(checkpoint),
        {"weight": mx.ones((2, 2), dtype=mx.bfloat16)},
    )

    model = _TinyBFloat16Weights()
    load_weights(model, str(checkpoint), verbose=False)

    assert model.weight.dtype == mx.bfloat16


def test_weight_loader_respects_existing_float32_destination_dtype(tmp_path):
    """PyTorch load_state_dict copies checkpoint values into the existing parameter dtype."""
    from trellmlx.weight_loader import load_weights

    checkpoint = tmp_path / "tiny-bf16.safetensors"
    mx.save_safetensors(
        str(checkpoint),
        {"weight": mx.ones((2, 2), dtype=mx.bfloat16)},
    )

    model = _TinyWeights()
    load_weights(model, str(checkpoint), verbose=False)

    assert model.weight.dtype == mx.float32


def test_sparse_structure_flow_uses_source_mixed_dtype_boundary():
    from trellmlx.models.sparse_structure_flow import SparseStructureFlowModel

    model = SparseStructureFlowModel(
        in_channels=2,
        out_channels=2,
        model_channels=12,
        num_heads=3,
        num_blocks=1,
        mlp_hidden=16,
        context_channels=4,
        resolution=2,
    )

    assert model.input_layer.weight.dtype == mx.float32
    assert model.t_embedder.mlp_0.weight.dtype == mx.float32
    assert model.adaLN_modulation.layers[1].weight.dtype == mx.float32
    assert model.out_layer.weight.dtype == mx.float32
    assert model.blocks[0].self_attn.to_qkv.weight.dtype == mx.bfloat16
    assert model.blocks[0].cross_attn.to_kv.weight.dtype == mx.bfloat16
    assert model.blocks[0].mlp.mlp_0.weight.dtype == mx.bfloat16
    assert model.blocks[0].modulation.dtype == mx.bfloat16
    assert model.blocks[0].self_attn.q_rms_norm.gamma.dtype == mx.float32


def test_slat_flow_uses_source_mixed_dtype_boundary():
    from trellmlx.models.slat_flow import SLatFlowModel

    model = SLatFlowModel(
        in_channels=2,
        out_channels=2,
        model_channels=12,
        num_heads=3,
        num_blocks=1,
        mlp_hidden=16,
        context_channels=4,
    )

    assert model.input_layer.weight.dtype == mx.float32
    assert model.t_embedder.mlp_0.weight.dtype == mx.float32
    assert model.adaLN_modulation.layers[1].weight.dtype == mx.float32
    assert model.out_layer.weight.dtype == mx.float32
    assert model.blocks[0].self_attn.to_qkv.weight.dtype == mx.bfloat16
    assert model.blocks[0].cross_attn.to_kv.weight.dtype == mx.bfloat16
    assert model.blocks[0].mlp.mlp_0.weight.dtype == mx.bfloat16
    assert model.blocks[0].modulation.dtype == mx.bfloat16
    assert model.blocks[0].self_attn.q_rms_norm.gamma.dtype == mx.float32


def test_modulated_block_casts_shared_modulation_to_timestep_dtype():
    from trellmlx.models.sparse_structure_flow import ModulatedBlock

    seen = {}

    class _CaptureSelfAttention:
        def __call__(self, x, rope_phases=None):
            seen["self_attn_input_dtype"] = x.dtype
            return mx.zeros_like(x)

    class _ZeroCrossAttention:
        def __call__(self, x, context=None, cached_kv=None):
            return mx.zeros_like(x)

    class _ZeroMLP:
        def __call__(self, x):
            return mx.zeros_like(x)

    block = ModulatedBlock(channels=4, num_heads=1, context_channels=4, mlp_hidden=8)
    block.self_attn = _CaptureSelfAttention()
    block.cross_attn = _ZeroCrossAttention()
    block.mlp = _ZeroMLP()
    block.modulation = mx.ones((24,), dtype=mx.float32)

    x = mx.ones((2, 4), dtype=mx.bfloat16)
    mod = mx.zeros((24,), dtype=mx.bfloat16)
    context = mx.zeros((1, 1, 4), dtype=mx.bfloat16)
    out = block(x, mod, context)
    mx.eval(out)

    assert seen["self_attn_input_dtype"] == mx.bfloat16


def test_layernorm32_affine_preserves_input_dtype():
    from trellmlx.modules.norm import LayerNorm32

    norm = LayerNorm32(4, affine=True)
    x = mx.random.normal((3, 4)).astype(mx.bfloat16)

    out = norm(x)
    mx.eval(out)

    assert out.dtype == mx.bfloat16


def test_noaffine_layernorm_uses_source_explicit_bfloat16_rounding():
    from trellmlx.models.sparse_structure_flow import _layernorm_noaffine

    mx.random.seed(35)
    x = mx.random.normal((4, 1536), dtype=mx.float32).astype(mx.bfloat16)

    out = _layernorm_noaffine(x)
    xf = x.astype(mx.float32)
    mean = mx.mean(xf, axis=-1, keepdims=True)
    var = mx.mean((xf - mean) * (xf - mean), axis=-1, keepdims=True)
    expected = ((xf - mean) * mx.rsqrt(var + 1e-6)).astype(mx.bfloat16)
    mx.eval(out, expected)

    assert out.dtype == mx.bfloat16
    assert mx.allclose(out, expected, atol=0.0, rtol=0.0).item()


def test_feedforward_uses_source_tanh_gelu_for_bfloat16_torso():
    from trellmlx.models.sparse_structure_flow import FeedForward

    mlp = FeedForward(4, 4)
    mlp.mlp_0.weight = mx.eye(4).astype(mx.bfloat16)
    mlp.mlp_0.bias = mx.zeros((4,)).astype(mx.bfloat16)
    mlp.mlp_2.weight = mx.eye(4).astype(mx.bfloat16)
    mlp.mlp_2.bias = mx.zeros((4,)).astype(mx.bfloat16)

    x = mx.array([[-3.0, -1.0, 1.0, 3.0]], dtype=mx.bfloat16)
    out = mlp(x)
    expected = mx.array(
        [[-0.0036315918, -0.15917969, 0.83984375, 3.0]],
        dtype=mx.float32,
    ).astype(mx.bfloat16)
    mx.eval(out, expected)

    assert out.dtype == mx.bfloat16
    assert mx.allclose(out, expected, atol=0.0, rtol=0.0).item()


def test_sparse_structure_flow_casts_block_inputs_to_checkpoint_dtype():
    from trellmlx.models.sparse_structure_flow import SparseStructureFlowModel

    model = SparseStructureFlowModel(
        in_channels=2,
        out_channels=2,
        model_channels=12,
        num_heads=3,
        num_blocks=1,
        mlp_hidden=16,
        context_channels=4,
        resolution=2,
    )
    model.apply(lambda value: value.astype(mx.bfloat16))
    recorder = _RecordingBlock()
    model.blocks = [recorder]

    x = mx.random.normal((1, 2, 2, 2, 2), dtype=mx.float32)
    t = mx.array([1000.0], dtype=mx.float32)
    cond = mx.random.normal((1, 5, 4), dtype=mx.float32)

    out = model(x, t, cond)
    mx.eval(out)

    assert recorder.calls[0]["x"] == mx.bfloat16
    assert recorder.calls[0]["mod"] == mx.bfloat16
    assert recorder.calls[0]["cond"] == mx.bfloat16
    assert out.dtype == mx.float32


def test_sparse_structure_flow_cache_uses_checkpoint_dtype():
    from trellmlx.models.sparse_structure_flow import SparseStructureFlowModel

    model = SparseStructureFlowModel(
        in_channels=2,
        out_channels=2,
        model_channels=12,
        num_heads=3,
        num_blocks=1,
        mlp_hidden=16,
        context_channels=4,
        resolution=2,
    )
    model.apply(lambda value: value.astype(mx.bfloat16))

    cache = model.build_cross_kv_cache(mx.random.normal((1, 5, 4), dtype=mx.float32))

    assert cache[0][0].dtype == mx.bfloat16
    assert cache[0][1].dtype == mx.bfloat16


def test_slat_flow_casts_block_inputs_to_checkpoint_dtype():
    from trellmlx.models.slat_flow import SLatFlowModel

    model = SLatFlowModel(
        in_channels=2,
        out_channels=2,
        model_channels=12,
        num_heads=3,
        num_blocks=1,
        mlp_hidden=16,
        context_channels=4,
    )
    model.apply(lambda value: value.astype(mx.bfloat16))
    recorder = _RecordingBlock()
    model.blocks = [recorder]

    x = mx.random.normal((6, 2), dtype=mx.float32)
    t = mx.array([1000.0], dtype=mx.float32)
    cond = mx.random.normal((1, 5, 4), dtype=mx.float32)
    coords = mx.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]])

    out = model(x, t, cond, coords=coords)
    mx.eval(out)

    assert recorder.calls[0]["x"] == mx.bfloat16
    assert recorder.calls[0]["mod"] == mx.bfloat16
    assert recorder.calls[0]["cond"] == mx.bfloat16
    assert out.dtype == mx.float32


def test_slat_flow_cache_uses_checkpoint_dtype():
    from trellmlx.models.slat_flow import SLatFlowModel

    model = SLatFlowModel(
        in_channels=2,
        out_channels=2,
        model_channels=12,
        num_heads=3,
        num_blocks=1,
        mlp_hidden=16,
        context_channels=4,
    )
    model.apply(lambda value: value.astype(mx.bfloat16))

    cache = model.build_cross_kv_cache(mx.random.normal((1, 5, 4), dtype=mx.float32))

    assert cache[0][0].dtype == mx.bfloat16
    assert cache[0][1].dtype == mx.bfloat16


def test_generate_does_not_override_sparse_flow_source_dtype():
    source = Path("generate.py").read_text()

    assert "ss_flow.apply(lambda x: x.astype(mx.float32))" not in source
