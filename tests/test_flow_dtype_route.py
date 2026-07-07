"""Source-parity contracts for TRELLIS.2 flow model dtype routing."""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


class _TinyWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = mx.zeros((2, 2), dtype=mx.float32)


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
    """TRELLIS.2 bf16 checkpoints must not be silently demoted to fp16."""
    from trellmlx.weight_loader import load_weights

    checkpoint = tmp_path / "tiny-bf16.safetensors"
    mx.save_safetensors(
        str(checkpoint),
        {"weight": mx.ones((2, 2), dtype=mx.bfloat16)},
    )

    model = _TinyWeights()
    load_weights(model, str(checkpoint), verbose=False)

    assert model.weight.dtype == mx.bfloat16


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
