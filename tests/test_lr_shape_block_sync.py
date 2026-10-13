"""The optional LR synchronization route must really evaluate every block."""

import pytest
import mlx.core as mx

from trellmlx.models import slat_flow


def test_shape_flow_per_block_sync_is_opt_in(monkeypatch):
    original_eval = mx.eval
    observed = []

    def record_eval(*arrays):
        observed.append(tuple(array.shape for array in arrays))
        return original_eval(*arrays)

    monkeypatch.setattr(slat_flow.mx, "eval", record_eval)
    inputs = (
        mx.zeros((2, 4)),
        mx.array([500.0]),
        mx.zeros((1, 2, 8)),
    )
    kwargs = dict(
        in_channels=4,
        out_channels=4,
        model_channels=24,
        num_heads=3,
        num_blocks=3,
        mlp_hidden=48,
        context_channels=8,
    )

    default_model = slat_flow.SLatFlowModel.for_shape(**kwargs)
    default_model(*inputs)
    assert [item for item in observed if item == ((2, 24),)] == []

    observed.clear()
    per_block_model = slat_flow.SLatFlowModel.for_shape(
        **kwargs, lr_sync_every_block=True
    )
    per_block_model(*inputs)
    assert [item for item in observed if item == ((2, 24),)] == [((2, 24),)] * 3
    with pytest.raises(ValueError, match="incompatible with compiled blocks"):
        per_block_model.compile()
