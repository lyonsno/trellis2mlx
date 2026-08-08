import mlx.core as mx
from pathlib import Path
import argparse
import subprocess
import sys

import pytest


def test_sparse_block_dispatches_noaffine_layernorm_through_sparse_backend(monkeypatch):
    import trellmlx.models.sparse_structure_flow as sparse_flow

    calls = []

    def fake_layernorm(x, eps=1e-6):
        calls.append((x.shape, eps))
        return mx.full(x.shape, 3, dtype=x.dtype)

    monkeypatch.setattr(
        sparse_flow,
        "_sparse_flow_layernorm_noaffine",
        fake_layernorm,
        raising=False,
    )
    block = sparse_flow.ModulatedBlock(
        channels=4,
        num_heads=1,
        context_channels=4,
        mlp_hidden=8,
        sparse_flow_layernorm=True,
    )
    x = mx.zeros((2, 4), dtype=mx.bfloat16)

    output = block._layernorm_noaffine(x, eps=1e-5)
    mx.eval(output)

    assert calls == [((2, 4), 1e-5)]
    assert bool(mx.all(output == mx.array(3, dtype=mx.bfloat16)))


def test_sparse_model_affine_norms_carry_sparse_backend_identity():
    from trellmlx.models.sparse_structure_flow import SparseStructureFlowModel

    model = SparseStructureFlowModel(
        model_channels=4,
        num_heads=1,
        num_blocks=1,
        mlp_hidden=8,
        context_channels=4,
        resolution=2,
    )

    assert model.blocks[0].norm2.sparse_flow_layernorm is True
    assert model.blocks[0].norm2.shape_flow_layernorm is False
    assert model.blocks[0].norm2.decoder_layernorm is False


def test_generate_exposes_distinct_sparse_flow_layernorm_selector():
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parents[1] / "generate.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--sparse-flow-layernorm-backend" in result.stdout
    assert "--shape-flow-layernorm-backend" in result.stdout


def test_rowwise_correction_rejects_nondefault_sparse_layernorm_backend():
    from generate import _validate_sparse_layernorm_correction_route

    parser = argparse.ArgumentParser()
    args = argparse.Namespace(
        sparse_flow_layernorm_correction_report="correction.json",
        sparse_flow_layernorm_backend="cuda-welford-turing-t4",
    )

    with pytest.raises(SystemExit):
        _validate_sparse_layernorm_correction_route(parser, args)
