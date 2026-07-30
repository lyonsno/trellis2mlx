"""Exact operation-boundary capture for shape-decoder level zero."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .models.shape_slat_decoder import (
    SparseConvNeXtBlock3d,
    _decoder_linear,
    _decoder_silu,
)
from .modules.sparse_conv import build_neighbor_map


def capture_mlx_decoder_level0_trace(
    decoder,
    feats: mx.array,
    coords: mx.array,
) -> dict[str, np.ndarray]:
    """Capture level-zero boundaries while proving the trace matches natural forward."""
    level = decoder.blocks[0]
    convnext_blocks = [
        block for block in level if isinstance(block, SparseConvNeXtBlock3d)
    ]
    if len(convnext_blocks) != 4:
        raise ValueError(
            "level-zero trace requires exactly four SparseConvNeXt blocks, "
            f"got {len(convnext_blocks)}"
        )
    upsample_blocks = [
        block for block in level if hasattr(block, "to_subdiv")
    ]
    if len(upsample_blocks) != 1:
        raise ValueError(
            "level-zero trace requires exactly one subdivision head, "
            f"got {len(upsample_blocks)}"
        )

    input_feats = feats
    projected_fp32 = decoder.from_latent(input_feats)
    torso_input = (
        projected_fp32.astype(mx.float16)
        if decoder.use_fp16
        else projected_fp32
    )
    neighbor_map = build_neighbor_map(coords)

    block_arrays = {}
    current = torso_input
    for block_index, block in enumerate(convnext_blocks):
        block_input = current
        conv = block.conv(block_input, neighbor_map)
        norm = block.norm(conv)
        mlp_fc1 = _decoder_linear(block.mlp_0, norm)
        silu = _decoder_silu(mlp_fc1)
        mlp_fc2 = _decoder_linear(block.mlp_2, silu)
        output = mlp_fc2 + block_input
        natural = block(block_input, neighbor_map)
        mx.eval(conv, norm, mlp_fc1, silu, mlp_fc2, output, natural)
        if not np.array_equal(np.asarray(output), np.asarray(natural)):
            raise RuntimeError(
                "manual level-zero block trace does not exactly reproduce "
                f"natural forward for block {block_index}"
            )
        block_arrays.update(
            {
                f"block{block_index}_conv": conv,
                f"block{block_index}_norm": norm,
                f"block{block_index}_mlp_fc1": mlp_fc1,
                f"block{block_index}_silu": silu,
                f"block{block_index}_mlp_fc2": mlp_fc2,
                f"block{block_index}_output": natural,
            }
        )
        current = natural

    level0_subdiv_logits = _decoder_linear(
        upsample_blocks[0].to_subdiv,
        current,
    )
    mx.eval(
        input_feats,
        projected_fp32,
        torso_input,
        *block_arrays.values(),
        level0_subdiv_logits,
    )

    arrays = {
        "coords": np.asarray(coords, dtype=np.int32),
        "input_feats": np.asarray(input_feats, dtype=np.float32),
        "from_latent_fp32": np.asarray(projected_fp32, dtype=np.float32),
        "torso_input": np.asarray(torso_input),
        **{
            name: np.asarray(values)
            for name, values in block_arrays.items()
        },
        "level0_subdiv_logits": np.asarray(level0_subdiv_logits),
    }
    return {
        name: np.ascontiguousarray(values)
        for name, values in arrays.items()
    }
