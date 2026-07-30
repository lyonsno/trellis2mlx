"""Exact operation-boundary capture for shape-decoder level zero."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .models.shape_slat_decoder import SparseConvNeXtBlock3d
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

    block0 = convnext_blocks[0]
    block0_conv = block0.conv(torso_input, neighbor_map)
    block0_norm = block0.norm(block0_conv)
    block0_mlp_fc1 = block0.mlp_0(block0_norm)
    block0_silu = nn.silu(block0_mlp_fc1)
    block0_mlp_fc2 = block0.mlp_2(block0_silu)
    block0_output = block0_mlp_fc2 + torso_input
    natural_block0 = block0(torso_input, neighbor_map)
    mx.eval(
        block0_conv,
        block0_norm,
        block0_mlp_fc1,
        block0_silu,
        block0_mlp_fc2,
        block0_output,
        natural_block0,
    )
    if not np.array_equal(
        np.asarray(block0_output),
        np.asarray(natural_block0),
    ):
        raise RuntimeError(
            "manual level-zero block trace does not exactly reproduce natural forward"
        )

    block_outputs = [natural_block0]
    current = natural_block0
    for block in convnext_blocks[1:]:
        current = block(current, neighbor_map)
        mx.eval(current)
        block_outputs.append(current)

    level0_subdiv_logits = upsample_blocks[0].to_subdiv(current)
    mx.eval(
        input_feats,
        projected_fp32,
        torso_input,
        *block_outputs,
        level0_subdiv_logits,
    )

    arrays = {
        "coords": np.asarray(coords, dtype=np.int32),
        "input_feats": np.asarray(input_feats, dtype=np.float32),
        "from_latent_fp32": np.asarray(projected_fp32, dtype=np.float32),
        "torso_input": np.asarray(torso_input),
        "block0_conv": np.asarray(block0_conv),
        "block0_norm": np.asarray(block0_norm),
        "block0_mlp_fc1": np.asarray(block0_mlp_fc1),
        "block0_silu": np.asarray(block0_silu),
        "block0_mlp_fc2": np.asarray(block0_mlp_fc2),
        "block0_output": np.asarray(block_outputs[0]),
        "block1_output": np.asarray(block_outputs[1]),
        "block2_output": np.asarray(block_outputs[2]),
        "block3_output": np.asarray(block_outputs[3]),
        "level0_subdiv_logits": np.asarray(level0_subdiv_logits),
    }
    return {
        name: np.ascontiguousarray(values)
        for name, values in arrays.items()
    }
