"""Focused MLX capture for the exact level-two block-zero frontier."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from .models.shape_slat_decoder import (
    SparseConvNeXtBlock3d,
    SparseResBlockC2S3d,
    _decoder_linear,
    _decoder_silu,
)
from .modules.sparse_conv import build_neighbor_map


def _only(blocks, block_type, label: str, expected: int):
    selected = [block for block in blocks if isinstance(block, block_type)]
    if len(selected) != expected:
        raise ValueError(
            f"focused block0 trace requires {expected} {label}, "
            f"got {len(selected)}"
        )
    return selected


def capture_mlx_decoder_level2_block0_trace(
    decoder,
    level0_output: mx.array,
    parent_coords: mx.array,
) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
    """Replay only the inherited path needed to enter and decompose block0."""
    first_upsample = _only(
        decoder.blocks[0],
        SparseResBlockC2S3d,
        "level-zero upsample block",
        1,
    )[0]
    level1_blocks = _only(
        decoder.blocks[1],
        SparseConvNeXtBlock3d,
        "level-one ConvNeXt blocks",
        16,
    )
    second_upsample = _only(
        decoder.blocks[1],
        SparseResBlockC2S3d,
        "level-one upsample block",
        1,
    )[0]
    level2_block0 = _only(
        decoder.blocks[2],
        SparseConvNeXtBlock3d,
        "level-two ConvNeXt blocks",
        8,
    )[0]

    parent_nmap = build_neighbor_map(parent_coords)
    level1_input, level1_coords, _ = first_upsample(
        level0_output,
        parent_coords,
        parent_nmap,
    )
    mx.eval(level1_input, level1_coords)
    level1_nmap = build_neighbor_map(level1_coords)
    level1_output = level1_input
    for block in level1_blocks:
        level1_output = block(level1_output, level1_nmap)
        mx.eval(level1_output)
    block0_input, level2_coords, _ = second_upsample(
        level1_output,
        level1_coords,
        level1_nmap,
    )
    mx.eval(block0_input, level2_coords)
    level2_nmap = build_neighbor_map(level2_coords)

    conv = level2_block0.conv(block0_input, level2_nmap)
    norm = level2_block0.norm(conv)
    fc1 = _decoder_linear(level2_block0.mlp_0, norm)
    silu = _decoder_silu(fc1)
    fc2 = _decoder_linear(level2_block0.mlp_2, silu)
    output = fc2 + block0_input
    natural = level2_block0(block0_input, level2_nmap)
    mx.eval(
        block0_input,
        level2_coords,
        conv,
        norm,
        fc1,
        silu,
        fc2,
        output,
        natural,
    )
    features_equal = np.array_equal(np.asarray(output), np.asarray(natural))
    if not features_equal:
        raise RuntimeError(
            "manual MLX level-two block0 decomposition does not exactly "
            "reproduce natural forward"
        )

    arrays = {
        "level2_child_coords": np.asarray(level2_coords, dtype=np.int32),
        "level1_upsample_output": np.asarray(block0_input),
        "level2_block0_conv": np.asarray(conv),
        "level2_block0_norm": np.asarray(norm),
        "level2_block0_mlp_fc1": np.asarray(fc1),
        "level2_block0_silu": np.asarray(silu),
        "level2_block0_mlp_fc2": np.asarray(fc2),
        "level2_block0_output": np.asarray(natural),
    }
    return arrays, {
        "features": features_equal,
        "coordinates": True,
    }
