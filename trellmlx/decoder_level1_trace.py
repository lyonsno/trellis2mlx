"""Exact operation-boundary capture for the first decoder upsample and level one."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from .models.shape_slat_decoder import (
    SparseChannel2Spatial,
    SparseConvNeXtBlock3d,
    SparseResBlockC2S3d,
    _decoder_linear,
    _decoder_silu,
)
from .modules.sparse_conv import build_neighbor_map


def capture_mlx_decoder_level1_trace(
    decoder,
    level0_output: mx.array,
    parent_coords: mx.array,
    *,
    hash_entry,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Capture the first upsample and level-one block 0 from an exact parent state."""
    level0_upsample = [
        block
        for block in decoder.blocks[0]
        if isinstance(block, SparseResBlockC2S3d)
    ]
    if len(level0_upsample) != 1:
        raise ValueError(
            "level-one trace requires exactly one level-zero upsample block, "
            f"got {len(level0_upsample)}"
        )
    level1_blocks = [
        block
        for block in decoder.blocks[1]
        if isinstance(block, SparseConvNeXtBlock3d)
    ]
    if len(level1_blocks) != 16:
        raise ValueError(
            "level-one trace requires exactly sixteen ConvNeXt blocks, "
            f"got {len(level1_blocks)}"
        )
    level1_upsample = [
        block
        for block in decoder.blocks[1]
        if isinstance(block, SparseResBlockC2S3d)
    ]
    if len(level1_upsample) != 1:
        raise ValueError(
            "level-one trace requires exactly one level-one upsample block, "
            f"got {len(level1_upsample)}"
        )
    upsample = level0_upsample[0]
    block0 = level1_blocks[0]
    parent_nmap = build_neighbor_map(parent_coords)

    subdiv_logits = _decoder_linear(upsample.to_subdiv, level0_output)
    subdiv_mask = subdiv_logits > 0
    norm1 = upsample.norm1(level0_output)
    silu1 = _decoder_silu(norm1)
    conv1 = upsample.conv1(silu1, parent_nmap)
    h_c2s, child_coords = SparseChannel2Spatial.upsample(
        conv1,
        parent_coords,
        subdiv_mask,
    )
    skip_c2s, skip_coords = SparseChannel2Spatial.upsample(
        level0_output,
        parent_coords,
        subdiv_mask,
    )
    if not np.array_equal(np.asarray(child_coords), np.asarray(skip_coords)):
        raise RuntimeError("upsample feature and skip coordinates differ")
    channels_per_child = upsample.channels // 8
    repeat_factor = upsample.out_channels // channels_per_child
    skip_repeated = mx.repeat(skip_c2s, repeat_factor, axis=1)

    norm2 = upsample.norm2(h_c2s)
    silu2 = _decoder_silu(norm2)
    child_nmap = build_neighbor_map(child_coords)
    conv2 = upsample.conv2(silu2, child_nmap)
    upsample_output = conv2 + skip_repeated

    natural_output, natural_coords, natural_subdiv = upsample(
        level0_output,
        parent_coords,
        parent_nmap,
    )
    mx.eval(
        subdiv_logits,
        norm1,
        silu1,
        conv1,
        h_c2s,
        skip_c2s,
        skip_repeated,
        norm2,
        silu2,
        conv2,
        upsample_output,
        natural_output,
        natural_coords,
        natural_subdiv,
    )
    for name, manual, natural in (
        ("features", upsample_output, natural_output),
        ("coordinates", child_coords, natural_coords),
        ("subdivision logits", subdiv_logits, natural_subdiv),
    ):
        if not np.array_equal(np.asarray(manual), np.asarray(natural)):
            raise RuntimeError(
                "manual first-upsample trace does not exactly reproduce "
                f"natural {name}"
            )

    block0_conv = block0.conv(upsample_output, child_nmap)
    block0_norm = block0.norm(block0_conv)
    block0_fc1 = _decoder_linear(block0.mlp_0, block0_norm)
    block0_silu = _decoder_silu(block0_fc1)
    block0_fc2 = _decoder_linear(block0.mlp_2, block0_silu)
    block0_output = block0_fc2 + upsample_output
    natural_block0 = block0(upsample_output, child_nmap)
    mx.eval(
        block0_conv,
        block0_norm,
        block0_fc1,
        block0_silu,
        block0_fc2,
        block0_output,
        natural_block0,
    )
    if not np.array_equal(
        np.asarray(block0_output),
        np.asarray(natural_block0),
    ):
        raise RuntimeError(
            "manual level-one block-0 trace does not exactly reproduce natural forward"
        )
    hash_entries = [
        hash_entry("level1_block0_output", np.asarray(natural_block0))
    ]
    level1_output = natural_block0
    for index, block in enumerate(level1_blocks[1:], start=1):
        level1_output = block(level1_output, child_nmap)
        mx.eval(level1_output)
        hash_entries.append(
            hash_entry(
                f"level1_block{index}_output",
                np.asarray(level1_output),
            )
        )

    next_upsample = level1_upsample[0]
    next_subdiv_logits = _decoder_linear(
        next_upsample.to_subdiv,
        level1_output,
    )
    next_subdiv_mask = next_subdiv_logits > 0
    next_norm1 = next_upsample.norm1(level1_output)
    next_silu1 = _decoder_silu(next_norm1)
    next_conv1 = next_upsample.conv1(next_silu1, child_nmap)
    next_h_c2s, next_child_coords = SparseChannel2Spatial.upsample(
        next_conv1,
        child_coords,
        next_subdiv_mask,
    )
    next_skip_c2s, next_skip_coords = SparseChannel2Spatial.upsample(
        level1_output,
        child_coords,
        next_subdiv_mask,
    )
    if not np.array_equal(
        np.asarray(next_child_coords),
        np.asarray(next_skip_coords),
    ):
        raise RuntimeError("second-upsample feature and skip coordinates differ")
    next_channels_per_child = next_upsample.channels // 8
    next_repeat_factor = (
        next_upsample.out_channels // next_channels_per_child
    )
    next_skip_repeated = mx.repeat(
        next_skip_c2s,
        next_repeat_factor,
        axis=1,
    )
    next_norm2 = next_upsample.norm2(next_h_c2s)
    next_silu2 = _decoder_silu(next_norm2)
    next_child_nmap = build_neighbor_map(next_child_coords)
    next_conv2 = next_upsample.conv2(next_silu2, next_child_nmap)
    next_upsample_output = next_conv2 + next_skip_repeated
    (
        natural_next_output,
        natural_next_coords,
        natural_next_subdiv,
    ) = next_upsample(
        level1_output,
        child_coords,
        child_nmap,
    )
    mx.eval(
        next_subdiv_logits,
        next_norm1,
        next_silu1,
        next_conv1,
        next_h_c2s,
        next_skip_c2s,
        next_skip_repeated,
        next_norm2,
        next_silu2,
        next_conv2,
        next_upsample_output,
        natural_next_output,
        natural_next_coords,
        natural_next_subdiv,
    )
    for name, manual, natural in (
        ("features", next_upsample_output, natural_next_output),
        ("coordinates", next_child_coords, natural_next_coords),
        ("subdivision logits", next_subdiv_logits, natural_next_subdiv),
    ):
        if not np.array_equal(np.asarray(manual), np.asarray(natural)):
            raise RuntimeError(
                "manual second-upsample trace does not exactly reproduce "
                f"natural {name}"
            )
    next_boundaries = {
        "level1_upsample_subdiv_logits": next_subdiv_logits,
        "level1_upsample_norm1": next_norm1,
        "level1_upsample_silu1": next_silu1,
        "level1_upsample_conv1": next_conv1,
        "level2_child_coords": next_child_coords,
        "level1_upsample_h_c2s": next_h_c2s,
        "level1_upsample_skip_c2s": next_skip_c2s,
        "level1_upsample_skip_repeated": next_skip_repeated,
        "level1_upsample_norm2": next_norm2,
        "level1_upsample_silu2": next_silu2,
        "level1_upsample_conv2": next_conv2,
        "level1_upsample_output": natural_next_output,
    }
    hash_entries.extend(
        hash_entry(name, np.asarray(values))
        for name, values in next_boundaries.items()
    )

    arrays = {
        "parent_coords": np.asarray(parent_coords, dtype=np.int32),
        "child_coords": np.asarray(child_coords, dtype=np.int32),
        "level0_output": np.asarray(level0_output),
        "upsample_subdiv_logits": np.asarray(subdiv_logits),
        "upsample_norm1": np.asarray(norm1),
        "upsample_silu1": np.asarray(silu1),
        "upsample_conv1": np.asarray(conv1),
        "upsample_h_c2s": np.asarray(h_c2s),
        "upsample_skip_c2s": np.asarray(skip_c2s),
        "upsample_skip_repeated": np.asarray(skip_repeated),
        "upsample_norm2": np.asarray(norm2),
        "upsample_silu2": np.asarray(silu2),
        "upsample_conv2": np.asarray(conv2),
        "upsample_output": np.asarray(natural_output),
        "level1_block0_conv": np.asarray(block0_conv),
        "level1_block0_norm": np.asarray(block0_norm),
        "level1_block0_mlp_fc1": np.asarray(block0_fc1),
        "level1_block0_silu": np.asarray(block0_silu),
        "level1_block0_mlp_fc2": np.asarray(block0_fc2),
        "level1_block0_output": np.asarray(natural_block0),
    }
    trace = {
        name: np.ascontiguousarray(values)
        for name, values in arrays.items()
    }
    return trace, hash_entries
