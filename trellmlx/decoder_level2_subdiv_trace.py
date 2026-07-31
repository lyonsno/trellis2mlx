"""Focused MLX capture for the decoder level-two subdivision head."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


PROJECTION_BACKENDS = ("turing_fda", "native")


def project_level2_subdiv(
    linear: Any,
    value: Any,
    backend: str,
    *,
    turing_linear: Callable[..., Any] | None = None,
) -> Any:
    backend = backend.lower()
    if backend not in PROJECTION_BACKENDS:
        raise ValueError(
            "level2 subdivision projection backend must be one of "
            f"{list(PROJECTION_BACKENDS)}, got {backend!r}"
        )
    if backend == "native":
        return linear(value)
    if turing_linear is None:
        from .turing_fda import turing_fda_linear

        turing_linear = turing_fda_linear
    if linear.bias is None:
        raise ValueError("Turing FDA subdivision projection requires a bias")
    return turing_linear(value, linear.weight.T, linear.bias)


def capture_mlx_decoder_level2_subdiv_trace(
    decoder: Any,
    block0_output: Any,
    child_coords: Any,
    *,
    projection_backend: str,
) -> dict[str, Any]:
    import mlx.core as mx

    from .models.shape_slat_decoder import (
        SparseConvNeXtBlock3d,
        SparseResBlockC2S3d,
    )
    from .modules.sparse_conv import build_neighbor_map

    blocks = [
        block
        for block in decoder.blocks[2]
        if isinstance(block, SparseConvNeXtBlock3d)
    ]
    upsample = [
        block
        for block in decoder.blocks[2]
        if isinstance(block, SparseResBlockC2S3d)
    ]
    if len(blocks) != 8 or len(upsample) != 1:
        raise ValueError(
            "level2 subdivision trace requires eight ConvNeXt blocks and "
            f"one upsample block, got {len(blocks)} and {len(upsample)}"
        )
    if (
        block0_output.ndim != 2
        or block0_output.shape[1] != 256
        or child_coords.shape != (block0_output.shape[0], 4)
    ):
        raise ValueError(
            "level2 subdivision parent must have feature shape [N, 256] "
            "and coordinate shape [N, 4]"
        )

    neighbor_map = build_neighbor_map(child_coords)
    block7_output = block0_output
    for block in blocks[1:]:
        block7_output = block(block7_output, neighbor_map)
        mx.eval(block7_output)

    linear = upsample[0].to_subdiv
    logits = project_level2_subdiv(
        linear,
        block7_output,
        projection_backend,
    )
    repeated = project_level2_subdiv(
        linear,
        block7_output,
        projection_backend,
    )
    mx.eval(logits, repeated, linear.weight, linear.bias)
    if not np.array_equal(np.asarray(logits), np.asarray(repeated)):
        raise RuntimeError(
            "level2 subdivision projection is not deterministic on replay"
        )

    arrays = {
        "level2_child_coords": np.asarray(child_coords),
        "level2_block0_output": np.asarray(block0_output),
        "level2_block7_output": np.asarray(block7_output),
        "level2_upsample_subdiv_weight": np.asarray(linear.weight),
        "level2_upsample_subdiv_bias": np.asarray(linear.bias),
        "level2_upsample_subdiv_logits": np.asarray(logits),
    }
    return {
        name: np.ascontiguousarray(values)
        for name, values in arrays.items()
    }
