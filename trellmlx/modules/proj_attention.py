"""Projection attention blocks for Pixal3D-conditioned TRELLIS flows."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn


class ProjectAttention(nn.Module):
    """Add per-token projected image features to global cross-attention."""

    def __init__(self, cross_attn_block: Any, channels: int, proj_in_channels: int):
        super().__init__()
        self.cross_attn_block = cross_attn_block
        self.proj_linear = nn.Linear(proj_in_channels, channels)

    def __call__(self, x: mx.array, context: dict[str, mx.array] | tuple[mx.array, mx.array]) -> mx.array:
        if isinstance(context, dict):
            global_context = context["global"]
            proj_context = context["proj"]
        else:
            global_context, proj_context = context

        global_out = self.cross_attn_block(x, global_context)
        proj_out = self.proj_linear(proj_context)
        if proj_out.ndim == 3 and global_out.ndim == 2:
            if proj_out.shape[0] != 1:
                raise ValueError(
                    "Projected context batch must be 1 when attention tokens are flattened"
                )
            proj_out = proj_out[0]
        if proj_out.shape != global_out.shape:
            raise ValueError(
                f"Projected context shape {proj_out.shape} does not match global attention {global_out.shape}"
            )
        return global_out + proj_out


class GatedProjectAttention(nn.Module):
    """Fuse semantic and color projected features before adding global attention."""

    def __init__(
        self,
        cross_attn_block: Any,
        channels: int,
        dino_in_channels: int,
        vae_in_channels: int,
    ):
        super().__init__()
        self.cross_attn_block = cross_attn_block
        self.proj_linear = nn.Linear(dino_in_channels + vae_in_channels, channels)
        self.proj_linear.weight = mx.zeros_like(self.proj_linear.weight)
        self.proj_linear.bias = mx.zeros_like(self.proj_linear.bias)

    def __call__(self, x: mx.array, context: dict[str, mx.array] | tuple[mx.array, mx.array, mx.array]) -> mx.array:
        if isinstance(context, dict):
            global_context = context["global"]
            proj_semantic = context["proj_semantic"]
            proj_color = context["proj_color"]
        else:
            global_context, proj_semantic, proj_color = context

        global_out = self.cross_attn_block(x, global_context)
        fused = self.proj_linear(mx.concatenate([proj_semantic, proj_color], axis=-1))
        if fused.ndim == 3 and global_out.ndim == 2:
            if fused.shape[0] != 1:
                raise ValueError("Projected context batch must be 1 when attention tokens are flattened")
            fused = fused[0]
        if fused.shape != global_out.shape:
            raise ValueError(
                f"Projected context shape {fused.shape} does not match global attention {global_out.shape}"
            )
        return global_out + fused
