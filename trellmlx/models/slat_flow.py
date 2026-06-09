"""SLatFlowModel — DiT over sparse tokens at occupied voxel positions.

Architecturally identical to SparseStructureFlowModel (same blocks, same
attention, same adaLN-Zero), but operates on variable-length sparse token
sequences instead of a dense 3D grid.

Input: SparseTensor with features [N, in_channels] at occupied coordinates
Output: SparseTensor with features [N, out_channels]

For the 512 and 1024 checkpoint variants, the model has:
- 30 blocks, 1536 model channels, 12 heads, 128 head dim
- 32 input/output channels (shape latent dimensionality)
- Same MLP hidden dim (8192)
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .sparse_structure_flow import (
    TimestepEmbedder,
    MultiHeadAttention,
    FeedForward,
    ModulatedBlock,
    _layernorm_noaffine,
)
from ..modules.norm import LayerNorm32
from ..modules.attention import MultiHeadRMSNorm
from ..modules.rope import build_rope_phases, apply_rope


class SLatFlowModel(nn.Module):
    """TRELLIS.2 Structured Latent Flow Model.

    Same architecture as SparseStructureFlowModel but takes sparse tokens
    at occupied voxel coordinates instead of a dense 3D grid.

    Config (from TRELLIS.2-4B, 512 variant):
        in_channels: 32
        out_channels: 32
        model_channels: 1536
        num_heads: 12
        num_blocks: 30
        mlp_hidden: 8192
        context_channels: 1024
        resolution: 512 (coordinate space, not grid resolution)
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        model_channels: int = 1536,
        num_heads: int = 12,
        num_blocks: int = 30,
        mlp_hidden: int = 8192,
        context_channels: int = 1024,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_heads = num_heads
        self.head_dim = model_channels // num_heads

        # Timestep embedder
        self.t_embedder = TimestepEmbedder(model_channels)

        # Input/output projections
        self.input_layer = nn.Linear(in_channels, model_channels)
        self.out_layer = nn.Linear(model_channels, out_channels)

        # Shared adaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 6 * model_channels),
        )

        # Transformer blocks (same as SparseStructureFlowModel)
        self.blocks = [
            ModulatedBlock(model_channels, num_heads, context_channels, mlp_hidden)
            for _ in range(num_blocks)
        ]

        # Compilation state (call .compile() to enable)
        self._compiled = False
        self._run_blocks = self._run_blocks_impl

    def __call__(
        self,
        x: mx.array,           # [N, in_channels] sparse token features
        t: mx.array,           # [B] timestep
        cond: mx.array,        # [B, L, context_channels]
        coords: mx.array = None,  # [N, 3] voxel coordinates for RoPE
        concat_cond: mx.array = None,  # [N, C'] features to concatenate with x
    ) -> mx.array:
        N = x.shape[0]
        B = t.shape[0] if len(t.shape) else 1
        cond_B = cond.shape[0] if cond is not None and len(cond.shape) else B
        if B != 1 or cond_B != 1:
            raise ValueError(
                f"Only B=1 supported for SLatFlowModel inference, got t B={B}, cond B={cond_B}"
            )

        if concat_cond is not None:
            x = mx.concatenate([x, concat_cond], axis=-1)

        # Timestep → shared modulation
        t_emb = self.t_embedder(t)
        mod = self.adaLN_modulation(t_emb)

        # Project to model channels
        x = self.input_layer(x)  # [N, C]

        # Build RoPE phases from coordinates if provided
        rope_phases = None
        if coords is not None:
            rope_phases = self._coords_to_rope_phases(coords)

        # Run through blocks (B=1 assumed)
        if self._compiled:
            # Compiled path: single fused graph, no internal evals.
            # mx.compile handles buffer reuse; the sampler evals between steps.
            x = self._run_blocks(x, mod[0], cond, rope_phases)
        else:
            # Eager path: periodic eval to yield memory bus back to CPU/display.
            # Without this, 30 blocks on 171K tokens queues a single
            # GPU burst that beachballs the entire machine.
            for i, block in enumerate(self.blocks):
                x = block(x, mod[0], cond, rope_phases=rope_phases)
                if (i + 1) % 6 == 0:
                    mx.eval(x)

        # Output projection
        x = _layernorm_noaffine(x)
        x = self.out_layer(x)  # [N, out_channels]

        return x

    def _run_blocks_impl(self, x, mod, cond, rope_phases):
        """Pure forward pass through all blocks (no mx.eval)."""
        for block in self.blocks:
            x = block(x, mod, cond, rope_phases=rope_phases)
        return x

    def compile(self):
        """Enable mx.compile for the transformer block loop."""
        self._compiled = True
        self._run_blocks = mx.compile(self._run_blocks_impl)

    def uncompile(self):
        """Disable mx.compile, fall back to eager with periodic eval."""
        self._compiled = False
        self._run_blocks = self._run_blocks_impl

    def _coords_to_rope_phases(self, coords: mx.array) -> mx.array:
        """Compute RoPE phases from sparse voxel coordinates.

        Args:
            coords: [N, 3] integer coordinates (z, y, x)

        Returns:
            [N, head_dim//2, 2] cos/sin pairs
        """
        import math

        freq_dim = self.head_dim // 2 // 3  # 3D coordinates
        freqs = np.arange(freq_dim, dtype=np.float32) / freq_dim
        freqs = 1.0 / (10000.0 ** freqs)
        freqs = mx.array(freqs)

        coords_f = coords.astype(mx.float32)

        # Compute phases for each spatial dimension
        all_angles = []
        for d in range(3):
            angles = coords_f[:, d:d+1] * freqs[None, :]  # [N, freq_dim]
            all_angles.append(angles)
        angles = mx.concatenate(all_angles, axis=-1)  # [N, 3*freq_dim]

        # Pad to head_dim//2 if needed
        target = self.head_dim // 2
        if angles.shape[-1] < target:
            pad = mx.zeros((angles.shape[0], target - angles.shape[-1]))
            angles = mx.concatenate([angles, pad], axis=-1)

        # Return as cos/sin pairs
        cos_p = mx.cos(angles)
        sin_p = mx.sin(angles)
        return mx.stack([cos_p, sin_p], axis=-1)  # [N, head_dim//2, 2]
