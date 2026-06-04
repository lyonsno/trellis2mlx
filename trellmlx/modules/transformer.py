"""Dense transformer blocks for TRELLIS.2 MLX port.

These are used by SparseStructureFlowModel which operates on a dense
3D voxel grid (flattened to a token sequence).
"""

import math

import mlx.core as mx
import mlx.nn as nn

from .attention import scaled_dot_product_attention, RMSNorm
from .norm import LayerNorm32


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding -> MLP -> model_channels.

    Identical to the standard diffusion timestep embedder:
    scalar t -> sin/cos(t * freqs) -> Linear -> SiLU -> Linear
    """

    def __init__(self, model_channels: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels),
        )

    def __call__(self, t: mx.array) -> mx.array:
        # Sinusoidal embedding
        half = self.freq_dim // 2
        freqs = mx.exp(-math.log(10000) * mx.arange(half, dtype=mx.float32) / half)
        args = t[:, None].astype(mx.float32) * freqs[None, :]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        return self.mlp(emb)


class AbsolutePositionEmbedder(nn.Module):
    """3D absolute position embedding from coordinate meshgrid."""

    def __init__(self, model_channels: int, in_channels: int = 3):
        super().__init__()
        self.linear = nn.Linear(in_channels, model_channels)

    def __call__(self, coords: mx.array) -> mx.array:
        return self.linear(coords.astype(mx.float32))


class FeedForward(nn.Module):
    """Standard MLP with GELU activation and 4x expansion."""

    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * mult)
        self.linear2 = nn.Linear(dim * mult, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear2(nn.gelu(self.linear1(x)))


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional QK RMSNorm and RoPE.

    For self-attention: fused QKV projection.
    For cross-attention: separate Q and KV projections.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        qk_norm: bool = True,
        context_channels: int = None,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        if context_channels is None:
            # Self-attention: fused QKV
            self.to_qkv = nn.Linear(channels, 3 * channels)
        else:
            # Cross-attention: separate Q and KV
            self.to_q = nn.Linear(channels, channels)
            self.to_kv = nn.Linear(context_channels, 2 * channels)

        self.to_out = nn.Linear(channels, channels)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_rms_norm = RMSNorm(self.head_dim)
            self.k_rms_norm = RMSNorm(self.head_dim)

    def __call__(
        self,
        x: mx.array,           # [T, C]
        context: mx.array = None,  # [B, L, C_ctx] for cross-attn
    ) -> mx.array:
        T = x.shape[0]

        if context is None:
            # Self-attention
            qkv = self.to_qkv(x)  # [T, 3*C]
            qkv = qkv.reshape(T, 3, self.num_heads, self.head_dim)
            q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each [T, H, D]
        else:
            # Cross-attention
            q = self.to_q(x).reshape(T, self.num_heads, self.head_dim)
            kv = self.to_kv(context)
            S = kv.shape[1] if kv.ndim == 3 else kv.shape[0]
            kv = kv.reshape(-1, S, 2, self.num_heads, self.head_dim)
            k, v = kv[..., 0, :, :].squeeze(0), kv[..., 1, :, :].squeeze(0)

        if self.qk_norm:
            q = self.q_rms_norm(q)
            k = self.k_rms_norm(k)

        # For dense attention: add batch dim, run SDPA, remove batch dim
        q = q[None].transpose(0, 2, 1, 3)  # [1, H, T, D]
        k = k[None].transpose(0, 2, 1, 3)  # [1, H, S, D]
        v = v[None].transpose(0, 2, 1, 3)  # [1, H, S, D]

        out = scaled_dot_product_attention(q, k, v)  # [1, H, T, D]
        out = out.transpose(0, 2, 1, 3).reshape(T, self.channels)  # [T, C]

        return self.to_out(out)


class ModulatedTransformerCrossBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning.

    self-attn -> cross-attn -> FFN, each modulated by timestep embedding.
    This is the core block of the TRELLIS.2 DiT.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        context_channels: int,
        mlp_ratio: int = 4,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.channels = channels

        # Norms (elementwise_affine=False, controlled by adaLN)
        self.norm1 = LayerNorm32(channels)
        self.norm2 = LayerNorm32(channels)
        self.norm3 = LayerNorm32(channels)

        # Self-attention
        self.self_attn = MultiHeadAttention(channels, num_heads, qk_norm=qk_norm)

        # Cross-attention
        self.cross_attn = MultiHeadAttention(
            channels, num_heads, qk_norm=qk_norm, context_channels=context_channels
        )

        # FFN
        self.mlp = FeedForward(channels, mlp_ratio)

        # adaLN-Zero: timestep -> 6 * channels (shift, scale, gate for MSA and MLP)
        # Cross-attn gets no separate modulation in TRELLIS.2
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 6 * channels),
        )

    def __call__(
        self,
        x: mx.array,           # [T, C]
        t_emb: mx.array,       # [B, C] timestep embedding
        context: mx.array,     # [B, L, C_ctx] image conditioning
    ) -> mx.array:
        # Compute modulation parameters from timestep
        mod = self.adaLN_modulation(t_emb)  # [B, 6*C]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(
            mod, 6, axis=-1
        )

        # Self-attention with adaLN-Zero
        h = self.norm1(x)
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h)
        h = h * gate_msa
        x = x + h

        # Cross-attention (no adaLN modulation in TRELLIS.2)
        h = self.norm2(x)
        h = self.cross_attn(h, context)
        x = x + h

        # FFN with adaLN-Zero
        h = self.norm3(x)
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h

        return x
