"""SparseStructureFlowModel — DiT over a dense 3D voxel grid.

This is the first stage of the TRELLIS.2 pipeline. It takes a noisy
dense 3D tensor [B, in_channels, R, R, R] and denoises it using a
DiT with adaLN-Zero conditioning on timestep + cross-attention to
image features.

The "sparse" in the name refers to the output being thresholded into
sparse occupancy coordinates — the model itself operates on a dense grid.
"""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..modules.norm import LayerNorm32
from ..modules.attention import scaled_dot_product_attention, MultiHeadRMSNorm
from ..modules.rope import apply_rope, build_rope_phases


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep → MLP → model_channels."""

    def __init__(self, model_channels: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp_0 = nn.Linear(freq_dim, model_channels)
        self.mlp_2 = nn.Linear(model_channels, model_channels)

    def __call__(self, t: mx.array) -> mx.array:
        half = self.freq_dim // 2
        freqs = mx.exp(-math.log(10000) * mx.arange(half, dtype=mx.float32) / half)
        args = t[:, None].astype(mx.float32) * freqs[None, :]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        return self.mlp_2(nn.silu(self.mlp_0(emb)))


class MultiHeadAttention(nn.Module):
    """Multi-head attention with QK RMSNorm.

    Self-attention uses fused QKV. Cross-attention uses separate Q and KV.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        context_channels: int = None,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        if context_channels is None:
            self.to_qkv = nn.Linear(channels, 3 * channels)
        else:
            self.to_q = nn.Linear(channels, channels)
            self.to_kv = nn.Linear(context_channels, 2 * channels)

        self.to_out = nn.Linear(channels, channels)
        self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)

    def project_kv(self, context: mx.array) -> tuple[mx.array, mx.array]:
        """Project context to K, V and apply RMSNorm. Cache-friendly."""
        kv = self.to_kv(context)
        if kv.ndim == 3:
            S = kv.shape[1]
            kv = kv.reshape(-1, S, 2, self.num_heads, self.head_dim)
            k = kv[:, :, 0]
            v = kv[:, :, 1]
        else:
            B_T = kv.shape[0]
            kv = kv.reshape(B_T, 2, self.num_heads, self.head_dim)
            k, v = kv[:, 0], kv[:, 1]
        k = self.k_rms_norm(k)
        return k, v

    def __call__(self, x: mx.array, context: mx.array = None, rope_phases: mx.array = None,
                 cached_kv: tuple = None) -> mx.array:
        B_T = x.shape[0]

        if context is None and cached_kv is None:
            qkv = self.to_qkv(x)
            qkv = qkv.reshape(B_T, 3, self.num_heads, self.head_dim)
            q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        elif cached_kv is not None:
            # Use precomputed K, V from cache
            q = self.to_q(x).reshape(B_T, self.num_heads, self.head_dim)
            k, v = cached_kv
        else:
            q = self.to_q(x).reshape(B_T, self.num_heads, self.head_dim)
            kv = self.to_kv(context)
            if kv.ndim == 3:
                S = kv.shape[1]
                kv = kv.reshape(-1, S, 2, self.num_heads, self.head_dim)
                k = kv[:, :, 0]
                v = kv[:, :, 1]
            else:
                kv = kv.reshape(B_T, 2, self.num_heads, self.head_dim)
                k, v = kv[:, 0], kv[:, 1]

        # QK RMSNorm (per-head)
        q = self.q_rms_norm(q)
        if cached_kv is None:
            k = self.k_rms_norm(k)

        # Apply RoPE to self-attention Q and K (not cross-attention)
        if rope_phases is not None and context is None:
            q = apply_rope(q, rope_phases)
            k = apply_rope(k, rope_phases)

        # Attention
        if q.ndim == 3 and k.ndim == 4:
            q = q[None].transpose(0, 2, 1, 3)
            k = k.transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
            out = scaled_dot_product_attention(q, k, v)
            out = out.transpose(0, 2, 1, 3).reshape(B_T, self.channels)
        else:
            q = q[None].transpose(0, 2, 1, 3)
            k = k[None].transpose(0, 2, 1, 3)
            v = v[None].transpose(0, 2, 1, 3)
            out = scaled_dot_product_attention(q, k, v)
            out = out.transpose(0, 2, 1, 3).reshape(B_T, self.channels)

        return self.to_out(out)


class FeedForward(nn.Module):
    """MLP with GELU. Matches TRELLIS weight layout: mlp.0 and mlp.2."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.mlp_0 = nn.Linear(dim, hidden_dim)
        self.mlp_2 = nn.Linear(hidden_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.mlp_2(nn.gelu_approx(self.mlp_0(x)))


class ModulatedBlock(nn.Module):
    """Single DiT block: self-attn + cross-attn + FFN with adaLN-Zero.

    Weight name mapping to TRELLIS checkpoints:
        self_attn.to_qkv   → blocks.N.self_attn.to_qkv
        self_attn.to_out    → blocks.N.self_attn.to_out
        self_attn.q/k_rms_norm → blocks.N.self_attn.q/k_rms_norm
        cross_attn.to_q     → blocks.N.cross_attn.to_q
        cross_attn.to_kv    → blocks.N.cross_attn.to_kv
        cross_attn.to_out   → blocks.N.cross_attn.to_out
        cross_attn.q/k_rms_norm → blocks.N.cross_attn.q/k_rms_norm
        norm2               → blocks.N.norm2  (affine LN for cross-attn)
        mlp.mlp_0           → blocks.N.mlp.mlp.0
        mlp.mlp_2           → blocks.N.mlp.mlp.2
        modulation          → blocks.N.modulation (learned bias, [6*C])
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        context_channels: int,
        mlp_hidden: int,
    ):
        super().__init__()
        self.channels = channels

        # Self-attention (no separate norm — adaLN handles it)
        self.self_attn = MultiHeadAttention(channels, num_heads)

        # Cross-attention with its own affine LayerNorm (fp32 accumulation)
        self.norm2 = LayerNorm32(channels, affine=True)
        self.cross_attn = MultiHeadAttention(channels, num_heads, context_channels)

        # FFN
        self.mlp = FeedForward(channels, mlp_hidden)

        # Per-block learned modulation bias (added to shared timestep modulation)
        self.modulation = mx.zeros((6 * channels,))

    def __call__(
        self,
        x: mx.array,        # [T, C]
        mod: mx.array,       # [6*C] shared modulation from timestep
        context: mx.array,   # [B, L, C_ctx]
        rope_phases: mx.array = None,  # [T, D//2, 2]
        cross_kv_cache: tuple = None,  # precomputed (K, V) for cross-attention
    ) -> mx.array:
        # Add per-block learned bias
        mod = mod + self.modulation

        # Split into 6 modulation params
        C = self.channels
        shift_msa = mod[0*C:1*C]
        scale_msa = mod[1*C:2*C]
        gate_msa  = mod[2*C:3*C]
        shift_mlp = mod[3*C:4*C]
        scale_mlp = mod[4*C:5*C]
        gate_mlp  = mod[5*C:6*C]

        # Self-attention with adaLN-Zero + RoPE
        h = _layernorm_noaffine(x)
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h, rope_phases=rope_phases)
        h = h * gate_msa
        x = x + h

        # Cross-attention (uses its own affine LayerNorm, no adaLN, no RoPE)
        h = self.norm2(x)
        if cross_kv_cache is not None:
            h = self.cross_attn(h, cached_kv=cross_kv_cache)
        else:
            h = self.cross_attn(h, context)
        x = x + h

        # FFN with adaLN-Zero
        h = _layernorm_noaffine(x)
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h

        return x

    def trace(
        self,
        x: mx.array,
        mod: mx.array,
        context: mx.array,
        rope_phases: mx.array = None,
        cross_kv_cache: tuple = None,
    ) -> tuple[mx.array, dict[str, mx.array]]:
        """Run this block once and return its internal boundary tensors."""
        mod = mod + self.modulation

        C = self.channels
        shift_msa = mod[0*C:1*C]
        scale_msa = mod[1*C:2*C]
        gate_msa  = mod[2*C:3*C]
        shift_mlp = mod[3*C:4*C]
        scale_mlp = mod[4*C:5*C]
        gate_mlp  = mod[5*C:6*C]

        trace: dict[str, mx.array] = {}

        h = _layernorm_noaffine(x)
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h, rope_phases=rope_phases)
        trace["block0_self_attn"] = h
        h = h * gate_msa
        x = x + h
        trace["block0_after_self"] = x

        h = self.norm2(x)
        if cross_kv_cache is not None:
            h = self.cross_attn(h, cached_kv=cross_kv_cache)
        else:
            h = self.cross_attn(h, context)
        trace["block0_cross_attn"] = h
        x = x + h
        trace["block0_after_cross"] = x

        h = _layernorm_noaffine(x)
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        trace["block0_mlp"] = h
        h = h * gate_mlp
        x = x + h
        trace["block0_after_mlp"] = x

        return x, trace


def _layernorm_noaffine(x: mx.array, eps: float = 1e-6) -> mx.array:
    """LayerNorm without learnable affine (controlled by adaLN)."""
    orig_dtype = x.dtype
    x = x.astype(mx.float32)
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    x = (x - mean) * mx.rsqrt(var + eps)
    return x.astype(orig_dtype)


class SparseStructureFlowModel(nn.Module):
    """TRELLIS.2 Sparse Structure Flow Model.

    Operates on a dense 3D grid [B, in_channels, R, R, R].
    Flattens to [B*R³, model_channels], runs through DiT blocks,
    reshapes back to [B, out_channels, R, R, R].

    Config (from TRELLIS.2-4B):
        in_channels: 8
        out_channels: 8
        model_channels: 1536
        num_heads: 12
        num_blocks: 30
        mlp_hidden: 8192
        context_channels: 1024 (DINOv3 image features)
        resolution: 16 (sparse structure operates on 16³ grid)
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 8,
        model_channels: int = 1536,
        num_heads: int = 12,
        num_blocks: int = 30,
        mlp_hidden: int = 8192,
        context_channels: int = 1024,
        resolution: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_heads = num_heads
        self.resolution = resolution

        # Timestep embedder
        self.t_embedder = TimestepEmbedder(model_channels)

        # Input/output projections
        self.input_layer = nn.Linear(in_channels, model_channels)
        self.out_layer = nn.Linear(model_channels, out_channels)

        # Shared adaLN modulation: timestep → 6*C
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 6 * model_channels),
        )

        # Transformer blocks
        self.blocks = [
            ModulatedBlock(model_channels, num_heads, context_channels, mlp_hidden)
            for _ in range(num_blocks)
        ]

        # Compilation state (call .compile() to enable)
        self._compiled = False
        self._run_blocks = self._run_blocks_impl

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

    def build_cross_kv_cache(self, cond: mx.array) -> list[tuple]:
        """Precompute cross-attention K, V for all blocks."""
        cache = []
        for block in self.blocks:
            k, v = block.cross_attn.project_kv(cond)
            cache.append((k, v))
        mx.eval(*[t for pair in cache for t in pair])
        return cache

    def trace_first_block(
        self,
        x: mx.array,
        t: mx.array,
        cond: mx.array,
        cross_kv_cache: list = None,
    ) -> dict[str, mx.array]:
        """Capture first-block sparse-flow internals for parity witnesses."""
        B = x.shape[0]
        R = x.shape[2]

        t_emb = self.t_embedder(t)
        mod = self.adaLN_modulation(t_emb)

        x = x.reshape(B, self.in_channels, -1)
        x = x.transpose(0, 2, 1)
        x = x.reshape(B * R * R * R, self.in_channels)
        x = self.input_layer(x)

        head_dim = self.model_channels // self.num_heads
        rope_phases = build_rope_phases(R, head_dim)

        assert B == 1, f"Only B=1 supported for inference, got B={B}"
        block_kv = cross_kv_cache[0] if cross_kv_cache is not None else None
        _x_after, trace = self.blocks[0].trace(
            x, mod[0], cond, rope_phases=rope_phases, cross_kv_cache=block_kv
        )
        trace["input_projected"] = x
        mx.eval(*trace.values())
        return trace

    def __call__(
        self,
        x: mx.array,           # [B, in_channels, R, R, R]
        t: mx.array,           # [B] timestep (scalar per batch)
        cond: mx.array,        # [B, L, context_channels] image conditioning
        cross_kv_cache: list = None,
    ) -> mx.array:
        B = x.shape[0]
        R = x.shape[2]

        # Timestep embedding → shared modulation
        t_emb = self.t_embedder(t)
        mod = self.adaLN_modulation(t_emb)

        # Flatten 3D grid to token sequence
        x = x.reshape(B, self.in_channels, -1)
        x = x.transpose(0, 2, 1)
        x = x.reshape(B * R * R * R, self.in_channels)

        # Project to model channels
        x = self.input_layer(x)

        # Compute 3D RoPE phases from actual input resolution
        head_dim = self.model_channels // self.num_heads
        rope_phases = build_rope_phases(R, head_dim)

        # Run through DiT blocks (B=1 only for inference)
        assert B == 1, f"Only B=1 supported for inference, got B={B}"
        if self._compiled and cross_kv_cache is None:
            x = self._run_blocks(x, mod[0], cond, rope_phases)
        else:
            for i, block in enumerate(self.blocks):
                block_kv = cross_kv_cache[i] if cross_kv_cache is not None else None
                x = block(x, mod[0], cond, rope_phases=rope_phases,
                          cross_kv_cache=block_kv)
                if (i + 1) % 6 == 0:
                    mx.eval(x)

        # Output projection
        # Match reference F.layer_norm default eps on the final output norm.
        x = _layernorm_noaffine(x, eps=1e-5)
        x = self.out_layer(x)                          # [B*R³, out_C]

        # Reshape back to 3D
        x = x.reshape(B, R, R, R, self.out_channels)
        x = x.transpose(0, 4, 1, 2, 3)                # [B, out_C, R, R, R]

        return x
