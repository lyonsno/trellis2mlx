"""SparseStructureFlowModel — DiT over a dense 3D voxel grid.

This is the first stage of the TRELLIS.2 pipeline. It takes a noisy
dense 3D tensor [B, in_channels, R, R, R] and denoises it using a
DiT with adaLN-Zero conditioning on timestep + cross-attention to
image features.

The "sparse" in the name refers to the output being thresholded into
sparse occupancy coordinates — the model itself operates on a dense grid.
"""

import hashlib
import math
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np

from ..modules.norm import LayerNorm32
from ..modules.attention import MultiHeadRMSNorm
from ..sparse_flow_attention import scaled_dot_product_attention
from ..sparse_flow_rope import (
    apply_sparse_flow_rope as apply_rope,
    build_sparse_flow_rope_phases as build_rope_phases,
)
from ..sparse_flow_layernorm import (
    layernorm_noaffine as _sparse_flow_layernorm_noaffine,
    layernorm_noaffine_float32_output as _sparse_flow_terminal_layernorm,
)
from ..source_cuda_gelu import SOURCE_CUDA_BF16_GELU_TANH_BITS_SHA256

_SOURCE_CUDA_BF16_GELU_TANH_TABLE: mx.array | None = None
_SOURCE_CUDA_TERMINAL_LINEAR_PARTITIONS = (0, 308, 616, 924, 1232, 1536)
_SOURCE_CUDA_SPARSE_TERMINAL_ROWS = 4096
_source_cuda_sparse_terminal_linear_kernel = None
MLX_NATIVE_SPARSE_TERMINAL_LINEAR_BACKEND = "mlx-native-linear"
SOURCE_CUDA_T4_SPARSE_TERMINAL_LINEAR_BACKEND = (
    "source-cuda-t4-volta-sgemm-32x128-tn-metal"
)
DEFAULT_SPARSE_TERMINAL_LINEAR_BACKEND = MLX_NATIVE_SPARSE_TERMINAL_LINEAR_BACKEND
SUPPORTED_SPARSE_TERMINAL_LINEAR_BACKENDS = (
    DEFAULT_SPARSE_TERMINAL_LINEAR_BACKEND,
    SOURCE_CUDA_T4_SPARSE_TERMINAL_LINEAR_BACKEND,
)


def sparse_flow_terminal_linear_backend_identity(
    row_count: int,
    *,
    input_width: int,
    output_width: int,
    has_bias: bool,
    backend: str = DEFAULT_SPARSE_TERMINAL_LINEAR_BACKEND,
) -> dict[str, object]:
    geometry = {
        "rows": row_count,
        "input_width": input_width,
        "output_width": output_width,
        "has_bias": has_bias,
    }
    if backend not in SUPPORTED_SPARSE_TERMINAL_LINEAR_BACKENDS:
        raise ValueError(
            f"unsupported sparse terminal linear backend {backend!r}; "
            f"expected one of {SUPPORTED_SPARSE_TERMINAL_LINEAR_BACKENDS}"
        )
    source_geometry = {
        "rows": _SOURCE_CUDA_SPARSE_TERMINAL_ROWS,
        "input_width": 1536,
        "output_width": 8,
        "has_bias": True,
    }
    if backend == SOURCE_CUDA_T4_SPARSE_TERMINAL_LINEAR_BACKEND:
        if geometry != source_geometry:
            raise ValueError(
                "source-CUDA sparse terminal linear requires authenticated "
                f"geometry {source_geometry}, got {geometry}"
            )
        return {
            "backend": backend,
            "algorithm": (
                "five-contiguous-fp32-fma-partitions-with-bias-epilogue"
            ),
            "experimental": True,
            "cuda_source_kernel": "volta_sgemm_32x128_tn",
            "authenticated_contract": {
                **geometry,
                "input_dtype": "float32",
                "weight_dtype": "float32",
                "bias_dtype": "float32",
                "output_dtype": "float32",
                "partition_bounds": list(
                    _SOURCE_CUDA_TERMINAL_LINEAR_PARTITIONS
                ),
                "cuda_device_anchor": "Tesla T4 sm_75",
                "torch_anchor": "2.10.0+cu128",
                "source_block_trace_sha256": (
                    "83d7e731dbda4c2244907d7402de166b69df7f1c2d3fef93e5842685064b5ded"
                ),
            },
        }
    return {
        "backend": backend,
        "algorithm": "mlx-nn-linear",
        "experimental": False,
        "effective_contract": geometry,
    }


def _source_cuda_t4_sparse_terminal_linear(
    x: mx.array,
    linear: nn.Linear,
) -> mx.array:
    """Execute the observed T4 SGEMM reduction law for sparse projection."""
    global _source_cuda_sparse_terminal_linear_kernel

    if x.ndim != 2 or x.shape[1] != 1536:
        raise ValueError("T4 sparse terminal linear requires width-1536 input")
    if linear.weight.ndim != 2 or linear.weight.shape != (8, 1536):
        raise ValueError("T4 sparse terminal linear requires 8x1536 weight")
    if linear.bias is None or linear.bias.shape != (8,):
        raise ValueError("T4 sparse terminal linear requires width-8 bias")

    if _source_cuda_sparse_terminal_linear_kernel is None:
        source = r"""
                constexpr uint reduction = 1536;
                constexpr uint columns = 8;
                constexpr uint partition_bounds[6] = {
                    0, 308, 616, 924, 1232, 1536
                };

                uint output_index = thread_position_in_grid.x;
                uint row_count = row_count_input[0];
                uint output_count = row_count * columns;
                if (output_index >= output_count) {
                    return;
                }

                uint row = output_index / columns;
                uint column = output_index - row * columns;
                uint input_offset = row * reduction;
                uint weight_offset = column * reduction;
                float accumulator = bias[column];

                for (uint partition = 0; partition < 5; ++partition) {
                    float partial = 0.0f;
                    for (uint k = partition_bounds[partition];
                         k < partition_bounds[partition + 1];
                         ++k) {
                        partial = metal::fma(
                            inp[input_offset + k],
                            weight[weight_offset + k],
                            partial);
                    }
                    accumulator = accumulator + partial;
                }
                out[output_index] = accumulator;
            """
        _source_cuda_sparse_terminal_linear_kernel = mx.fast.metal_kernel(
            name="sparse_flow_source_cuda_t4_terminal_linear_fp32",
            input_names=["inp", "weight", "bias", "row_count_input"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )

    return _source_cuda_sparse_terminal_linear_kernel(
        inputs=[
            x.astype(mx.float32),
            linear.weight.astype(mx.float32),
            linear.bias.astype(mx.float32),
            mx.array([x.shape[0]], dtype=mx.uint32),
        ],
        grid=(x.shape[0] * 8, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(x.shape[0], 8)],
        output_dtypes=[mx.float32],
    )[0]


def _sparse_terminal_linear(
    x: mx.array,
    linear: nn.Linear,
    *,
    backend: str = DEFAULT_SPARSE_TERMINAL_LINEAR_BACKEND,
) -> mx.array:
    identity = sparse_flow_terminal_linear_backend_identity(
        int(x.shape[0]),
        input_width=int(linear.weight.shape[1]),
        output_width=int(linear.weight.shape[0]),
        has_bias=linear.bias is not None,
        backend=backend,
    )
    if identity["backend"] == SOURCE_CUDA_T4_SPARSE_TERMINAL_LINEAR_BACKEND:
        return _source_cuda_t4_sparse_terminal_linear(x, linear)
    return linear(x)


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


def _source_shared_modulation(
    t: mx.array,
    t_embedder: TimestepEmbedder,
    adaLN_modulation: nn.Sequential,
    dtype: mx.Dtype,
) -> mx.array:
    """Source-compatible shared timestep modulation.

    The reference keeps this small MLP in float32 and only casts its output to
    the transformer torso dtype. MLX's float32 matmul can round a few entries
    differently from PyTorch; numpy's float32 path matches the reference for
    this control-surface-sized computation.
    """
    t_np = np.array(t, dtype=np.float32)
    if t_np.ndim == 0:
        t_np = t_np.reshape(1)
    half = t_embedder.freq_dim // 2
    freqs = np.exp(
        -math.log(10000) * np.arange(half, dtype=np.float32) / half
    ).astype(np.float32)
    args = t_np[:, None].astype(np.float32) * freqs[None, :]
    emb = np.concatenate([np.cos(args), np.sin(args)], axis=-1).astype(np.float32)
    t0 = _source_linear(emb, t_embedder.mlp_0)
    t1 = _source_linear(_source_silu(t0), t_embedder.mlp_2)
    mod = _source_linear(_source_silu(t1), adaLN_modulation.layers[1])
    return mx.array(mod).astype(dtype)


def _sparse_shared_modulation(
    t: mx.array,
    t_embedder: TimestepEmbedder,
    adaLN_modulation: nn.Sequential,
    compute_dtype: mx.Dtype,
    sparse_timestep_modulation_lut=None,
) -> mx.array:
    if sparse_timestep_modulation_lut is not None:
        return sparse_timestep_modulation_lut.lookup_mlx(t, compute_dtype)
    return _source_shared_modulation(
        t,
        t_embedder,
        adaLN_modulation,
        compute_dtype,
    )


def _source_linear(x: np.ndarray, linear: nn.Linear) -> np.ndarray:
    weight = _mx_to_float32_np(linear.weight)
    bias = _mx_to_float32_np(linear.bias) if linear.bias is not None else None
    y = x @ weight.T
    if bias is not None:
        y = y + bias
    return y.astype(np.float32)


def _mx_to_float32_np(value: mx.array) -> np.ndarray:
    return np.array(value.astype(mx.float32), dtype=np.float32)


def _source_silu(x: np.ndarray) -> np.ndarray:
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


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

    def trace_self_attention(
        self,
        x: mx.array,
        rope_phases: mx.array = None,
        trace_prefix: str = "block0",
    ) -> tuple[mx.array, dict[str, mx.array]]:
        B_T = x.shape[0]
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(B_T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        trace = {
            f"{trace_prefix}_q_pre_norm": q,
            f"{trace_prefix}_k_pre_norm": k,
            f"{trace_prefix}_v": v,
        }

        q = self.q_rms_norm(q)
        k = self.k_rms_norm(k)
        trace[f"{trace_prefix}_q_post_norm"] = q
        trace[f"{trace_prefix}_k_post_norm"] = k

        if rope_phases is not None:
            q = apply_rope(q, rope_phases)
            k = apply_rope(k, rope_phases)
        trace[f"{trace_prefix}_q_post_rope"] = q
        trace[f"{trace_prefix}_k_post_rope"] = k

        q_b = q[None].transpose(0, 2, 1, 3)
        k_b = k[None].transpose(0, 2, 1, 3)
        v_b = v[None].transpose(0, 2, 1, 3)
        out = scaled_dot_product_attention(q_b, k_b, v_b)
        out = out.transpose(0, 2, 1, 3).reshape(B_T, self.channels)
        trace[f"{trace_prefix}_attention_raw"] = out

        return self.to_out(out), trace

    def trace_cross_attention(
        self,
        x: mx.array,
        context: mx.array,
        cached_kv: tuple = None,
        trace_prefix: str = "block0",
    ) -> tuple[mx.array, dict[str, mx.array]]:
        B_T = x.shape[0]
        q = self.to_q(x).reshape(B_T, self.num_heads, self.head_dim)
        kv = self.to_kv(context)
        if kv.ndim == 3:
            S = kv.shape[1]
            kv = kv.reshape(-1, S, 2, self.num_heads, self.head_dim)
            k_pre = kv[:, :, 0]
            v_pre = kv[:, :, 1]
        else:
            kv = kv.reshape(B_T, 2, self.num_heads, self.head_dim)
            k_pre, v_pre = kv[:, 0], kv[:, 1]

        trace = {
            f"{trace_prefix}_cross_q_pre_norm": q,
            f"{trace_prefix}_cross_k_pre_norm": k_pre,
            f"{trace_prefix}_cross_v": v_pre,
        }

        q = self.q_rms_norm(q)
        k_norm = self.k_rms_norm(k_pre)
        trace[f"{trace_prefix}_cross_q_post_norm"] = q
        trace[f"{trace_prefix}_cross_k_post_norm"] = k_norm

        if cached_kv is not None:
            k, v = cached_kv
            trace[f"{trace_prefix}_cross_k_cached_post_norm"] = k
            trace[f"{trace_prefix}_cross_v_cached"] = v
        else:
            k, v = k_norm, v_pre

        q_b = q[None].transpose(0, 2, 1, 3)
        if k.ndim == 4:
            k_b = k.transpose(0, 2, 1, 3)
            v_b = v.transpose(0, 2, 1, 3)
        else:
            k_b = k[None].transpose(0, 2, 1, 3)
            v_b = v[None].transpose(0, 2, 1, 3)
        out = scaled_dot_product_attention(q_b, k_b, v_b)
        out = out.transpose(0, 2, 1, 3).reshape(B_T, self.channels)
        trace[f"{trace_prefix}_cross_attention_raw"] = out

        return self.to_out(out), trace


class FeedForward(nn.Module):
    """MLP with GELU. Matches TRELLIS weight layout: mlp.0 and mlp.2."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.mlp_0 = nn.Linear(dim, hidden_dim)
        self.mlp_2 = nn.Linear(hidden_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.mlp_2(_gelu_tanh(self.mlp_0(x)))


def _gelu_tanh(x: mx.array) -> mx.array:
    """PyTorch nn.GELU(approximate="tanh") with source dtype restoration."""
    orig_dtype = x.dtype
    if orig_dtype == mx.bfloat16:
        table = _source_cuda_bf16_gelu_tanh_table()
        indices = mx.view(x, mx.uint16).astype(mx.uint32)
        return mx.view(mx.take(table, indices), mx.bfloat16)
    x = x.astype(mx.float32)
    tanh_scale = math.sqrt(2.0 / math.pi)
    cubic_scale = 0.044715
    x = 0.5 * x * (1.0 + mx.tanh(tanh_scale * (x + cubic_scale * x * x * x)))
    return x.astype(orig_dtype)


def _source_cuda_bf16_gelu_tanh_table() -> mx.array:
    global _SOURCE_CUDA_BF16_GELU_TANH_TABLE
    if _SOURCE_CUDA_BF16_GELU_TANH_TABLE is None:
        table_path = Path(__file__).with_name("source_cuda_bf16_gelu_tanh_table.npy")
        bits = np.load(table_path, allow_pickle=False)
        if bits.shape != (65536,) or bits.dtype != np.uint16:
            raise ValueError(
                "source CUDA BF16 GELU table must contain 65,536 uint16 entries"
            )
        digest = hashlib.sha256(bits.tobytes()).hexdigest()
        if digest != SOURCE_CUDA_BF16_GELU_TANH_BITS_SHA256:
            raise ValueError(
                "source CUDA BF16 GELU table digest mismatch: "
                f"expected {SOURCE_CUDA_BF16_GELU_TANH_BITS_SHA256}, got {digest}"
            )
        _SOURCE_CUDA_BF16_GELU_TANH_TABLE = mx.array(bits)
        mx.eval(_SOURCE_CUDA_BF16_GELU_TANH_TABLE)
    return _SOURCE_CUDA_BF16_GELU_TANH_TABLE


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
        *,
        sparse_flow_layernorm: bool = False,
        shape_flow_layernorm: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.sparse_flow_layernorm = sparse_flow_layernorm
        self.shape_flow_layernorm = shape_flow_layernorm

        # Self-attention (no separate norm — adaLN handles it)
        self.self_attn = MultiHeadAttention(channels, num_heads)

        # Cross-attention with its own affine LayerNorm (fp32 accumulation)
        self.norm2 = LayerNorm32(
            channels,
            affine=True,
            sparse_flow_layernorm=sparse_flow_layernorm,
            shape_flow_layernorm=shape_flow_layernorm,
        )
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
        mod = (self.modulation + mod).astype(mod.dtype)

        # Split into 6 modulation params
        C = self.channels
        shift_msa = mod[0*C:1*C]
        scale_msa = mod[1*C:2*C]
        gate_msa  = mod[2*C:3*C]
        shift_mlp = mod[3*C:4*C]
        scale_mlp = mod[4*C:5*C]
        gate_mlp  = mod[5*C:6*C]

        # Self-attention with adaLN-Zero + RoPE
        h = self._layernorm_noaffine(x)
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
        h = self._layernorm_noaffine(x)
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h

        return x

    def forward_with_injection(
        self,
        x: mx.array,
        mod: mx.array,
        context: mx.array,
        *,
        injection,
        branch: str,
        rope_phases: mx.array = None,
        cross_kv_cache: tuple = None,
    ) -> mx.array:
        mod = (self.modulation + mod).astype(mod.dtype)

        C = self.channels
        shift_msa = mod[0*C:1*C]
        scale_msa = mod[1*C:2*C]
        gate_msa = mod[2*C:3*C]
        shift_mlp = mod[3*C:4*C]
        scale_mlp = mod[4*C:5*C]
        gate_mlp = mod[5*C:6*C]

        if _is_rowwise_layernorm_correction(injection):
            h = _layernorm_noaffine_rowwise_perturbed(
                x,
                injection,
                branch=branch,
                eps=1e-6,
            )
        else:
            h = self._layernorm_noaffine(x)
        norm1_injection = _injection_at_stage(injection, "norm1")
        if norm1_injection is not None:
            h = _injected_tensor_like(norm1_injection, h, branch=branch)
        h = h * (1 + scale_msa) + shift_msa
        modulated_injection = _injection_at_stage(injection, "modulated_self_input")
        if modulated_injection is not None:
            h = _injected_tensor_like(modulated_injection, h, branch=branch)
        attention_injection = _injection_at_stage(injection, "attention_raw")
        if attention_injection is not None:
            _, attn_trace = self.self_attn.trace_self_attention(
                h,
                rope_phases=rope_phases,
                trace_prefix="injected",
            )
            h = _injected_tensor_like(
                attention_injection,
                attn_trace["injected_attention_raw"],
                branch=branch,
            )
            h = self.self_attn.to_out(h)
        else:
            h = self.self_attn(h, rope_phases=rope_phases)
        h = h * gate_msa
        x = x + h
        after_self_injection = _injection_at_stage(injection, "after_self")
        if after_self_injection is not None:
            x = _injected_tensor_like(after_self_injection, x, branch=branch)

        h = self.norm2(x)
        cross_attention_injection = _injection_at_stage(injection, "cross_attention_raw")
        if cross_attention_injection is not None:
            _, cross_trace = self.cross_attn.trace_cross_attention(
                h,
                context,
                cached_kv=cross_kv_cache,
                trace_prefix="injected",
            )
            h = _injected_tensor_like(
                cross_attention_injection,
                cross_trace["injected_cross_attention_raw"],
                branch=branch,
            )
            h = self.cross_attn.to_out(h)
        elif cross_kv_cache is not None:
            h = self.cross_attn(h, cached_kv=cross_kv_cache)
        else:
            h = self.cross_attn(h, context)
        x = x + h
        after_cross_injection = _injection_at_stage(injection, "after_cross")
        if after_cross_injection is not None:
            x = _injected_tensor_like(after_cross_injection, x, branch=branch)

        h = self._layernorm_noaffine(x)
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        after_mlp_injection = _injection_at_stage(injection, "after_mlp")
        if after_mlp_injection is not None:
            x = _injected_tensor_like(after_mlp_injection, x, branch=branch)
        return x

    def trace(
        self,
        x: mx.array,
        mod: mx.array,
        context: mx.array,
        rope_phases: mx.array = None,
        cross_kv_cache: tuple = None,
        trace_prefix: str = "block0",
        injection=None,
        branch: str = "pos",
    ) -> tuple[mx.array, dict[str, mx.array]]:
        mod = (self.modulation + mod).astype(mod.dtype)

        C = self.channels
        shift_msa = mod[0*C:1*C]
        scale_msa = mod[1*C:2*C]
        gate_msa  = mod[2*C:3*C]
        shift_mlp = mod[3*C:4*C]
        scale_mlp = mod[4*C:5*C]
        gate_mlp  = mod[5*C:6*C]

        trace: dict[str, mx.array] = {
            f"{trace_prefix}_shift_msa": shift_msa,
            f"{trace_prefix}_scale_msa": scale_msa,
            f"{trace_prefix}_gate_msa": gate_msa,
            f"{trace_prefix}_shift_mlp": shift_mlp,
            f"{trace_prefix}_scale_mlp": scale_mlp,
            f"{trace_prefix}_gate_mlp": gate_mlp,
        }

        if _is_rowwise_layernorm_correction(injection):
            h = _layernorm_noaffine_rowwise_perturbed(
                x,
                injection,
                branch=branch,
                eps=1e-6,
            )
        else:
            h = self._layernorm_noaffine(x)
        norm1_injection = _injection_at_stage(injection, "norm1")
        if norm1_injection is not None:
            h = _injected_tensor_like(norm1_injection, h, branch=branch)
        trace[f"{trace_prefix}_norm1"] = h
        h = h * (1 + scale_msa) + shift_msa
        modulated_injection = _injection_at_stage(injection, "modulated_self_input")
        if modulated_injection is not None:
            h = _injected_tensor_like(modulated_injection, h, branch=branch)
        trace[f"{trace_prefix}_modulated_self_input"] = h
        h, attn_trace = self.self_attn.trace_self_attention(
            h,
            rope_phases=rope_phases,
            trace_prefix=trace_prefix,
        )
        attention_injection = _injection_at_stage(injection, "attention_raw")
        if attention_injection is not None:
            attention_raw = _injected_tensor_like(
                attention_injection,
                attn_trace[f"{trace_prefix}_attention_raw"],
                branch=branch,
            )
            attn_trace[f"{trace_prefix}_attention_raw"] = attention_raw
            h = self.self_attn.to_out(attention_raw)
        trace.update(attn_trace)
        trace[f"{trace_prefix}_self_attn"] = h
        h = h * gate_msa
        x = x + h
        after_self_injection = _injection_at_stage(injection, "after_self")
        if after_self_injection is not None:
            x = _injected_tensor_like(after_self_injection, x, branch=branch)
        trace[f"{trace_prefix}_after_self"] = x

        h = self.norm2(x)
        trace[f"{trace_prefix}_norm2"] = h
        h, cross_trace = self.cross_attn.trace_cross_attention(
            h,
            context,
            cached_kv=cross_kv_cache,
            trace_prefix=trace_prefix,
        )
        cross_attention_injection = _injection_at_stage(injection, "cross_attention_raw")
        if cross_attention_injection is not None:
            cross_attention_raw = _injected_tensor_like(
                cross_attention_injection,
                cross_trace[f"{trace_prefix}_cross_attention_raw"],
                branch=branch,
            )
            cross_trace[f"{trace_prefix}_cross_attention_raw"] = cross_attention_raw
            h = self.cross_attn.to_out(cross_attention_raw)
        trace.update(cross_trace)
        trace[f"{trace_prefix}_cross_attn"] = h
        x = x + h
        after_cross_injection = _injection_at_stage(injection, "after_cross")
        if after_cross_injection is not None:
            x = _injected_tensor_like(after_cross_injection, x, branch=branch)
        trace[f"{trace_prefix}_after_cross"] = x

        h = self._layernorm_noaffine(x)
        h = h * (1 + scale_mlp) + shift_mlp
        trace[f"{trace_prefix}_mlp_input"] = h
        h_fc1 = self.mlp.mlp_0(h)
        trace[f"{trace_prefix}_mlp_fc1"] = h_fc1
        h_gelu = _gelu_tanh(h_fc1)
        trace[f"{trace_prefix}_mlp_gelu"] = h_gelu
        h = self.mlp.mlp_2(h_gelu)
        trace[f"{trace_prefix}_mlp_fc2"] = h
        trace[f"{trace_prefix}_mlp"] = h
        h = h * gate_mlp
        trace[f"{trace_prefix}_mlp_gated"] = h
        x = x + h
        after_mlp_injection = _injection_at_stage(injection, "after_mlp")
        if after_mlp_injection is not None:
            x = _injected_tensor_like(after_mlp_injection, x, branch=branch)
        trace[f"{trace_prefix}_after_mlp"] = x

        return x, trace

    def _layernorm_noaffine(self, x: mx.array, eps: float = 1e-6) -> mx.array:
        if self.sparse_flow_layernorm:
            return _sparse_flow_layernorm_noaffine(x, eps=eps)
        if self.shape_flow_layernorm:
            from ..shape_flow_layernorm import layernorm_noaffine

            return layernorm_noaffine(x, eps=eps)
        return _layernorm_noaffine(x, eps=eps)


def _layernorm_noaffine(x: mx.array, eps: float = 1e-6) -> mx.array:
    """LayerNorm without learnable affine (controlled by adaLN)."""
    if x.dtype in (mx.bfloat16, mx.float16):
        input_dtype = x.dtype
        xf = x.astype(mx.float32)
        mean = mx.mean(xf, axis=-1, keepdims=True)
        var = mx.mean((xf - mean) * (xf - mean), axis=-1, keepdims=True)
        return ((xf - mean) * mx.rsqrt(var + eps)).astype(input_dtype)
    return mx.fast.layer_norm(x, None, None, eps)


def _is_rowwise_layernorm_correction(injection) -> bool:
    return getattr(injection, "stage", None) in {"norm1_rowwise_scale", "norm1_rowwise_bias"}


def _injection_at_stage(injection, stage: str):
    if injection is None:
        return None
    candidates = injection if isinstance(injection, (tuple, list)) else (injection,)
    matches = [candidate for candidate in candidates if getattr(candidate, "stage", None) == stage]
    if len(matches) > 1:
        raise ValueError(f"multiple active shape block injections target stage {stage!r}")
    return matches[0] if matches else None


def _injections_for_block(injection, block_index: int):
    if injection is None:
        return None
    if hasattr(injection, "injections_for_block"):
        matches = injection.injections_for_block(block_index)
        return matches or None
    if hasattr(injection, "injection_for_block"):
        return injection.injection_for_block(block_index)
    return injection if block_index == injection.block_index else None


def _layernorm_noaffine_rowwise_perturbed(
    x: mx.array,
    correction,
    *,
    branch: str,
    eps: float = 1e-6,
) -> mx.array:
    """No-affine LayerNorm with diagnostic row-local perturbations before BF16 cast."""
    base = _layernorm_noaffine(x, eps=eps)
    rows = np.asarray(correction.array_for_branch(branch), dtype=np.float32)
    if rows.size == 0:
        return base
    if rows.ndim != 2 or rows.shape[1] != 3:
        raise ValueError(
            "rowwise LayerNorm correction rows must have shape [N, 3] "
            f"for batch/token/value, got {rows.shape}"
        )

    leading_shape = tuple(int(dim) for dim in x.shape[:-1])
    if len(leading_shape) == 1:
        values = np.zeros(leading_shape, dtype=np.float32)
        active = np.zeros(leading_shape, dtype=np.bool_)
        for batch_f, token_f, value_f in rows:
            batch = int(batch_f)
            token = int(token_f)
            if batch != 0:
                raise ValueError(
                    "rowwise LayerNorm correction for 2D tokens requires batch 0, "
                    f"got batch {batch}"
                )
            if token < 0 or token >= leading_shape[0]:
                raise ValueError(
                    f"rowwise LayerNorm correction token {token} outside [0, {leading_shape[0]})"
                )
            values[token] = float(value_f)
            active[token] = True
    elif len(leading_shape) == 2:
        values = np.zeros(leading_shape, dtype=np.float32)
        active = np.zeros(leading_shape, dtype=np.bool_)
        for batch_f, token_f, value_f in rows:
            batch = int(batch_f)
            token = int(token_f)
            if batch < 0 or batch >= leading_shape[0] or token < 0 or token >= leading_shape[1]:
                raise ValueError(
                    "rowwise LayerNorm correction coordinate "
                    f"({batch}, {token}) outside {leading_shape}"
                )
            values[batch, token] = float(value_f)
            active[batch, token] = True
    else:
        raise ValueError(
            "rowwise LayerNorm correction supports [tokens, channels] or "
            f"[batch, tokens, channels], got {x.shape}"
        )

    xf = x.astype(mx.float32)
    mean = mx.mean(xf, axis=-1, keepdims=True)
    centered = xf - mean
    var = mx.mean(centered * centered, axis=-1, keepdims=True)
    normalized = centered * mx.rsqrt(var + eps)
    row_values = mx.array(values)[..., None]
    row_active = mx.array(active)[..., None]
    if correction.mode == "scale":
        corrected = normalized * (1.0 + row_values)
    elif correction.mode == "bias":
        corrected = normalized + row_values
    else:
        raise ValueError(f"unknown rowwise LayerNorm correction mode: {correction.mode!r}")
    return mx.where(row_active, corrected.astype(base.dtype), base)


def _injected_tensor_like(injection, reference: mx.array, *, branch: str) -> mx.array:
    array = injection.array_for_branch(branch)
    injected = mx.array(array)
    if injected.ndim == reference.ndim + 1 and injected.shape[0] == 1:
        injected = injected[0]
    if injected.shape != reference.shape:
        raise ValueError(
            "block injection shape mismatch for "
            f"{branch} {injection.stage}: expected {reference.shape}, got {injected.shape}"
        )
    source_delta_scale = float(getattr(injection, "source_delta_scale", 1.0))
    if source_delta_scale != 1.0:
        injected = (
            reference.astype(mx.float32)
            + source_delta_scale
            * (injected.astype(mx.float32) - reference.astype(mx.float32))
        )
    return injected.astype(reference.dtype)


def _infer_compute_dtype(module: nn.Module) -> mx.Dtype:
    """Infer the source checkpoint dtype currently loaded into a flow module."""
    for _, value in mlx.utils.tree_flatten(module.parameters()):
        if hasattr(value, "dtype") and value.dtype in (mx.bfloat16, mx.float16):
            return value.dtype
    return mx.float32


def _cast_block_linears(block: ModulatedBlock, dtype: mx.Dtype) -> None:
    block.modulation = block.modulation.astype(dtype)
    linears = (
        block.self_attn.to_qkv,
        block.self_attn.to_out,
        block.cross_attn.to_q,
        block.cross_attn.to_kv,
        block.cross_attn.to_out,
        block.mlp.mlp_0,
        block.mlp.mlp_2,
    )
    for linear in linears:
        linear.weight = linear.weight.astype(dtype)
        if linear.bias is not None:
            linear.bias = linear.bias.astype(dtype)


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
        terminal_linear_backend: str = DEFAULT_SPARSE_TERMINAL_LINEAR_BACKEND,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_heads = num_heads
        self.resolution = resolution
        if terminal_linear_backend not in SUPPORTED_SPARSE_TERMINAL_LINEAR_BACKENDS:
            raise ValueError(
                f"unsupported sparse terminal linear backend "
                f"{terminal_linear_backend!r}"
            )
        self.terminal_linear_backend = terminal_linear_backend

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
            ModulatedBlock(
                model_channels,
                num_heads,
                context_channels,
                mlp_hidden,
                sparse_flow_layernorm=True,
            )
            for _ in range(num_blocks)
        ]
        for block in self.blocks:
            _cast_block_linears(block, mx.bfloat16)

        # Compilation state (call .compile() to enable)
        self._compiled = False
        self._run_blocks = self._run_blocks_impl

    def terminal_linear_backend_identity(self, row_count: int) -> dict[str, object]:
        return sparse_flow_terminal_linear_backend_identity(
            row_count,
            input_width=int(self.out_layer.weight.shape[1]),
            output_width=int(self.out_layer.weight.shape[0]),
            has_bias=self.out_layer.bias is not None,
            backend=self.terminal_linear_backend,
        )

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
        cond = cond.astype(_infer_compute_dtype(self))
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
        sparse_timestep_modulation_lut=None,
    ) -> dict[str, mx.array]:
        return self.trace_block(
            x,
            t,
            cond,
            block_index=0,
            cross_kv_cache=cross_kv_cache,
            sparse_timestep_modulation_lut=sparse_timestep_modulation_lut,
        )

    def trace_block(
        self,
        x: mx.array,
        t: mx.array,
        cond: mx.array,
        block_index: int = 0,
        cross_kv_cache: list = None,
        sparse_block_injection=None,
        sparse_block_injection_branch: str | None = None,
        sparse_timestep_modulation_lut=None,
    ) -> dict[str, mx.array]:
        if block_index < 0 or block_index >= len(self.blocks):
            raise ValueError(f"block_index must be in [0, {len(self.blocks) - 1}], got {block_index}")

        input_dtype = x.dtype
        B = x.shape[0]
        R = x.shape[2]

        compute_dtype = _infer_compute_dtype(self)
        mod = _sparse_shared_modulation(
            t,
            self.t_embedder,
            self.adaLN_modulation,
            compute_dtype,
            sparse_timestep_modulation_lut,
        )

        x = x.reshape(B, self.in_channels, -1)
        x = x.transpose(0, 2, 1)
        x = x.reshape(B * R * R * R, self.in_channels)
        x = self.input_layer(x)
        x = x.astype(compute_dtype)
        cond = cond.astype(compute_dtype)

        head_dim = self.model_channels // self.num_heads
        rope_phases = build_rope_phases(R, head_dim)

        assert B == 1, f"Only B=1 supported for inference, got B={B}"
        trace: dict[str, mx.array] = {"input_projected": x}
        for i, block in enumerate(self.blocks):
            block_kv = cross_kv_cache[i] if cross_kv_cache is not None else None
            block_injection = None
            if sparse_block_injection is not None:
                block_injection = _injections_for_block(sparse_block_injection, i)
            if i == block_index:
                trace_prefix = f"block{block_index}"
                trace[f"{trace_prefix}_input"] = x
                _x_after, block_trace = block.trace(
                    x,
                    mod[0],
                    cond,
                    rope_phases=rope_phases,
                    cross_kv_cache=block_kv,
                    trace_prefix=trace_prefix,
                    injection=block_injection,
                    branch=sparse_block_injection_branch or "pos",
                )
                trace.update(block_trace)
                if i == len(self.blocks) - 1:
                    x = _x_after
                    trace["final_input"] = x
                    x = _sparse_flow_terminal_layernorm(x, eps=1e-5)
                    trace["final_norm"] = x
                    x = _sparse_terminal_linear(
                        x,
                        self.out_layer,
                        backend=self.terminal_linear_backend,
                    )
                    trace["final_out_flat"] = x
                    trace["final_output"] = x.reshape(B, R, R, R, self.out_channels).transpose(0, 4, 1, 2, 3)
                break
            if block_injection is not None:
                x = block.forward_with_injection(
                    x,
                    mod[0],
                    cond,
                    injection=block_injection,
                    branch=sparse_block_injection_branch or "pos",
                    rope_phases=rope_phases,
                    cross_kv_cache=block_kv,
                )
            else:
                x = block(
                    x,
                    mod[0],
                    cond,
                    rope_phases=rope_phases,
                    cross_kv_cache=block_kv,
                )
            if (i + 1) % 6 == 0:
                mx.eval(x)
        mx.eval(*trace.values())
        return trace

    def trace_projected_block_input(
        self,
        block_input: mx.array,
        t: mx.array,
        cond: mx.array,
        block_index: int = 0,
        resolution: int | None = None,
        cross_kv_cache: list = None,
        sparse_block_injection=None,
        sparse_block_injection_branch: str | None = None,
        sparse_timestep_modulation_lut=None,
    ) -> dict[str, mx.array]:
        """Trace one block starting from a saved projected block input.

        `trace_block` replays the whole dense sparse-flow prefix before the
        selected block. This diagnostic helper is for same-input local parity:
        feed a reference block input directly through the corresponding MLX
        block under the requested timestep/conditioning route.
        """
        if block_index < 0 or block_index >= len(self.blocks):
            raise ValueError(f"block_index must be in [0, {len(self.blocks) - 1}], got {block_index}")
        if block_input.ndim == 3:
            if block_input.shape[0] != 1:
                raise ValueError(
                    "projected block input with a batch dimension must have B=1, "
                    f"got {block_input.shape}"
                )
            block_input = block_input[0]
        elif block_input.ndim != 2:
            raise ValueError(
                "projected block input must have shape [T,C] or [1,T,C], "
                f"got {block_input.shape}"
            )
        if block_input.shape[1] != self.model_channels:
            raise ValueError(
                "projected block input channel dimension must match model_channels "
                f"{self.model_channels}, got {block_input.shape}"
            )

        token_count = block_input.shape[0]
        if resolution is None:
            inferred_resolution = round(token_count ** (1.0 / 3.0))
            if inferred_resolution ** 3 != token_count:
                raise ValueError(
                    "projected block input token count must be a perfect cube when "
                    f"resolution is omitted, got {token_count}"
                )
            resolution = inferred_resolution
        elif resolution ** 3 != token_count:
            raise ValueError(
                f"resolution {resolution} implies {resolution ** 3} tokens, "
                f"got projected block input with {token_count}"
            )

        input_dtype = block_input.dtype
        compute_dtype = _infer_compute_dtype(self)
        mod = _sparse_shared_modulation(
            t,
            self.t_embedder,
            self.adaLN_modulation,
            compute_dtype,
            sparse_timestep_modulation_lut,
        )
        x = block_input.astype(compute_dtype)
        cond = cond.astype(compute_dtype)

        head_dim = self.model_channels // self.num_heads
        rope_phases = build_rope_phases(resolution, head_dim)
        block_kv = cross_kv_cache[block_index] if cross_kv_cache is not None else None

        trace_prefix = f"block{block_index}"
        trace: dict[str, mx.array] = {f"{trace_prefix}_input": x}
        block_injection = None
        if sparse_block_injection is not None:
            block_injection = _injections_for_block(sparse_block_injection, block_index)
        x_after, block_trace = self.blocks[block_index].trace(
            x,
            mod[0],
            cond,
            rope_phases=rope_phases,
            cross_kv_cache=block_kv,
            trace_prefix=trace_prefix,
            injection=block_injection,
            branch=sparse_block_injection_branch or "pos",
        )
        trace.update(block_trace)
        if block_index == len(self.blocks) - 1:
            trace["final_input"] = x_after
            x_final = _sparse_flow_terminal_layernorm(x_after, eps=1e-5)
            trace["final_norm"] = x_final
            x_final = _sparse_terminal_linear(
                x_final,
                self.out_layer,
                backend=self.terminal_linear_backend,
            )
            trace["final_out_flat"] = x_final
            trace["final_output"] = (
                x_final.reshape(1, resolution, resolution, resolution, self.out_channels)
                .transpose(0, 4, 1, 2, 3)
            )
        mx.eval(*trace.values())
        return trace

    def __call__(
        self,
        x: mx.array,           # [B, in_channels, R, R, R]
        t: mx.array,           # [B] timestep (scalar per batch)
        cond: mx.array,        # [B, L, context_channels] image conditioning
        cross_kv_cache: list = None,
        sparse_block_injection=None,
        sparse_block_injection_branch: str | None = None,
        sparse_timestep_modulation_lut=None,
    ) -> mx.array:
        input_dtype = x.dtype
        B = x.shape[0]
        R = x.shape[2]

        # Flatten 3D grid to token sequence
        x = x.reshape(B, self.in_channels, -1)
        x = x.transpose(0, 2, 1)
        x = x.reshape(B * R * R * R, self.in_channels)

        # Project to model channels
        x = self.input_layer(x)
        compute_dtype = _infer_compute_dtype(self)
        mod = _sparse_shared_modulation(
            t,
            self.t_embedder,
            self.adaLN_modulation,
            compute_dtype,
            sparse_timestep_modulation_lut,
        )
        x = x.astype(compute_dtype)
        cond = cond.astype(compute_dtype)

        # Compute 3D RoPE phases from actual input resolution
        head_dim = self.model_channels // self.num_heads
        rope_phases = build_rope_phases(R, head_dim)

        # Run through DiT blocks (B=1 only for inference)
        assert B == 1, f"Only B=1 supported for inference, got B={B}"
        if self._compiled and cross_kv_cache is None and sparse_block_injection is None:
            x = self._run_blocks(x, mod[0], cond, rope_phases)
        else:
            for i, block in enumerate(self.blocks):
                block_kv = cross_kv_cache[i] if cross_kv_cache is not None else None
                block_injection = None
                if sparse_block_injection is not None:
                    block_injection = _injections_for_block(sparse_block_injection, i)
                if block_injection is not None:
                    x = block.forward_with_injection(
                        x,
                        mod[0],
                        cond,
                        injection=block_injection,
                        branch=sparse_block_injection_branch or "pos",
                        rope_phases=rope_phases,
                        cross_kv_cache=block_kv,
                    )
                else:
                    x = block(x, mod[0], cond, rope_phases=rope_phases,
                              cross_kv_cache=block_kv)
                if (i + 1) % 6 == 0:
                    mx.eval(x)

        # Output projection
        # PyTorch F.layer_norm defaults to eps=1e-5 in the TRELLIS.2 source.
        x = _sparse_flow_terminal_layernorm(x, eps=1e-5)
        x = _sparse_terminal_linear(
            x,
            self.out_layer,
            backend=self.terminal_linear_backend,
        )  # [B*R³, out_C]

        # Reshape back to 3D
        x = x.reshape(B, R, R, R, self.out_channels)
        x = x.transpose(0, 4, 1, 2, 3)                # [B, out_C, R, R, R]

        return x
