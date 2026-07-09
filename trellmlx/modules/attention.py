"""Attention modules for TRELLIS.2 MLX port.

Handles both dense attention (for SparseStructureFlowModel on dense grid)
and variable-length attention (for SLatFlowModel on sparse tokens).
"""

import math
import os

import mlx.core as mx
import mlx.nn as nn


def _manual_scaled_dot_product_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    mask: mx.array = None,
) -> mx.array:
    """Chunked MLX matmul/softmax attention for source-parity diagnostics."""
    out_chunks = []
    chunk_size = int(os.environ.get("TRELLIS2MLX_ATTENTION_CHUNK_SIZE", "512"))
    if chunk_size <= 0:
        raise ValueError("TRELLIS2MLX_ATTENTION_CHUNK_SIZE must be positive")

    for start in range(0, q.shape[2], chunk_size):
        stop = min(start + chunk_size, q.shape[2])
        q_chunk = q[:, :, start:stop, :]
        scores = (q_chunk.astype(mx.float32) @ k.astype(mx.float32).transpose(0, 1, 3, 2)) * scale
        if mask is not None:
            scores = scores + mask[:, :, start:stop, :].astype(mx.float32)
        probs = mx.softmax(scores, axis=-1)
        out = probs @ v.astype(mx.float32)
        out_chunks.append(out.astype(q.dtype))
    return mx.concatenate(out_chunks, axis=2)


def scaled_dot_product_attention(
    q: mx.array,  # [B, H, T, D]
    k: mx.array,  # [B, H, S, D]
    v: mx.array,  # [B, H, S, D]
    mask: mx.array = None,  # [B, 1, T, S] or None
) -> mx.array:
    """Scaled dot-product attention using MLX's fused Flash Attention kernel.

    MLX's mx.fast.scaled_dot_product_attention uses tiled online softmax
    and never materializes the full N×N attention matrix (O(N) memory).
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    backend = os.environ.get("TRELLIS2MLX_ATTENTION_BACKEND", "fast").lower()
    if backend in {"fast", "mlx-fast"}:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    if backend in {"manual", "mlx-manual"}:
        return _manual_scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    raise ValueError(
        "TRELLIS2MLX_ATTENTION_BACKEND must be one of 'fast' or 'manual', "
        f"got {backend!r}"
    )


def varlen_attention_padded(
    q: mx.array,       # [T_total, H, D]
    k: mx.array,       # [S_total, H, D]
    v: mx.array,       # [S_total, H, D]
    q_seqlens: list,    # lengths per batch element for queries
    kv_seqlens: list,   # lengths per batch element for keys/values
) -> mx.array:
    """Variable-length attention via padding to max length.

    This is the same strategy as the MPS SDPA path in the PyTorch version.
    Pads variable-length sequences into a batch, runs attention with masks,
    then unpads the result.

    For a future optimization: replace with a proper segmented attention
    kernel that avoids the padding overhead.
    """
    B = len(q_seqlens)
    H = q.shape[1]
    D = q.shape[2]

    max_q = max(q_seqlens)
    max_kv = max(kv_seqlens)

    # Pad into [B, H, max_len, D] batches
    q_padded = mx.zeros((B, H, max_q, D), dtype=q.dtype)
    k_padded = mx.zeros((B, H, max_kv, D), dtype=k.dtype)
    v_padded = mx.zeros((B, H, max_kv, D), dtype=v.dtype)

    # Build attention mask: -inf for padding positions
    mask = mx.full((B, 1, max_q, max_kv), float("-inf"), dtype=q.dtype)

    q_offset = 0
    kv_offset = 0
    for i in range(B):
        ql = q_seqlens[i]
        kvl = kv_seqlens[i]
        q_padded[i, :, :ql, :] = q[q_offset : q_offset + ql].transpose(1, 0, 2)
        k_padded[i, :, :kvl, :] = k[kv_offset : kv_offset + kvl].transpose(1, 0, 2)
        v_padded[i, :, :kvl, :] = v[kv_offset : kv_offset + kvl].transpose(1, 0, 2)
        mask[i, :, :ql, :kvl] = 0.0
        q_offset += ql
        kv_offset += kvl

    # Run standard attention
    out_padded = scaled_dot_product_attention(q_padded, k_padded, v_padded, mask)

    # Unpad: gather valid tokens back into flat [T_total, H, D]
    results = []
    for i in range(B):
        ql = q_seqlens[i]
        results.append(out_padded[i, :, :ql, :].transpose(1, 0, 2))

    return mx.concatenate(results, axis=0)


class MultiHeadRMSNorm(nn.Module):
    """Per-head L2-normalize + learned scale for QK norm.

    Matches TRELLIS.2's MultiHeadRMSNorm:
        F.normalize(x, dim=-1) * gamma * sqrt(dim)
    gamma shape is [num_heads, head_dim].
    """

    def __init__(self, head_dim: int, num_heads: int):
        super().__init__()
        self.scale = head_dim ** 0.5
        self.gamma = mx.ones((num_heads, head_dim))

    def __call__(self, x: mx.array) -> mx.array:
        # x: [..., H, D] — L2 normalize along last dim, then scale
        orig_dtype = x.dtype
        x = x.astype(mx.float32)
        norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True) + 1e-12)
        x = (x / norm) * self.gamma * self.scale
        return x.astype(orig_dtype)
