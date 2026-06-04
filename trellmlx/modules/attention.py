"""Attention modules for TRELLIS.2 MLX port.

Handles both dense attention (for SparseStructureFlowModel on dense grid)
and variable-length attention (for SLatFlowModel on sparse tokens).
"""

import math

import mlx.core as mx
import mlx.nn as nn


def scaled_dot_product_attention(
    q: mx.array,  # [B, H, T, D]
    k: mx.array,  # [B, H, S, D]
    v: mx.array,  # [B, H, S, D]
    mask: mx.array = None,  # [B, 1, T, S] or None
) -> mx.array:
    """Standard scaled dot-product attention."""
    scale = math.sqrt(q.shape[-1])
    scores = (q @ k.transpose(0, 1, 3, 2)) / scale
    if mask is not None:
        scores = scores + mask
    weights = mx.softmax(scores, axis=-1)
    return weights @ v


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


class RMSNorm(nn.Module):
    """Per-head RMS normalization for QK norm."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = mx.ones((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return self.gamma * (x / rms)
