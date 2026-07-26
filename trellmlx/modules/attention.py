"""Attention modules for TRELLIS.2 MLX port.

Handles both dense attention (for SparseStructureFlowModel on dense grid)
and variable-length attention (for SLatFlowModel on sparse tokens).
"""

import math
import os

import mlx.core as mx
import mlx.nn as nn


DEFAULT_QK_NORM_BACKEND = "source-cuda-warp32"
MLX_QK_NORM_BACKEND = "mlx-sum"
SUPPORTED_QK_NORM_BACKENDS = (
    DEFAULT_QK_NORM_BACKEND,
    MLX_QK_NORM_BACKEND,
)

_source_cuda_warp32_norm_kernel = None


def get_qk_norm_backend() -> str:
    backend = os.environ.get(
        "TRELLIS2MLX_QK_NORM_BACKEND", DEFAULT_QK_NORM_BACKEND
    ).lower()
    if backend not in SUPPORTED_QK_NORM_BACKENDS:
        raise ValueError(
            "TRELLIS2MLX_QK_NORM_BACKEND must be one of "
            f"{SUPPORTED_QK_NORM_BACKENDS}, got {backend!r}"
        )
    return backend


def qk_norm_backend_identity() -> dict:
    backend = get_qk_norm_backend()
    if backend == MLX_QK_NORM_BACKEND:
        return {
            "backend": backend,
            "algorithm": "mlx-fp32-sum-sqrt",
            "experimental": False,
        }
    return {
        "backend": backend,
        "algorithm": "pytorch-vectorized-cuda-warp32-sum-on-metal",
        "experimental": True,
        "cuda_source_tag": "pytorch-v2.10.0",
        "cuda_source_kernel": "ATen-native-cuda-Reduce.cuh",
        "authenticated_contract": {
            "input_dtype": "bfloat16",
            "head_dim": 128,
            "accumulator_dtype": "float32",
            "warp_width": 32,
            "vector_width": 4,
            "shuffle_offsets": [16, 8, 4, 2, 1],
            "cuda_device_anchor": "Tesla T4",
        },
        "non_authenticated_geometry": "mlx-fp32-sum-sqrt",
    }


def _source_cuda_warp32_l2_norm(x: mx.array) -> mx.array:
    global _source_cuda_warp32_norm_kernel
    if _source_cuda_warp32_norm_kernel is None:
        source = r"""
            constexpr uint width = 128;
            constexpr uint vector_width = 4;

            uint lane = thread_position_in_threadgroup.x;
            uint row = threadgroup_position_in_grid.z;
            uint offset = row * width + lane * vector_width;

            float value0 = static_cast<float>(inp[offset]);
            float value1 = static_cast<float>(inp[offset + 1]);
            float value2 = static_cast<float>(inp[offset + 2]);
            float value3 = static_cast<float>(inp[offset + 3]);
            float sum = value0 * value0;
            sum += value1 * value1;
            sum += value2 * value2;
            sum += value3 * value3;

            for (ushort delta = 16; delta > 0; delta >>= 1) {
                sum += simd_shuffle_down(sum, delta);
            }
            if (lane == 0) {
                out[row] = metal::precise::sqrt(sum);
            }
        """
        _source_cuda_warp32_norm_kernel = mx.fast.metal_kernel(
            name="qk_norm_source_cuda_warp32_bf16_128",
            input_names=["inp"],
            output_names=["out"],
            source=source,
        )

    rows = x.size // 128
    norm_shape = (*x.shape[:-1], 1)
    return _source_cuda_warp32_norm_kernel(
        inputs=[x],
        template=[("T", mx.bfloat16)],
        grid=(32, 1, rows),
        threadgroup=(32, 1, 1),
        output_shapes=[norm_shape],
        output_dtypes=[mx.float32],
    )[0]


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
        xf = x.astype(mx.float32)
        backend = get_qk_norm_backend()
        if (
            backend == DEFAULT_QK_NORM_BACKEND
            and orig_dtype == mx.bfloat16
            and x.shape[-1] == 128
        ):
            norm = mx.maximum(_source_cuda_warp32_l2_norm(x), 1e-12)
        else:
            norm = mx.sqrt(
                mx.sum(xf * xf, axis=-1, keepdims=True) + 1e-12
            )
        return ((xf / norm) * self.gamma * self.scale).astype(orig_dtype)
