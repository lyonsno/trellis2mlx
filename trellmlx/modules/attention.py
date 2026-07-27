"""Attention modules for TRELLIS.2 MLX port.

Handles both dense attention (for SparseStructureFlowModel on dense grid)
and variable-length attention (for SLatFlowModel on sparse tokens).
"""

import math
import os

import mlx.core as mx
import mlx.nn as nn

from trellmlx.source_cuda_ex2 import SOURCE_CUDA_EX2_METAL_HEADER


DEFAULT_QK_NORM_BACKEND = "source-cuda-warp32"
MLX_QK_NORM_BACKEND = "mlx-sum"
LIBRARY_DEFAULT_QK_NORM_BACKEND = MLX_QK_NORM_BACKEND
SUPPORTED_QK_NORM_BACKENDS = (
    DEFAULT_QK_NORM_BACKEND,
    MLX_QK_NORM_BACKEND,
)

_source_cuda_warp32_norm_kernel = None
_source_cuda_sequential_value_kernel = None
_source_cuda_long_row_softmax_kernel = None
_source_cuda_warp_softmax_kernel = None


def get_qk_norm_backend() -> str:
    backend = os.environ.get(
        "TRELLIS2MLX_QK_NORM_BACKEND", LIBRARY_DEFAULT_QK_NORM_BACKEND
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
    value_backend = os.environ.get(
        "TRELLIS2MLX_ATTENTION_VALUE_BACKEND",
        "mlx-matmul",
    ).lower()
    softmax_backend = os.environ.get(
        "TRELLIS2MLX_ATTENTION_SOFTMAX_BACKEND",
        "mlx-softmax",
    ).lower()
    if softmax_backend not in {"mlx-softmax", "source-cuda-turing"}:
        raise ValueError(
            "TRELLIS2MLX_ATTENTION_SOFTMAX_BACKEND must be one of "
            "'mlx-softmax' or 'source-cuda-turing', "
            f"got {softmax_backend!r}"
        )
    if value_backend not in {"mlx-matmul", "source-cuda-sequential"}:
        raise ValueError(
            "TRELLIS2MLX_ATTENTION_VALUE_BACKEND must be one of "
            "'mlx-matmul' or 'source-cuda-sequential', "
            f"got {value_backend!r}"
        )

    scaling_factor = math.sqrt(scale)
    q32 = q.astype(mx.float32)
    k32 = k.astype(mx.float32) * scaling_factor
    v32 = v.astype(mx.float32)
    k_transposed = k32.transpose(0, 1, 3, 2)
    for start in range(0, q.shape[2], chunk_size):
        stop = min(start + chunk_size, q.shape[2])
        q_chunk = q32[:, :, start:stop, :] * scaling_factor
        scores = q_chunk @ k_transposed
        if mask is not None:
            scores = scores + mask[:, :, start:stop, :].astype(mx.float32)
        if softmax_backend == "source-cuda-turing":
            probs = _source_cuda_long_row_softmax(scores)
        else:
            probs = mx.softmax(scores, axis=-1)
        if value_backend == "source-cuda-sequential":
            out = _source_cuda_sequential_value_projection(probs, v32)
        else:
            out = probs @ v32
        out_chunks.append(out.astype(q.dtype))
    return mx.concatenate(out_chunks, axis=2)


def _source_cuda_sequential_value_projection(
    probs: mx.array,
    values: mx.array,
) -> mx.array:
    """Project values with source CUDA's observed left-to-right FP32 sum."""
    global _source_cuda_sequential_value_kernel

    if probs.ndim != 4 or values.ndim != 4:
        raise ValueError("probabilities and values must be rank-4 arrays")
    if probs.dtype != mx.float32 or values.dtype != mx.float32:
        raise ValueError("probabilities and values must use float32")
    if probs.shape[:2] != values.shape[:2]:
        raise ValueError("probabilities and values must share batch and head axes")
    if probs.shape[-1] != values.shape[-2]:
        raise ValueError("probability width must match the value token axis")
    if probs.shape[-2] <= 0:
        raise ValueError("probability query axis must be positive")
    if probs.shape[-1] <= 0:
        raise ValueError("probability source token axis must be positive")
    if values.shape[-1] <= 0 or values.shape[-1] > 1024:
        raise ValueError("value head dimension must be in [1, 1024]")

    if _source_cuda_sequential_value_kernel is None:
        source = r"""
            uint component = thread_position_in_threadgroup.x;
            uint row = threadgroup_position_in_grid.y;
            uint query_count_value = query_count[0];
            uint source_width_value = source_width[0];
            uint head_dim_value = head_dim[0];
            uint value_matrix = row / query_count_value;
            uint probability_offset = row * source_width_value;
            uint value_offset =
                value_matrix * source_width_value * head_dim_value + component;
            float accumulator = 0.0f;

            for (uint token = 0; token < source_width_value; ++token) {
                accumulator = metal::fma(
                    probs[probability_offset + token],
                    values[value_offset + token * head_dim_value],
                    accumulator);
            }
            out[row * head_dim_value + component] = accumulator;
        """
        _source_cuda_sequential_value_kernel = mx.fast.metal_kernel(
            name="attention_source_cuda_sequential_value_fp32",
            input_names=[
                "probs",
                "values",
                "query_count",
                "source_width",
                "head_dim",
            ],
            output_names=["out"],
            source=source,
        )

    query_count = int(probs.shape[-2])
    source_width = int(probs.shape[-1])
    head_dim = int(values.shape[-1])
    row_count = probs.size // source_width
    output_shape = (*probs.shape[:-1], head_dim)
    return _source_cuda_sequential_value_kernel(
        inputs=[
            probs,
            values,
            mx.array([query_count], dtype=mx.uint32),
            mx.array([source_width], dtype=mx.uint32),
            mx.array([head_dim], dtype=mx.uint32),
        ],
        template=[],
        grid=(head_dim, row_count, 1),
        threadgroup=(head_dim, 1, 1),
        output_shapes=[output_shape],
        output_dtypes=[mx.float32],
    )[0]


def _source_cuda_long_row_softmax(scores: mx.array) -> mx.array:
    """Reproduce authenticated source CUDA FP32 softmax schedules."""
    global _source_cuda_long_row_softmax_kernel
    global _source_cuda_warp_softmax_kernel

    if scores.ndim < 1:
        raise ValueError("softmax scores must have at least one axis")
    if scores.dtype != mx.float32:
        raise ValueError("softmax scores must use float32")
    width = int(scores.shape[-1])
    if width not in {1029, 7697}:
        raise ValueError(
            "source CUDA softmax requires an authenticated width of 1029 or 7697"
        )
    row_count = scores.size // scores.shape[-1]
    if row_count <= 0:
        raise ValueError("softmax row count must be positive")

    if width == 1029:
        if _source_cuda_warp_softmax_kernel is None:
            source = r"""
                constexpr uint width = 1029;
                constexpr uint warp_size = 32;
                constexpr uint warp_iterations = 64;
                constexpr uint warps_per_threadgroup = 4;
                constexpr float lowest = -3.402823466e+38f;

                uint tid = thread_position_in_threadgroup.x;
                uint lane = tid & 31;
                uint warp = tid >> 5;
                uint row =
                    threadgroup_position_in_grid.y * warps_per_threadgroup + warp;
                uint row_count_value = row_count[0];
                if (row >= row_count_value) {
                    return;
                }
                uint row_offset = row * width;
                float elements[warp_iterations];

                for (uint iteration = 0; iteration < warp_iterations; ++iteration) {
                    uint offset = lane + iteration * warp_size;
                    elements[iteration] =
                        offset < width ? scores[row_offset + offset] : lowest;
                }

                float row_max = elements[0];
                for (uint iteration = 0; iteration < warp_iterations; ++iteration) {
                    float value = elements[iteration];
                    row_max = row_max > value ? row_max : value;
                }
                for (ushort delta = 16; delta > 0; delta >>= 1) {
                    float peer = simd_shuffle_xor(row_max, delta);
                    row_max = row_max < peer ? peer : row_max;
                }

                float row_sum = 0.0f;
                for (uint iteration = 0; iteration < warp_iterations; ++iteration) {
                    elements[iteration] = source_cuda_expf(
                        elements[iteration] - row_max);
                    row_sum = row_sum + elements[iteration];
                }
                for (ushort delta = 16; delta > 0; delta >>= 1) {
                    row_sum = row_sum + simd_shuffle_xor(row_sum, delta);
                }

                for (uint iteration = 0; iteration < warp_iterations; ++iteration) {
                    uint offset = lane + iteration * warp_size;
                    if (offset < width) {
                        out[row_offset + offset] =
                            elements[iteration] / row_sum;
                    }
                }
            """
            _source_cuda_warp_softmax_kernel = mx.fast.metal_kernel(
                name="attention_source_cuda_warp_softmax_fp32_1029",
                input_names=["scores", "row_count"],
                output_names=["out"],
                header=SOURCE_CUDA_EX2_METAL_HEADER,
                source=source,
            )
        return _source_cuda_warp_softmax_kernel(
            inputs=[
                scores,
                mx.array([row_count], dtype=mx.uint32),
            ],
            template=[],
            grid=(128, math.ceil(row_count / 4), 1),
            threadgroup=(128, 1, 1),
            output_shapes=[scores.shape],
            output_dtypes=[mx.float32],
        )[0]

    if _source_cuda_long_row_softmax_kernel is None:
        source = r"""
            constexpr uint width = 7697;
            constexpr uint threads = 1024;
            constexpr uint warps = 32;
            constexpr uint registers_per_thread = 8;
            constexpr float lowest = -3.402823466e+38f;

            uint tid = thread_position_in_threadgroup.x;
            uint lane = tid & 31;
            uint warp = tid >> 5;
            uint row = threadgroup_position_in_grid.y;
            uint row_offset = row * width;
            threadgroup float shared[warps];
            float registers[registers_per_thread];
            float thread_max = lowest;

            for (uint reg = 0; reg < registers_per_thread; ++reg) {
                uint offset = tid + reg * threads;
                if (offset < width) {
                    float value = scores[row_offset + offset];
                    registers[reg] = value;
                    thread_max = thread_max < value ? value : thread_max;
                }
            }

            for (ushort delta = 16; delta > 0; delta >>= 1) {
                float upper = simd_shuffle_down(thread_max, delta);
                thread_max = thread_max < upper ? upper : thread_max;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) {
                shared[warp] = thread_max;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float row_max = tid < warps ? shared[lane] : lowest;
            if (warp == 0) {
                for (ushort delta = 16; delta > 0; delta >>= 1) {
                    float upper = simd_shuffle_down(row_max, delta);
                    row_max = row_max < upper ? upper : row_max;
                }
            }
            if (tid == 0) {
                shared[0] = row_max;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            row_max = shared[0];

            float thread_sum = 0.0f;
            for (uint reg = 0; reg < registers_per_thread; ++reg) {
                uint offset = tid + reg * threads;
                if (offset < width) {
                    registers[reg] = source_cuda_expf(
                        registers[reg] - row_max);
                    thread_sum = thread_sum + registers[reg];
                }
            }

            for (ushort delta = 16; delta > 0; delta >>= 1) {
                thread_sum += simd_shuffle_down(thread_sum, delta);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) {
                shared[warp] = thread_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float row_sum = tid < warps ? shared[lane] : 0.0f;
            if (warp == 0) {
                for (ushort delta = 16; delta > 0; delta >>= 1) {
                    row_sum += simd_shuffle_down(row_sum, delta);
                }
            }
            if (tid == 0) {
                shared[0] = row_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            row_sum = shared[0];

            for (uint reg = 0; reg < registers_per_thread; ++reg) {
                uint offset = tid + reg * threads;
                if (offset < width) {
                    out[row_offset + offset] = registers[reg] / row_sum;
                }
            }
        """
        _source_cuda_long_row_softmax_kernel = mx.fast.metal_kernel(
            name="attention_source_cuda_long_row_softmax_fp32_7697",
            input_names=["scores"],
            output_names=["out"],
            header=SOURCE_CUDA_EX2_METAL_HEADER,
            source=source,
        )

    return _source_cuda_long_row_softmax_kernel(
        inputs=[scores],
        template=[],
        grid=(1024, row_count, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[scores.shape],
        output_dtypes=[mx.float32],
    )[0]


def scaled_dot_product_attention(
    q: mx.array,  # [B, H, T, D]
    k: mx.array,  # [B, H, S, D]
    v: mx.array,  # [B, H, S, D]
    mask: mx.array = None,  # [B, 1, T, S] or None
) -> mx.array:
    """Scaled dot-product attention using the selected diagnostic backend.

    MLX's mx.fast.scaled_dot_product_attention uses tiled online softmax
    and never materializes the full N×N attention matrix (O(N) memory).
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    backend = os.environ.get("TRELLIS2MLX_ATTENTION_BACKEND", "fast").lower()
    if backend in {"fast", "mlx-fast"}:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    if backend in {"manual", "mlx-manual"}:
        return _manual_scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    if backend == "source-cuda-self":
        softmax_backend = os.environ.get(
            "TRELLIS2MLX_ATTENTION_SOFTMAX_BACKEND",
            "mlx-softmax",
        ).lower()
        value_backend = os.environ.get(
            "TRELLIS2MLX_ATTENTION_VALUE_BACKEND",
            "mlx-matmul",
        ).lower()
        if (
            softmax_backend != "source-cuda-turing"
            or value_backend != "source-cuda-sequential"
        ):
            raise ValueError(
                "source-cuda-self requires source-cuda-turing softmax and "
                "source-cuda-sequential value projection"
            )
        if k.shape[-2] in {1029, 7697}:
            return _manual_scaled_dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                mask=mask,
            )
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    raise ValueError(
        "TRELLIS2MLX_ATTENTION_BACKEND must be one of "
        "'fast', 'manual', or 'source-cuda-self', "
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
        if backend == DEFAULT_QK_NORM_BACKEND:
            if orig_dtype != mx.bfloat16 or x.shape[-1] != 128:
                raise ValueError(
                    f"{DEFAULT_QK_NORM_BACKEND} requires bfloat16 input with "
                    f"head dimension 128, got dtype {orig_dtype} and "
                    f"head dimension {x.shape[-1]}"
                )
            norm = mx.maximum(_source_cuda_warp32_l2_norm(x), 1e-12)
        else:
            norm = mx.sqrt(
                mx.sum(xf * xf, axis=-1, keepdims=True) + 1e-12
            )
        return ((xf / norm) * self.gamma * self.scale).astype(orig_dtype)
