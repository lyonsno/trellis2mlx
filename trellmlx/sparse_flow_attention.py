"""Model-specific attention routing for the dense sparse-structure flow."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from .modules.attention import (
    scaled_dot_product_attention as inherited_scaled_dot_product_attention,
    scaled_dot_product_attention_for_backend,
)


DEFAULT_BACKEND = "inherit"
SOURCE_CUDA_MATH_TURING_T4_BACKEND = "source-cuda-math-turing-t4"
SUPPORTED_BACKENDS = (
    DEFAULT_BACKEND,
    SOURCE_CUDA_MATH_TURING_T4_BACKEND,
)

_backend = DEFAULT_BACKEND


def configure_sparse_flow_attention_backend(name: str) -> None:
    global _backend
    if name not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"unsupported sparse-flow attention backend {name!r}; "
            f"expected one of {SUPPORTED_BACKENDS}"
        )
    _backend = name


def get_sparse_flow_attention_backend() -> str:
    return _backend


def sparse_flow_attention_backend_identity() -> dict[str, Any]:
    if _backend == DEFAULT_BACKEND:
        return {
            "backend": _backend,
            "scope": "sparse-structure-flow",
            "algorithm": "inherited-global-attention-route",
            "experimental": False,
        }
    return {
        "backend": _backend,
        "scope": "sparse-structure-flow",
        "algorithm": "pytorch-math-sdpa-on-metal",
        "source_runtime": "torch-2.10.0+cu128",
        "source_device": "Tesla T4",
        "source_backend": "aten::_scaled_dot_product_attention_math",
        "self_attention_width": 4096,
        "cross_attention_width": 1029,
        "score_compute_dtype": "float32",
        "split_sqrt_qk_scaling": True,
        "softmax_threads": 1024,
        "softmax_registers_per_thread": 4,
        "value_projection": "forward-fp32-fma",
        "experimental": True,
    }


def scaled_dot_product_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    mask: mx.array = None,
) -> mx.array:
    if _backend == DEFAULT_BACKEND:
        return inherited_scaled_dot_product_attention(q, k, v, mask)
    return scaled_dot_product_attention_for_backend(
        q,
        k,
        v,
        mask,
        backend="source-cuda-self",
        softmax_backend="source-cuda-turing",
        value_backend="source-cuda-sequential",
    )
