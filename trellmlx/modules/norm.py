"""Normalization layers for TRELLIS.2 MLX port."""

import mlx.core as mx
import mlx.nn as nn


class LayerNorm32(nn.Module):
    """LayerNorm that accumulates in float32 for numerical stability.

    TRELLIS.2 uses elementwise_affine=False (controlled by adaLN-Zero),
    so this is just normalize + optional affine.
    """

    def __init__(self, dims: int, eps: float = 1e-6, affine: bool = False):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = mx.ones((dims,))
            self.bias = mx.zeros((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        orig_dtype = x.dtype
        x = x.astype(mx.float32)
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - mean) * mx.rsqrt(var + self.eps)
        if self.affine:
            x = x * self.weight + self.bias
        return x.astype(orig_dtype)
