"""Normalization layers for TRELLIS.2 MLX port."""

import mlx.core as mx
import mlx.nn as nn


class LayerNorm32(nn.Module):
    """LayerNorm that accumulates in float32 for numerical stability.

    TRELLIS.2 uses elementwise_affine=False (controlled by adaLN-Zero),
    so this is just normalize + optional affine.
    """

    def __init__(
        self,
        dims: int,
        eps: float = 1e-6,
        affine: bool = False,
        *,
        sparse_flow_layernorm: bool = False,
        shape_flow_layernorm: bool = False,
        decoder_layernorm: bool = False,
    ):
        super().__init__()
        if sum((sparse_flow_layernorm, shape_flow_layernorm, decoder_layernorm)) > 1:
            raise ValueError(
                "LayerNorm32 cannot use sparse-flow, shape-flow, and decoder "
                "routes together"
            )
        self.eps = eps
        self.affine = affine
        self.sparse_flow_layernorm = sparse_flow_layernorm
        self.shape_flow_layernorm = shape_flow_layernorm
        self.decoder_layernorm = decoder_layernorm
        if affine:
            self.weight = mx.ones((dims,))
            self.bias = mx.zeros((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        orig_dtype = x.dtype
        weight = self.weight if self.affine else None
        bias = self.bias if self.affine else None
        if self.sparse_flow_layernorm:
            from ..sparse_flow_layernorm import layernorm_affine, layernorm_noaffine

            if self.affine:
                return layernorm_affine(x, weight, bias, self.eps).astype(orig_dtype)
            return layernorm_noaffine(x, self.eps).astype(orig_dtype)
        if self.shape_flow_layernorm and self.affine:
            from ..shape_flow_layernorm import layernorm_affine

            return layernorm_affine(x, weight, bias, self.eps).astype(orig_dtype)
        if self.decoder_layernorm and self.affine:
            from ..decoder_turing_layernorm import layernorm_affine

            return layernorm_affine(x, weight, bias, self.eps).astype(orig_dtype)
        if self.decoder_layernorm:
            from ..decoder_turing_layernorm import layernorm_noaffine

            return layernorm_noaffine(x, self.eps).astype(orig_dtype)
        return mx.fast.layer_norm(x, weight, bias, self.eps).astype(orig_dtype)
