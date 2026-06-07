"""Quantize trellis2mlx models for faster inference.

MLX's INT4 quantization reduces weight memory ~6.4x (5.17GB → 0.81GB)
and proportionally reduces memory bandwidth, directly speeding up
inference on bandwidth-bound Apple Silicon.

Usage:
    from trellmlx.quantize import quantize_model
    quantize_model(model, bits=4)  # in-place quantization
"""

import mlx.nn as nn


def quantize_model(model: nn.Module, bits: int = 4, group_size: int = 64):
    """Quantize all sufficiently large Linear layers in-place.

    Skips layers where either dimension is smaller than group_size
    (e.g., input_layer [1536, 8] or out_layer [8, 1536]).

    Args:
        model: The model to quantize.
        bits: Quantization bit width (4 or 8).
        group_size: Quantization group size.
    """
    def should_quantize(path, module):
        if isinstance(module, nn.Linear):
            w = module.weight
            return all(d >= group_size for d in w.shape)
        return False

    nn.quantize(model, bits=bits, group_size=group_size, class_predicate=should_quantize)
