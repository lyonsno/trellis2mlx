"""Load TRELLIS.2 safetensors weights into trellmlx models.

Handles key remapping between PyTorch checkpoint naming and MLX module tree,
plus weight transposition for Linear layers (PyTorch [out, in] → MLX [in, out]).
"""

import re

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np
from safetensors import safe_open


# Key remapping: checkpoint key → MLX parameter tree key
_REMAP_PATTERNS = [
    # MLX nn.Sequential inserts .layers. prefix
    (r"^adaLN_modulation\.(\d+)\.", r"adaLN_modulation.layers.\1."),
    # FeedForward uses mlp_0/mlp_2 attrs instead of Sequential .0/.2
    (r"\.mlp\.mlp\.(\d+)\.", r".mlp.mlp_\1."),
    # Decoder ConvNeXt blocks: blocks.X.Y.mlp.0 → blocks.X.Y.mlp_0
    (r"(blocks\.\d+\.\d+)\.mlp\.(\d+)\.", r"\1.mlp_\2."),
    # TimestepEmbedder same pattern
    (r"^t_embedder\.mlp\.(\d+)\.", r"t_embedder.mlp_\1."),
    # Decoder out_layer Sequential: out_layer.0 → out_layer_0, out_layer.2 → out_layer_2
    (r"^out_layer\.(\d+)\.", r"out_layer_\1."),
    # Middle block Sequential: middle_block.N → middle_block.N (list indexing, should work as-is)
]


def _remap_key(key: str) -> str:
    """Remap a checkpoint key to match the MLX parameter tree."""
    for pattern, replacement in _REMAP_PATTERNS:
        key = re.sub(pattern, replacement, key)
    return key


def _should_transpose(key: str, shape: tuple) -> bool:
    """No 2D transposition needed — MLX Linear matches PyTorch convention."""
    return False


def _should_permute_conv(key: str, shape: tuple) -> bool:
    """Check if a 5D weight needs permutation for MLX conv_general.

    PyTorch dense Conv3d: [out_C, in_C, kD, kH, kW] → permute to [out_C, kD, kH, kW, in_C]
    Sparse conv (flex_gemm): [out_C, kD, kH, kW, in_C] → already correct, don't permute

    Heuristic: if shape is [C, 3, 3, 3, C'] with dims 1-3 all equal (kernel),
    it's already in flex_gemm layout. If shape is [C, C', 3, 3, 3] with dims 2-4
    all equal, it's PyTorch layout and needs permutation.
    """
    if len(shape) != 5 or not key.endswith(".weight"):
        return False
    # Check if dims 2,3,4 are all equal (PyTorch layout: [out, in, K, K, K])
    if shape[2] == shape[3] == shape[4] and shape[2] <= 7:
        return True
    # Check if dims 1,2,3 are all equal (flex_gemm layout: [out, K, K, K, in])
    if shape[1] == shape[2] == shape[3] and shape[1] <= 7:
        return False  # already correct
    return False


def load_weights(model: nn.Module, checkpoint_path: str, verbose: bool = True):
    """Load TRELLIS.2 safetensors weights into an MLX model.

    Args:
        model: The MLX model to load weights into.
        checkpoint_path: Path to a .safetensors file.
        verbose: Print loading progress.

    Returns:
        List of checkpoint keys that were not loaded (missing in model).
    """
    # Get the model's parameter tree
    model_params = dict(mlx.utils.tree_flatten(model.parameters()))
    model_keys = set(model_params)

    weights = {}
    unloaded = []

    # Try numpy first, fall back to MLX direct loading for bf16.
    # safetensors numpy can't handle bf16 — the error occurs in get_tensor,
    # not safe_open, so we probe by trying to load the first tensor.
    _framework = "numpy"
    try:
        with safe_open(checkpoint_path, framework="numpy") as sf:
            first_key = list(sf.keys())[0]
            _ = sf.get_tensor(first_key)
    except TypeError:
        _framework = "mlx"

    if _framework == "mlx":
        # bf16 checkpoint — load directly via MLX's safetensors support
        raw_weights = mx.load(checkpoint_path)
        for ckpt_key, tensor in raw_weights.items():
            mlx_key = _remap_key(ckpt_key)
            if mlx_key not in model_keys:
                unloaded.append(ckpt_key)
                if verbose:
                    print(f"  SKIP {ckpt_key} -> {mlx_key} (not in model)")
                continue

            # Transpose linear weights
            if _should_transpose(mlx_key, tuple(tensor.shape)):
                tensor = tensor.T

            # Permute Conv3d weights: PyTorch [out, in, D, H, W] → MLX [out, D, H, W, in]
            if _should_permute_conv(mlx_key, tuple(tensor.shape)):
                tensor = tensor.transpose(0, 2, 3, 4, 1)

            tensor = _cast_to_destination_dtype(tensor, model_params[mlx_key])
            weights[mlx_key] = tensor
    else:
        with safe_open(checkpoint_path, framework="numpy") as sf:
            for ckpt_key in sf.keys():
                mlx_key = _remap_key(ckpt_key)
                tensor = sf.get_tensor(ckpt_key)

                if mlx_key not in model_keys:
                    unloaded.append(ckpt_key)
                    if verbose:
                        print(f"  SKIP {ckpt_key} -> {mlx_key} (not in model)")
                    continue

                if _should_transpose(mlx_key, tensor.shape):
                    tensor = tensor.T

                # Permute Conv3d: PyTorch [out, in, D, H, W] → MLX [out, D, H, W, in]
                if _should_permute_conv(mlx_key, tensor.shape):
                    tensor = tensor.transpose(0, 2, 3, 4, 1)

                tensor = mx.array(tensor)
                tensor = _cast_to_destination_dtype(tensor, model_params[mlx_key])
                weights[mlx_key] = tensor

    # Check for model keys not found in checkpoint
    loaded_keys = set(weights.keys())
    missing = model_keys - loaded_keys
    if missing and verbose:
        print(f"  {len(missing)} model parameters not in checkpoint:")
        for k in sorted(missing):
            print(f"    MISSING {k}")

    # Load into model
    model.load_weights(list(weights.items()))

    if verbose:
        print(f"  Loaded {len(weights)}/{len(model_keys)} parameters")
        if unloaded:
            print(f"  Skipped {len(unloaded)} checkpoint keys")

    return unloaded


def _cast_to_destination_dtype(tensor: mx.array, destination: mx.array) -> mx.array:
    if hasattr(destination, "dtype") and hasattr(tensor, "astype") and tensor.dtype != destination.dtype:
        return tensor.astype(destination.dtype)
    return tensor
