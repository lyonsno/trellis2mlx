"""Convert TRELLIS.2 safetensors weights to MLX format.

Usage:
    python -m trellmlx.convert [--model microsoft/TRELLIS.2-4B] [--output weights/]

Downloads from HuggingFace (or reads cached weights) and writes MLX-compatible
.safetensors files with transposed linear weights (MLX convention: [out, in] -> [in, out]).
"""

import argparse
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file as np_save_file


# TRELLIS.2-4B checkpoint layout
CHECKPOINTS = {
    "ss_flow": "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
    "shape_flow_512": "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
    "shape_flow_1024": "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
    "tex_flow_512": "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors",
    "tex_flow_1024": "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
    "shape_dec": "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
    "tex_dec": "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
}


def _torch_dtype_to_numpy(tensor_np, torch_dtype_str: str):
    """Handle bfloat16 -> float16 conversion (numpy doesn't support bf16)."""
    if torch_dtype_str == "torch.bfloat16":
        # safetensors already converts bf16 to f32 when loading to numpy
        return tensor_np.astype(np.float16)
    elif torch_dtype_str == "torch.float16":
        return tensor_np
    elif torch_dtype_str == "torch.float32":
        return tensor_np
    return tensor_np


def _should_transpose(key: str, shape: tuple) -> bool:
    """Determine if a weight should be transposed for MLX convention.

    PyTorch Linear: weight is [out_features, in_features]
    MLX Linear: weight is [in_features, out_features] (transposed)

    We transpose 2D weight matrices that belong to linear layers.
    Conv weights (5D: [out, kd, kh, kw, in]) are NOT transposed.
    """
    if len(shape) != 2:
        return False
    # These are linear weights — transpose them
    if any(suffix in key for suffix in [".weight"]):
        # But not embedding weights, norm weights, or 1D params
        if any(skip in key for skip in ["norm", "gamma", "embedding", "pos_emb"]):
            return False
        return True
    return False


def convert_checkpoint(src_path: str, dst_path: str, verbose: bool = True):
    """Convert a single safetensors checkpoint to MLX format."""
    weights = {}
    with safe_open(src_path, framework="numpy") as sf:
        metadata = sf.metadata()
        for key in sf.keys():
            tensor = sf.get_tensor(key)

            if _should_transpose(key, tensor.shape):
                tensor = tensor.T
                if verbose:
                    print(f"  T {key}: {tensor.T.shape} -> {tensor.shape}")

            # Convert bf16 -> f16 (MLX supports f16 natively)
            if tensor.dtype == np.float32:
                # Check if original was bf16 (safetensors loads bf16 as f32)
                tensor = tensor.astype(np.float16)

            weights[key] = tensor

    np_save_file(weights, dst_path)
    size_mb = os.path.getsize(dst_path) / 1024 / 1024
    if verbose:
        print(f"  -> {dst_path} ({size_mb:.0f}MB, {len(weights)} tensors)")


def convert_all(model_id: str = "microsoft/TRELLIS.2-4B", output_dir: str = "weights"):
    """Convert all TRELLIS.2 checkpoints to MLX format."""
    from huggingface_hub import hf_hub_download

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, rel_path in CHECKPOINTS.items():
        print(f"\nConverting {name}...")
        src = hf_hub_download(model_id, rel_path)
        dst = str(out / f"{name}.safetensors")
        convert_checkpoint(src, dst)

    # Also download and save the pipeline config
    config_path = hf_hub_download(model_id, "pipeline.json")
    import shutil
    shutil.copy(config_path, str(out / "pipeline.json"))
    print(f"\nSaved pipeline.json")
    print(f"\nAll conversions complete. Output: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Convert TRELLIS.2 weights to MLX format")
    parser.add_argument("--model", default="microsoft/TRELLIS.2-4B",
                        help="HuggingFace model ID")
    parser.add_argument("--output", default="weights",
                        help="Output directory for converted weights")
    args = parser.parse_args()
    convert_all(args.model, args.output)


if __name__ == "__main__":
    main()
