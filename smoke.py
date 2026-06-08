"""Smoke test: generate a 3D occupancy grid from an image using MLX.

Usage:
    python smoke.py [--image path/to/image.png]

Without --image, uses random conditioning (won't look like anything real,
but proves the sampling loop works).

Outputs:
    /tmp/trellis-mlx-smoke.glb — voxel mesh of the occupancy grid
"""

import argparse
import os
import time

import mlx.core as mx
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Input image (uses native MLX DINOv3)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--output", default="/tmp/trellis-mlx-smoke.glb")
    args = parser.parse_args()

    mx.random.seed(args.seed)

    # Load models
    from trellmlx.models.sparse_structure_flow import SparseStructureFlowModel
    from trellmlx.models.sparse_structure_decoder import SparseStructureDecoder
    from trellmlx.weight_loader import load_weights

    print("Loading SparseStructureFlowModel (1.29B params)...", flush=True)
    flow_model = SparseStructureFlowModel()
    flow_ckpt = os.path.expanduser(
        "~/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/"
        "snapshots/af44b45f2e35a493886929c6d786e563ec68364d/"
        "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors"
    )
    load_weights(flow_model, flow_ckpt, verbose=False)
    print(f"  Loaded. Resolution: {flow_model.resolution}", flush=True)

    print("Loading SparseStructureDecoder (73.7M params)...", flush=True)
    decoder = SparseStructureDecoder()
    dec_ckpt = os.path.expanduser(
        "~/.cache/huggingface/hub/models--microsoft--TRELLIS-image-large/"
        "snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/"
        "ckpts/ss_dec_conv3d_16l8_fp16.safetensors"
    )
    load_weights(decoder, dec_ckpt, verbose=False)
    print("  Loaded.", flush=True)

    # Get image conditioning
    if args.image:
        print(f"Extracting image features from {args.image}...", flush=True)
        cond = _extract_image_features(args.image)
    else:
        print("No image — using random conditioning (structure will be random)", flush=True)
        cond = mx.random.normal((1, 10, 1024))

    neg_cond = mx.zeros_like(cond)

    # Sample sparse structure latent
    R = flow_model.resolution
    in_channels = flow_model.in_channels
    noise = mx.random.normal((1, in_channels, R, R, R))

    print(f"Sampling latent ({args.steps} steps, {R}³ grid)...", flush=True)
    from trellmlx.samplers import flow_euler_sample

    t0 = time.perf_counter()
    z_s = flow_euler_sample(
        flow_model, noise, cond, neg_cond,
        steps=args.steps,
        guidance_strength=7.5,
        guidance_rescale=0.7,
        guidance_interval=(0.6, 1.0),
        rescale_t=5.0,
    )
    mx.eval(z_s)
    sample_time = time.perf_counter() - t0
    print(f"  Sampled in {sample_time:.1f}s", flush=True)

    # Free flow model to save memory before decode
    import gc
    del flow_model
    gc.collect()

    # Decode latent → 64³ occupancy
    print("Decoding to occupancy grid...", flush=True)
    t0 = time.perf_counter()
    logits = decoder(z_s.astype(mx.float32))  # float32 to avoid GroupNorm overflow
    mx.eval(logits)
    decode_time = time.perf_counter() - t0
    print(f"  Decoded in {decode_time:.1f}s", flush=True)

    occupancy = logits[0, 0] > 0  # [64, 64, 64] boolean
    mx.eval(occupancy)
    n_occupied = occupancy.sum().item()
    total = 64 * 64 * 64
    print(f"  Occupancy: {n_occupied}/{total} voxels ({n_occupied/total*100:.1f}%)", flush=True)

    # Export as voxel mesh
    print("Exporting voxel mesh...", flush=True)
    _export_voxel_mesh(np.array(occupancy), args.output)
    print(f"  Saved: {args.output}", flush=True)

    # Release Metal resources
    from trellmlx.cleanup import cleanup
    cleanup()


def _extract_image_features(image_path: str, resolution: int = 512) -> mx.array:
    """Extract DINOv3 image features, preferring native MLX path."""
    try:
        from trellmlx.models.dinov3 import extract_features
        features = extract_features(image_path, image_size=resolution)
        print(f"  Features: {features.shape} (MLX)", flush=True)
        return features
    except Exception as e:
        print(f"  MLX DINOv3 failed ({e}), trying PyTorch...", flush=True)

    try:
        import torch
        import sys
        sys.path.insert(0, os.path.expanduser("~/dev/trellis-mac/TRELLIS.2"))
        from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor
        from PIL import Image

        extractor = DinoV3FeatureExtractor(
            "facebook/dinov3-vitl16-pretrain-lvd1689m",
            image_size=resolution,
        )
        extractor.to("cpu")

        img = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            features = extractor([img])  # [1, N, 1024]

        print(f"  Features: shape={features.shape}", flush=True)
        return mx.array(features.numpy())
    except Exception as e:
        raise RuntimeError(
            f"Image feature extraction failed for {image_path!r}. "
            "Download the native DINOv3 weights with "
            "`hf download facebook/dinov3-vitl16-pretrain-lvd1689m`, "
            "or omit --image for random conditioning."
        ) from e


def _export_voxel_mesh(occupancy: np.ndarray, output_path: str):
    """Export a boolean 3D grid as a mesh using trimesh VoxelGrid."""
    import trimesh

    R = occupancy.shape[0]
    coords = np.argwhere(occupancy)  # [N, 3]

    if len(coords) == 0:
        print("  WARNING: No occupied voxels!", flush=True)
        trimesh.Trimesh().export(output_path)
        return

    print(f"  {len(coords)} occupied voxels in {R}³ grid", flush=True)

    # Use trimesh's VoxelGrid for efficient mesh construction
    voxel_size = 1.0 / R
    voxel_grid = trimesh.voxel.VoxelGrid(
        trimesh.voxel.encoding.DenseEncoding(occupancy)
    )
    # Scale and center
    mesh = voxel_grid.marching_cubes
    mesh.apply_scale(voxel_size)
    mesh.apply_translation([-0.5, -0.5, -0.5])
    mesh.export(output_path)


if __name__ == "__main__":
    main()
