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
    parser.add_argument("--image", help="Input image (requires PyTorch for DINOv3)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--output", default="/tmp/trellis-mlx-smoke.glb")
    args = parser.parse_args()

    mx.random.seed(args.seed)

    # Load model
    print("Loading SparseStructureFlowModel...", flush=True)
    from trellmlx.models.sparse_structure_flow import SparseStructureFlowModel
    from trellmlx.weight_loader import load_weights

    model = SparseStructureFlowModel()
    ckpt = os.path.expanduser(
        "~/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/"
        "snapshots/af44b45f2e35a493886929c6d786e563ec68364d/"
        "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors"
    )
    load_weights(model, ckpt, verbose=False)
    print(f"  Loaded. Resolution: {model.resolution}", flush=True)

    # Get image conditioning
    if args.image:
        print(f"Extracting image features from {args.image}...", flush=True)
        cond = _extract_image_features(args.image)
    else:
        print("No image — using random conditioning (structure will be random)", flush=True)
        cond = mx.random.normal((1, 10, 1024))

    neg_cond = mx.zeros_like(cond)

    # Sample
    R = model.resolution
    in_channels = model.in_channels
    noise = mx.random.normal((1, in_channels, R, R, R))

    print(f"Sampling ({args.steps} steps, {R}³ grid)...", flush=True)
    from trellmlx.samplers import flow_euler_sample

    t0 = time.perf_counter()
    output = flow_euler_sample(
        model, noise, cond, neg_cond,
        steps=args.steps,
        guidance_strength=7.5,
        guidance_rescale=0.7,
        guidance_interval=(0.6, 1.0),
        rescale_t=5.0,
    )
    mx.eval(output)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s", flush=True)

    # Decode: threshold to get occupancy
    # The sparse structure decoder thresholds channel 0
    occupancy = output[0, 0] > 0  # [R, R, R] boolean
    mx.eval(occupancy)
    n_occupied = occupancy.sum().item()
    total = R * R * R
    print(f"  Occupancy: {n_occupied}/{total} voxels ({n_occupied/total*100:.1f}%)", flush=True)

    # Export as voxel mesh
    print("Exporting voxel mesh...", flush=True)
    _export_voxel_mesh(np.array(occupancy), args.output)
    print(f"  Saved: {args.output}", flush=True)


def _extract_image_features(image_path: str) -> mx.array:
    """Extract DINOv3 image features using PyTorch (temporary)."""
    try:
        import torch
        import sys
        sys.path.insert(0, os.path.expanduser("~/dev/trellis-mac/TRELLIS.2"))
        from trellis2.modules.image_feature_extractor import DINOv3ViTModel
        from transformers import AutoImageProcessor
        from PIL import Image

        processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov3-vitl16-pretrain-lvd1689m"
        )
        dino = DINOv3ViTModel.from_pretrained(
            "facebook/dinov3-vitl16-pretrain-lvd1689m"
        )
        dino.eval()

        img = Image.open(image_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            features = dino(**inputs).last_hidden_state  # [1, L, 1024]

        return mx.array(features.numpy())
    except ImportError:
        print("  PyTorch/transformers not available — using random features", flush=True)
        return mx.random.normal((1, 10, 1024))


def _export_voxel_mesh(occupancy: np.ndarray, output_path: str):
    """Export a boolean 3D grid as a mesh of cubes."""
    import trimesh

    R = occupancy.shape[0]
    coords = np.argwhere(occupancy)  # [N, 3]

    if len(coords) == 0:
        print("  WARNING: No occupied voxels!", flush=True)
        trimesh.Trimesh().export(output_path)
        return

    # Build a mesh of unit cubes at each occupied voxel
    voxel_size = 1.0 / R
    meshes = []
    for coord in coords:
        box = trimesh.primitives.Box(
            extents=[voxel_size] * 3,
            transform=trimesh.transformations.translation_matrix(
                (coord + 0.5) * voxel_size - 0.5
            ),
        )
        meshes.append(box)

    combined = trimesh.util.concatenate(meshes)
    combined.export(output_path)


if __name__ == "__main__":
    main()
