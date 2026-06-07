"""Smoke test: stages 1+2 — sparse structure + shape latent generation.

Runs:
1. SparseStructureFlowModel → occupancy grid
2. SparseStructureDecoder → 64³ occupancy → coordinates
3. SLatFlowModel → shape latents at occupied coordinates
4. Visualize latent magnitudes as colored voxels

Usage:
    python smoke_stage2.py --image photo.png
"""

import argparse
import gc
import os
import time

import mlx.core as mx
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Input image")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="/tmp/trellis-mlx-stage2.glb")
    args = parser.parse_args()

    mx.random.seed(args.seed)

    from trellmlx.models.sparse_structure_flow import SparseStructureFlowModel
    from trellmlx.models.sparse_structure_decoder import SparseStructureDecoder
    from trellmlx.models.slat_flow import SLatFlowModel
    from trellmlx.weight_loader import load_weights
    from trellmlx.samplers import flow_euler_sample

    HF_BASE = os.path.expanduser(
        "~/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/"
        "snapshots/af44b45f2e35a493886929c6d786e563ec68364d/ckpts/"
    )
    HF_LARGE = os.path.expanduser(
        "~/.cache/huggingface/hub/models--microsoft--TRELLIS-image-large/"
        "snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/"
    )

    # === Stage 1: Sparse Structure ===
    print("=== Stage 1: Sparse Structure ===", flush=True)
    print("Loading flow model...", flush=True)
    ss_flow = SparseStructureFlowModel()
    load_weights(ss_flow, HF_BASE + "ss_flow_img_dit_1_3B_64_bf16.safetensors", verbose=False)

    print("Loading decoder...", flush=True)
    ss_dec = SparseStructureDecoder()
    load_weights(ss_dec, HF_LARGE + "ss_dec_conv3d_16l8_fp16.safetensors", verbose=False)

    # Image conditioning
    if args.image:
        cond = _extract_image_features(args.image)
    else:
        print("No image — random conditioning", flush=True)
        cond = mx.random.normal((1, 10, 1024))
    neg_cond = mx.zeros_like(cond)

    # Sample structure
    noise = mx.random.normal((1, 8, 16, 16, 16))
    print("Sampling sparse structure (12 steps)...", flush=True)
    t0 = time.perf_counter()
    z_s = flow_euler_sample(ss_flow, noise, cond, neg_cond, steps=12, verbose=False)
    mx.eval(z_s)
    print(f"  {time.perf_counter()-t0:.1f}s", flush=True)

    # Decode to occupancy
    logits = ss_dec(z_s.astype(mx.float32))
    mx.eval(logits)
    decoded = np.array(logits[0, 0] > 0)  # [64, 64, 64]
    print(f"  Decoded: {decoded.shape}, {decoded.sum()} occupied at 64³", flush=True)

    # Downsample to target resolution (matches PyTorch pipeline's max_pool3d)
    # The SLat model operates on coords at the flow model's resolution (16³),
    # not the decoder's full 64³ output.
    target_res = 16
    if decoded.shape[0] != target_res:
        ratio = decoded.shape[0] // target_res
        # Max-pool: if any voxel in the ratio³ block is occupied, the downsampled voxel is occupied
        decoded_ds = decoded.reshape(
            target_res, ratio, target_res, ratio, target_res, ratio
        ).any(axis=(1, 3, 5))
        print(f"  Downsampled: {decoded_ds.shape}, {decoded_ds.sum()} occupied at {target_res}³", flush=True)
    else:
        decoded_ds = decoded

    coords = np.argwhere(decoded_ds)  # [N, 3]
    print(f"  {len(coords)} sparse tokens for SLat", flush=True)

    # Free stage 1 models
    del ss_flow, ss_dec, z_s, logits
    gc.collect()

    # === Stage 2: Shape SLat ===
    print("\n=== Stage 2: Shape Latent ===", flush=True)
    print("Loading SLatFlowModel...", flush=True)
    slat_flow = SLatFlowModel()
    load_weights(slat_flow, HF_BASE + "slat_flow_img2shape_dit_1_3B_512_bf16.safetensors", verbose=False)

    # Create noise at occupied coordinates
    N = len(coords)
    slat_noise = mx.random.normal((N, 32))  # [N, 32]
    coords_mx = mx.array(coords)

    print(f"Sampling shape latent ({N} tokens, 12 steps)...", flush=True)
    t0 = time.perf_counter()
    shape_slat = flow_euler_sample(
        slat_flow, slat_noise, cond, neg_cond,
        steps=12, verbose=False,
    )
    mx.eval(shape_slat)
    elapsed = time.perf_counter() - t0
    print(f"  {elapsed:.1f}s", flush=True)
    print(f"  Shape latent: [{N}, 32], mean={shape_slat.mean().item():.4f}", flush=True)

    # === Visualize ===
    print("\nExporting colored voxel mesh...", flush=True)
    _export_colored_voxels(coords, np.array(shape_slat), args.output)
    print(f"  Saved: {args.output}", flush=True)


def _extract_image_features(image_path, resolution=512):
    try:
        import torch, sys
        sys.path.insert(0, os.path.expanduser("~/dev/trellis-mac/TRELLIS.2"))
        from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor
        from PIL import Image
        extractor = DinoV3FeatureExtractor("facebook/dinov3-vitl16-pretrain-lvd1689m", image_size=resolution)
        extractor.to("cpu")
        img = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            features = extractor([img])
        print(f"  Features: {features.shape}", flush=True)
        return mx.array(features.numpy())
    except Exception as e:
        print(f"  Feature extraction failed: {e}", flush=True)
        return mx.random.normal((1, 10, 1024))


def _export_colored_voxels(coords, latents, output_path):
    """Export voxels colored by latent magnitude."""
    import trimesh

    N = len(coords)
    if N == 0:
        trimesh.Trimesh().export(output_path)
        return

    # Color by L2 norm of latent vector (normalized to 0-1)
    magnitudes = np.linalg.norm(latents, axis=-1)
    mag_norm = (magnitudes - magnitudes.min()) / (magnitudes.max() - magnitudes.min() + 1e-8)

    # Use trimesh VoxelGrid with marching cubes
    R = coords.max() + 1
    grid = np.zeros((R, R, R), dtype=bool)
    grid[coords[:, 0], coords[:, 1], coords[:, 2]] = True

    voxel_grid = trimesh.voxel.VoxelGrid(
        trimesh.voxel.encoding.DenseEncoding(grid)
    )
    mesh = voxel_grid.marching_cubes
    voxel_size = 1.0 / R
    mesh.apply_scale(voxel_size)
    mesh.apply_translation([-0.5, -0.5, -0.5])

    # Color vertices by nearest voxel's latent magnitude
    vert_coords = ((mesh.vertices + 0.5) / voxel_size).astype(int).clip(0, R - 1)
    # Map vertex to nearest occupied voxel
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    _, idx = tree.query(vert_coords)
    vert_mags = mag_norm[idx]

    # Colormap: blue (low) → red (high)
    colors = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
    colors[:, 0] = (vert_mags * 255).astype(np.uint8)  # red
    colors[:, 2] = ((1 - vert_mags) * 255).astype(np.uint8)  # blue
    colors[:, 3] = 255  # alpha
    mesh.visual.vertex_colors = colors

    mesh.export(output_path)
    print(f"  {N} voxels, {len(mesh.vertices)} vertices", flush=True)


if __name__ == "__main__":
    main()
