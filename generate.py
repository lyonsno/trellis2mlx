"""Generate a 3D mesh from a single image using trellis2mlx.

Two-pass pipeline matching the TRELLIS.2 reference:
  1. Image → DINOv3 features
  2. Sparse structure flow → occupancy grid → LR coordinates
  3. LR SLat flow → denormalize → decoder upsample → HR coordinates
  4. HR SLat flow → denormalize → full decode → mesh extraction

Usage:
    PYTHONPATH=. python generate.py --image photo.png --output mesh.glb
"""

import argparse
import gc
import os
import time

import mlx.core as mx
import numpy as np

# SLat normalization from pipeline.json
SHAPE_SLAT_MEAN = np.array([
    0.781296, 0.018091, -0.495192, -0.558457, 1.06053, 0.093252,
    1.518149, -0.933218, -0.732996, 2.604095, -0.118341, -2.143904,
    0.495076, -2.179512, -2.130751, -0.996944, 0.261421, -2.217463,
    1.260067, -0.150213, 3.790713, 1.481266, -1.046058, -1.523667,
    -0.059621, 2.22078, 1.621212, 0.87723, 0.567247, -3.175944,
    -3.186688, 1.578665,
], dtype=np.float32)

SHAPE_SLAT_STD = np.array([
    5.972266, 4.706852, 5.44501, 5.209927, 5.32022, 4.547237,
    5.020802, 5.444004, 5.226681, 5.683095, 4.831436, 5.286469,
    5.652043, 5.367606, 5.525084, 4.730578, 4.805265, 5.124013,
    5.530808, 5.619001, 5.10393, 5.41767, 5.269677, 5.547194,
    5.634698, 5.235274, 6.110351, 5.511298, 6.237273, 4.879207,
    5.347008, 5.405691,
], dtype=np.float32)


def _denormalize_slat(slat: mx.array) -> mx.array:
    return slat * mx.array(SHAPE_SLAT_STD) + mx.array(SHAPE_SLAT_MEAN)


def _requantize_coords(hr_coords_np, lr_resolution, hr_resolution):
    """Requantize decoder output coords to the target resolution.

    Maps from the decoder's upsampled space back to hr_resolution,
    deduplicates, and returns unique coords.

    Args:
        hr_coords_np: [N, 4] int array (batch, z, y, x) at decoder output res
        lr_resolution: input coordinate resolution (e.g. 32)
        hr_resolution: target mesh resolution (e.g. 256)

    Returns:
        unique_coords: [M, 4] int array at hr_resolution
    """
    spatial = hr_coords_np[:, 1:4].astype(np.float64)
    spatial = ((spatial + 0.5) / lr_resolution * (hr_resolution // 16)).astype(np.int32)
    result = hr_coords_np.copy()
    result[:, 1:4] = spatial
    unique_coords = np.unique(result, axis=0)
    return unique_coords


def main():
    parser = argparse.ArgumentParser(description="Generate 3D mesh from image via MLX")
    parser.add_argument("--image", help="Input image (requires PyTorch for DINOv3)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="/tmp/trellis-mlx-mesh.glb")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Mesh resolution (default: 256 from checkpoint config)")
    parser.add_argument("--max-tokens", type=int, default=49152,
                        help="Max tokens for HR SLat pass (reduces resolution if exceeded)")
    args = parser.parse_args()

    mx.random.seed(args.seed)
    t_total = time.perf_counter()

    HF_4B = os.path.expanduser(
        "~/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/"
        "snapshots/af44b45f2e35a493886929c6d786e563ec68364d/ckpts/"
    )
    HF_LARGE = os.path.expanduser(
        "~/.cache/huggingface/hub/models--microsoft--TRELLIS-image-large/"
        "snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/"
    )

    from trellmlx.weight_loader import load_weights
    from trellmlx.samplers import flow_euler_sample
    from trellmlx.cleanup import cleanup_model, cleanup

    # === Image conditioning ===
    if args.image:
        cond = _extract_image_features(args.image)
    else:
        print("No image — random conditioning", flush=True)
        cond = mx.random.normal((1, 10, 1024))
    neg_cond = mx.zeros_like(cond)

    # === Stage 1: Sparse Structure ===
    print("=== Stage 1: Sparse Structure ===", flush=True)
    from trellmlx.models.sparse_structure_flow import SparseStructureFlowModel
    from trellmlx.models.sparse_structure_decoder import SparseStructureDecoder

    ss_flow = SparseStructureFlowModel()
    load_weights(ss_flow, HF_4B + "ss_flow_img_dit_1_3B_64_bf16.safetensors", verbose=False)
    ss_dec = SparseStructureDecoder()
    load_weights(ss_dec, HF_LARGE + "ss_dec_conv3d_16l8_fp16.safetensors", verbose=False)

    noise = mx.random.normal((1, 8, 16, 16, 16))
    t0 = time.perf_counter()
    z_s = flow_euler_sample(ss_flow, noise, cond, neg_cond, steps=12, verbose=False)
    mx.eval(z_s)
    print(f"  Sampled: {time.perf_counter()-t0:.1f}s", flush=True)

    logits = ss_dec(z_s.astype(mx.float32))
    mx.eval(logits)
    decoded = np.array(logits[0, 0] > 0)

    lr_resolution = 32
    ratio = decoded.shape[0] // lr_resolution
    decoded_ds = decoded.reshape(
        lr_resolution, ratio, lr_resolution, ratio, lr_resolution, ratio
    ).any(axis=(1, 3, 5))
    lr_coords = np.argwhere(decoded_ds)
    print(f"  {len(lr_coords)} sparse voxels at {lr_resolution}³", flush=True)

    cleanup_model(ss_flow, ss_dec)

    # === Stage 2a: LR Shape Latent ===
    print("\n=== Stage 2a: LR Shape Latent ===", flush=True)
    from trellmlx.models.slat_flow import SLatFlowModel

    slat_flow = SLatFlowModel()
    load_weights(slat_flow, HF_4B + "slat_flow_img2shape_dit_1_3B_512_bf16.safetensors", verbose=False)

    N_lr = len(lr_coords)
    lr_noise = mx.random.normal((N_lr, 32))
    lr_coords_4d = np.column_stack([np.zeros(N_lr, dtype=np.int32), lr_coords])

    t0 = time.perf_counter()
    lr_slat = flow_euler_sample(
        slat_flow, lr_noise, cond, neg_cond,
        steps=12, verbose=False,
        coords=mx.array(lr_coords),
    )
    mx.eval(lr_slat)
    print(f"  Sampled: {time.perf_counter()-t0:.1f}s ({N_lr} tokens)", flush=True)

    lr_slat = _denormalize_slat(lr_slat)
    mx.eval(lr_slat)

    # === Stage 2b: Upsample to get HR coordinates ===
    print("\n=== Stage 2b: Upsample → HR coordinates ===", flush=True)
    from trellmlx.models.shape_slat_decoder import ShapeSLatDecoder

    decoder = ShapeSLatDecoder()
    load_weights(decoder, HF_4B + "shape_dec_next_dc_f16c32_fp16.safetensors", verbose=False)

    t0 = time.perf_counter()
    hr_coords_raw = decoder.upsample(lr_slat, mx.array(lr_coords_4d), upsample_times=4)
    mx.eval(hr_coords_raw)
    print(f"  Upsampled: {time.perf_counter()-t0:.1f}s ({hr_coords_raw.shape[0]:,} voxels)", flush=True)

    # Requantize to target resolution, dedup
    hr_resolution = args.resolution
    hr_coords_np = np.array(hr_coords_raw)
    while True:
        quant_coords = _requantize_coords(hr_coords_np, lr_resolution, hr_resolution)
        num_tokens = len(quant_coords)
        if num_tokens < args.max_tokens or hr_resolution == 1024:
            if hr_resolution != args.resolution:
                print(f"  Resolution reduced to {hr_resolution} ({num_tokens:,} tokens)", flush=True)
            break
        hr_resolution -= 128

    print(f"  HR coords: {num_tokens:,} tokens at res {hr_resolution}", flush=True)

    cleanup_model(decoder)
    del decoder
    gc.collect()

    # === Stage 2c: HR Shape Latent (second SLat pass) ===
    print("\n=== Stage 2c: HR Shape Latent ===", flush=True)

    hr_coords_3d = quant_coords[:, 1:4]  # drop batch dim for RoPE
    hr_noise = mx.random.normal((num_tokens, 32))

    t0 = time.perf_counter()
    hr_slat = flow_euler_sample(
        slat_flow, hr_noise, cond, neg_cond,
        steps=12, verbose=False,
        coords=mx.array(hr_coords_3d),
    )
    mx.eval(hr_slat)
    print(f"  Sampled: {time.perf_counter()-t0:.1f}s ({num_tokens:,} tokens)", flush=True)

    hr_slat = _denormalize_slat(hr_slat)
    mx.eval(hr_slat)

    cleanup_model(slat_flow)
    del slat_flow
    gc.collect()

    # === Stage 3: Full Decode ===
    print("\n=== Stage 3: Decode to Mesh ===", flush=True)

    decoder = ShapeSLatDecoder()
    load_weights(decoder, HF_4B + "shape_dec_next_dc_f16c32_fp16.safetensors", verbose=False)

    t0 = time.perf_counter()
    dec_out, dec_coords = decoder(hr_slat, mx.array(quant_coords))
    mx.eval(dec_out)
    print(f"  Decoded: {time.perf_counter()-t0:.1f}s ({dec_out.shape[0]:,} voxels)", flush=True)

    cleanup_model(decoder)
    del decoder
    gc.collect()

    # === Mesh Extraction ===
    print("\n=== Mesh Extraction ===", flush=True)
    from trellmlx.mesh_extract import decoder_output_to_mesh

    dec_coords_np = np.array(dec_coords)
    dec_feats_np = np.array(dec_out)

    # The decoder output coords are used directly for mesh extraction.
    # grid_size controls world-space scaling: voxel_size = 1/grid_size.
    # The decoder upsamples 4x (2^4=16) from the input resolution, so
    # coords span [0, hr_resolution*16). Use that as grid_size for correct
    # [-0.5, 0.5] world-space scaling.
    mesh_grid_size = hr_resolution * 16
    print(f"  {dec_coords_np.shape[0]:,} voxels, coord range "
          f"[{dec_coords_np[:,1:].min()}, {dec_coords_np[:,1:].max()}], "
          f"grid_size={mesh_grid_size}", flush=True)

    t0 = time.perf_counter()
    vertices, faces = decoder_output_to_mesh(
        dec_feats_np,
        dec_coords_np,
        resolution=mesh_grid_size,
    )
    print(f"  Extracted: {time.perf_counter()-t0:.1f}s", flush=True)
    print(f"  {len(vertices):,} vertices, {len(faces):,} faces", flush=True)

    # === Export ===
    import trimesh
    if len(vertices) > 0 and len(faces) > 0:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.export(args.output)
        print(f"  Saved: {args.output} ({os.path.getsize(args.output)/1e6:.1f}MB)", flush=True)
    else:
        print("  WARNING: Empty mesh!", flush=True)

    total = time.perf_counter() - t_total
    print(f"\nTotal: {total:.1f}s", flush=True)
    cleanup()


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


if __name__ == "__main__":
    main()
