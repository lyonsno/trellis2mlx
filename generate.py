"""Generate a 3D mesh from a single image using trellis2mlx.

Full pipeline: image → DINOv3 features → sparse structure → shape latent → decode → mesh

Usage:
    PYTHONPATH=. python generate.py --image photo.png --output mesh.glb
"""

import argparse
import gc
import os
import time

import mlx.core as mx
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Generate 3D mesh from image via MLX")
    parser.add_argument("--image", help="Input image (requires PyTorch for DINOv3)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="/tmp/trellis-mlx-mesh.glb")
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

    # Downsample to resolution 32
    target_res = 32
    ratio = decoded.shape[0] // target_res
    decoded_ds = decoded.reshape(
        target_res, ratio, target_res, ratio, target_res, ratio
    ).any(axis=(1, 3, 5))
    coords = np.argwhere(decoded_ds)
    print(f"  {len(coords)} sparse voxels at {target_res}³", flush=True)

    cleanup_model(ss_flow, ss_dec)

    # === Stage 2: Shape Latent ===
    print("=== Stage 2: Shape Latent ===", flush=True)
    from trellmlx.models.slat_flow import SLatFlowModel

    slat_flow = SLatFlowModel()
    load_weights(slat_flow, HF_4B + "slat_flow_img2shape_dit_1_3B_512_bf16.safetensors", verbose=False)

    N = len(coords)
    slat_noise = mx.random.normal((N, 32))
    # Add batch dim to coords for the flow model
    coords_4d = np.column_stack([np.zeros(N, dtype=np.int32), coords])
    coords_mx = mx.array(coords_4d)

    t0 = time.perf_counter()
    shape_slat = flow_euler_sample(
        slat_flow, slat_noise, cond, neg_cond,
        steps=12, verbose=False,
        coords=mx.array(coords),  # 3D coords for RoPE
    )
    mx.eval(shape_slat)
    print(f"  Sampled: {time.perf_counter()-t0:.1f}s ({N} tokens)", flush=True)

    cleanup_model(slat_flow)

    # === Stage 3: Decode to Mesh ===
    print("=== Stage 3: Decode to Mesh ===", flush=True)
    from trellmlx.models.shape_slat_decoder import ShapeSLatDecoder

    decoder = ShapeSLatDecoder()
    load_weights(decoder, HF_4B + "shape_dec_next_dc_f16c32_fp16.safetensors", verbose=False)

    t0 = time.perf_counter()
    dec_out, dec_coords = decoder(shape_slat, coords_mx)
    mx.eval(dec_out)
    print(f"  Decoded: {time.perf_counter()-t0:.1f}s ({dec_out.shape[0]} voxels)", flush=True)

    cleanup_model(decoder)

    # === Mesh Extraction ===
    print("=== Mesh Extraction ===", flush=True)
    from trellmlx.mesh_extract import decoder_output_to_mesh

    t0 = time.perf_counter()
    vertices, faces = decoder_output_to_mesh(
        np.array(dec_out),
        np.array(dec_coords),
        resolution=256,  # decoder config resolution
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
