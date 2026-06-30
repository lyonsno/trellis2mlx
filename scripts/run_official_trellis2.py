"""Run the Trellis-Mac pipeline end-to-end and save raw mesh + decoder output.

Runs Microsoft's TRELLIS.2 pipeline via Trellis-Mac (PyTorch MPS) to produce
ground-truth mesh topology and decoder features for comparison against
trellis2mlx (MLX reimplementation).

Usage (via Greenroom):
    /Users/noahlyons/dev/trellis-mac/.venv/bin/python -u \
        /private/tmp/trellis2mlx-molten-remesh-parity-0629/scripts/run_official_trellis2.py \
        --image /path/to/T.png \
        --output-dir /path/to/output \
        --save-raw-mesh
"""

import argparse
import json
import os
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Run Trellis-Mac pipeline")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--save-raw-mesh", action="store_true",
                        help="Save raw mesh and decoder output before cleanup")
    parser.add_argument("--remesh", action="store_true",
                        help="Run with remesh=True in to_glb")
    parser.add_argument("--target-faces", type=int, default=350000)
    parser.add_argument("--texture-size", type=int, default=1024)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Add TRELLIS.2 to path
    trellis_dir = "/Users/noahlyons/dev/trellis-mac/TRELLIS.2"
    sys.path.insert(0, trellis_dir)
    os.chdir(trellis_dir)

    import torch
    print(f"PyTorch {torch.__version__}, MPS: {torch.backends.mps.is_available()}")

    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    # Load pipeline
    print("Loading pipeline...", flush=True)
    t0 = time.perf_counter()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.to("mps")
    print(f"Pipeline loaded: {time.perf_counter()-t0:.1f}s", flush=True)

    # Hook the shape decoder to capture raw 7-channel features before extraction
    _captured = {}

    if args.save_raw_mesh:
        decoder = pipeline.models['shape_slat_decoder']
        _orig_forward = decoder.forward

        @torch.no_grad()
        def _hooked_forward(x, **kwargs):
            # Run the parent decoder (NeXtDecoderSC.forward) to get raw output
            decoded = decoder.__class__.__bases__[0].forward(decoder, x, **kwargs)
            out_list = list(decoded) if isinstance(decoded, tuple) else [decoded]
            h = out_list[0]

            # Capture raw 7-channel features before sigmoid/threshold/softplus
            _captured['feats'] = h.feats.detach().cpu().numpy()
            _captured['coords'] = h.coords.detach().cpu().numpy()
            print(f"  [hook] Captured decoder output: {h.feats.shape[0]:,} voxels, "
                  f"{h.feats.shape[1]} channels", flush=True)

            # Continue with normal forward (extraction happens here)
            return _orig_forward(x, **kwargs)

        decoder.forward = _hooked_forward

    # Load image
    image = Image.open(args.image)
    print(f"Image: {args.image} ({image.size})", flush=True)

    # Run pipeline
    print("Running pipeline...", flush=True)
    t0 = time.perf_counter()
    meshes = pipeline.run(image)
    mesh = meshes[0]
    print(f"Pipeline done: {time.perf_counter()-t0:.1f}s", flush=True)

    # Save raw mesh topology metrics
    verts = mesh.vertices.cpu().numpy()
    faces = mesh.faces.cpu().numpy()
    print(f"Raw mesh: {len(verts):,}V {len(faces):,}F", flush=True)

    if args.save_raw_mesh:
        # Save raw mesh
        raw_path = os.path.join(args.output_dir, "raw_mesh.npz")
        np.savez(raw_path, vertices=verts, faces=faces)
        print(f"Saved raw mesh: {raw_path}", flush=True)

        # Save captured decoder output
        if _captured:
            decoder_path = os.path.join(args.output_dir, "decoder_output.npz")
            np.savez(decoder_path, **_captured)
            print(f"Saved decoder output: {decoder_path} "
                  f"(feats: {_captured['feats'].shape}, coords: {_captured['coords'].shape})",
                  flush=True)

        # Compute topology metrics
        from collections import Counter
        edge_count = Counter()
        for face in faces:
            for i in range(3):
                e = (min(face[i], face[(i+1)%3]), max(face[i], face[(i+1)%3]))
                edge_count[e] += 1
        boundary_edges = sum(1 for c in edge_count.values() if c == 1)
        non_manifold = sum(1 for c in edge_count.values() if c > 2)
        print(f"Raw topology: {boundary_edges:,} boundary edges, "
              f"{non_manifold:,} non-manifold edges", flush=True)

        metrics = {
            "vertices": len(verts),
            "faces": len(faces),
            "boundary_edges": boundary_edges,
            "non_manifold_edges": non_manifold,
        }
        if _captured:
            metrics["decoder_feats_shape"] = list(_captured['feats'].shape)
            metrics["decoder_coords_shape"] = list(_captured['coords'].shape)

        with open(os.path.join(args.output_dir, "raw_mesh_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    # Skip to_glb for now — it crashes on MPS device mismatch in texture baking
    print("Skipping to_glb (known MPS device mismatch in texture baking)", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
