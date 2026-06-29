"""Run the official TRELLIS.2 pipeline end-to-end and save raw mesh at each stage.

This runs the Microsoft TRELLIS.2 pipeline (PyTorch MPS) to produce ground-truth
mesh topology for comparison against our trellis2mlx reimplementation.

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
    parser = argparse.ArgumentParser(description="Run official TRELLIS.2 pipeline")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--save-raw-mesh", action="store_true",
                        help="Save raw mesh before any cleanup/remesh")
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
    # Use MPS (Metal) instead of CUDA
    pipeline.to("mps")
    print(f"Pipeline loaded: {time.perf_counter()-t0:.1f}s", flush=True)

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
        raw_path = os.path.join(args.output_dir, "raw_mesh.npz")
        np.savez(raw_path, vertices=verts, faces=faces)
        print(f"Saved raw mesh: {raw_path}", flush=True)

        # Compute topology metrics inline
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

        # Save metrics
        metrics = {
            "vertices": len(verts),
            "faces": len(faces),
            "boundary_edges": boundary_edges,
            "non_manifold_edges": non_manifold,
        }
        with open(os.path.join(args.output_dir, "raw_mesh_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    # Simplify (nvdiffrast limit)
    mesh.simplify(16777216)

    # Export GLB via official to_glb
    import o_voxel
    print(f"Running to_glb (remesh={args.remesh})...", flush=True)
    t0 = time.perf_counter()
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.target_faces,
        texture_size=args.texture_size,
        remesh=args.remesh,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    print(f"to_glb done: {time.perf_counter()-t0:.1f}s", flush=True)

    output_path = os.path.join(args.output_dir, "output.glb")
    glb.export(output_path)
    print(f"Saved: {output_path} ({os.path.getsize(output_path)/1e6:.1f}MB)", flush=True)


if __name__ == "__main__":
    main()
