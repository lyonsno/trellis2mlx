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


class _StopAfterSparse(Exception):
    """Internal control-flow sentinel for sparse-only diagnostic runs."""


def main():
    parser = argparse.ArgumentParser(description="Run Trellis-Mac pipeline")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--save-raw-mesh", action="store_true",
                        help="Save raw mesh and decoder output before cleanup")
    parser.add_argument("--remesh", action="store_true",
                        help="Run with remesh=True in to_glb")
    parser.add_argument("--shared-noise", metavar="PATH",
                        help="Load shared noise tensors from .npz for matched comparison")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed passed to Trellis2ImageTo3DPipeline.run")
    parser.add_argument("--steps", type=int, default=0,
                        help="Override sampler steps for all stages (0=use pipeline defaults)")
    parser.add_argument("--pipeline-type", default=None,
                        help="Pipeline type: '512', '1024', '1024_cascade'. "
                             "Default: auto (usually 1024_cascade for 4B model)")
    parser.add_argument("--no-preprocess", action="store_true",
                        help="Pass preprocess_image=False to the Trellis-Mac pipeline")
    parser.add_argument("--save-sparse-coords", action="store_true",
                        help="Save sparse structure coordinates to sparse_coords.npz")
    parser.add_argument("--stop-after-sparse", action="store_true",
                        help="Exit successfully after saving sparse structure coordinates")
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

    if args.save_sparse_coords or args.stop_after_sparse:
        _install_sparse_capture_hook(
            pipeline,
            args.output_dir,
            stop_after_sparse=args.stop_after_sparse,
        )

    # Load image
    image = Image.open(args.image)
    print(f"Image: {args.image} ({image.size})", flush=True)

    # Inject shared noise by monkey-patching torch.randn
    if args.shared_noise:
        _shared = np.load(args.shared_noise)
        _noise_calls = [0]
        _noise_map = {
            (1, 8, 16, 16, 16): torch.tensor(_shared["ss_noise"]),
        }
        _slat_pool = torch.tensor(_shared["slat_noise_pool"])
        _tex_pool = torch.tensor(_shared["tex_noise_pool"])
        _original_randn = torch.randn

        def _patched_randn(*args_t, **kwargs):
            shape = args_t if len(args_t) > 1 else args_t[0] if args_t else None
            if isinstance(shape, (list, tuple)):
                shape = tuple(shape)
            elif hasattr(shape, 'shape'):
                shape = tuple(shape.shape)

            # Match by shape
            if shape in _noise_map:
                result = _noise_map[shape].clone()
                if 'device' in kwargs:
                    result = result.to(kwargs['device'])
                print(f"  [shared noise] Injected {shape}", flush=True)
                return result

            # For SLat-sized noise (N, 32), use pool
            if len(shape) == 2 and shape[1] == 32:
                n = shape[0]
                _noise_calls[0] += 1
                if _noise_calls[0] <= 1:
                    pool = _slat_pool[:n]
                else:
                    pool = _tex_pool[:n]
                result = pool.clone()
                if 'device' in kwargs:
                    result = result.to(kwargs['device'])
                print(f"  [shared noise] Injected SLat {shape} (call #{_noise_calls[0]})",
                      flush=True)
                return result

            return _original_randn(*args_t, **kwargs)

        torch.randn = _patched_randn
        print(f"Shared noise loaded from {args.shared_noise}", flush=True)

    # Run pipeline
    run_kwargs = _build_run_kwargs(args)
    print(f"Running pipeline (type={args.pipeline_type or 'default'}, "
          f"steps={args.steps or 'default'}, seed={args.seed}, "
          f"preprocess={not args.no_preprocess})...", flush=True)
    t0 = time.perf_counter()
    try:
        meshes = pipeline.run(image, **run_kwargs)
    except _StopAfterSparse:
        print(f"Sparse-only run done: {time.perf_counter()-t0:.1f}s", flush=True)
        print("Done.", flush=True)
        return
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


def _build_run_kwargs(args):
    run_kwargs = {
        'seed': args.seed,
    }
    if args.pipeline_type:
        run_kwargs['pipeline_type'] = args.pipeline_type
    if args.no_preprocess:
        run_kwargs['preprocess_image'] = False
    if args.steps > 0:
        step_override = {'steps': args.steps}
        run_kwargs['sparse_structure_sampler_params'] = step_override
        run_kwargs['shape_slat_sampler_params'] = step_override
        run_kwargs['tex_slat_sampler_params'] = step_override
    return run_kwargs


def _install_sparse_capture_hook(pipeline, output_dir, *, stop_after_sparse=False):
    original = pipeline.sample_sparse_structure

    def _hooked_sample_sparse_structure(*args, **kwargs):
        coords = original(*args, **kwargs)
        coords_np = _to_numpy(coords).astype(np.int32, copy=False)
        os.makedirs(output_dir, exist_ok=True)
        sparse_path = os.path.join(output_dir, "sparse_coords.npz")
        np.savez(sparse_path, coords=coords_np)
        print(f"Saved sparse coords: {sparse_path} ({coords_np.shape})", flush=True)
        if stop_after_sparse:
            raise _StopAfterSparse()
        return coords

    pipeline.sample_sparse_structure = _hooked_sample_sparse_structure


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


if __name__ == "__main__":
    main()
