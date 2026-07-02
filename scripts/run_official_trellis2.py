"""Run the local Trellis-Mac TRELLIS.2 route and save comparison artifacts.

This is a local-reference route for stage comparison against trellis2mlx. It is
not Microsoft CUDA TRELLIS.2 evidence. The script intentionally writes route
identity before importing PyTorch so failed runs still leave a usable report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


SCHEMA = "trellis2mlx.official_trellis2_route.v1"
TRELLIS_DIR = Path("/Users/noahlyons/dev/trellis-mac/TRELLIS.2")


class _StopAfterSparse(Exception):
    """Internal control-flow sentinel for sparse-only diagnostic runs."""


class _StopAfterConditioning(Exception):
    """Internal control-flow sentinel for conditioning-only diagnostic runs."""


class _StopAfterShapeSLat(Exception):
    """Internal control-flow sentinel for shape-SLat-only diagnostic runs."""


class _StopAfterShapeFlowStep(Exception):
    """Internal control-flow sentinel for first-step shape-flow diagnostic runs."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local Trellis-Mac TRELLIS.2 pipeline")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--save-raw-mesh",
        action="store_true",
        help="Save raw mesh and decoder output before cleanup/export",
    )
    parser.add_argument("--remesh", action="store_true", help="Reserved; final GLB is not emitted")
    parser.add_argument(
        "--shared-noise",
        metavar="PATH",
        help="Load shared noise tensors from .npz for matched comparison",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Override sampler steps for all stages (0=use pipeline defaults)",
    )
    parser.add_argument(
        "--pipeline-type",
        default=None,
        help="Pipeline type: '512', '1024', or '1024_cascade'. Empty means pipeline default.",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Pass preprocess_image=False to the Trellis-Mac pipeline",
    )
    parser.add_argument("--save-conditioning", action="store_true")
    parser.add_argument("--stop-after-conditioning", action="store_true")
    parser.add_argument("--save-sparse-coords", action="store_true")
    parser.add_argument("--stop-after-sparse", action="store_true")
    parser.add_argument("--save-shape-slat", action="store_true")
    parser.add_argument("--stop-after-shape-slat", action="store_true")
    parser.add_argument("--save-shape-flow-step", action="store_true")
    parser.add_argument("--stop-after-shape-flow-step", action="store_true")
    parser.add_argument(
        "--stop-after-raw-mesh",
        action="store_true",
        help="Exit after raw mesh/decoder artifacts are written",
    )
    parser.add_argument("--target-faces", type=int, default=350000)
    parser.add_argument("--texture-size", type=int, default=1024)
    return parser


def build_route_identity(
    args: argparse.Namespace, *, command: list[str] | None = None
) -> dict[str, Any]:
    image_path = str(Path(args.image))
    output_dir = str(Path(args.output_dir))
    shared_noise = args.shared_noise or ""
    requested_stops = {
        "conditioning": bool(args.stop_after_conditioning),
        "sparse": bool(args.stop_after_sparse),
        "shape_slat": bool(args.stop_after_shape_slat),
        "shape_flow_step": bool(args.stop_after_shape_flow_step),
        "raw_mesh": bool(args.stop_after_raw_mesh),
    }
    return {
        "schema": SCHEMA,
        "route": {
            "family": "local-reference/trellis-mac",
            "backend": "pytorch-mps",
            "trellis_dir": str(TRELLIS_DIR),
            "pipeline_type": args.pipeline_type or None,
            "seed": args.seed,
            "steps": args.steps,
            "preprocess_image": not args.no_preprocess,
            "target_faces": args.target_faces,
            "texture_size": args.texture_size,
            "shared_noise_path": shared_noise or None,
            "shared_noise_sha256": _sha256_file(shared_noise) if shared_noise else None,
        },
        "source": {
            "image_path": image_path,
            "image_sha256": _sha256_file(image_path),
        },
        "requested_outputs": {
            "conditioning": bool(args.save_conditioning or args.stop_after_conditioning),
            "sparse_coords": bool(args.save_sparse_coords or args.stop_after_sparse),
            "shape_slat": bool(args.save_shape_slat or args.stop_after_shape_slat),
            "shape_flow_step": bool(args.save_shape_flow_step or args.stop_after_shape_flow_step),
            "raw_mesh": bool(args.save_raw_mesh),
            "decoder_output": bool(args.save_raw_mesh),
            "final_glb": False,
        },
        "requested_stops": requested_stops,
        "output_dir": output_dir,
        "env": {
            "ATTN_BACKEND": os.environ.get("ATTN_BACKEND"),
            "SPARSE_ATTN_BACKEND": os.environ.get("SPARSE_ATTN_BACKEND"),
            "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
        },
        "command": command or sys.argv,
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": _sha256_file(Path(__file__).resolve()),
        "forbidden_inferences": [
            "not Microsoft CUDA TRELLIS.2 evidence",
            "not final-GLB parity evidence",
            "not texture/bake parity evidence",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    route_identity = build_route_identity(args)
    route_path = output_dir / "route_identity.json"
    _write_json(route_path, route_identity)
    _write_json(
        output_dir / "run_report.json",
        {
            "schema": "trellis2mlx.official_trellis2_run_report.v1",
            "status": "starting",
            "route_identity": route_identity,
            "last_trustworthy_phase": "route_identity_written",
            "primary_output_status": "not_started",
        },
    )

    # Add TRELLIS.2 to path after route identity is durable.
    sys.path.insert(0, str(TRELLIS_DIR))
    os.chdir(TRELLIS_DIR)

    import torch
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    print(f"PyTorch {torch.__version__}, MPS: {torch.backends.mps.is_available()}", flush=True)

    print("Loading pipeline...", flush=True)
    load_start = time.perf_counter()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.to("mps")
    load_elapsed = time.perf_counter() - load_start
    print(f"Pipeline loaded: {load_elapsed:.1f}s", flush=True)

    captured: dict[str, np.ndarray] = {}
    if args.save_raw_mesh:
        _install_decoder_capture_hook(pipeline, captured)
    if args.save_sparse_coords or args.stop_after_sparse:
        _install_sparse_capture_hook(pipeline, output_dir, stop_after_sparse=args.stop_after_sparse)
    if args.save_conditioning or args.stop_after_conditioning:
        _install_conditioning_capture_hook(
            pipeline, output_dir, stop_after_conditioning=args.stop_after_conditioning
        )
    if args.save_shape_slat or args.stop_after_shape_slat:
        _install_shape_slat_capture_hook(
            pipeline, output_dir, stop_after_shape_slat=args.stop_after_shape_slat
        )
    if args.save_shape_flow_step or args.stop_after_shape_flow_step:
        _install_shape_flow_step_capture_hook(
            pipeline, output_dir, stop_after_shape_flow_step=args.stop_after_shape_flow_step
        )

    image = Image.open(args.image)
    print(f"Image: {args.image} ({image.size})", flush=True)

    if args.shared_noise:
        _install_shared_noise(torch, args.shared_noise)

    run_kwargs = _build_run_kwargs(args)
    print(
        f"Running pipeline (type={args.pipeline_type or 'default'}, "
        f"steps={args.steps or 'default'}, seed={args.seed}, "
        f"preprocess={not args.no_preprocess})...",
        flush=True,
    )
    run_start = time.perf_counter()
    try:
        meshes = _run_mesh_pipeline(pipeline, image, args)
    except _StopAfterConditioning:
        return _finish_stage_only(output_dir, route_identity, "conditioning", run_start)
    except _StopAfterSparse:
        return _finish_stage_only(output_dir, route_identity, "sparse", run_start)
    except _StopAfterShapeSLat:
        return _finish_stage_only(output_dir, route_identity, "shape_slat", run_start)
    except _StopAfterShapeFlowStep:
        return _finish_stage_only(output_dir, route_identity, "shape_flow_step", run_start)

    run_elapsed = time.perf_counter() - run_start
    mesh = meshes[0]
    print(f"Pipeline done: {run_elapsed:.1f}s", flush=True)

    verts = mesh.vertices.cpu().numpy()
    faces = mesh.faces.cpu().numpy()
    print(f"Raw mesh: {len(verts):,}V {len(faces):,}F", flush=True)

    artifact_status: dict[str, Any] = {}
    if args.save_raw_mesh:
        raw_path = output_dir / "raw_mesh.npz"
        np.savez(raw_path, vertices=verts, faces=faces)
        print(f"Saved raw mesh: {raw_path}", flush=True)
        artifact_status["raw_mesh"] = str(raw_path)

        if captured:
            decoder_path = output_dir / "decoder_output.npz"
            np.savez(decoder_path, **captured)
            print(
                f"Saved decoder output: {decoder_path} "
                f"(feats: {captured['feats'].shape}, coords: {captured['coords'].shape})",
                flush=True,
            )
            artifact_status["decoder_output"] = str(decoder_path)

        metrics = _raw_mesh_metrics(verts, faces)
        if captured:
            metrics["decoder_feats_shape"] = list(captured["feats"].shape)
            metrics["decoder_coords_shape"] = list(captured["coords"].shape)
        _write_json(output_dir / "raw_mesh_metrics.json", metrics)
        artifact_status["raw_mesh_metrics"] = str(output_dir / "raw_mesh_metrics.json")

    _write_json(
        output_dir / "run_report.json",
        {
            "schema": "trellis2mlx.official_trellis2_run_report.v1",
            "status": "done",
            "route_identity": route_identity,
            "last_trustworthy_phase": "raw_mesh_saved" if args.save_raw_mesh else "pipeline_run_done",
            "primary_output_status": "written" if args.save_raw_mesh else "not_requested",
            "elapsed_seconds": {
                "pipeline_load": load_elapsed,
                "pipeline_run": run_elapsed,
            },
            "artifacts": artifact_status,
            "final_glb_status": (
                "not_requested_stage_stop"
                if args.stop_after_raw_mesh
                else "not_emitted_known_mps_texture_bake_mismatch"
            ),
        },
    )
    if args.stop_after_raw_mesh:
        print("Raw-mesh-only run done.", flush=True)
    else:
        print("Skipping to_glb (known MPS device mismatch in texture baking)", flush=True)
    print("Done.", flush=True)
    return 0


def _build_run_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    run_kwargs: dict[str, Any] = {"seed": args.seed}
    if args.pipeline_type:
        run_kwargs["pipeline_type"] = args.pipeline_type
    if args.no_preprocess:
        run_kwargs["preprocess_image"] = False
    if args.steps > 0:
        step_override = {"steps": args.steps}
        run_kwargs["sparse_structure_sampler_params"] = step_override
        run_kwargs["shape_slat_sampler_params"] = step_override
        run_kwargs["tex_slat_sampler_params"] = step_override
    return run_kwargs


def _run_mesh_pipeline(pipeline: Any, image: Any, args: argparse.Namespace) -> list[Any]:
    if args.stop_after_raw_mesh:
        return _run_raw_mesh_only(pipeline, image, args)
    return pipeline.run(image, **_build_run_kwargs(args))


def _run_raw_mesh_only(pipeline: Any, image: Any, args: argparse.Namespace) -> list[Any]:
    import torch

    pipeline_type = args.pipeline_type or pipeline.default_pipeline_type
    if pipeline_type not in {"512", "1024"}:
        raise ValueError(
            "--stop-after-raw-mesh currently supports non-cascade pipeline_type "
            f"'512' or '1024', found {pipeline_type!r}"
        )

    if args.no_preprocess:
        processed_image = image
    else:
        processed_image = pipeline.preprocess_image(image)

    torch.manual_seed(args.seed)
    cond_512 = pipeline.get_cond([processed_image], 512)
    if pipeline_type == "512":
        shape_cond = cond_512
        ss_res = 32
        shape_model = pipeline.models["shape_slat_flow_model_512"]
        resolution = 512
    else:
        shape_cond = pipeline.get_cond([processed_image], 1024)
        ss_res = 64
        shape_model = pipeline.models["shape_slat_flow_model_1024"]
        resolution = 1024

    sparse_params, shape_params = _sampler_param_overrides(args)
    coords = pipeline.sample_sparse_structure(cond_512, ss_res, 1, sparse_params)
    shape_slat = pipeline.sample_shape_slat(shape_cond, shape_model, coords, shape_params)
    decoded = pipeline.decode_shape_slat(shape_slat, resolution)
    if isinstance(decoded, tuple):
        meshes, _subs = decoded
        return meshes
    return decoded


def _sampler_param_overrides(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    if args.steps <= 0:
        return {}, {}
    override = {"steps": args.steps}
    return override, override


def _install_decoder_capture_hook(pipeline: Any, captured: dict[str, np.ndarray]) -> None:
    import torch

    decoder = pipeline.models["shape_slat_decoder"]
    original_forward = decoder.forward

    @torch.no_grad()
    def _hooked_forward(x, **kwargs):
        decoded = decoder.__class__.__bases__[0].forward(decoder, x, **kwargs)
        out_list = list(decoded) if isinstance(decoded, tuple) else [decoded]
        h = out_list[0]
        captured["feats"] = h.feats.detach().cpu().numpy()
        captured["coords"] = h.coords.detach().cpu().numpy()
        print(
            f"  [hook] Captured decoder output: {h.feats.shape[0]:,} voxels, "
            f"{h.feats.shape[1]} channels",
            flush=True,
        )
        return original_forward(x, **kwargs)

    decoder.forward = _hooked_forward


def _install_sparse_capture_hook(
    pipeline: Any, output_dir: Path, *, stop_after_sparse: bool = False
) -> None:
    original = pipeline.sample_sparse_structure

    def _hooked_sample_sparse_structure(*args, **kwargs):
        coords = original(*args, **kwargs)
        coords_np = _to_numpy(coords).astype(np.int32, copy=False)
        sparse_path = output_dir / "sparse_coords.npz"
        np.savez(sparse_path, coords=coords_np)
        print(f"Saved sparse coords: {sparse_path} ({coords_np.shape})", flush=True)
        if stop_after_sparse:
            raise _StopAfterSparse()
        return coords

    pipeline.sample_sparse_structure = _hooked_sample_sparse_structure


def _install_conditioning_capture_hook(
    pipeline: Any, output_dir: Path, *, stop_after_conditioning: bool = False
) -> None:
    original = pipeline.get_cond

    def _hooked_get_cond(*args, **kwargs):
        cond = original(*args, **kwargs)
        cond_np = _to_numpy(cond["cond"]).astype(np.float32, copy=False)
        neg_cond_np = _to_numpy(cond["neg_cond"]).astype(np.float32, copy=False)
        resolution = args[1] if len(args) > 1 else kwargs.get("resolution")
        conditioning_path = output_dir / "conditioning.npz"
        np.savez(conditioning_path, cond=cond_np, neg_cond=neg_cond_np)
        _write_json(
            output_dir / "conditioning.json",
            {
                "resolution": int(resolution) if resolution is not None else None,
                "shape": list(cond_np.shape),
                "tokens": int(cond_np.shape[1]) if cond_np.ndim == 3 else None,
                "channels": int(cond_np.shape[2]) if cond_np.ndim == 3 else None,
            },
        )
        print(
            f"Saved conditioning: {conditioning_path} "
            f"(cond: {cond_np.shape}, neg_cond: {neg_cond_np.shape})",
            flush=True,
        )
        if stop_after_conditioning:
            raise _StopAfterConditioning()
        return cond

    pipeline.get_cond = _hooked_get_cond


def _install_shape_slat_capture_hook(
    pipeline: Any, output_dir: Path, *, stop_after_shape_slat: bool = False
) -> None:
    original = pipeline.sample_shape_slat

    def _hooked_sample_shape_slat(*args, **kwargs):
        shape_slat = original(*args, **kwargs)
        feats_np = _to_numpy(shape_slat.feats).astype(np.float32, copy=False)
        coords_np = _to_numpy(shape_slat.coords).astype(np.int32, copy=False)
        slat_path = output_dir / "shape_slat.npz"
        np.savez(slat_path, feats=feats_np, coords=coords_np)
        print(
            f"Saved shape SLat: {slat_path} "
            f"(feats: {feats_np.shape}, coords: {coords_np.shape})",
            flush=True,
        )
        if stop_after_shape_slat:
            raise _StopAfterShapeSLat()
        return shape_slat

    pipeline.sample_shape_slat = _hooked_sample_shape_slat


def _install_shape_flow_step_capture_hook(
    pipeline: Any, output_dir: Path, *, stop_after_shape_flow_step: bool = False
) -> None:
    sampler = pipeline.shape_slat_sampler
    original = sampler._get_model_prediction
    captured = {"done": False}

    def _hooked_get_model_prediction(model, x_t, t, cond=None, **kwargs):
        pred_x0, pred_eps, pred_v = original(model, x_t, t, cond, **kwargs)
        if not captured["done"]:
            captured["done"] = True
            step_path = output_dir / "shape_flow_step0.npz"
            np.savez(
                step_path,
                sample_feats=_to_numpy(x_t.feats).astype(np.float32, copy=False),
                coords=_to_numpy(x_t.coords).astype(np.int32, copy=False),
                pred_x0_feats=_to_numpy(pred_x0.feats).astype(np.float32, copy=False),
                pred_eps_feats=_to_numpy(pred_eps.feats).astype(np.float32, copy=False),
                pred_v_feats=_to_numpy(pred_v.feats).astype(np.float32, copy=False),
                t=np.array(t, dtype=np.float32),
            )
            print(f"Saved first shape-flow step: {step_path}", flush=True)
            if stop_after_shape_flow_step:
                raise _StopAfterShapeFlowStep()
        return pred_x0, pred_eps, pred_v

    sampler._get_model_prediction = _hooked_get_model_prediction


def _install_shared_noise(torch: Any, shared_noise_path: str) -> None:
    shared = np.load(shared_noise_path)
    noise_calls = [0]
    noise_map = {
        (1, 8, 16, 16, 16): torch.tensor(shared["ss_noise"]),
    }
    slat_pool = torch.tensor(shared["slat_noise_pool"])
    tex_pool = torch.tensor(shared["tex_noise_pool"])
    original_randn = torch.randn

    def _patched_randn(*args_t, **kwargs):
        shape = args_t if len(args_t) > 1 else args_t[0] if args_t else None
        if isinstance(shape, (list, tuple)):
            shape = tuple(shape)
        elif hasattr(shape, "shape"):
            shape = tuple(shape.shape)

        if shape in noise_map:
            result = noise_map[shape].clone()
            if "device" in kwargs:
                result = result.to(kwargs["device"])
            print(f"  [shared noise] Injected {shape}", flush=True)
            return result

        if isinstance(shape, tuple) and len(shape) == 2 and shape[1] == 32:
            n = shape[0]
            noise_calls[0] += 1
            pool = slat_pool[:n] if noise_calls[0] <= 1 else tex_pool[:n]
            result = pool.clone()
            if "device" in kwargs:
                result = result.to(kwargs["device"])
            print(f"  [shared noise] Injected SLat {shape} (call #{noise_calls[0]})", flush=True)
            return result

        return original_randn(*args_t, **kwargs)

    torch.randn = _patched_randn
    print(f"Shared noise loaded from {shared_noise_path}", flush=True)


def _finish_stage_only(
    output_dir: Path, route_identity: dict[str, Any], stage: str, run_start: float
) -> int:
    elapsed = time.perf_counter() - run_start
    _write_json(
        output_dir / "run_report.json",
        {
            "schema": "trellis2mlx.official_trellis2_run_report.v1",
            "status": "done",
            "route_identity": route_identity,
            "last_trustworthy_phase": f"{stage}_saved",
            "primary_output_status": "written",
            "elapsed_seconds": {"pipeline_run": elapsed},
            "final_glb_status": "not_requested_stage_stop",
        },
    )
    print(f"{stage} run done: {elapsed:.1f}s", flush=True)
    print("Done.", flush=True)
    return 0


def _raw_mesh_metrics(verts: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    from collections import Counter

    edge_count: Counter[tuple[int, int]] = Counter()
    same_direction: Counter[tuple[int, int]] = Counter()
    for face in faces:
        for i in range(3):
            a = int(face[i])
            b = int(face[(i + 1) % 3])
            edge_count[(min(a, b), max(a, b))] += 1
            same_direction[(a, b)] += 1
    boundary_edges = sum(1 for c in edge_count.values() if c == 1)
    non_manifold = sum(1 for c in edge_count.values() if c > 2)
    same_dir_conflicts = 0
    for (a, b), count in same_direction.items():
        if a < b and count and same_direction.get((b, a), 0) == 0 and edge_count[(a, b)] > 1:
            same_dir_conflicts += edge_count[(a, b)]
    print(
        f"Raw topology: {boundary_edges:,} boundary edges, "
        f"{non_manifold:,} non-manifold edges",
        flush=True,
    )
    return {
        "vertices": int(len(verts)),
        "faces": int(len(faces)),
        "boundary_edges": int(boundary_edges),
        "non_manifold_edges": int(non_manifold),
        "same_direction_conflict_edges": int(same_dir_conflicts),
    }


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _sha256_file(path: str | os.PathLike[str]) -> str | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
