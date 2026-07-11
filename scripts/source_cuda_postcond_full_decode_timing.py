"""Time official TRELLIS.2 source CUDA decode from saved conditioning.

This is a post-conditioning timing witness. It intentionally does not instantiate
the source DINO/rembg path because DINOv3 is a gated dependency and the existing
diagnostic surface already has durable conditioning tensors.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tarfile
import time
from typing import Any

import numpy as np


MODEL_NAMES_BY_PIPELINE_TYPE = {
    "512": (
        "sparse_structure_decoder",
        "sparse_structure_flow_model",
        "shape_slat_decoder",
        "shape_slat_flow_model_512",
        "tex_slat_decoder",
        "tex_slat_flow_model_512",
    ),
    "1024_cascade": (
        "sparse_structure_decoder",
        "sparse_structure_flow_model",
        "shape_slat_decoder",
        "shape_slat_flow_model_512",
        "shape_slat_flow_model_1024",
        "tex_slat_decoder",
        "tex_slat_flow_model_1024",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--conditioning", default="conditioning.npz", type=Path)
    parser.add_argument("--conditioning-1024", type=Path)
    parser.add_argument("--source-tar", default="trellis2_source_tarball.bin", type=Path)
    parser.add_argument("--model-repo", default="microsoft/TRELLIS.2-4B")
    parser.add_argument("--pipeline-config", default="pipeline.json")
    parser.add_argument("--pipeline-type", default="512")
    parser.add_argument("--steps", default=8, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max-num-tokens", default=49152, type=int)
    parser.add_argument(
        "--sparse-conv-backend",
        default="none",
        choices=("none", "spconv", "torchsparse", "flex_gemm"),
        help=(
            "Official source sparse convolution backend. The default avoids "
            "Kaggle's missing flex_gemm extension while preserving source code."
        ),
    )
    parser.add_argument(
        "--sparse-attn-backend",
        default="sdpa",
        choices=("xformers", "flash_attn", "flash_attn_3", "sdpa", "naive"),
        help="Official source sparse attention backend.",
    )
    return parser


def required_model_names(pipeline_type: str) -> tuple[str, ...]:
    try:
        return MODEL_NAMES_BY_PIPELINE_TYPE[pipeline_type]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_NAMES_BY_PIPELINE_TYPE))
        raise ValueError(f"unsupported pipeline_type {pipeline_type!r}; expected one of {supported}") from exc


def resolve_model_ref(model_repo: str, model_spec: str) -> str:
    if model_spec.startswith("ckpts/"):
        return f"{model_repo}/{model_spec}"
    return model_spec


def apply_sparse_backend_env(conv_backend: str, attn_backend: str) -> dict[str, str]:
    os.environ["SPARSE_CONV_BACKEND"] = conv_backend
    os.environ["SPARSE_ATTN_BACKEND"] = attn_backend
    os.environ["ATTN_BACKEND"] = attn_backend
    return {
        "SPARSE_CONV_BACKEND": conv_backend,
        "SPARSE_ATTN_BACKEND": attn_backend,
        "ATTN_BACKEND": attn_backend,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema": "trellis2mlx.source_cuda_postcond_full_decode_timing.v1",
        "status": "failed",
        "failure_phase": None,
        "requested_backend": "official TRELLIS.2 source post-conditioning decode on CUDA",
        "pipeline_type": args.pipeline_type,
        "steps": int(args.steps),
        "seed": int(args.seed),
        "phase_timings": {},
        "model_loads": {},
        "forbidden_inferences": [
            "not image-encoder timing",
            "not rembg timing",
            "not full image-to-3d wall clock",
            "not GLB export timing unless export_glb_elapsed_seconds is present",
        ],
    }
    phase = "setup"

    try:
        output_json = Path(args.output_json)
        output_npz = Path(args.output_npz)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_npz.parent.mkdir(parents=True, exist_ok=True)

        phase = "validate_args"
        if args.steps <= 0:
            raise ValueError("--steps must be positive")
        model_names = required_model_names(args.pipeline_type)
        if args.pipeline_type != "512" and args.conditioning_1024 is None:
            raise ValueError(f"--conditioning-1024 is required for pipeline_type={args.pipeline_type}")
        report["required_model_names"] = list(model_names)
        report["requested_sparse_backend"] = apply_sparse_backend_env(
            args.sparse_conv_backend,
            args.sparse_attn_backend,
        )

        phase = "extract_source"
        phase_started = time.perf_counter()
        source_root = extract_source(Path(args.source_tar), Path.cwd())
        sys.path.insert(0, str(source_root))
        report["phase_timings"][phase] = elapsed(phase_started)
        report["source_root"] = str(source_root)

        phase = "import_runtime"
        phase_started = time.perf_counter()
        import torch
        from huggingface_hub import hf_hub_download

        from trellis2 import models as source_models
        from trellis2.pipelines import samplers
        from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
        from trellis2.modules.sparse import config as sparse_config

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        device = torch.device("cuda")
        torch.set_grad_enabled(False)
        report.update(
            {
                "torch": torch.__version__,
                "cuda_device": torch.cuda.get_device_name(0),
                "sparse_attention_backend": getattr(sparse_config, "ATTN", None),
                "sparse_conv_backend": getattr(sparse_config, "CONV", None),
            }
        )
        report["phase_timings"][phase] = elapsed(phase_started)

        phase = "load_pipeline_config"
        phase_started = time.perf_counter()
        pipeline_config_path = Path(hf_hub_download(args.model_repo, args.pipeline_config))
        with pipeline_config_path.open() as handle:
            pipeline_args = json.load(handle)["args"]
        report["pipeline_config"] = {
            "model_repo": args.model_repo,
            "pipeline_config": args.pipeline_config,
            "path": str(pipeline_config_path),
        }
        report["phase_timings"][phase] = elapsed(phase_started)

        phase = "load_models"
        phase_started = time.perf_counter()
        loaded_models = {}
        for name in model_names:
            model_ref = resolve_model_ref(args.model_repo, pipeline_args["models"][name])
            model_started = time.perf_counter()
            loaded_models[name] = source_models.from_pretrained(model_ref)
            report["model_loads"][name] = {
                "model_ref": model_ref,
                "elapsed_seconds": elapsed(model_started),
                "parameter_count": parameter_count(loaded_models[name]),
            }
        report["phase_timings"][phase] = elapsed(phase_started)
        report["model_download_and_load_elapsed_seconds"] = report["phase_timings"][phase]

        phase = "build_pipeline"
        phase_started = time.perf_counter()
        pipeline = Trellis2ImageTo3DPipeline(
            models=loaded_models,
            sparse_structure_sampler=getattr(samplers, pipeline_args["sparse_structure_sampler"]["name"])(
                **pipeline_args["sparse_structure_sampler"]["args"]
            ),
            shape_slat_sampler=getattr(samplers, pipeline_args["shape_slat_sampler"]["name"])(
                **pipeline_args["shape_slat_sampler"]["args"]
            ),
            tex_slat_sampler=getattr(samplers, pipeline_args["tex_slat_sampler"]["name"])(
                **pipeline_args["tex_slat_sampler"]["args"]
            ),
            sparse_structure_sampler_params=pipeline_args["sparse_structure_sampler"]["params"],
            shape_slat_sampler_params=pipeline_args["shape_slat_sampler"]["params"],
            tex_slat_sampler_params=pipeline_args["tex_slat_sampler"]["params"],
            shape_slat_normalization=pipeline_args["shape_slat_normalization"],
            tex_slat_normalization=pipeline_args["tex_slat_normalization"],
            image_cond_model=None,
            rembg_model=None,
            low_vram=True,
            default_pipeline_type=args.pipeline_type,
        )
        pipeline.to(device)
        report["phase_timings"][phase] = elapsed(phase_started)

        phase = "load_conditioning"
        phase_started = time.perf_counter()
        cond_512 = load_conditioning(Path(args.conditioning), device, torch)
        cond_1024 = (
            load_conditioning(Path(args.conditioning_1024), device, torch)
            if args.conditioning_1024 is not None
            else None
        )
        report["conditioning"] = {
            "cond_512": conditioning_identity(Path(args.conditioning), cond_512),
            "cond_1024": (
                conditioning_identity(Path(args.conditioning_1024), cond_1024)
                if args.conditioning_1024 is not None and cond_1024 is not None
                else None
            ),
        }
        report["phase_timings"][phase] = elapsed(phase_started)

        phase = "post_conditioning_decode"
        decode_started = time.perf_counter()
        torch.manual_seed(args.seed)
        stage_timings: dict[str, float] = {}

        stage_started = time.perf_counter()
        coords = pipeline.sample_sparse_structure(
            cond_512,
            32,
            1,
            {"steps": args.steps},
        )
        sync_cuda(torch)
        stage_timings["sample_sparse_structure_elapsed_seconds"] = elapsed(stage_started)

        if args.pipeline_type == "512":
            stage_started = time.perf_counter()
            shape_slat = pipeline.sample_shape_slat(
                cond_512,
                pipeline.models["shape_slat_flow_model_512"],
                coords,
                {"steps": args.steps},
            )
            sync_cuda(torch)
            stage_timings["sample_shape_slat_elapsed_seconds"] = elapsed(stage_started)

            stage_started = time.perf_counter()
            tex_slat = pipeline.sample_tex_slat(
                cond_512,
                pipeline.models["tex_slat_flow_model_512"],
                shape_slat,
                {"steps": args.steps},
            )
            sync_cuda(torch)
            stage_timings["sample_tex_slat_elapsed_seconds"] = elapsed(stage_started)
            resolution = 512
        elif args.pipeline_type == "1024_cascade":
            assert cond_1024 is not None
            stage_started = time.perf_counter()
            shape_slat, resolution = pipeline.sample_shape_slat_cascade(
                cond_512,
                cond_1024,
                pipeline.models["shape_slat_flow_model_512"],
                pipeline.models["shape_slat_flow_model_1024"],
                512,
                1024,
                coords,
                {"steps": args.steps},
                args.max_num_tokens,
            )
            sync_cuda(torch)
            stage_timings["sample_shape_slat_elapsed_seconds"] = elapsed(stage_started)

            stage_started = time.perf_counter()
            tex_slat = pipeline.sample_tex_slat(
                cond_1024,
                pipeline.models["tex_slat_flow_model_1024"],
                shape_slat,
                {"steps": args.steps},
            )
            sync_cuda(torch)
            stage_timings["sample_tex_slat_elapsed_seconds"] = elapsed(stage_started)
        else:  # pragma: no cover - guarded by required_model_names
            raise AssertionError(args.pipeline_type)

        stage_started = time.perf_counter()
        meshes = pipeline.decode_latent(shape_slat, tex_slat, resolution)
        sync_cuda(torch)
        stage_timings["decode_latent_elapsed_seconds"] = elapsed(stage_started)

        report["stage_timings"] = stage_timings
        report["post_conditioning_decode_elapsed_seconds"] = elapsed(decode_started)
        report["mesh_summary"] = [mesh_summary(mesh) for mesh in meshes]
        report["sparse_coords_count"] = int(coords.shape[0])
        report["shape_slat_summary"] = sparse_tensor_summary(shape_slat)
        report["tex_slat_summary"] = sparse_tensor_summary(tex_slat)
        report["resolution"] = int(resolution)

        phase = "write_outputs"
        np.savez(
            output_npz,
            sparse_coords_count=np.asarray(report["sparse_coords_count"], dtype=np.int64),
            resolution=np.asarray(report["resolution"], dtype=np.int64),
            post_conditioning_decode_elapsed_seconds=np.asarray(
                report["post_conditioning_decode_elapsed_seconds"],
                dtype=np.float64,
            ),
            mesh_vertices=np.asarray(
                [entry["vertices"] for entry in report["mesh_summary"]],
                dtype=np.int64,
            ),
            mesh_faces=np.asarray(
                [entry["faces"] for entry in report["mesh_summary"]],
                dtype=np.int64,
            ),
        )
        report.update(
            {
                "status": "done",
                "failure_phase": None,
                "elapsed_seconds": elapsed(started),
                "output_npz": str(output_npz),
            }
        )
        output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return 0
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "failure_phase": phase,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": elapsed(started),
            }
        )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return 1


def extract_source(source_tar: Path, base: Path) -> Path:
    source_tree = base / "trellis2_source"
    if source_tree.is_dir():
        return source_tree
    if not source_tar.is_file():
        alternate = base / "trellis2_source.tar.gz"
        if alternate.is_file():
            source_tar = alternate
        else:
            raise FileNotFoundError(source_tar)
    target = base / "source"
    with tarfile.open(source_tar, "r:gz") as tf:
        tf.extractall(target)
    return target


def load_conditioning(path: Path, device: Any, torch_module: Any) -> dict[str, Any]:
    with np.load(path) as data:
        cond = torch_module.from_numpy(np.asarray(data["cond"], dtype=np.float32)).to(
            device=device,
            dtype=torch_module.float32,
        )
        neg_cond = torch_module.from_numpy(np.asarray(data["neg_cond"], dtype=np.float32)).to(
            device=device,
            dtype=torch_module.float32,
        )
    return {"cond": cond, "neg_cond": neg_cond}


def conditioning_identity(path: Path, cond: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "cond_shape": [int(v) for v in cond["cond"].shape],
        "neg_cond_shape": [int(v) for v in cond["neg_cond"].shape],
    }


def parameter_count(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def sparse_tensor_summary(value: Any) -> dict[str, Any]:
    feats = value.feats.detach()
    coords = value.coords.detach()
    return {
        "coords_shape": [int(v) for v in coords.shape],
        "feats_shape": [int(v) for v in feats.shape],
        "feats_mean": float(feats.float().mean().cpu()),
        "feats_std": float(feats.float().std().cpu()),
    }


def mesh_summary(mesh: Any) -> dict[str, Any]:
    attrs = getattr(mesh, "attrs", None)
    coords = getattr(mesh, "coords", None)
    return {
        "vertices": int(mesh.vertices.shape[0]),
        "faces": int(mesh.faces.shape[0]),
        "attrs": int(attrs.shape[0]) if attrs is not None else None,
        "coords": int(coords.shape[0]) if coords is not None else None,
    }


def sync_cuda(torch_module: Any) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def elapsed(started: float) -> float:
    return max(0.0, time.perf_counter() - started)


if __name__ == "__main__":
    raise SystemExit(main())
