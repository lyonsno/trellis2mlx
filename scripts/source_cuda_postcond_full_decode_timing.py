"""Time official TRELLIS.2 source CUDA decode from saved conditioning.

This is a post-conditioning timing witness. It intentionally does not instantiate
the source DINO/rembg path because DINOv3 is a gated dependency and the existing
diagnostic surface already has durable conditioning tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import tempfile
import time
import traceback
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def load_decoder_level0_trace_contract():
    if not __package__:
        contract_path = Path(__file__).resolve().with_name(
            "decoder_level0_trace_contract.py"
        )
        if not contract_path.is_file():
            raise ModuleNotFoundError(
                "standalone decoder trace runner requires adjacent contract "
                f"{contract_path}",
                name="decoder_level0_trace_contract",
            )
        spec = importlib.util.spec_from_file_location(
            "decoder_level0_trace_contract",
            contract_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(
                f"cannot load decoder level-zero trace contract from {contract_path}"
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    for module_name in (
        "scripts.decoder_level0_trace_contract",
        "decoder_level0_trace_contract",
    ):
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name not in {module_name, module_name.split(".", 1)[0]}:
                raise
    raise ModuleNotFoundError("decoder_level0_trace_contract")


def load_decoder_level1_trace_contract():
    if not __package__:
        contract_path = Path(__file__).resolve().with_name(
            "decoder_level1_trace_contract.py"
        )
        if not contract_path.is_file():
            raise ModuleNotFoundError(
                "standalone decoder trace runner requires adjacent contract "
                f"{contract_path}",
                name="decoder_level1_trace_contract",
            )
        spec = importlib.util.spec_from_file_location(
            "decoder_level1_trace_contract",
            contract_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(
                f"cannot load decoder level-one trace contract from {contract_path}"
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    for module_name in (
        "scripts.decoder_level1_trace_contract",
        "decoder_level1_trace_contract",
    ):
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name not in {module_name, module_name.split(".", 1)[0]}:
                raise
    raise ModuleNotFoundError("decoder_level1_trace_contract")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-npz", type=Path)
    parser.add_argument("--output-ply", default=Path("cuda_result_mesh.ply"), type=Path)
    parser.add_argument("--output-mesh-state", type=Path)
    parser.add_argument("--output-shape-slat", type=Path)
    parser.add_argument("--output-shape-flow-step", type=Path)
    parser.add_argument(
        "--shape-flow-noise-sample",
        type=Path,
        help=(
            "Optional NPZ containing shape-flow first-step noise/sample_feats "
            "and coords. Coords must exactly match the source support so CUDA "
            "and MLX first-step captures can share the same input tensor."
        ),
    )
    parser.add_argument("--conditioning", default="conditioning.npz", type=Path)
    parser.add_argument("--conditioning-1024", type=Path)
    parser.add_argument("--source-tar", default="trellis2_source_tarball.bin", type=Path)
    parser.add_argument("--mesh-override", default=Path("o_voxel_override_convert.py"), type=Path)
    parser.add_argument("--model-repo", default="microsoft/TRELLIS.2-4B")
    parser.add_argument("--pipeline-config", default="pipeline.json")
    parser.add_argument("--pipeline-type", default="512")
    parser.add_argument(
        "--shape-slat-grid",
        type=Path,
        help="Admitted block29 basin NPZ to decode directly with the official shape decoder.",
    )
    parser.add_argument(
        "--shape-slat-grid-report",
        type=Path,
        help="Authoritative source-CUDA basin report bound to --shape-slat-grid.",
    )
    parser.add_argument(
        "--shape-slat-grid-sha256",
        help=(
            "Expected SHA256 of --shape-slat-grid. Required for suffix-ladder "
            "selective decode so the caller binds the accepted NPZ before cleanup."
        ),
    )
    parser.add_argument(
        "--shape-slat-grid-report-sha256",
        help=(
            "Expected SHA256 of --shape-slat-grid-report. Required for "
            "suffix-ladder selective decode."
        ),
    )
    parser.add_argument(
        "--shape-slat-point",
        action="append",
        default=[],
        help="Coordinate key to decode, such as alpha-1_beta-1. Repeat for every point.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for selective raw and hole-filled PLY artifacts.",
    )
    parser.add_argument(
        "--decoder-state-only",
        action="store_true",
        help=(
            "Run the official shape decoder superclass in eval mode and emit "
            "raw sparse decoder state without mesh conversion."
        ),
    )
    parser.add_argument(
        "--decoder-level0-trace",
        action="store_true",
        help=(
            "Capture exact operation boundaries through the official shape "
            "decoder's first level without subdivision or mesh conversion."
        ),
    )
    parser.add_argument(
        "--decoder-level1-trace",
        action="store_true",
        help=(
            "Capture the official shape decoder's first channel-to-spatial "
            "upsample and level-one block 0 without mesh conversion."
        ),
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Validate selective-decode inputs and stale-output handling, then stop before model download.",
    )
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


def install_mesh_override(source_root: Path, override_path: Path) -> dict[str, Any]:
    override_path = Path(override_path)
    if not override_path.is_file():
        return {"status": "missing", "source": str(override_path)}
    destination = Path(source_root) / "stubs" / "o_voxel_override_convert.py"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(override_path, destination)
    return {
        "status": "installed",
        "source": str(override_path),
        "path": str(destination),
        "sha256": sha256_file(destination),
        "size_bytes": destination.stat().st_size,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if any(
        (
            args.shape_slat_grid is not None,
            args.shape_slat_grid_report is not None,
            bool(args.shape_slat_point),
            args.output_dir is not None,
            args.no_download,
        )
    ):
        return run_shape_slat_grid_decode(args)
    started = time.perf_counter()
    decode_started: float | None = None
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
        if args.output_npz is None:
            raise ValueError("--output-npz is required for full post-conditioning decode")
        output_npz = Path(args.output_npz)
        output_ply = Path(args.output_ply)
        output_mesh_state = Path(args.output_mesh_state) if args.output_mesh_state is not None else None
        output_shape_slat = Path(args.output_shape_slat) if args.output_shape_slat is not None else None
        output_shape_flow_step = (
            Path(args.output_shape_flow_step)
            if args.output_shape_flow_step is not None
            else None
        )
        shape_flow_noise_sample_path = (
            Path(args.shape_flow_noise_sample)
            if args.shape_flow_noise_sample is not None
            else None
        )
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        output_ply.parent.mkdir(parents=True, exist_ok=True)
        if output_mesh_state is not None:
            output_mesh_state.parent.mkdir(parents=True, exist_ok=True)
        if output_shape_slat is not None:
            output_shape_slat.parent.mkdir(parents=True, exist_ok=True)
        if output_shape_flow_step is not None:
            output_shape_flow_step.parent.mkdir(parents=True, exist_ok=True)

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
        report["mesh_override"] = install_mesh_override(source_root, Path(args.mesh_override))
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
        report["stage_timings"] = {}
        stage_timings = report["stage_timings"]
        shape_flow_step_payload = None
        shape_flow_noise_sample = None
        shape_flow_noise_key = None

        stage_started = time.perf_counter()
        coords = pipeline.sample_sparse_structure(
            cond_512,
            32,
            1,
            {"steps": args.steps},
        )
        sync_cuda(torch)
        stage_timings["sample_sparse_structure_elapsed_seconds"] = elapsed(stage_started)
        report["post_conditioning_partial_elapsed_seconds"] = elapsed(decode_started)

        if args.pipeline_type == "512":
            if output_shape_flow_step is not None:
                stage_started = time.perf_counter()
                if shape_flow_noise_sample_path is not None:
                    shape_flow_noise_sample, shape_flow_noise_key = load_shape_flow_noise_sample(
                        shape_flow_noise_sample_path,
                        tensor_to_numpy(coords).astype(np.int32, copy=False),
                    )
                    report["shape_flow_noise_sample"] = {
                        "path": str(shape_flow_noise_sample_path),
                        "sha256": sha256_file(shape_flow_noise_sample_path),
                        "coords_shape": [int(v) for v in tensor_to_numpy(coords).shape],
                        "noise_shape": [int(v) for v in shape_flow_noise_sample.shape],
                        "noise_key": shape_flow_noise_key,
                    }
                with torch.random.fork_rng(devices=[device.index or 0]):
                    shape_flow_step_payload = capture_source_shape_flow_first_step(
                        pipeline,
                        cond_512,
                        pipeline.models["shape_slat_flow_model_512"],
                        coords,
                        {"steps": args.steps},
                        noise_feats=shape_flow_noise_sample,
                    )
                sync_cuda(torch)
                stage_timings["capture_shape_flow_first_step_elapsed_seconds"] = elapsed(stage_started)

            stage_started = time.perf_counter()
            shape_slat = pipeline.sample_shape_slat(
                cond_512,
                pipeline.models["shape_slat_flow_model_512"],
                coords,
                {"steps": args.steps},
            )
            sync_cuda(torch)
            stage_timings["sample_shape_slat_elapsed_seconds"] = elapsed(stage_started)
            report["post_conditioning_partial_elapsed_seconds"] = elapsed(decode_started)

            stage_started = time.perf_counter()
            tex_slat = pipeline.sample_tex_slat(
                cond_512,
                pipeline.models["tex_slat_flow_model_512"],
                shape_slat,
                {"steps": args.steps},
            )
            sync_cuda(torch)
            stage_timings["sample_tex_slat_elapsed_seconds"] = elapsed(stage_started)
            report["post_conditioning_partial_elapsed_seconds"] = elapsed(decode_started)
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
            report["post_conditioning_partial_elapsed_seconds"] = elapsed(decode_started)

            stage_started = time.perf_counter()
            tex_slat = pipeline.sample_tex_slat(
                cond_1024,
                pipeline.models["tex_slat_flow_model_1024"],
                shape_slat,
                {"steps": args.steps},
            )
            sync_cuda(torch)
            stage_timings["sample_tex_slat_elapsed_seconds"] = elapsed(stage_started)
            report["post_conditioning_partial_elapsed_seconds"] = elapsed(decode_started)
        else:  # pragma: no cover - guarded by required_model_names
            raise AssertionError(args.pipeline_type)

        stage_started = time.perf_counter()
        meshes = pipeline.decode_latent(shape_slat, tex_slat, resolution)
        sync_cuda(torch)
        stage_timings["decode_latent_elapsed_seconds"] = elapsed(stage_started)

        report["post_conditioning_decode_elapsed_seconds"] = elapsed(decode_started)
        report["mesh_summary"] = [mesh_summary(mesh) for mesh in meshes]
        report["sparse_coords_count"] = int(coords.shape[0])
        report["shape_slat_summary"] = sparse_tensor_summary(shape_slat)
        report["tex_slat_summary"] = sparse_tensor_summary(tex_slat)
        report["resolution"] = int(resolution)

        phase = "write_outputs"
        mesh_artifacts = []
        if meshes:
            write_binary_mesh_ply(output_ply, meshes[0])
            mesh_artifacts.append(
                {
                    "path": str(output_ply),
                    "sha256": sha256_file(output_ply),
                    "size_bytes": output_ply.stat().st_size,
                    "format": "binary_little_endian_ply",
                    "mesh_index": 0,
                }
            )
            report["output_ply"] = str(output_ply)
            if output_mesh_state is not None:
                write_mesh_state_npz(output_mesh_state, meshes[0])
                mesh_artifacts.append(
                    {
                        "path": str(output_mesh_state),
                        "sha256": sha256_file(output_mesh_state),
                        "size_bytes": output_mesh_state.stat().st_size,
                        "format": "mesh_with_voxel_state_npz",
                        "mesh_index": 0,
                        "artifact_scope": "post_decode_mesh_with_voxel_state_for_source_finalization",
                    }
                )
                report["output_mesh_state"] = str(output_mesh_state)
        report["mesh_artifacts"] = mesh_artifacts
        if output_shape_flow_step is not None and shape_flow_step_payload is not None:
            report["shape_flow_step_artifact"] = write_source_shape_flow_step_npz(
                output_shape_flow_step,
                shape_flow_step_payload,
                normalization=pipeline_args["shape_slat_normalization"],
            )
        if output_shape_slat is not None:
            report["shape_slat_artifact"] = write_sparse_tensor_npz(
                output_shape_slat,
                shape_slat,
                stage="shape_slat",
                normalization=pipeline_args["shape_slat_normalization"],
            )

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
                "traceback": traceback.format_exc(),
                "elapsed_seconds": elapsed(started),
            }
        )
        if decode_started is not None:
            report["post_conditioning_partial_elapsed_seconds"] = elapsed(decode_started)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return 1


SHAPE_SLAT_POINT_RE = re.compile(
    r"^(?:alpha-(?:0|0p5|1)_beta-(?:0|0p5|1)|switch-[0-8])$"
)
SHAPE_SLAT_BASIN_ROUTE = {
    "route": "official-source-cuda-full-eight-step-shape-flow-with-fixed-block29-endpoints",
    "device_type": "cuda",
    "attention_backend": "sdpa",
    "conv_backend": "none",
    "block_index": 29,
    "step_index": 0,
    "steps": 8,
    "one_model_load": True,
    "endpoint_semantics": "current + scale * (source - current)",
}
SHAPE_SLAT_SUFFIX_ROUTE = {
    "route": "official-source-cuda-shape-flow-suffix-ladder-from-exact-mlx-prefixes",
    "device_type": "cuda",
    "attention_backend": "sdpa",
    "conv_backend": "none",
    "steps": 8,
    "switch_steps": list(range(9)),
    "one_model_load": True,
    "comparison_class": "exact-mlx-prefix-plus-source-cuda-suffix",
}


def _resolved_path(path: Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _existing_file_identity(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_dev, stat.st_ino


def _path_collides(path: Path, protected: tuple[Path, ...]) -> bool:
    resolved = _resolved_path(path)
    identity = _existing_file_identity(path)
    for candidate in protected:
        if resolved == _resolved_path(candidate):
            return True
        candidate_identity = _existing_file_identity(candidate)
        if identity is not None and identity == candidate_identity:
            return True
    return False


def _selective_failure_report_path(
    requested: Path,
    protected: tuple[Path, ...],
) -> Path:
    candidate = requested.with_name(f"{requested.name}.selective-decode-failure.json")
    index = 2
    while _path_collides(candidate, protected):
        candidate = requested.with_name(
            f"{requested.name}.selective-decode-failure-{index}.json"
        )
        index += 1
    return candidate


def _write_selective_decode_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _validate_source_shape_slat_route(source_report: dict[str, Any]) -> dict[str, Any]:
    schema = source_report.get("schema")
    supported_schemas = {
        "trellis2mlx.source_cuda_shape_block29_basin_map.v1",
        "trellis2mlx.source_cuda_shape_flow_suffix_ladder.v1",
    }
    if schema not in supported_schemas:
        raise ValueError("unsupported source shape-SLat report schema")
    if source_report.get("status") != "done":
        raise ValueError("source shape-SLat report is not done")
    route = source_report.get("effective_route")
    if not isinstance(route, dict):
        raise ValueError("source shape-SLat report is missing effective_route")
    expected_route = (
        SHAPE_SLAT_BASIN_ROUTE
        if schema == "trellis2mlx.source_cuda_shape_block29_basin_map.v1"
        else SHAPE_SLAT_SUFFIX_ROUTE
    )
    route_label = (
        "source basin"
        if schema == "trellis2mlx.source_cuda_shape_block29_basin_map.v1"
        else "source suffix"
    )
    for key, expected in expected_route.items():
        if route.get(key) != expected:
            raise ValueError(
                f"source shape-SLat route mismatch for {key}: "
                f"expected {expected!r}, got {route.get(key)!r}"
            )
    if not isinstance(route.get("cuda_device"), str) or not route["cuda_device"].strip():
        raise ValueError(f"{route_label} route is missing cuda_device")
    if schema == "trellis2mlx.source_cuda_shape_block29_basin_map.v1":
        source_control = source_report.get("source_control")
        if not isinstance(source_control, dict):
            raise ValueError("source basin report is missing source_control")
        if source_control.get("exact") is not True:
            raise ValueError("source basin report did not prove an exact source control")
        if source_control.get("coordinate") != {"alpha": 1.0, "beta": 1.0}:
            raise ValueError("source control coordinate mismatch")
    else:
        timing = source_report.get("timing")
        if not isinstance(timing, dict):
            raise ValueError("source suffix report is missing timing")
        expected_timing = {
            "source_steps_completed": 36,
            "source_steps_requested": 36,
            "switch_points_completed": 9,
            "switch_points_requested": 9,
        }
        for key, expected in expected_timing.items():
            if timing.get(key) != expected:
                raise ValueError(
                    f"source suffix {key} must be {expected}, got {timing.get(key)!r}"
                )
        primary = source_report.get("primary_output", {})
        validation = primary.get("validation", {}) if isinstance(primary, dict) else {}
        if validation.get("point_arrays_bound") is not True:
            raise ValueError("source suffix primary output does not bind point arrays")
        if validation.get("switch_count") != 9:
            raise ValueError("source suffix primary output does not bind all nine switches")
    return {
        "source_schema": schema,
        **{key: route[key] for key in (*expected_route, "cuda_device") if key in route},
    }


def _coordinate_from_point_name(point_name: str) -> dict[str, float]:
    alpha_token, beta_token = point_name.removeprefix("alpha-").split("_beta-", 1)
    values = {"0": 0.0, "0p5": 0.5, "1": 1.0}
    return {"alpha": values[alpha_token], "beta": values[beta_token]}


def _load_selected_shape_slat_inputs(
    grid_path: Path,
    source_report: dict[str, Any],
    point_names: list[str],
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict[str, Any]]]:
    primary = source_report.get("primary_output")
    if not isinstance(primary, dict):
        raise ValueError("source shape-SLat report is missing primary_output")
    actual_primary_sha = sha256_file(grid_path)
    if primary.get("sha256") != actual_primary_sha:
        raise ValueError(
            "primary output digest mismatch: "
            f"report={primary.get('sha256')!r}, actual={actual_primary_sha!r}"
        )
    if primary.get("size_bytes") != grid_path.stat().st_size:
        raise ValueError("primary output size mismatch")
    primary_keys = primary.get("keys")
    if not isinstance(primary_keys, list) or not all(isinstance(key, str) for key in primary_keys):
        raise ValueError("primary output keys must be a string list")
    if "coords" not in primary_keys:
        raise ValueError("primary output key list omits coords")

    point_rows = source_report.get("points")
    if not isinstance(point_rows, list):
        raise ValueError("source shape-SLat report points must be a list")
    schema = source_report.get("schema")
    by_name: dict[str, dict[str, Any]] = {}
    for row in point_rows:
        if not isinstance(row, dict):
            raise ValueError("source shape-SLat report contains an invalid point row")
        if schema == "trellis2mlx.source_cuda_shape_block29_basin_map.v1":
            name = row.get("coordinate_key")
            if not isinstance(name, str):
                raise ValueError("source basin report contains an invalid point row")
        else:
            step = row.get("switch_step")
            if not isinstance(step, int) or step not in range(9):
                raise ValueError("source suffix report contains an invalid switch row")
            name = f"switch-{step}"
            if row.get("source_step_indices") != list(range(step, 8)):
                raise ValueError(f"source suffix switch {step} has invalid source step indices")
            if row.get("source_step_count") != 8 - step:
                raise ValueError(f"source suffix switch {step} has invalid source step count")
            if row.get("output_key") != f"switch_{step}_shape_slat":
                raise ValueError(f"source suffix switch {step} has noncanonical output key")
        if name in by_name:
            raise ValueError(f"source shape-SLat report contains duplicate point {name!r}")
        by_name[name] = row
    expected_names = (
        {f"switch-{step}" for step in range(9)}
        if schema == "trellis2mlx.source_cuda_shape_flow_suffix_ladder.v1"
        else None
    )
    if expected_names is not None and set(by_name) != expected_names:
        raise ValueError("source suffix report does not contain exactly all nine switches")

    selected_arrays: dict[str, np.ndarray] = {}
    selected_rows: list[dict[str, Any]] = []
    with np.load(grid_path, allow_pickle=False) as data:
        if "coords" not in data.files:
            raise ValueError("shape-SLat grid is missing coords")
        coords = np.asarray(data["coords"])
        if coords.dtype != np.int32 or coords.ndim != 2 or coords.shape[1] != 4:
            raise ValueError(f"coords must be int32 [N, 4], got {coords.dtype} {coords.shape}")
        coords = np.ascontiguousarray(coords)
        point_names_to_validate = point_names
        if schema == "trellis2mlx.source_cuda_shape_flow_suffix_ladder.v1":
            if set(primary_keys) != set(data.files):
                missing = sorted(set(primary_keys) - set(data.files))
                unreported = sorted(set(data.files) - set(primary_keys))
                raise ValueError(
                    "primary output key manifest differs from archive: "
                    f"missing={missing}, unreported={unreported}"
                )
            canonical_output_keys = {
                f"switch_{step}_shape_slat" for step in range(9)
            }
            for output_key in sorted(canonical_output_keys):
                if output_key not in data.files:
                    raise ValueError(
                        f"source suffix is missing canonical suffix array {output_key!r}"
                    )
            if "switch_steps" not in data.files:
                raise ValueError("source suffix is missing switch_steps")
            switch_steps = np.asarray(data["switch_steps"])
            if (
                switch_steps.dtype != np.int32
                or switch_steps.shape != (9,)
                or not np.array_equal(switch_steps, np.arange(9, dtype=np.int32))
            ):
                raise ValueError("source suffix switch_steps must be int32 [0..8]")
            if "metadata_json" not in data.files:
                raise ValueError("source suffix is missing metadata_json")
            raw_metadata = np.asarray(data["metadata_json"])
            if raw_metadata.shape != () or raw_metadata.dtype.kind not in {"U", "S"}:
                raise ValueError("source suffix metadata_json must be a string scalar")
            try:
                metadata = json.loads(str(raw_metadata.item()))
            except json.JSONDecodeError as exc:
                raise ValueError("source suffix metadata_json is invalid JSON") from exc
            if (
                metadata.get("schema")
                != "trellis2mlx.source_cuda_shape_flow_suffix_ladder.artifact.v1"
            ):
                raise ValueError("source suffix artifact metadata schema is invalid")
            if metadata.get("artifact_status") != "computed_pending_serialization":
                raise ValueError("source suffix artifact metadata status is invalid")
            if metadata.get("external_report_required") is not True:
                raise ValueError(
                    "source suffix artifact metadata must require the external report"
                )
            metadata_fields = (
                "effective_route",
                "inputs",
                "points",
                "pairwise",
                "timing",
                "forbidden_inferences",
            )
            for field in metadata_fields:
                if metadata.get(field) != source_report.get(field):
                    label = "point manifest" if field == "points" else field
                    raise ValueError(
                        f"source suffix artifact {label} differs from external report"
                    )
            point_names_to_validate = [
                f"switch-{step}" for step in range(9)
            ]

        validated_arrays: dict[str, np.ndarray] = {}
        validated_rows: dict[str, dict[str, Any]] = {}
        for point_name in point_names_to_validate:
            row = by_name.get(point_name)
            if row is None:
                raise ValueError(
                    f"selected point {point_name!r} is absent from source shape-SLat report"
                )
            output_key = row.get("output_key")
            if not isinstance(output_key, str) or output_key not in data.files:
                raise ValueError(f"selected point {point_name!r} is missing output array {output_key!r}")
            if output_key not in primary_keys:
                raise ValueError(
                    f"primary output key list omits selected array {output_key!r}"
                )
            if schema == "trellis2mlx.source_cuda_shape_block29_basin_map.v1":
                expected_coordinate = _coordinate_from_point_name(point_name)
                if row.get("coordinate") != expected_coordinate:
                    raise ValueError(
                        f"coordinate mismatch for {point_name!r}: "
                        f"expected {expected_coordinate!r}, got {row.get('coordinate')!r}"
                    )
            values = np.asarray(data[output_key])
            if values.dtype != np.float32 or values.ndim != 2 or values.shape != (coords.shape[0], 32):
                raise ValueError(
                    f"selected array {output_key!r} must be float32 [N, 32], got {values.dtype} {values.shape}"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"selected array {output_key!r} contains non-finite values")
            actual_digest = hashlib.sha256(values.tobytes()).hexdigest()
            if row.get("sha256") != actual_digest:
                raise ValueError(
                    f"selected array digest mismatch for {point_name!r}: "
                    f"report={row.get('sha256')!r}, actual={actual_digest!r}"
                )
            if row.get("shape") != [int(v) for v in values.shape]:
                raise ValueError(f"selected array shape report mismatch for {point_name!r}")
            validated_arrays[point_name] = np.ascontiguousarray(values)
            validated_row = {
                "coordinate_key": point_name,
                "output_key": output_key,
                "sha256": actual_digest,
                "shape": [int(v) for v in values.shape],
            }
            if schema == "trellis2mlx.source_cuda_shape_block29_basin_map.v1":
                validated_row["coordinate"] = row.get("coordinate")
            else:
                validated_row["switch_step"] = row.get("switch_step")
            validated_rows[point_name] = validated_row
        for point_name in point_names:
            selected_arrays[point_name] = validated_arrays[point_name]
            selected_rows.append(validated_rows[point_name])
    return coords, selected_arrays, selected_rows


def run_shape_slat_grid_decode(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    decoder_state_only = bool(args.decoder_state_only)
    decoder_level0_trace = bool(args.decoder_level0_trace)
    decoder_level1_trace = bool(args.decoder_level1_trace)
    decoder_trace_mode = decoder_level0_trace or decoder_level1_trace
    route_name = (
        "official-source-cuda-shape-decoder-level1-trace"
        if decoder_level1_trace
        else "official-source-cuda-shape-decoder-level0-trace"
        if decoder_level0_trace
        else "official-source-cuda-shape-slat-decoder-raw-state"
        if decoder_state_only
        else "official-source-cuda-shape-slat-decoder"
    )
    requested_output_json = Path(args.output_json)
    grid_path = Path(args.shape_slat_grid) if args.shape_slat_grid is not None else None
    source_report_path = (
        Path(args.shape_slat_grid_report)
        if args.shape_slat_grid_report is not None
        else None
    )
    protected_paths = tuple(
        path
        for path in (grid_path, source_report_path, Path(args.source_tar), Path(args.mesh_override))
        if path is not None
    )
    output_json = requested_output_json
    collision = _path_collides(requested_output_json, protected_paths)
    if collision:
        output_json = _selective_failure_report_path(
            requested_output_json,
            protected_paths,
        )

    report: dict[str, Any] = {
        "schema": "trellis2mlx.source_cuda_shape_slat_grid_decode.v1",
        "status": "failed",
        "failure_phase": None,
        "last_trustworthy_phase": "request_received",
        "requested_output_json": str(requested_output_json),
        "effective_output_json": str(output_json),
        "requested_route": {
            "route": route_name,
            "resolution": 512,
            "decoder_state_only": decoder_state_only,
            "decoder_level0_trace": decoder_level0_trace,
            "decoder_level1_trace": decoder_level1_trace,
            "raw_meshes": not decoder_state_only and not decoder_trace_mode,
            "post_fill_holes_snapshots": (
                not decoder_state_only and not decoder_trace_mode
            ),
            "mesh_conversion": not decoder_state_only and not decoder_trace_mode,
            "one_model_load": True,
            "no_download": bool(args.no_download),
        },
        "requested_source_shape_slat_identity": {
            "report_sha256": args.shape_slat_grid_report_sha256,
            "primary_sha256": args.shape_slat_grid_sha256,
        },
        "mesh_artifacts": [],
        "decoder_state_artifacts": [],
        "decoder_trace_artifacts": [],
        "written_artifact_count": 0,
        "forbidden_inferences": [
            "not a textured mesh or GLB",
            "not a winding-correctness claim",
            "a post-fill_holes snapshot is not proof that geometry changed",
            "not evidence that every exact tensor quotient is a distinct visual basin",
            "not a full MLX continuation",
        ],
    }
    phase = "request_validation"

    try:
        if collision:
            raise ValueError("--output-json collides with protected input")
        if sum(
            (
                decoder_state_only,
                decoder_level0_trace,
                decoder_level1_trace,
            )
        ) > 1:
            raise ValueError(
                "--decoder-state-only, --decoder-level0-trace, and "
                "--decoder-level1-trace are mutually exclusive"
            )
        if grid_path is None:
            raise ValueError("--shape-slat-grid is required for selective decode")
        if source_report_path is None:
            raise ValueError("--shape-slat-grid-report is required for selective decode")
        if args.output_dir is None:
            raise ValueError("--output-dir is required for selective decode")
        point_names = list(args.shape_slat_point)
        if not point_names:
            raise ValueError("at least one --shape-slat-point is required")
        invalid_names = [name for name in point_names if not SHAPE_SLAT_POINT_RE.fullmatch(name)]
        if invalid_names:
            raise ValueError(f"invalid --shape-slat-point values: {invalid_names!r}")
        if len(set(point_names)) != len(point_names):
            raise ValueError("duplicate --shape-slat-point values are not allowed")

        output_dir = Path(args.output_dir)
        if decoder_level1_trace:
            expected_paths = [
                output_dir / f"{point_name}.decoder-level1-trace.npz"
                for point_name in point_names
            ]
        elif decoder_level0_trace:
            expected_paths = [
                output_dir / f"{point_name}.decoder-level0-trace.npz"
                for point_name in point_names
            ]
        elif decoder_state_only:
            expected_paths = [
                output_dir / f"{point_name}.decoder-state.npz"
                for point_name in point_names
            ]
        else:
            expected_paths = [
                output_dir / f"{point_name}.{variant}.ply"
                for point_name in point_names
                for variant in ("raw", "filled")
            ]
        protected_output_collisions = sorted(
            str(path)
            for path in expected_paths
            if _path_collides(path, protected_paths)
        )
        if protected_output_collisions:
            raise ValueError(
                "expected mesh output collides with protected input: "
                f"{protected_output_collisions}"
            )
        expected_output_collision = _path_collides(
            requested_output_json,
            tuple(expected_paths),
        )
        if expected_output_collision:
            output_json = _selective_failure_report_path(
                requested_output_json,
                protected_paths + tuple(expected_paths),
            )
            report["effective_output_json"] = str(output_json)

        expected_grid_sha256 = args.shape_slat_grid_sha256
        expected_report_sha256 = args.shape_slat_grid_report_sha256
        suffix_selection = any(name.startswith("switch-") for name in point_names)
        if suffix_selection and not (
            expected_grid_sha256 and expected_report_sha256
        ):
            raise ValueError(
                "expected report and NPZ SHA256 values are required for "
                "suffix selective decode"
            )
        if bool(expected_grid_sha256) != bool(expected_report_sha256):
            raise ValueError(
                "--shape-slat-grid-sha256 and "
                "--shape-slat-grid-report-sha256 must be provided together"
            )
        if expected_grid_sha256 and expected_report_sha256:
            for label, expected in (
                ("primary", expected_grid_sha256),
                ("report", expected_report_sha256),
            ):
                if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
                    raise ValueError(
                        f"expected source shape-SLat {label} SHA256 is not "
                        "canonical lowercase hex"
                    )
            actual_report_sha256 = sha256_file(source_report_path)
            if actual_report_sha256 != expected_report_sha256:
                raise ValueError(
                    "expected source shape-SLat report digest mismatch: "
                    f"expected={expected_report_sha256}, "
                    f"actual={actual_report_sha256}"
                )
            actual_grid_sha256 = sha256_file(grid_path)
            if actual_grid_sha256 != expected_grid_sha256:
                raise ValueError(
                    "expected source shape-SLat primary digest mismatch: "
                    f"expected={expected_grid_sha256}, actual={actual_grid_sha256}"
                )

        output_dir.mkdir(parents=True, exist_ok=True)
        for path in expected_paths:
            path.unlink(missing_ok=True)
        report["selected_point_names"] = point_names
        report["expected_artifact_count"] = len(expected_paths)
        if decoder_trace_mode:
            trace_level = 1 if decoder_level1_trace else 0
            report["decoder_trace_artifacts"] = [
                {
                    "coordinate_key": point_name,
                    "path": str(
                        output_dir
                        / f"{point_name}.decoder-level{trace_level}-trace.npz"
                    ),
                    "status": "not_written",
                }
                for point_name in point_names
            ]
        elif decoder_state_only:
            report["decoder_state_artifacts"] = [
                {
                    "coordinate_key": point_name,
                    "path": str(output_dir / f"{point_name}.decoder-state.npz"),
                    "status": "not_written",
                }
                for point_name in point_names
            ]
        else:
            report["mesh_artifacts"] = [
                {
                    "coordinate_key": point_name,
                    "variant": variant,
                    "path": str(output_dir / f"{point_name}.{variant}.ply"),
                    "status": "not_written",
                }
                for point_name in point_names
                for variant in ("raw", "filled")
            ]
        if expected_output_collision:
            output_kind = (
                "decoder-level1-trace"
                if decoder_level1_trace
                else "decoder-level0-trace"
                if decoder_level0_trace
                else "decoder-state"
                if decoder_state_only
                else "mesh"
            )
            raise ValueError(
                f"--output-json collides with an expected {output_kind} output"
            )

        phase = "input_validation"
        source_report = json.loads(source_report_path.read_text())
        source_shape_slat_route = _validate_source_shape_slat_route(source_report)
        coords, selected_arrays, selected_rows = _load_selected_shape_slat_inputs(
            grid_path,
            source_report,
            point_names,
        )
        source_shape_slat_report = {
            "path": str(source_report_path),
            "sha256": sha256_file(source_report_path),
        }
        source_shape_slat_primary = {
            "path": str(grid_path),
            "sha256": sha256_file(grid_path),
            "size_bytes": grid_path.stat().st_size,
        }
        report.update(
            {
                "source_shape_slat_report": source_shape_slat_report,
                "source_shape_slat_primary": source_shape_slat_primary,
                "source_shape_slat_route": source_shape_slat_route,
                "selected_points": selected_rows,
                "coords_shape": [int(v) for v in coords.shape],
                "last_trustworthy_phase": phase,
            }
        )
        if source_report.get("schema") == "trellis2mlx.source_cuda_shape_block29_basin_map.v1":
            report.update(
                {
                    "source_basin_report": source_shape_slat_report,
                    "source_basin_primary": source_shape_slat_primary,
                    "source_basin_route": source_shape_slat_route,
                }
            )

        if args.no_download:
            for artifact in (
                report["mesh_artifacts"]
                + report["decoder_state_artifacts"]
                + report["decoder_trace_artifacts"]
            ):
                artifact["status"] = "not_written_no_download"
            report.update(
                {
                    "status": "preflight_stopped",
                    "failure_phase": None,
                    "effective_route": {
                        "route": route_name,
                        "device_type": "not_loaded_no_download",
                        "resolution": 512,
                        "decoder_state_only": decoder_state_only,
                        "decoder_level0_trace": decoder_level0_trace,
                        "decoder_level1_trace": decoder_level1_trace,
                        "raw_meshes": (
                            not decoder_state_only and not decoder_trace_mode
                        ),
                        "post_fill_holes_snapshots": (
                            not decoder_state_only and not decoder_trace_mode
                        ),
                        "mesh_conversion": (
                            not decoder_state_only and not decoder_trace_mode
                        ),
                        "one_model_load": True,
                    },
                    "elapsed_seconds": elapsed(started),
                }
            )
            _write_selective_decode_report(output_json, report)
            return 0

        phase = "extract_source"
        phase_started = time.perf_counter()
        source_root = extract_source(Path(args.source_tar), Path.cwd())
        sys.path.insert(0, str(source_root))
        report["mesh_override"] = install_mesh_override(source_root, Path(args.mesh_override))
        report["source_root"] = str(source_root)
        report.setdefault("phase_timings", {})[phase] = elapsed(phase_started)

        phase = "import_runtime"
        phase_started = time.perf_counter()
        report["requested_sparse_backend"] = apply_sparse_backend_env(
            args.sparse_conv_backend,
            args.sparse_attn_backend,
        )
        import torch
        from huggingface_hub import hf_hub_download

        from trellis2 import models as source_models
        from trellis2.modules.sparse import SparseTensor
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
        pipeline_args = json.loads(pipeline_config_path.read_text())["args"]
        model_ref = resolve_model_ref(
            args.model_repo,
            pipeline_args["models"]["shape_slat_decoder"],
        )
        report["phase_timings"][phase] = elapsed(phase_started)

        phase = "load_shape_decoder"
        phase_started = time.perf_counter()
        decoder = source_models.from_pretrained(model_ref)
        training_before_eval = bool(decoder.training)
        decoder.eval()
        decoder.set_resolution(512)
        decoder.to(device)
        decoder.low_vram = True
        report["model_load"] = {
            "name": "shape_slat_decoder",
            "model_ref": model_ref,
            "parameter_count": parameter_count(decoder),
            "elapsed_seconds": elapsed(phase_started),
            "training_before_eval": training_before_eval,
            "training": bool(decoder.training),
        }
        report["phase_timings"][phase] = report["model_load"]["elapsed_seconds"]
        report["last_trustworthy_phase"] = phase
        if decoder.training:
            raise RuntimeError("shape-SLat decoder remained in training mode after eval()")

        phase = "decode_selected_points"
        decode_started = time.perf_counter()
        written_artifacts: list[dict[str, Any]] = []
        point_results: list[dict[str, Any]] = []
        coords_tensor = torch.from_numpy(coords.copy()).to(device=device)
        artifact_by_key = {
            (row["coordinate_key"], row["variant"]): row
            for row in report["mesh_artifacts"]
        }
        decoder_artifact_by_key = {
            row["coordinate_key"]: row
            for row in report["decoder_state_artifacts"]
        }
        decoder_trace_artifact_by_key = {
            row["coordinate_key"]: row
            for row in report["decoder_trace_artifacts"]
        }
        for point_name in point_names:
            point_started = time.perf_counter()
            feats_tensor = torch.from_numpy(selected_arrays[point_name].copy()).to(device=device)
            shape_slat = SparseTensor(feats=feats_tensor, coords=coords_tensor)
            if decoder_trace_mode:
                trace_contract = (
                    load_decoder_level1_trace_contract()
                    if decoder_level1_trace
                    else load_decoder_level0_trace_contract()
                )
                with torch.no_grad():
                    trace_result = (
                        capture_source_decoder_level1_trace(
                            decoder,
                            shape_slat,
                            trace_contract=trace_contract,
                        )
                        if decoder_level1_trace
                        else capture_source_decoder_level0_trace(
                            decoder,
                            shape_slat,
                        )
                    )
                sync_cuda(torch)
                trace_level = 1 if decoder_level1_trace else 0
                trace_path = (
                    output_dir
                    / f"{point_name}.decoder-level{trace_level}-trace.npz"
                )
                if decoder_level1_trace:
                    trace_arrays, hash_entries = trace_result
                    hash_ledger = (
                        trace_contract.validate_decoder_level1_hash_ledger(
                            {
                                "schema": (
                                    trace_contract.LEVEL1_HASH_LEDGER_SCHEMA
                                ),
                                "entries": hash_entries,
                            }
                        )
                    )
                    validation = trace_contract.write_decoder_level1_trace_npz(
                        trace_path,
                        trace_arrays,
                        parent_channels=1024,
                        child_channels=512,
                        torso_dtype=np.float16,
                    )
                    input_tensor_sha256 = (
                        trace_contract.decoder_level1_trace_input_sha256(
                            trace_arrays["level0_output"],
                            trace_arrays["parent_coords"],
                        )
                    )
                else:
                    trace_arrays = trace_result
                    hash_ledger = None
                    validation = trace_contract.write_decoder_level0_trace_npz(
                        trace_path,
                        trace_arrays,
                        latent_channels=32,
                        channels=1024,
                        torso_dtype=np.float16,
                    )
                    input_tensor_sha256 = (
                        trace_contract.decoder_trace_input_sha256(
                            selected_arrays[point_name],
                            coords,
                        )
                    )
                trace_artifact = decoder_trace_artifact_by_key[point_name]
                trace_artifact.update(
                    {
                        "status": "written",
                        "sha256": sha256_file(trace_path),
                        "size_bytes": trace_path.stat().st_size,
                        "input_tensor_sha256": input_tensor_sha256,
                        "validation": validation,
                    }
                )
                if hash_ledger is not None:
                    trace_artifact["hash_ledger"] = hash_ledger
                written_artifacts.append(trace_artifact)
                report["written_artifact_count"] = len(written_artifacts)
                point_results.append(
                    {
                        "coordinate_key": point_name,
                        "elapsed_seconds": elapsed(point_started),
                    }
                )
                continue
            if decoder_state_only:
                with torch.no_grad():
                    decoded, subs = decode_shape_slat_raw(decoder, shape_slat)
                sync_cuda(torch)
                state_path = output_dir / f"{point_name}.decoder-state.npz"
                validation = write_decoder_state_npz(
                    state_path,
                    decoded,
                    subs,
                )
                state_artifact = decoder_artifact_by_key[point_name]
                state_artifact.update(
                    {
                        "status": "written",
                        "sha256": sha256_file(state_path),
                        "size_bytes": state_path.stat().st_size,
                        "validation": validation,
                    }
                )
                written_artifacts.append(state_artifact)
                report["written_artifact_count"] = len(written_artifacts)
                point_results.append(
                    {
                        "coordinate_key": point_name,
                        "elapsed_seconds": elapsed(point_started),
                    }
                )
                continue
            with torch.no_grad():
                meshes, _subs = decoder(shape_slat, return_subs=True)
            sync_cuda(torch)
            if len(meshes) != 1:
                raise ValueError(f"expected one decoded mesh for {point_name!r}, got {len(meshes)}")
            mesh = meshes[0]
            raw_path = output_dir / f"{point_name}.raw.ply"
            write_binary_mesh_ply(raw_path, mesh)
            raw_artifact = artifact_by_key[(point_name, "raw")]
            raw_artifact.update(
                {
                    "status": "written",
                    "sha256": sha256_file(raw_path),
                    "size_bytes": raw_path.stat().st_size,
                    "mesh_summary": mesh_summary(mesh),
                }
            )
            written_artifacts.append(raw_artifact)
            report["written_artifact_count"] = len(written_artifacts)
            raw_sha256 = raw_artifact["sha256"]

            fill_started = time.perf_counter()
            mesh.fill_holes()
            sync_cuda(torch)
            filled_path = output_dir / f"{point_name}.filled.ply"
            write_binary_mesh_ply(filled_path, mesh)
            filled_sha256 = sha256_file(filled_path)
            fill_holes_effective_change = filled_sha256 != raw_sha256
            filled_artifact = artifact_by_key[(point_name, "filled")]
            filled_artifact.update(
                {
                    "status": "written",
                    "sha256": filled_sha256,
                    "size_bytes": filled_path.stat().st_size,
                    "mesh_summary": mesh_summary(mesh),
                    "fill_holes_effective_change": fill_holes_effective_change,
                }
            )
            written_artifacts.append(filled_artifact)
            report["written_artifact_count"] = len(written_artifacts)
            point_results.append(
                {
                    "coordinate_key": point_name,
                    "elapsed_seconds": elapsed(point_started),
                    "fill_holes_elapsed_seconds": elapsed(fill_started),
                    "fill_holes_effective_change": fill_holes_effective_change,
                }
            )
        report["decode_selected_points_elapsed_seconds"] = elapsed(decode_started)
        report["point_results"] = point_results
        if len(written_artifacts) != len(expected_paths):
            raise RuntimeError(
                f"partial output: wrote {len(written_artifacts)} of {len(expected_paths)} expected artifacts"
            )

        effective_route = {
            "route": route_name,
            "device_type": "cuda",
            "cuda_device": report["cuda_device"],
            "sparse_attention_backend": report["sparse_attention_backend"],
            "sparse_conv_backend": report["sparse_conv_backend"],
            "model_ref": model_ref,
            "model_training": bool(decoder.training),
            "resolution": 512,
            "decoder_state_only": decoder_state_only,
            "decoder_level0_trace": decoder_level0_trace,
            "decoder_level1_trace": decoder_level1_trace,
            "raw_meshes": not decoder_state_only and not decoder_trace_mode,
            "post_fill_holes_snapshots": (
                not decoder_state_only and not decoder_trace_mode
            ),
            "mesh_conversion": not decoder_state_only and not decoder_trace_mode,
            "one_model_load": True,
        }
        if not decoder_state_only and not decoder_trace_mode:
            effective_route["fill_holes_effective_change_count"] = sum(
                bool(point["fill_holes_effective_change"])
                for point in point_results
            )
        report.update(
            {
                "status": "done",
                "failure_phase": None,
                "last_trustworthy_phase": (
                    "all_selected_decoder_traces_written"
                    if decoder_trace_mode
                    else
                    "all_selected_decoder_states_written"
                    if decoder_state_only
                    else "all_selected_meshes_written"
                ),
                "effective_route": effective_route,
                "elapsed_seconds": elapsed(started),
            }
        )
        _write_selective_decode_report(output_json, report)
        return 0
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "failure_phase": phase,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": elapsed(started),
            }
        )
        _write_selective_decode_report(output_json, report)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sparse_tensor_summary(value: Any) -> dict[str, Any]:
    feats = value.feats.detach()
    coords = value.coords.detach()
    return {
        "coords_shape": [int(v) for v in coords.shape],
        "feats_shape": [int(v) for v in feats.shape],
        "feats_mean": float(feats.float().mean().cpu()),
        "feats_std": float(feats.float().std().cpu()),
    }


def capture_source_decoder_level0_trace(
    decoder: Any,
    shape_slat: Any,
) -> dict[str, np.ndarray]:
    from trellis2.models.sc_vaes.sparse_unet_vae import (
        SparseConvNeXtBlock3d,
    )

    level = decoder.blocks[0]
    convnext_blocks = [
        block for block in level if isinstance(block, SparseConvNeXtBlock3d)
    ]
    if len(convnext_blocks) != 4:
        raise ValueError(
            "source level-zero trace requires exactly four SparseConvNeXt "
            f"blocks, got {len(convnext_blocks)}"
        )
    upsample_blocks = [
        block for block in level if hasattr(block, "to_subdiv")
    ]
    if len(upsample_blocks) != 1:
        raise ValueError(
            "source level-zero trace requires exactly one subdivision head, "
            f"got {len(upsample_blocks)}"
        )

    projected_fp32 = decoder.from_latent(shape_slat)
    torso_input = projected_fp32.type(decoder.dtype)
    block_arrays = {}
    current = torso_input
    for block_index, block in enumerate(convnext_blocks):
        block_input = current
        conv_state = block.conv(block_input)
        norm = block.norm(conv_state.feats)
        mlp_fc1 = block.mlp[0](norm)
        silu = block.mlp[1](mlp_fc1)
        mlp_fc2 = block.mlp[2](silu)
        output = block_input.replace(mlp_fc2 + block_input.feats)
        natural = block(block_input)
        if not np.array_equal(
            tensor_to_numpy(output.feats),
            tensor_to_numpy(natural.feats),
        ):
            raise RuntimeError(
                "manual source level-zero block trace does not exactly "
                f"reproduce natural forward for block {block_index}"
            )
        if not np.array_equal(
            tensor_to_numpy(output.coords),
            tensor_to_numpy(natural.coords),
        ):
            raise RuntimeError(
                "manual source level-zero block trace changed sparse "
                f"coordinates for block {block_index}"
            )
        block_arrays.update(
            {
                f"block{block_index}_conv": tensor_to_numpy(conv_state.feats),
                f"block{block_index}_norm": tensor_to_numpy(norm),
                f"block{block_index}_mlp_fc1": tensor_to_numpy(mlp_fc1),
                f"block{block_index}_silu": tensor_to_numpy(silu),
                f"block{block_index}_mlp_fc2": tensor_to_numpy(mlp_fc2),
                f"block{block_index}_output": tensor_to_numpy(natural.feats),
            }
        )
        current = natural
    level0_subdiv = upsample_blocks[0].to_subdiv(current)

    arrays = {
        "coords": tensor_to_numpy(shape_slat.coords),
        "input_feats": tensor_to_numpy(shape_slat.feats),
        "from_latent_fp32": tensor_to_numpy(projected_fp32.feats),
        "torso_input": tensor_to_numpy(torso_input.feats),
        **block_arrays,
        "level0_subdiv_logits": tensor_to_numpy(level0_subdiv.feats),
    }
    return {
        name: np.ascontiguousarray(values)
        for name, values in arrays.items()
    }


def capture_source_decoder_level1_trace(
    decoder: Any,
    shape_slat: Any,
    *,
    trace_contract: Any,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    import torch.nn.functional as torch_functional

    from trellis2.models.sc_vaes.sparse_unet_vae import (
        SparseConvNeXtBlock3d,
        SparseResBlockC2S3d,
    )

    level0 = decoder.blocks[0]
    level0_blocks = [
        block for block in level0 if isinstance(block, SparseConvNeXtBlock3d)
    ]
    level0_upsample = [
        block for block in level0 if isinstance(block, SparseResBlockC2S3d)
    ]
    level1_blocks = [
        block
        for block in decoder.blocks[1]
        if isinstance(block, SparseConvNeXtBlock3d)
    ]
    level1_upsample = [
        block
        for block in decoder.blocks[1]
        if isinstance(block, SparseResBlockC2S3d)
    ]
    level2_blocks = [
        block
        for block in decoder.blocks[2]
        if isinstance(block, SparseConvNeXtBlock3d)
    ]
    level2_upsample = [
        block
        for block in decoder.blocks[2]
        if isinstance(block, SparseResBlockC2S3d)
    ]
    if len(level0_blocks) != 4 or len(level0_upsample) != 1:
        raise ValueError(
            "source level-one trace requires four level-zero ConvNeXt blocks "
            f"and one upsample, got {len(level0_blocks)} and "
            f"{len(level0_upsample)}"
        )
    if len(level1_blocks) != 16:
        raise ValueError(
            "source level-one trace requires sixteen level-one ConvNeXt "
            f"blocks, got {len(level1_blocks)}"
        )
    if len(level1_upsample) != 1:
        raise ValueError(
            "source level-one trace requires one level-one upsample, "
            f"got {len(level1_upsample)}"
        )
    if len(level2_blocks) != 8:
        raise ValueError(
            "source level-one trace requires eight level-two ConvNeXt "
            f"blocks, got {len(level2_blocks)}"
        )
    if len(level2_upsample) != 1:
        raise ValueError(
            "source level-one trace requires one level-two upsample, "
            f"got {len(level2_upsample)}"
        )

    current = decoder.from_latent(shape_slat).type(decoder.dtype)
    for block in level0_blocks:
        current = block(current)
    level0_output = current
    upsample = level0_upsample[0]

    subdiv = upsample.to_subdiv(level0_output)
    norm1 = upsample.norm1(level0_output.feats)
    silu1 = torch_functional.silu(norm1)
    conv1 = upsample.conv1(level0_output.replace(silu1))
    subdiv_binarized = subdiv.replace(subdiv.feats > 0)
    h_c2s = upsample.updown(conv1, subdiv_binarized)
    skip_c2s = upsample.updown(level0_output, subdiv_binarized)
    norm2 = upsample.norm2(h_c2s.feats)
    silu2 = torch_functional.silu(norm2)
    conv2 = upsample.conv2(h_c2s.replace(silu2))
    skip_repeated = upsample.skip_connection(skip_c2s)
    upsample_output = conv2 + skip_repeated

    natural_output, natural_subdiv = upsample(level0_output)
    for name, manual, natural in (
        ("features", upsample_output.feats, natural_output.feats),
        ("coordinates", upsample_output.coords, natural_output.coords),
        ("subdivision logits", subdiv.feats, natural_subdiv.feats),
    ):
        if not np.array_equal(
            tensor_to_numpy(manual),
            tensor_to_numpy(natural),
        ):
            raise RuntimeError(
                "manual source first-upsample trace does not exactly "
                f"reproduce natural {name}"
            )
    if not np.array_equal(
        tensor_to_numpy(h_c2s.coords),
        tensor_to_numpy(skip_c2s.coords),
    ):
        raise RuntimeError(
            "source upsample feature and skip coordinates differ"
        )

    block0 = level1_blocks[0]
    block0_conv_state = block0.conv(upsample_output)
    block0_norm = block0.norm(block0_conv_state.feats)
    block0_fc1 = block0.mlp[0](block0_norm)
    block0_silu = block0.mlp[1](block0_fc1)
    block0_fc2 = block0.mlp[2](block0_silu)
    block0_output = upsample_output.replace(
        block0_fc2 + upsample_output.feats
    )
    natural_block0 = block0(upsample_output)
    for name, manual, natural in (
        ("features", block0_output.feats, natural_block0.feats),
        ("coordinates", block0_output.coords, natural_block0.coords),
    ):
        if not np.array_equal(
            tensor_to_numpy(manual),
            tensor_to_numpy(natural),
        ):
            raise RuntimeError(
                "manual source level-one block-0 trace does not exactly "
                f"reproduce natural {name}"
            )
    hash_entries = [
        trace_contract.decoder_level1_hash_entry(
            "level1_block0_output",
            tensor_to_numpy(natural_block0.feats),
        )
    ]
    level1_output = natural_block0
    for index, block in enumerate(level1_blocks[1:], start=1):
        level1_output = block(level1_output)
        hash_entries.append(
            trace_contract.decoder_level1_hash_entry(
                f"level1_block{index}_output",
                tensor_to_numpy(level1_output.feats),
            )
        )

    next_upsample = level1_upsample[0]
    next_subdiv = next_upsample.to_subdiv(level1_output)
    next_norm1 = next_upsample.norm1(level1_output.feats)
    next_silu1 = torch_functional.silu(next_norm1)
    next_conv1 = next_upsample.conv1(level1_output.replace(next_silu1))
    next_subdiv_binarized = next_subdiv.replace(next_subdiv.feats > 0)
    next_h_c2s = next_upsample.updown(next_conv1, next_subdiv_binarized)
    next_skip_c2s = next_upsample.updown(
        level1_output,
        next_subdiv_binarized,
    )
    next_norm2 = next_upsample.norm2(next_h_c2s.feats)
    next_silu2 = torch_functional.silu(next_norm2)
    next_conv2 = next_upsample.conv2(next_h_c2s.replace(next_silu2))
    next_skip_repeated = next_upsample.skip_connection(next_skip_c2s)
    next_upsample_output = next_conv2 + next_skip_repeated
    natural_next_output, natural_next_subdiv = next_upsample(level1_output)
    for name, manual, natural in (
        ("features", next_upsample_output.feats, natural_next_output.feats),
        (
            "coordinates",
            next_upsample_output.coords,
            natural_next_output.coords,
        ),
        ("subdivision logits", next_subdiv.feats, natural_next_subdiv.feats),
    ):
        if not np.array_equal(
            tensor_to_numpy(manual),
            tensor_to_numpy(natural),
        ):
            raise RuntimeError(
                "manual source second-upsample trace does not exactly "
                f"reproduce natural {name}"
            )
    if not np.array_equal(
        tensor_to_numpy(next_h_c2s.coords),
        tensor_to_numpy(next_skip_c2s.coords),
    ):
        raise RuntimeError(
            "source second-upsample feature and skip coordinates differ"
        )
    next_boundaries = {
        "level1_upsample_subdiv_logits": next_subdiv.feats,
        "level1_upsample_norm1": next_norm1,
        "level1_upsample_silu1": next_silu1,
        "level1_upsample_conv1": next_conv1.feats,
        "level2_child_coords": natural_next_output.coords,
        "level1_upsample_h_c2s": next_h_c2s.feats,
        "level1_upsample_skip_c2s": next_skip_c2s.feats,
        "level1_upsample_skip_repeated": next_skip_repeated.feats,
        "level1_upsample_norm2": next_norm2,
        "level1_upsample_silu2": next_silu2,
        "level1_upsample_conv2": next_conv2.feats,
        "level1_upsample_output": natural_next_output.feats,
    }
    hash_entries.extend(
        trace_contract.decoder_level1_hash_entry(
            name,
            tensor_to_numpy(values),
        )
        for name, values in next_boundaries.items()
    )

    level2_output = natural_next_output
    for index, block in enumerate(level2_blocks):
        level2_output = block(level2_output)
        hash_entries.append(
            trace_contract.decoder_level1_hash_entry(
                f"level2_block{index}_output",
                tensor_to_numpy(level2_output.feats),
            )
        )

    final_upsample = level2_upsample[0]
    final_subdiv = final_upsample.to_subdiv(level2_output)
    final_norm1 = final_upsample.norm1(level2_output.feats)
    final_silu1 = torch_functional.silu(final_norm1)
    final_conv1 = final_upsample.conv1(
        level2_output.replace(final_silu1)
    )
    final_subdiv_binarized = final_subdiv.replace(final_subdiv.feats > 0)
    final_h_c2s = final_upsample.updown(
        final_conv1,
        final_subdiv_binarized,
    )
    final_skip_c2s = final_upsample.updown(
        level2_output,
        final_subdiv_binarized,
    )
    final_norm2 = final_upsample.norm2(final_h_c2s.feats)
    final_silu2 = torch_functional.silu(final_norm2)
    final_conv2 = final_upsample.conv2(final_h_c2s.replace(final_silu2))
    final_skip_repeated = final_upsample.skip_connection(final_skip_c2s)
    final_upsample_output = final_conv2 + final_skip_repeated
    natural_final_output, natural_final_subdiv = final_upsample(level2_output)
    for name, manual, natural in (
        (
            "features",
            final_upsample_output.feats,
            natural_final_output.feats,
        ),
        (
            "coordinates",
            final_upsample_output.coords,
            natural_final_output.coords,
        ),
        (
            "subdivision logits",
            final_subdiv.feats,
            natural_final_subdiv.feats,
        ),
    ):
        if not np.array_equal(
            tensor_to_numpy(manual),
            tensor_to_numpy(natural),
        ):
            raise RuntimeError(
                "manual source third-upsample trace does not exactly "
                f"reproduce natural {name}"
            )
    if not np.array_equal(
        tensor_to_numpy(final_h_c2s.coords),
        tensor_to_numpy(final_skip_c2s.coords),
    ):
        raise RuntimeError(
            "source third-upsample feature and skip coordinates differ"
        )
    final_boundaries = {
        "level2_upsample_subdiv_logits": final_subdiv.feats,
        "level2_upsample_norm1": final_norm1,
        "level2_upsample_silu1": final_silu1,
        "level2_upsample_conv1": final_conv1.feats,
        "level3_child_coords": natural_final_output.coords,
        "level2_upsample_h_c2s": final_h_c2s.feats,
        "level2_upsample_skip_c2s": final_skip_c2s.feats,
        "level2_upsample_skip_repeated": final_skip_repeated.feats,
        "level2_upsample_norm2": final_norm2,
        "level2_upsample_silu2": final_silu2,
        "level2_upsample_conv2": final_conv2.feats,
        "level2_upsample_output": natural_final_output.feats,
    }
    hash_entries.extend(
        trace_contract.decoder_level1_hash_entry(
            name,
            tensor_to_numpy(values),
        )
        for name, values in final_boundaries.items()
    )

    arrays = {
        "parent_coords": tensor_to_numpy(level0_output.coords),
        "child_coords": tensor_to_numpy(natural_output.coords),
        "level0_output": tensor_to_numpy(level0_output.feats),
        "upsample_subdiv_logits": tensor_to_numpy(subdiv.feats),
        "upsample_norm1": tensor_to_numpy(norm1),
        "upsample_silu1": tensor_to_numpy(silu1),
        "upsample_conv1": tensor_to_numpy(conv1.feats),
        "upsample_h_c2s": tensor_to_numpy(h_c2s.feats),
        "upsample_skip_c2s": tensor_to_numpy(skip_c2s.feats),
        "upsample_skip_repeated": tensor_to_numpy(skip_repeated.feats),
        "upsample_norm2": tensor_to_numpy(norm2),
        "upsample_silu2": tensor_to_numpy(silu2),
        "upsample_conv2": tensor_to_numpy(conv2.feats),
        "upsample_output": tensor_to_numpy(natural_output.feats),
        "level1_block0_conv": tensor_to_numpy(block0_conv_state.feats),
        "level1_block0_norm": tensor_to_numpy(block0_norm),
        "level1_block0_mlp_fc1": tensor_to_numpy(block0_fc1),
        "level1_block0_silu": tensor_to_numpy(block0_silu),
        "level1_block0_mlp_fc2": tensor_to_numpy(block0_fc2),
        "level1_block0_output": tensor_to_numpy(natural_block0.feats),
    }
    trace = {
        name: np.ascontiguousarray(values)
        for name, values in arrays.items()
    }
    return trace, hash_entries


def decode_shape_slat_raw(decoder: Any, shape_slat: Any) -> tuple[Any, list[Any]]:
    from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder

    decoded = SparseUnetVaeDecoder.forward(
        decoder,
        shape_slat,
        return_subs=True,
    )
    if not isinstance(decoded, tuple) or len(decoded) != 2:
        raise ValueError("raw shape decoder must return (state, subdivisions)")
    state, subdivisions = decoded
    if not isinstance(subdivisions, list):
        raise ValueError("raw shape decoder subdivisions must be a list")
    return state, subdivisions


def write_decoder_state_npz(
    path: Path,
    state: Any,
    subdivisions: list[Any],
) -> dict[str, Any]:
    feats = tensor_to_numpy(state.feats)
    coords = tensor_to_numpy(state.coords)
    if feats.dtype != np.dtype(np.float32):
        raise ValueError(
            f"decoder-state feats must have dtype float32, got {feats.dtype}"
        )
    if coords.dtype != np.dtype(np.int32):
        raise ValueError(
            f"decoder-state coords must have dtype int32, got {coords.dtype}"
        )
    if feats.ndim != 2 or feats.shape[1] != 7:
        raise ValueError(f"decoder-state feats must have shape [N, 7], got {feats.shape}")
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"decoder-state coords must have shape [N, 4], got {coords.shape}")
    if feats.shape[0] == 0:
        raise ValueError("decoder-state output must be nonempty")
    if feats.shape[0] != coords.shape[0]:
        raise ValueError(
            "decoder-state feats and coords row counts must match, "
            f"got {feats.shape[0]} and {coords.shape[0]}"
        )
    if not np.isfinite(feats).all():
        raise ValueError("decoder-state feats contain non-finite values")
    if np.unique(coords, axis=0).shape[0] != coords.shape[0]:
        raise ValueError("decoder-state coords contain duplicates")
    if len(subdivisions) != 4:
        raise ValueError(
            "raw shape decoder must return exactly 4 subdivision levels, "
            f"got {len(subdivisions)}"
        )

    arrays: dict[str, np.ndarray] = {
        "feats": np.ascontiguousarray(feats),
        "coords": np.ascontiguousarray(coords),
    }
    subdivision_shapes: list[list[int]] = []
    subdivision_coordinate_shapes: list[list[int]] = []
    for index, subdivision in enumerate(subdivisions):
        if not hasattr(subdivision, "feats") or not hasattr(subdivision, "coords"):
            raise ValueError(
                f"decoder subdivision level {index} must carry feats and coords"
            )
        values = tensor_to_numpy(subdivision.feats)
        level_coords = tensor_to_numpy(subdivision.coords)
        if values.dtype != np.dtype(np.float16):
            raise ValueError(
                "decoder subdivision logits must have dtype float16 "
                f"at level {index}, got {values.dtype}"
            )
        if level_coords.dtype != np.dtype(np.int32):
            raise ValueError(
                "decoder subdivision coords must have dtype int32 "
                f"at level {index}, got {level_coords.dtype}"
            )
        if values.ndim != 2 or values.shape[1] != 8:
            raise ValueError(
                "decoder subdivision logits must have shape [N, 8], "
                f"got {values.shape} at level {index}"
            )
        if values.shape[0] == 0:
            raise ValueError(f"decoder subdivision level {index} must be nonempty")
        if level_coords.ndim != 2 or level_coords.shape[1] != 4:
            raise ValueError(
                "decoder subdivision coords must have shape [N, 4], "
                f"got {level_coords.shape} at level {index}"
            )
        if level_coords.shape[0] != values.shape[0]:
            raise ValueError(
                "decoder subdivision logits and coords row counts must match, "
                f"got {values.shape[0]} and {level_coords.shape[0]} at level {index}"
            )
        if not np.isfinite(values).all():
            raise ValueError(
                f"decoder subdivision logits contain non-finite values at level {index}"
            )
        if np.unique(level_coords, axis=0).shape[0] != level_coords.shape[0]:
            raise ValueError(
                f"decoder subdivision coords contain duplicates at level {index}"
            )
        arrays[f"shape_subs_{index}"] = np.ascontiguousarray(values)
        arrays[f"shape_subs_{index}_coords"] = np.ascontiguousarray(level_coords)
        subdivision_shapes.append([int(value) for value in values.shape])
        subdivision_coordinate_shapes.append(
            [int(value) for value in level_coords.shape]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp.npz",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        np.savez(temporary_path, **arrays)

        expected_keys = list(arrays)
        with np.load(temporary_path, allow_pickle=False) as reopened:
            if reopened.files != expected_keys:
                raise ValueError(
                    "decoder-state primary keys changed after write: "
                    f"expected={expected_keys}, actual={reopened.files}"
                )
            for key, expected in arrays.items():
                actual = np.asarray(reopened[key])
                if actual.dtype != expected.dtype or actual.shape != expected.shape:
                    raise ValueError(
                        f"decoder-state primary {key!r} changed dtype or shape after write"
                    )
                if not np.array_equal(actual, expected):
                    raise ValueError(
                        f"decoder-state primary {key!r} changed values after write"
                    )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return {
        "feats_shape": [int(value) for value in feats.shape],
        "feats_dtype": str(feats.dtype),
        "coords_shape": [int(value) for value in coords.shape],
        "coords_dtype": str(coords.dtype),
        "subdivision_shapes": subdivision_shapes,
        "subdivision_coordinate_shapes": subdivision_coordinate_shapes,
        "subdivision_dtypes": [
            {
                "logits": str(arrays[f"shape_subs_{index}"].dtype),
                "coords": str(arrays[f"shape_subs_{index}_coords"].dtype),
            }
            for index in range(len(subdivisions))
        ],
        "finite": True,
        "reopened_exact": True,
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


def write_binary_mesh_ply(path: Path, mesh: Any) -> None:
    vertices = tensor_to_numpy(mesh.vertices).astype("<f4", copy=False)
    faces = tensor_to_numpy(mesh.faces).astype("<i4", copy=False)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"mesh vertices must have shape [N, 3], got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"mesh faces must have shape [F, 3], got {faces.shape}")

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {vertices.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {faces.shape[0]}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    face_records = np.empty(
        faces.shape[0],
        dtype=np.dtype([("count", "u1"), ("indices", "<i4", (3,))]),
    )
    face_records["count"] = 3
    face_records["indices"] = faces

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(np.ascontiguousarray(vertices).tobytes())
        handle.write(face_records.tobytes())


def write_mesh_state_npz(path: Path, mesh: Any) -> None:
    vertices = tensor_to_numpy(mesh.vertices).astype(np.float32, copy=False)
    faces = tensor_to_numpy(mesh.faces).astype(np.int32, copy=False)
    attrs = getattr(mesh, "attrs", None)
    coords = getattr(mesh, "coords", None)
    if attrs is None:
        raise ValueError("mesh state export requires mesh.attrs")
    if coords is None:
        raise ValueError("mesh state export requires mesh.coords")
    attrs_array = tensor_to_numpy(attrs).astype(np.float32, copy=False)
    coords_array = tensor_to_numpy(coords).astype(np.int32, copy=False)
    origin = tensor_to_numpy(getattr(mesh, "origin")).astype(np.float32, copy=False)
    voxel_size = np.asarray(getattr(mesh, "voxel_size"), dtype=np.float64)
    voxel_shape = np.asarray(list(getattr(mesh, "voxel_shape")), dtype=np.int64)
    layout_json = json.dumps(serialize_layout(getattr(mesh, "layout", {})), sort_keys=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        vertices=np.ascontiguousarray(vertices),
        faces=np.ascontiguousarray(faces),
        attrs=np.ascontiguousarray(attrs_array),
        coords=np.ascontiguousarray(coords_array),
        origin=np.ascontiguousarray(origin),
        voxel_size=voxel_size,
        voxel_shape=voxel_shape,
        layout_json=np.asarray(layout_json),
    )


def write_sparse_tensor_npz(
    path: Path,
    value: Any,
    *,
    stage: str,
    normalization: str,
) -> dict[str, Any]:
    coords = tensor_to_numpy(value.coords).astype(np.int32, copy=False)
    feats = tensor_to_numpy(value.feats).astype(np.float32, copy=False)
    if coords.ndim != 2:
        raise ValueError(f"sparse tensor coords must have shape [N, D], got {coords.shape}")
    if feats.ndim != 2:
        raise ValueError(f"sparse tensor feats must have shape [N, C], got {feats.shape}")
    if coords.shape[0] != feats.shape[0]:
        raise ValueError(
            "sparse tensor coords/feats row mismatch: "
            f"{coords.shape[0]} coords vs {feats.shape[0]} feats"
        )
    metadata = {
        "artifact_scope": "source_cuda_shape_slat",
        "normalization": normalization,
        "stage": stage,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        coords=np.ascontiguousarray(coords),
        feats=np.ascontiguousarray(feats),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return {
        **metadata,
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "format": "sparse_tensor_npz",
        "coords_shape": [int(v) for v in coords.shape],
        "feats_shape": [int(v) for v in feats.shape],
    }


def capture_source_shape_flow_first_step(
    pipeline: Any,
    cond: dict[str, Any],
    flow_model: Any,
    coords: Any,
    sampler_params: dict[str, Any],
    *,
    noise_feats: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    from trellis2.modules.sparse import SparseTensor

    torch = __import__("torch")
    params = {**pipeline.shape_slat_sampler_params, **sampler_params}
    steps = int(params.get("steps", 50))
    if steps <= 0:
        raise ValueError("shape flow first-step capture requires positive steps")
    rescale_t = float(params.get("rescale_t", 1.0))
    guidance_strength = float(params.get("guidance_strength", 1.0))
    guidance_rescale = float(params.get("guidance_rescale", 0.0))
    guidance_interval = tuple(float(v) for v in params.get("guidance_interval", (0.0, 1.0)))
    if len(guidance_interval) != 2:
        raise ValueError(f"guidance_interval must have two values, got {guidance_interval!r}")

    if noise_feats is None:
        feats = torch.randn(coords.shape[0], flow_model.in_channels).to(pipeline.device)
    else:
        noise_array = np.asarray(noise_feats, dtype=np.float32)
        expected_shape = (int(coords.shape[0]), int(flow_model.in_channels))
        if noise_array.shape != expected_shape:
            raise ValueError(
                "shape flow noise tensor shape mismatch: "
                f"expected {expected_shape}, got {noise_array.shape}"
            )
        feats = torch.as_tensor(noise_array, device=pipeline.device, dtype=torch.float32)
    noise = SparseTensor(feats=feats, coords=coords)
    if pipeline.low_vram:
        flow_model.to(pipeline.device)
    try:
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t = float(t_seq[0])
        t_prev = float(t_seq[1])
        t_model = torch.tensor([1000.0 * t] * noise.shape[0], device=noise.device, dtype=torch.float32)
        pred_pos = flow_model(noise, t_model, cond["cond"])
        pred_neg = flow_model(noise, t_model, cond["neg_cond"])
        guidance_active = guidance_interval[0] <= t <= guidance_interval[1]
        if guidance_active:
            pred_cfg = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg
        else:
            pred_cfg = pred_pos
        x0_pos = pipeline.shape_slat_sampler._pred_to_xstart(noise, t, pred_pos)
        x0_cfg = pipeline.shape_slat_sampler._pred_to_xstart(noise, t, pred_cfg)
        std_pos = x0_pos.std(dim=list(range(1, x0_pos.ndim)), keepdim=True)
        std_cfg = x0_cfg.std(dim=list(range(1, x0_cfg.ndim)), keepdim=True)
        ratio_raw = std_pos / std_cfg
        x0_rescaled = x0_cfg * ratio_raw
        if guidance_active and guidance_rescale > 0:
            x0_after_rescale = guidance_rescale * x0_rescaled + (1 - guidance_rescale) * x0_cfg
            pred_final = pipeline.shape_slat_sampler._xstart_to_pred(noise, t, x0_after_rescale)
            ratio_effective = ratio_raw
        else:
            x0_after_rescale = x0_cfg
            pred_final = pred_cfg
            ratio_effective = torch.ones_like(ratio_raw)
        sample_next = noise - (t - t_prev) * pred_final
    finally:
        if pipeline.low_vram:
            flow_model.cpu()

    return {
        "noise": tensor_to_numpy(noise.feats).astype(np.float32, copy=False),
        "sample_feats": tensor_to_numpy(noise.feats).astype(np.float32, copy=False),
        "coords": tensor_to_numpy(noise.coords).astype(np.int32, copy=False),
        "coords_3d": tensor_to_numpy(noise.coords[:, 1:]).astype(np.int32, copy=False),
        "pred_pos": tensor_to_numpy(pred_pos.feats).astype(np.float32, copy=False),
        "pred_neg": tensor_to_numpy(pred_neg.feats).astype(np.float32, copy=False),
        "pred_cfg": tensor_to_numpy(pred_cfg.feats).astype(np.float32, copy=False),
        "x0_pos": tensor_to_numpy(x0_pos.feats).astype(np.float32, copy=False),
        "x0_cfg": tensor_to_numpy(x0_cfg.feats).astype(np.float32, copy=False),
        "std_pos": tensor_to_numpy(std_pos).astype(np.float32, copy=False).squeeze(),
        "std_cfg": tensor_to_numpy(std_cfg).astype(np.float32, copy=False).squeeze(),
        "ratio_raw": tensor_to_numpy(ratio_raw).astype(np.float32, copy=False).squeeze(),
        "std_ratio": tensor_to_numpy(ratio_raw).astype(np.float32, copy=False).squeeze(),
        "ratio_effective": tensor_to_numpy(ratio_effective).astype(np.float32, copy=False).squeeze(),
        "x0_rescaled": tensor_to_numpy(x0_rescaled.feats).astype(np.float32, copy=False),
        "x0_after_rescale": tensor_to_numpy(x0_after_rescale.feats).astype(np.float32, copy=False),
        "pred_final": tensor_to_numpy(pred_final.feats).astype(np.float32, copy=False),
        "pred_v_feats": tensor_to_numpy(pred_final.feats).astype(np.float32, copy=False),
        "sample_next": tensor_to_numpy(sample_next.feats).astype(np.float32, copy=False),
        "t": np.asarray(t, dtype=np.float32),
        "t_prev": np.asarray(t_prev, dtype=np.float32),
        "steps": np.asarray(steps, dtype=np.int32),
        "guidance_strength": np.asarray(guidance_strength, dtype=np.float32),
        "guidance_rescale": np.asarray(guidance_rescale, dtype=np.float32),
        "guidance_interval": np.asarray(guidance_interval, dtype=np.float32),
        "rescale_t": np.asarray(rescale_t, dtype=np.float32),
    }


def load_shape_flow_noise_sample(path: Path, expected_coords: np.ndarray) -> tuple[np.ndarray, str]:
    expected = np.asarray(expected_coords, dtype=np.int32)
    if expected.ndim != 2:
        raise ValueError(f"expected coords must have shape [N, D], got {expected.shape}")
    with np.load(path) as data:
        if "coords" not in data:
            raise ValueError(f"shape flow noise sample {path} missing coords")
        coords = np.asarray(data["coords"], dtype=np.int32)
        if "noise" in data:
            key = "noise"
        elif "sample_feats" in data:
            key = "sample_feats"
        else:
            raise ValueError(f"shape flow noise sample {path} missing noise/sample_feats")
        noise = np.asarray(data[key], dtype=np.float32)

    if coords.shape != expected.shape or not np.array_equal(coords, expected):
        raise ValueError(
            "shape flow noise sample coords do not exactly match source coords: "
            f"sample {coords.shape}, expected {expected.shape}"
        )
    if noise.ndim != 2:
        raise ValueError(f"shape flow noise must have shape [N, C], got {noise.shape}")
    if noise.shape[0] != expected.shape[0]:
        raise ValueError(
            "shape flow noise row mismatch: "
            f"{noise.shape[0]} noise rows vs {expected.shape[0]} coords"
        )
    return np.ascontiguousarray(noise), key


def write_source_shape_flow_step_npz(
    path: Path,
    payload: dict[str, Any],
    *,
    normalization: str,
) -> dict[str, Any]:
    required = {
        "noise",
        "sample_feats",
        "coords",
        "coords_3d",
        "pred_pos",
        "pred_neg",
        "pred_cfg",
        "x0_pos",
        "x0_cfg",
        "std_pos",
        "std_cfg",
        "ratio_raw",
        "std_ratio",
        "ratio_effective",
        "x0_rescaled",
        "x0_after_rescale",
        "pred_final",
        "pred_v_feats",
        "sample_next",
        "t",
        "t_prev",
        "steps",
        "guidance_strength",
        "guidance_rescale",
        "guidance_interval",
        "rescale_t",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"shape flow step payload missing keys: {missing}")
    arrays = {key: np.asarray(payload[key]) for key in sorted(required)}
    coords = arrays["coords"].astype(np.int32, copy=False)
    sample_next = arrays["sample_next"].astype(np.float32, copy=False)
    if coords.ndim != 2:
        raise ValueError(f"shape flow coords must have shape [N, D], got {coords.shape}")
    if sample_next.ndim != 2:
        raise ValueError(f"shape flow sample_next must have shape [N, C], got {sample_next.shape}")
    if coords.shape[0] != sample_next.shape[0]:
        raise ValueError(
            "shape flow coords/sample_next row mismatch: "
            f"{coords.shape[0]} coords vs {sample_next.shape[0]} sample_next rows"
        )
    metadata = {
        "artifact_scope": "source_cuda_shape_flow_first_step",
        "normalization": normalization,
        "stage": "shape_flow_step",
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **{key: np.ascontiguousarray(value) for key, value in arrays.items()},
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return {
        **metadata,
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "format": "shape_flow_step_npz",
        "coords_shape": [int(v) for v in coords.shape],
        "sample_next_shape": [int(v) for v in sample_next.shape],
    }


def serialize_layout(layout: dict[str, Any]) -> dict[str, list[int | None]]:
    serialized: dict[str, list[int | None]] = {}
    for key, value in layout.items():
        if not isinstance(value, slice):
            raise ValueError(f"mesh layout entry {key!r} must be a slice, got {type(value).__name__}")
        serialized[str(key)] = [value.start, value.stop, value.step]
    return serialized


def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def sync_cuda(torch_module: Any) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def elapsed(started: float) -> float:
    return max(0.0, time.perf_counter() - started)


if __name__ == "__main__":
    raise SystemExit(main())
