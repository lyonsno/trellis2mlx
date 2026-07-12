"""Run a source-CUDA full decode from a sparse-flow perturbation witness.

This is the load-bearing basin experiment: it does not run Trellis-Mac and it
does not restart sparse sampling from fresh noise. It starts from a named
post-step sparse-flow latent, applies one MLX-minus-CUDA delta scale, finishes
the remaining sparse-flow steps on the official TRELLIS.2 source CUDA path,
then runs source CUDA sparse decode, shape, texture, and mesh decode.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.source_cuda_postcond_full_decode_timing import (
        apply_sparse_backend_env,
        conditioning_identity,
        elapsed,
        extract_source,
        install_mesh_override,
        load_conditioning,
        mesh_summary,
        parameter_count,
        required_model_names,
        resolve_model_ref,
        sha256_file,
        sparse_tensor_summary,
        sync_cuda,
        write_binary_mesh_ply,
        write_mesh_state_npz,
    )
    from scripts.source_cuda_sparse_flow_basin_map import (
        _guided_pred,
        _parse_guidance_interval,
        _schedule_pairs,
        _select_candidate_post,
        compare_arrays,
        remaining_step_indices,
    )
except ModuleNotFoundError:
    from source_cuda_postcond_full_decode_timing import (
        apply_sparse_backend_env,
        conditioning_identity,
        elapsed,
        extract_source,
        install_mesh_override,
        load_conditioning,
        mesh_summary,
        parameter_count,
        required_model_names,
        resolve_model_ref,
        sha256_file,
        sparse_tensor_summary,
        sync_cuda,
        write_binary_mesh_ply,
        write_mesh_state_npz,
    )
    from source_cuda_sparse_flow_basin_map import (
        _guided_pred,
        _parse_guidance_interval,
        _schedule_pairs,
        _select_candidate_post,
        compare_arrays,
        remaining_step_indices,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--output-ply", default=Path("cuda_perturbed_mesh.ply"), type=Path)
    parser.add_argument("--output-mesh-state", type=Path)
    parser.add_argument("--source-steps", default="source_cuda_steps.npz", type=Path)
    parser.add_argument("--candidate-step", default="mlx_step2_capture.npz", type=Path)
    parser.add_argument("--conditioning", default="conditioning.npz", type=Path)
    parser.add_argument("--conditioning-1024", type=Path)
    parser.add_argument("--source-tar", default="trellis2_source_tarball.bin", type=Path)
    parser.add_argument("--mesh-override", default=Path("o_voxel_override_convert.py"), type=Path)
    parser.add_argument("--model-repo", default="microsoft/TRELLIS.2-4B")
    parser.add_argument("--pipeline-config", default="pipeline.json")
    parser.add_argument("--pipeline-type", default="512")
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--start-after-step-index", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max-num-tokens", default=49152, type=int)
    parser.add_argument("--guidance-strength", type=float, default=7.5)
    parser.add_argument("--guidance-rescale", type=float, default=0.7)
    parser.add_argument("--guidance-interval", default="0.6,1.0")
    parser.add_argument("--rescale-t", type=float, default=5.0)
    parser.add_argument("--sigma-min", type=float, default=1e-5)
    parser.add_argument(
        "--block-injection-trace",
        type=Path,
        help="NPZ trace containing a tensor to inject into a source-CUDA sparse-flow block.",
    )
    parser.add_argument("--block-injection-step-index", type=int, default=2)
    parser.add_argument("--block-injection-block-index", type=int, default=0)
    parser.add_argument(
        "--block-injection-branch",
        choices=("pos", "neg", "both"),
        default="pos",
        help="CFG branch that receives the injected block tensor.",
    )
    parser.add_argument(
        "--block-injection-stage",
        choices=("norm1", "modulated_self_input", "after_self"),
        default="modulated_self_input",
    )
    parser.add_argument(
        "--block-injection-array-key",
        help="Override trace array key; default is <branch>_block<index>_<stage>.",
    )
    parser.add_argument(
        "--sparse-conv-backend",
        default="none",
        choices=("none", "spconv", "torchsparse", "flex_gemm"),
        help="Official source sparse convolution backend.",
    )
    parser.add_argument(
        "--sparse-attn-backend",
        default="sdpa",
        choices=("xformers", "flash_attn", "flash_attn_3", "sdpa", "naive"),
        help="Official source sparse attention backend.",
    )
    return parser


@dataclass(frozen=True)
class BlockInjection:
    trace_path: Path
    array_key: str
    branch: str
    step_index: int
    block_index: int
    stage: str
    array: np.ndarray
    trace_identity: dict[str, Any]

    def applies(self, *, step_index: int, branch: str, block_index: int) -> bool:
        return (
            int(step_index) == self.step_index
            and int(block_index) == self.block_index
            and (self.branch == "both" or self.branch == branch)
        )

    def report_identity(self) -> dict[str, Any]:
        return {
            "trace_path": str(self.trace_path),
            "array_key": self.array_key,
            "branch": self.branch,
            "step_index": self.step_index,
            "block_index": self.block_index,
            "stage": self.stage,
            "array_shape": list(self.array.shape),
            "array_dtype": str(self.array.dtype),
            "trace_identity": self.trace_identity,
            "comparison_class": "source_cuda_sparse_flow_with_named_block_tensor_injection",
            "route_identity_evidence": True,
        }


def load_block_injection(
    trace_path: Path,
    *,
    branch: str,
    step_index: int,
    block_index: int,
    stage: str,
    array_key: str | None,
) -> BlockInjection:
    trace_path = Path(trace_path)
    selected_key = array_key or f"{branch}_block{block_index}_{stage}"
    with np.load(trace_path) as trace:
        if selected_key not in trace:
            raise KeyError(f"block injection trace missing required array {selected_key!r}")
        array = np.asarray(trace[selected_key], dtype=np.float32)
        trace_identity: dict[str, Any] = {}
        if "route_identity_json" in trace:
            raw_identity = np.asarray(trace["route_identity_json"])
            if raw_identity.shape == ():
                trace_identity = json.loads(str(raw_identity.item()))
    return BlockInjection(
        trace_path=trace_path,
        array_key=selected_key,
        branch=branch,
        step_index=int(step_index),
        block_index=int(block_index),
        stage=stage,
        array=array,
        trace_identity=trace_identity,
    )


def build_single_perturbed_start(
    source_post: np.ndarray,
    candidate_post: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    source_post = np.asarray(source_post, dtype=np.float32)
    candidate_post = np.asarray(candidate_post, dtype=np.float32)
    if source_post.shape != candidate_post.shape:
        raise ValueError(f"source/candidate post-step shapes differ: {source_post.shape} vs {candidate_post.shape}")
    if source_post.ndim != 5:
        raise ValueError(f"post-step arrays must be [B,C,Z,Y,X], got {source_post.shape}")
    delta = candidate_post - source_post
    start = source_post + np.float32(alpha) * delta
    return start.astype(np.float32), delta.astype(np.float32)


def decoded_mask_to_source_coords(decoded: Any, *, resolution: int):
    torch = _torch_from_tensor(decoded)
    if resolution != decoded.shape[2]:
        ratio = decoded.shape[2] // resolution
        decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
    return torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    decode_started: float | None = None
    phase = "setup"
    report: dict[str, Any] = {
        "schema": "trellis2mlx.source_cuda_perturbed_sparse_full_decode.v1",
        "status": "failed",
        "failure_phase": None,
        "requested_backend": "official TRELLIS.2 source CUDA continuation from perturbed sparse-flow latent",
        "pipeline_type": args.pipeline_type,
        "alpha": float(args.alpha),
        "start_after_step_index": int(args.start_after_step_index),
        "steps": int(args.steps),
        "seed": int(args.seed),
        "phase_timings": {},
        "stage_timings": {},
        "model_loads": {},
        "forbidden_inferences": [
            "not Trellis-Mac",
            "not fresh sparse sampling",
            "not a GLB/finalization parity claim unless a finalization artifact is listed",
        ],
    }

    try:
        output_json = Path(args.output_json)
        output_npz = Path(args.output_npz)
        output_ply = Path(args.output_ply)
        output_mesh_state = Path(args.output_mesh_state) if args.output_mesh_state is not None else None
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        output_ply.parent.mkdir(parents=True, exist_ok=True)
        if output_mesh_state is not None:
            output_mesh_state.parent.mkdir(parents=True, exist_ok=True)

        phase = "validate_args"
        if args.steps <= 0:
            raise ValueError("--steps must be positive")
        if args.pipeline_type != "512" and args.conditioning_1024 is None:
            raise ValueError(f"--conditioning-1024 is required for pipeline_type={args.pipeline_type}")
        guidance_interval = _parse_guidance_interval(args.guidance_interval)
        model_names = required_model_names(args.pipeline_type)
        step_indices = remaining_step_indices(args.steps, start_after_step_index=args.start_after_step_index)
        report["required_model_names"] = list(model_names)
        report["remaining_step_indices"] = step_indices
        report["requested_sparse_backend"] = apply_sparse_backend_env(
            args.sparse_conv_backend,
            args.sparse_attn_backend,
        )
        block_injection = None
        if args.block_injection_trace is not None:
            block_injection = load_block_injection(
                args.block_injection_trace,
                branch=args.block_injection_branch,
                step_index=args.block_injection_step_index,
                block_index=args.block_injection_block_index,
                stage=args.block_injection_stage,
                array_key=args.block_injection_array_key,
            )
            report["block_injection"] = block_injection.report_identity()

        phase = "extract_source"
        phase_started = time.perf_counter()
        source_root = extract_source(Path(args.source_tar), Path.cwd())
        import sys

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

        phase = "load_inputs"
        phase_started = time.perf_counter()
        with np.load(args.source_steps) as source_steps:
            source_post = np.asarray(source_steps["sample_next"][args.start_after_step_index], dtype=np.float32)
            source_final = np.asarray(source_steps["sample_next"][-1], dtype=np.float32)
            source_arrays = sorted(source_steps.files)
        with np.load(args.candidate_step) as candidate:
            candidate_post = _select_candidate_post(np.asarray(candidate["sample_next"], dtype=np.float32))
            candidate_arrays = sorted(candidate.files)
        perturbed_start, delta = build_single_perturbed_start(source_post, candidate_post, alpha=args.alpha)
        cond_512 = load_conditioning(Path(args.conditioning), device, torch)
        cond_1024 = (
            load_conditioning(Path(args.conditioning_1024), device, torch)
            if args.conditioning_1024 is not None
            else None
        )
        report["inputs"] = {
            "source_steps": str(args.source_steps),
            "candidate_step": str(args.candidate_step),
            "source_arrays": source_arrays,
            "candidate_arrays": candidate_arrays,
            "source_post_shape": list(source_post.shape),
            "candidate_post_shape": list(candidate_post.shape),
            "delta_step2_mlx_minus_cuda": compare_arrays(candidate_post, source_post),
            "perturbed_start_vs_source_post": compare_arrays(perturbed_start, source_post),
            "perturbed_start_vs_candidate_post": compare_arrays(perturbed_start, candidate_post),
        }
        report["conditioning"] = {
            "cond_512": conditioning_identity(Path(args.conditioning), cond_512),
            "cond_1024": (
                conditioning_identity(Path(args.conditioning_1024), cond_1024)
                if args.conditioning_1024 is not None and cond_1024 is not None
                else None
            ),
        }
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

        phase = "perturbed_post_conditioning_decode"
        decode_started = time.perf_counter()
        torch.manual_seed(args.seed)
        stage_timings = report["stage_timings"]

        stage_started = time.perf_counter()
        z_s, sparse_trace = finish_sparse_flow_on_source_cuda(
            pipeline.models["sparse_structure_flow_model"],
            perturbed_start,
            cond_512,
            torch_module=torch,
            device=device,
            step_indices=step_indices,
            steps=args.steps,
            guidance_strength=args.guidance_strength,
            guidance_rescale=args.guidance_rescale,
            guidance_interval=guidance_interval,
            rescale_t=args.rescale_t,
            sigma_min=args.sigma_min,
            block_injection=block_injection,
        )
        sync_cuda(torch)
        stage_timings["finish_sparse_flow_elapsed_seconds"] = elapsed(stage_started)
        report["sparse_flow_trace"] = sparse_trace
        report["final_sparse_latent_vs_source_cuda_final"] = compare_arrays(
            z_s.detach().float().cpu().numpy(),
            source_final,
        )
        report["post_conditioning_partial_elapsed_seconds"] = elapsed(decode_started)

        stage_started = time.perf_counter()
        coords = decode_sparse_structure_latent(pipeline, z_s, resolution=32)
        sync_cuda(torch)
        stage_timings["decode_sparse_structure_elapsed_seconds"] = elapsed(stage_started)
        report["sparse_coords_count"] = int(coords.shape[0])
        report["post_conditioning_partial_elapsed_seconds"] = elapsed(decode_started)

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
        else:  # pragma: no cover - guarded by required_model_names
            raise AssertionError(args.pipeline_type)
        report["post_conditioning_partial_elapsed_seconds"] = elapsed(decode_started)

        stage_started = time.perf_counter()
        meshes = pipeline.decode_latent(shape_slat, tex_slat, resolution)
        sync_cuda(torch)
        stage_timings["decode_latent_elapsed_seconds"] = elapsed(stage_started)

        report["post_conditioning_decode_elapsed_seconds"] = elapsed(decode_started)
        report["mesh_summary"] = [mesh_summary(mesh) for mesh in meshes]
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
                        "artifact_scope": "perturbed_sparse_flow_source_cuda_post_decode_mesh_state",
                    }
                )
                report["output_mesh_state"] = str(output_mesh_state)
        report["mesh_artifacts"] = mesh_artifacts

        np.savez(
            output_npz,
            alpha=np.asarray(args.alpha, dtype=np.float32),
            start_after_step_index=np.asarray(args.start_after_step_index, dtype=np.int32),
            remaining_step_indices=np.asarray(step_indices, dtype=np.int32),
            delta_step2_mlx_minus_cuda=delta.astype(np.float32),
            source_step2_post=source_post.astype(np.float32),
            candidate_step2_post=candidate_post.astype(np.float32),
            perturbed_start=perturbed_start.astype(np.float32),
            final_sparse_latent=z_s.detach().float().cpu().numpy().astype(np.float32),
            sparse_coords_count=np.asarray(report["sparse_coords_count"], dtype=np.int64),
            resolution=np.asarray(report["resolution"], dtype=np.int64),
            mesh_vertices=np.asarray([entry["vertices"] for entry in report["mesh_summary"]], dtype=np.int64),
            mesh_faces=np.asarray([entry["faces"] for entry in report["mesh_summary"]], dtype=np.int64),
            post_conditioning_decode_elapsed_seconds=np.asarray(
                report["post_conditioning_decode_elapsed_seconds"],
                dtype=np.float64,
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
        if decode_started is not None:
            report["post_conditioning_partial_elapsed_seconds"] = elapsed(decode_started)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return 1


def finish_sparse_flow_on_source_cuda(
    model: Any,
    start: np.ndarray,
    cond: dict[str, Any],
    *,
    torch_module: Any,
    device: Any,
    step_indices: list[int],
    steps: int,
    guidance_strength: float,
    guidance_rescale: float,
    guidance_interval: tuple[float, float],
    rescale_t: float,
    sigma_min: float,
    block_injection: BlockInjection | None = None,
) -> tuple[Any, dict[str, Any]]:
    t_pairs = _schedule_pairs(steps, rescale_t)
    sample = torch_module.from_numpy(np.asarray(start, dtype=np.float32)).to(device=device, dtype=torch_module.float32)
    sample_next_summaries = []
    if hasattr(model, "to"):
        model.to(device)
    for step_index in step_indices:
        t, t_prev = t_pairs[step_index]
        step_started = time.perf_counter()
        t_tensor = torch_module.tensor([1000 * t] * sample.shape[0], device=device, dtype=torch_module.float32)
        pred_pos = _predict_sparse_flow_branch(
            model,
            sample,
            t_tensor,
            cond["cond"],
            torch_module=torch_module,
            branch="pos",
            step_index=step_index,
            block_injection=block_injection,
        )
        pred_neg = _predict_sparse_flow_branch(
            model,
            sample,
            t_tensor,
            cond["neg_cond"],
            torch_module=torch_module,
            branch="neg",
            step_index=step_index,
            block_injection=block_injection,
        )
        pred_final = _guided_pred(
            sample,
            pred_pos,
            pred_neg,
            t=t,
            t_prev=t_prev,
            guidance_strength=guidance_strength,
            guidance_rescale=guidance_rescale,
            guidance_interval=guidance_interval,
            sigma_min=sigma_min,
        )
        sample = sample - (t - t_prev) * pred_final
        sample_next_summaries.append(
            {
                "step_index": int(step_index),
                "t": float(t),
                "t_prev": float(t_prev),
                "elapsed_seconds": elapsed(step_started),
                "block_injection_applied": bool(
                    block_injection is not None
                    and any(
                        block_injection.applies(
                            step_index=step_index,
                            branch=branch,
                            block_index=block_injection.block_index,
                        )
                        for branch in ("pos", "neg")
                    )
                ),
                "mean": float(sample.detach().float().mean().cpu()),
                "std": float(sample.detach().float().std().cpu()),
            }
        )
    return sample, {"remaining_sample_next": sample_next_summaries}


def _predict_sparse_flow_branch(
    model: Any,
    sample: Any,
    t_tensor: Any,
    cond: Any,
    *,
    torch_module: Any,
    branch: str,
    step_index: int,
    block_injection: BlockInjection | None,
):
    if block_injection is None:
        return model(sample, t_tensor, cond)
    if not any(
        block_injection.applies(step_index=step_index, branch=branch, block_index=index)
        for index in range(len(model.blocks))
    ):
        return model(sample, t_tensor, cond)
    return _forward_sparse_flow_with_block_injection(
        model,
        sample,
        t_tensor,
        cond,
        torch_module=torch_module,
        branch=branch,
        step_index=step_index,
        injection=block_injection,
    )


def _forward_sparse_flow_with_block_injection(
    model: Any,
    sample: Any,
    t_tensor: Any,
    cond: Any,
    *,
    torch_module: Any,
    branch: str,
    step_index: int,
    injection: BlockInjection,
):
    from trellis2.modules.utils import manual_cast

    h = sample.view(*sample.shape[:2], -1).permute(0, 2, 1).contiguous()
    h = model.input_layer(h)
    if model.pe_mode == "ape":
        h = h + model.pos_emb[None]
    t_emb = model.t_embedder(t_tensor)
    if model.share_mod:
        t_emb = model.adaLN_modulation(t_emb)
    t_emb = manual_cast(t_emb, model.dtype)
    h = manual_cast(h, model.dtype)
    cond = manual_cast(cond, model.dtype)
    for block_index, block in enumerate(model.blocks):
        if injection.applies(step_index=step_index, branch=branch, block_index=block_index):
            h = _block_forward_with_injection(
                torch_module,
                model,
                block,
                h,
                t_emb,
                cond,
                injection=injection,
            )
        else:
            h = block(h, t_emb, cond, model.rope_phases)
    h = manual_cast(h, sample.dtype)
    final_norm = torch_module.nn.functional.layer_norm(h, h.shape[-1:])
    final_flat = model.out_layer(final_norm)
    return final_flat.permute(0, 2, 1).view(
        sample.shape[0],
        model.out_channels,
        model.resolution,
        model.resolution,
        model.resolution,
    )


def _block_forward_with_injection(
    torch_module: Any,
    model: Any,
    block: Any,
    x: Any,
    mod: Any,
    context: Any,
    *,
    injection: BlockInjection,
):
    from trellis2.modules.attention import RotaryPositionEmbedder
    from trellis2.modules.attention.full_attn import scaled_dot_product_attention

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = _split_block_modulation(
        block,
        mod,
    )

    h = block.norm1(x)
    if injection.stage == "norm1":
        h = _injected_tensor_like(torch_module, injection, h)
    h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
    if injection.stage == "modulated_self_input":
        h = _injected_tensor_like(torch_module, injection, h)

    attn = block.self_attn
    qkv = attn.to_qkv(h)
    qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, attn.num_heads, -1)
    q, k, v = qkv.unbind(dim=2)
    if getattr(attn, "qk_rms_norm", False):
        q = attn.q_rms_norm(q)
        k = attn.k_rms_norm(k)
    if getattr(attn, "use_rope", False):
        q = RotaryPositionEmbedder.apply_rotary_embedding(q, model.rope_phases)
        k = RotaryPositionEmbedder.apply_rotary_embedding(k, model.rope_phases)
    raw = scaled_dot_product_attention(q, k, v)
    raw_flat = raw.reshape(h.shape[0], h.shape[1], -1)
    h = attn.to_out(raw_flat)
    h = h * gate_msa.unsqueeze(1)
    x = x + h
    if injection.stage == "after_self":
        x = _injected_tensor_like(torch_module, injection, x)

    h = block.norm2(x)
    attn = block.cross_attn
    q = attn.to_q(h)
    kv = attn.to_kv(context)
    q = q.reshape(q.shape[0], q.shape[1], attn.num_heads, -1)
    kv = kv.reshape(kv.shape[0], kv.shape[1], 2, attn.num_heads, -1)
    k, v = kv.unbind(dim=2)
    if getattr(attn, "qk_rms_norm", False):
        q = attn.q_rms_norm(q)
        k = attn.k_rms_norm(k)
    raw = scaled_dot_product_attention(q, k, v)
    raw_flat = raw.reshape(h.shape[0], h.shape[1], -1)
    h = attn.to_out(raw_flat)
    x = x + h

    h = block.norm3(x)
    h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
    fc1, fc2 = _source_mlp_linears(block.mlp)
    h = fc2(_source_gelu(torch_module, block.mlp, fc1(h)))
    h = h * gate_mlp.unsqueeze(1)
    return x + h


def _split_block_modulation(block: Any, mod: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
    if getattr(block, "share_mod", False):
        return (block.modulation + mod).type(mod.dtype).chunk(6, dim=1)
    return block.adaLN_modulation(mod).chunk(6, dim=1)


def _source_mlp_linears(mlp: Any) -> tuple[Any, Any]:
    if hasattr(mlp, "mlp"):
        return mlp.mlp[0], mlp.mlp[2]
    return mlp.mlp_0, mlp.mlp_2


def _source_gelu(torch_module: Any, mlp: Any, value: Any) -> Any:
    if hasattr(mlp, "mlp"):
        return mlp.mlp[1](value)
    return torch_module.nn.functional.gelu(value, approximate="tanh")


def _injected_tensor_like(torch_module: Any, injection: BlockInjection, target: Any) -> Any:
    array = np.asarray(injection.array, dtype=np.float32)
    if array.ndim == target.ndim - 1:
        array = array[None, ...]
    if tuple(array.shape) != tuple(target.shape):
        raise ValueError(
            f"block injection {injection.array_key!r} shape {array.shape} does not match "
            f"target {injection.stage} shape {tuple(target.shape)}"
        )
    return torch_module.from_numpy(array).to(device=target.device, dtype=target.dtype)


def decode_sparse_structure_latent(pipeline: Any, z_s: Any, *, resolution: int):
    decoder = pipeline.models["sparse_structure_decoder"]
    if pipeline.low_vram:
        decoder.to(pipeline.device)
    decoded = decoder(z_s) > 0
    if pipeline.low_vram:
        decoder.cpu()
    return decoded_mask_to_source_coords(decoded, resolution=resolution)


def _torch_from_tensor(value: Any):
    module_name = type(value).__module__.split(".", 1)[0]
    if module_name != "torch":
        raise TypeError(f"decoded mask must be a torch tensor, got {type(value).__name__}")
    import torch

    return torch


if __name__ == "__main__":
    raise SystemExit(main())
