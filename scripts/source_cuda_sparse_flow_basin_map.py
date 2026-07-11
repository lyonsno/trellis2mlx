"""Run source-CUDA sparse-flow continuations from MLX-minus-CUDA perturbations."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--source-steps", default="source_cuda_steps.npz", type=Path)
    parser.add_argument("--candidate-step", default="mlx_step2_capture.npz", type=Path)
    parser.add_argument("--conditioning", default="conditioning.npz", type=Path)
    parser.add_argument("--checkpoint", default="ss_flow_img_dit_1_3B_64_bf16.safetensors", type=Path)
    parser.add_argument("--old-steps", default="old_5ccda85_sparse_flow_steps.npz", type=Path)
    parser.add_argument("--current-steps", default="current_60508eb_sparse_flow_steps.npz", type=Path)
    parser.add_argument("--alphas", default="-0.5,0,0.015625,0.03125,0.0625,0.125,0.25,0.5,1")
    parser.add_argument("--start-after-step-index", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance-strength", type=float, default=7.5)
    parser.add_argument("--guidance-rescale", type=float, default=0.7)
    parser.add_argument("--guidance-interval", default="0.6,1.0")
    parser.add_argument("--rescale-t", type=float, default=5.0)
    parser.add_argument("--sigma-min", type=float, default=1e-5)
    return parser


def parse_alphas(value: str) -> np.ndarray:
    alphas = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not alphas:
        raise ValueError("--alphas must contain at least one value")
    return np.asarray(alphas, dtype=np.float32)


def remaining_step_indices(steps: int, *, start_after_step_index: int) -> list[int]:
    if start_after_step_index < -1 or start_after_step_index >= steps:
        raise ValueError(f"start_after_step_index={start_after_step_index} outside steps={steps}")
    return list(range(start_after_step_index + 1, steps))


def build_perturbed_starts(
    source_post: np.ndarray,
    candidate_post: np.ndarray,
    alphas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_post = np.asarray(source_post, dtype=np.float32)
    candidate_post = np.asarray(candidate_post, dtype=np.float32)
    alphas = np.asarray(alphas, dtype=np.float32)
    if source_post.shape != candidate_post.shape:
        raise ValueError(f"source/candidate post-step shapes differ: {source_post.shape} vs {candidate_post.shape}")
    if source_post.ndim != 5:
        raise ValueError(f"post-step arrays must be [B,C,Z,Y,X], got {source_post.shape}")
    delta = candidate_post - source_post
    starts = source_post[None, ...] + alphas.reshape((-1,) + (1,) * source_post.ndim) * delta[None, ...]
    return starts.astype(np.float32), delta.astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    phase = "setup"
    started = time.time()
    report: dict[str, Any] = {
        "schema": "trellis2mlx.source_cuda_sparse_flow_basin_map.v1",
        "status": "failed",
        "failure_phase": None,
        "phases": [],
    }
    try:
        alphas = parse_alphas(args.alphas)
        guidance_interval = _parse_guidance_interval(args.guidance_interval)
        step_indices = remaining_step_indices(args.steps, start_after_step_index=args.start_after_step_index)
        report["request"] = {
            "alphas": [float(v) for v in alphas],
            "start_after_step_index": int(args.start_after_step_index),
            "remaining_step_indices": step_indices,
            "steps": int(args.steps),
            "guidance_strength": float(args.guidance_strength),
            "guidance_rescale": float(args.guidance_rescale),
            "guidance_interval": [float(guidance_interval[0]), float(guidance_interval[1])],
            "rescale_t": float(args.rescale_t),
            "sigma_min": float(args.sigma_min),
        }

        phase = "extract_source"
        _install_source_path(Path.cwd())
        report["phases"].append(phase)

        phase = "import_torch_source"
        import torch
        from safetensors.torch import load_file

        from trellis2.models.sparse_structure_flow import SparseStructureFlowModel
        from trellis2.modules.attention import config as attention_config

        attention_config.BACKEND = "sdpa"
        device = torch.device("cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.set_grad_enabled(False)
        report.update(
            {
                "torch": torch.__version__,
                "cuda_device": torch.cuda.get_device_name(0),
                "attention_backend": attention_config.BACKEND,
            }
        )
        report["phases"].append(phase)

        phase = "load_inputs"
        with np.load(args.source_steps) as source_steps:
            source_post = np.asarray(source_steps["sample_next"][args.start_after_step_index], dtype=np.float32)
            source_final = np.asarray(source_steps["sample_next"][-1], dtype=np.float32)
            source_arrays = sorted(source_steps.files)
        with np.load(args.candidate_step) as candidate:
            candidate_post = _select_candidate_post(np.asarray(candidate["sample_next"], dtype=np.float32))
            candidate_arrays = sorted(candidate.files)
        perturbed_starts, delta = build_perturbed_starts(source_post, candidate_post, alphas)
        with np.load(args.conditioning) as conditioning:
            cond_np = np.asarray(conditioning["cond"], dtype=np.float32)
            neg_cond_np = np.asarray(conditioning["neg_cond"], dtype=np.float32)
        old_steps = _load_optional_steps(args.old_steps)
        current_steps = _load_optional_steps(args.current_steps)
        report["inputs"] = {
            "source_steps": str(args.source_steps),
            "candidate_step": str(args.candidate_step),
            "source_arrays": source_arrays,
            "candidate_arrays": candidate_arrays,
            "source_post_shape": list(source_post.shape),
            "candidate_post_shape": list(candidate_post.shape),
            "delta_mean_abs": float(np.mean(np.abs(delta))),
            "delta_max_abs": float(np.max(np.abs(delta))),
        }
        report["phases"].append(phase)

        phase = "load_model"
        model = SparseStructureFlowModel(
            resolution=16,
            in_channels=8,
            out_channels=8,
            model_channels=1536,
            cond_channels=1024,
            num_blocks=30,
            num_heads=12,
            mlp_ratio=5.3334,
            pe_mode="rope",
            share_mod=True,
            initialization="scaled",
            qk_rms_norm=True,
            qk_rms_norm_cross=True,
            dtype="bfloat16",
        )
        state = load_file(str(args.checkpoint), device="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        model.to(device).eval()
        report["model_load"] = {
            "missing": list(missing),
            "unexpected": list(unexpected),
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
        }
        report["phases"].append(phase)

        phase = "continuation_loop"
        cond = torch.from_numpy(cond_np).to(device=device, dtype=torch.float32)
        neg_cond = torch.from_numpy(neg_cond_np).to(device=device, dtype=torch.float32)
        t_pairs = _schedule_pairs(args.steps, args.rescale_t)
        final_samples = []
        remaining_sample_next = []
        remaining_pred_final = []
        alpha_reports = []
        continuation_started = time.time()
        for alpha, start in zip(alphas, perturbed_starts):
            alpha_started = time.time()
            sample = torch.from_numpy(start).to(device=device, dtype=torch.float32)
            sample_next_rows = []
            pred_final_rows = []
            for step_index in step_indices:
                t, t_prev = t_pairs[step_index]
                t_tensor = torch.tensor([1000 * t] * sample.shape[0], device=device, dtype=torch.float32)
                pred_pos = model(sample, t_tensor, cond)
                pred_neg = model(sample, t_tensor, neg_cond)
                pred_final = _guided_pred(
                    sample,
                    pred_pos,
                    pred_neg,
                    t=t,
                    t_prev=t_prev,
                    guidance_strength=args.guidance_strength,
                    guidance_rescale=args.guidance_rescale,
                    guidance_interval=guidance_interval,
                    sigma_min=args.sigma_min,
                )
                sample_next = sample - (t - t_prev) * pred_final
                sample_next_rows.append(sample_next.detach().float().cpu().numpy())
                pred_final_rows.append(pred_final.detach().float().cpu().numpy())
                sample = sample_next
            final_np = sample.detach().float().cpu().numpy()
            final_samples.append(final_np)
            remaining_sample_next.append(np.stack(sample_next_rows, axis=0).astype(np.float32))
            remaining_pred_final.append(np.stack(pred_final_rows, axis=0).astype(np.float32))
            alpha_reports.append(
                _alpha_report(
                    alpha=float(alpha),
                    elapsed_seconds=time.time() - alpha_started,
                    final=final_np,
                    source_final=source_final,
                    old_steps=old_steps,
                    current_steps=current_steps,
                )
            )
        report["continuation_elapsed_seconds"] = time.time() - continuation_started
        report["alpha_reports"] = alpha_reports
        report["phases"].append(phase)

        phase = "write_outputs"
        output_arrays = {
            "alpha_values": alphas.astype(np.float32),
            "perturbed_start": perturbed_starts.astype(np.float32),
            "delta_step2_mlx_minus_cuda": delta.astype(np.float32),
            "source_step2_post": source_post.astype(np.float32),
            "candidate_step2_post": candidate_post.astype(np.float32),
            "final_sample_next": np.stack(final_samples, axis=0).astype(np.float32),
            "remaining_step_indices": np.asarray(step_indices, dtype=np.int32),
            "remaining_sample_next": np.stack(remaining_sample_next, axis=0).astype(np.float32),
            "remaining_pred_final": np.stack(remaining_pred_final, axis=0).astype(np.float32),
        }
        np.savez_compressed(args.output_npz, **output_arrays)
        report.update(
            {
                "status": "done",
                "failure_phase": None,
                "elapsed_seconds": time.time() - started,
                "output_npz": str(args.output_npz),
            }
        )
        report["phases"].append(phase)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return 0
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "failure_phase": phase,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": time.time() - started,
            }
        )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return 1


def _guided_pred(
    sample,
    pred_pos,
    pred_neg,
    *,
    t: float,
    t_prev: float,
    guidance_strength: float,
    guidance_rescale: float,
    guidance_interval: tuple[float, float],
    sigma_min: float,
):
    del t_prev
    if guidance_interval[0] <= t <= guidance_interval[1]:
        pred_cfg = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg
    else:
        pred_cfg = pred_pos
    x0_pos = _pred_to_xstart(sample, t, pred_pos, sigma_min)
    x0_cfg = _pred_to_xstart(sample, t, pred_cfg, sigma_min)
    std_pos = x0_pos.std(dim=list(range(1, x0_pos.ndim)), keepdim=True)
    std_cfg = x0_cfg.std(dim=list(range(1, x0_cfg.ndim)), keepdim=True)
    x0_rescaled = x0_cfg * (std_pos / std_cfg)
    if guidance_rescale > 0 and guidance_interval[0] <= t <= guidance_interval[1]:
        x0_after_rescale = guidance_rescale * x0_rescaled + (1 - guidance_rescale) * x0_cfg
        return _xstart_to_pred(sample, t, x0_after_rescale, sigma_min)
    return pred_cfg


def _pred_to_xstart(x_t, t: float, pred, sigma_min: float):
    return (1 - sigma_min) * x_t - (sigma_min + (1 - sigma_min) * t) * pred


def _xstart_to_pred(x_t, t: float, x0, sigma_min: float):
    return ((1 - sigma_min) * x_t - x0) / (sigma_min + (1 - sigma_min) * t)


def _alpha_report(
    *,
    alpha: float,
    elapsed_seconds: float,
    final: np.ndarray,
    source_final: np.ndarray,
    old_steps: dict[str, np.ndarray] | None,
    current_steps: dict[str, np.ndarray] | None,
) -> dict[str, Any]:
    report = {
        "alpha": alpha,
        "continuation_elapsed_seconds": float(elapsed_seconds),
        "vs_source_cuda_final": compare_arrays(final, source_final),
    }
    if old_steps is not None and "sample_next" in old_steps:
        report["vs_old_final"] = compare_arrays(final, old_steps["sample_next"][-1])
    if current_steps is not None and "sample_next" in current_steps:
        report["vs_current_final"] = compare_arrays(final, current_steps["sample_next"][-1])
    anchors = {
        key: value
        for key, value in (
            ("source_cuda", report.get("vs_source_cuda_final")),
            ("old", report.get("vs_old_final")),
            ("current", report.get("vs_current_final")),
        )
        if value and value.get("shape_match")
    }
    if anchors:
        report["best_final_anchor"] = min(anchors.items(), key=lambda item: item[1]["mean_abs"])[0]
    return report


def compare_arrays(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        return {"shape_a": list(a.shape), "shape_b": list(b.shape), "shape_match": False}
    diff = np.abs(a - b)
    return {
        "shape_match": True,
        "mean_abs": float(diff.mean()),
        "max_abs": float(diff.max()),
        "nonzero": int(np.count_nonzero(diff)),
    }


def _schedule_pairs(steps: int, rescale_t: float) -> list[tuple[float, float]]:
    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    return [(float(t_seq[i]), float(t_seq[i + 1])) for i in range(steps)]


def _select_candidate_post(array: np.ndarray) -> np.ndarray:
    if array.ndim == 5:
        return array
    if array.ndim == 6 and array.shape[0] == 1:
        return array[0]
    raise ValueError(f"candidate sample_next must be [B,C,Z,Y,X] or [1,B,C,Z,Y,X], got {array.shape}")


def _load_optional_steps(path: Path) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    with np.load(path) as data:
        return {name: np.asarray(data[name], dtype=np.float32) for name in data.files}


def _parse_guidance_interval(value: str) -> tuple[float, float]:
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("--guidance-interval must contain exactly two floats")
    return (parts[0], parts[1])


def _install_source_path(base: Path) -> None:
    source_tree = base / "trellis2_source"
    if source_tree.is_dir():
        sys.path.insert(0, str(source_tree))
        return
    source_tar = base / "trellis2_source_tarball.bin"
    if not source_tar.is_file():
        source_tar = base / "trellis2_source.tar.gz"
    if not source_tar.is_file():
        raise FileNotFoundError(base / "trellis2_source_tarball.bin")
    with tarfile.open(source_tar, "r:gz") as tf:
        tf.extractall(base / "source")
    sys.path.insert(0, str(base / "source"))


if __name__ == "__main__":
    raise SystemExit(main())
