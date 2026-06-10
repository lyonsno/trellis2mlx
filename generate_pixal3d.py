#!/usr/bin/env python3
"""Pixal3D-MLX route-control harness.

This is not yet the full image-to-GLB pipeline. It proves that the effective
route is the Pixal3D projected-conditioning route, writes a JSON report, and
fails loud if the requested route silently collapses to vanilla TRELLIS global
conditioning.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.utils
import numpy as np


SCHEMA = "trellis2mlx.pixal3d_route.v1"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def default_report_path() -> Path:
    run_id = time.strftime("%Y%m%d-%H%M%S")
    unique = f"{os.getpid()}-{time.time_ns()}"
    return Path("/tmp") / f"pixal3d-mlx-route-{run_id}-{unique}" / "report.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-route", action="store_true", help="Run the projected-conditioning route smoke")
    parser.add_argument("--report", type=Path, default=None, help="JSON report path")
    parser.add_argument("--overwrite-report", action="store_true", help="Overwrite an existing JSON report")
    parser.add_argument("--repo-root", type=Path, default=None, help="Effective pixal3d-mlx repo root")
    parser.add_argument("--checkpoint", type=Path, help="Optional Pixal3D denoiser checkpoint to inventory")
    parser.add_argument("--grid-resolution", type=_positive_int, default=2)
    parser.add_argument("--image-size", type=_positive_int, default=32)
    parser.add_argument("--patch-size", type=_positive_int, default=16)
    parser.add_argument("--context-channels", type=_positive_int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.repo_root is None:
        args.repo_root = Path.cwd()
    if args.report is None:
        args.report = default_report_path()
    return args


def ensure_report_writable(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"report already exists: {path}; pass --overwrite-report to replace it")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n")


def _run_text(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def _repo_identity(repo_root: Path) -> dict[str, Any]:
    return {
        "root": str(repo_root),
        "head": _run_text(["git", "rev-parse", "HEAD"], cwd=repo_root),
        "branch": _run_text(["git", "branch", "--show-current"], cwd=repo_root),
        "status_short": _run_text(["git", "status", "--short"], cwd=repo_root),
    }


def _host_identity() -> dict[str, Any]:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "default_device": str(mx.default_device()),
    }


def _model_keys(model) -> set[str]:
    return {key for key, _ in mlx.utils.tree_flatten(model.parameters())}


def inspect_checkpoint(checkpoint: Path | None, models: dict[str, Any]) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "path": str(checkpoint) if checkpoint is not None else None,
        "exists": checkpoint.exists() if checkpoint is not None else None,
        "size_bytes": checkpoint.stat().st_size if checkpoint is not None and checkpoint.exists() else None,
        "loaded": 0,
        "missing": None,
        "skipped": 0,
        "projection_keys_present": None,
    }
    if checkpoint is None:
        return identity
    if not checkpoint.exists():
        identity["error"] = "checkpoint does not exist"
        return identity

    from safetensors import safe_open
    from trellmlx.weight_loader import _remap_key

    model_keys = set()
    for model in models.values():
        model_keys.update(_model_keys(model))

    checkpoint_keys = []
    with safe_open(str(checkpoint), framework="numpy") as sf:
        checkpoint_keys = list(sf.keys())

    remapped = [_remap_key(key) for key in checkpoint_keys]
    matched = [key for key in remapped if key in model_keys]
    identity.update({
        "total_keys": len(checkpoint_keys),
        "loaded": len(matched),
        "missing": len(model_keys - set(matched)),
        "skipped": len(checkpoint_keys) - len(matched),
        "projection_keys_present": any(".cross_attn.proj_linear." in key for key in remapped),
    })
    return identity


def build_synthetic_context(args: argparse.Namespace) -> dict[str, mx.array]:
    from trellmlx.models.dinov3_proj import DINOv3ProjectionAdapter

    patch_grid = args.image_size // args.patch_size
    if patch_grid * args.patch_size != args.image_size:
        raise ValueError("image-size must be divisible by patch-size")

    prefix = mx.zeros((1, 5, args.context_channels), dtype=mx.float32)
    patches = mx.ones((1, patch_grid * patch_grid, args.context_channels), dtype=mx.float32)
    features = mx.concatenate([prefix, patches], axis=1)

    adapter = DINOv3ProjectionAdapter(
        image_size=args.image_size,
        patch_size=args.patch_size,
        grid_resolution=args.grid_resolution,
        num_prefix_tokens=5,
    )
    return adapter(
        features,
        camera_angle_x=mx.array([np.pi / 2], dtype=mx.float32),
        distance=mx.array([2.0], dtype=mx.float32),
        mesh_scale=mx.array([1.0], dtype=mx.float32),
    )


def build_smoke_models(args: argparse.Namespace) -> dict[str, Any]:
    from trellmlx.models.pixal3d_flow import Pixal3DSLatFlowModel, Pixal3DSparseStructureFlowModel

    return {
        "ss_flow": Pixal3DSparseStructureFlowModel(
            in_channels=8,
            out_channels=8,
            model_channels=64,
            num_heads=4,
            num_blocks=1,
            mlp_hidden=128,
            context_channels=args.context_channels,
            proj_in_channels=args.context_channels,
            resolution=args.grid_resolution,
        ),
        "slat_flow": Pixal3DSLatFlowModel(
            in_channels=32,
            out_channels=32,
            model_channels=64,
            num_heads=4,
            num_blocks=1,
            mlp_hidden=128,
            context_channels=args.context_channels,
            proj_in_channels=args.context_channels,
        ),
    }


def validate_effective_route(route: dict[str, Any]) -> None:
    errors = []
    if route.get("requested") == "pixal3d-proj" and route.get("effective") != "pixal3d-proj":
        errors.append("effective route is not pixal3d-proj")
    if "proj" not in route.get("context_keys", []):
        errors.append("projected conditioning key 'proj' is missing")
    if route.get("projected_shape") is None:
        errors.append("projected conditioning shape is missing")
    for role, class_name in route.get("model_classes", {}).items():
        if not class_name.startswith("Pixal3D"):
            errors.append(f"{role} model is not a Pixal3D model: {class_name}")
    if errors:
        route["fallback_detected"] = True
        raise RuntimeError("Pixal3D projected conditioning route invalid: " + "; ".join(errors))
    route["fallback_detected"] = False
    route["pixal3d_projected_conditioning"] = True


def mark_failure(
    report: dict[str, Any],
    report_path: Path,
    *,
    phase: str,
    error: str,
    extra: dict[str, Any] | None = None,
) -> None:
    report["status"] = "failed"
    report["phase"] = phase
    report["error"] = error
    report["last_trustworthy_evidence"] = {
        "phase": phase,
        "report_existed_before_write": report_path.exists(),
    }
    if extra:
        report.update(extra)
    write_report(report_path, report)


def run_smoke_route(args: argparse.Namespace, command_line: list[str] | None = None) -> dict[str, Any]:
    ensure_report_writable(args.report, overwrite=args.overwrite_report)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command_line": command_line or sys.argv,
        "repo": _repo_identity(args.repo_root),
        "host": _host_identity(),
        "settings": {
            "seed": args.seed,
            "grid_resolution": args.grid_resolution,
            "image_size": args.image_size,
            "patch_size": args.patch_size,
            "context_channels": args.context_channels,
        },
        "last_trustworthy_evidence": {"phase": "initialized"},
    }
    write_report(args.report, report)

    try:
        mx.random.seed(args.seed)
        context = build_synthetic_context(args)
        models = build_smoke_models(args)
        route = {
            "requested": "pixal3d-proj",
            "effective": "pixal3d-proj",
            "context_keys": sorted(context.keys()),
            "projected_shape": list(context["proj"].shape) if "proj" in context else None,
            "global_shape": list(context["global"].shape) if "global" in context else None,
            "model_classes": {name: model.__class__.__name__ for name, model in models.items()},
        }
        validate_effective_route(route)

        x_ss = mx.random.normal((1, 8, args.grid_resolution, args.grid_resolution, args.grid_resolution))
        t = mx.array([500.0])
        ss_out = models["ss_flow"](x_ss, t, context)

        token_count = args.grid_resolution ** 3
        x_slat = mx.random.normal((token_count, 32))
        coords = mx.array(np.indices((args.grid_resolution, args.grid_resolution, args.grid_resolution)).reshape(3, -1).T)
        slat_out = models["slat_flow"](x_slat, t, context, coords=coords)
        mx.eval(ss_out, slat_out)

        report["checkpoint"] = inspect_checkpoint(args.checkpoint, models)
        report["route"] = route
        report["smoke"] = {
            "ss_flow_output_shape": list(ss_out.shape),
            "slat_flow_output_shape": list(slat_out.shape),
            "ss_flow_output_std": float(mx.std(ss_out).item()),
            "slat_flow_output_std": float(mx.std(slat_out).item()),
        }
        report["status"] = "ok"
        report["last_trustworthy_evidence"] = {"phase": "smoke_route"}
        write_report(args.report, report)
        return report
    except Exception as exc:
        mark_failure(report, args.report, phase="smoke_route", error=str(exc))
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.smoke_route:
        raise SystemExit("Only --smoke-route is implemented in this checkpoint")
    run_smoke_route(args, command_line=["generate_pixal3d.py", *(argv or sys.argv[1:])])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
