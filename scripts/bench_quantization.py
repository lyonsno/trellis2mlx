#!/usr/bin/env python3
"""Benchmark TRELLIS2MLX flow-model quantization variants.

The benchmark intentionally isolates flow stages touched by ``generate.py
--quantize``. It writes an incremental JSON report so failed runs still leave
route identity and the last trustworthy phase behind.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "trellis2mlx.quant_bench.v1"
VARIANT_BITS = {
    "fp16": 0,
    "int8": 8,
    "int4": 4,
}
STAGES = ("ss-flow", "slat-flow")
DEFAULT_CHECKPOINT_ROOT = (
    "~/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/"
    "snapshots/af44b45f2e35a493886929c6d786e563ec68364d/ckpts/"
)
CHECKPOINT_FILES = {
    "ss-flow": "ss_flow_img_dit_1_3B_64_bf16.safetensors",
    "slat-flow": "slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
}


def parse_variants(value: str) -> list[tuple[str, int]]:
    variants: list[tuple[str, int]] = []
    for raw in value.split(","):
        label = raw.strip().lower()
        if not label:
            continue
        if label not in VARIANT_BITS:
            raise ValueError(f"unknown quantization variant: {label}")
        variants.append((label, VARIANT_BITS[label]))
    if not variants:
        raise ValueError("at least one quantization variant is required")
    return variants


def parse_stages(value: str) -> list[str]:
    stages: list[str] = []
    for raw in value.split(","):
        stage = raw.strip().lower()
        if not stage:
            continue
        if stage not in STAGES:
            raise ValueError(f"unknown benchmark stage: {stage}")
        stages.append(stage)
    if not stages:
        raise ValueError("at least one benchmark stage is required")
    return stages


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _variant_arg(value: str) -> str:
    try:
        parse_variants(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


def _stage_arg(value: str) -> str:
    try:
        parse_stages(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


def default_report_path() -> Path:
    run_id = time.strftime("%Y%m%d-%H%M%S")
    unique = f"{os.getpid()}-{time.time_ns()}"
    return Path("/tmp") / f"trellis2mlx-quant-bench-{run_id}-{unique}" / "report.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Input image for native MLX DINOv3 conditioning")
    parser.add_argument("--report", type=Path, default=None, help="JSON report path")
    parser.add_argument("--overwrite-report", action="store_true", help="Overwrite an existing JSON report")
    parser.add_argument("--repo-root", type=Path, default=None, help="Effective trellis2mlx repo root")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path(os.environ.get("TRELLIS2MLX_CHECKPOINT_ROOT", DEFAULT_CHECKPOINT_ROOT)).expanduser(),
        help="Directory containing converted TRELLIS.2-4B checkpoint safetensors",
    )
    parser.add_argument("--variants", type=_variant_arg, default="fp16,int8,int4")
    parser.add_argument("--stages", type=_stage_arg, default="ss-flow,slat-flow")
    parser.add_argument("--slat-tokens", type=_positive_int, default=12043)
    parser.add_argument("--warmup-steps", type=_positive_int, default=1)
    parser.add_argument("--timed-steps", type=_positive_int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=_positive_int, default=512)
    parser.add_argument("--random-conditioning-tokens", type=_positive_int, default=10)
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


def _host_identity() -> dict[str, Any]:
    try:
        import mlx.core as mx

        mlx_version = getattr(mx, "__version__", "unknown")
        default_device = str(mx.default_device())
    except Exception as exc:
        mlx_version = f"unavailable: {exc}"
        default_device = None

    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "chip": _run_text(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "mem_bytes": _run_text(["sysctl", "-n", "hw.memsize"]),
        "mlx_version": mlx_version,
        "default_device": default_device,
    }


def _repo_identity(repo_root: Path) -> dict[str, Any]:
    return {
        "root": str(repo_root),
        "head": _run_text(["git", "rev-parse", "HEAD"], cwd=repo_root),
        "status_short": _run_text(["git", "status", "--short"], cwd=repo_root),
    }


def checkpoint_path(args: argparse.Namespace, stage: str) -> Path:
    return args.checkpoint_root / CHECKPOINT_FILES[stage]


def _checkpoint_identity(args: argparse.Namespace, stages: list[str]) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for stage in stages:
        path = checkpoint_path(args, stage)
        files[stage] = {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
        }
    return {
        "root": str(args.checkpoint_root),
        "files": files,
    }


def initial_report(args: argparse.Namespace, command_line: list[str] | None = None) -> dict[str, Any]:
    variants = parse_variants(args.variants)
    stages = parse_stages(args.stages)
    image_path = args.image
    events = []
    if image_path is None:
        events.append({
            "level": "warning",
            "code": "random_conditioning_not_representative",
        })
    return {
        "schema": SCHEMA,
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command_line": command_line or sys.argv,
        "repo": _repo_identity(args.repo_root),
        "host": _host_identity(),
        "checkpoints": _checkpoint_identity(args, stages),
        "asset": {
            "path": str(image_path) if image_path is not None else None,
            "exists": image_path.exists() if image_path is not None else None,
            "size_bytes": image_path.stat().st_size if image_path is not None and image_path.exists() else None,
        },
        "settings": {
            "seed": args.seed,
            "warmup_steps": args.warmup_steps,
            "timed_steps": args.timed_steps,
            "variants": [label for label, _ in variants],
            "variant_bits": {label: bits for label, bits in variants},
            "stages": stages,
            "slat_tokens": args.slat_tokens,
            "image_size": args.image_size,
            "random_conditioning_tokens": args.random_conditioning_tokens,
        },
        "results": [],
        "events": events,
    }


def mark_failure(
    report: dict[str, Any],
    report_path: Path,
    *,
    phase: str,
    error: str,
    extra: dict[str, Any] | None = None,
) -> None:
    existed_before = report_path.exists()
    report["status"] = "failed"
    report["phase"] = phase
    report["error"] = error
    report["last_trustworthy_evidence"] = {
        "report_existed_before_write": existed_before,
        "result_count": len(report.get("results", [])),
    }
    if extra:
        report.update(extra)
    write_report(report_path, report)


def _load_conditioning(args: argparse.Namespace, report: dict[str, Any], report_path: Path):
    import mlx.core as mx

    if args.image is None:
        print(
            "warning: using random conditioning; timings are not representative of native-DINO image conditioning",
            file=sys.stderr,
            flush=True,
        )
        mx.random.seed(args.seed)
        cond = mx.random.normal((1, args.random_conditioning_tokens, 1024))
        mx.eval(cond)
        report["asset"]["conditioning_route"] = "random"
        report["asset"]["feature_shape"] = list(cond.shape)
        report["asset"]["feature_dtype"] = str(cond.dtype)
        write_report(report_path, report)
        return np.array(cond)

    if not args.image.exists():
        raise FileNotFoundError(f"input image does not exist: {args.image}")

    from trellmlx.models.dinov3 import extract_features

    t0 = time.perf_counter()
    cond = extract_features(str(args.image), image_size=args.image_size)
    mx.eval(cond)
    cond_np = np.array(cond)
    report["asset"]["conditioning_route"] = "native_mlx_dinov3"
    report["asset"]["feature_shape"] = list(cond_np.shape)
    report["asset"]["feature_dtype"] = str(cond.dtype)
    report["asset"]["feature_time_s"] = time.perf_counter() - t0
    write_report(report_path, report)
    return cond_np


def add_result_warnings(result: dict[str, Any]) -> None:
    if result.get("status") != "ok":
        return
    warnings = result.setdefault("warnings", [])
    output_std = result.get("output_std")
    if output_std is not None and output_std <= 1e-12:
        warnings.append("degenerate_output_std")
    sample_time = result.get("sample_time_s")
    time_per_step = result.get("sample_time_per_step_s")
    if (
        sample_time is not None
        and time_per_step is not None
        and (sample_time <= 0 or time_per_step <= 0)
    ):
        warnings.append("implausible_sample_time")
    if not warnings:
        result.pop("warnings", None)


def _cleanup_runtime(model=None) -> None:
    try:
        from trellmlx.cleanup import cleanup_model, cleanup

        if model is not None:
            cleanup_model(model)
        cleanup()
    except Exception:
        pass
    gc.collect()


def _time_ss_flow(*, args: argparse.Namespace, label: str, bits: int, cond_np: np.ndarray) -> dict[str, Any]:
    import mlx.core as mx

    from trellmlx.models.sparse_structure_flow import SparseStructureFlowModel
    from trellmlx.quantize import quantize_model
    from trellmlx.samplers import flow_euler_sample
    from trellmlx.weight_loader import load_weights

    result: dict[str, Any] = {
        "stage": "ss-flow",
        "variant": label,
        "bits": bits,
        "status": "running",
    }
    model = None
    try:
        mx.random.seed(args.seed)
        t0 = time.perf_counter()
        model = SparseStructureFlowModel()
        load_weights(
            model,
            str(checkpoint_path(args, "ss-flow")),
            verbose=False,
        )
        result["load_time_s"] = time.perf_counter() - t0
        result["resolution"] = model.resolution
        result["in_channels"] = model.in_channels

        if bits:
            t0 = time.perf_counter()
            quantize_model(model, bits=bits)
            result["quantize_time_s"] = time.perf_counter() - t0
        else:
            result["quantize_time_s"] = 0.0

        cond = mx.array(cond_np)
        neg_cond = mx.zeros_like(cond)
        noise = mx.random.normal((1, model.in_channels, model.resolution, model.resolution, model.resolution))
        mx.eval(noise, cond, neg_cond)

        t0 = time.perf_counter()
        warm = flow_euler_sample(model, noise, cond, neg_cond, steps=args.warmup_steps, verbose=False)
        mx.eval(warm)
        result["warmup_time_s"] = time.perf_counter() - t0

        mx.random.seed(args.seed)
        noise = mx.random.normal((1, model.in_channels, model.resolution, model.resolution, model.resolution))
        mx.eval(noise)
        t0 = time.perf_counter()
        out = flow_euler_sample(model, noise, cond, neg_cond, steps=args.timed_steps, verbose=False)
        mx.eval(out)
        result["sample_time_s"] = time.perf_counter() - t0
        result["sample_steps"] = args.timed_steps
        result["sample_time_per_step_s"] = result["sample_time_s"] / args.timed_steps
        out_np = np.array(out)
        result["output_shape"] = list(out_np.shape)
        result["output_mean"] = float(out_np.mean())
        result["output_std"] = float(out_np.std())
        result["status"] = "ok"
        return result
    except Exception as exc:
        result["status"] = "failed"
        result["failure_phase"] = "ss-flow"
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        return result
    finally:
        _cleanup_runtime(model)


def _synthetic_slat_coords(tokens: int) -> np.ndarray:
    index = np.arange(tokens, dtype=np.int32)
    return np.stack([
        index % 64,
        (index // 64) % 64,
        (index // (64 * 64)) % 64,
    ], axis=1)


def _time_slat_flow(*, args: argparse.Namespace, label: str, bits: int, cond_np: np.ndarray) -> dict[str, Any]:
    import mlx.core as mx

    from trellmlx.models.slat_flow import SLatFlowModel
    from trellmlx.quantize import quantize_model
    from trellmlx.samplers import flow_euler_sample
    from trellmlx.weight_loader import load_weights

    result: dict[str, Any] = {
        "stage": "slat-flow",
        "variant": label,
        "bits": bits,
        "tokens": args.slat_tokens,
        "status": "running",
    }
    model = None
    try:
        mx.random.seed(args.seed)
        t0 = time.perf_counter()
        model = SLatFlowModel()
        load_weights(
            model,
            str(checkpoint_path(args, "slat-flow")),
            verbose=False,
        )
        result["load_time_s"] = time.perf_counter() - t0

        if bits:
            t0 = time.perf_counter()
            quantize_model(model, bits=bits)
            result["quantize_time_s"] = time.perf_counter() - t0
        else:
            result["quantize_time_s"] = 0.0

        cond = mx.array(cond_np)
        neg_cond = mx.zeros_like(cond)
        coords = mx.array(_synthetic_slat_coords(args.slat_tokens))
        noise = mx.random.normal((args.slat_tokens, 32))
        mx.eval(noise, coords, cond, neg_cond)

        t0 = time.perf_counter()
        warm = flow_euler_sample(
            model, noise, cond, neg_cond, steps=args.warmup_steps, verbose=False, coords=coords,
        )
        mx.eval(warm)
        result["warmup_time_s"] = time.perf_counter() - t0

        mx.random.seed(args.seed)
        noise = mx.random.normal((args.slat_tokens, 32))
        mx.eval(noise)
        t0 = time.perf_counter()
        out = flow_euler_sample(
            model, noise, cond, neg_cond, steps=args.timed_steps, verbose=False, coords=coords,
        )
        mx.eval(out)
        result["sample_time_s"] = time.perf_counter() - t0
        result["sample_steps"] = args.timed_steps
        result["sample_time_per_step_s"] = result["sample_time_s"] / args.timed_steps
        out_np = np.array(out)
        result["output_shape"] = list(out_np.shape)
        result["output_mean"] = float(out_np.mean())
        result["output_std"] = float(out_np.std())
        result["status"] = "ok"
        return result
    except Exception as exc:
        result["status"] = "failed"
        result["failure_phase"] = "slat-flow"
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        return result
    finally:
        _cleanup_runtime(model)


def run_benchmark(args: argparse.Namespace, report: dict[str, Any], report_path: Path) -> dict[str, Any]:
    cond_np = _load_conditioning(args, report, report_path)
    variants = parse_variants(args.variants)
    stages = parse_stages(args.stages)
    for stage in stages:
        for label, bits in variants:
            print(f"=== {stage} {label} ===", flush=True)
            if stage == "ss-flow":
                result = _time_ss_flow(args=args, label=label, bits=bits, cond_np=cond_np)
            else:
                result = _time_slat_flow(args=args, label=label, bits=bits, cond_np=cond_np)
            add_result_warnings(result)
            report["results"].append(result)
            write_report(report_path, report)
            if result["status"] == "ok":
                print(
                    f"  {result['sample_time_per_step_s']:.2f}s/step "
                    f"(load {result['load_time_s']:.2f}s, quant {result['quantize_time_s']:.2f}s)",
                    flush=True,
                )
            else:
                print(f"  failed: {result.get('error')}", flush=True)
    report["status"] = "ok" if all(r["status"] == "ok" for r in report["results"]) else "partial"
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_report(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report_path = args.report
    try:
        ensure_report_writable(report_path, overwrite=args.overwrite_report)
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    report = initial_report(args)
    write_report(report_path, report)
    try:
        run_benchmark(args, report, report_path)
    except Exception as exc:
        mark_failure(
            report,
            report_path,
            phase="benchmark",
            error=repr(exc),
            extra={"traceback": traceback.format_exc()},
        )
        print(f"benchmark failed: {exc}", file=sys.stderr)
        print(f"FINAL_REPORT={report_path}", flush=True)
        return 1
    print(f"FINAL_REPORT={report_path}", flush=True)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
