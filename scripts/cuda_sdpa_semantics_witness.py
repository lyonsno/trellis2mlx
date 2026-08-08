#!/usr/bin/env python3
"""Identify CUDA SDPA semantics for an authenticated sparse self-attention witness."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import traceback
from typing import Any
import warnings

import numpy as np


SCHEMA = "trellis2mlx.cuda_sdpa_semantics_witness.v1"
EXPECTED_TORCH = "2.10.0+cu128"
EXPECTED_DEVICE = "Tesla T4"
EXPECTED_SHAPE = (4096, 12, 128)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def requested_witness_identity(expected_sha256: str | None) -> dict[str, str]:
    if expected_sha256 is None or re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ) is None:
        raise ValueError(
            "expected witness sha256 must be 64 lowercase hexadecimal"
        )
    return {"sha256": expected_sha256}


def validate_runtime(
    *,
    torch_version: str,
    cuda_available: bool,
    cuda_device: str | None,
) -> None:
    if torch_version != EXPECTED_TORCH:
        raise ValueError(
            f"expected Torch {EXPECTED_TORCH}, got {torch_version}"
        )
    if not cuda_available:
        raise ValueError("CUDA is unavailable")
    if cuda_device != EXPECTED_DEVICE:
        raise ValueError(
            f"expected CUDA device {EXPECTED_DEVICE}, got {cuda_device}"
        )


def _exact_metric(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    if actual.shape != expected.shape:
        return {
            "shape_match": False,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "exact": False,
        }
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {
        "shape_match": True,
        "exact": bool(np.array_equal(actual, expected)),
        "nonzero": int(np.count_nonzero(actual != expected)),
        "mean_abs": float(delta.mean()) if delta.size else 0.0,
        "max_abs": float(delta.max(initial=0.0)),
        "rms": float(np.sqrt(np.mean(np.square(delta))))
        if delta.size
        else 0.0,
    }


def analyze_sdpa_results(
    *,
    default_output: np.ndarray,
    expected_output: np.ndarray,
    candidate_outputs: dict[str, np.ndarray],
) -> dict[str, Any]:
    default_metric = _exact_metric(default_output, expected_output)
    if not default_metric.get("exact", False):
        raise ValueError(
            "default CUDA SDPA does not replay source: "
            f"{default_metric}"
        )

    candidate_metrics = {
        name: _exact_metric(output, default_output)
        for name, output in sorted(candidate_outputs.items())
    }
    exact_matches = sorted(
        name
        for name, metric in candidate_metrics.items()
        if metric.get("exact", False)
    )
    if not exact_matches:
        raise ValueError(
            "no forced CUDA SDPA backend exactly reproduces the authenticated "
            "default output"
        )
    return {
        "default_self_authentication": default_metric,
        "candidates": candidate_metrics,
        "exact_default_matches": exact_matches,
        "backend_identification": "unique"
        if len(exact_matches) == 1
        else "ambiguous",
    }


def _load_witness(path: Path) -> dict[str, np.ndarray]:
    required = ("q", "k", "v", "expected_output")
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise ValueError(f"witness archive missing arrays: {missing}")
        arrays = {
            name: np.ascontiguousarray(np.asarray(archive[name]))
            for name in required
        }

    for name, array in arrays.items():
        if array.shape != EXPECTED_SHAPE:
            raise ValueError(
                f"{name} shape {array.shape} != {EXPECTED_SHAPE}"
            )
        if array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"{name} must contain finite float32 values")
        if np.any(array.view(np.uint32) & np.uint32(0xFFFF)):
            raise ValueError(f"{name} must be exactly BF16-representable")
    return arrays


def _backend_specs(torch: Any) -> tuple[tuple[str, Any], ...]:
    from torch.nn.attention import SDPBackend

    specs: list[tuple[str, Any]] = [
        ("flash_attention", SDPBackend.FLASH_ATTENTION),
        ("efficient_attention", SDPBackend.EFFICIENT_ATTENTION),
        ("math", SDPBackend.MATH),
    ]
    cudnn = getattr(SDPBackend, "CUDNN_ATTENTION", None)
    if cudnn is not None:
        specs.append(("cudnn_attention", cudnn))
    return tuple(specs)


def _run_variant(
    torch: Any,
    *,
    q: Any,
    k: Any,
    v: Any,
    backend: Any | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch.nn.functional as functional
    from torch.profiler import ProfilerActivity, profile

    started = time.perf_counter()
    caught: list[str] = []
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        ) as profiler:
            if backend is None:
                output = functional.scaled_dot_product_attention(q, k, v)
            else:
                from torch.nn.attention import sdpa_kernel

                with sdpa_kernel(backends=[backend]):
                    output = functional.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()
        caught = [str(item.message) for item in observed]

    output_np = (
        output.squeeze(0)
        .transpose(0, 1)
        .to(dtype=torch.float32)
        .contiguous()
        .cpu()
        .numpy()
    )
    events = sorted(
        {
            event.key
            for event in profiler.key_averages()
            if any(
                token in event.key.lower()
                for token in (
                    "attention",
                    "cudnn",
                    "efficient",
                    "flash",
                    "scaled_dot_product",
                )
            )
        }
    )
    return output_np, {
        "elapsed_seconds": time.perf_counter() - started,
        "profiler_events": events,
        "warnings": caught,
    }


def _run_cuda(
    torch: Any,
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    tensors = {
        name: torch.from_numpy(arrays[name])
        .to(device="cuda", dtype=torch.bfloat16)
        .transpose(0, 1)
        .unsqueeze(0)
        for name in ("q", "k", "v")
    }
    outputs: dict[str, np.ndarray] = {}
    variants: dict[str, Any] = {}

    default_output, default_details = _run_variant(
        torch, backend=None, **tensors
    )
    outputs["default"] = default_output
    variants["default"] = {
        "status": "done",
        "requested_backend": "default",
        **default_details,
    }

    for name, backend in _backend_specs(torch):
        try:
            output, details = _run_variant(
                torch, backend=backend, **tensors
            )
            outputs[name] = output
            variants[name] = {
                "status": "done",
                "requested_backend": name,
                **details,
            }
        except Exception as exc:
            variants[name] = {
                "status": "unavailable",
                "requested_backend": name,
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            torch.cuda.empty_cache()

    return outputs, variants


def _write_result(
    path: Path,
    *,
    arrays: dict[str, np.ndarray],
    outputs: dict[str, np.ndarray],
) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    payload = {
        "expected_output": arrays["expected_output"],
        **{
            f"{name}_output": np.asarray(output, dtype=np.float32)
            for name, output in outputs.items()
        },
    }
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witness", required=True, type=Path)
    parser.add_argument("--expected-witness-sha256", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    started = time.time()
    phase = "argument_parsing"
    last_trustworthy = "process_started"
    args: argparse.Namespace | None = None
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "failure_phase": phase,
        "last_trustworthy_phase": last_trustworthy,
    }
    try:
        args = _parse_args(argv)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_npz.parent.mkdir(parents=True, exist_ok=True)
        args.output_npz.unlink(missing_ok=True)
        requested_identity = requested_witness_identity(
            args.expected_witness_sha256
        )
        report["witness_identity_requested"] = requested_identity
        phase = "runtime_validation"
        last_trustworthy = "output_paths_validated"

        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_device = (
            torch.cuda.get_device_name(0) if cuda_available else None
        )
        report.update(
            {
                "torch": torch.__version__,
                "cuda_available": cuda_available,
                "cuda_device": cuda_device,
            }
        )
        validate_runtime(
            torch_version=torch.__version__,
            cuda_available=cuda_available,
            cuda_device=cuda_device,
        )
        phase = "witness_validation"
        last_trustworthy = "runtime_validated"

        effective_sha = sha256_file(args.witness)
        report["witness_identity_effective"] = {"sha256": effective_sha}
        if effective_sha != requested_identity["sha256"]:
            raise ValueError(
                "witness sha256 mismatch: requested "
                f"{requested_identity['sha256']}, got {effective_sha}"
            )
        arrays = _load_witness(args.witness)
        report["input_arrays"] = {
            name: {
                "sha256": sha256_array(array),
                "shape": list(array.shape),
            }
            for name, array in arrays.items()
        }
        phase = "cuda_execution"
        last_trustworthy = "witness_validated"

        outputs, variants = _run_cuda(torch, arrays)
        phase = "result_write"
        last_trustworthy = "cuda_execution_completed"
        _write_result(args.output_npz, arrays=arrays, outputs=outputs)
        phase = "cuda_self_authentication"
        last_trustworthy = "cuda_output_preserved"

        analysis = analyze_sdpa_results(
            default_output=outputs["default"],
            expected_output=arrays["expected_output"],
            candidate_outputs={
                name: output
                for name, output in outputs.items()
                if name != "default"
            },
        )
        for name, metric in analysis["candidates"].items():
            variants[name]["vs_default"] = metric
        variants["default"]["vs_source"] = analysis[
            "default_self_authentication"
        ]
        report.update(
            {
                "status": "done",
                "failure_phase": None,
                "last_trustworthy_phase": "result_archive_authenticated",
                "analysis": analysis,
                "variants": variants,
                "output_arrays": {
                    name: {
                        "sha256": sha256_array(output),
                        "shape": list(output.shape),
                    }
                    for name, output in outputs.items()
                },
                "primary_output": {
                    "path": str(args.output_npz),
                    "exists": True,
                    "authority": "authenticated",
                    "sha256": sha256_file(args.output_npz),
                    "size_bytes": args.output_npz.stat().st_size,
                },
                "elapsed_seconds": time.time() - started,
            }
        )
        _write_report(args.output_json, report)
        return 0
    except Exception as exc:
        output_exists = bool(args is not None and args.output_npz.exists())
        primary_output: dict[str, Any] = {
            "path": str(args.output_npz) if args is not None else None,
            "exists": output_exists,
        }
        if output_exists and args is not None:
            primary_output.update(
                {
                    "authority": "diagnostic-only",
                    "sha256": sha256_file(args.output_npz),
                    "size_bytes": args.output_npz.stat().st_size,
                }
            )
        report.update(
            {
                "status": "failed",
                "failure_phase": phase,
                "last_trustworthy_phase": last_trustworthy,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "primary_output": primary_output,
                "elapsed_seconds": time.time() - started,
            }
        )
        if args is not None:
            _write_report(args.output_json, report)
        else:
            print(json.dumps(report, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
