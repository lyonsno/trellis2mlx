#!/usr/bin/env python3
"""Replay the decoder level-two width-128 LayerNorm on Metal."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.decoder_level2_norm2_trace_contract import (
    TRACE_SCHEMA,
    load_decoder_level2_norm2_trace,
)
from scripts.run_mlx_decoder_level1_trace import (
    _failure_sibling,
    _load_turing_rsqrt_lut,
    _sha256_file,
    _validate_digest,
    _write_report,
)

SCHEMA = "trellis2mlx.mlx_decoder_level2_norm2_replay.v1"
ROUTE = "mlx-shape-decoder-level2-norm2-diagnostic-replay"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-trace", required=True, type=Path)
    parser.add_argument("--expected-source-trace-sha256", required=True)
    parser.add_argument("--turing-rsqrt-lut", required=True, type=Path)
    parser.add_argument("--expected-turing-rsqrt-lut-sha256", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    return parser


def compare_norm2(
    source: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    source = np.asarray(source)
    candidate = np.asarray(candidate)
    for values, label in ((source, "source"), (candidate, "candidate")):
        if values.dtype != np.dtype(np.float16):
            raise ValueError(
                f"{label} norm2 must have dtype float16, got {values.dtype}"
            )
        if values.ndim != 2 or values.shape[1] != 128 or values.shape[0] == 0:
            raise ValueError(
                f"{label} norm2 must have nonempty shape [N, 128], "
                f"got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{label} norm2 contains non-finite values")
    if source.shape != candidate.shape:
        raise ValueError(
            "source and candidate norm2 shapes differ: "
            f"{source.shape} != {candidate.shape}"
        )
    delta = candidate.astype(np.float32) - source.astype(np.float32)
    absolute = np.abs(delta)
    return {
        "shape": [int(value) for value in source.shape],
        "dtype": "float16",
        "source_array_sha256": hashlib.sha256(
            np.ascontiguousarray(source).tobytes()
        ).hexdigest(),
        "candidate_array_sha256": hashlib.sha256(
            np.ascontiguousarray(candidate).tobytes()
        ).hexdigest(),
        "exact": bool(np.array_equal(source, candidate)),
        "nonzero_count": int(np.count_nonzero(delta)),
        "mean_abs": float(np.mean(absolute, dtype=np.float64)),
        "rms": float(
            np.sqrt(np.mean(np.square(delta), dtype=np.float64))
        ),
        "max_abs": float(np.max(absolute)),
    }


def _run_mlx_candidate(
    source_arrays: dict[str, np.ndarray],
    turing_rsqrt_lut: np.ndarray,
    turing_rsqrt_lut_digest: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    import mlx.core as mx

    from trellmlx.decoder_turing_layernorm import (
        turing_layernorm_noaffine_fp16,
    )

    mx.set_default_device(mx.gpu)
    effective_device = str(mx.default_device())
    if "gpu" not in effective_device.lower():
        raise RuntimeError(
            f"norm2 replay requires Metal GPU, got {effective_device}"
        )
    input_values = mx.array(source_arrays["level2_upsample_h_c2s"])
    lut = mx.array(turing_rsqrt_lut)
    first = turing_layernorm_noaffine_fp16(
        input_values,
        lut,
        1e-6,
    )
    second = turing_layernorm_noaffine_fp16(
        input_values,
        lut,
        1e-6,
    )
    mx.eval(first, second)
    first_values = np.ascontiguousarray(np.asarray(first))
    second_values = np.ascontiguousarray(np.asarray(second))
    if not np.array_equal(first_values, second_values):
        raise RuntimeError("width-128 Metal LayerNorm replay is nondeterministic")
    return first_values, {
        "route": ROUTE,
        "device_type": "metal",
        "effective_device": effective_device,
        "backend": "cuda-welford-turing-t4",
        "cuda_source_tag": "pytorch-v2.10.0",
        "cuda_source_kernel": "vectorized_layer_norm_kernel",
        "cuda_architecture": "sm_75",
        "cuda_device_anchor": "Tesla T4",
        "turing_rsqrt_lut_artifact_sha256_attested": (
            turing_rsqrt_lut_digest
        ),
        "candidate_contract": {
            "input_dtype": "float16",
            "hidden_width": 128,
            "affine": False,
            "eps": 1e-6,
            "reduction": {
                "threads": 128,
                "warps": 4,
                "vector_width": 4,
                "active_values_per_thread": 4,
                "average_values_per_launched_thread": 1,
                "active_vector_threads": 32,
                "inactive_vector_threads": 96,
                "accumulator_dtype": "float32",
            },
        },
        "production_enrollment": False,
        "evidence_scope": (
            "diagnostic width-128 schedule candidate pending exact "
            "source-CUDA comparison"
        ),
        "deterministic_replay": True,
    }


def _write_candidate_npz(
    path: Path,
    candidate: np.ndarray,
) -> dict[str, Any]:
    candidate = np.ascontiguousarray(candidate)
    if (
        candidate.dtype != np.dtype(np.float16)
        or candidate.ndim != 2
        or candidate.shape[0] == 0
        or candidate.shape[1] != 128
        or not np.isfinite(candidate).all()
    ):
        raise ValueError(
            "candidate output must be finite nonempty float16[N, 128]"
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
        np.savez(
            temporary_path,
            level2_upsample_norm2_candidate=candidate,
        )
        with np.load(temporary_path, allow_pickle=False) as archive:
            if archive.files != ["level2_upsample_norm2_candidate"]:
                raise ValueError("candidate NPZ member set changed after write")
            reopened = np.asarray(
                archive["level2_upsample_norm2_candidate"]
            )
        if not np.array_equal(reopened, candidate):
            raise ValueError("candidate output changed after write")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {
        "exists": True,
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "array_name": "level2_upsample_norm2_candidate",
        "shape": [int(value) for value in candidate.shape],
        "dtype": "float16",
        "reopened_exact": True,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_path = args.output_json
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "failure_phase": None,
        "last_trustworthy_phase": "request_received",
        "requested_route": {
            "route": ROUTE,
            "device_type": "metal",
            "backend": "cuda-welford-turing-t4",
            "source_trace_schema": TRACE_SCHEMA,
            "candidate_contract": {
                "hidden_width": 128,
                "affine": False,
            },
            "production_enrollment": False,
        },
        "effective_route": None,
        "requested_report_path": str(args.output_json),
        "effective_report_path": str(report_path),
        "stale_primary_invalidated": False,
        "primary_output": {
            "path": str(args.output_npz),
            "exists": False,
        },
    }
    phase = "input_validation"
    primary_is_safe_to_remove = False
    try:
        inputs = {
            args.source_trace.resolve(),
            args.turing_rsqrt_lut.resolve(),
        }
        protected_reports = inputs | {args.output_npz.resolve()}
        report_collision = args.output_json.resolve() in protected_reports
        if report_collision:
            report_path = _failure_sibling(
                args.output_json,
                protected_reports,
            )
            report["effective_report_path"] = str(report_path)
        if args.output_npz.resolve() in inputs:
            raise ValueError("--output-npz collides with an input path")
        primary_is_safe_to_remove = True
        stale_primary_existed = args.output_npz.exists()
        args.output_npz.unlink(missing_ok=True)
        report["stale_primary_invalidated"] = stale_primary_existed
        if report_collision:
            raise ValueError(
                "--output-json collides with an input or primary path"
            )
        _validate_digest(
            args.expected_source_trace_sha256,
            "--expected-source-trace-sha256",
        )
        _validate_digest(
            args.expected_turing_rsqrt_lut_sha256,
            "--expected-turing-rsqrt-lut-sha256",
        )
        if not args.source_trace.is_file():
            raise FileNotFoundError(
                f"source trace does not exist: {args.source_trace}"
            )
        source_digest = _sha256_file(args.source_trace)
        if source_digest != args.expected_source_trace_sha256:
            raise ValueError(
                "source trace sha256 mismatch: "
                f"expected={args.expected_source_trace_sha256}, "
                f"actual={source_digest}"
            )
        source_arrays = load_decoder_level2_norm2_trace(
            args.source_trace
        )
        report["source_trace"] = {
            "path": str(args.source_trace.resolve()),
            "sha256": source_digest,
            "schema": TRACE_SCHEMA,
            "rows": int(source_arrays["level3_child_coords"].shape[0]),
            "channels": 128,
        }
        report["last_trustworthy_phase"] = phase

        phase = "turing_rsqrt_lut_validation"
        turing_rsqrt_lut, lut_identity = _load_turing_rsqrt_lut(
            args.turing_rsqrt_lut,
            args.expected_turing_rsqrt_lut_sha256,
        )
        report["turing_rsqrt_lut"] = lut_identity
        report["last_trustworthy_phase"] = phase

        phase = "metal_replay"
        candidate, effective_route = _run_mlx_candidate(
            source_arrays,
            turing_rsqrt_lut,
            args.expected_turing_rsqrt_lut_sha256,
        )
        report["effective_route"] = effective_route
        report["last_trustworthy_phase"] = phase

        phase = "comparison"
        comparison = compare_norm2(
            source_arrays["level2_upsample_norm2"],
            candidate,
        )
        report["comparison"] = comparison
        report["last_trustworthy_phase"] = phase

        phase = "artifact_write"
        primary = _write_candidate_npz(args.output_npz, candidate)
        report["primary_output"] = {
            "path": str(args.output_npz),
            **primary,
        }
        report.update(
            {
                "status": "done",
                "failure_phase": None,
                "last_trustworthy_phase": phase,
            }
        )
        _write_report(report_path, report)
        return 0
    except Exception as exc:
        if primary_is_safe_to_remove:
            args.output_npz.unlink(missing_ok=True)
        primary_exists = args.output_npz.exists()
        report.update(
            {
                "status": "failed",
                "failure_phase": phase,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "primary_output": {
                    "path": str(args.output_npz),
                    "exists": primary_exists,
                    "protected_input_collision": (
                        primary_exists and not primary_is_safe_to_remove
                    ),
                },
            }
        )
        _write_report(report_path, report)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
