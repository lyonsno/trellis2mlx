#!/usr/bin/env python3
"""Capture the full CUDA center-kernel GEMM at the first decoder fork."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cuda_decoder_block0_gemm_witness import (  # noqa: E402
    EXPECTED_DEVICE,
    EXPECTED_TORCH,
    _canonical_sha256,
    _load_npz,
    _to_cpu_numpy_preserve_dtype,
    _write_json,
    _write_npz_atomic,
    sha256_file,
    validate_witness_arrays,
)


SCHEMA = "trellis2mlx.cuda_decoder_block0_center_gemm_oracle.v1"
OUTPUT_NAME = "cuda_default_full_product"


def analyze_full_product(
    *,
    product: np.ndarray,
    witness: dict[str, Any],
) -> dict[str, Any]:
    value = np.asarray(product)
    expected_shape = witness["torso_input"].shape[:-1] + (
        witness["center_weight"].shape[1],
    )
    if value.dtype != np.float16:
        raise ValueError(
            f"full center GEMM product must have dtype float16, got {value.dtype}"
        )
    if value.shape != expected_shape:
        raise ValueError(
            f"full center GEMM product must have shape {expected_shape}, "
            f"got {value.shape}"
        )
    if not np.all(np.isfinite(value)):
        raise ValueError("full center GEMM product contains non-finite values")
    row_index = int(witness["row_index"])
    selected_plus_bias = (
        value[row_index] + witness["bias"]
    ).astype(np.float16)
    if not np.array_equal(
        selected_plus_bias,
        witness["source_trace_row"],
    ):
        raise ValueError(
            "full CUDA center GEMM selected row plus bias does not reproduce "
            "source trace"
        )
    return {
        "selected_plus_bias_exact_source": True,
        "selected_row_index": row_index,
        "matrix_shape": list(value.shape),
        "matrix_dtype": str(value.dtype),
        "finite": True,
    }


def _run_cuda(
    witness: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    if torch.__version__ != EXPECTED_TORCH:
        raise ValueError(
            f"Torch route must be {EXPECTED_TORCH}, got {torch.__version__}"
        )
    if not torch.cuda.is_available():
        raise ValueError("CUDA route is unavailable")
    device = torch.cuda.get_device_name(0)
    if device != EXPECTED_DEVICE:
        raise ValueError(
            f"CUDA device route must be {EXPECTED_DEVICE}, got {device}"
        )
    reduction = bool(
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    )
    if reduction is not True:
        raise ValueError(
            "default CUDA route disabled FP16 reduced-precision reduction"
        )

    x = torch.from_numpy(witness["torso_input"].copy()).to(
        device="cuda",
        dtype=torch.float16,
    )
    weight = torch.from_numpy(witness["center_weight"].copy()).to(
        device="cuda",
        dtype=torch.float16,
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        product = x @ weight
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return _to_cpu_numpy_preserve_dtype(product), {
        "torch": torch.__version__,
        "cuda_device": device,
        "allow_fp16_reduced_precision_reduction": reduction,
        "gemm_seconds": elapsed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witness", required=True, type=Path)
    parser.add_argument("--expected-witness-sha256", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int, default=7697)
    parser.add_argument("--channels", type=int, default=1024)
    parser.add_argument("--expected-row", type=int, default=7693)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "failure_phase": "request_validation",
        "last_trustworthy_phase": None,
        "requested_witness": {
            "path": str(args.witness),
            "sha256": args.expected_witness_sha256,
        },
        "primary_output": {
            "path": str(args.output_npz),
            "exists": False,
        },
    }
    try:
        resolved = [
            args.witness.resolve(),
            args.output_json.resolve(),
            args.output_npz.resolve(),
        ]
        if len(set(resolved)) != len(resolved):
            raise ValueError("witness and output paths must be distinct")
        args.output_json.unlink(missing_ok=True)
        args.output_npz.unlink(missing_ok=True)
        report["last_trustworthy_phase"] = "output_paths_validated"
        report["failure_phase"] = "input_validation"
        if not _canonical_sha256(args.expected_witness_sha256):
            raise ValueError(
                "expected witness sha256 must be canonical lowercase hex"
            )
        actual_digest = sha256_file(args.witness)
        if actual_digest != args.expected_witness_sha256:
            raise ValueError(
                "witness sha256 mismatch: "
                f"expected {args.expected_witness_sha256}, got {actual_digest}"
            )
        witness = validate_witness_arrays(
            _load_npz(args.witness),
            expected_rows=args.expected_rows,
            channels=args.channels,
            expected_row=args.expected_row,
        )
        report["witness"] = {
            "path": str(args.witness),
            "sha256": actual_digest,
            "size_bytes": args.witness.stat().st_size,
            "route_identity": witness["route_identity"],
            "rows": args.expected_rows,
            "channels": args.channels,
            "row_index": args.expected_row,
            "neighbor_count": witness["neighbor_count"],
        }
        report["last_trustworthy_phase"] = "input_validated"
        report["failure_phase"] = "cuda_execution"
        product, runtime = _run_cuda(witness)
        analysis = analyze_full_product(
            product=product,
            witness=witness,
        )
        report["last_trustworthy_phase"] = "cuda_authenticated"
        report["failure_phase"] = "output_publication"
        outputs = {OUTPUT_NAME: product}
        _write_npz_atomic(args.output_npz, outputs)
        reopened = _load_npz(args.output_npz)
        if (
            set(reopened) != {OUTPUT_NAME}
            or not np.array_equal(reopened[OUTPUT_NAME], product)
            or reopened[OUTPUT_NAME].dtype != product.dtype
        ):
            raise ValueError("published NPZ does not reopen exactly")
        report.update(
            {
                "status": "done",
                "failure_phase": None,
                "last_trustworthy_phase": "output_reopened_exact",
                "runtime": runtime,
                "analysis": analysis,
                "primary_output": {
                    "path": str(args.output_npz),
                    "exists": True,
                    "sha256": sha256_file(args.output_npz),
                    "size_bytes": args.output_npz.stat().st_size,
                    "keys": [OUTPUT_NAME],
                    "reopened_exact": True,
                },
            }
        )
    except Exception as exc:
        args.output_npz.unlink(missing_ok=True)
        report.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "primary_output": {
                    "path": str(args.output_npz),
                    "exists": False,
                },
            }
        )
        _write_json(args.output_json, report)
        return 1
    _write_json(args.output_json, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
