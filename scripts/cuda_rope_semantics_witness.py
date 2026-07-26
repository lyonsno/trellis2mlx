#!/usr/bin/env python3
"""Capture CUDA RoPE phase and complex-multiply semantics on a Tesla T4."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import traceback
from typing import Any

import numpy as np


SCHEMA = "trellis2mlx.cuda_rope_semantics_witness.v1"
EXPECTED_TORCH = "2.10.0+cu128"
EXPECTED_DEVICE = "Tesla T4"
EXPECTED_COORDINATE_COUNT = 64
EXPECTED_FREQUENCY_COUNT = 21


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


def _exact_metric(
    actual: np.ndarray, expected: np.ndarray
) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    if actual.shape != expected.shape:
        return {
            "shape_match": False,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
        }
    delta = np.abs(actual - expected)
    return {
        "shape_match": True,
        "nonzero": int(np.count_nonzero(delta)),
        "mean_abs": float(delta.mean(dtype=np.float64)),
        "max_abs": float(delta.max(initial=0.0)),
    }


def analyze_cuda_results(
    *,
    phase_pairs: np.ndarray,
    case_output: np.ndarray,
    expected_case_output: np.ndarray,
    coordinate_count: int,
    frequency_count: int,
) -> dict[str, Any]:
    phase_pairs = np.asarray(phase_pairs)
    expected_shape = (coordinate_count, frequency_count, 2)
    if phase_pairs.shape != expected_shape:
        raise ValueError(
            f"phase-pair shape {phase_pairs.shape} != {expected_shape}"
        )
    phase_finite = bool(np.isfinite(phase_pairs).all())
    if phase_pairs.dtype != np.float32 or not phase_finite:
        raise ValueError("phase pairs must be finite float32 values")

    case_metric = _exact_metric(case_output, expected_case_output)
    if (
        not case_metric.get("shape_match", False)
        or case_metric.get("nonzero") != 0
    ):
        raise ValueError(
            "CUDA RoPE cases do not reproduce source outputs: "
            f"{case_metric}"
        )

    return {
        "phase_pairs": {
            "shape": list(phase_pairs.shape),
            "dtype": str(phase_pairs.dtype),
            "finite": phase_finite,
        },
        "case_self_authentication": case_metric,
    }


def _load_witness(path: Path) -> dict[str, np.ndarray]:
    required = (
        "coordinate_values",
        "frequencies",
        "case_input",
        "case_coordinate_index",
        "case_frequency_index",
        "expected_case_output",
    )
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise ValueError(f"witness archive missing arrays: {missing}")
        arrays = {name: np.asarray(archive[name]) for name in required}

    coordinates = arrays["coordinate_values"]
    frequencies = arrays["frequencies"]
    case_input = arrays["case_input"]
    case_coordinate_index = arrays["case_coordinate_index"]
    case_frequency_index = arrays["case_frequency_index"]
    expected_case_output = arrays["expected_case_output"]

    if coordinates.shape != (EXPECTED_COORDINATE_COUNT,):
        raise ValueError(
            "coordinate_values must contain the complete 0..63 domain"
        )
    if coordinates.dtype != np.int32:
        raise ValueError("coordinate_values must be int32")
    if not np.array_equal(
        coordinates, np.arange(EXPECTED_COORDINATE_COUNT, dtype=np.int32)
    ):
        raise ValueError("coordinate_values must equal int32 0..63")
    if frequencies.shape != (EXPECTED_FREQUENCY_COUNT,):
        raise ValueError("frequencies must contain exactly 21 values")
    if frequencies.dtype != np.float32 or not np.isfinite(frequencies).all():
        raise ValueError("frequencies must be finite float32 values")
    if case_input.ndim != 2 or case_input.shape[1] != 2:
        raise ValueError("case_input must have shape [N, 2]")
    if expected_case_output.shape != case_input.shape:
        raise ValueError("expected_case_output must match case_input shape")
    if case_input.dtype != np.float32 or expected_case_output.dtype != np.float32:
        raise ValueError("case input and output must be float32")
    if not np.isfinite(case_input).all() or not np.isfinite(
        expected_case_output
    ).all():
        raise ValueError("case input and output must be finite")
    if np.any(
        np.ascontiguousarray(case_input).view(np.uint32)
        & np.uint32(0xFFFF)
    ):
        raise ValueError("case_input must be exactly BF16-representable")
    case_count = case_input.shape[0]
    if case_count == 0:
        raise ValueError("witness must contain at least one boundary case")
    if case_coordinate_index.shape != (case_count,):
        raise ValueError("case_coordinate_index must have shape [N]")
    if case_frequency_index.shape != (case_count,):
        raise ValueError("case_frequency_index must have shape [N]")
    if not np.issubdtype(case_coordinate_index.dtype, np.integer):
        raise ValueError("case_coordinate_index must have integer dtype")
    if not np.issubdtype(case_frequency_index.dtype, np.integer):
        raise ValueError("case_frequency_index must have integer dtype")
    if np.any(case_coordinate_index < 0) or np.any(
        case_coordinate_index >= EXPECTED_COORDINATE_COUNT
    ):
        raise ValueError("case coordinate index escapes the phase table")
    if np.any(case_frequency_index < 0) or np.any(
        case_frequency_index >= EXPECTED_FREQUENCY_COUNT
    ):
        raise ValueError("case frequency index escapes the phase table")
    return arrays


def _run_cuda(
    torch: Any, arrays: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = torch.from_numpy(arrays["coordinate_values"]).cuda()
    frequencies = torch.from_numpy(arrays["frequencies"]).cuda()
    angles = torch.outer(coordinates, frequencies)
    phases = torch.polar(torch.ones_like(angles), angles)

    case_input = torch.from_numpy(arrays["case_input"]).to(
        device="cuda", dtype=torch.bfloat16
    )
    coordinate_index = torch.from_numpy(
        arrays["case_coordinate_index"].astype(np.int64, copy=False)
    ).cuda()
    frequency_index = torch.from_numpy(
        arrays["case_frequency_index"].astype(np.int64, copy=False)
    ).cuda()
    case_phases = phases[coordinate_index, frequency_index]
    case_complex = torch.view_as_complex(case_input.float().reshape(-1, 2))
    case_output = torch.view_as_real(case_complex * case_phases)
    case_output = case_output.to(torch.bfloat16).float()
    torch.cuda.synchronize()

    phase_pairs = torch.view_as_real(phases).float().cpu().numpy()
    return phase_pairs, case_output.cpu().numpy()


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _write_result(
    path: Path,
    *,
    phase_pairs: np.ndarray,
    case_output: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            phase_pairs=np.asarray(phase_pairs, dtype=np.float32),
            case_output=np.asarray(case_output, dtype=np.float32),
            coordinate_values=arrays["coordinate_values"],
            frequencies=arrays["frequencies"],
            case_coordinate_index=arrays["case_coordinate_index"],
            case_frequency_index=arrays["case_frequency_index"],
            expected_case_output=arrays["expected_case_output"],
        )
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witness", type=Path, default=Path("rope_witness.npz"))
    parser.add_argument("--expected-witness-sha256")
    parser.add_argument(
        "--output-json", type=Path, default=Path("cuda_rope_result.json")
    )
    parser.add_argument(
        "--output-npz", type=Path, default=Path("cuda_rope_result.npz")
    )
    return parser.parse_args()


def main() -> int:
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
        args = _parse_args()
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_npz.parent.mkdir(parents=True, exist_ok=True)
        args.output_npz.unlink(missing_ok=True)
        phase = "runtime_validation"
        last_trustworthy = "output_paths_validated"

        requested_identity = requested_witness_identity(
            args.expected_witness_sha256
        )
        report["witness_identity_requested"] = requested_identity

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

        actual_sha256 = sha256_file(args.witness)
        report["witness_identity_effective"] = {"sha256": actual_sha256}
        if actual_sha256 != requested_identity["sha256"]:
            raise ValueError(
                "witness sha256 mismatch: "
                f"requested {requested_identity['sha256']}, got {actual_sha256}"
            )
        arrays = _load_witness(args.witness)
        phase = "cuda_execution"
        last_trustworthy = "witness_validated"

        phase_pairs, case_output = _run_cuda(torch, arrays)
        analysis = analyze_cuda_results(
            phase_pairs=phase_pairs,
            case_output=case_output,
            expected_case_output=arrays["expected_case_output"],
            coordinate_count=EXPECTED_COORDINATE_COUNT,
            frequency_count=EXPECTED_FREQUENCY_COUNT,
        )
        phase = "result_write"
        last_trustworthy = "cuda_output_authenticated"

        _write_result(
            args.output_npz,
            phase_pairs=phase_pairs,
            case_output=case_output,
            arrays=arrays,
        )
        report.update(
            {
                "status": "done",
                "failure_phase": None,
                "last_trustworthy_phase": "result_archive_written",
                "analysis": analysis,
                "arrays": {
                    "phase_pairs": {
                        "sha256": sha256_array(phase_pairs),
                        "shape": list(phase_pairs.shape),
                    },
                    "case_output": {
                        "sha256": sha256_array(case_output),
                        "shape": list(case_output.shape),
                    },
                },
                "primary_output": {
                    "path": str(args.output_npz),
                    "exists": True,
                    "sha256": sha256_file(args.output_npz),
                    "size_bytes": args.output_npz.stat().st_size,
                },
                "elapsed_seconds": time.time() - started,
            }
        )
        _write_report(args.output_json, report)
        return 0
    except Exception as exc:
        if args is not None:
            args.output_npz.unlink(missing_ok=True)
        report.update(
            {
                "status": "failed",
                "failure_phase": phase,
                "last_trustworthy_phase": last_trustworthy,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "primary_output": {
                    "path": (
                        str(args.output_npz) if args is not None else None
                    ),
                    "exists": False,
                },
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
