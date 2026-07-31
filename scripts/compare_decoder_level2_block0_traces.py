"""Compare focused CUDA/Metal block0 traces against immutable v2 receipts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.decoder_level2_block0_trace_contract import (
    COMPARISON_SCHEMA,
    authenticate_parent_receipt_file,
    compare_decoder_level2_block0_traces,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--local", required=True, type=Path)
    parser.add_argument("--local-report", required=True, type=Path)
    parser.add_argument("--parent-receipt", required=True, type=Path)
    parser.add_argument(
        "--expected-parent-receipt-sha256",
        required=True,
        help="Caller-bound SHA256 of the exact parent receipt file",
    )
    parser.add_argument("--turing-rsqrt-lut", required=True, type=Path)
    parser.add_argument(
        "--expected-turing-rsqrt-lut-sha256",
        required=True,
        help="Caller-bound SHA256 of the exact Turing rsqrt LUT artifact",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _failure_sibling(requested: Path, protected: set[Path]) -> Path:
    for index in range(len(protected) + 1):
        suffix = ".failure.json" if index == 0 else f".failure.{index}.json"
        candidate = requested.with_name(requested.name + suffix)
        if candidate.resolve() not in protected:
            return candidate
    raise RuntimeError("could not derive a non-colliding failure report path")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    effective_output = args.output
    report: dict[str, Any] = {
        "schema": COMPARISON_SCHEMA,
        "status": "failed",
        "failure_phase": None,
        "last_trustworthy_phase": "request_received",
        "first_nonexact_boundary": None,
        "requested": {
            "source": str(args.source),
            "source_report": str(args.source_report),
            "local": str(args.local),
            "local_report": str(args.local_report),
            "parent_receipt": str(args.parent_receipt),
            "expected_parent_receipt_sha256": (
                args.expected_parent_receipt_sha256
            ),
            "turing_rsqrt_lut": str(args.turing_rsqrt_lut),
            "expected_turing_rsqrt_lut_sha256": (
                args.expected_turing_rsqrt_lut_sha256
            ),
            "output": str(args.output),
        },
        "effective_output": str(effective_output),
    }
    phase = "request_validation"
    try:
        protected = {
            args.source.resolve(),
            args.source_report.resolve(),
            args.local.resolve(),
            args.local_report.resolve(),
            args.parent_receipt.resolve(),
            args.turing_rsqrt_lut.resolve(),
        }
        if args.output.resolve() in protected:
            effective_output = _failure_sibling(args.output, protected)
            report["effective_output"] = str(effective_output)
            raise ValueError("comparison output collides with an input path")
        if args.output.exists():
            args.output.unlink()
        report["last_trustworthy_phase"] = phase

        phase = "parent_receipt_authentication"
        authenticate_parent_receipt_file(
            args.parent_receipt,
            args.expected_parent_receipt_sha256,
        )
        report["last_trustworthy_phase"] = phase

        phase = "comparison"
        report = compare_decoder_level2_block0_traces(
            source_path=args.source,
            source_report_path=args.source_report,
            local_path=args.local,
            local_report_path=args.local_report,
            parent_receipt_path=args.parent_receipt,
            expected_parent_receipt_sha256=(
                args.expected_parent_receipt_sha256
            ),
            turing_rsqrt_lut_path=args.turing_rsqrt_lut,
            expected_turing_rsqrt_lut_sha256=(
                args.expected_turing_rsqrt_lut_sha256
            ),
        )
        report.update(
            {
                "failure_phase": None,
                "last_trustworthy_phase": "block0_deltas_complete",
            }
        )
        _write_json_atomic(effective_output, report)
        return 0
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "failure_phase": phase,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        _write_json_atomic(effective_output, report)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
