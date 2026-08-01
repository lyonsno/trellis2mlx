"""Compare authenticated source-CUDA and MLX full-decoder hash ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_decoder_level1_traces import compare_level1_traces
from scripts.decoder_full_hash_ledger_contract import (
    compare_decoder_full_hash_ledgers,
    validate_decoder_full_hash_ledger,
)
from scripts.decoder_level1_trace_contract import (
    validate_decoder_level1_hash_ledger,
)


SOURCE_ROUTE = "official-source-cuda-shape-decoder-full-hash-ledger"
LOCAL_ROUTE = "mlx-shape-decoder-level1-trace"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_reported_path(value: object, report_path: Path) -> Path:
    path = Path(str(value or ""))
    if not path.is_absolute():
        path = Path(report_path).resolve().parent / path
    return path.resolve()


def _require_full_route(
    label: str,
    report: Mapping[str, Any],
) -> None:
    requested = report.get("requested_route")
    effective = report.get("effective_route")
    if not isinstance(requested, Mapping):
        raise ValueError(f"{label} full-decoder requested route is missing")
    if not isinstance(effective, Mapping):
        raise ValueError(f"{label} full-decoder effective route is missing")
    expected_route = SOURCE_ROUTE if label == "source" else LOCAL_ROUTE
    expected_output_head = (
        "torch-sparse-linear-fp32"
        if label == "source"
        else "mlx-native-fp32"
    )
    for route_label, route in (
        ("requested", requested),
        ("effective", effective),
    ):
        if route.get("route") != expected_route:
            raise ValueError(
                f"{label} full-decoder {route_label} route mismatch"
            )
        if route.get("full_decoder_hash_ledger") is not True:
            raise ValueError(
                f"{label} {route_label} route did not enable the "
                "full-decoder hash ledger"
            )
        if route.get("decoder_output_head_backend") != expected_output_head:
            raise ValueError(
                f"{label} {route_label} terminal output-head route mismatch"
            )
        if label == "local":
            if route_label == "requested":
                layernorm_backend = route.get(
                    "decoder_layernorm_backend"
                )
            else:
                layernorm = route.get("decoder_layernorm")
                layernorm_backend = (
                    layernorm.get("backend")
                    if isinstance(layernorm, Mapping)
                    else None
                )
            if layernorm_backend != "cuda-welford-turing-t4":
                raise ValueError(
                    "local full-decoder LayerNorm route must be "
                    "cuda-welford-turing-t4"
                )


def _load_full_primary(
    label: str,
    *,
    report_path: Path,
    primary_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    report = json.loads(Path(report_path).read_text())
    if not isinstance(report, dict):
        raise ValueError(f"{label} report must be an object")
    _require_full_route(label, report)
    if label == "source":
        matching = [
            artifact
            for artifact in report.get("decoder_trace_artifacts", [])
            if isinstance(artifact, Mapping)
            and _resolve_reported_path(artifact.get("path"), report_path)
            == Path(primary_path).resolve()
        ]
        if len(matching) != 1:
            raise ValueError(
                "source report does not identify exactly one full-decoder "
                "trace primary"
            )
        primary = dict(matching[0])
    else:
        candidate = report.get("primary")
        if not isinstance(candidate, Mapping):
            raise ValueError("local report omits full-decoder trace primary")
        primary = dict(candidate)
        if (
            _resolve_reported_path(primary.get("path"), report_path)
            != Path(primary_path).resolve()
        ):
            raise ValueError("local full-decoder primary path mismatch")
    full_ledger = validate_decoder_full_hash_ledger(
        primary.get("full_decoder_hash_ledger")
    )
    baseline_ledger = validate_decoder_level1_hash_ledger(
        primary.get("hash_ledger")
    )
    return report, full_ledger, baseline_ledger


def _parent_entry(ledger: Mapping[str, Any]) -> dict[str, Any]:
    matching = [
        entry
        for entry in ledger["entries"]
        if entry["name"] == "level2_upsample_output"
    ]
    if len(matching) != 1:
        raise ValueError(
            "baseline ledger does not identify exactly one authenticated "
            "level-two output"
        )
    return dict(matching[0])


def compare_decoder_full_hash_reports(
    *,
    source_path: Path,
    source_report_path: Path,
    local_path: Path,
    local_report_path: Path,
) -> dict[str, Any]:
    baseline = compare_level1_traces(
        source_path=source_path,
        source_report_path=source_report_path,
        local_path=local_path,
        local_report_path=local_report_path,
    )
    (
        source_report,
        source_full,
        source_baseline,
    ) = _load_full_primary(
        "source",
        report_path=source_report_path,
        primary_path=source_path,
    )
    (
        local_report,
        local_full,
        local_baseline,
    ) = _load_full_primary(
        "local",
        report_path=local_report_path,
        primary_path=local_path,
    )
    comparison = compare_decoder_full_hash_ledgers(
        source_full,
        local_full,
        source_parent_entry=_parent_entry(source_baseline),
        local_parent_entry=_parent_entry(local_baseline),
    )
    comparison.update(
        {
            "baseline": {
                "schema": baseline["schema"],
                "input_tensor_sha256": baseline["input_tensor_sha256"],
                "first_nonexact_boundary": baseline[
                    "first_nonexact_boundary"
                ],
                "first_nonexact_hash_boundary": baseline[
                    "first_nonexact_hash_boundary"
                ],
            },
            "artifacts": baseline["artifacts"],
            "reports": {
                "source": {
                    "path": str(Path(source_report_path)),
                    "sha256": _sha256_file(source_report_path),
                    "effective_route": source_report["effective_route"],
                },
                "local": {
                    "path": str(Path(local_report_path)),
                    "sha256": _sha256_file(local_report_path),
                    "effective_route": local_report["effective_route"],
                },
            },
        }
    )
    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--local", required=True, type=Path)
    parser.add_argument("--local-report", required=True, type=Path)
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
        "schema": "trellis2mlx.decoder_full_hash_comparison.v1",
        "status": "failed",
        "failure_phase": None,
        "last_trustworthy_phase": "request_received",
        "requested": {
            "source": str(args.source),
            "source_report": str(args.source_report),
            "local": str(args.local),
            "local_report": str(args.local_report),
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
        }
        if args.output.resolve() in protected:
            effective_output = _failure_sibling(args.output, protected)
            report["effective_output"] = str(effective_output)
            raise ValueError("comparison output collides with an input path")
        if args.output.exists():
            args.output.unlink()
        report["last_trustworthy_phase"] = phase

        phase = "comparison"
        report = compare_decoder_full_hash_reports(
            source_path=args.source,
            source_report_path=args.source_report,
            local_path=args.local,
            local_report_path=args.local_report,
        )
        report.update(
            {
                "failure_phase": None,
                "last_trustworthy_phase": "full_hash_comparison_complete",
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
