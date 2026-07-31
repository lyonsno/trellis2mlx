"""Compare focused official-CUDA and MLX level-two subdivision traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import traceback
from typing import Any, Mapping

from scripts.decoder_level2_subdiv_trace_contract import (
    COMPARISON_SCHEMA,
    PROJECTION_DISPOSITIONS,
    TRACE_SCHEMA,
    compare_decoder_level2_subdiv_traces,
    load_decoder_level2_subdiv_trace,
)

SOURCE_REPORT_SCHEMA = "trellis2mlx.source_cuda_shape_slat_grid_decode.v1"
SOURCE_ROUTE = "official-source-cuda-shape-decoder-level2-subdiv-trace"
LOCAL_ROUTE = "mlx-shape-decoder-level2-subdiv-trace"
SHA256_LENGTH = 64


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--local", required=True, type=Path)
    parser.add_argument("--local-report", required=True, type=Path)
    parser.add_argument("--block0-comparison", required=True, type=Path)
    parser.add_argument("--expected-block0-comparison-sha256", required=True)
    parser.add_argument("--ledger-comparison", required=True, type=Path)
    parser.add_argument("--expected-ledger-comparison-sha256", required=True)
    parser.add_argument(
        "--projection-disposition",
        required=True,
        choices=PROJECTION_DISPOSITIONS,
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_canonical_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
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


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _authenticate_json(
    path: Path,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, Any], str]:
    if not _is_canonical_sha256(expected_sha256):
        raise ValueError(f"{label} expected SHA256 is not canonical")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} SHA256 mismatch: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    return _load_json_object(path, label), actual_sha256


def _resolve_reported_path(value: object, report_path: Path) -> Path:
    path = Path(str(value or ""))
    if not path.is_absolute():
        path = report_path.resolve().parent / path
    return path.resolve()


def _require_artifact_identity(
    *,
    label: str,
    artifact: Mapping[str, Any],
    artifact_path: Path,
    report_path: Path,
) -> str:
    if artifact.get("status") != "written":
        raise ValueError(f"{label} artifact status is not written")
    reported_path = _resolve_reported_path(artifact.get("path"), report_path)
    if reported_path != artifact_path.resolve():
        raise ValueError(f"{label} artifact path mismatch")
    reported_sha256 = artifact.get("sha256")
    if not _is_canonical_sha256(reported_sha256):
        raise ValueError(f"{label} artifact SHA256 is not canonical")
    actual_sha256 = _sha256_file(artifact_path)
    if actual_sha256 != reported_sha256:
        raise ValueError(f"{label} artifact SHA256 mismatch")
    return actual_sha256


def _validate_source_report(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    source_path: Path,
) -> tuple[str, str]:
    if report.get("schema") != SOURCE_REPORT_SCHEMA:
        raise ValueError("source report schema mismatch")
    if report.get("status") != "done":
        raise ValueError("source report status is not done")
    route = report.get("effective_route")
    if not isinstance(route, Mapping):
        raise ValueError("source effective route must be an object")
    expected_route = {
        "route": SOURCE_ROUTE,
        "device_type": "cuda",
        "decoder_level2_subdiv_trace": True,
        "projection_backend": "torch-F.linear",
    }
    for field, expected in expected_route.items():
        if route.get(field) != expected:
            raise ValueError(
                f"source route field {field!r} mismatch: "
                f"expected={expected!r}, actual={route.get(field)!r}"
            )
    device = route.get("cuda_device")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("source route omits CUDA device identity")
    artifacts = report.get("decoder_trace_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise ValueError(
            "source report must contain exactly one decoder trace artifact"
        )
    artifact = artifacts[0]
    if not isinstance(artifact, Mapping):
        raise ValueError("source trace artifact must be an object")
    if artifact.get("projection_backend") != "torch-F.linear":
        raise ValueError("source artifact projection backend mismatch")
    sha256 = _require_artifact_identity(
        label="source trace",
        artifact=artifact,
        artifact_path=source_path,
        report_path=report_path,
    )
    return sha256, "torch-F.linear"


def _validate_local_report(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    local_path: Path,
    projection_disposition: str,
) -> tuple[str, str]:
    if report.get("schema") != TRACE_SCHEMA:
        raise ValueError("local report schema mismatch")
    if report.get("status") != "done":
        raise ValueError("local report status is not done")
    route = report.get("effective_route")
    if not isinstance(route, Mapping):
        raise ValueError("local effective route must be an object")
    expected_route = {
        "route": LOCAL_ROUTE,
        "device_type": "metal",
        "decoder_linear_backend": "turing_fda",
        "sparse_conv_matmul_backend": "turing_fda",
    }
    for field, expected in expected_route.items():
        if route.get(field) != expected:
            raise ValueError(
                f"local route field {field!r} mismatch: "
                f"expected={expected!r}, actual={route.get(field)!r}"
            )
    projection_backend = route.get("projection_backend")
    expected_projection = (
        "turing_fda"
        if projection_disposition == "historical-turing-fda"
        else "native"
    )
    if projection_backend != expected_projection:
        raise ValueError(
            "local projection backend mismatch: "
            f"disposition={projection_disposition!r}, "
            f"expected={expected_projection!r}, "
            f"actual={projection_backend!r}"
        )
    primary = report.get("primary")
    if not isinstance(primary, Mapping):
        raise ValueError("local report omits primary identity")
    validation = primary.get("validation")
    if (
        not isinstance(validation, Mapping)
        or validation.get("reopened_exact") is not True
    ):
        raise ValueError("local primary lacks exact reopen validation")
    sha256 = _require_artifact_identity(
        label="local trace",
        artifact=primary,
        artifact_path=local_path,
        report_path=report_path,
    )
    return sha256, str(projection_backend)


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
            "block0_comparison": str(args.block0_comparison),
            "expected_block0_comparison_sha256": (
                args.expected_block0_comparison_sha256
            ),
            "ledger_comparison": str(args.ledger_comparison),
            "expected_ledger_comparison_sha256": (
                args.expected_ledger_comparison_sha256
            ),
            "projection_disposition": args.projection_disposition,
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
            args.block0_comparison.resolve(),
            args.ledger_comparison.resolve(),
        }
        if args.output.resolve() in protected:
            effective_output = _failure_sibling(args.output, protected)
            report["effective_output"] = str(effective_output)
            raise ValueError("comparison output collides with an input path")
        if args.output.exists():
            args.output.unlink()
            report["stale_output_invalidated"] = True
        else:
            report["stale_output_invalidated"] = False
        report["last_trustworthy_phase"] = phase

        phase = "evidence_authentication"
        block0_comparison, block0_sha256 = _authenticate_json(
            args.block0_comparison,
            args.expected_block0_comparison_sha256,
            "block0 comparison",
        )
        ledger_comparison, ledger_sha256 = _authenticate_json(
            args.ledger_comparison,
            args.expected_ledger_comparison_sha256,
            "ledger comparison",
        )
        report["last_trustworthy_phase"] = phase

        phase = "report_validation"
        source_report = _load_json_object(
            args.source_report,
            "source report",
        )
        local_report = _load_json_object(args.local_report, "local report")
        source_sha256, source_projection = _validate_source_report(
            source_report,
            report_path=args.source_report,
            source_path=args.source,
        )
        local_sha256, local_projection = _validate_local_report(
            local_report,
            report_path=args.local_report,
            local_path=args.local,
            projection_disposition=args.projection_disposition,
        )
        report["last_trustworthy_phase"] = phase

        phase = "trace_validation"
        source_arrays = load_decoder_level2_subdiv_trace(args.source)
        local_arrays = load_decoder_level2_subdiv_trace(args.local)
        report["last_trustworthy_phase"] = phase

        phase = "comparison"
        comparison = compare_decoder_level2_subdiv_traces(
            source_arrays,
            local_arrays,
            block0_comparison=block0_comparison,
            ledger_comparison=ledger_comparison,
            projection_disposition=args.projection_disposition,
        )
        comparison.update(
            {
                "failure_phase": None,
                "last_trustworthy_phase": "subdivision_comparison_complete",
                "requested": report["requested"],
                "effective_output": str(effective_output),
                "stale_output_invalidated": report[
                    "stale_output_invalidated"
                ],
                "artifacts": {
                    "source": {
                        "path": str(args.source.resolve()),
                        "sha256": source_sha256,
                        "report_path": str(args.source_report.resolve()),
                        "report_sha256": _sha256_file(args.source_report),
                        "projection_backend": source_projection,
                    },
                    "local": {
                        "path": str(args.local.resolve()),
                        "sha256": local_sha256,
                        "report_path": str(args.local_report.resolve()),
                        "report_sha256": _sha256_file(args.local_report),
                        "projection_backend": local_projection,
                    },
                    "block0_comparison": {
                        "path": str(args.block0_comparison.resolve()),
                        "sha256": block0_sha256,
                    },
                    "ledger_comparison": {
                        "path": str(args.ledger_comparison.resolve()),
                        "sha256": ledger_sha256,
                    },
                },
            }
        )
        _write_json_atomic(effective_output, comparison)
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
