"""Compare official source-CUDA and MLX first-upsample decoder traces."""

from __future__ import annotations

import argparse
import hashlib
import json
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

from scripts.decoder_level1_trace_contract import (
    TRACE_NAMES,
    decoder_level1_trace_input_sha256,
    load_decoder_level1_trace,
)


SOURCE_ROUTE = "official-source-cuda-shape-decoder-level1-trace"
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


def _require_primary_validation(label: str, primary: object) -> dict[str, Any]:
    if not isinstance(primary, dict):
        raise ValueError(f"{label} trace report omits primary identity")
    if primary.get("status") != "written":
        raise ValueError(f"{label} trace primary status is not written")
    validation = primary.get("validation")
    if (
        not isinstance(validation, dict)
        or validation.get("reopened_exact") is not True
        or validation.get("child_expansion_exact") is not True
    ):
        raise ValueError(
            f"{label} trace primary lacks exact reopen and child expansion validation"
        )
    return primary


def _require_local_route(
    route: object,
    report: dict[str, Any],
    report_path: Path,
    input_identity: object,
) -> dict[str, Any]:
    if not isinstance(route, dict):
        raise ValueError("local trace effective route must be an object")
    expected = {
        "route": LOCAL_ROUTE,
        "device_type": "metal",
        "decoder_linear_backend": "turing_fda",
        "sparse_conv_matmul_backend": "turing_fda",
    }
    for field, value in expected.items():
        if route.get(field) != value:
            raise ValueError(
                f"local trace route field {field!r} mismatch: "
                f"expected={value!r}, actual={route.get(field)!r}"
            )
    device = route.get("device")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("local trace route omits Metal device identity")

    layernorm = route.get("decoder_layernorm")
    expected_layernorm = {
        "backend": "mlx-fast-layer-norm",
        "algorithm": "mlx-fast-layer-norm",
        "experimental": False,
    }
    if not isinstance(layernorm, dict):
        raise ValueError("local trace route omits decoder LayerNorm identity")
    for field, value in expected_layernorm.items():
        if layernorm.get(field) != value:
            raise ValueError(
                f"local decoder LayerNorm field {field!r} mismatch: "
                f"expected={value!r}, actual={layernorm.get(field)!r}"
            )

    silu = route.get("decoder_silu")
    if not isinstance(silu, dict):
        raise ValueError("local trace route omits decoder SiLU identity")
    expected_silu = {
        "backend": "cuda-turing-t4-fp16-lut",
        "algorithm": "exhaustive-fp16-bit-pattern-output-lookup",
        "experimental": True,
        "cuda_architecture": "sm_75",
        "cuda_device_anchor": "Tesla T4",
        "cuda_source_operation": "torch.nn.functional.silu",
        "cuda_source_version": "torch-2.10.0+cu128",
    }
    for field, value in expected_silu.items():
        if silu.get(field) != value:
            raise ValueError(
                f"local decoder SiLU field {field!r} mismatch: "
                f"expected={value!r}, actual={silu.get(field)!r}"
            )
    contract = silu.get("authenticated_contract")
    if contract != {
        "input_dtype": "float16",
        "output_dtype": "float16",
        "domain": "all-65536-bit-patterns",
    }:
        raise ValueError("local decoder SiLU authenticated contract mismatch")
    attested = silu.get("output_lut_artifact_sha256_attested")
    effective = silu.get("output_lut_artifact_sha256_effective")
    if (
        not isinstance(attested, str)
        or len(attested) != 64
        or attested != effective
    ):
        raise ValueError("local decoder SiLU artifact identity mismatch")
    silu_path = _resolve_reported_path(
        silu.get("output_lut_artifact_path"),
        report_path,
    )
    if not silu_path.is_file() or _sha256_file(silu_path) != effective:
        raise ValueError("local decoder SiLU artifact bytes do not match identity")

    parent_state = route.get("parent_state")
    parent_trace = report.get("parent_trace")
    if not isinstance(parent_state, dict) or not isinstance(parent_trace, dict):
        raise ValueError("local trace route omits parent-state custody")
    for field in ("sha256", "input_tensor_sha256"):
        if parent_state.get(field) != parent_trace.get(field):
            raise ValueError(f"local parent-state field {field!r} mismatch")
    if parent_state.get("input_tensor_sha256") != input_identity:
        raise ValueError("local parent-state tensor identity mismatch")
    route_parent_path = _resolve_reported_path(
        parent_state.get("path"),
        report_path,
    )
    report_parent_path = _resolve_reported_path(
        parent_trace.get("path"),
        report_path,
    )
    if route_parent_path != report_parent_path:
        raise ValueError("local parent-state path mismatch")
    if (
        not route_parent_path.is_file()
        or _sha256_file(route_parent_path) != parent_state.get("sha256")
    ):
        raise ValueError("local parent-state artifact bytes do not match identity")
    return route


def _load_report(
    label: str,
    report_path: Path,
    primary_path: Path,
) -> dict[str, Any]:
    report = json.loads(Path(report_path).read_text())
    if label == "source":
        if (
            report.get("schema")
            != "trellis2mlx.source_cuda_shape_slat_grid_decode.v1"
            or report.get("status") != "done"
        ):
            raise ValueError("source level-one trace report is not done")
        route = report.get("effective_route")
        expected = {
            "route": SOURCE_ROUTE,
            "device_type": "cuda",
            "sparse_conv_backend": "none",
            "decoder_state_only": False,
            "decoder_level0_trace": False,
            "decoder_level1_trace": True,
            "raw_meshes": False,
            "post_fill_holes_snapshots": False,
            "mesh_conversion": False,
            "one_model_load": True,
        }
        if not isinstance(route, dict):
            raise ValueError("source trace effective route must be an object")
        for field, value in expected.items():
            if route.get(field) != value:
                raise ValueError(
                    f"source trace route field {field!r} mismatch: "
                    f"expected={value!r}, actual={route.get(field)!r}"
                )
        cuda_device = route.get("cuda_device")
        if not isinstance(cuda_device, str) or not cuda_device.strip():
            raise ValueError("source trace route omits CUDA device identity")
        matching = [
            artifact
            for artifact in report.get("decoder_trace_artifacts", [])
            if _resolve_reported_path(artifact.get("path"), report_path)
            == Path(primary_path).resolve()
        ]
        if len(matching) != 1:
            raise ValueError(
                "source trace report does not identify exactly one primary"
            )
        primary = _require_primary_validation("source", matching[0])
        input_identity = primary.get("input_tensor_sha256")
        effective_route = route
    else:
        if (
            report.get("schema")
            != "trellis2mlx.decoder_level1_trace_run.v1"
            or report.get("status") != "done"
        ):
            raise ValueError("local level-one trace report is not done")
        primary = _require_primary_validation("local", report.get("primary"))
        input_identity = report.get("input_tensor_sha256")
        effective_route = _require_local_route(
            report.get("effective_route"),
            report,
            Path(report_path),
            input_identity,
        )
    actual_digest = _sha256_file(primary_path)
    if primary.get("sha256") != actual_digest:
        raise ValueError(f"{label} trace primary digest mismatch")
    if (
        _resolve_reported_path(primary.get("path"), report_path)
        != Path(primary_path).resolve()
    ):
        raise ValueError(f"{label} trace primary path mismatch")
    if (
        not isinstance(input_identity, str)
        or len(input_identity) != 64
    ):
        raise ValueError(f"{label} trace input tensor identity is invalid")
    return {
        "report": report,
        "effective_route": effective_route,
        "input_tensor_sha256": input_identity,
    }


def _numeric_delta(source: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    source64 = source.astype(np.float64)
    candidate64 = candidate.astype(np.float64)
    delta = candidate64 - source64
    absolute = np.abs(delta)
    return {
        "source_dtype": str(source.dtype),
        "candidate_dtype": str(candidate.dtype),
        "shape": [int(value) for value in source.shape],
        "mean_abs": float(absolute.mean()),
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(absolute.max()),
        "nonzero_count": int(np.count_nonzero(delta)),
    }


def compare_level1_traces(
    *,
    source_path: Path,
    source_report_path: Path,
    local_path: Path,
    local_report_path: Path,
) -> dict[str, Any]:
    paths = {
        "source": Path(source_path),
        "local": Path(local_path),
    }
    reports = {
        "source": _load_report(
            "source",
            Path(source_report_path),
            paths["source"],
        ),
        "local": _load_report(
            "local",
            Path(local_report_path),
            paths["local"],
        ),
    }
    traces = {
        label: load_decoder_level1_trace(path)
        for label, path in paths.items()
    }
    recomputed = {
        label: decoder_level1_trace_input_sha256(
            trace["level0_output"],
            trace["parent_coords"],
        )
        for label, trace in traces.items()
    }
    for label in ("source", "local"):
        if recomputed[label] != reports[label]["input_tensor_sha256"]:
            raise ValueError(
                f"{label} trace input tensor identity mismatch"
            )
    if recomputed["source"] != recomputed["local"]:
        raise ValueError("source and local parent-state identities differ")
    for name in ("parent_coords", "child_coords", "level0_output"):
        if not np.array_equal(traces["source"][name], traces["local"][name]):
            raise ValueError(f"local trace {name} does not exactly match source")

    first_nonexact_boundary = None
    stages: dict[str, Any] = {}
    for name in TRACE_NAMES:
        delta = _numeric_delta(traces["source"][name], traces["local"][name])
        if first_nonexact_boundary is None and delta["nonzero_count"]:
            first_nonexact_boundary = name
        stages[name] = delta

    return {
        "schema": "trellis2mlx.decoder_level1_trace_comparison.v1",
        "status": "done",
        "input_tensor_sha256": recomputed["source"],
        "first_nonexact_boundary": first_nonexact_boundary,
        "artifacts": {
            label: {
                "path": str(paths[label]),
                "sha256": _sha256_file(paths[label]),
                "effective_route": reports[label]["effective_route"],
            }
            for label in ("source", "local")
        },
        "stages": stages,
    }


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
        "schema": "trellis2mlx.decoder_level1_trace_comparison.v1",
        "status": "failed",
        "failure_phase": None,
        "last_trustworthy_phase": "request_received",
        "first_nonexact_boundary": None,
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
        report = compare_level1_traces(
            source_path=args.source,
            source_report_path=args.source_report,
            local_path=args.local,
            local_report_path=args.local_report,
        )
        report.update(
            {
                "failure_phase": None,
                "last_trustworthy_phase": "stage_deltas_complete",
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
