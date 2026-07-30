"""Compare source-CUDA, MLX-FP16, and MLX-FP32 decoder level-zero traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.decoder_level0_trace_contract import (
    TRACE_NAMES,
    decoder_trace_input_sha256,
    load_decoder_level0_trace,
)


EXPECTED_ROUTES = {
    "source": "official-source-cuda-shape-decoder-level0-trace",
    "local_fp16": "mlx-shape-decoder-level0-trace-fp16",
    "local_fp32": "mlx-shape-decoder-level0-trace-fp32",
}


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


def _load_report(
    label: str,
    report_path: Path,
    primary_path: Path,
) -> dict[str, Any]:
    report = json.loads(Path(report_path).read_text())
    if report.get("schema") == "trellis2mlx.decoder_level0_trace_run.v1":
        if report.get("status") != "passed":
            raise ValueError(f"{label} trace report status is not passed")
        effective_route = report.get("effective_route")
        effective_route_details = {"route": effective_route}
        primary = report.get("primary")
        input_tensor_sha256 = report.get(
            "input_tensor_sha256",
            report.get("input_slat_sha256"),
        )
    elif (
        label == "source"
        and report.get("schema")
        == "trellis2mlx.source_cuda_shape_slat_grid_decode.v1"
    ):
        if report.get("status") != "done":
            raise ValueError(f"{label} trace report status is not done")
        route = report.get("effective_route")
        effective_route = route.get("route") if isinstance(route, dict) else None
        if not isinstance(route, dict):
            raise ValueError("source trace effective route must be an object")
        expected_source_route = {
            "device_type": "cuda",
            "sparse_conv_backend": "none",
            "decoder_state_only": False,
            "decoder_level0_trace": True,
            "raw_meshes": False,
            "post_fill_holes_snapshots": False,
            "mesh_conversion": False,
            "one_model_load": True,
        }
        for field, expected in expected_source_route.items():
            if route.get(field) != expected:
                raise ValueError(
                    f"source trace effective route field {field!r} mismatch: "
                    f"expected={expected!r}, actual={route.get(field)!r}"
                )
        cuda_device = route.get("cuda_device")
        if not isinstance(cuda_device, str) or not cuda_device.strip():
            raise ValueError(
                "source trace effective route field 'cuda_device' mismatch: "
                "expected nonempty string"
            )
        effective_route_details = dict(route)
        matching_artifacts = [
            artifact
            for artifact in report.get("decoder_trace_artifacts", [])
            if _resolve_reported_path(artifact.get("path"), report_path)
            == Path(primary_path).resolve()
        ]
        if len(matching_artifacts) != 1:
            raise ValueError(
                "source trace report does not identify exactly one requested primary"
            )
        primary = matching_artifacts[0]
        input_tensor_sha256 = primary.get("input_tensor_sha256")
    else:
        raise ValueError(f"{label} trace report schema mismatch")
    if effective_route != EXPECTED_ROUTES[label]:
        raise ValueError(f"{label} trace effective route mismatch")
    if not isinstance(primary, dict):
        raise ValueError(f"{label} trace report omits primary identity")
    actual_digest = _sha256_file(primary_path)
    if primary.get("sha256") != actual_digest:
        raise ValueError(f"{label} trace primary digest mismatch")
    reported_path = _resolve_reported_path(primary.get("path"), report_path)
    if reported_path != Path(primary_path).resolve():
        raise ValueError(f"{label} trace primary path mismatch")
    input_digest = input_tensor_sha256
    if not isinstance(input_digest, str) or len(input_digest) != 64:
        raise ValueError(f"{label} trace input tensor identity is invalid")
    return {
        "effective_route": effective_route,
        "effective_route_details": effective_route_details,
        "input_tensor_sha256": input_digest,
        "primary": primary,
        "source_report": report,
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


def compare_level0_traces(
    *,
    source_path: Path,
    source_report_path: Path,
    local_fp16_path: Path,
    local_fp16_report_path: Path,
    local_fp32_path: Path,
    local_fp32_report_path: Path,
    latent_channels: int = 32,
    channels: int = 1024,
) -> dict[str, Any]:
    paths = {
        "source": Path(source_path),
        "local_fp16": Path(local_fp16_path),
        "local_fp32": Path(local_fp32_path),
    }
    reports = {
        "source": _load_report("source", source_report_path, paths["source"]),
        "local_fp16": _load_report(
            "local_fp16",
            local_fp16_report_path,
            paths["local_fp16"],
        ),
        "local_fp32": _load_report(
            "local_fp32",
            local_fp32_report_path,
            paths["local_fp32"],
        ),
    }
    input_identities = {
        report["input_tensor_sha256"] for report in reports.values()
    }
    if len(input_identities) != 1:
        raise ValueError("trace input SLat identities do not match")

    traces = {
        "source": load_decoder_level0_trace(
            paths["source"],
            latent_channels=latent_channels,
            channels=channels,
            torso_dtype=np.float16,
        ),
        "local_fp16": load_decoder_level0_trace(
            paths["local_fp16"],
            latent_channels=latent_channels,
            channels=channels,
            torso_dtype=np.float16,
        ),
        "local_fp32": load_decoder_level0_trace(
            paths["local_fp32"],
            latent_channels=latent_channels,
            channels=channels,
            torso_dtype=np.float32,
        ),
    }
    recomputed_input_identities = {
        label: decoder_trace_input_sha256(
            trace["input_feats"],
            trace["coords"],
        )
        for label, trace in traces.items()
    }
    for label, identity in recomputed_input_identities.items():
        if identity != reports[label]["input_tensor_sha256"]:
            raise ValueError(
                f"{label} trace input tensor identity mismatch: "
                f"report={reports[label]['input_tensor_sha256']}, "
                f"recomputed={identity}"
            )
    source_coords = traces["source"]["coords"]
    for label in ("local_fp16", "local_fp32"):
        if not np.array_equal(source_coords, traces[label]["coords"]):
            raise ValueError(f"{label} trace coordinates do not exactly match source")
        if not np.array_equal(
            traces["source"]["input_feats"],
            traces[label]["input_feats"],
        ):
            raise ValueError(f"{label} trace input features do not exactly match source")

    first_numeric_fork = {
        "local_fp16": None,
        "local_fp32": None,
    }
    stages: dict[str, Any] = {}
    for name in TRACE_NAMES:
        local_fp16_delta = _numeric_delta(
            traces["source"][name],
            traces["local_fp16"][name],
        )
        local_fp32_delta = _numeric_delta(
            traces["source"][name],
            traces["local_fp32"][name],
        )
        for label, delta in (
            ("local_fp16", local_fp16_delta),
            ("local_fp32", local_fp32_delta),
        ):
            if (
                first_numeric_fork[label] is None
                and delta["nonzero_count"] > 0
            ):
                first_numeric_fork[label] = name
        fp16_rms = local_fp16_delta["rms"]
        fp32_rms = local_fp32_delta["rms"]
        nearest = (
            "tie"
            if fp16_rms == fp32_rms
            else "local_fp16"
            if fp16_rms < fp32_rms
            else "local_fp32"
        )
        stages[name] = {
            "local_fp16": local_fp16_delta,
            "local_fp32": local_fp32_delta,
            "nearest_local_island": nearest,
        }

    return {
        "schema": "trellis2mlx.decoder_level0_trace_comparison.v1",
        "input_tensor_sha256": recomputed_input_identities["source"],
        "artifacts": {
            label: {
                "path": str(path),
                "sha256": _sha256_file(path),
                "effective_route": reports[label]["effective_route"],
                "effective_route_details": reports[label][
                    "effective_route_details"
                ],
            }
            for label, path in paths.items()
        },
        "first_numeric_fork": first_numeric_fork,
        "stages": stages,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--local-fp16", required=True, type=Path)
    parser.add_argument("--local-fp16-report", required=True, type=Path)
    parser.add_argument("--local-fp32", required=True, type=Path)
    parser.add_argument("--local-fp32-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = compare_level0_traces(
        source_path=args.source,
        source_report_path=args.source_report,
        local_fp16_path=args.local_fp16,
        local_fp16_report_path=args.local_fp16_report,
        local_fp32_path=args.local_fp32,
        local_fp32_report_path=args.local_fp32_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
