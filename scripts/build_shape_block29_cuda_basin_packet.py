#!/usr/bin/env python3
"""Build a compact, route-bound block29 MLX/source-CUDA endpoint packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "trellis2mlx.shape_block29_cuda_basin_endpoints.v1"
COMPARISON_CLASS = "fixed_block29_endpoint_affine_plane"
ENDPOINT_SEMANTICS = "current + scale * (source - current)"
BRANCH_STAGE_KEYS = (
    "pos_block29_after_self",
    "pos_block29_cross_attention_raw",
    "neg_block29_after_self",
    "neg_block29_cross_attention_raw",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def encode_bf16_words(array: np.ndarray, *, name: str) -> np.ndarray:
    values = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    words32 = values.view(np.uint32)
    nonzero_low = int(np.count_nonzero(words32 & np.uint32(0xFFFF)))
    if nonzero_low:
        raise ValueError(
            f"{name} is not exactly representable as BF16: {nonzero_low} values have low bits"
        )
    return np.ascontiguousarray((words32 >> np.uint32(16)).astype(np.uint16))


def decode_bf16_words(words: np.ndarray) -> np.ndarray:
    packed = np.ascontiguousarray(np.asarray(words, dtype=np.uint16))
    return np.ascontiguousarray((packed.astype(np.uint32) << np.uint32(16)).view(np.float32))


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"{label} is blank: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _resolve_recorded_path(recorded: str, *, owner: Path) -> Path:
    path = Path(recorded)
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve()


def _current_point(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("status") != "done":
        raise ValueError("grid summary must have status='done'")
    if summary.get("comparison_class") != "block29_after_self_cross_attention_raw_delta_grid":
        raise ValueError("grid summary comparison class is not the admitted block29 alpha-beta grid")
    matches = [
        point
        for point in summary.get("points", [])
        if point.get("coordinate") == {"alpha": 0.0, "beta": 0.0}
    ]
    if len(matches) != 1:
        raise ValueError("grid summary must contain exactly one current coordinate (0,0)")
    point = matches[0]
    route = point.get("route", {})
    required = {
        "backend": "mlx-metal",
        "family": "trellis2mlx/mlx",
        "attention_backend": "fast",
        "shape_flow_trace_block_index": 29,
        "shape_flow_trace_step_index": 0,
        "shape_flow_trace_key_selection": "explicit",
        "steps": 8,
    }
    for key, expected in required.items():
        if route.get(key) != expected:
            raise ValueError(f"current route {key} must be {expected!r}, got {route.get(key)!r}")
    selected = route.get("shape_flow_trace_keys")
    if not isinstance(selected, list) or not set(BRANCH_STAGE_KEYS).issubset(selected):
        raise ValueError("current route does not explicitly select all four endpoint arrays")
    return point


def _validate_source_route(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("status") != "done" or report.get("primary_output_status") != "written":
        raise ValueError("source report must be done with a written primary output")
    route = report.get("route_identity", {})
    required = {
        "backend": "source-trellis",
        "device": "cuda",
        "effective_device_type": "cuda",
        "effective_route": "official-trellis2-source-cuda-shape-flow-block-trace",
        "branch": "both",
        "steps": 8,
    }
    for key, expected in required.items():
        if route.get(key) != expected:
            if key == "effective_device_type":
                raise ValueError(
                    "source route requires effective_device_type='cuda', got "
                    f"{route.get(key)!r}"
                )
            raise ValueError(f"source route {key} must be {expected!r}, got {route.get(key)!r}")
    blocks = route.get("shape_flow_trace_block_indices", route.get("block_indices"))
    if blocks != [29]:
        raise ValueError(f"source route must trace only block29, got {blocks!r}")
    return route


def _required_digest(route: dict[str, Any], key: str, *, label: str) -> str:
    value = route.get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} has no valid {key}")
    return value


def _scalar_int(archive: np.lib.npyio.NpzFile, key: str) -> int:
    if key not in archive.files:
        raise ValueError(f"source trace has no {key}")
    value = np.asarray(archive[key])
    if value.size != 1:
        raise ValueError(f"source trace {key} must contain one value")
    return int(value.reshape(-1)[0])


def build_packet(
    *,
    grid_summary_path: Path,
    source_report_path: Path,
    output_npz: Path,
    output_json: Path,
) -> dict[str, Any]:
    grid_summary_path = Path(grid_summary_path).resolve()
    source_report_path = Path(source_report_path).resolve()
    output_npz = Path(output_npz).resolve()
    output_json = Path(output_json).resolve()
    summary = _load_json(grid_summary_path, label="grid summary")
    source_report = _load_json(source_report_path, label="source report")
    current_point = _current_point(summary)
    current_route = current_point["route"]
    source_route = _validate_source_route(source_report)

    current_trace = _resolve_recorded_path(current_point["artifact"], owner=grid_summary_path)
    source_primary = source_report.get("primary_output", {})
    source_trace = _resolve_recorded_path(source_primary.get("path", ""), owner=source_report_path)
    for path, label in ((current_trace, "current trace"), (source_trace, "source trace")):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"{label} is missing or blank: {path}")
    if _sha256(current_trace) != current_point.get("artifact_sha256"):
        raise ValueError("current trace digest does not match admitted grid summary")
    if _sha256(source_trace) != source_primary.get("sha256"):
        raise ValueError("source trace digest does not match source report")

    shared_digest_keys = (
        ("conditioning_sample_sha256", "conditioning_sha256"),
        ("shape_slat_support_sample_sha256", "shape_slat_support_sample_sha256"),
        ("shape_flow_noise_sample_sha256", "shape_flow_noise_sample_sha256"),
    )
    for current_key, source_key in shared_digest_keys:
        current_digest = _required_digest(current_route, current_key, label="current route")
        source_digest = _required_digest(source_route, source_key, label="source route")
        if current_digest != source_digest:
            raise ValueError(
                f"current/source route mismatch for {current_key}: "
                f"{current_digest} != {source_digest}"
            )

    packed: dict[str, np.ndarray] = {}
    endpoint_shapes: dict[str, list[int]] = {}
    endpoint_digests: dict[str, dict[str, str]] = {}
    with np.load(current_trace, allow_pickle=False) as current, np.load(
        source_trace, allow_pickle=False
    ) as source:
        for archive, label in ((current, "current"), (source, "source")):
            missing = sorted(set(("coords", *BRANCH_STAGE_KEYS)) - set(archive.files))
            if missing:
                raise ValueError(f"{label} trace is missing arrays: {missing}")
        if _scalar_int(source, "trace_block_index") != 29:
            raise ValueError("source trace block index is not 29")
        if _scalar_int(source, "shape_flow_trace_step_index") != 0:
            raise ValueError("source trace step index is not 0")
        if _scalar_int(source, "steps") != 8:
            raise ValueError("source trace steps is not 8")
        current_coords = np.asarray(current["coords"], dtype=np.int32)
        source_coords = np.asarray(source["coords"], dtype=np.int32)
        if current_coords.shape != source_coords.shape or not np.array_equal(
            current_coords, source_coords
        ):
            raise ValueError("current/source trace coordinates differ")
        packed["coords"] = np.ascontiguousarray(current_coords)
        for key in BRANCH_STAGE_KEYS:
            current_values = np.asarray(current[key], dtype=np.float32)
            source_values = np.asarray(source[key], dtype=np.float32)
            if current_values.ndim == 4 and current_values.shape[0] == 1:
                current_values = current_values.reshape(1, current_values.shape[1], -1)
            if source_values.ndim == 4 and source_values.shape[0] == 1:
                source_values = source_values.reshape(1, source_values.shape[1], -1)
            if current_values.shape != source_values.shape:
                raise ValueError(
                    f"endpoint shape mismatch for {key}: "
                    f"{current_values.shape} != {source_values.shape}"
                )
            current_words = encode_bf16_words(current_values, name=f"current {key}")
            source_words = encode_bf16_words(source_values, name=f"source {key}")
            packed[f"{key}_current_bf16_words"] = current_words
            packed[f"{key}_source_bf16_words"] = source_words
            endpoint_shapes[key] = [int(v) for v in current_values.shape]
            endpoint_digests[key] = {
                "current_float32_sha256": hashlib.sha256(current_values.tobytes()).hexdigest(),
                "source_float32_sha256": hashlib.sha256(source_values.tobytes()).hexdigest(),
            }

    metadata = {
        "schema": SCHEMA,
        "status": "done",
        "comparison_class": COMPARISON_CLASS,
        "endpoint_semantics": ENDPOINT_SEMANTICS,
        "block_index": 29,
        "step_index": 0,
        "steps": 8,
        "branches": ["pos", "neg"],
        "stages": ["after_self", "cross_attention_raw"],
        "coords_shape": [int(v) for v in packed["coords"].shape],
        "endpoint_shapes": endpoint_shapes,
        "endpoint_digests": endpoint_digests,
        "current_trace": str(current_trace),
        "current_trace_sha256": _sha256(current_trace),
        "source_trace": str(source_trace),
        "source_trace_sha256": _sha256(source_trace),
        "grid_summary": str(grid_summary_path),
        "grid_summary_sha256": _sha256(grid_summary_path),
        "source_report": str(source_report_path),
        "source_report_sha256": _sha256(source_report_path),
        "current_route": current_route,
        "source_route": source_route,
        "forbidden_inferences": [
            "not a final mesh, texture, winding, or GLB result",
            "not a claim that local MLX continuation equals source-CUDA continuation",
            "not an implementation patch",
        ],
    }
    packed["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False))
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_npz, **packed)
    result = {
        **metadata,
        "primary_output_status": "written",
        "primary_output": {
            "path": str(output_npz),
            "sha256": _sha256(output_npz),
            "size_bytes": output_npz.stat().st_size,
            "keys": sorted(packed),
        },
    }
    _write_json(output_json, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-summary", required=True, type=Path)
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    try:
        build_packet(
            grid_summary_path=args.grid_summary,
            source_report_path=args.source_report,
            output_npz=args.output_npz,
            output_json=args.output_json,
        )
        return 0
    except Exception as exc:
        payload = {
            "schema": f"{SCHEMA}.failure",
            "status": "failed",
            "failure_phase": "input_validation",
            "last_trustworthy_phase": "arguments_parsed",
            "primary_output_status": "written" if args.output_npz.exists() else "missing",
            "elapsed_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
            "inputs": {
                "grid_summary": str(args.grid_summary),
                "source_report": str(args.source_report),
            },
        }
        _write_json(args.output_json, payload)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
