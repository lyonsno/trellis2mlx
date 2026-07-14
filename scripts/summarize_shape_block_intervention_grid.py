"""Admit and summarize a block29 two-site intervention grid without projection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PLAN_SCHEMA = "trellis2mlx.shape_block_intervention_grid_plan.v1"
SUMMARY_SCHEMA = "trellis2mlx.shape_block_intervention_grid_summary.v1"
COMPARISON_CLASS = "block29_after_self_cross_attention_raw_delta_grid"
COMPARED_ARRAYS = tuple(
    f"{branch}_{stage}"
    for branch in ("pos", "neg")
    for stage in (
        "block29_after_self",
        "block29_cross_attention_raw",
        "block29_cross_attn",
        "block29_after_cross",
        "block29_after_mlp",
        "final_output",
    )
)
ROUTE_FIELDS = (
    "family",
    "backend",
    "attention_backend",
    "repo_root",
    "conditioning_sample_sha256",
    "shape_flow_noise_sample_sha256",
    "shape_slat_support_sample_sha256",
    "shared_noise_sha256",
    "shape_flow_trace_block_index",
    "shape_flow_trace_step_index",
    "steps",
)
SHA_FIELDS = {
    "conditioning_sample_sha256",
    "shape_flow_noise_sample_sha256",
    "shape_slat_support_sample_sha256",
    "shared_noise_sha256",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-index", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def summarize_grid(index_path: Path) -> dict:
    index_path = Path(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    _validate_plan(index)
    source_trace_path = Path(index["source_traces"]["block29"])
    if not source_trace_path.is_file() or source_trace_path.stat().st_size == 0:
        raise ValueError(f"source block29 trace is missing or blank: {source_trace_path}")

    admitted_points = []
    common_route = None
    for point in index["points"]:
        admitted = _admit_point(point, source_trace_path=source_trace_path)
        if common_route is None:
            common_route = admitted["route"]
        elif admitted["route"] != common_route:
            raise ValueError(
                f"grid point {point['name']} route vector differs from prior admitted points"
            )
        admitted_points.append(admitted)

    return {
        "schema": SUMMARY_SCHEMA,
        "status": "done",
        "comparison_class": COMPARISON_CLASS,
        "grid_index": str(index_path),
        "grid_index_sha256": _sha256(index_path),
        "axes": index["axes"],
        "point_count": len(admitted_points),
        "route_vector": common_route,
        "source_trace": str(source_trace_path),
        "source_trace_sha256": _sha256(source_trace_path),
        "compared_arrays": list(COMPARED_ARRAYS),
        "points": admitted_points,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        summary = summarize_grid(args.grid_index)
    except Exception as exc:
        failure = {
            "schema": SUMMARY_SCHEMA,
            "status": "failed",
            "failure_phase": "admit_grid_runs",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "last_trustworthy_evidence": {
                "grid_index": str(args.grid_index),
                "grid_index_sha256": (
                    _sha256(args.grid_index) if args.grid_index.is_file() else None
                ),
            },
        }
        _write_json(args.output_json, failure)
        return 1
    _write_json(args.output_json, summary)
    return 0


def _validate_plan(index: Any) -> None:
    if not isinstance(index, dict) or index.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"grid index must use schema {PLAN_SCHEMA}")
    axes = index.get("axes")
    if not isinstance(axes, dict):
        raise ValueError("grid index has no axes")
    alphas = _axis_values(axes.get("alpha"), "alpha")
    betas = _axis_values(axes.get("beta"), "beta")
    points = index.get("points")
    if not isinstance(points, list):
        raise ValueError("grid index points must be a list")
    expected = {(alpha, beta) for alpha in alphas for beta in betas}
    observed = []
    for point in points:
        if not isinstance(point, dict) or not isinstance(point.get("coordinate"), dict):
            raise ValueError("grid point must carry a coordinate object")
        observed.append(
            (float(point["coordinate"]["alpha"]), float(point["coordinate"]["beta"]))
        )
    if len(observed) != len(set(observed)):
        raise ValueError("grid index contains duplicate coordinates")
    missing = sorted(expected - set(observed))
    extra = sorted(set(observed) - expected)
    if missing or extra:
        raise ValueError(f"grid index is not the full Cartesian product: missing={missing}, extra={extra}")


def _axis_values(raw: Any, name: str) -> tuple[float, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"grid {name} axis must be a non-empty list")
    values = tuple(float(value) for value in raw)
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"grid {name} axis contains a non-finite value")
    if len(values) != len(set(values)):
        raise ValueError(f"grid {name} axis contains duplicate values")
    return values


def _admit_point(point: dict, *, source_trace_path: Path) -> dict:
    name = str(point.get("name", "unnamed"))
    manifest_path = Path(point["manifest_path"])
    expected_manifest_sha = point.get("manifest_sha256")
    _require_file(manifest_path, f"{name} manifest")
    observed_manifest_sha = _sha256(manifest_path)
    if observed_manifest_sha != expected_manifest_sha:
        raise ValueError(f"{name} manifest SHA does not match grid index")

    output_dir = Path(point["output_dir"])
    trace_path = Path(point["expected_trace_path"])
    run_report_path = output_dir / "run_report.json"
    route_identity_path = output_dir / "route_identity.json"
    _require_file(run_report_path, f"{name} run report")
    _require_file(route_identity_path, f"{name} route identity")
    _require_file(trace_path, f"{name} primary trace")
    run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
    route_identity = json.loads(route_identity_path.read_text(encoding="utf-8"))
    if (
        run_report.get("status") != "done"
        or run_report.get("exit_code") != 0
        or run_report.get("failure_phase") is not None
        or run_report.get("primary_output_status") != "written"
    ):
        raise ValueError(f"{name} run report does not prove a complete successful primary trace")
    artifact = run_report.get("artifacts", {}).get("shape_flow_block_trace.npz")
    if not isinstance(artifact, dict):
        raise ValueError(f"{name} run report omits the primary trace artifact")
    if Path(artifact.get("path", "")) != trace_path:
        raise ValueError(f"{name} run report primary trace path differs from the grid index")
    trace_sha = _sha256(trace_path)
    if artifact.get("sha256") != trace_sha or artifact.get("size_bytes") != trace_path.stat().st_size:
        raise ValueError(f"{name} run report primary trace digest or size is stale")

    report_route = run_report.get("route_identity", {}).get("route")
    external_route = route_identity.get("route")
    if report_route != external_route:
        raise ValueError(f"{name} run report and route identity disagree")
    route = _route_vector(report_route, name=name)
    if report_route.get("shape_flow_block_injection_manifest_sha256") != expected_manifest_sha:
        raise ValueError(f"{name} effective route used a different injection manifest")

    coordinate = {
        "alpha": float(point["coordinate"]["alpha"]),
        "beta": float(point["coordinate"]["beta"]),
    }
    with np.load(trace_path, allow_pickle=False) as trace:
        _validate_injection_evidence(
            trace,
            name=name,
            coordinate=coordinate,
            manifest_sha=expected_manifest_sha,
        )
        source_metrics = _source_metrics(trace, source_trace_path, name=name)
        control_exact = None
        if point.get("control_role") is not None:
            control_path_raw = point.get("control_reference")
            if not control_path_raw:
                raise ValueError(f"{name} semantic control has no accepted control reference")
            control_path = Path(control_path_raw)
            _require_file(control_path, f"{name} control reference")
            _require_exact_control(trace, control_path, name=name)
            control_exact = True

    return {
        "name": name,
        "coordinate": coordinate,
        "artifact": str(trace_path),
        "artifact_sha256": trace_sha,
        "manifest": str(manifest_path),
        "manifest_sha256": expected_manifest_sha,
        "control_role": point.get("control_role"),
        "control_reference": point.get("control_reference"),
        "control_exact": control_exact,
        "route": route,
        "source_metrics": source_metrics,
    }


def _route_vector(route: Any, *, name: str) -> dict:
    if not isinstance(route, dict):
        raise ValueError(f"{name} route identity has no route object")
    vector = {field: route.get(field) for field in ROUTE_FIELDS}
    missing = [field for field, value in vector.items() if value in (None, "")]
    if missing:
        raise ValueError(f"{name} route vector is missing {missing}")
    if vector["family"] != "trellis2mlx/mlx" or vector["backend"] != "mlx-metal":
        raise ValueError(f"{name} route is not the MLX Metal path")
    if vector["attention_backend"] != "fast":
        raise ValueError(f"{name} route did not use the requested fast attention backend")
    if int(vector["shape_flow_trace_block_index"]) != 29 or int(
        vector["shape_flow_trace_step_index"]
    ) != 0:
        raise ValueError(f"{name} route did not capture block29 at step0")
    for field in SHA_FIELDS:
        _require_sha256(vector[field], f"{name} {field}")
    vector["shape_flow_trace_block_index"] = int(vector["shape_flow_trace_block_index"])
    vector["shape_flow_trace_step_index"] = int(vector["shape_flow_trace_step_index"])
    vector["steps"] = int(vector["steps"])
    return vector


def _validate_injection_evidence(
    trace: Any, *, name: str, coordinate: dict[str, float], manifest_sha: str
) -> None:
    if "shape_flow_block_injection_json" not in trace:
        raise ValueError(f"{name} trace omits injection evidence")
    evidence = json.loads(str(np.asarray(trace["shape_flow_block_injection_json"]).item()))
    identity = evidence.get("manifest_identity")
    if not evidence.get("route_identity_evidence") or not isinstance(identity, dict):
        raise ValueError(f"{name} injection evidence is incomplete")
    if identity.get("comparison_class") != COMPARISON_CLASS:
        raise ValueError(f"{name} injection comparison class is wrong")
    observed_coordinate = identity.get("grid_coordinate")
    if observed_coordinate != coordinate:
        raise ValueError(f"{name} injection coordinate differs from the grid index")
    if evidence.get("manifest_sha256") != manifest_sha:
        raise ValueError(f"{name} trace injection manifest digest differs from the grid index")
    sites = evidence.get("sites")
    if not isinstance(sites, list) or len(sites) != 3:
        raise ValueError(f"{name} injection evidence must contain exactly three sites")
    expected = (
        (28, "after_mlp", 1.0),
        (29, "after_self", coordinate["alpha"]),
        (29, "cross_attention_raw", coordinate["beta"]),
    )
    for site, (block, stage, scale) in zip(sites, expected):
        if (
            site.get("block_index") != block
            or site.get("step_index") != 0
            or site.get("stage") != stage
            or site.get("branch") != "both"
            or float(site.get("source_delta_scale")) != scale
        ):
            raise ValueError(f"{name} injection site contradicts coordinate {coordinate}")


def _source_metrics(trace: Any, source_trace_path: Path, *, name: str) -> dict:
    metrics = {}
    with np.load(source_trace_path, allow_pickle=False) as source:
        for array_name in COMPARED_ARRAYS:
            candidate = _required_array(trace, array_name, name)
            source_array = _required_array(source, array_name, "source")
            if candidate.shape != source_array.shape:
                raise ValueError(f"{name} {array_name} shape differs from source")
            delta = candidate.astype(np.float32) - source_array.astype(np.float32)
            source_norm = float(np.linalg.norm(source_array.astype(np.float32).ravel()))
            delta_norm = float(np.linalg.norm(delta.ravel()))
            metrics[array_name] = {
                "mean_abs": float(np.mean(np.abs(delta), dtype=np.float64)),
                "max_abs": float(np.max(np.abs(delta))),
                "nonzero": int(np.count_nonzero(delta)),
                "relative_norm": delta_norm / source_norm if source_norm else None,
            }
    return metrics


def _require_exact_control(trace: Any, control_path: Path, *, name: str) -> None:
    with np.load(control_path, allow_pickle=False) as control:
        for array_name in COMPARED_ARRAYS:
            candidate = _required_array(trace, array_name, name)
            expected = _required_array(control, array_name, f"{name} control")
            if candidate.shape != expected.shape or not np.array_equal(candidate, expected):
                raise ValueError(f"{name} control is not exact at {array_name}")


def _required_array(trace: Any, name: str, label: str) -> np.ndarray:
    if name not in trace:
        raise ValueError(f"{label} trace omits {name}")
    array = np.asarray(trace[name])
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} {name} is blank or non-finite")
    if "cross_attention_raw" in name and array.ndim == 4 and array.shape[0] == 1:
        array = array.reshape(array.shape[1], array.shape[2] * array.shape[3])
    else:
        while array.ndim > 2 and array.shape[0] == 1:
            array = array[0]
    if array.ndim != 2:
        raise ValueError(f"{label} {name} does not normalize to [tokens, channels]: {array.shape}")
    return array


def _require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"{label} is missing or blank: {path}")


def _require_sha256(value: Any, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a SHA256 digest")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
