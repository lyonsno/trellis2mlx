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
SUMMARY_SCHEMA = "trellis2mlx.shape_block_intervention_grid_summary.v2"
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
    "shape_flow_trace_key_selection",
    "shape_flow_trace_keys",
    "steps",
)
SHA_FIELDS = {
    "conditioning_sample_sha256",
    "shape_flow_noise_sample_sha256",
    "shape_slat_support_sample_sha256",
    "shared_noise_sha256",
}


class CoordinateGeometryError(ValueError):
    pass


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

    try:
        coordinate_geometry = _build_coordinate_geometry(index["axes"], admitted_points)
    except CoordinateGeometryError:
        raise
    except Exception as exc:
        raise CoordinateGeometryError(str(exc)) from exc

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
        "coordinate_geometry": coordinate_geometry,
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
            "failure_phase": (
                "coordinate_geometry"
                if isinstance(exc, CoordinateGeometryError)
                else "admit_grid_runs"
            ),
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
    point_names = []
    for point in points:
        if not isinstance(point, dict) or not isinstance(point.get("coordinate"), dict):
            raise ValueError("grid point must carry a coordinate object")
        point_name = point.get("name")
        if not isinstance(point_name, str) or not point_name:
            raise ValueError("grid point must carry a non-empty name")
        point_names.append(point_name)
        observed.append(
            (float(point["coordinate"]["alpha"]), float(point["coordinate"]["beta"]))
        )
    if len(point_names) != len(set(point_names)):
        raise ValueError("grid index contains duplicate point names")
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
    if vector["shape_flow_trace_key_selection"] != "explicit":
        raise ValueError(f"{name} trace key selection was not explicit")
    if vector["shape_flow_trace_keys"] != list(COMPARED_ARRAYS):
        raise ValueError(
            f"{name} effective trace key selection does not match the evidence contract"
        )
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


def _build_coordinate_geometry(axes: dict[str, list[float]], points: list[dict]) -> dict:
    alpha_values = tuple(sorted(_axis_values(axes.get("alpha"), "alpha")))
    beta_values = tuple(sorted(_axis_values(axes.get("beta"), "beta")))
    for point in points:
        point["state_digests"] = {}

    quotient_classes = {}
    combined_cells = None
    for array_name in COMPARED_ARRAYS:
        states = {}
        for point in points:
            coordinate = point["coordinate"]
            key = (float(coordinate["alpha"]), float(coordinate["beta"]))
            with np.load(point["artifact"], allow_pickle=False) as trace:
                array = np.array(
                    _required_array(trace, array_name, point["name"]),
                    copy=True,
                )
            states[key] = (point["name"], array)

        array_geometry = _summarize_array_geometry(
            array_name,
            alpha_values=alpha_values,
            beta_values=beta_values,
            states=states,
        )
        quotient_classes[array_name] = array_geometry["quotient_classes"]
        digest_by_coordinate = {
            (state["coordinate"]["alpha"], state["coordinate"]["beta"]): state[
                "state_digest"
            ]
            for state in array_geometry["states"]
        }
        for point in points:
            coordinate = point["coordinate"]
            point["state_digests"][array_name] = digest_by_coordinate[
                (float(coordinate["alpha"]), float(coordinate["beta"]))
            ]

        if combined_cells is None:
            combined_cells = [
                {
                    "bounds": cell["bounds"],
                    "delta": cell["delta"],
                    "arrays": {
                        array_name: {
                            key: value
                            for key, value in cell.items()
                            if key not in {"bounds", "delta"}
                        }
                    },
                }
                for cell in array_geometry["cells"]
            ]
        else:
            if len(combined_cells) != len(array_geometry["cells"]):
                raise ValueError("coordinate geometry produced inconsistent cell counts")
            for combined, cell in zip(combined_cells, array_geometry["cells"]):
                if combined["bounds"] != cell["bounds"] or combined["delta"] != cell["delta"]:
                    raise ValueError("coordinate geometry produced inconsistent cell bounds")
                combined["arrays"][array_name] = {
                    key: value for key, value in cell.items() if key not in {"bounds", "delta"}
                }

    return {
        "coordinate_system": {
            "alpha": "source_delta_scale at block29 after_self",
            "beta": "source_delta_scale at block29 cross_attention_raw",
            "projection": "none",
        },
        "sorted_axes": {"alpha": list(alpha_values), "beta": list(beta_values)},
        "quotient_classes": quotient_classes,
        "cells": combined_cells or [],
    }


def _summarize_array_geometry(
    array_name: str,
    *,
    alpha_values: tuple[float, ...],
    beta_values: tuple[float, ...],
    states: dict[tuple[float, float], tuple[str, np.ndarray]],
) -> dict:
    alphas = tuple(sorted(float(value) for value in alpha_values))
    betas = tuple(sorted(float(value) for value in beta_values))
    if not alphas or not betas:
        raise ValueError(f"{array_name} coordinate geometry axes must be non-empty")
    if any(not math.isfinite(value) for value in (*alphas, *betas)):
        raise ValueError(f"{array_name} coordinate geometry axes contain non-finite values")
    if len(alphas) != len(set(alphas)) or len(betas) != len(set(betas)):
        raise ValueError(f"{array_name} coordinate geometry axes contain duplicates")
    expected = {(alpha, beta) for alpha in alphas for beta in betas}
    observed = {(float(alpha), float(beta)) for alpha, beta in states}
    if observed != expected:
        raise ValueError(
            f"{array_name} coordinate geometry is not the full Cartesian product: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )

    normalized = {}
    expected_shape = None
    state_rows = []
    quotient_groups: dict[str, list[dict]] = {}
    for coordinate in sorted(expected):
        point_name, raw_array = states[coordinate]
        array = np.asarray(raw_array)
        if array.ndim != 2:
            raise ValueError(
                f"{array_name} state at {coordinate} does not have normalized 2-D shape: {array.shape}"
            )
        if array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError(f"{array_name} state at {coordinate} is blank or non-finite")
        if expected_shape is None:
            expected_shape = array.shape
        elif array.shape != expected_shape:
            raise ValueError(
                f"{array_name} state shape differs across coordinates: "
                f"expected {expected_shape}, got {array.shape} at {coordinate}"
            )
        array = np.ascontiguousarray(array)
        normalized[coordinate] = array
        digest = _state_digest(array)
        row = {
            "point_name": str(point_name),
            "coordinate": {"alpha": coordinate[0], "beta": coordinate[1]},
            "state_digest": digest,
        }
        state_rows.append(row)
        quotient_groups.setdefault(digest, []).append(row)

    quotient_classes = []
    for digest, rows in sorted(
        quotient_groups.items(),
        key=lambda item: (
            item[1][0]["coordinate"]["alpha"],
            item[1][0]["coordinate"]["beta"],
        ),
    ):
        quotient_classes.append(
            {
                "state_digest": digest,
                "point_count": len(rows),
                "point_names": [row["point_name"] for row in rows],
                "coordinates": [row["coordinate"] for row in rows],
            }
        )

    cells = []
    for alpha0, alpha1 in zip(alphas, alphas[1:]):
        delta_alpha = alpha1 - alpha0
        if delta_alpha <= 0:
            raise ValueError(f"{array_name} alpha cell width must be positive")
        for beta0, beta1 in zip(betas, betas[1:]):
            delta_beta = beta1 - beta0
            if delta_beta <= 0:
                raise ValueError(f"{array_name} beta cell width must be positive")
            y00 = normalized[(alpha0, beta0)].astype(np.float64)
            y10 = normalized[(alpha1, beta0)].astype(np.float64)
            y01 = normalized[(alpha0, beta1)].astype(np.float64)
            y11 = normalized[(alpha1, beta1)].astype(np.float64)
            alpha_lower = (y10 - y00) / delta_alpha
            alpha_upper = (y11 - y01) / delta_alpha
            beta_lower = (y01 - y00) / delta_beta
            beta_upper = (y11 - y10) / delta_beta
            mixed = (y11 - y10 - y01 + y00) / (delta_alpha * delta_beta)
            cells.append(
                {
                    "bounds": {
                        "alpha": [alpha0, alpha1],
                        "beta": [beta0, beta1],
                    },
                    "delta": {"alpha": delta_alpha, "beta": delta_beta},
                    "lower_corner_tangents": {
                        "alpha": _vector_metrics(alpha_lower),
                        "beta": _vector_metrics(beta_lower),
                        "cosine": _cosine(alpha_lower, beta_lower),
                    },
                    "opposite_edge_transport": {
                        "alpha": {
                            "difference": _vector_metrics(alpha_upper - alpha_lower),
                            "cosine": _cosine(alpha_lower, alpha_upper),
                        },
                        "beta": {
                            "difference": _vector_metrics(beta_upper - beta_lower),
                            "cosine": _cosine(beta_lower, beta_upper),
                        },
                    },
                    "mixed_second_difference": _vector_metrics(mixed),
                }
            )

    return {
        "array_name": array_name,
        "states": state_rows,
        "quotient_classes": quotient_classes,
        "cells": cells,
    }


def _state_digest(array: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(array))
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("state digest input is blank or non-finite")
    digest = hashlib.sha256()
    digest.update(b"trellis2mlx.coordinate-state.v1\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(size) for size in array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _vector_metrics(array: np.ndarray) -> dict:
    array = np.asarray(array, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("coordinate geometry produced a blank or non-finite vector")
    absolute = np.abs(array)
    return {
        "mean_abs": float(np.mean(absolute, dtype=np.float64)),
        "max_abs": float(np.max(absolute)),
        "l2_norm": float(np.linalg.norm(array.ravel())),
        "nonzero": int(np.count_nonzero(array)),
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    left_flat = np.asarray(left, dtype=np.float64).ravel()
    right_flat = np.asarray(right, dtype=np.float64).ravel()
    left_norm = float(np.linalg.norm(left_flat))
    right_norm = float(np.linalg.norm(right_flat))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    value = float(np.dot(left_flat, right_flat) / (left_norm * right_norm))
    return min(1.0, max(-1.0, value))


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
