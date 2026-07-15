"""Render latent and decoded mesh distances on their shared alpha-beta plane."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any


LATENT_SCHEMA = "trellis2mlx.source_cuda_block29_basin_plane_summary.v1"
ATLAS_SCHEMA = "trellis2mlx.mesh_surface_support_atlas.v1"
DECODE_SCHEMA = "trellis2mlx.source_cuda_shape_slat_grid_decode.v1"
REPORT_SCHEMA = "trellis2mlx.decoded_basin_plane_render.v1"
SOURCE_ROUTE = "official-source-cuda-full-eight-step-shape-flow-with-fixed-block29-endpoints"
DECODE_ROUTE = "official-source-cuda-shape-slat-decoder"
SHAPE_MODEL = "microsoft/TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16"
DECODER_MODEL = "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16"


class BasinPlaneContractError(ValueError):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-summary", required=True, type=Path)
    parser.add_argument("--mesh-atlas", required=True, type=Path)
    parser.add_argument("--decode-report", required=True, action="append", type=Path)
    parser.add_argument("--output-html", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_paths = [args.latent_summary, args.mesh_atlas, *args.decode_report]
    collisions = _path_collisions(
        [
            ("latent_summary", args.latent_summary),
            ("mesh_atlas", args.mesh_atlas),
            *[(f"decode_report:{index}", path) for index, path in enumerate(args.decode_report)],
            ("output_html", args.output_html),
            ("output_report", args.output_report),
        ]
    )
    effective_report = args.output_report
    if any("output_report" in collision for collision in collisions):
        effective_report = _failure_report_path(args.output_report, [*input_paths, args.output_html])
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "failed",
        "failure_phase": None,
        "latent_summary": str(args.latent_summary),
        "mesh_atlas": str(args.mesh_atlas),
        "decode_reports": [str(path) for path in args.decode_report],
        "output_html": str(args.output_html),
        "requested_output_report": str(args.output_report),
        "effective_output_report": str(effective_report),
        "path_collisions": collisions,
        "last_trustworthy_evidence": {},
    }
    phase = "validate_paths"
    try:
        if not _aliases_any(args.output_html, input_paths):
            args.output_html.unlink(missing_ok=True)
        if collisions:
            raise BasinPlaneContractError(f"requested paths must be distinct: {collisions}")

        phase = "read_inputs"
        all_inputs = [args.latent_summary, args.mesh_atlas, *args.decode_report]
        for path in all_inputs:
            if not path.is_file() or path.stat().st_size == 0:
                raise BasinPlaneContractError(f"input is missing or blank: {path}")
        hashes = {str(path): _sha256(path) for path in all_inputs}
        report["last_trustworthy_evidence"] = {"input_sha256": hashes}
        latent = _read_json(args.latent_summary)
        atlas = _read_json(args.mesh_atlas)
        decode_reports = [_read_json(path) for path in args.decode_report]

        phase = "validate_inputs"
        payload = build_payload(
            latent=latent,
            atlas=atlas,
            decode_reports=decode_reports,
            input_sha256={
                "latent": hashes[str(args.latent_summary)],
                "atlas": hashes[str(args.mesh_atlas)],
            },
            decode_report_sha256=[hashes[str(path)] for path in args.decode_report],
        )

        phase = "render_html"
        html = render_html(payload)
        args.output_html.parent.mkdir(parents=True, exist_ok=True)
        args.output_html.write_text(html, encoding="utf-8")
        if args.output_html.stat().st_size == 0:
            raise BasinPlaneContractError("rendered HTML is blank")
        report.update(
            {
                "status": "done",
                "failure_phase": None,
                "source_shape_route": payload["source_shape_route"],
                "geometry_decode_route": payload["geometry_decode_route"],
                "axes": payload["axes"],
                "scales": payload["scales"],
                "comparison": payload["comparison"],
                "output_html_sha256": _sha256(args.output_html),
                "output_html_size_bytes": args.output_html.stat().st_size,
            }
        )
    except Exception as exc:
        if args.output_html.exists() and not _aliases_any(args.output_html, input_paths):
            args.output_html.unlink()
        report.update(
            {
                "status": "failed",
                "failure_phase": phase,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        _write_json(effective_report, report)
        return 1
    _write_json(effective_report, report)
    return 0


def build_payload(
    *,
    latent: Any,
    atlas: Any,
    decode_reports: list[Any],
    input_sha256: dict[str, str],
    decode_report_sha256: list[str],
) -> dict[str, Any]:
    _validate_hashes(input_sha256, decode_report_sha256)
    axes, points, latent_pairs, source_route = _validate_latent(latent)
    names = [_coordinate_name(point["alpha"], point["beta"]) for point in points]
    atlas_scales, source_hashes = _validate_atlas(atlas, names)
    decode_route = _validate_decode_reports(
        decode_reports,
        names=names,
        source_hashes=source_hashes,
        latent=latent,
    )

    index = {name: position for position, name in enumerate(atlas["names"])}
    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(points):
        left_name = _coordinate_name(left["alpha"], left["beta"])
        for right in points[left_index + 1 :]:
            right_name = _coordinate_name(right["alpha"], right["beta"])
            pair_key = tuple(sorted((left_name, right_name)))
            row = {
                "a": left,
                "b": right,
                "a_name": left_name,
                "b_name": right_name,
                "lattice_distance": math.hypot(
                    left["alpha"] - right["alpha"], left["beta"] - right["beta"]
                ),
                "latent_mean_abs": latent_pairs[pair_key],
                "support_jaccard_distance": {},
            }
            for scale in atlas_scales:
                matrix = atlas["scales"][str(scale)]["pairwise_jaccard_distance"]
                row["support_jaccard_distance"][str(scale)] = matrix[
                    index[left_name]
                ][index[right_name]]
            pairs.append(row)

    comparison: dict[str, Any] = {}
    latent_values = [pair["latent_mean_abs"] for pair in pairs]
    lattice_values = [pair["lattice_distance"] for pair in pairs]
    for scale in atlas_scales:
        support = [pair["support_jaccard_distance"][str(scale)] for pair in pairs]
        adjacent = [
            value
            for value, lattice in zip(support, lattice_values)
            if _is_adjacent(lattice, axes)
        ]
        nonadjacent = [
            value
            for value, lattice in zip(support, lattice_values)
            if not _is_adjacent(lattice, axes)
        ]
        comparison[str(scale)] = {
            "latent_vs_support": {
                "pearson_r": _pearson(latent_values, support),
                "spearman_rho": _pearson(_ranks(latent_values), _ranks(support)),
            },
            "lattice_vs_support": {
                "pearson_r": _pearson(lattice_values, support),
                "spearman_rho": _pearson(_ranks(lattice_values), _ranks(support)),
            },
            "support_distance": _summary(support),
            "adjacent_support_distance": _summary(adjacent),
            "nonadjacent_support_distance": _summary(nonadjacent),
        }

    occupied = {
        str(scale): atlas["scales"][str(scale)]["occupied_cells"] for scale in atlas_scales
    }
    nearest_chords = {
        "latent": _nearest_chords(
            points,
            pairs,
            lambda pair: pair["latent_mean_abs"],
        ),
        "support": {
            str(scale): _nearest_chords(
                points,
                pairs,
                lambda pair, scale=scale: pair["support_jaccard_distance"][str(scale)],
            )
            for scale in atlas_scales
        },
    }
    return {
        "schema": REPORT_SCHEMA,
        "input_sha256": input_sha256,
        "decode_report_sha256": decode_report_sha256,
        "axes": axes,
        "points": [
            {
                "name": _coordinate_name(point["alpha"], point["beta"]),
                "coordinate": point,
                "raw_mesh_sha256": source_hashes[
                    _coordinate_name(point["alpha"], point["beta"])
                ],
            }
            for point in points
        ],
        "pairs": pairs,
        "nearest_chords": nearest_chords,
        "scales": atlas_scales,
        "occupied_cells": occupied,
        "comparison": comparison,
        "source_shape_route": source_route,
        "geometry_decode_route": decode_route,
        "forbidden_inferences": list(atlas["forbidden_inferences"])
        + [
            "decoded support similarity is not final textured-GLB correctness",
            "this chart does not establish winding, watertightness, or topology",
        ],
    }


def _validate_latent(
    latent: Any,
) -> tuple[dict[str, list[float]], list[dict[str, float]], dict[tuple[str, str], float], dict[str, Any]]:
    if not isinstance(latent, dict) or latent.get("schema") != LATENT_SCHEMA:
        raise BasinPlaneContractError(f"latent summary must use schema {LATENT_SCHEMA}")
    if latent.get("status") != "done":
        raise BasinPlaneContractError("latent summary status is not done")
    for field in ("primary_sha256", "report_sha256", "receipt_sha256"):
        _require_sha(latent.get(field), f"latent {field}")
    source_route = _validate_source_route(latent.get("effective_route"))
    axes = latent.get("axes")
    if not isinstance(axes, dict):
        raise BasinPlaneContractError("latent axes are missing")
    normalized_axes = {
        "alpha": _axis(axes.get("alpha"), "alpha"),
        "beta": _axis(axes.get("beta"), "beta"),
    }
    expected = {
        (alpha, beta)
        for alpha in normalized_axes["alpha"]
        for beta in normalized_axes["beta"]
    }
    raw_points = latent.get("points")
    if not isinstance(raw_points, list):
        raise BasinPlaneContractError("latent points are missing")
    observed: set[tuple[float, float]] = set()
    for row in raw_points:
        coordinate = row.get("coordinate") if isinstance(row, dict) else None
        if not isinstance(coordinate, dict):
            raise BasinPlaneContractError("latent point coordinate is malformed")
        point = (
            _finite(coordinate.get("alpha"), "point alpha"),
            _finite(coordinate.get("beta"), "point beta"),
        )
        if point in observed:
            raise BasinPlaneContractError("latent point coordinates are duplicated")
        observed.add(point)
    if observed != expected:
        raise BasinPlaneContractError("latent points are not the full Cartesian plane")
    points = [{"alpha": alpha, "beta": beta} for alpha, beta in sorted(observed)]

    expected_pairs = {
        tuple(sorted((_coordinate_name(*left), _coordinate_name(*right))))
        for index, left in enumerate(sorted(observed))
        for right in sorted(observed)[index + 1 :]
    }
    pairs: dict[tuple[str, str], float] = {}
    for row in latent.get("pairwise", []):
        if not isinstance(row, dict):
            raise BasinPlaneContractError("latent pairwise row is malformed")
        left = _coordinate(row.get("a"), "latent pair a")
        right = _coordinate(row.get("b"), "latent pair b")
        key = tuple(sorted((_coordinate_name(*left), _coordinate_name(*right))))
        if left == right or key in pairs:
            raise BasinPlaneContractError("latent pairwise rows duplicate a pair")
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            raise BasinPlaneContractError("latent pair metrics are missing")
        pairs[key] = _finite(metrics.get("mean_abs"), "latent pair mean_abs")
    if set(pairs) != expected_pairs:
        raise BasinPlaneContractError("latent pairwise matrix is incomplete")
    return normalized_axes, points, pairs, source_route


def _validate_atlas(
    atlas: Any,
    expected_names: list[str],
) -> tuple[list[int], dict[str, str]]:
    if not isinstance(atlas, dict) or atlas.get("schema") != ATLAS_SCHEMA:
        raise BasinPlaneContractError(f"mesh atlas must use schema {ATLAS_SCHEMA}")
    if atlas.get("status") != "done" or atlas.get("route") != "shared_grid_vertex_surface_support":
        raise BasinPlaneContractError("mesh atlas route is not admitted")
    if atlas.get("embedding_authority") != "none":
        raise BasinPlaneContractError("mesh atlas claims an embedding authority")
    names = atlas.get("names")
    if not isinstance(names, list) or set(names) != set(expected_names) or len(names) != len(set(names)):
        raise BasinPlaneContractError("mesh atlas names do not match the latent plane")
    sources = atlas.get("sources")
    if not isinstance(sources, list) or len(sources) != len(names):
        raise BasinPlaneContractError("mesh atlas sources are incomplete")
    source_hashes: dict[str, str] = {}
    for source in sources:
        if not isinstance(source, dict) or source.get("name") not in names:
            raise BasinPlaneContractError("mesh atlas source is malformed")
        name = source["name"]
        if name in source_hashes:
            raise BasinPlaneContractError("mesh atlas sources duplicate a name")
        _require_sha(source.get("sha256"), f"mesh atlas source {name}")
        source_hashes[name] = source["sha256"]
    scales = atlas.get("grid_sizes")
    if not isinstance(scales, list) or not scales:
        raise BasinPlaneContractError("mesh atlas grid sizes are missing")
    normalized_scales = sorted({_positive_int(value, "grid size") for value in scales})
    scale_rows = atlas.get("scales")
    if not isinstance(scale_rows, dict) or set(scale_rows) != {str(value) for value in normalized_scales}:
        raise BasinPlaneContractError("mesh atlas scales are stale or incomplete")
    for scale in normalized_scales:
        row = scale_rows[str(scale)]
        occupied = row.get("occupied_cells") if isinstance(row, dict) else None
        matrix = row.get("pairwise_jaccard_distance") if isinstance(row, dict) else None
        if not isinstance(occupied, dict) or set(occupied) != set(names):
            raise BasinPlaneContractError(f"grid {scale} occupied-cell map is incomplete")
        for name, value in occupied.items():
            _positive_int(value, f"grid {scale} occupied cells for {name}")
        _validate_distance_matrix(matrix, names, scale)
    forbidden = atlas.get("forbidden_inferences")
    if not isinstance(forbidden, list) or len(forbidden) < 3:
        raise BasinPlaneContractError("mesh atlas forbidden inferences are missing")
    return normalized_scales, source_hashes


def _validate_decode_reports(
    reports: list[Any],
    *,
    names: list[str],
    source_hashes: dict[str, str],
    latent: dict[str, Any],
) -> dict[str, Any]:
    if not reports:
        raise BasinPlaneContractError("at least one decode report is required")
    raw_hashes: dict[str, str] = {}
    effective_route: dict[str, Any] | None = None
    fill_holes_effective_change_count = 0
    identity_sources: set[str] = set()
    for report in reports:
        if not isinstance(report, dict) or report.get("schema") != DECODE_SCHEMA:
            raise BasinPlaneContractError(f"decode report must use schema {DECODE_SCHEMA}")
        if report.get("status") != "done" or report.get("failure_phase") is not None:
            raise BasinPlaneContractError("decode report is not a complete success")
        route = _validate_decode_route(report.get("effective_route"))
        model = report.get("model_load")
        if (
            not isinstance(model, dict)
            or model.get("model_ref") != DECODER_MODEL
            or model.get("training_before_eval") is not True
            or model.get("training") is not False
        ):
            raise BasinPlaneContractError("decode model mode identity is not admitted")
        route_change_count = route.pop("fill_holes_effective_change_count")
        route_identity_source = route.pop("identity_source")
        identity_sources.add(route_identity_source)
        if effective_route is None:
            effective_route = route
        elif route != effective_route:
            raise BasinPlaneContractError("decode reports disagree on effective route")
        primary = report.get("source_basin_primary")
        source_report = report.get("source_basin_report")
        if not isinstance(primary, dict) or primary.get("sha256") != latent["primary_sha256"]:
            raise BasinPlaneContractError("decode report uses a different latent primary")
        if not isinstance(source_report, dict) or source_report.get("sha256") != latent["report_sha256"]:
            raise BasinPlaneContractError("decode report uses a different latent report")
        source_route = report.get("source_basin_route")
        _validate_source_route(source_route, allow_missing_model=True)
        selected = report.get("selected_point_names")
        artifacts = report.get("mesh_artifacts")
        if not isinstance(selected, list) or not isinstance(artifacts, list):
            raise BasinPlaneContractError("decode report artifacts are missing")
        if (
            len(selected) != len(set(selected))
            or any(name not in names for name in selected)
            or report.get("expected_artifact_count") != 2 * len(selected)
            or report.get("written_artifact_count") != 2 * len(selected)
            or len(artifacts) != 2 * len(selected)
        ):
            raise BasinPlaneContractError("decode report completion accounting is inconsistent")
        report_raw: dict[str, str] = {}
        report_filled: dict[str, tuple[str, bool | None]] = {}
        for artifact in artifacts:
            if not isinstance(artifact, dict) or artifact.get("status") != "written":
                raise BasinPlaneContractError("decode report contains an unwritten artifact")
            name = artifact.get("coordinate_key")
            variant = artifact.get("variant")
            if name not in selected or variant not in {"raw", "filled"}:
                raise BasinPlaneContractError("decode artifacts contradict selected points")
            _require_sha(artifact.get("sha256"), f"decode {variant} mesh {name}")
            if variant == "raw":
                if name in report_raw:
                    raise BasinPlaneContractError("decode report duplicates a selected raw mesh")
                report_raw[name] = artifact["sha256"]
            else:
                changed = artifact.get("fill_holes_effective_change")
                if name in report_filled or (
                    changed is not None and not isinstance(changed, bool)
                ):
                    raise BasinPlaneContractError("decode filled artifacts are incomplete")
                report_filled[name] = (artifact["sha256"], changed)
        if set(report_raw) != set(selected) or set(report_filled) != set(selected):
            raise BasinPlaneContractError("decode report is missing a selected raw/filled mesh pair")
        point_results = report.get("point_results")
        if not isinstance(point_results, list) or len(point_results) != len(selected):
            raise BasinPlaneContractError("decode point result accounting is incomplete")
        result_changes: dict[str, bool | None] = {}
        for result in point_results:
            if not isinstance(result, dict):
                raise BasinPlaneContractError("decode point result is malformed")
            name = result.get("coordinate_key")
            changed = result.get("fill_holes_effective_change")
            if (
                name not in selected
                or name in result_changes
                or (changed is not None and not isinstance(changed, bool))
            ):
                raise BasinPlaneContractError("decode point results contradict selected points")
            result_changes[name] = changed
        artifact_changes = {name: changed for name, (_, changed) in report_filled.items()}
        hash_changes = {
            name: report_filled[name][0] != report_raw[name]
            for name in selected
        }
        explicit_changes = [*result_changes.values(), *artifact_changes.values()]
        if any(changed is not None for changed in explicit_changes):
            if (
                any(changed is None for changed in explicit_changes)
                or result_changes != artifact_changes
                or result_changes != hash_changes
            ):
                raise BasinPlaneContractError("decode fill-hole evidence is self-contradictory")
        elif route_identity_source != "legacy-derived-from-written-pairs":
            raise BasinPlaneContractError("decode fill-hole evidence is missing")
        actual_change_count = sum(hash_changes.values())
        if route_change_count is not None and route_change_count != actual_change_count:
            raise BasinPlaneContractError("decode route fill-hole count contradicts point results")
        fill_holes_effective_change_count += actual_change_count
        for name, digest in report_raw.items():
            if name in raw_hashes:
                raise BasinPlaneContractError("decode reports duplicate a coordinate")
            raw_hashes[name] = digest
    if set(raw_hashes) != set(names):
        raise BasinPlaneContractError("decode reports do not cover the full atlas")
    if raw_hashes != source_hashes:
        raise BasinPlaneContractError("atlas source hashes do not match decoded raw meshes")
    assert effective_route is not None
    effective_route["fill_holes_effective_change_count"] = fill_holes_effective_change_count
    effective_route["identity_sources"] = sorted(identity_sources)
    return effective_route


def _validate_source_route(route: Any, *, allow_missing_model: bool = False) -> dict[str, Any]:
    expected = {
        "route": SOURCE_ROUTE,
        "device_type": "cuda",
        "cuda_device": "Tesla T4",
        "attention_backend": "sdpa",
        "conv_backend": "none",
        "block_index": 29,
        "step_index": 0,
        "steps": 8,
        "endpoint_semantics": "current + scale * (source - current)",
        "one_model_load": True,
    }
    if not allow_missing_model:
        expected["model_ref"] = SHAPE_MODEL
    if not isinstance(route, dict) or any(route.get(key) != value for key, value in expected.items()):
        raise BasinPlaneContractError("source shape route identity is wrong or incomplete")
    return {key: route[key] for key in expected}


def _validate_decode_route(route: Any) -> dict[str, Any]:
    expected = {
        "route": DECODE_ROUTE,
        "device_type": "cuda",
        "cuda_device": "Tesla T4",
        "model_ref": DECODER_MODEL,
        "model_training": False,
        "one_model_load": True,
        "sparse_attention_backend": "sdpa",
        "sparse_conv_backend": "none",
        "resolution": 512,
    }
    if not isinstance(route, dict) or any(route.get(key) != value for key, value in expected.items()):
        raise BasinPlaneContractError("geometry decode route identity is wrong or incomplete")
    modern_fields = {
        "raw_meshes": True,
        "post_fill_holes_snapshots": True,
    }
    if all(route.get(key) == value for key, value in modern_fields.items()):
        change_count = route.get("fill_holes_effective_change_count")
        if isinstance(change_count, bool) or not isinstance(change_count, int) or change_count < 0:
            raise BasinPlaneContractError("geometry decode fill-hole count is invalid")
        identity_source = "current-explicit"
    elif (
        route.get("raw_and_filled_meshes") is True
        and all(key not in route for key in (*modern_fields, "fill_holes_effective_change_count"))
    ):
        change_count = None
        identity_source = "legacy-derived-from-written-pairs"
    else:
        raise BasinPlaneContractError("geometry decode raw/fill route identity is wrong or incomplete")
    return {
        **expected,
        **modern_fields,
        "fill_holes_effective_change_count": change_count,
        "identity_source": identity_source,
    }


def _validate_distance_matrix(matrix: Any, names: list[str], scale: int) -> None:
    size = len(names)
    if not isinstance(matrix, list) or len(matrix) != size:
        raise BasinPlaneContractError(f"grid {scale} distance matrix has wrong row count")
    for row in matrix:
        if not isinstance(row, list) or len(row) != size:
            raise BasinPlaneContractError(f"grid {scale} distance matrix has wrong shape")
    for left in range(size):
        for right in range(size):
            value = _finite(matrix[left][right], f"grid {scale} distance")
            if value < 0 or value > 1:
                raise BasinPlaneContractError(f"grid {scale} distance is outside [0,1]")
            if left == right and value != 0:
                raise BasinPlaneContractError(f"grid {scale} distance diagonal is nonzero")
            if not math.isclose(value, float(matrix[right][left]), rel_tol=0, abs_tol=1e-12):
                raise BasinPlaneContractError(f"grid {scale} distance matrix is asymmetric")


def _nearest_chords(
    points: list[dict[str, float]],
    pairs: list[dict[str, Any]],
    value: Any,
) -> list[dict[str, str]]:
    chords: set[tuple[str, str]] = set()
    for point in points:
        name = _coordinate_name(point["alpha"], point["beta"])
        options = [pair for pair in pairs if name in (pair["a_name"], pair["b_name"])]
        if not options:
            raise BasinPlaneContractError(f"point {name} has no pairwise distances")
        best = min(
            options,
            key=lambda pair: (value(pair), pair["a_name"], pair["b_name"]),
        )
        chords.add(tuple(sorted((best["a_name"], best["b_name"]))))
    return [
        {"a_name": left, "b_name": right}
        for left, right in sorted(chords)
    ]


def render_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    latent_sha = payload["input_sha256"]["latent"]
    default_scale = payload["scales"][-1]
    static_chart = _render_static_chart(payload, default_scale)
    static_detail = _render_static_detail(payload, default_scale)
    return f'''<!doctype html>
<html lang="en" data-latent-sha256="{latent_sha}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TRELLIS.2 Latent-to-Geometry Basin Plane</title>
<style>
:root{{--bg:#f3f1ed;--surface:#fff;--ink:#202426;--muted:#667075;--line:#c8cecf;--grid:#dfe3e2;--teal:#087f8c;--red:#c34632;--gold:#9a6b00;--violet:#6557a8}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:0}}header{{display:flex;justify-content:space-between;gap:24px;align-items:end;padding:16px 20px;border-bottom:1px solid var(--line)}}h1{{font-size:20px;margin:0 0 4px;letter-spacing:0}}p{{margin:0;color:var(--muted)}}.truth{{color:var(--gold);text-align:right}}.controls{{display:flex;gap:14px;align-items:end;flex-wrap:wrap;padding:10px 20px;border-bottom:1px solid var(--line);background:var(--surface)}}label{{display:grid;gap:4px;color:var(--muted);font-size:11px}}select{{min-height:34px;border:1px solid var(--line);border-radius:4px;background:var(--surface);color:var(--ink);padding:5px 28px 5px 8px;font:inherit}}.toggle{{display:flex;align-items:center;gap:7px;min-height:34px}}.layout{{display:grid;grid-template-columns:minmax(0,1fr) 320px}}main{{padding:18px 20px;min-width:0}}svg{{display:block;width:100%;height:auto;aspect-ratio:900/630;background:var(--surface);border:1px solid var(--line)}}aside{{border-left:1px solid var(--line);padding:18px 16px;background:var(--surface);overflow-wrap:anywhere}}aside h2{{font-size:13px;margin:0 0 8px}}.detail{{white-space:pre-wrap;color:var(--muted);font-size:11px}}.axis{{stroke:var(--muted);stroke-width:1}}.grid{{stroke:var(--grid);stroke-width:1}}.edge{{stroke:var(--teal);stroke-linecap:round;cursor:pointer}}.nearest{{stroke:var(--red);stroke-width:2;stroke-dasharray:7 5;fill:none;pointer-events:none}}.node{{fill:var(--surface);stroke:var(--violet);stroke-width:3;cursor:pointer}}.label{{fill:var(--muted);font:12px ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:0}}.edge-label{{fill:var(--ink);font:11px ui-monospace,SFMono-Regular,Menlo,monospace;paint-order:stroke;stroke:var(--surface);stroke-width:4;stroke-linejoin:round;letter-spacing:0;pointer-events:none}}.node-label{{fill:var(--ink);font:11px ui-monospace,SFMono-Regular,Menlo,monospace;text-anchor:middle;letter-spacing:0;pointer-events:none}}.legend{{display:flex;gap:18px;flex-wrap:wrap;padding:9px 20px;border-top:1px solid var(--line);background:var(--surface);color:var(--muted);font-size:11px}}.swatch{{display:inline-block;width:22px;height:3px;margin:0 6px 3px 0;background:var(--teal)}}.swatch.nearest-swatch{{background:var(--red)}}@media(max-width:820px){{header{{align-items:start;flex-direction:column}}.truth{{text-align:left}}.layout{{grid-template-columns:1fr}}aside{{border-left:0;border-top:1px solid var(--line)}}main{{padding:12px 8px}}.edge-label,.node-label,.label{{font-size:15px}}}}
</style>
</head>
<body>
<header><div><h1>TRELLIS.2 Latent-to-Geometry Basin Plane</h1><p>block29 source-endpoint perturbation coordinates</p></div><p class="truth">No PCA or learned embedding defines these coordinates.</p></header>
<div class="controls"><label>Edge measure<select id="measure"><option value="support">support Jaccard distance</option><option value="latent">latent mean absolute</option></select></label><label>Support grid<select id="scale">{''.join(f'<option value="{scale}"{(" selected" if scale == payload["scales"][-1] else "")}>{scale} cubed</option>' for scale in payload['scales'])}</select></label><label class="toggle"><input id="nearest" type="checkbox" checked> nearest-neighbor chords</label></div>
<div class="layout"><main><svg id="chart" viewBox="0 0 900 630" role="img" aria-labelledby="chart-title chart-desc" data-static-chart="support-jaccard-{default_scale}"><title id="chart-title">Direct alpha-beta latent and decoded support plane</title><desc id="chart-desc">Adjacent lattice edges encode support Jaccard distance and dashed chords show nearest neighbors.</desc>{static_chart}</svg></main><aside><h2>Selected evidence</h2><div id="detail" class="detail" data-static-detail="true">{static_detail}</div></aside></div>
<div class="legend"><span><i class="swatch"></i>adjacent selected distance</span><span><i class="swatch nearest-swatch"></i>global nearest-neighbor chord</span><span>node: decoded coordinate</span></div>
<script type="application/json" id="plane-data">{data}</script>
<script>
const D=JSON.parse(document.getElementById('plane-data').textContent),svg=document.getElementById('chart'),measure=document.getElementById('measure'),scale=document.getElementById('scale'),nearest=document.getElementById('nearest'),detail=document.getElementById('detail'),NS='http:'+'//www.w3.org/2000/svg';
const el=(n,a={{}})=>{{const x=document.createElementNS(NS,n);for(const[k,v]of Object.entries(a))x.setAttribute(k,String(v));return x}},A=D.axes.alpha,B=D.axes.beta,P={{l:105,r:65,t:55,b:95}},W=900,H=630,loA=Math.min(...A),hiA=Math.max(...A),loB=Math.min(...B),hiB=Math.max(...B),X=a=>P.l+(a-loA)*(W-P.l-P.r)/(hiA-loA||1),Y=b=>H-P.b-(b-loB)*(H-P.t-P.b)/(hiB-loB||1),pairKey=(a,b)=>[a,b].sort().join('|'),pairs=new Map(D.pairs.map(p=>[pairKey(p.a_name,p.b_name),p])),pointMap=new Map(D.points.map(p=>[p.name,p]));
const value=p=>measure.value==='latent'?p.latent_mean_abs:p.support_jaccard_distance[scale.value],show=v=>detail.textContent=JSON.stringify(v,null,2),fmt=v=>Number(v).toExponential(3);
function draw(){{svg.querySelectorAll(':scope > :not(title):not(desc)').forEach(n=>n.remove());for(const a of A){{svg.append(el('line',{{x1:X(a),y1:P.t,x2:X(a),y2:H-P.b,class:'grid'}}));const t=el('text',{{x:X(a),y:H-P.b+28,'text-anchor':'middle',class:'label'}});t.textContent=a;svg.append(t)}}for(const b of B){{svg.append(el('line',{{x1:P.l,y1:Y(b),x2:W-P.r,y2:Y(b),class:'grid'}}));const t=el('text',{{x:P.l-15,y:Y(b)+4,'text-anchor':'end',class:'label'}});t.textContent=b;svg.append(t)}}
const adjacent=D.pairs.filter(p=>{{const da=Math.abs(p.a.alpha-p.b.alpha),db=Math.abs(p.a.beta-p.b.beta);return (da===0&&db>0&&!B.some(x=>x>Math.min(p.a.beta,p.b.beta)&&x<Math.max(p.a.beta,p.b.beta)))||(db===0&&da>0&&!A.some(x=>x>Math.min(p.a.alpha,p.b.alpha)&&x<Math.max(p.a.alpha,p.b.alpha)))}}),vals=adjacent.map(value),lo=Math.min(...vals),hi=Math.max(...vals),width=v=>2+8*(hi===lo?.5:(v-lo)/(hi-lo));
for(const p of adjacent){{const v=value(p),line=el('line',{{x1:X(p.a.alpha),y1:Y(p.a.beta),x2:X(p.b.alpha),y2:Y(p.b.beta),class:'edge','stroke-width':width(v)}});line.addEventListener('click',()=>show({{measure:measure.value,scale:Number(scale.value),pair:p}}));svg.append(line);const t=el('text',{{x:(X(p.a.alpha)+X(p.b.alpha))/2,y:(Y(p.a.beta)+Y(p.b.beta))/2-8,'text-anchor':'middle',class:'edge-label'}});t.textContent=fmt(v);svg.append(t)}}
if(nearest.checked){{const chords=measure.value==='latent'?D.nearest_chords.latent:D.nearest_chords.support[scale.value];for(const chord of chords){{const p=pointMap.get(chord.a_name),q=pointMap.get(chord.b_name);svg.append(el('path',{{d:`M${{X(p.coordinate.alpha)}},${{Y(p.coordinate.beta)}} L${{X(q.coordinate.alpha)}},${{Y(q.coordinate.beta)}}`,class:'nearest'}}))}}}}
for(const point of D.points){{const c=el('circle',{{cx:X(point.coordinate.alpha),cy:Y(point.coordinate.beta),r:10,class:'node'}});c.addEventListener('click',()=>show({{point,occupied_cells:D.occupied_cells[scale.value][point.name],raw_mesh_sha256:point.raw_mesh_sha256}}));svg.append(c)}}svg.append(el('line',{{x1:P.l,y1:H-P.b,x2:W-P.r,y2:H-P.b,class:'axis'}}),el('line',{{x1:P.l,y1:P.t,x2:P.l,y2:H-P.b,class:'axis'}}));const xl=el('text',{{x:(P.l+W-P.r)/2,y:H-28,'text-anchor':'middle',class:'label'}});xl.textContent='alpha: after_self source-delta scale';svg.append(xl);const yl=el('text',{{x:25,y:(P.t+H-P.b)/2,transform:`rotate(-90 25 ${{(P.t+H-P.b)/2}})`,'text-anchor':'middle',class:'label'}});yl.textContent='beta: cross-attention-raw source-delta scale';svg.append(yl);show({{measure:measure.value,scale:Number(scale.value),comparison:D.comparison[scale.value],forbidden_inferences:D.forbidden_inferences}})}}
for(const control of[measure,scale,nearest])control.addEventListener('change',draw);draw();
</script>
</body>
</html>'''


def _render_static_chart(payload: dict[str, Any], scale: int) -> str:
    alpha = payload["axes"]["alpha"]
    beta = payload["axes"]["beta"]
    padding = {"left": 105.0, "right": 65.0, "top": 55.0, "bottom": 95.0}
    width, height = 900.0, 630.0

    def x(value: float) -> float:
        span = max(alpha) - min(alpha)
        return padding["left"] + (value - min(alpha)) * (
            width - padding["left"] - padding["right"]
        ) / (span or 1)

    def y(value: float) -> float:
        span = max(beta) - min(beta)
        return height - padding["bottom"] - (value - min(beta)) * (
            height - padding["top"] - padding["bottom"]
        ) / (span or 1)

    adjacent = [pair for pair in payload["pairs"] if _pair_is_grid_adjacent(pair, alpha, beta)]
    values = [pair["support_jaccard_distance"][str(scale)] for pair in adjacent]
    low, high = min(values), max(values)

    def stroke_width(value: float) -> float:
        position = 0.5 if high == low else (value - low) / (high - low)
        return 2 + 8 * position

    parts: list[str] = []
    for value in alpha:
        parts.append(
            f'<line x1="{x(value):.3f}" y1="{padding["top"]:.3f}" '
            f'x2="{x(value):.3f}" y2="{height - padding["bottom"]:.3f}" class="grid static"/>'
        )
        parts.append(
            f'<text x="{x(value):.3f}" y="{height - padding["bottom"] + 28:.3f}" '
            f'text-anchor="middle" class="label static">{value:g}</text>'
        )
    for value in beta:
        parts.append(
            f'<line x1="{padding["left"]:.3f}" y1="{y(value):.3f}" '
            f'x2="{width - padding["right"]:.3f}" y2="{y(value):.3f}" class="grid static"/>'
        )
        parts.append(
            f'<text x="{padding["left"] - 15:.3f}" y="{y(value) + 4:.3f}" '
            f'text-anchor="end" class="label static">{value:g}</text>'
        )
    for pair in adjacent:
        value = pair["support_jaccard_distance"][str(scale)]
        parts.append(
            f'<line x1="{x(pair["a"]["alpha"]):.3f}" y1="{y(pair["a"]["beta"]):.3f}" '
            f'x2="{x(pair["b"]["alpha"]):.3f}" y2="{y(pair["b"]["beta"]):.3f}" '
            f'class="edge static" stroke-width="{stroke_width(value):.3f}"/>'
        )
        parts.append(
            f'<text x="{(x(pair["a"]["alpha"]) + x(pair["b"]["alpha"])) / 2:.3f}" '
            f'y="{(y(pair["a"]["beta"]) + y(pair["b"]["beta"])) / 2 - 8:.3f}" '
            f'text-anchor="middle" class="edge-label static">{value:.3e}</text>'
        )
    point_map = {point["name"]: point for point in payload["points"]}
    for chord in payload["nearest_chords"]["support"][str(scale)]:
        point = point_map[chord["a_name"]]
        other = point_map[chord["b_name"]]
        parts.append(
            f'<path d="M{x(point["coordinate"]["alpha"]):.3f},{y(point["coordinate"]["beta"]):.3f} '
            f'L{x(other["coordinate"]["alpha"]):.3f},{y(other["coordinate"]["beta"]):.3f}" '
            'class="nearest static"/>'
        )
    for point in payload["points"]:
        occupied = payload["occupied_cells"][str(scale)][point["name"]]
        parts.append(
            f'<circle cx="{x(point["coordinate"]["alpha"]):.3f}" '
            f'cy="{y(point["coordinate"]["beta"]):.3f}" r="10" class="node static" '
            f'data-occupied-cells="{occupied}"/>'
        )
    parts.extend(
        [
            f'<line x1="{padding["left"]:.3f}" y1="{height - padding["bottom"]:.3f}" '
            f'x2="{width - padding["right"]:.3f}" y2="{height - padding["bottom"]:.3f}" class="axis static"/>',
            f'<line x1="{padding["left"]:.3f}" y1="{padding["top"]:.3f}" '
            f'x2="{padding["left"]:.3f}" y2="{height - padding["bottom"]:.3f}" class="axis static"/>',
            f'<text x="{(padding["left"] + width - padding["right"]) / 2:.3f}" y="{height - 28:.3f}" '
            'text-anchor="middle" class="label static">alpha: after_self source-delta scale</text>',
            f'<text x="25" y="{(padding["top"] + height - padding["bottom"]) / 2:.3f}" '
            f'transform="rotate(-90 25 {(padding["top"] + height - padding["bottom"]) / 2:.3f})" '
            'text-anchor="middle" class="label static">beta: cross-attention-raw source-delta scale</text>',
        ]
    )
    return "".join(parts)


def _render_static_detail(payload: dict[str, Any], scale: int) -> str:
    comparison = payload["comparison"][str(scale)]
    latent = comparison["latent_vs_support"]
    support = comparison["support_distance"]
    lines = [
        "measure: support Jaccard distance",
        f"support grid: {scale} cubed",
        f"latent vs support Pearson: {latent['pearson_r']:.6f}",
        f"latent vs support Spearman: {latent['spearman_rho']:.6f}",
        f"support distance range: {support['min']:.6f} to {support['max']:.6f}",
    ]
    return html.escape("\n".join(lines))


def _pair_is_grid_adjacent(pair: dict[str, Any], alpha: list[float], beta: list[float]) -> bool:
    alpha_delta = abs(pair["a"]["alpha"] - pair["b"]["alpha"])
    beta_delta = abs(pair["a"]["beta"] - pair["b"]["beta"])
    alpha_between = any(
        value > min(pair["a"]["alpha"], pair["b"]["alpha"])
        and value < max(pair["a"]["alpha"], pair["b"]["alpha"])
        for value in alpha
    )
    beta_between = any(
        value > min(pair["a"]["beta"], pair["b"]["beta"])
        and value < max(pair["a"]["beta"], pair["b"]["beta"])
        for value in beta
    )
    return (alpha_delta == 0 and beta_delta > 0 and not beta_between) or (
        beta_delta == 0 and alpha_delta > 0 and not alpha_between
    )


def _coordinate_name(alpha: float, beta: float) -> str:
    def part(value: float) -> str:
        return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")

    return f"alpha-{part(alpha)}_beta-{part(beta)}"


def _coordinate(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, dict):
        raise BasinPlaneContractError(f"{label} is malformed")
    return _finite(value.get("alpha"), f"{label} alpha"), _finite(
        value.get("beta"), f"{label} beta"
    )


def _axis(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) < 2:
        raise BasinPlaneContractError(f"{label} axis is missing")
    result = [_finite(item, f"{label} axis") for item in value]
    if result != sorted(set(result)):
        raise BasinPlaneContractError(f"{label} axis must be sorted and unique")
    return result


def _is_adjacent(distance: float, axes: dict[str, list[float]]) -> bool:
    steps = [
        right - left
        for values in axes.values()
        for left, right in zip(values, values[1:])
    ]
    return any(math.isclose(distance, step, rel_tol=0, abs_tol=1e-12) for step in steps)


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise BasinPlaneContractError("distance group is empty")
    mean = sum(values) / len(values)
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "std": math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)),
    }


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise BasinPlaneContractError("correlation inputs have incompatible lengths")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_norm = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_norm = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2 + 1
        for position in order[index:end]:
            ranks[position] = rank
        index = end
    return ranks


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise BasinPlaneContractError(f"{label} must be finite numeric")
    return float(value)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BasinPlaneContractError(f"{label} must be a positive integer")
    return value


def _require_sha(value: Any, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise BasinPlaneContractError(f"{label} SHA256 is invalid")


def _validate_hashes(input_sha256: dict[str, str], decode_report_sha256: list[str]) -> None:
    if set(input_sha256) != {"latent", "atlas"} or not decode_report_sha256:
        raise BasinPlaneContractError("input identity map is incomplete")
    for label, value in input_sha256.items():
        _require_sha(value, f"{label} input")
    for index, value in enumerate(decode_report_sha256):
        _require_sha(value, f"decode report {index}")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BasinPlaneContractError(f"cannot read JSON {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _aliases_any(path: Path, candidates: list[Path]) -> bool:
    return any(_paths_alias(path, candidate) for candidate in candidates)


def _paths_alias(left: Path, right: Path) -> bool:
    try:
        return left.resolve(strict=False) == right.resolve(strict=False)
    except OSError:
        return str(left.absolute()) == str(right.absolute())


def _path_collisions(named_paths: list[tuple[str, Path]]) -> list[str]:
    collisions = []
    for index, (left_name, left_path) in enumerate(named_paths):
        for right_name, right_path in named_paths[index + 1 :]:
            if _paths_alias(left_path, right_path):
                collisions.append(f"{left_name}={right_name}")
    return collisions


def _failure_report_path(requested: Path, protected: list[Path]) -> Path:
    candidate = requested.with_name(requested.name + ".failure.json")
    suffix = 1
    while _aliases_any(candidate, protected):
        candidate = requested.with_name(requested.name + f".failure-{suffix}.json")
        suffix += 1
    return candidate


if __name__ == "__main__":
    raise SystemExit(main())
