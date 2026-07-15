"""Build a multiscale shared-grid atlas of raw mesh vertex support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_raw_mesh_preview import (
    choose_failure_report_path,
    map_binary_ply,
    paths_alias,
    sha256_file,
)


ROUTE = "shared_grid_vertex_surface_support"
FORBIDDEN_INFERENCES = [
    "vertex support occupancy is not watertight volume occupancy",
    "Jaccard distance is not global learned-manifold distance",
    "projected support deltas are not topology or winding evidence",
]
PROJECTIONS = (
    ("front_xz", 1),
    ("side_yz", 0),
    ("top_xy", 2),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", action="append", required=True, metavar="NAME=PLY")
    parser.add_argument("--grid-size", action="append", required=True, type=int)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-png", required=True, type=Path)
    return parser


def build_mesh_support_atlas(
    *,
    meshes: dict[str, Path],
    grid_sizes: list[int],
    reference: str,
    output_json: Path,
    output_png: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    meshes = {str(name): Path(path) for name, path in meshes.items()}
    output_json = Path(output_json)
    output_png = Path(output_png)
    named_paths = [(f"mesh:{name}", path) for name, path in meshes.items()]
    named_paths.extend((("output_json", output_json), ("output_png", output_png)))
    path_collisions = find_named_path_collisions(named_paths)
    effective_output_json = output_json
    if any("output_json" in collision for collision in path_collisions):
        effective_output_json = choose_failure_report_path(
            output_json,
            protected_paths=[path for _, path in named_paths],
        )
    report: dict[str, Any] = {
        "schema": "trellis2mlx.mesh_surface_support_atlas.v1",
        "status": "failed",
        "phase": None,
        "route": ROUTE,
        "embedding_authority": "none",
        "requested_output_json": str(output_json),
        "effective_output_json": str(effective_output_json),
        "output_png": str(output_png),
        "reference": reference,
        "grid_sizes": sorted(set(int(size) for size in grid_sizes)),
        "path_collisions": path_collisions,
        "forbidden_inferences": FORBIDDEN_INFERENCES,
        "last_trustworthy_evidence": {},
    }
    phase = "validate_request"
    try:
        output_png_is_protected = any(
            paths_alias(output_png, path) for path in meshes.values()
        )
        if not output_png_is_protected:
            output_png.unlink(missing_ok=True)
        validate_request(
            meshes=meshes,
            grid_sizes=report["grid_sizes"],
            reference=reference,
            path_collisions=path_collisions,
        )
        report["last_trustworthy_evidence"] = {"validated_request": True}
        output_png.unlink(missing_ok=True)

        phase = "read_source_identity"
        states: dict[str, dict[str, Any]] = {}
        sources: list[dict[str, Any]] = []
        for name, path in meshes.items():
            input_sha256 = sha256_file(path)
            vertices, face_records, layout = map_binary_ply(path)
            if vertices.shape[0] == 0:
                raise ValueError(f"mesh {name!r} has no vertices")
            phase = "validate_source_vertices"
            validate_finite_vertices(vertices, mesh_name=name)
            phase = "read_source_identity"
            bounds_min, bounds_max = chunked_bounds(vertices)
            source = {
                "name": name,
                "path": str(path),
                "sha256": input_sha256,
                "size_bytes": path.stat().st_size,
                "vertices": int(vertices.shape[0]),
                "faces": int(face_records.shape[0]),
                "bounds_min": bounds_min.tolist(),
                "bounds_max": bounds_max.tolist(),
                "read_mode": "numpy_memmap",
                "mapped_bytes": int(layout["mapped_bytes"]),
            }
            sources.append(source)
            states[name] = {"vertices": vertices, "source": source}
        report["sources"] = sources
        report["last_trustworthy_evidence"]["source_sha256"] = {
            source["name"]: source["sha256"] for source in sources
        }

        phase = "build_shared_coordinate_frame"
        shared_min = np.min([source["bounds_min"] for source in sources], axis=0)
        shared_max = np.max([source["bounds_max"] for source in sources], axis=0)
        extent = shared_max - shared_min
        extent[extent == 0] = 1.0
        report["shared_bounds"] = {
            "min": shared_min.tolist(),
            "max": shared_max.tolist(),
        }

        phase = "quantize_vertex_support"
        names = list(meshes)
        occupancies: dict[int, dict[str, np.ndarray]] = {
            size: {} for size in report["grid_sizes"]
        }
        for name in names:
            vertices = states[name]["vertices"]
            for size in report["grid_sizes"]:
                occupancies[size][name] = quantize_vertex_support(
                    vertices,
                    grid_size=size,
                    shared_min=shared_min,
                    extent=extent,
                )

        phase = "measure_multiscale_support"
        scales: dict[str, Any] = {}
        for size in report["grid_sizes"]:
            scale_occupancies = occupancies[size]
            scales[str(size)] = {
                "occupied_cells": {
                    name: int(np.count_nonzero(scale_occupancies[name])) for name in names
                },
                "pairwise_jaccard": pairwise_jaccard(names, scale_occupancies),
                "pairwise_jaccard_distance": pairwise_jaccard_distance(
                    names,
                    scale_occupancies,
                ),
            }
        report["names"] = names
        report["scales"] = scales

        phase = "render_support_atlas"
        output_png.parent.mkdir(parents=True, exist_ok=True)
        render_support_atlas(
            names=names,
            occupancies=occupancies[report["grid_sizes"][-1]],
            reference=reference,
            output_png=output_png,
            grid_size=report["grid_sizes"][-1],
        )

        phase = "validate_visual_output"
        output_size = output_png.stat().st_size
        if output_size <= 0:
            raise ValueError("support atlas PNG is blank")
        with Image.open(output_png) as image:
            pixels = np.asarray(image.convert("RGB"))
            if pixels.size == 0 or float(pixels.std()) == 0.0:
                raise ValueError("support atlas PNG has no visual variation")
            visual_size = [int(image.width), int(image.height)]

        report.update(
            {
                "status": "done",
                "phase": "done",
                "elapsed_seconds": elapsed(started),
                "output_png_sha256": sha256_file(output_png),
                "output_png_size_bytes": output_size,
                "output_png_dimensions": visual_size,
            }
        )
        write_report(effective_output_json, report)
        return report
    except Exception as exc:
        if phase in {"render_support_atlas", "validate_visual_output"} and output_png.exists():
            report["invalid_output_observed"] = {
                "sha256": sha256_file(output_png),
                "size_bytes": output_png.stat().st_size,
            }
            output_png.unlink()
            report["invalid_output_removed"] = True
        report.update(
            {
                "status": "failed",
                "phase": phase,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": elapsed(started),
            }
        )
        write_report(effective_output_json, report)
        raise


def validate_request(
    *,
    meshes: dict[str, Path],
    grid_sizes: list[int],
    reference: str,
    path_collisions: list[str],
) -> None:
    if path_collisions:
        raise ValueError(f"requested paths must be distinct: {path_collisions}")
    if len(meshes) < 2:
        raise ValueError("at least two named meshes are required")
    if any(not name for name in meshes):
        raise ValueError("mesh names must be non-empty")
    if reference not in meshes:
        raise ValueError(f"reference {reference!r} is not a named mesh")
    if not grid_sizes or any(size < 2 for size in grid_sizes):
        raise ValueError("grid sizes must be integers greater than one")


def quantize_vertex_support(
    vertices: np.ndarray,
    *,
    grid_size: int,
    shared_min: np.ndarray,
    extent: np.ndarray,
    chunk_size: int = 1_000_000,
) -> np.ndarray:
    occupancy = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    for start in range(0, vertices.shape[0], chunk_size):
        chunk = np.asarray(vertices[start : start + chunk_size], dtype=np.float32)
        normalized = (chunk - shared_min) / extent
        indices = np.floor(normalized * grid_size).astype(np.int64)
        np.clip(indices, 0, grid_size - 1, out=indices)
        occupancy[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    return occupancy


def pairwise_jaccard(names: list[str], occupancies: dict[str, np.ndarray]) -> list[list[float]]:
    matrix: list[list[float]] = []
    for left_name in names:
        row = []
        left = occupancies[left_name]
        for right_name in names:
            right = occupancies[right_name]
            intersection = int(np.count_nonzero(left & right))
            union = int(np.count_nonzero(left | right))
            row.append(float(intersection / union) if union else 1.0)
        matrix.append(row)
    return matrix


def pairwise_jaccard_distance(
    names: list[str],
    occupancies: dict[str, np.ndarray],
) -> list[list[float]]:
    return [[1.0 - value for value in row] for row in pairwise_jaccard(names, occupancies)]


def render_support_atlas(
    *,
    names: list[str],
    occupancies: dict[str, np.ndarray],
    reference: str,
    output_png: Path,
    grid_size: int,
) -> None:
    panel_size = 256
    label_width = 230
    header_height = 34
    row_height = panel_size + 28
    gap = 8
    delta_names = [name for name in names if name != reference]
    row_labels = [f"support: {name}" for name in names]
    row_labels.extend(f"delta: {reference} -> {name}" for name in delta_names)
    width = label_width + len(PROJECTIONS) * panel_size + (len(PROJECTIONS) - 1) * gap
    height = header_height + len(row_labels) * row_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (label, _) in enumerate(PROJECTIONS):
        x = label_width + column * (panel_size + gap)
        draw.text((x + 4, 10), label, fill=(20, 20, 20))

    row_index = 0
    for name in names:
        draw.text((8, header_height + row_index * row_height + 8), row_labels[row_index], fill=(20, 20, 20))
        for column, (_, axis) in enumerate(PROJECTIONS):
            panel = render_support_panel(occupancies[name], axis=axis)
            paste_panel(canvas, panel, row_index, column, label_width, header_height, row_height, panel_size, gap)
        row_index += 1

    reference_occupancy = occupancies[reference]
    for name in delta_names:
        draw.text((8, header_height + row_index * row_height + 8), row_labels[row_index], fill=(20, 20, 20))
        candidate = occupancies[name]
        for column, (_, axis) in enumerate(PROJECTIONS):
            panel = render_delta_panel(reference_occupancy, candidate, axis=axis)
            paste_panel(canvas, panel, row_index, column, label_width, header_height, row_height, panel_size, gap)
        row_index += 1

    draw.text(
        (8, height - 18),
        f"grid={grid_size}; support=dark; shared=gray; reference-only=red; candidate-only=blue",
        fill=(20, 20, 20),
    )
    canvas.save(output_png)


def render_support_panel(occupancy: np.ndarray, *, axis: int) -> np.ndarray:
    projected = orient_projection(np.any(occupancy, axis=axis))
    panel = np.full((*projected.shape, 3), 255, dtype=np.uint8)
    panel[projected] = np.array([45, 50, 52], dtype=np.uint8)
    return panel


def render_delta_panel(reference: np.ndarray, candidate: np.ndarray, *, axis: int) -> np.ndarray:
    shared = orient_projection(np.any(reference & candidate, axis=axis))
    reference_only = orient_projection(np.any(reference & ~candidate, axis=axis))
    candidate_only = orient_projection(np.any(candidate & ~reference, axis=axis))
    panel = np.full((*shared.shape, 3), 255, dtype=np.uint8)
    panel[shared] = np.array([175, 178, 180], dtype=np.uint8)
    panel[reference_only] = np.array([220, 65, 65], dtype=np.uint8)
    panel[candidate_only] = np.array([55, 105, 220], dtype=np.uint8)
    projected_overlap = reference_only & candidate_only
    panel[projected_overlap] = np.array([150, 70, 175], dtype=np.uint8)
    return panel


def orient_projection(projected: np.ndarray) -> np.ndarray:
    return np.flipud(np.asarray(projected).T)


def paste_panel(
    canvas: Image.Image,
    panel: np.ndarray,
    row: int,
    column: int,
    label_width: int,
    header_height: int,
    row_height: int,
    panel_size: int,
    gap: int,
) -> None:
    image = Image.fromarray(panel, mode="RGB").resize(
        (panel_size, panel_size),
        resample=Image.Resampling.NEAREST,
    )
    x = label_width + column * (panel_size + gap)
    y = header_height + row * row_height
    canvas.paste(image, (x, y))


def chunked_bounds(
    vertices: np.ndarray,
    *,
    chunk_size: int = 1_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    bounds_min = np.full(3, np.inf, dtype=np.float64)
    bounds_max = np.full(3, -np.inf, dtype=np.float64)
    for start in range(0, vertices.shape[0], chunk_size):
        chunk = vertices[start : start + chunk_size]
        bounds_min = np.minimum(bounds_min, chunk.min(axis=0))
        bounds_max = np.maximum(bounds_max, chunk.max(axis=0))
    return bounds_min, bounds_max


def validate_finite_vertices(
    vertices: np.ndarray,
    *,
    mesh_name: str,
    chunk_size: int = 1_000_000,
) -> None:
    for start in range(0, vertices.shape[0], chunk_size):
        chunk = vertices[start : start + chunk_size]
        finite = np.isfinite(chunk)
        if bool(np.all(finite)):
            continue
        first = np.argwhere(~finite)[0]
        vertex_index = start + int(first[0])
        coordinate_index = int(first[1])
        chunk_nonfinite_count = int(finite.size - np.count_nonzero(finite))
        raise ValueError(
            f"mesh {mesh_name!r} has non-finite vertex coordinates; "
            f"first_vertex={vertex_index}, coordinate={coordinate_index}, "
            f"chunk_nonfinite_count={chunk_nonfinite_count}"
        )


def find_named_path_collisions(named_paths: list[tuple[str, Path]]) -> list[str]:
    collisions: list[str] = []
    for index, (left_name, left_path) in enumerate(named_paths):
        for right_name, right_path in named_paths[index + 1 :]:
            if paths_alias(left_path, right_path):
                collisions.append(f"{left_name}={right_name}")
    return collisions


def parse_mesh_args(values: list[str]) -> dict[str, Path]:
    meshes: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"--mesh must use NAME=PLY, got {value!r}")
        if name in meshes:
            raise ValueError(f"duplicate mesh name: {name}")
        meshes[name] = Path(path)
    return meshes


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


def elapsed(started: float) -> float:
    return max(0.0, time.perf_counter() - started)


def write_request_failure_report(args: argparse.Namespace, exc: Exception) -> None:
    output_json = Path(args.output_json)
    output_png = Path(args.output_png)
    possible_mesh_paths = []
    for value in args.mesh:
        _, separator, path = value.partition("=")
        if separator and path:
            possible_mesh_paths.append(Path(path))
    protected_paths = [output_png, *possible_mesh_paths]
    effective_output_json = output_json
    if any(paths_alias(output_json, path) for path in protected_paths):
        effective_output_json = choose_failure_report_path(
            output_json,
            protected_paths=[output_json, *protected_paths],
        )
    else:
        output_json.unlink(missing_ok=True)
    if not any(paths_alias(output_png, path) for path in possible_mesh_paths):
        output_png.unlink(missing_ok=True)
    write_report(
        effective_output_json,
        {
            "schema": "trellis2mlx.mesh_surface_support_atlas.v1",
            "status": "failed",
            "phase": "request_validation",
            "route": ROUTE,
            "embedding_authority": "none",
            "requested_output_json": str(output_json),
            "effective_output_json": str(effective_output_json),
            "output_png": str(output_png),
            "reference": args.reference,
            "grid_sizes": sorted(set(int(size) for size in args.grid_size)),
            "forbidden_inferences": FORBIDDEN_INFERENCES,
            "last_trustworthy_evidence": {},
            "error_type": type(exc).__name__,
            "error": str(exc),
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        meshes = parse_mesh_args(args.mesh)
    except Exception as exc:
        write_request_failure_report(args, exc)
        return 1
    try:
        build_mesh_support_atlas(
            meshes=meshes,
            grid_sizes=args.grid_size,
            reference=args.reference,
            output_json=args.output_json,
            output_png=args.output_png,
        )
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
