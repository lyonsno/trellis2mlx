#!/usr/bin/env python3
"""Attribute visible backfacing pixels to GLB, UV, and clean mesh faces."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


SCHEMA = "trellis2mlx.mesh_culling_attribution.v1"
ROUTE = "cpu_visible_backface_face_attribution"
VISIBLE_EXTERIOR_VIEWS = {
    "+X": (0, 1),
    "-X": (0, -1),
    "+Y": (1, 1),
    "-Y": (1, -1),
    "+Z": (2, 1),
    "-Z": (2, -1),
}
PANELS = ("front_xz", "side_yz", "top_xy")
PANEL_FRONT_FACE = {
    "front_xz": "ccw",
    "side_yz": "cw",
    "top_xy": "cw",
}


class AttributionError(RuntimeError):
    def __init__(self, phase: str, message: str):
        super().__init__(message)
        self.phase = phase


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n")


def _cyclic_orders(face: np.ndarray) -> set[tuple[int, int, int]]:
    a, b, c = [int(v) for v in face]
    return {(a, b, c), (b, c, a), (c, a, b)}


def build_source_face_index_map(
    *,
    source_faces: np.ndarray,
    uv_faces: np.ndarray,
    vmapping: np.ndarray,
) -> dict[str, Any]:
    """Map each UV face row back to a source face row and orientation class."""
    source_by_key: dict[tuple[int, int, int], list[tuple[int, np.ndarray]]] = defaultdict(list)
    for source_index, face in enumerate(np.asarray(source_faces, dtype=np.int64)):
        key = tuple(sorted(int(v) for v in face))
        source_by_key[key].append((source_index, face))

    uv_faces = np.asarray(uv_faces, dtype=np.int64)
    mapped = np.asarray(vmapping, dtype=np.int64)[uv_faces]
    source_face_index = np.full(len(uv_faces), -1, dtype=np.int64)
    orientation = np.empty(len(uv_faces), dtype=object)
    summary = Counter()

    for face_index, face in enumerate(mapped):
        key = tuple(sorted(int(v) for v in face))
        candidates = source_by_key.get(key, [])
        if not candidates:
            orientation[face_index] = "unmatched"
            summary["unmatched"] += 1
            continue

        same_matches = [
            source_index
            for source_index, source_face in candidates
            if tuple(int(v) for v in face) in _cyclic_orders(source_face)
        ]
        reversed_matches = [
            source_index
            for source_index, source_face in candidates
            if tuple(int(v) for v in face[::-1]) in _cyclic_orders(source_face)
        ]

        if same_matches and not reversed_matches:
            source_face_index[face_index] = same_matches[0]
            orientation[face_index] = "same"
            summary["same"] += 1
        elif reversed_matches and not same_matches:
            source_face_index[face_index] = reversed_matches[0]
            orientation[face_index] = "reversed"
            summary["reversed"] += 1
        elif len(candidates) == 1:
            source_face_index[face_index] = candidates[0][0]
            orientation[face_index] = "ambiguous"
            summary["ambiguous"] += 1
        else:
            orientation[face_index] = "ambiguous"
            summary["ambiguous"] += 1

    for key in ("same", "reversed", "unmatched", "ambiguous"):
        summary.setdefault(key, 0)

    return {
        "source_face_index": source_face_index,
        "orientation": orientation,
        "summary": dict(summary),
    }


def _edge_function(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    return float((p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0]))


def _face_normals_and_areas(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    areas = lengths * 0.5
    unit = np.zeros_like(normals, dtype=np.float64)
    valid = lengths > 1e-12
    unit[valid] = normals[valid] / lengths[valid, None]
    return unit, areas


def visible_backface_attribution(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    view_name: str,
    image_size: int,
) -> dict[str, Any]:
    """Rasterize one axis view and return visible/backfacing face pixel counts."""
    if view_name not in VISIBLE_EXTERIOR_VIEWS:
        raise ValueError(f"unknown view {view_name}")
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    normals, _ = _face_normals_and_areas(vertices, faces)

    depth_axis, sign = VISIBLE_EXTERIOR_VIEWS[view_name]
    axes = [axis for axis in range(3) if axis != depth_axis]
    outward_axis = np.zeros(3, dtype=np.float64)
    outward_axis[depth_axis] = float(sign)

    coords = vertices[:, axes]
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = maxs - mins
    span[span == 0.0] = 1.0

    margin = max(2.0, image_size * 0.05)
    scale = min((image_size - 2.0 * margin) / span[0], (image_size - 2.0 * margin) / span[1])
    offset = np.array(
        [
            (image_size - span[0] * scale) / 2.0 - mins[0] * scale,
            (image_size - span[1] * scale) / 2.0 - mins[1] * scale,
        ],
        dtype=np.float64,
    )
    projected = coords * scale + offset
    depths = vertices[:, depth_axis] * float(sign)

    z_buffer = np.full((image_size, image_size), -np.inf, dtype=np.float64)
    face_buffer = np.full((image_size, image_size), -1, dtype=np.int64)

    for face_index, face in enumerate(faces):
        tri_2d = projected[face]
        if not np.isfinite(tri_2d).all():
            continue
        area2 = _edge_function(tri_2d[0], tri_2d[1], tri_2d[2])
        if abs(area2) < 1e-8:
            continue

        min_xy = np.floor(tri_2d.min(axis=0)).astype(int)
        max_xy = np.ceil(tri_2d.max(axis=0)).astype(int)
        x0 = max(0, int(min_xy[0]))
        y0 = max(0, int(min_xy[1]))
        x1 = min(image_size - 1, int(max_xy[0]))
        y1 = min(image_size - 1, int(max_xy[1]))
        if x0 > x1 or y0 > y1:
            continue

        tri_depths = depths[face]
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                sample = np.array([x + 0.5, y + 0.5], dtype=np.float64)
                w0 = _edge_function(tri_2d[1], tri_2d[2], sample) / area2
                w1 = _edge_function(tri_2d[2], tri_2d[0], sample) / area2
                w2 = _edge_function(tri_2d[0], tri_2d[1], sample) / area2
                if w0 < -1e-8 or w1 < -1e-8 or w2 < -1e-8:
                    continue
                depth = w0 * tri_depths[0] + w1 * tri_depths[1] + w2 * tri_depths[2]
                if depth > z_buffer[y, x]:
                    z_buffer[y, x] = depth
                    face_buffer[y, x] = face_index

    visible = face_buffer >= 0
    visible_face_ids = face_buffer[visible]
    visible_pixels_by_face = Counter(int(face) for face in visible_face_ids)
    if len(visible_face_ids) == 0:
        return {
            "view": view_name,
            "visible_pixels": 0,
            "backfacing_visible_pixels": 0,
            "backfacing_visible_ratio": 0.0,
            "visible_pixels_by_face": {},
            "backface_pixels_by_face": {},
        }

    dots = normals[visible_face_ids] @ outward_axis
    back = dots < -1e-6
    back_face_ids = visible_face_ids[back]
    backface_pixels_by_face = Counter(int(face) for face in back_face_ids)
    return {
        "view": view_name,
        "image_size": int(image_size),
        "visible_pixels": int(len(visible_face_ids)),
        "backfacing_visible_pixels": int(back.sum()),
        "backfacing_visible_ratio": float(back.sum() / len(visible_face_ids)),
        "visible_faces": int(len(visible_pixels_by_face)),
        "backfacing_visible_faces": int(len(backface_pixels_by_face)),
        "visible_pixels_by_face": dict(visible_pixels_by_face),
        "backface_pixels_by_face": dict(backface_pixels_by_face),
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise AttributionError("load_inputs", f"missing checkpoint: {path}")
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _load_glb(path: Path) -> trimesh.Trimesh:
    if not path.exists():
        raise AttributionError("load_inputs", f"missing GLB: {path}")
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        loaded = trimesh.util.concatenate(meshes) if meshes else trimesh.Trimesh()
    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise AttributionError("load_inputs", "GLB mesh is empty")
    return loaded


def _panel_axes(panel: str) -> tuple[int, int, int]:
    if panel == "front_xz":
        return 0, 2, 1
    if panel == "side_yz":
        return 1, 2, 0
    if panel == "top_xy":
        return 0, 1, 2
    raise ValueError(panel)


def _uv_vertices_to_glb_space(uv_vertices: np.ndarray) -> np.ndarray:
    export_vertices = np.asarray(uv_vertices, dtype=np.float64).copy()
    export_vertices[:, 1], export_vertices[:, 2] = (
        export_vertices[:, 2].copy(),
        -export_vertices[:, 1].copy(),
    )
    return export_vertices


def export_space_identity(*, uv_vertices: np.ndarray, glb_vertices: np.ndarray) -> dict[str, Any]:
    uv_vertices = np.asarray(uv_vertices, dtype=np.float64)
    glb_vertices = np.asarray(glb_vertices, dtype=np.float64)
    if uv_vertices.shape != glb_vertices.shape:
        return {
            "transform": "glb_xyz_from_uv_x_z_neg_y",
            "vertices_match_export_transform": False,
            "max_abs_error": None,
            "shape_mismatch": {
                "uv_vertices": list(uv_vertices.shape),
                "glb_vertices": list(glb_vertices.shape),
            },
        }
    expected = _uv_vertices_to_glb_space(uv_vertices)
    max_abs_error = float(np.max(np.abs(expected - glb_vertices))) if expected.size else 0.0
    return {
        "transform": "glb_xyz_from_uv_x_z_neg_y",
        "vertices_match_export_transform": bool(max_abs_error <= 1e-6),
        "max_abs_error": max_abs_error,
    }


def default_front_face_for_panel(panel: str) -> str:
    try:
        return PANEL_FRONT_FACE[panel]
    except KeyError as exc:
        raise ValueError(panel) from exc


def _projected_panel_orientation(vertices: np.ndarray, faces: np.ndarray) -> dict[str, dict[str, Any]]:
    result = {}
    for panel in PANELS:
        axis_a, axis_b, _ = _panel_axes(panel)
        coords = vertices[:, [axis_a, axis_b]]
        front = 0
        back = 0
        degenerate = 0
        for face in faces:
            pts = coords[face]
            area = _signed_polygon_area(pts)
            if abs(area) < 1e-12:
                degenerate += 1
            elif area > 0:
                front += 1
            else:
                back += 1
        result[panel] = {
            "front_faces": front,
            "back_faces": back,
            "degenerate_faces": degenerate,
            "back_ratio": float(back / (front + back)) if (front + back) else 0.0,
        }
    return result


def projected_front_face_missing_attribution(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    panel: str,
    image_size: int,
    front_face: str = "ccw",
) -> dict[str, Any]:
    """Approximate render_glb_witness front-face holes and attribute them to faces."""
    if front_face not in {"ccw", "cw"}:
        raise ValueError(front_face)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    axis_a, axis_b, depth_axis = _panel_axes(panel)

    coords = vertices[:, [axis_a, axis_b]]
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = maxs - mins
    span[span == 0.0] = 1.0

    margin = image_size * 0.09
    scale = min((image_size - 2.0 * margin) / span[0], (image_size - 2.0 * margin) / span[1])
    offset = np.array(
        [
            (image_size - span[0] * scale) / 2.0 - mins[0] * scale,
            (image_size - span[1] * scale) / 2.0 - mins[1] * scale,
        ],
        dtype=np.float64,
    )
    projected = coords * scale + offset
    projected[:, 1] = image_size - projected[:, 1]

    depths = vertices[faces, depth_axis].mean(axis=1)
    order = np.argsort(depths)
    double_buffer = np.full((image_size, image_size), -1, dtype=np.int64)
    front_buffer = np.full((image_size, image_size), -1, dtype=np.int64)

    projected_is_front = np.zeros(len(faces), dtype=bool)
    projected_valid = np.zeros(len(faces), dtype=bool)
    for face_index in order:
        pts = projected[faces[face_index]]
        if not np.isfinite(pts).all():
            continue
        signed_area = _signed_polygon_area(pts)
        if abs(signed_area) < 0.02:
            continue
        projected_valid[face_index] = True
        is_front = signed_area > 0 if front_face == "ccw" else signed_area < 0
        projected_is_front[face_index] = is_front

        min_xy = np.floor(pts.min(axis=0)).astype(int)
        max_xy = np.ceil(pts.max(axis=0)).astype(int)
        x0 = max(0, int(min_xy[0]))
        y0 = max(0, int(min_xy[1]))
        x1 = min(image_size - 1, int(max_xy[0]))
        y1 = min(image_size - 1, int(max_xy[1]))
        if x0 > x1 or y0 > y1:
            continue

        area2 = _edge_function(pts[0], pts[1], pts[2])
        if abs(area2) < 1e-8:
            continue
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                sample = np.array([x + 0.5, y + 0.5], dtype=np.float64)
                w0 = _edge_function(pts[1], pts[2], sample) / area2
                w1 = _edge_function(pts[2], pts[0], sample) / area2
                w2 = _edge_function(pts[0], pts[1], sample) / area2
                if w0 < -1e-8 or w1 < -1e-8 or w2 < -1e-8:
                    continue
                double_buffer[y, x] = face_index
                if is_front:
                    front_buffer[y, x] = face_index

    double_drawn = double_buffer >= 0
    front_drawn = front_buffer >= 0
    missing = double_drawn & ~front_drawn
    missing_faces = double_buffer[missing]
    missing_by_face = Counter(int(face) for face in missing_faces)
    return {
        "panel": panel,
        "image_size": int(image_size),
        "front_face": front_face,
        "double_sided_pixels": int(double_drawn.sum()),
        "front_face_pixels": int(front_drawn.sum()),
        "missing_pixels": int(missing.sum()),
        "missing_pixel_ratio_vs_double_sided": (
            float(missing.sum() / double_drawn.sum()) if double_drawn.sum() else 0.0
        ),
        "projected_valid_faces": int(projected_valid.sum()),
        "projected_front_faces": int((projected_valid & projected_is_front).sum()),
        "projected_back_faces": int((projected_valid & ~projected_is_front).sum()),
        "missing_faces": int(len(missing_by_face)),
        "missing_pixels_by_face": dict(missing_by_face),
    }


def _signed_polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _component_labels(faces: np.ndarray) -> np.ndarray:
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_index, face in enumerate(np.asarray(faces, dtype=np.int64)):
        for corner in range(3):
            edge = tuple(sorted((int(face[corner]), int(face[(corner + 1) % 3]))))
            edge_to_faces[edge].append(face_index)

    rows = []
    cols = []
    for incident in edge_to_faces.values():
        for i in range(len(incident)):
            for j in range(i + 1, len(incident)):
                rows.extend([incident[i], incident[j]])
                cols.extend([incident[j], incident[i]])
    if not rows:
        return np.zeros(len(faces), dtype=np.int32)
    graph = sparse.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(len(faces), len(faces)))
    _, labels = connected_components(graph, directed=False)
    return labels.astype(np.int32)


def _boundary_face_mask(faces: np.ndarray) -> np.ndarray:
    edge_counts: Counter[tuple[int, int]] = Counter()
    face_edges = []
    for face in np.asarray(faces, dtype=np.int64):
        edges = []
        for corner in range(3):
            edge = tuple(sorted((int(face[corner]), int(face[(corner + 1) % 3]))))
            edge_counts[edge] += 1
            edges.append(edge)
        face_edges.append(edges)
    return np.array([any(edge_counts[edge] == 1 for edge in edges) for edges in face_edges], dtype=bool)


def _top_entries(
    *,
    pixel_counter: Counter[int],
    source_face_index: np.ndarray,
    orientation: np.ndarray,
    clean_vertices: np.ndarray,
    clean_faces: np.ndarray,
    clean_component_labels: np.ndarray,
    clean_boundary_mask: np.ndarray,
    limit: int,
) -> list[dict[str, Any]]:
    clean_normals, clean_areas = _face_normals_and_areas(clean_vertices, clean_faces)
    centers = clean_vertices[clean_faces].mean(axis=1)
    mesh_center = clean_vertices.mean(axis=0)
    radial = centers - mesh_center
    radial_norm = np.linalg.norm(radial, axis=1)
    radial_dot = np.zeros(len(clean_faces), dtype=np.float64)
    usable = radial_norm > 1e-12
    radial_dot[usable] = np.sum(clean_normals[usable] * radial[usable], axis=1) / radial_norm[usable]

    entries = []
    for glb_face, pixels in pixel_counter.most_common(limit):
        clean_face = int(source_face_index[glb_face]) if glb_face < len(source_face_index) else -1
        entry: dict[str, Any] = {
            "glb_face": int(glb_face),
            "uv_face": int(glb_face),
            "pixels": int(pixels),
            "source_orientation": str(orientation[glb_face]) if glb_face < len(orientation) else "out_of_range",
            "clean_face": clean_face,
        }
        if clean_face >= 0:
            entry.update(
                {
                    "clean_component": int(clean_component_labels[clean_face]),
                    "clean_boundary_face": bool(clean_boundary_mask[clean_face]),
                    "clean_area": float(clean_areas[clean_face]),
                    "clean_radial_dot": float(radial_dot[clean_face]),
                    "clean_center": centers[clean_face].tolist(),
                    "clean_normal": clean_normals[clean_face].tolist(),
                }
            )
        entries.append(entry)
    return entries


def _top_projected_missing_entries(
    *,
    panel_reports: dict[str, dict[str, Any]],
    source_face_index: np.ndarray,
    orientation: np.ndarray,
    clean_vertices: np.ndarray,
    clean_faces: np.ndarray,
    clean_component_labels: np.ndarray,
    clean_boundary_mask: np.ndarray,
    limit: int,
) -> list[dict[str, Any]]:
    merged: Counter[int] = Counter()
    by_panel: dict[int, dict[str, int]] = defaultdict(dict)
    for panel, report in panel_reports.items():
        for face, pixels in report["missing_pixels_by_face"].items():
            face_i = int(face)
            pixels_i = int(pixels)
            merged[face_i] += pixels_i
            by_panel[face_i][panel] = pixels_i
    entries = _top_entries(
        pixel_counter=merged,
        source_face_index=source_face_index,
        orientation=orientation,
        clean_vertices=clean_vertices,
        clean_faces=clean_faces,
        clean_component_labels=clean_component_labels,
        clean_boundary_mask=clean_boundary_mask,
        limit=limit,
    )
    for entry in entries:
        entry["pixels_by_panel"] = by_panel.get(entry["glb_face"], {})
    return entries


def build_report(
    *,
    checkpoint_dir: Path,
    glb: Path,
    report_path: Path,
    image_size: int,
    top_n: int,
) -> dict[str, Any]:
    mesh_clean = _load_npz(checkpoint_dir / "mesh_clean.npz")
    mesh_uv = _load_npz(checkpoint_dir / "mesh_uv.npz")
    if "vmapping" not in mesh_uv:
        raise AttributionError("load_inputs", f"{checkpoint_dir / 'mesh_uv.npz'} lacks vmapping")

    clean_vertices = np.asarray(mesh_clean["vertices"], dtype=np.float64)
    clean_faces = np.asarray(mesh_clean["faces"], dtype=np.int64)
    uv_faces = np.asarray(mesh_uv["faces"], dtype=np.int64)
    uv_vertices = np.asarray(mesh_uv["vertices"], dtype=np.float64)
    vmapping = np.asarray(mesh_uv["vmapping"], dtype=np.int64)
    mesh = _load_glb(glb)
    glb_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    glb_faces = np.asarray(mesh.faces, dtype=np.int64)

    if glb_faces.shape != uv_faces.shape:
        raise AttributionError("validate_inputs", f"GLB faces {glb_faces.shape} do not match mesh_uv faces {uv_faces.shape}")
    face_rows_equal = bool(np.array_equal(glb_faces, uv_faces))
    if not face_rows_equal:
        raise AttributionError("validate_inputs", "GLB face rows differ from mesh_uv faces; face-index attribution would lie")

    source_map = build_source_face_index_map(
        source_faces=clean_faces,
        uv_faces=uv_faces,
        vmapping=vmapping,
    )
    clean_source = source_map["source_face_index"]
    source_orientation = source_map["orientation"]

    per_view = {}
    all_backface_pixels: Counter[int] = Counter()
    all_visible_pixels: Counter[int] = Counter()
    for view_name in VISIBLE_EXTERIOR_VIEWS:
        view = visible_backface_attribution(
            vertices=glb_vertices,
            faces=glb_faces,
            view_name=view_name,
            image_size=image_size,
        )
        per_view[view_name] = {
            key: value
            for key, value in view.items()
            if key not in ("visible_pixels_by_face", "backface_pixels_by_face")
        }
        all_visible_pixels.update({int(k): int(v) for k, v in view["visible_pixels_by_face"].items()})
        all_backface_pixels.update({int(k): int(v) for k, v in view["backface_pixels_by_face"].items()})

    projected_missing = {
        panel: projected_front_face_missing_attribution(
            vertices=glb_vertices,
            faces=glb_faces,
            panel=panel,
            image_size=image_size,
            front_face=default_front_face_for_panel(panel),
        )
        for panel in PANELS
    }
    projected_missing_pixels: Counter[int] = Counter()
    for panel_report in projected_missing.values():
        projected_missing_pixels.update(
            {int(k): int(v) for k, v in panel_report["missing_pixels_by_face"].items()}
        )

    clean_component_labels = _component_labels(clean_faces)
    clean_boundary_mask = _boundary_face_mask(clean_faces)
    mapped_backface_clean = np.array(
        [clean_source[int(face)] for face in all_backface_pixels if clean_source[int(face)] >= 0],
        dtype=np.int64,
    )
    mapped_visible_clean = np.array(
        [clean_source[int(face)] for face in all_visible_pixels if clean_source[int(face)] >= 0],
        dtype=np.int64,
    )
    mapped_projected_missing_clean = np.array(
        [clean_source[int(face)] for face in projected_missing_pixels if clean_source[int(face)] >= 0],
        dtype=np.int64,
    )

    component_sizes = Counter(int(label) for label in clean_component_labels)
    backface_components = Counter(int(clean_component_labels[face]) for face in mapped_backface_clean)
    visible_components = Counter(int(clean_component_labels[face]) for face in mapped_visible_clean)
    boundary_backface = int(clean_boundary_mask[mapped_backface_clean].sum()) if len(mapped_backface_clean) else 0
    boundary_visible = int(clean_boundary_mask[mapped_visible_clean].sum()) if len(mapped_visible_clean) else 0
    boundary_projected_missing = (
        int(clean_boundary_mask[mapped_projected_missing_clean].sum())
        if len(mapped_projected_missing_clean)
        else 0
    )
    normal_backface_face_set = set(int(face) for face in all_backface_pixels)
    projected_missing_face_set = set(int(face) for face in projected_missing_pixels)
    overlap_face_set = normal_backface_face_set & projected_missing_face_set
    overlap_missing_pixels = sum(projected_missing_pixels[face] for face in overlap_face_set)
    total_projected_missing_pixels = sum(projected_missing_pixels.values())

    top_backface_entries = _top_entries(
        pixel_counter=all_backface_pixels,
        source_face_index=clean_source,
        orientation=source_orientation,
        clean_vertices=clean_vertices,
        clean_faces=clean_faces,
        clean_component_labels=clean_component_labels,
        clean_boundary_mask=clean_boundary_mask,
        limit=top_n,
    )
    top_projected_missing_entries = _top_projected_missing_entries(
        panel_reports=projected_missing,
        source_face_index=clean_source,
        orientation=source_orientation,
        clean_vertices=clean_vertices,
        clean_faces=clean_faces,
        clean_component_labels=clean_component_labels,
        clean_boundary_mask=clean_boundary_mask,
        limit=top_n,
    )

    return {
        "schema": SCHEMA,
        "status": "ok",
        "route": ROUTE,
        "evidence_use_class": "diagnostic_face_level_visible_backface_attribution",
        "checkpoint_dir": str(checkpoint_dir),
        "glb": str(glb),
        "report_json": str(report_path),
        "image_size": int(image_size),
        "face_identity": {
            "glb_faces": int(len(glb_faces)),
            "mesh_uv_faces": int(len(uv_faces)),
            "mesh_clean_faces": int(len(clean_faces)),
            "glb_faces_equal_mesh_uv_faces": face_rows_equal,
            "uv_to_clean_summary": source_map["summary"],
            "export_space_identity": export_space_identity(
                uv_vertices=uv_vertices,
                glb_vertices=glb_vertices,
            ),
        },
        "visible_backface_summary": {
            "visible_faces": int(len(all_visible_pixels)),
            "backfacing_faces": int(len(all_backface_pixels)),
            "visible_pixels": int(sum(all_visible_pixels.values())),
            "backfacing_pixels": int(sum(all_backface_pixels.values())),
            "backfacing_pixel_ratio": (
                float(sum(all_backface_pixels.values()) / sum(all_visible_pixels.values()))
                if all_visible_pixels
                else 0.0
            ),
            "mapped_backfacing_clean_faces": int(len(mapped_backface_clean)),
            "mapped_visible_clean_faces": int(len(mapped_visible_clean)),
            "boundary_backfacing_clean_faces": boundary_backface,
            "boundary_visible_clean_faces": boundary_visible,
            "boundary_backfacing_clean_face_ratio": (
                float(boundary_backface / len(mapped_backface_clean)) if len(mapped_backface_clean) else 0.0
            ),
            "boundary_visible_clean_face_ratio": (
                float(boundary_visible / len(mapped_visible_clean)) if len(mapped_visible_clean) else 0.0
            ),
        },
        "views": per_view,
        "component_summary": {
            "component_count": int(len(component_sizes)),
            "largest_components": [
                {
                    "component": int(component),
                    "faces": int(size),
                    "visible_backface_faces": int(backface_components.get(component, 0)),
                    "visible_faces": int(visible_components.get(component, 0)),
                }
                for component, size in component_sizes.most_common(10)
            ],
        },
        "projected_panel_orientation": _projected_panel_orientation(glb_vertices, glb_faces),
        "projected_front_face_missing": {
            panel: {
                key: value
                for key, value in report.items()
                if key != "missing_pixels_by_face"
            }
            for panel, report in projected_missing.items()
        },
        "projected_missing_summary": {
            "projected_missing_faces": int(len(projected_missing_pixels)),
            "projected_missing_pixels": int(total_projected_missing_pixels),
            "mapped_projected_missing_clean_faces": int(len(mapped_projected_missing_clean)),
            "boundary_projected_missing_clean_faces": int(boundary_projected_missing),
            "boundary_projected_missing_clean_face_ratio": (
                float(boundary_projected_missing / len(mapped_projected_missing_clean))
                if len(mapped_projected_missing_clean)
                else 0.0
            ),
            "normal_backface_face_overlap": int(len(overlap_face_set)),
            "normal_backface_overlap_ratio_of_projected_missing_faces": (
                float(len(overlap_face_set) / len(projected_missing_face_set))
                if projected_missing_face_set
                else 0.0
            ),
            "normal_backface_overlap_projected_missing_pixels": int(overlap_missing_pixels),
            "normal_backface_overlap_ratio_of_projected_missing_pixels": (
                float(overlap_missing_pixels / total_projected_missing_pixels)
                if total_projected_missing_pixels
                else 0.0
            ),
        },
        "top_backface_faces": top_backface_entries,
        "top_projected_missing_faces": top_projected_missing_entries,
        "forbidden_to_prove": [
            "not a renderer-ground-truth hardware culling proof",
            "not root-cause closure",
            "not proof that visible artifacts are fixed or absent",
        ],
    }


def _failure_report(
    *,
    phase: str,
    error: str,
    checkpoint_dir: Path,
    glb: Path,
    report_path: Path,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "error",
        "route": ROUTE,
        "phase": phase,
        "error": error,
        "checkpoint_dir": str(checkpoint_dir),
        "glb": str(glb),
        "report_json": str(report_path),
        "last_trustworthy_evidence": {
            "checkpoint_dir_exists": checkpoint_dir.exists(),
            "glb_exists": glb.exists(),
            "mesh_clean_exists": (checkpoint_dir / "mesh_clean.npz").exists(),
            "mesh_uv_exists": (checkpoint_dir / "mesh_uv.npz").exists(),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--glb", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--top-n", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.image_size < 16:
            raise AttributionError("parse_args", "--image-size must be at least 16")
        report = build_report(
            checkpoint_dir=args.checkpoint_dir,
            glb=args.glb,
            report_path=args.report,
            image_size=args.image_size,
            top_n=args.top_n,
        )
        _write_json(args.report, report)
        print(f"wrote report: {args.report}", flush=True)
        return 0
    except AttributionError as exc:
        _write_json(
            args.report,
            _failure_report(
                phase=exc.phase,
                error=str(exc),
                checkpoint_dir=args.checkpoint_dir,
                glb=args.glb,
                report_path=args.report,
            ),
        )
        print(f"{exc.phase}: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
