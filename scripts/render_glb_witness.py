#!/usr/bin/env python3
"""Render a deterministic GLB witness image and JSON evidence report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import trimesh


PANELS = ("front_xz", "side_yz", "top_xy")
ROUTE = "software_projected_mesh_witness"
BACKGROUND = (248, 248, 246)
LINE = (72, 76, 82)
LABEL = (30, 34, 40)


class WitnessError(RuntimeError):
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


def _last_trustworthy_evidence(input_path: Path, output_path: Path) -> dict[str, Any]:
    return {
        "input_exists": input_path.exists(),
        "input_size_bytes": input_path.stat().st_size if input_path.exists() else None,
        "output_exists": output_path.exists(),
        "output_size_bytes": output_path.stat().st_size if output_path.exists() else None,
    }


def _failure_report(
    *,
    phase: str,
    error: str,
    input_path: Path,
    output_path: Path,
    report_path: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "error",
        "route": ROUTE,
        "phase": phase,
        "input_glb": str(input_path),
        "output_png": str(output_path),
        "report_json": str(report_path),
        "error": error,
        "last_trustworthy_evidence": _last_trustworthy_evidence(input_path, output_path),
    }
    if extra:
        payload.update(extra)
    return payload


def _load_mesh(path: Path) -> trimesh.Trimesh:
    if not path.exists():
        raise WitnessError("load_mesh", f"input GLB does not exist: {path}")

    try:
        loaded = trimesh.load(path, force="mesh", process=False)
    except Exception as exc:  # pragma: no cover - exercised by malformed local artifacts
        raise WitnessError("load_mesh", f"failed to load GLB: {exc}") from exc

    if isinstance(loaded, trimesh.Scene):
        geometries = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geometries:
            return trimesh.Trimesh()
        if len(geometries) == 1:
            return geometries[0]
        return trimesh.util.concatenate(geometries)

    return loaded


def _validate_mesh(mesh: trimesh.Trimesh) -> None:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[0] == 0:
        raise WitnessError("validate_mesh", "mesh has no vertices")
    if faces.ndim != 2 or faces.shape[0] == 0:
        raise WitnessError("validate_mesh", "mesh has no faces")
    if faces.shape[1] != 3:
        raise WitnessError("validate_mesh", f"expected triangular faces, found shape {faces.shape}")
    if not np.isfinite(vertices).all():
        raise WitnessError("validate_mesh", "mesh contains non-finite vertex coordinates")


def _material_base_color(mesh: trimesh.Trimesh) -> np.ndarray:
    material = getattr(getattr(mesh, "visual", None), "material", None)
    factor = getattr(material, "baseColorFactor", None)
    if factor is None:
        factor = getattr(material, "main_color", None)
    if factor is None:
        return np.array([186, 188, 190], dtype=np.float64)
    color = np.asarray(factor, dtype=np.float64)
    if color.size >= 3:
        color = color[:3]
        if color.max(initial=0) <= 1.0:
            color *= 255.0
        return np.clip(color, 0, 255)
    return np.array([186, 188, 190], dtype=np.float64)


def _texture_image(mesh: trimesh.Trimesh) -> Image.Image | None:
    material = getattr(getattr(mesh, "visual", None), "material", None)
    for attr in ("baseColorTexture", "image"):
        texture = getattr(material, attr, None)
        if texture is None:
            continue
        if isinstance(texture, Image.Image):
            return texture.convert("RGB")
        if isinstance(texture, np.ndarray):
            return Image.fromarray(texture.astype(np.uint8)).convert("RGB")
    return None


def _sample_texture_colors(mesh: trimesh.Trimesh) -> np.ndarray | None:
    uv = getattr(getattr(mesh, "visual", None), "uv", None)
    image = _texture_image(mesh)
    if uv is None or image is None:
        return None

    faces = np.asarray(mesh.faces)
    uv = np.asarray(uv, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[0] <= faces.max(initial=-1) or uv.shape[1] < 2:
        return None

    width, height = image.size
    pixels = np.asarray(image, dtype=np.float64)
    face_uv = uv[faces].mean(axis=1)
    u = np.mod(face_uv[:, 0], 1.0)
    v = np.mod(face_uv[:, 1], 1.0)
    x = np.clip(np.rint(u * (width - 1)).astype(np.int64), 0, width - 1)
    y = np.clip(np.rint((1.0 - v) * (height - 1)).astype(np.int64), 0, height - 1)
    return pixels[y, x, :3]


def _face_colors(mesh: trimesh.Trimesh) -> tuple[np.ndarray, str]:
    texture_colors = _sample_texture_colors(mesh)
    if texture_colors is not None:
        return np.clip(texture_colors, 0, 255).astype(np.uint8), "texture_uv_centroid"

    visual = getattr(mesh, "visual", None)
    vertex_colors = getattr(visual, "vertex_colors", None)
    if vertex_colors is not None and len(vertex_colors) >= len(mesh.vertices):
        colors = np.asarray(vertex_colors, dtype=np.float64)[:, :3]
        return np.clip(colors[np.asarray(mesh.faces)].mean(axis=1), 0, 255).astype(np.uint8), "vertex_color_mean"

    face_colors = getattr(visual, "face_colors", None)
    if face_colors is not None and len(face_colors) >= len(mesh.faces):
        colors = np.asarray(face_colors, dtype=np.float64)[:, :3]
        return np.clip(colors, 0, 255).astype(np.uint8), "face_color"

    color = _material_base_color(mesh)
    colors = np.repeat(color[None, :], len(mesh.faces), axis=0)
    return np.clip(colors, 0, 255).astype(np.uint8), "material_or_default"


def _panel_axes(panel: str) -> tuple[int, int, int]:
    if panel == "front_xz":
        return 0, 2, 1
    if panel == "side_yz":
        return 1, 2, 0
    if panel == "top_xy":
        return 0, 1, 2
    raise ValueError(panel)


def _project_panel(
    *,
    mesh: trimesh.Trimesh,
    face_colors: np.ndarray,
    panel: str,
    size: int,
) -> tuple[Image.Image, dict[str, Any]]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    axis_a, axis_b, depth_axis = _panel_axes(panel)

    coords = vertices[:, [axis_a, axis_b]]
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = maxs - mins
    span[span == 0] = 1.0

    margin = size * 0.09
    scale = min((size - 2 * margin) / span[0], (size - 2 * margin) / span[1])
    offset = np.array(
        [
            (size - span[0] * scale) / 2.0 - mins[0] * scale,
            (size - span[1] * scale) / 2.0 - mins[1] * scale,
        ]
    )
    projected = coords * scale + offset
    projected[:, 1] = size - projected[:, 1]

    depths = vertices[faces, depth_axis].mean(axis=1)
    order = np.argsort(depths)

    image = Image.new("RGB", (size, size), BACKGROUND)
    draw = ImageDraw.Draw(image, "RGBA")

    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    if normals.shape[0] != faces.shape[0]:
        normals = np.zeros((faces.shape[0], 3), dtype=np.float64)
    light = np.array([0.35, -0.45, 0.82], dtype=np.float64)
    light /= np.linalg.norm(light)
    faces_drawn = 0
    faces_skipped = 0

    for face_index in order:
        pts = projected[faces[face_index]]
        if not np.isfinite(pts).all():
            faces_skipped += 1
            continue
        if abs(_polygon_area(pts)) < 0.02:
            faces_skipped += 1
            continue

        base = face_colors[face_index].astype(np.float64)
        shade = 0.72 + 0.28 * max(float(np.dot(normals[face_index], light)), 0.0)
        color = tuple(np.clip(base * shade, 0, 255).astype(np.uint8).tolist()) + (235,)
        outline = LINE + (105,)
        draw.polygon([tuple(p) for p in pts], fill=color, outline=outline)
        faces_drawn += 1

    draw.text((18, 16), panel, fill=LABEL)
    return image, {
        "panel": panel,
        "faces_rendered": int(faces.shape[0]),
        "faces_drawn": faces_drawn,
        "faces_skipped": faces_skipped,
        "scale": float(scale),
        "axes": [axis_a, axis_b],
        "depth_axis": depth_axis,
    }


def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _render_witness(mesh: trimesh.Trimesh, output_path: Path, size: int) -> dict[str, Any]:
    face_colors, color_route = _face_colors(mesh)
    panel_images = []
    panel_reports = []
    for panel in PANELS:
        image, report = _project_panel(mesh=mesh, face_colors=face_colors, panel=panel, size=size)
        panel_images.append(image)
        panel_reports.append(report)

    output = Image.new("RGB", (size * len(panel_images), size), BACKGROUND)
    for index, image in enumerate(panel_images):
        output.paste(image, (index * size, 0))

    pixels = np.asarray(output.convert("RGB"))
    pixel_std = float(pixels.std())
    faces_drawn = sum(panel["faces_drawn"] for panel in panel_reports)
    nonblank = bool(
        faces_drawn > 0
        and pixel_std > 1.0
        and np.unique(pixels.reshape(-1, 3), axis=0).shape[0] > 8
    )
    if not nonblank:
        raise WitnessError(
            "validate_witness",
            f"rendered witness is blank or near-blank; faces_drawn={faces_drawn}, pixel_std={pixel_std:.4f}",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)
    return {
        "nonblank": nonblank,
        "pixel_std": pixel_std,
        "size": [output.width, output.height],
        "panels": list(PANELS),
        "panel_reports": panel_reports,
        "faces_drawn": faces_drawn,
        "color_route": color_route,
    }


def _success_report(
    *,
    mesh: trimesh.Trimesh,
    witness: dict[str, Any],
    input_path: Path,
    output_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    extents = np.asarray(mesh.extents, dtype=np.float64)
    return {
        "status": "ok",
        "route": ROUTE,
        "phase": "complete",
        "input_glb": str(input_path),
        "output_png": str(output_path),
        "report_json": str(report_path),
        "mesh": {
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "bounds_min": bounds[0].tolist() if bounds.shape == (2, 3) else None,
            "bounds_max": bounds[1].tolist() if bounds.shape == (2, 3) else None,
            "extents": extents.tolist(),
            "is_watertight": bool(mesh.is_watertight),
        },
        "witness": witness,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input GLB path.")
    parser.add_argument("--output", required=True, type=Path, help="Output PNG witness path.")
    parser.add_argument("--report", required=True, type=Path, help="Output JSON report path.")
    parser.add_argument("--size", type=int, default=720, help="Per-panel square size in pixels.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.size < 128:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        _write_json(
            args.report,
            _failure_report(
                phase="parse_args",
                error="--size must be at least 128",
                input_path=args.input,
                output_path=args.output,
                report_path=args.report,
            ),
        )
        return 2

    try:
        mesh = _load_mesh(args.input)
        _validate_mesh(mesh)
        witness = _render_witness(mesh, args.output, args.size)
        _write_json(
            args.report,
            _success_report(
                mesh=mesh,
                witness=witness,
                input_path=args.input,
                output_path=args.output,
                report_path=args.report,
            ),
        )
    except WitnessError as exc:
        if args.output.exists():
            args.output.unlink()
        _write_json(
            args.report,
            _failure_report(
                phase=exc.phase,
                error=str(exc),
                input_path=args.input,
                output_path=args.output,
                report_path=args.report,
            ),
        )
        print(f"{exc.phase}: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive durable failure report
        if args.output.exists():
            args.output.unlink()
        phase = "unexpected"
        _write_json(
            args.report,
            _failure_report(
                phase=phase,
                error=f"{type(exc).__name__}: {exc}",
                input_path=args.input,
                output_path=args.output,
                report_path=args.report,
            ),
        )
        print(f"{phase}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"wrote witness: {args.output}")
    print(f"wrote report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
