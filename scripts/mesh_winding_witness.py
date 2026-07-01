#!/usr/bin/env python3
"""Write a stage-by-stage mesh winding witness report."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


SCHEMA = "trellis2mlx.mesh_winding_witness.v1"
ROUTE = "cpu_mesh_winding_witness"
CHECKPOINT_STAGES = ("mesh_raw", "mesh_clean", "mesh_uv")


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


def _face_normals_and_areas(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(faces) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    areas = lengths * 0.5
    unit = np.zeros_like(normals, dtype=np.float64)
    valid = lengths > 1e-12
    unit[valid] = normals[valid] / lengths[valid, None]
    return unit, areas


def _orientation_summary(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    normals, areas = _face_normals_and_areas(vertices, faces)
    if len(faces) == 0:
        return {
            "checked_faces": 0,
            "outward_faces": 0,
            "inward_faces": 0,
            "near_tangent_faces": 0,
            "degenerate_faces": 0,
            "inward_ratio": 0.0,
        }

    tri = vertices[faces]
    centers = tri.mean(axis=1)
    mesh_center = vertices.mean(axis=0)
    radial = centers - mesh_center
    radial_norm = np.linalg.norm(radial, axis=1)
    usable = (areas > 1e-12) & (radial_norm > 1e-12)

    dots = np.zeros(len(faces), dtype=np.float64)
    dots[usable] = np.sum(normals[usable] * radial[usable], axis=1) / radial_norm[usable]
    outward = dots > 1e-8
    inward = dots < -1e-8
    near = usable & ~(outward | inward)
    checked = int(usable.sum())
    inward_faces = int((inward & usable).sum())
    return {
        "checked_faces": checked,
        "outward_faces": int((outward & usable).sum()),
        "inward_faces": inward_faces,
        "near_tangent_faces": int(near.sum()),
        "degenerate_faces": int((areas <= 1e-12).sum()),
        "inward_ratio": float(inward_faces / checked) if checked else 0.0,
    }


def _edge_consistency_summary(faces: np.ndarray) -> dict[str, Any]:
    undirected_edges: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for face_index, face in enumerate(np.asarray(faces, dtype=np.int64)):
        for corner in range(3):
            a = int(face[corner])
            b = int(face[(corner + 1) % 3])
            undirected_edges[tuple(sorted((a, b)))].append((face_index, a, b))

    boundary_edges = 0
    manifold_edges = 0
    opposite_direction_edges = 0
    same_direction_conflict_edges = 0
    nonmanifold_edges = 0
    duplicate_directed_nonmanifold_edges = 0
    conflict_examples = []

    for edge_faces in undirected_edges.values():
        if len(edge_faces) == 1:
            boundary_edges += 1
        elif len(edge_faces) == 2:
            manifold_edges += 1
            (_, a0, b0), (_, a1, b1) = edge_faces
            if a0 == a1 and b0 == b1:
                same_direction_conflict_edges += 1
                if len(conflict_examples) < 5:
                    conflict_examples.append(edge_faces)
            else:
                opposite_direction_edges += 1
        else:
            nonmanifold_edges += 1
            directed = {(a, b) for _, a, b in edge_faces}
            if len(directed) < len(edge_faces):
                duplicate_directed_nonmanifold_edges += 1

    return {
        "edges": int(len(undirected_edges)),
        "boundary_edges": int(boundary_edges),
        "manifold_edges": int(manifold_edges),
        "opposite_direction_edges": int(opposite_direction_edges),
        "same_direction_conflict_edges": int(same_direction_conflict_edges),
        "nonmanifold_edges": int(nonmanifold_edges),
        "duplicate_directed_nonmanifold_edges": int(duplicate_directed_nonmanifold_edges),
        "conflict_examples": conflict_examples,
    }


def _changed_face_rows(before: np.ndarray, after: np.ndarray) -> int | None:
    if before.shape != after.shape:
        return None
    return int((before != after).any(axis=1).sum())


def analyze_mesh(stage: str, vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    """Analyze winding/normal evidence for one mesh stage."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise WitnessError("analyze_mesh", f"{stage}: vertices must have shape [V, 3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise WitnessError("analyze_mesh", f"{stage}: faces must have shape [F, 3]")
    if len(vertices) == 0 or len(faces) == 0:
        raise WitnessError("analyze_mesh", f"{stage}: mesh has no vertices or no faces")
    if not np.isfinite(vertices).all():
        raise WitnessError("analyze_mesh", f"{stage}: mesh contains non-finite vertices")
    if faces.min(initial=0) < 0 or faces.max(initial=-1) >= len(vertices):
        raise WitnessError("analyze_mesh", f"{stage}: face index out of vertex range")

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    orientation = _orientation_summary(vertices, faces)

    fixed = mesh.copy()
    before_faces = np.asarray(fixed.faces).copy()
    trimesh.repair.fix_normals(fixed)
    after_faces = np.asarray(fixed.faces, dtype=np.int64)
    after_vertices = np.asarray(fixed.vertices, dtype=np.float64)

    return {
        "stage": stage,
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": int(mesh.euler_number),
        "body_count": int(getattr(mesh, "body_count", 0)),
        "volume": float(mesh.volume) if np.isfinite(mesh.volume) else None,
        "orientation": orientation,
        "edge_consistency": _edge_consistency_summary(faces),
        "fix_normals_counterfactual": {
            "changed_faces": _changed_face_rows(before_faces, after_faces),
            "face_count_after": int(len(after_faces)),
            "is_winding_consistent_after": bool(fixed.is_winding_consistent),
            "after_orientation": _orientation_summary(after_vertices, after_faces),
        },
    }


def _load_checkpoint_mesh(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        if "vertices" not in data or "faces" not in data:
            raise WitnessError("load_inputs", f"{path} must contain vertices and faces arrays")
        result = {key: data[key] for key in data.files}
    return result


def _load_glb_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise WitnessError("load_inputs", f"GLB does not exist: {path}")
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        loaded = trimesh.util.concatenate(meshes) if meshes else trimesh.Trimesh()
    return np.asarray(loaded.vertices), np.asarray(loaded.faces)


def _cyclic_orders(face: np.ndarray) -> set[tuple[int, int, int]]:
    a, b, c = [int(v) for v in face]
    return {(a, b, c), (b, c, a), (c, a, b)}


def _source_face_mapping(
    *,
    source_stage: str,
    source_faces: np.ndarray,
    uv_faces: np.ndarray,
    vmapping: np.ndarray,
) -> dict[str, Any]:
    source_by_key: dict[tuple[int, int, int], list[np.ndarray]] = {}
    for face in np.asarray(source_faces, dtype=np.int64):
        key = tuple(sorted(int(v) for v in face))
        source_by_key.setdefault(key, []).append(face)

    same = 0
    reversed_count = 0
    unmatched = 0
    ambiguous = 0
    mapped = np.asarray(vmapping, dtype=np.int64)[np.asarray(uv_faces, dtype=np.int64)]
    for face in mapped:
        key = tuple(sorted(int(v) for v in face))
        candidates = source_by_key.get(key, [])
        if not candidates:
            unmatched += 1
            continue
        same_match = any(tuple(int(v) for v in face) in _cyclic_orders(src) for src in candidates)
        reversed_match = any(
            tuple(int(v) for v in face[::-1]) in _cyclic_orders(src) for src in candidates
        )
        if same_match and not reversed_match:
            same += 1
        elif reversed_match and not same_match:
            reversed_count += 1
        elif same_match and reversed_match:
            ambiguous += 1
        else:
            unmatched += 1

    return {
        "source_stage": source_stage,
        "mapped_faces": int(len(mapped)),
        "same_orientation_faces": int(same),
        "reversed_orientation_faces": int(reversed_count),
        "unmatched_faces": int(unmatched),
        "ambiguous_faces": int(ambiguous),
    }


def build_report(*, checkpoint_dir: Path | None, glb: Path | None, report_path: Path) -> dict[str, Any]:
    stages: dict[str, dict[str, Any]] = {}
    stage_arrays: dict[str, dict[str, np.ndarray]] = {}

    if checkpoint_dir is not None:
        for stage in CHECKPOINT_STAGES:
            path = checkpoint_dir / f"{stage}.npz"
            if not path.exists():
                continue
            arrays = _load_checkpoint_mesh(path)
            stage_arrays[stage] = arrays
            stages[stage] = analyze_mesh(stage, arrays["vertices"], arrays["faces"])
            stages[stage]["path"] = str(path)

    if glb is not None:
        vertices, faces = _load_glb_mesh(glb)
        stages["export_glb"] = analyze_mesh("export_glb", vertices, faces)
        stages["export_glb"]["path"] = str(glb)

    if "mesh_uv" in stage_arrays and "vmapping" in stage_arrays["mesh_uv"]:
        source_stage = "mesh_clean" if "mesh_clean" in stage_arrays else "mesh_raw"
        if source_stage in stage_arrays:
            stages["mesh_uv"]["source_face_mapping"] = _source_face_mapping(
                source_stage=source_stage,
                source_faces=stage_arrays[source_stage]["faces"],
                uv_faces=stage_arrays["mesh_uv"]["faces"],
                vmapping=stage_arrays["mesh_uv"]["vmapping"],
            )

    if not stages:
        raise WitnessError("load_inputs", "no mesh stages found; pass --checkpoint-dir and/or --glb")

    return {
        "schema": SCHEMA,
        "status": "ok",
        "route": ROUTE,
        "evidence_use_class": "diagnostic_winding_stage_witness",
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        "glb": str(glb) if glb is not None else None,
        "report_json": str(report_path),
        "stages": stages,
    }


def _failure_report(
    *,
    phase: str,
    error: str,
    checkpoint_dir: Path | None,
    glb: Path | None,
    report_path: Path,
    loaded_stages: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "error",
        "route": ROUTE,
        "phase": phase,
        "error": error,
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        "glb": str(glb) if glb is not None else None,
        "report_json": str(report_path),
        "last_trustworthy_evidence": {
            "checkpoint_dir_exists": checkpoint_dir.exists() if checkpoint_dir is not None else None,
            "glb_exists": glb.exists() if glb is not None else None,
            "loaded_stages": loaded_stages or [],
        },
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, help="Directory containing mesh_*.npz checkpoints.")
    parser.add_argument("--glb", type=Path, help="Exported GLB to analyze.")
    parser.add_argument("--report", required=True, type=Path, help="Output JSON report path.")
    args = parser.parse_args(argv)
    if args.checkpoint_dir is None and args.glb is None:
        parser.error("at least one of --checkpoint-dir or --glb is required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    loaded: list[str] = []
    try:
        report = build_report(checkpoint_dir=args.checkpoint_dir, glb=args.glb, report_path=args.report)
        loaded = list(report["stages"].keys())
        _write_json(args.report, report)
    except WitnessError as exc:
        _write_json(
            args.report,
            _failure_report(
                phase=exc.phase,
                error=str(exc),
                checkpoint_dir=args.checkpoint_dir,
                glb=args.glb,
                report_path=args.report,
                loaded_stages=loaded,
            ),
        )
        print(f"{exc.phase}: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive durable failure report
        _write_json(
            args.report,
            _failure_report(
                phase="unexpected",
                error=str(exc),
                checkpoint_dir=args.checkpoint_dir,
                glb=args.glb,
                report_path=args.report,
                loaded_stages=loaded,
            ),
        )
        print(f"unexpected: {exc}", file=sys.stderr)
        return 1

    print(f"wrote report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
