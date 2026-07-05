#!/usr/bin/env python3
"""Probe winding evidence across mesh cleanup orientation stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np

from scripts.mesh_winding_witness import WitnessError, analyze_mesh
from trellmlx.mesh_cleanup import (
    fill_small_holes,
    orient_components_outward_by_radial_heuristic,
    orient_faces_by_adjacency,
    remove_duplicate_faces,
    remove_same_direction_manifold_conflicts,
    remove_small_components,
    repair_non_manifold_edges,
)


SCHEMA = "trellis2mlx.cleanup_orientation_probe.v1"
ROUTE = "cpu_cleanup_orientation_step_probe"


class ProbeError(RuntimeError):
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


def _load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise ProbeError("load_inputs", f"missing input mesh: {path}")
    try:
        with np.load(path) as data:
            if "vertices" not in data or "faces" not in data:
                raise ProbeError("load_inputs", f"{path} must contain vertices and faces arrays")
            vertices = np.asarray(data["vertices"])
            faces = np.asarray(data["faces"])
    except ProbeError:
        raise
    except Exception as exc:
        raise ProbeError("load_inputs", f"failed to load {path}: {exc}") from exc
    return vertices, faces


def _changed_face_rows(before: np.ndarray, after: np.ndarray) -> int | None:
    if before.shape != after.shape:
        return None
    return int((before != after).any(axis=1).sum())


def _save_stage(output_dir: Path, stage: str, vertices: np.ndarray, faces: np.ndarray) -> Path:
    path = output_dir / f"{stage}.npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, vertices=np.asarray(vertices), faces=np.asarray(faces))
    return path


def _record_stage(
    *,
    stages: dict[str, dict[str, Any]],
    output_dir: Path,
    stage: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    previous_faces: np.ndarray | None,
    include_visible: bool,
) -> None:
    path = _save_stage(output_dir, stage, vertices, faces)
    try:
        report = analyze_mesh(
            stage,
            vertices,
            faces,
            include_visible_exterior=include_visible,
        )
    except WitnessError as exc:
        raise ProbeError("analyze_stage", str(exc)) from exc
    report["path"] = str(path)
    report["changed_face_rows_from_previous"] = (
        None if previous_faces is None else _changed_face_rows(previous_faces, np.asarray(faces))
    )
    stages[stage] = report


CleanupStep = tuple[str, Callable[..., tuple[np.ndarray, np.ndarray]]]


def build_probe_report(
    *,
    mesh_path: Path,
    output_dir: Path,
    report_path: Path,
    include_visible: bool,
    max_hole_perimeter: float,
    min_component_ratio: float,
) -> dict[str, Any]:
    vertices, faces = _load_mesh(mesh_path)
    stages: dict[str, dict[str, Any]] = {}

    _record_stage(
        stages=stages,
        output_dir=output_dir,
        stage="input",
        vertices=vertices,
        faces=faces,
        previous_faces=None,
        include_visible=include_visible,
    )

    steps: list[tuple[str, Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]]] = [
        (
            "after_duplicate_face_removal",
            lambda v, f: remove_duplicate_faces(v, f, verbose=False),
        ),
        (
            "after_non_manifold_repair",
            lambda v, f: repair_non_manifold_edges(v, f, verbose=False),
        ),
        (
            "after_small_component_filter",
            lambda v, f: remove_small_components(
                v,
                f,
                min_ratio=min_component_ratio,
                verbose=False,
            ),
        ),
        (
            "after_small_hole_fill",
            lambda v, f: fill_small_holes(
                v,
                f,
                max_hole_perimeter=max_hole_perimeter,
                verbose=False,
            ),
        ),
        (
            "after_adjacency_orient",
            lambda v, f: orient_faces_by_adjacency(v, f, verbose=False),
        ),
        (
            "after_radial_component_orient",
            lambda v, f: orient_components_outward_by_radial_heuristic(v, f, verbose=False),
        ),
        (
            "after_residual_conflict_prune",
            lambda v, f: remove_same_direction_manifold_conflicts(v, f, verbose=False),
        ),
    ]

    for stage, step in steps:
        previous_faces = np.asarray(faces)
        try:
            vertices, faces = step(vertices, faces)
        except Exception as exc:
            raise ProbeError(stage, f"{stage} failed: {exc}") from exc
        _record_stage(
            stages=stages,
            output_dir=output_dir,
            stage=stage,
            vertices=vertices,
            faces=faces,
            previous_faces=previous_faces,
            include_visible=include_visible,
        )

    return {
        "schema": SCHEMA,
        "status": "ok",
        "route": ROUTE,
        "evidence_use_class": "diagnostic_cleanup_orientation_stage_probe",
        "input_mesh": str(mesh_path),
        "output_dir": str(output_dir),
        "report_json": str(report_path),
        "parameters": {
            "include_visible": bool(include_visible),
            "max_hole_perimeter": float(max_hole_perimeter),
            "min_component_ratio": float(min_component_ratio),
        },
        "stages": stages,
    }


def _failure_report(
    *,
    phase: str,
    error: str,
    mesh_path: Path,
    output_dir: Path,
    report_path: Path,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "error",
        "route": ROUTE,
        "phase": phase,
        "error": error,
        "input_mesh": str(mesh_path),
        "output_dir": str(output_dir),
        "report_json": str(report_path),
        "last_trustworthy_evidence": {
            "input_exists": mesh_path.exists(),
            "output_dir_exists": output_dir.exists(),
            "written_stage_files": sorted(path.name for path in output_dir.glob("*.npz"))
            if output_dir.exists()
            else [],
        },
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path, help="Input mesh npz with vertices/faces.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for stage npz outputs.")
    parser.add_argument("--report", required=True, type=Path, help="Output JSON report path.")
    parser.add_argument("--skip-visible", action="store_true", help="Skip raster visible-exterior metrics.")
    parser.add_argument("--max-hole-perimeter", type=float, default=3e-2)
    parser.add_argument("--min-component-ratio", type=float, default=1e-5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        report = build_probe_report(
            mesh_path=args.mesh,
            output_dir=args.output_dir,
            report_path=args.report,
            include_visible=not args.skip_visible,
            max_hole_perimeter=args.max_hole_perimeter,
            min_component_ratio=args.min_component_ratio,
        )
        _write_json(args.report, report)
    except ProbeError as exc:
        _write_json(
            args.report,
            _failure_report(
                phase=exc.phase,
                error=str(exc),
                mesh_path=args.mesh,
                output_dir=args.output_dir,
                report_path=args.report,
            ),
        )
        print(f"{exc.phase}: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - durable fallback
        _write_json(
            args.report,
            _failure_report(
                phase="unexpected",
                error=str(exc),
                mesh_path=args.mesh,
                output_dir=args.output_dir,
                report_path=args.report,
            ),
        )
        print(f"unexpected: {exc}", file=sys.stderr)
        return 1

    print(f"wrote report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
