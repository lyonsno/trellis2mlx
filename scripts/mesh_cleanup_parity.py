#!/usr/bin/env python
"""Write a reference-cleanup parity report for a fixture or raw mesh NPZ."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fixture", choices=["two-triangle-sheet"])
    source.add_argument("--raw-mesh", type=Path, help="NPZ containing vertices and faces arrays")
    parser.add_argument("--target-faces", type=int, default=200_000)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--overwrite-report", action="store_true")
    return parser.parse_args(argv)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n")


def _ensure_report_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"report already exists: {path}; pass --overwrite-report to replace it")


def _fixture_two_triangle_sheet() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    return vertices, faces


def _load_mesh(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, Any], str]:
    if args.fixture:
        vertices, faces = _fixture_two_triangle_sheet()
        return vertices, faces, {"route": "fixture", "name": args.fixture}, f"fixture:{args.fixture}"

    if not args.raw_mesh.exists():
        raise FileNotFoundError(f"raw mesh does not exist: {args.raw_mesh}")
    data = np.load(args.raw_mesh)
    if "vertices" not in data or "faces" not in data:
        raise ValueError(f"raw mesh NPZ must contain vertices and faces arrays: {args.raw_mesh}")
    return (
        np.asarray(data["vertices"]),
        np.asarray(data["faces"]),
        {"route": "raw-mesh", "path": str(args.raw_mesh)},
        "raw-mesh",
    )


def _probe_reference_backend() -> dict[str, Any]:
    try:
        import cumesh  # type: ignore  # noqa: F401
    except Exception as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    return {"status": "available", "module": "cumesh"}


def _fixture_simplify(vertices, faces, target_reduction=None, target_count=None):
    if target_count is None:
        return vertices, faces
    return vertices, faces[:target_count]


def _fixture_cleanup(vertices, faces, **kwargs):
    return vertices, faces


def _run_local_reference_cleanup(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    target_faces: int,
    fixture_mode: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    from generate import _cleanup_and_simplify_mesh

    operation_trace: list[dict[str, Any]] = []
    kwargs: dict[str, Any] = {}
    if fixture_mode:
        from trellmlx.mesh_cleanup import orient_faces_by_adjacency

        kwargs.update(
            cleanup_mesh=_fixture_cleanup,
            simplify=_fixture_simplify,
            orient_faces_by_adjacency=orient_faces_by_adjacency,
        )

    output_vertices, output_faces = _cleanup_and_simplify_mesh(
        vertices,
        faces,
        target_faces=target_faces,
        no_cleanup=False,
        reference_cleanup=True,
        operation_trace=operation_trace,
        log=lambda *args, **kwargs: None,
        **kwargs,
    )
    return output_vertices, output_faces, operation_trace


def _failure_report(
    *,
    requested_route: str,
    report_path: Path,
    phase: str,
    error: str,
) -> dict[str, Any]:
    return {
        "schema": "trellis2mlx.mesh_cleanup_parity_report.v1",
        "status": "failed",
        "requested_route": requested_route,
        "effective_route": "none",
        "failure_phase": phase,
        "error": error,
        "report_path": str(report_path),
        "last_trustworthy_evidence": {"report_written": True},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    from trellmlx.mesh_cleanup_parity import build_mesh_cleanup_parity_report

    vertices, faces, asset, requested_route = _load_mesh(args)
    t0 = time.perf_counter()
    output_vertices, output_faces, operation_trace = _run_local_reference_cleanup(
        vertices,
        faces,
        target_faces=args.target_faces,
        fixture_mode=asset["route"] == "fixture",
    )
    report = build_mesh_cleanup_parity_report(
        requested_route=requested_route,
        effective_route="local-reference-cleanup",
        input_vertices=vertices,
        input_faces=faces,
        output_vertices=output_vertices,
        output_faces=output_faces,
        operation_trace=operation_trace,
        reference_backend=_probe_reference_backend(),
    )
    report.update({
        "status": "ok",
        "asset": asset,
        "settings": {"target_faces": args.target_faces},
        "elapsed_seconds": time.perf_counter() - t0,
    })
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _ensure_report_writable(args.report, args.overwrite_report)
    requested_route = "raw-mesh" if args.raw_mesh else f"fixture:{args.fixture}"
    try:
        report = run(args)
    except Exception as exc:
        report = _failure_report(
            requested_route=requested_route,
            report_path=args.report,
            phase="load_mesh" if isinstance(exc, (FileNotFoundError, ValueError)) else "run_cleanup",
            error=f"{type(exc).__name__}: {exc}",
        )
        _write_json(args.report, report)
        return 1
    _write_json(args.report, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
