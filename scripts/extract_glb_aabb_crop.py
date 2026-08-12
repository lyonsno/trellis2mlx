#!/usr/bin/env python3
"""Extract a deterministic core-plus-halo volume from a simple triangle GLB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from trellmlx.glb_aabb_crop import (
    CropError,
    ROUTE,
    open_triangle_glb,
    output_temporary_path,
    provenance_temporary_path,
    remove_partial_output_surface,
    remove_output_surface,
    select_aabb_crop,
    sha256_file,
    validate_request,
    validate_output_paths,
    write_geometry_glb,
    write_provenance,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--provenance-dir", required=True, type=Path)
    parser.add_argument("--core-min", required=True, nargs=3, type=float)
    parser.add_argument("--core-max", required=True, nargs=3, type=float)
    parser.add_argument("--halo-fraction", type=float, default=0.5)
    parser.add_argument("--chunk-faces", type=int, default=250_000)
    return parser.parse_args(argv)


def _request(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "input_glb": str(args.input),
        "output_glb": str(args.output),
        "report_json": str(args.report),
        "provenance_dir": str(args.provenance_dir),
        "core_min": list(args.core_min),
        "core_max": list(args.core_max),
        "halo_fraction": args.halo_fraction,
        "chunk_faces": args.chunk_faces,
    }


def _failure_report_path(args: argparse.Namespace) -> Path:
    """Keep validation errors from overwriting a protected request path."""
    source = args.input.resolve()
    output = args.output.resolve()
    output_temporary = output_temporary_path(args.output).resolve()
    provenance = args.provenance_dir.resolve()
    provenance_temporary = provenance_temporary_path(args.provenance_dir).resolve()

    def protected(candidate: Path) -> bool:
        resolved = candidate.resolve()
        return resolved in {
            source,
            output,
            output_temporary,
            provenance,
            provenance_temporary,
        }

    if not protected(args.report):
        return args.report

    candidate = args.input.with_name(args.input.name + ".crop-error.json")
    while protected(candidate):
        candidate = candidate.with_name(candidate.name + ".crop-error.json")
    return candidate


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    request = _request(args)
    report_path = _failure_report_path(args)
    report_identity = {
        "requested_path": str(args.report),
        "effective_path": str(report_path),
    }
    try:
        validate_output_paths(
            input_path=args.input,
            output_path=args.output,
            report_path=args.report,
            provenance_dir=args.provenance_dir,
        )
    except CropError as exc:
        input_exists = args.input.exists()
        _write_json(
            report_path,
            {
                "status": "error",
                "phase": exc.phase,
                "route": ROUTE,
                "error": str(exc),
                "request": request,
                "report": report_identity,
                "last_trustworthy_evidence": {
                    "input_exists": input_exists,
                    "input_size_bytes": args.input.stat().st_size if input_exists else None,
                    "source_preserved": input_exists,
                },
            },
        )
        print(f"{exc.phase}: {exc}", file=sys.stderr)
        return 1

    last_trustworthy: dict[str, Any] = {
        "input_exists": args.input.exists(),
        "input_size_bytes": args.input.stat().st_size if args.input.exists() else None,
        "preexisting_output_removed": False,
        "preexisting_provenance_removed": False,
    }
    started = time.perf_counter()

    try:
        cleanup = remove_output_surface(args.output, args.provenance_dir)
        last_trustworthy.update(cleanup)
        core_min = np.asarray(args.core_min, dtype=np.float64)
        core_max = np.asarray(args.core_max, dtype=np.float64)
        validate_request(core_min, core_max, args.halo_fraction, args.chunk_faces)
        source_sha256_before = sha256_file(args.input)
        with open_triangle_glb(args.input) as view:
            last_trustworthy.update(
                {
                    "source_faces": int(len(view.faces)),
                    "source_vertices": int(len(view.vertices)),
                    "position_accessor": view.position_accessor,
                    "index_accessor": view.index_accessor,
                }
            )
            selection = select_aabb_crop(
                view,
                core_min=core_min,
                core_max=core_max,
                halo_fraction=args.halo_fraction,
                chunk_faces=args.chunk_faces,
            )
        source_sha256_after = sha256_file(args.input)
        if source_sha256_after != source_sha256_before:
            raise CropError(
                "source_identity",
                "input GLB changed while the crop was being selected",
                {
                    "source_sha256_before": source_sha256_before,
                    "source_sha256_after": source_sha256_after,
                },
            )
        write_geometry_glb(args.output, selection.vertices, selection.faces)
        write_provenance(args.provenance_dir, selection)

        output_sha256 = sha256_file(args.output)
        payload = {
            "status": "ok",
            "phase": "complete",
            "route": ROUTE,
            "request": request,
            "report": report_identity,
            "effective_config": {
                "selection_rule": "triangle_bounds_overlap_outer_aabb",
                "chunk_faces": args.chunk_faces,
                "face_limit": None,
                "halo_fraction_per_core_span_per_side": args.halo_fraction,
            },
            "source": {
                "path": str(args.input),
                "bytes": args.input.stat().st_size,
                "sha256": source_sha256_after,
                "vertices": selection.source_vertex_count,
                "faces": selection.source_face_count,
                "identity_stable_during_selection": True,
            },
            "selection": {
                "rule": "triangle_bounds_overlap_outer_aabb",
                "core_min": selection.core_min.tolist(),
                "core_max": selection.core_max.tolist(),
                "outer_min": selection.outer_min.tolist(),
                "outer_max": selection.outer_max.tolist(),
                "source_faces": selection.source_face_count,
                "selected_faces": int(len(selection.faces)),
                "selected_vertices": int(len(selection.vertices)),
                "core_faces": int(selection.core_face_mask.sum()),
                "halo_only_faces": int((~selection.core_face_mask).sum()),
            },
            "output": {
                "path": str(args.output),
                "bytes": args.output.stat().st_size,
                "sha256": output_sha256,
                "provenance_dir": str(args.provenance_dir),
            },
            "timing_seconds": {"total": time.perf_counter() - started},
        }
        _write_json(report_path, payload)
    except CropError as exc:
        failure_cleanup = remove_partial_output_surface(args.output, args.provenance_dir)
        last_trustworthy.update(failure_cleanup)
        last_trustworthy.update(exc.evidence)
        _write_json(
            report_path,
            {
                "status": "error",
                "phase": exc.phase,
                "route": ROUTE,
                "error": str(exc),
                "request": request,
                "report": report_identity,
                "last_trustworthy_evidence": last_trustworthy,
                "timing_seconds": {"total": time.perf_counter() - started},
            },
        )
        print(f"{exc.phase}: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive durable failure path
        failure_cleanup = remove_partial_output_surface(args.output, args.provenance_dir)
        last_trustworthy.update(failure_cleanup)
        _write_json(
            report_path,
            {
                "status": "error",
                "phase": "unexpected",
                "route": ROUTE,
                "error": f"{type(exc).__name__}: {exc}",
                "request": request,
                "report": report_identity,
                "last_trustworthy_evidence": last_trustworthy,
                "timing_seconds": {"total": time.perf_counter() - started},
            },
        )
        print(f"unexpected: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"wrote crop: {args.output}")
    print(f"wrote provenance: {args.provenance_dir}")
    print(f"wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
