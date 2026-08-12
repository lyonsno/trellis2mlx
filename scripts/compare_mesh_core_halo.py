#!/usr/bin/env python3
"""Compare authenticated simplifier outputs inside a shared core AABB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np

from trellmlx.glb_aabb_crop import open_triangle_glb, sha256_file


ROUTE = "authenticated-glb-core-halo-comparison-v1"


class ComparisonError(RuntimeError):
    def __init__(self, phase: str, message: str):
        super().__init__(message)
        self.phase = phase


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh",
        action="append",
        required=True,
        nargs=3,
        metavar=("LABEL", "GLB", "EXPECTED_SHA256"),
    )
    parser.add_argument("--core-min", required=True, nargs=3, type=float)
    parser.add_argument("--core-max", required=True, nargs=3, type=float)
    parser.add_argument("--chunk-faces", type=int, default=250_000)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args(argv)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _mesh_requests(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {"label": label, "path": str(Path(path)), "expected_sha256": expected}
        for label, path, expected in args.mesh
    ]


def _effective_report(args: argparse.Namespace) -> tuple[Path, bool]:
    requested = args.report.resolve()
    inputs = {Path(path).resolve() for _label, path, _digest in args.mesh}
    if requested not in inputs:
        return args.report, False
    candidate = next(Path(path) for _label, path, _digest in args.mesh)
    candidate = candidate.with_name(candidate.name + ".comparison-error.json")
    while candidate.resolve() in inputs:
        candidate = candidate.with_name(candidate.name + ".comparison-error.json")
    return candidate, True


def _validate(args: argparse.Namespace, report_path: Path) -> None:
    labels = [label for label, _path, _digest in args.mesh]
    if len(labels) != len(set(labels)):
        raise ComparisonError("validate_request", "duplicate mesh labels are not allowed")
    if any(not label.strip() for label in labels):
        raise ComparisonError("validate_request", "mesh labels must be nonempty")
    if len({Path(path).resolve() for _label, path, _digest in args.mesh}) != len(args.mesh):
        raise ComparisonError("validate_request", "each mesh path must be unique")
    for _label, path, digest in args.mesh:
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ComparisonError("validate_request", "expected SHA256 must be lowercase hex")
        if report_path.resolve() == Path(path).resolve():
            raise ComparisonError("validate_paths", "effective report collides with an input")
        if args.report.resolve() == Path(path).resolve():
            raise ComparisonError("validate_paths", "requested report collides with an input")
    core_min = np.asarray(args.core_min, dtype=np.float64)
    core_max = np.asarray(args.core_max, dtype=np.float64)
    if not np.isfinite(core_min).all() or not np.isfinite(core_max).all():
        raise ComparisonError("validate_request", "core bounds must be finite")
    if np.any(core_min >= core_max):
        raise ComparisonError("validate_request", "each core minimum must be below its maximum")
    if args.chunk_faces <= 0:
        raise ComparisonError("validate_request", "chunk face count must be positive")


def _region_summary(
    *,
    faces: int,
    degenerate_faces: int,
    area: float,
    areas: list[np.ndarray],
    max_edges: list[np.ndarray],
    total_faces: int,
    total_area: float,
) -> dict[str, Any]:
    area_values = np.concatenate(areas) if areas else np.empty(0, dtype=np.float64)
    edge_values = np.concatenate(max_edges) if max_edges else np.empty(0, dtype=np.float64)

    def percentile(values: np.ndarray, quantile: float) -> float | None:
        return float(np.quantile(values, quantile)) if len(values) else None

    return {
        "faces": faces,
        "face_fraction": faces / max(total_faces, 1),
        "degenerate_faces": degenerate_faces,
        "degenerate_fraction": degenerate_faces / max(faces, 1),
        "area": area,
        "area_fraction": area / total_area if total_area > 0 else None,
        "median_face_area": percentile(area_values, 0.5),
        "p95_face_area": percentile(area_values, 0.95),
        "median_max_edge": percentile(edge_values, 0.5),
        "p95_max_edge": percentile(edge_values, 0.95),
    }


def _measure_mesh(
    path: Path,
    *,
    expected_sha256: str,
    core_min: np.ndarray,
    core_max: np.ndarray,
    chunk_faces: int,
) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ComparisonError(
            "authenticate_inputs",
            f"SHA256 mismatch for {path}: expected {expected_sha256}, found {observed}",
        )

    stats = {
        "core": {"faces": 0, "degenerate": 0, "area": 0.0, "areas": [], "edges": []},
        "halo": {"faces": 0, "degenerate": 0, "area": 0.0, "areas": [], "edges": []},
    }
    with open_triangle_glb(path) as view:
        total_faces = int(len(view.faces))
        total_vertices = int(len(view.vertices))
        for start in range(0, total_faces, chunk_faces):
            stop = min(start + chunk_faces, total_faces)
            triangles = np.asarray(view.vertices[view.faces[start:stop]], dtype=np.float64)
            edge_01 = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
            edge_12 = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
            edge_20 = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)
            max_edge = np.maximum(np.maximum(edge_01, edge_12), edge_20)
            area = 0.5 * np.linalg.norm(
                np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
                axis=1,
            )
            centroids = triangles.mean(axis=1)
            core = np.logical_and(centroids >= core_min, centroids <= core_max).all(axis=1)
            for key, mask in (("core", core), ("halo", ~core)):
                selected_area = area[mask]
                selected_edges = max_edge[mask]
                current = stats[key]
                current["faces"] += int(mask.sum())
                current["degenerate"] += int(np.count_nonzero(selected_area == 0.0))
                current["area"] += float(selected_area.sum(dtype=np.float64))
                current["areas"].append(selected_area)
                current["edges"].append(selected_edges)

    if sha256_file(path) != observed:
        raise ComparisonError("authenticate_inputs", f"input changed during measurement: {path}")
    total_area = stats["core"]["area"] + stats["halo"]["area"]
    degenerate_faces = stats["core"]["degenerate"] + stats["halo"]["degenerate"]
    return {
        "path": str(path),
        "sha256": observed,
        "expected_sha256": expected_sha256,
        "identity_match": True,
        "vertices": total_vertices,
        "faces": total_faces,
        "degenerate_faces": degenerate_faces,
        "degenerate_fraction": degenerate_faces / max(total_faces, 1),
        "total_area": total_area,
        "core_centroid": _region_summary(
            faces=stats["core"]["faces"],
            degenerate_faces=stats["core"]["degenerate"],
            area=stats["core"]["area"],
            areas=stats["core"]["areas"],
            max_edges=stats["core"]["edges"],
            total_faces=total_faces,
            total_area=total_area,
        ),
        "halo_centroid": _region_summary(
            faces=stats["halo"]["faces"],
            degenerate_faces=stats["halo"]["degenerate"],
            area=stats["halo"]["area"],
            areas=stats["halo"]["areas"],
            max_edges=stats["halo"]["edges"],
            total_faces=total_faces,
            total_area=total_area,
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    effective_report, rerouted = _effective_report(args)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "status": "failed",
        "failure_phase": "validate_paths",
        "route": ROUTE,
        "request": {
            "meshes": _mesh_requests(args),
            "core_min": args.core_min,
            "core_max": args.core_max,
            "chunk_faces": args.chunk_faces,
            "report": str(args.report),
        },
        "report": {
            "requested_path": str(args.report),
            "effective_path": str(effective_report),
            "rerouted": rerouted,
        },
        "effective_config": {
            "classification": "triangle-centroid-in-core-aabb",
            "core_min": args.core_min,
            "core_max": args.core_max,
            "chunk_faces": args.chunk_faces,
        },
        "meshes": None,
        "primary_output_status": "not_started",
        "last_trustworthy_evidence": {"inputs_preserved": True},
        "elapsed_seconds": None,
    }
    try:
        _validate(args, effective_report)
        _write_json(effective_report, report)
        report["failure_phase"] = "authenticate_inputs"
        identities: dict[str, tuple[Path, str]] = {}
        for label, path_text, expected in args.mesh:
            path = Path(path_text)
            if not path.is_file():
                raise ComparisonError("authenticate_inputs", f"mesh does not exist: {path}")
            observed = sha256_file(path)
            if observed != expected:
                raise ComparisonError(
                    "authenticate_inputs",
                    f"SHA256 mismatch for {label}: expected {expected}, found {observed}",
                )
            identities[label] = (path, observed)
        report["last_trustworthy_evidence"] = {
            "inputs_preserved": True,
            "authenticated_labels": list(identities),
        }
        _write_json(effective_report, report)

        report["failure_phase"] = "measure_meshes"
        core_min = np.asarray(args.core_min, dtype=np.float64)
        core_max = np.asarray(args.core_max, dtype=np.float64)
        meshes = {
            label: _measure_mesh(
                path,
                expected_sha256=digest,
                core_min=core_min,
                core_max=core_max,
                chunk_faces=args.chunk_faces,
            )
            for label, (path, digest) in identities.items()
        }
        report.update(
            status="completed",
            failure_phase=None,
            meshes=meshes,
            primary_output_status="validated",
            last_trustworthy_evidence={
                "inputs_preserved": True,
                "authenticated_labels": list(identities),
                "measured_labels": list(meshes),
            },
        )
    except BaseException as exc:
        report["status"] = "failed"
        report["failure_phase"] = getattr(exc, "phase", report["failure_phase"])
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        _write_json(effective_report, report)
    if report["status"] != "completed":
        print(f"{report['failure_phase']}: {report['error']}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
