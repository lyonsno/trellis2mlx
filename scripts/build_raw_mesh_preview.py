"""Build a deterministic face-stride GLB preview of a raw extraction PLY.

The preview exists only to make very large raw meshes cheap enough to inspect.
The JSON report preserves the full source identity and labels every inference
that the derived GLB cannot support.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.postprocess_raw_cuda_mesh import export_glb, sha256_file, write_binary_ply


ROUTE = "deterministic_face_stride_preview"
FACE_DTYPE = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])
FORBIDDEN_INFERENCES = [
    "preview is not the complete raw mesh",
    "preview topology metrics are not full-mesh topology evidence",
    "preview is not cleanup, hole-fill, UV, texture, or final-GLB evidence",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", required=True, type=Path)
    parser.add_argument("--output-glb", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--face-stride", required=True, type=int)
    return parser


def build_raw_mesh_preview(
    *,
    input_ply: Path,
    output_glb: Path,
    report_json: Path,
    face_stride: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    input_ply = Path(input_ply)
    output_glb = Path(output_glb)
    report_json = Path(report_json)
    path_collisions = find_path_collisions(input_ply, output_glb, report_json)
    effective_report_json = report_json
    if any("report_json" in collision for collision in path_collisions):
        effective_report_json = choose_failure_report_path(
            report_json,
            protected_paths=[input_ply, output_glb, report_json],
        )
    report: dict[str, Any] = {
        "schema": "trellis2mlx.raw_mesh_face_stride_preview.v1",
        "status": "failed",
        "phase": None,
        "route": ROUTE,
        "input_ply": str(input_ply),
        "output_glb": str(output_glb),
        "requested_report_json": str(report_json),
        "effective_report_json": str(effective_report_json),
        "path_collisions": path_collisions,
        "requested_face_stride": int(face_stride),
        "effective_face_stride": None,
        "forbidden_inferences": FORBIDDEN_INFERENCES,
        "last_trustworthy_evidence": {},
    }
    phase = "validate_request"
    try:
        if path_collisions:
            raise ValueError(f"requested paths must be distinct: {path_collisions}")
        if face_stride <= 0:
            raise ValueError("--face-stride must be positive")
        output_glb.unlink(missing_ok=True)

        phase = "read_source_identity"
        input_sha256 = sha256_file(input_ply)
        report["input_ply_sha256"] = input_sha256
        report["last_trustworthy_evidence"] = {
            "input_ply_sha256": input_sha256,
            "input_ply_size_bytes": input_ply.stat().st_size,
        }

        phase = "load_source_mesh"
        vertices, face_records, source_layout = map_binary_ply(input_ply)
        report["source_read_mode"] = "numpy_memmap"
        report["source_mapped_bytes"] = int(source_layout["mapped_bytes"])

        phase = "validate_source_mesh"
        if vertices.shape[0] == 0:
            raise ValueError("source mesh has no vertices")
        if face_records.shape[0] == 0:
            raise ValueError("source mesh has no triangular faces")
        validate_face_records(face_records, vertices.shape[0])
        source_summary = mesh_summary_from_layout(vertices, face_records.shape[0])
        report["source_mesh"] = source_summary
        report["last_trustworthy_evidence"]["source_mesh"] = source_summary

        phase = "subsample_faces"
        sampled_faces = np.asarray(face_records["indices"][::face_stride], dtype=np.int32).copy()
        used_vertices, inverse = np.unique(sampled_faces.reshape(-1), return_inverse=True)
        preview_vertices = np.asarray(vertices[used_vertices], dtype=np.float32)
        preview_faces = np.asarray(inverse.reshape(-1, 3), dtype=np.int32)
        preview_summary = mesh_summary(preview_vertices, preview_faces)
        report["effective_face_stride"] = int(face_stride)
        report["preview_mesh"] = preview_summary
        report["preview_materialized_bytes"] = int(preview_vertices.nbytes + preview_faces.nbytes)

        phase = "write_preview_glb"
        output_glb.parent.mkdir(parents=True, exist_ok=True)
        export_glb(output_glb, preview_vertices, preview_faces)

        phase = "validate_preview_glb"
        output_size = output_glb.stat().st_size
        if output_size <= 0:
            raise ValueError("preview GLB is blank")
        loaded = trimesh.load(output_glb, force="mesh", process=False)
        if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
            raise ValueError("preview GLB reload produced blank geometry")
        if len(loaded.vertices) != preview_vertices.shape[0] or len(loaded.faces) != preview_faces.shape[0]:
            raise ValueError("preview GLB reload changed mesh counts")

        report.update(
            {
                "status": "done",
                "phase": "done",
                "elapsed_seconds": elapsed(started),
                "output_glb_sha256": sha256_file(output_glb),
                "output_glb_size_bytes": output_size,
            }
        )
        write_report(effective_report_json, report)
        return report
    except Exception as exc:
        if phase in {"write_preview_glb", "validate_preview_glb"} and output_glb.exists():
            invalid_output: dict[str, Any] = {"size_bytes": output_glb.stat().st_size}
            try:
                invalid_output["sha256"] = sha256_file(output_glb)
            except Exception as digest_exc:
                invalid_output["sha256_error"] = f"{type(digest_exc).__name__}: {digest_exc}"
            report["invalid_output_observed"] = invalid_output
            try:
                output_glb.unlink()
                report["invalid_output_removed"] = True
            except Exception as cleanup_exc:
                report["invalid_output_removed"] = False
                report["invalid_output_cleanup_error"] = (
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
        report.update(
            {
                "status": "failed",
                "phase": phase,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": elapsed(started),
            }
        )
        write_report(effective_report_json, report)
        raise


def mesh_summary(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    vertices = np.asarray(vertices)
    return {
        "vertices": int(vertices.shape[0]),
        "faces": int(np.asarray(faces).shape[0]),
        "bounds_min": [float(value) for value in vertices.min(axis=0)],
        "bounds_max": [float(value) for value in vertices.max(axis=0)],
    }


def mesh_summary_from_layout(vertices: np.ndarray, face_count: int) -> dict[str, Any]:
    bounds_min, bounds_max = chunked_vertex_bounds(vertices)
    return {
        "vertices": int(vertices.shape[0]),
        "faces": int(face_count),
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
    }


def map_binary_ply(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    with Path(path).open("rb") as handle:
        header_lines: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("PLY ended before end_header")
            decoded = line.decode("ascii").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break
        if "format binary_little_endian 1.0" not in header_lines:
            raise ValueError("only binary_little_endian PLY is supported")
        vertex_count = header_count(header_lines, "vertex")
        face_count = header_count(header_lines, "face")
        vertex_offset = handle.tell()

    vertex_bytes = vertex_count * 3 * np.dtype("<f4").itemsize
    face_bytes = face_count * FACE_DTYPE.itemsize
    face_offset = vertex_offset + vertex_bytes
    required_size = face_offset + face_bytes
    actual_size = Path(path).stat().st_size
    if actual_size < required_size:
        raise ValueError(
            f"PLY payload is truncated: expected at least {required_size} bytes, got {actual_size}"
        )
    vertices: np.ndarray
    face_records: np.ndarray
    if vertex_count:
        vertices = np.memmap(
            path,
            dtype="<f4",
            mode="r",
            offset=vertex_offset,
            shape=(vertex_count, 3),
        )
    else:
        vertices = np.empty((0, 3), dtype=np.float32)
    if face_count:
        face_records = np.memmap(
            path,
            dtype=FACE_DTYPE,
            mode="r",
            offset=face_offset,
            shape=(face_count,),
        )
    else:
        face_records = np.empty((0,), dtype=FACE_DTYPE)
    return vertices, face_records, {
        "vertex_offset": int(vertex_offset),
        "face_offset": int(face_offset),
        "mapped_bytes": int(vertex_bytes + face_bytes),
    }


def validate_face_records(
    face_records: np.ndarray,
    vertex_count: int,
    *,
    chunk_size: int = 1_000_000,
) -> None:
    for start in range(0, face_records.shape[0], chunk_size):
        chunk = face_records[start : start + chunk_size]
        if not np.all(chunk["count"] == 3):
            raise ValueError("only triangular PLY faces are supported")
        indices = chunk["indices"]
        if indices.min() < 0 or indices.max() >= vertex_count:
            raise ValueError("PLY face index is outside the vertex range")


def chunked_vertex_bounds(
    vertices: np.ndarray,
    *,
    chunk_size: int = 1_000_000,
) -> tuple[list[float], list[float]]:
    bounds_min = np.full(3, np.inf, dtype=np.float64)
    bounds_max = np.full(3, -np.inf, dtype=np.float64)
    for start in range(0, vertices.shape[0], chunk_size):
        chunk = vertices[start : start + chunk_size]
        bounds_min = np.minimum(bounds_min, chunk.min(axis=0))
        bounds_max = np.maximum(bounds_max, chunk.max(axis=0))
    return (
        [float(value) for value in bounds_min],
        [float(value) for value in bounds_max],
    )


def header_count(header_lines: list[str], element: str) -> int:
    prefix = f"element {element} "
    for line in header_lines:
        if line.startswith(prefix):
            return int(line.split()[-1])
    raise ValueError(f"missing PLY element count for {element}")


def find_path_collisions(input_ply: Path, output_glb: Path, report_json: Path) -> list[str]:
    named_paths = [
        ("input_ply", Path(input_ply)),
        ("output_glb", Path(output_glb)),
        ("report_json", Path(report_json)),
    ]
    collisions: list[str] = []
    for index, (left_name, left_path) in enumerate(named_paths):
        for right_name, right_path in named_paths[index + 1 :]:
            if paths_alias(left_path, right_path):
                collisions.append(f"{left_name}={right_name}")
    return collisions


def paths_alias(left: Path, right: Path) -> bool:
    if left.resolve(strict=False) == right.resolve(strict=False):
        return True
    try:
        return left.samefile(right)
    except (FileNotFoundError, OSError):
        return False


def choose_failure_report_path(requested: Path, *, protected_paths: list[Path]) -> Path:
    candidate = requested.with_name(requested.name + ".failure.json")
    suffix = 1
    while any(paths_alias(candidate, protected) for protected in protected_paths):
        candidate = requested.with_name(requested.name + f".failure.{suffix}.json")
        suffix += 1
    return candidate


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def elapsed(started: float) -> float:
    return max(0.0, time.perf_counter() - started)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        build_raw_mesh_preview(
            input_ply=args.input_ply,
            output_glb=args.output_glb,
            report_json=args.report_json,
            face_stride=args.face_stride,
        )
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
