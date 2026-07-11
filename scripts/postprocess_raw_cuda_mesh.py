"""Geometry-only cleanup for raw source-CUDA mesh extraction PLYs.

This is deliberately narrower than source ``o_voxel.postprocess.to_glb``. It
has no voxel attributes, no UV unwrap, and no texture bake. Its purpose is to
make a raw extraction visually inspectable without laundering it into final GLB
evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import trimesh

from trellmlx.source_mtlmesh import postprocess_source_native


FORBIDDEN_INFERENCES = [
    "not full source o_voxel.postprocess.to_glb output",
    "not texture bake evidence",
    "not final material evidence",
    "not proof of source final winding without operator inspection",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", required=True, type=Path)
    parser.add_argument("--output-glb", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--target-faces", required=True, type=int)
    parser.add_argument("--reference-python", type=Path)
    parser.add_argument("--expected-source-root", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser


def postprocess_raw_cuda_mesh(
    *,
    input_ply: Path,
    output_glb: Path,
    report_json: Path,
    target_faces: int,
    reference_python: str | Path | None = None,
    expected_source_root: str | Path | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema": "trellis2mlx.raw_cuda_mesh_geometry_postprocess.v1",
        "status": "failed",
        "phase": None,
        "artifact_scope": "geometry_only_raw_extraction_cleanup",
        "input_ply": str(input_ply),
        "output_glb": str(output_glb),
        "target_faces": int(target_faces),
        "reference_python": str(reference_python) if reference_python is not None else None,
        "expected_source_root": str(expected_source_root) if expected_source_root is not None else None,
        "forbidden_inferences": FORBIDDEN_INFERENCES,
    }
    phase = "load_ply"
    try:
        if target_faces <= 0:
            raise ValueError("--target-faces must be positive")
        vertices, faces = read_binary_ply(input_ply)
        report["input_mesh"] = mesh_summary(vertices, faces)

        phase = "source_native_geometry_postprocess"
        out_vertices, out_faces, trace = postprocess_source_native(
            vertices,
            faces,
            int(target_faces),
            verbose=verbose,
            reference_python=reference_python,
            expected_source_root=expected_source_root,
        )
        report["operation_trace"] = trace
        report["output_mesh"] = mesh_summary(out_vertices, out_faces)

        phase = "write_glb"
        output_glb.parent.mkdir(parents=True, exist_ok=True)
        export_glb(output_glb, out_vertices, out_faces)
        report.update(
            {
                "status": "done",
                "phase": "done",
                "elapsed_seconds": elapsed(started),
                "output_glb_sha256": sha256_file(output_glb),
                "output_glb_size_bytes": output_glb.stat().st_size,
            }
        )
        write_report(report_json, report)
        return report
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "phase": phase,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": elapsed(started),
            }
        )
        write_report(report_json, report)
        raise


def read_binary_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
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
        header = "\n".join(header_lines)
        if "format binary_little_endian 1.0" not in header:
            raise ValueError("only binary_little_endian PLY is supported")
        vertex_count = _header_count(header_lines, "vertex")
        face_count = _header_count(header_lines, "face")
        vertices_bytes = handle.read(vertex_count * 3 * 4)
        if len(vertices_bytes) != vertex_count * 3 * 4:
            raise ValueError("PLY ended before all vertices were read")
        vertices = np.frombuffer(vertices_bytes, dtype="<f4").reshape(vertex_count, 3).copy()
        face_dtype = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])
        face_bytes = handle.read(face_count * face_dtype.itemsize)
        if len(face_bytes) != face_count * face_dtype.itemsize:
            raise ValueError("PLY ended before all faces were read")
        face_records = np.frombuffer(face_bytes, dtype=face_dtype)
        if not np.all(face_records["count"] == 3):
            raise ValueError("only triangular PLY faces are supported")
        faces = np.asarray(face_records["indices"], dtype=np.int32).copy()
    return vertices, faces


def write_binary_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    vertices = np.asarray(vertices, dtype="<f4")
    faces = np.asarray(faces, dtype="<i4")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape [N, 3], got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape [F, 3], got {faces.shape}")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {vertices.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {faces.shape[0]}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    face_records = np.empty(
        faces.shape[0],
        dtype=np.dtype([("count", "u1"), ("indices", "<i4", (3,))]),
    )
    face_records["count"] = 3
    face_records["indices"] = faces
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(np.ascontiguousarray(vertices).tobytes())
        handle.write(face_records.tobytes())


def export_glb(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int32),
        process=False,
    )
    mesh.visual.face_colors = np.tile(np.array([[140, 145, 142, 255]], dtype=np.uint8), (len(mesh.faces), 1))
    mesh.export(path)


def mesh_summary(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    return {
        "vertices": int(np.asarray(vertices).shape[0]),
        "faces": int(np.asarray(faces).shape[0]),
    }


def _header_count(header_lines: list[str], element: str) -> int:
    prefix = f"element {element} "
    for line in header_lines:
        if line.startswith(prefix):
            return int(line.split()[-1])
    raise ValueError(f"missing PLY element count for {element}")


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def elapsed(started: float) -> float:
    return max(0.0, time.perf_counter() - started)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        postprocess_raw_cuda_mesh(
            input_ply=args.input_ply,
            output_glb=args.output_glb,
            report_json=args.report_json,
            target_faces=args.target_faces,
            reference_python=args.reference_python,
            expected_source_root=args.expected_source_root,
            verbose=not args.quiet,
        )
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
