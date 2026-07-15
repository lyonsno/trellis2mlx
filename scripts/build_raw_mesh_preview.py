"""Build a deterministic face-stride GLB preview of a raw extraction PLY.

The preview exists only to make very large raw meshes cheap enough to inspect.
The JSON report preserves the full source identity and labels every inference
that the derived GLB cannot support.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import trimesh

from scripts.postprocess_raw_cuda_mesh import (
    export_glb,
    read_binary_ply,
    sha256_file,
    write_binary_ply,
)


ROUTE = "deterministic_face_stride_preview"
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
    report: dict[str, Any] = {
        "schema": "trellis2mlx.raw_mesh_face_stride_preview.v1",
        "status": "failed",
        "phase": None,
        "route": ROUTE,
        "input_ply": str(input_ply),
        "output_glb": str(output_glb),
        "requested_face_stride": int(face_stride),
        "effective_face_stride": None,
        "forbidden_inferences": FORBIDDEN_INFERENCES,
        "last_trustworthy_evidence": {},
    }
    phase = "validate_request"
    try:
        if input_ply == output_glb:
            raise ValueError("input PLY and output GLB must be different paths")
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
        vertices, faces = read_binary_ply(input_ply)

        phase = "validate_source_mesh"
        if vertices.shape[0] == 0:
            raise ValueError("source mesh has no vertices")
        if faces.shape[0] == 0:
            raise ValueError("source mesh has no triangular faces")
        source_summary = mesh_summary(vertices, faces)
        report["source_mesh"] = source_summary
        report["last_trustworthy_evidence"]["source_mesh"] = source_summary

        phase = "subsample_faces"
        sampled_faces = np.asarray(faces[::face_stride], dtype=np.int32)
        used_vertices, inverse = np.unique(sampled_faces.reshape(-1), return_inverse=True)
        preview_vertices = np.asarray(vertices[used_vertices], dtype=np.float32)
        preview_faces = np.asarray(inverse.reshape(-1, 3), dtype=np.int32)
        preview_summary = mesh_summary(preview_vertices, preview_faces)
        report["effective_face_stride"] = int(face_stride)
        report["preview_mesh"] = preview_summary

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


def mesh_summary(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    vertices = np.asarray(vertices)
    return {
        "vertices": int(vertices.shape[0]),
        "faces": int(np.asarray(faces).shape[0]),
        "bounds_min": [float(value) for value in vertices.min(axis=0)],
        "bounds_max": [float(value) for value in vertices.max(axis=0)],
    }


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
