"""Compare hole-fill outputs while factoring out loop and face ordering."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


class StructureComparisonError(RuntimeError):
    pass


def compare_prefill_arrays(
    reference_vertices: np.ndarray,
    reference_faces: np.ndarray,
    candidate_vertices: np.ndarray,
    candidate_faces: np.ndarray,
    *,
    input_vertex_count: int,
    input_face_count: int,
) -> dict[str, Any]:
    reference_vertices = np.asarray(reference_vertices, dtype=np.float32)
    candidate_vertices = np.asarray(candidate_vertices, dtype=np.float32)
    reference_faces = np.asarray(reference_faces, dtype=np.int32)
    candidate_faces = np.asarray(candidate_faces, dtype=np.int32)
    _validate_mesh(reference_vertices, reference_faces, "reference")
    _validate_mesh(candidate_vertices, candidate_faces, "candidate")
    if input_vertex_count < 0 or input_face_count < 0:
        raise StructureComparisonError("input counts must be non-negative")
    if input_vertex_count > len(reference_vertices) or input_vertex_count > len(
        candidate_vertices
    ):
        raise StructureComparisonError("input vertex count exceeds a stage mesh")
    if input_face_count > len(reference_faces) or input_face_count > len(candidate_faces):
        raise StructureComparisonError("input face count exceeds a stage mesh")

    input_vertices_exact = bool(
        np.array_equal(
            reference_vertices[:input_vertex_count],
            candidate_vertices[:input_vertex_count],
        )
    )
    input_faces_exact = bool(
        np.array_equal(
            reference_faces[:input_face_count],
            candidate_faces[:input_face_count],
        )
    )
    if not input_vertices_exact:
        raise StructureComparisonError("input vertex prefix changed before appended geometry")
    if not input_faces_exact:
        raise StructureComparisonError("input face prefix changed before appended geometry")

    reference_new_vertices = reference_vertices[input_vertex_count:]
    candidate_new_vertices = candidate_vertices[input_vertex_count:]
    reference_new_faces = reference_faces[input_face_count:]
    candidate_new_faces = candidate_faces[input_face_count:]
    reference_edges, reference_centers = _edge_aligned_centers(
        reference_vertices,
        reference_new_faces,
        input_vertex_count=input_vertex_count,
        role="reference",
    )
    candidate_edges, candidate_centers = _edge_aligned_centers(
        candidate_vertices,
        candidate_new_faces,
        input_vertex_count=input_vertex_count,
        role="candidate",
    )
    boundary_edges_exact = bool(np.array_equal(reference_edges, candidate_edges))
    aligned_centers: dict[str, Any]
    if boundary_edges_exact:
        aligned_centers = {
            "comparable": True,
            "exact": bool(np.array_equal(reference_centers, candidate_centers)),
            "ordered_float_delta": _float_delta(reference_centers, candidate_centers),
            "float32_ulp": _float32_ulp_delta(reference_centers, candidate_centers),
        }
    else:
        aligned_centers = {
            "comparable": False,
            "exact": False,
            "ordered_float_delta": None,
            "float32_ulp": None,
        }

    cub_centers, cub_center_ids, cub_loop_sizes = reproduce_cub_hole_centers(
        reference_vertices[:input_vertex_count],
        reference_new_faces,
        input_vertex_count=input_vertex_count,
    )
    reference_center_vertices = reference_vertices[cub_center_ids]

    return {
        "schema": "trellis2mlx.prefill_stage_structure_comparison.v1",
        "status": "done",
        "input_prefix": {
            "vertex_count": int(input_vertex_count),
            "face_count": int(input_face_count),
            "vertices_exact": input_vertices_exact,
            "faces_exact": input_faces_exact,
        },
        "appended": {
            "reference_vertices": int(len(reference_new_vertices)),
            "candidate_vertices": int(len(candidate_new_vertices)),
            "reference_faces": int(len(reference_new_faces)),
            "candidate_faces": int(len(candidate_new_faces)),
            "vertices_ordered_exact": bool(
                np.array_equal(reference_new_vertices, candidate_new_vertices)
            ),
            "vertices_multiset_exact": _vertex_multiset_exact(
                reference_new_vertices,
                candidate_new_vertices,
            ),
            "faces_ordered_exact": bool(
                np.array_equal(reference_new_faces, candidate_new_faces)
            ),
        },
        "boundary_edges": {
            "reference_count": int(len(reference_edges)),
            "candidate_count": int(len(candidate_edges)),
            "multiset_exact": boundary_edges_exact,
        },
        "edge_aligned_centers": aligned_centers,
        "cub_center_reproduction": {
            "source": "CCCL 2.7.0 DeviceSegmentedReduce::Sum Vec3f",
            "center_count": int(len(cub_centers)),
            "loop_size": {
                "min": int(np.min(cub_loop_sizes)) if len(cub_loop_sizes) else 0,
                "max": int(np.max(cub_loop_sizes)) if len(cub_loop_sizes) else 0,
                "mean": float(np.mean(cub_loop_sizes)) if len(cub_loop_sizes) else 0.0,
            },
            "exact": bool(np.array_equal(reference_center_vertices, cub_centers)),
            "ordered_float_delta": _float_delta(
                reference_center_vertices,
                cub_centers,
            ),
            "float32_ulp": _float32_ulp_delta(
                reference_center_vertices,
                cub_centers,
            ),
        },
    }


def compare_prefill_files(
    reference_ply: Path,
    candidate_ply: Path,
    *,
    input_vertex_count: int,
    input_face_count: int,
    expected_reference_sha256: str,
    expected_candidate_sha256: str,
) -> dict[str, Any]:
    reference_ply = Path(reference_ply)
    candidate_ply = Path(candidate_ply)
    reference_sha256 = sha256_file(reference_ply)
    candidate_sha256 = sha256_file(candidate_ply)
    if reference_sha256 != expected_reference_sha256:
        raise StructureComparisonError(
            "reference PLY SHA256 mismatch: "
            f"expected {expected_reference_sha256}, got {reference_sha256}"
        )
    if candidate_sha256 != expected_candidate_sha256:
        raise StructureComparisonError(
            "candidate PLY SHA256 mismatch: "
            f"expected {expected_candidate_sha256}, got {candidate_sha256}"
        )
    reference_vertices, reference_faces = read_binary_ply(reference_ply)
    candidate_vertices, candidate_faces = read_binary_ply(candidate_ply)
    report = compare_prefill_arrays(
        reference_vertices,
        reference_faces,
        candidate_vertices,
        candidate_faces,
        input_vertex_count=input_vertex_count,
        input_face_count=input_face_count,
    )
    report["reference"] = {
        "path": str(reference_ply),
        "sha256": reference_sha256,
        "vertices": int(len(reference_vertices)),
        "faces": int(len(reference_faces)),
    }
    report["candidate"] = {
        "path": str(candidate_ply),
        "sha256": candidate_sha256,
        "vertices": int(len(candidate_vertices)),
        "faces": int(len(candidate_faces)),
    }
    return report


def _edge_aligned_centers(
    vertices: np.ndarray,
    appended_faces: np.ndarray,
    *,
    input_vertex_count: int,
    role: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not len(appended_faces):
        return np.empty((0, 2), dtype=np.int32), np.empty((0, 3), dtype=np.float32)
    new_mask = appended_faces >= input_vertex_count
    new_counts = np.sum(new_mask, axis=1)
    if not np.all(new_counts == 1):
        bad = int(np.flatnonzero(new_counts != 1)[0])
        raise StructureComparisonError(
            f"{role} appended face {bad} does not contain exactly one new center vertex"
        )
    center_ids = appended_faces[new_mask].reshape(-1)
    edge_ids = appended_faces[~new_mask].reshape(-1, 2)
    edge_ids.sort(axis=1)
    if np.any(center_ids >= len(vertices)):
        raise StructureComparisonError(f"{role} appended face references missing center vertex")
    packed_edges = (
        edge_ids[:, 0].astype(np.uint64)
        | (edge_ids[:, 1].astype(np.uint64) << np.uint64(32))
    )
    order = np.argsort(packed_edges, kind="stable")
    packed_sorted = packed_edges[order]
    if len(packed_sorted) > 1 and np.any(packed_sorted[1:] == packed_sorted[:-1]):
        raise StructureComparisonError(f"{role} appended boundary edges are not unique")
    return edge_ids[order], vertices[center_ids[order]]


def cub_vec3f_segmented_sum(values: np.ndarray) -> np.ndarray:
    """Emulate CCCL 2.7 AgentReduce for one 16-byte Vec3f segment."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3:
        raise StructureComparisonError(
            f"Vec3f segment must have shape [N, 3], got {values.shape}"
        )
    if not len(values):
        return np.zeros(3, dtype=np.float32)

    block_threads = 256
    items_per_thread = 4
    tile_items = block_threads * items_per_thread
    thread_aggregates = np.zeros((block_threads, 3), dtype=np.float32)
    initialized = np.zeros(block_threads, dtype=bool)
    tile_offset = 0

    while tile_offset + tile_items <= len(values):
        for thread in range(block_threads):
            for item in range(items_per_thread):
                value = values[tile_offset + thread + block_threads * item]
                if initialized[thread]:
                    thread_aggregates[thread] = np.add(
                        thread_aggregates[thread],
                        value,
                        dtype=np.float32,
                    )
                else:
                    thread_aggregates[thread] = value
                    initialized[thread] = True
        tile_offset += tile_items

    tail_items = len(values) - tile_offset
    for thread in range(min(block_threads, tail_items)):
        item = thread
        while item < tail_items:
            value = values[tile_offset + item]
            if initialized[thread]:
                thread_aggregates[thread] = np.add(
                    thread_aggregates[thread],
                    value,
                    dtype=np.float32,
                )
            else:
                thread_aggregates[thread] = value
                initialized[thread] = True
            item += block_threads

    valid_threads = min(block_threads, len(values))
    warp_aggregates = []
    for warp_start in range(0, valid_threads, 32):
        warp = thread_aggregates[warp_start : min(warp_start + 32, valid_threads)].copy()
        valid_lanes = len(warp)
        for offset in (1, 2, 4, 8, 16):
            if offset >= valid_lanes:
                break
            previous = warp.copy()
            warp[: valid_lanes - offset] = np.add(
                previous[: valid_lanes - offset],
                previous[offset:valid_lanes],
                dtype=np.float32,
            )
        warp_aggregates.append(warp[0])

    block_aggregate = warp_aggregates[0]
    for warp_aggregate in warp_aggregates[1:]:
        block_aggregate = np.add(
            block_aggregate,
            warp_aggregate,
            dtype=np.float32,
        )
    return np.add(
        np.zeros(3, dtype=np.float32),
        block_aggregate,
        dtype=np.float32,
    )


def reproduce_cub_hole_centers(
    input_vertices: np.ndarray,
    appended_faces: np.ndarray,
    *,
    input_vertex_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_vertices = np.asarray(input_vertices, dtype=np.float32)
    appended_faces = np.asarray(appended_faces, dtype=np.int32)
    if input_vertices.shape != (input_vertex_count, 3):
        raise StructureComparisonError(
            "input vertex prefix shape does not match input_vertex_count"
        )
    if appended_faces.ndim != 2 or appended_faces.shape[1] != 3:
        raise StructureComparisonError(
            f"appended faces must have shape [F, 3], got {appended_faces.shape}"
        )
    if not len(appended_faces):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )

    new_mask = appended_faces >= input_vertex_count
    if not np.all(np.sum(new_mask, axis=1) == 1):
        raise StructureComparisonError(
            "each appended face must contain exactly one center vertex"
        )
    center_ids_per_face = appended_faces[new_mask].reshape(-1)
    boundary_edges = appended_faces[~new_mask].reshape(-1, 2)
    if np.any(boundary_edges < 0) or np.any(boundary_edges >= input_vertex_count):
        raise StructureComparisonError(
            "appended fan faces reference a non-prefix boundary vertex"
        )

    center_ids, first_indices = np.unique(center_ids_per_face, return_index=True)
    order = np.argsort(first_indices, kind="stable")
    center_ids = center_ids[order].astype(np.int32, copy=False)
    centers = []
    loop_sizes = []
    for center_id in center_ids:
        face_indices = np.flatnonzero(center_ids_per_face == center_id)
        if len(face_indices) > 1 and np.any(np.diff(face_indices) != 1):
            raise StructureComparisonError(
                f"center vertex {center_id} fan faces are not contiguous"
            )
        edges = boundary_edges[face_indices]
        midpoints = np.multiply(
            np.add(
                input_vertices[edges[:, 0]],
                input_vertices[edges[:, 1]],
                dtype=np.float32,
            ),
            np.float32(0.5),
            dtype=np.float32,
        )
        center_sum = cub_vec3f_segmented_sum(midpoints)
        centers.append(
            np.divide(
                center_sum,
                np.float32(len(midpoints)),
                dtype=np.float32,
            )
        )
        loop_sizes.append(len(midpoints))
    return (
        np.asarray(centers, dtype=np.float32),
        center_ids,
        np.asarray(loop_sizes, dtype=np.int32),
    )


def _vertex_multiset_exact(reference: np.ndarray, candidate: np.ndarray) -> bool:
    if reference.shape != candidate.shape:
        return False
    if not len(reference):
        return True
    reference_bits = np.ascontiguousarray(reference).view(np.uint32).reshape(-1, 3)
    candidate_bits = np.ascontiguousarray(candidate).view(np.uint32).reshape(-1, 3)
    ref_order = np.lexsort(
        (reference_bits[:, 2], reference_bits[:, 1], reference_bits[:, 0])
    )
    cand_order = np.lexsort(
        (candidate_bits[:, 2], candidate_bits[:, 1], candidate_bits[:, 0])
    )
    return bool(np.array_equal(reference_bits[ref_order], candidate_bits[cand_order]))


def _float_delta(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {"shape_match": False}
    diff = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    return {
        "shape_match": True,
        "nonzero": int(np.count_nonzero(diff)),
        "max_abs": float(np.max(diff)) if diff.size else 0.0,
        "mean_abs": float(np.mean(diff)) if diff.size else 0.0,
    }


def _float32_ulp_delta(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {"shape_match": False}
    reference_ordered = _ordered_float32_bits(reference)
    candidate_ordered = _ordered_float32_bits(candidate)
    ulp = np.abs(
        reference_ordered.astype(np.int64) - candidate_ordered.astype(np.int64)
    )
    return {
        "shape_match": True,
        "nonzero": int(np.count_nonzero(ulp)),
        "max": int(np.max(ulp)) if ulp.size else 0,
        "mean": float(np.mean(ulp)) if ulp.size else 0.0,
    }


def _ordered_float32_bits(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    negative = (bits & np.uint32(0x80000000)) != 0
    return np.where(
        negative,
        ~bits,
        bits ^ np.uint32(0x80000000),
    ).astype(np.uint32)


def _validate_mesh(vertices: np.ndarray, faces: np.ndarray, role: str) -> None:
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise StructureComparisonError(
            f"{role} vertices must have shape [N, 3], got {vertices.shape}"
        )
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise StructureComparisonError(
            f"{role} faces must have shape [F, 3], got {faces.shape}"
        )
    if len(faces) and (np.min(faces) < 0 or np.max(faces) >= len(vertices)):
        raise StructureComparisonError(f"{role} faces contain invalid vertex indices")


def read_binary_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Path(path).open("rb") as handle:
        header_lines = []
        while True:
            line = handle.readline()
            if not line:
                raise StructureComparisonError("PLY ended before end_header")
            decoded = line.decode("ascii").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break
        if "format binary_little_endian 1.0" not in header_lines:
            raise StructureComparisonError("only binary_little_endian PLY is supported")
        vertex_count = _header_count(header_lines, "vertex")
        face_count = _header_count(header_lines, "face")
        vertex_bytes = handle.read(vertex_count * 3 * 4)
        if len(vertex_bytes) != vertex_count * 3 * 4:
            raise StructureComparisonError("PLY ended before all vertices were read")
        vertices = np.frombuffer(vertex_bytes, dtype="<f4").reshape(vertex_count, 3).copy()
        face_dtype = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])
        face_bytes = handle.read(face_count * face_dtype.itemsize)
        if len(face_bytes) != face_count * face_dtype.itemsize:
            raise StructureComparisonError("PLY ended before all faces were read")
        records = np.frombuffer(face_bytes, dtype=face_dtype)
        if not np.all(records["count"] == 3):
            raise StructureComparisonError("only triangular PLY faces are supported")
        faces = np.asarray(records["indices"], dtype=np.int32).copy()
        if handle.read(1):
            raise StructureComparisonError("PLY contains trailing bytes")
    return vertices, faces


def _header_count(header_lines: list[str], element: str) -> int:
    prefix = f"element {element} "
    for line in header_lines:
        if line.startswith(prefix):
            return int(line.split()[-1])
    raise StructureComparisonError(f"missing PLY element count for {element}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-ply", type=Path, required=True)
    parser.add_argument("--candidate-ply", type=Path, required=True)
    parser.add_argument("--input-vertex-count", type=int, required=True)
    parser.add_argument("--input-face-count", type=int, required=True)
    parser.add_argument("--expected-reference-sha256", required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = compare_prefill_files(
        args.reference_ply,
        args.candidate_ply,
        input_vertex_count=args.input_vertex_count,
        input_face_count=args.input_face_count,
        expected_reference_sha256=args.expected_reference_sha256,
        expected_candidate_sha256=args.expected_candidate_sha256,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
