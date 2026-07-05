"""Compare reference and candidate TRELLIS stage artifacts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


_SPARSE_FLOW_BLOCK_TRACE_KEYS = (
    "pos_input_projected",
    "pos_block0_norm1",
    "pos_block0_modulated_self_input",
    "pos_block0_q_pre_norm",
    "pos_block0_k_pre_norm",
    "pos_block0_v",
    "pos_block0_q_post_norm",
    "pos_block0_k_post_norm",
    "pos_block0_q_post_rope",
    "pos_block0_k_post_rope",
    "pos_block0_attention_raw",
    "pos_block0_self_attn",
    "pos_block0_after_self",
    "pos_block0_cross_attn",
    "pos_block0_after_cross",
    "pos_block0_mlp",
    "pos_block0_after_mlp",
    "neg_input_projected",
    "neg_block0_norm1",
    "neg_block0_modulated_self_input",
    "neg_block0_q_pre_norm",
    "neg_block0_k_pre_norm",
    "neg_block0_v",
    "neg_block0_q_post_norm",
    "neg_block0_k_post_norm",
    "neg_block0_q_post_rope",
    "neg_block0_k_post_rope",
    "neg_block0_attention_raw",
    "neg_block0_self_attn",
    "neg_block0_after_self",
    "neg_block0_cross_attn",
    "neg_block0_after_cross",
    "neg_block0_mlp",
    "neg_block0_after_mlp",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare TRELLIS stage artifacts")
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "conditioning",
            "sparse_coords",
            "sparse_flow_step",
            "sparse_flow_block_trace",
            "sparse_internals",
            "shape_slat",
            "decoder_output",
            "mesh_raw",
            "mesh_clean",
            "mesh_uv",
        ],
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = compare_stage(args.stage, args.reference, args.candidate)
    _write_json(args.output, report)
    print(json.dumps(_compact_console_summary(report), sort_keys=True))
    return 0


def compare_stage(stage: str, reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    with np.load(reference_path) as ref, np.load(candidate_path) as cand:
        report: dict[str, Any] = {
            "schema": "trellis2mlx.stage_artifact_comparison.v1",
            "stage": stage,
            "reference": _artifact_identity(reference_path),
            "candidate": _artifact_identity(candidate_path),
        }
        if stage == "conditioning":
            report["arrays"] = {
                name: _array_delta(ref[name], cand[name])
                for name in ("cond", "neg_cond")
            }
            return report
        if stage == "sparse_flow_step":
            report["arrays"] = {
                name: _array_delta(ref[name], cand[name])
                for name in (
                    "noise",
                    "pred_pos",
                    "pred_neg",
                    "pred_cfg",
                    "x0_pos",
                    "x0_cfg",
                    "std_pos",
                    "std_cfg",
                    "ratio_raw",
                    "std_ratio",
                    "ratio_effective",
                    "x0_rescaled",
                    "x0_after_rescale",
                    "pred_final",
                    "sample_next",
                    "t",
                    "t_prev",
                )
                if name in ref and name in cand
            }
            return report
        if stage == "sparse_flow_block_trace":
            report["arrays"] = {
                name: _array_delta(ref[name], cand[name])
                for name in _SPARSE_FLOW_BLOCK_TRACE_KEYS
                if name in ref and name in cand
            }
            return report
        if stage == "sparse_internals":
            report["arrays"] = {
                name: _array_delta(ref[name], cand[name])
                for name in ("z_s", "logits", "decoded", "decoded_ds")
            }
            coords_report, _ref_order, _cand_order = _coord_overlap(ref["coords"], cand["coords"])
            report["coords"] = coords_report
            return report
        if stage in {"mesh_raw", "mesh_clean", "mesh_uv"}:
            report.update(_mesh_stage_delta(ref, cand, include_uv=(stage == "mesh_uv")))
            return report

        coords_report, ref_order, cand_order = _coord_overlap(ref["coords"], cand["coords"])
        report["coords"] = coords_report
        if stage in {"shape_slat", "decoder_output"} and "feats" in ref and "feats" in cand:
            report["features"] = _feature_delta(ref["feats"], cand["feats"], ref_order, cand_order)
        return report


def _array_delta(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    shape_match = reference.shape == candidate.shape
    dtype_match = str(reference.dtype) == str(candidate.dtype)
    summary: dict[str, Any] = {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "shape_match": shape_match,
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "dtype_match": dtype_match,
    }
    if not shape_match:
        return summary
    diff = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    summary.update(_diff_summary(diff))
    return summary


def _coord_overlap(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> tuple[dict[str, Any], list[int], list[int]]:
    ref_keys = _coord_index(reference)
    cand_keys = _coord_index(candidate)
    ref_set = set(ref_keys)
    cand_set = set(cand_keys)
    common = sorted(ref_set & cand_set)
    union = ref_set | cand_set
    ref_order = [ref_keys[key] for key in common]
    cand_order = [cand_keys[key] for key in common]
    report = {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "reference_count": int(reference.shape[0]),
        "candidate_count": int(candidate.shape[0]),
        "common_count": len(common),
        "reference_only_count": len(ref_set - cand_set),
        "candidate_only_count": len(cand_set - ref_set),
        "union_count": len(union),
        "jaccard": (len(common) / len(union)) if union else 1.0,
    }
    return report, ref_order, cand_order


def _feature_delta(
    reference_feats: np.ndarray,
    candidate_feats: np.ndarray,
    ref_order: list[int],
    cand_order: list[int],
) -> dict[str, Any]:
    if not ref_order:
        return {
            "common_shape": [0, int(reference_feats.shape[1]) if reference_feats.ndim == 2 else 0],
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "rms_diff": None,
        }
    ref_common = reference_feats[np.asarray(ref_order)]
    cand_common = candidate_feats[np.asarray(cand_order)]
    diff = np.abs(ref_common.astype(np.float64) - cand_common.astype(np.float64))
    return {
        "common_shape": list(ref_common.shape),
        **_diff_summary(diff),
    }


def _diff_summary(diff: np.ndarray) -> dict[str, Any]:
    return {
        "max_abs_diff": float(np.max(diff)) if diff.size else 0.0,
        "mean_abs_diff": float(np.mean(diff)) if diff.size else 0.0,
        "rms_diff": float(np.sqrt(np.mean(np.square(diff)))) if diff.size else 0.0,
    }


def _mesh_stage_delta(reference: Any, candidate: Any, *, include_uv: bool) -> dict[str, Any]:
    _require_keys(reference, ("vertices", "faces"), "reference")
    _require_keys(candidate, ("vertices", "faces"), "candidate")
    ref_vertices = np.asarray(reference["vertices"])
    cand_vertices = np.asarray(candidate["vertices"])
    ref_faces = np.asarray(reference["faces"])
    cand_faces = np.asarray(candidate["faces"])

    report: dict[str, Any] = {
        "vertices": _array_delta(ref_vertices, cand_vertices),
        "faces": {
            **_face_array_summary(ref_faces, cand_faces),
            "index_validity": {
                "reference": _face_index_validity(ref_faces, len(ref_vertices)),
                "candidate": _face_index_validity(cand_faces, len(cand_vertices)),
            },
        },
        "reference_bounds": _vertex_bounds(ref_vertices),
        "candidate_bounds": _vertex_bounds(cand_vertices),
        "face_orientation_overlap": _face_orientation_overlap(ref_faces, cand_faces),
        "reference_edge_consistency": _edge_consistency_summary(ref_faces),
        "candidate_edge_consistency": _edge_consistency_summary(cand_faces),
    }
    if include_uv:
        report["uvs"] = _optional_array_delta(reference, candidate, "uvs")
        report["vmapping"] = _vmapping_delta(reference, candidate)
    return report


def _require_keys(npz: Any, keys: tuple[str, ...], side: str) -> None:
    missing = [key for key in keys if key not in npz]
    if missing:
        raise KeyError(f"{side} artifact missing required arrays: {', '.join(missing)}")


def _optional_array_delta(reference: Any, candidate: Any, name: str) -> dict[str, Any]:
    exists = {"reference_exists": name in reference, "candidate_exists": name in candidate}
    if name not in reference or name not in candidate:
        return exists
    return {**exists, **_array_delta(reference[name], candidate[name])}


def _vmapping_delta(reference: Any, candidate: Any) -> dict[str, Any]:
    exists = {"reference_exists": "vmapping" in reference, "candidate_exists": "vmapping" in candidate}
    if "vmapping" not in reference or "vmapping" not in candidate:
        return exists
    ref = np.asarray(reference["vmapping"])
    cand = np.asarray(candidate["vmapping"])
    shape_match = ref.shape == cand.shape
    report: dict[str, Any] = {
        **exists,
        "reference_shape": list(ref.shape),
        "candidate_shape": list(cand.shape),
        "shape_match": shape_match,
        "reference_dtype": str(ref.dtype),
        "candidate_dtype": str(cand.dtype),
        "dtype_match": str(ref.dtype) == str(cand.dtype),
        "exact_match": bool(shape_match and np.array_equal(ref, cand)),
    }
    if shape_match:
        report["mismatched_entries"] = int(np.count_nonzero(ref != cand))
    return report


def _face_array_summary(reference_faces: np.ndarray, candidate_faces: np.ndarray) -> dict[str, Any]:
    shape_match = reference_faces.shape == candidate_faces.shape
    dtype_match = str(reference_faces.dtype) == str(candidate_faces.dtype)
    exact_row_match = bool(shape_match and np.array_equal(reference_faces, candidate_faces))
    report: dict[str, Any] = {
        "reference_shape": list(reference_faces.shape),
        "candidate_shape": list(candidate_faces.shape),
        "shape_match": shape_match,
        "reference_dtype": str(reference_faces.dtype),
        "candidate_dtype": str(candidate_faces.dtype),
        "dtype_match": dtype_match,
        "exact_row_match": exact_row_match,
    }
    if shape_match:
        report["mismatched_rows"] = int(np.count_nonzero(np.any(reference_faces != candidate_faces, axis=1)))
    return report


def _vertex_bounds(vertices: np.ndarray) -> dict[str, Any]:
    vertices = np.asarray(vertices)
    if vertices.ndim != 2 or vertices.shape[0] == 0:
        return {"shape": list(vertices.shape), "min": None, "max": None}
    return {
        "shape": list(vertices.shape),
        "min": [float(v) for v in np.min(vertices.astype(np.float64), axis=0)],
        "max": [float(v) for v in np.max(vertices.astype(np.float64), axis=0)],
    }


def _face_index_validity(faces: np.ndarray, vertex_count: int) -> dict[str, Any]:
    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.size == 0:
        return {
            "valid_triangle_array": bool(faces.ndim == 2 and faces.shape[1] == 3),
            "min_index": None,
            "max_index": None,
            "valid_indices": bool(faces.ndim == 2 and faces.shape[1] == 3 and faces.size == 0),
        }
    min_index = int(np.min(faces))
    max_index = int(np.max(faces))
    return {
        "valid_triangle_array": True,
        "min_index": min_index,
        "max_index": max_index,
        "valid_indices": bool(min_index >= 0 and max_index < vertex_count),
    }


def _cyclic_orders(face: np.ndarray) -> set[tuple[int, int, int]]:
    a, b, c = [int(v) for v in face]
    return {(a, b, c), (b, c, a), (c, a, b)}


def _face_orientation_overlap(reference_faces: np.ndarray, candidate_faces: np.ndarray) -> dict[str, Any]:
    ref_by_key: dict[tuple[int, int, int], list[np.ndarray]] = defaultdict(list)
    cand_by_key: dict[tuple[int, int, int], list[np.ndarray]] = defaultdict(list)
    for face in np.asarray(reference_faces, dtype=np.int64):
        ref_by_key[tuple(sorted(int(v) for v in face))].append(face)
    for face in np.asarray(candidate_faces, dtype=np.int64):
        cand_by_key[tuple(sorted(int(v) for v in face))].append(face)

    common_keys = sorted(set(ref_by_key) & set(cand_by_key))
    same = 0
    reversed_count = 0
    ambiguous = 0
    unmatched_orientation = 0
    duplicate_common_face_sets = 0
    examples: list[dict[str, Any]] = []
    for key in common_keys:
        refs = ref_by_key[key]
        cands = cand_by_key[key]
        if len(refs) != 1 or len(cands) != 1:
            duplicate_common_face_sets += 1
        for cand in cands:
            same_match = any(tuple(int(v) for v in cand) in _cyclic_orders(ref) for ref in refs)
            reversed_match = any(tuple(int(v) for v in cand[::-1]) in _cyclic_orders(ref) for ref in refs)
            if same_match and not reversed_match:
                same += 1
            elif reversed_match and not same_match:
                reversed_count += 1
                if len(examples) < 5:
                    examples.append({"face_key": list(key), "candidate_face": [int(v) for v in cand]})
            elif same_match and reversed_match:
                ambiguous += 1
            else:
                unmatched_orientation += 1

    ref_set = set(ref_by_key)
    cand_set = set(cand_by_key)
    union_count = len(ref_set | cand_set)
    return {
        "reference_face_sets": len(ref_set),
        "candidate_face_sets": len(cand_set),
        "common_face_sets": len(common_keys),
        "reference_only_face_sets": len(ref_set - cand_set),
        "candidate_only_face_sets": len(cand_set - ref_set),
        "jaccard": (len(common_keys) / union_count) if union_count else 1.0,
        "same_orientation_faces": int(same),
        "reversed_orientation_faces": int(reversed_count),
        "ambiguous_orientation_faces": int(ambiguous),
        "unmatched_orientation_faces": int(unmatched_orientation),
        "duplicate_common_face_sets": int(duplicate_common_face_sets),
        "reversed_examples": examples,
    }


def _edge_consistency_summary(faces: np.ndarray) -> dict[str, Any]:
    undirected_edges: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    if np.asarray(faces).ndim != 2 or np.asarray(faces).shape[-1] != 3:
        return {
            "edges": 0,
            "boundary_edges": 0,
            "manifold_edges": 0,
            "opposite_direction_edges": 0,
            "same_direction_conflict_edges": 0,
            "nonmanifold_edges": 0,
            "duplicate_directed_nonmanifold_edges": 0,
            "conflict_examples": [],
        }
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
                    conflict_examples.append([list(item) for item in edge_faces])
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


def _coord_index(coords: np.ndarray) -> dict[tuple[int, ...], int]:
    return {tuple(int(v) for v in row): i for i, row in enumerate(coords)}


def _artifact_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": _sha256_file(path) if path.exists() else None,
    }


def _compact_console_summary(report: dict[str, Any]) -> dict[str, Any]:
    if "coords" in report:
        summary = {
            "stage": report["stage"],
            "common": report["coords"]["common_count"],
            "jaccard": report["coords"]["jaccard"],
            "reference_count": report["coords"]["reference_count"],
            "candidate_count": report["coords"]["candidate_count"],
        }
        if "features" in report:
            summary["feature_max_abs_diff"] = report["features"]["max_abs_diff"]
            summary["feature_mean_abs_diff"] = report["features"]["mean_abs_diff"]
        return summary
    return {
        "stage": report["stage"],
        "array_max_abs_diff": {
            name: values.get("max_abs_diff")
            for name, values in report.get("arrays", {}).items()
        },
        "array_mean_abs_diff": {
            name: values.get("mean_abs_diff")
            for name, values in report.get("arrays", {}).items()
        },
    }


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


if __name__ == "__main__":
    raise SystemExit(main())
