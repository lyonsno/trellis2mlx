"""Compare authenticated CUDA CuMesh and Metal mtlmesh geometry stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.source_cuda_cumesh_postprocess_witness import (
    STAGE_SPECS,
    read_binary_ply,
    sha256_file,
)


class ComparisonError(RuntimeError):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def compare_witnesses(
    reference_report: Path,
    candidate_report: Path,
) -> dict[str, Any]:
    reference = _load_validated_report(Path(reference_report), "reference")
    candidate = _load_validated_report(Path(candidate_report), "candidate")
    reference_input = reference["payload"]["input_mesh"].get("sha256")
    candidate_input = candidate["payload"]["input_mesh"].get("sha256")
    if reference_input != candidate_input:
        raise ComparisonError(
            f"input SHA mismatch: reference {reference_input}, candidate {candidate_input}"
        )
    reference_target = reference["payload"]["requested_route"].get("target_faces")
    candidate_target = candidate["payload"]["requested_route"].get("target_faces")
    if reference_target != candidate_target:
        raise ComparisonError(
            "target face mismatch: "
            f"reference {reference_target}, candidate {candidate_target}"
        )

    stages = []
    first_ordered_content_divergence = None
    first_cardinality_divergence = None
    for index, (operation, _filename) in enumerate(STAGE_SPECS, start=1):
        ref_item = reference["artifacts"][index - 1]
        cand_item = candidate["artifacts"][index - 1]
        ref_vertices, ref_faces = reference["arrays"][index - 1]
        cand_vertices, cand_faces = candidate["arrays"][index - 1]
        dimensions_exact = (
            ref_vertices.shape == cand_vertices.shape
            and ref_faces.shape == cand_faces.shape
        )
        vertices_exact = bool(
            ref_vertices.shape == cand_vertices.shape
            and np.array_equal(ref_vertices, cand_vertices)
        )
        faces_exact = bool(
            ref_faces.shape == cand_faces.shape
            and np.array_equal(ref_faces, cand_faces)
        )
        ordered_content_exact = vertices_exact and faces_exact
        stage = {
            "index": index,
            "operation": operation,
            "dimensions_exact": dimensions_exact,
            "ordered_content_exact": ordered_content_exact,
            "ply_bytes_exact": ref_item["sha256"] == cand_item["sha256"],
            "reference": {
                "vertices": int(len(ref_vertices)),
                "faces": int(len(ref_faces)),
                "sha256": ref_item["sha256"],
                "path": ref_item["path"],
                "vertex_bounds": _vertex_bounds(ref_vertices),
            },
            "candidate": {
                "vertices": int(len(cand_vertices)),
                "faces": int(len(cand_faces)),
                "sha256": cand_item["sha256"],
                "path": cand_item["path"],
                "vertex_bounds": _vertex_bounds(cand_vertices),
            },
            "ordered_vertices": _ordered_numeric_delta(ref_vertices, cand_vertices),
            "ordered_faces": _ordered_numeric_delta(ref_faces, cand_faces),
        }
        stages.append(stage)
        if not ordered_content_exact and first_ordered_content_divergence is None:
            first_ordered_content_divergence = {
                "index": index,
                "operation": operation,
            }
        if not dimensions_exact and first_cardinality_divergence is None:
            first_cardinality_divergence = {
                "index": index,
                "operation": operation,
            }

    return {
        "schema": "trellis2mlx.cumesh_postprocess_comparison.v1",
        "status": "done",
        "comparison_scope": (
            "ordered PLY arrays and cardinality; content divergence may include "
            "benign reordering until canonical topology comparison is run"
        ),
        "reference_report": str(Path(reference_report)),
        "candidate_report": str(Path(candidate_report)),
        "input_sha256": reference_input,
        "target_faces": reference_target,
        "first_ordered_content_divergence": first_ordered_content_divergence,
        "first_cardinality_divergence": first_cardinality_divergence,
        "all_ordered_content_exact": first_ordered_content_divergence is None,
        "all_cardinalities_exact": first_cardinality_divergence is None,
        "stages": stages,
    }


def _load_validated_report(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise ComparisonError(f"{role} report does not exist: {path}")
    payload = json.loads(path.read_text())
    if payload.get("status") != "done":
        raise ComparisonError(f"{role} report status is not done: {payload.get('status')}")
    if payload.get("primary_output_status") != "validated":
        raise ComparisonError(
            f"{role} report primary_output_status is not validated: "
            f"{payload.get('primary_output_status')}"
        )
    artifacts = payload.get("stage_artifacts")
    if not isinstance(artifacts, list):
        raise ComparisonError(f"{role} report stage_artifacts is not a list")
    expected_operations = [operation for operation, _ in STAGE_SPECS]
    actual_operations = [item.get("operation") for item in artifacts]
    if actual_operations != expected_operations:
        raise ComparisonError(
            f"{role} report stage set mismatch: "
            f"expected {expected_operations}, got {actual_operations}"
        )

    arrays = []
    for index, item in enumerate(artifacts, start=1):
        if item.get("status") != "validated":
            raise ComparisonError(f"{role} stage {index} is not validated")
        artifact_path = Path(item["path"])
        if not artifact_path.is_absolute():
            artifact_path = path.parent / artifact_path
        if not artifact_path.is_file():
            raise ComparisonError(f"{role} stage artifact is missing: {artifact_path}")
        actual_sha256 = sha256_file(artifact_path)
        if actual_sha256 != item.get("sha256"):
            raise ComparisonError(
                f"{role} stage artifact hash mismatch for {item.get('operation')}: "
                f"expected {item.get('sha256')}, got {actual_sha256}"
            )
        vertices, faces = read_binary_ply(artifact_path)
        if len(vertices) != item.get("output_vertices"):
            raise ComparisonError(
                f"{role} stage vertex count mismatch for {item.get('operation')}"
            )
        if len(faces) != item.get("output_faces"):
            raise ComparisonError(
                f"{role} stage face count mismatch for {item.get('operation')}"
            )
        item["path"] = str(artifact_path)
        arrays.append((vertices, faces))
    return {"payload": payload, "artifacts": artifacts, "arrays": arrays}


def _ordered_numeric_delta(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "exact": False,
        }
    reference_flat = reference.reshape(-1)
    candidate_flat = candidate.reshape(-1)
    nonzero = 0
    max_abs = 0.0
    total_abs = 0.0
    chunk_size = 1_000_000
    for start in range(0, reference_flat.size, chunk_size):
        stop = min(start + chunk_size, reference_flat.size)
        diff = np.abs(
            reference_flat[start:stop].astype(np.float64)
            - candidate_flat[start:stop].astype(np.float64)
        )
        nonzero += int(np.count_nonzero(diff))
        if diff.size:
            max_abs = max(max_abs, float(np.max(diff)))
            total_abs += float(np.sum(diff))
    return {
        "shape_match": True,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "exact": nonzero == 0,
        "nonzero": nonzero,
        "max_abs": max_abs,
        "mean_abs": total_abs / reference_flat.size if reference_flat.size else 0.0,
    }


def _vertex_bounds(vertices: np.ndarray) -> dict[str, Any]:
    if not len(vertices):
        return {"min": None, "max": None, "centroid": None}
    vertices64 = np.asarray(vertices, dtype=np.float64)
    return {
        "min": np.min(vertices64, axis=0).tolist(),
        "max": np.max(vertices64, axis=0).tolist(),
        "centroid": np.mean(vertices64, axis=0).tolist(),
    }


def main() -> int:
    args = build_parser().parse_args()
    report = compare_witnesses(args.reference_report, args.candidate_report)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "first_ordered_content_divergence": report[
                    "first_ordered_content_divergence"
                ],
                "first_cardinality_divergence": report[
                    "first_cardinality_divergence"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
