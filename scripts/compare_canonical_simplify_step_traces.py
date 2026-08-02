"""Compare ordered state digests for canonical CUDA and Metal simplification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class TraceComparisonError(RuntimeError):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def compare_traces(
    reference_report: Path,
    candidate_report: Path,
) -> dict[str, Any]:
    reference = _load_trace_report(Path(reference_report), "reference")
    candidate = _load_trace_report(Path(candidate_report), "candidate")
    first_ordered = None
    first_cardinality = None
    stages = []
    for operation in ("simplify_coarse", "simplify_final"):
        reference_steps = reference[operation]
        candidate_steps = candidate[operation]
        if len(reference_steps) != len(candidate_steps):
            raise TraceComparisonError(
                f"{operation} step count mismatch: "
                f"{len(reference_steps)} versus {len(candidate_steps)}"
            )
        compared_steps = []
        for ref_step, cand_step in zip(
            reference_steps,
            candidate_steps,
            strict=True,
        ):
            if ref_step["iteration"] != cand_step["iteration"]:
                raise TraceComparisonError(
                    f"{operation} iteration identity mismatch"
                )
            iteration = int(ref_step["iteration"])
            vertices_exact = (
                ref_step["observation"]["vertices_sha256"]
                == cand_step["observation"]["vertices_sha256"]
            )
            faces_exact = (
                ref_step["observation"]["faces_sha256"]
                == cand_step["observation"]["faces_sha256"]
            )
            ordered_state_exact = vertices_exact and faces_exact
            cardinality_exact = all(
                ref_step[field] == cand_step[field]
                for field in ("output_vertices", "output_faces")
            )
            compared_steps.append(
                {
                    "iteration": iteration,
                    "threshold_exact": (
                        ref_step["threshold"] == cand_step["threshold"]
                    ),
                    "cardinality_exact": cardinality_exact,
                    "vertices_exact": vertices_exact,
                    "faces_exact": faces_exact,
                    "ordered_state_exact": ordered_state_exact,
                    "reference": ref_step,
                    "candidate": cand_step,
                }
            )
            if not ordered_state_exact and first_ordered is None:
                first_ordered = {
                    "operation": operation,
                    "iteration": iteration,
                    "vertices_exact": vertices_exact,
                    "faces_exact": faces_exact,
                }
            if not cardinality_exact and first_cardinality is None:
                first_cardinality = {
                    "operation": operation,
                    "iteration": iteration,
                }
        stages.append(
            {
                "operation": operation,
                "steps": compared_steps,
            }
        )
    return {
        "schema": (
            "trellis2mlx.canonical_simplify_step_trace_comparison.v1"
        ),
        "status": "done",
        "reference_report": str(Path(reference_report)),
        "candidate_report": str(Path(candidate_report)),
        "first_ordered_state_divergence": first_ordered,
        "first_cardinality_divergence": first_cardinality,
        "all_ordered_states_exact": first_ordered is None,
        "all_cardinalities_exact": first_cardinality is None,
        "stages": stages,
    }


def _load_trace_report(path: Path, role: str) -> dict[str, list[dict]]:
    if not path.is_file():
        raise TraceComparisonError(f"{role} report does not exist: {path}")
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != "done"
        or payload.get("primary_output_status") != "validated"
    ):
        raise TraceComparisonError(f"{role} report is not validated")
    route = payload.get("effective_route") or {}
    if (
        route.get("adjacency_order")
        != "ascending-face-id-per-vertex"
        or route.get("reuse_vertex_face_adjacency") is not True
        or route.get("record_simplify_step_digests") is not True
    ):
        raise TraceComparisonError(
            f"{role} report does not identify the digest-enabled route"
        )
    stages: dict[str, list[dict]] = {}
    for stage in payload.get("stage_artifacts") or []:
        operation = stage.get("operation")
        if operation not in {"simplify_coarse", "simplify_final"}:
            continue
        details = stage.get("details") or {}
        if details.get("record_step_digests") is not True:
            raise TraceComparisonError(
                f"{role} {operation} does not record step digests"
            )
        steps = details.get("simplifier_step_trace")
        if not isinstance(steps, list):
            raise TraceComparisonError(
                f"{role} {operation} step trace is missing"
            )
        for step in steps:
            observation = step.get("observation") or {}
            for field in ("vertices_sha256", "faces_sha256"):
                value = observation.get(field)
                if not isinstance(value, str) or len(value) != 64:
                    raise TraceComparisonError(
                        f"{role} {operation} has invalid {field}"
                    )
        stages[operation] = steps
    expected = {"simplify_coarse", "simplify_final"}
    if set(stages) != expected:
        raise TraceComparisonError(
            f"{role} simplify stage set mismatch: {sorted(stages)}"
        )
    return stages


def main() -> int:
    args = build_parser().parse_args()
    report = compare_traces(
        args.reference_report,
        args.candidate_report,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "first_ordered_state_divergence": report[
                    "first_ordered_state_divergence"
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
