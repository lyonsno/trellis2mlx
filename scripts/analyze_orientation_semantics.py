"""Compare face-orientation outputs on their satisfiable topology."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.source_cuda_cumesh_postprocess_witness import (
    read_binary_ply,
    sha256_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--reference-ply", type=Path, required=True)
    parser.add_argument("--candidate-ply", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-reference-sha256", required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    return parser


def _manifold_constraints(
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    face_count = len(faces)
    face_ids = np.arange(face_count, dtype=np.int32)
    sources = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    targets = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    edge_faces = np.concatenate([face_ids, face_ids, face_ids])
    low = np.minimum(sources, targets)
    high = np.maximum(sources, targets)
    order = np.lexsort((high, low))
    low = low[order]
    high = high[order]
    edge_faces = edge_faces[order]
    directions = sources[order] < targets[order]
    starts = np.r_[
        0,
        np.flatnonzero(
            (low[1:] != low[:-1]) | (high[1:] != high[:-1])
        )
        + 1,
    ]
    ends = np.r_[starts[1:], len(low)]
    manifold = starts[(ends - starts) == 2]
    return (
        edge_faces[manifold],
        edge_faces[manifold + 1],
        (directions[manifold] == directions[manifold + 1]).astype(
            np.uint8
        ),
        int(len(starts)),
    )


def _solve_parity_components(
    face_count: int,
    face_a: np.ndarray,
    face_b: np.ndarray,
    required_xor: np.ndarray,
) -> dict[str, Any]:
    parent = np.arange(face_count, dtype=np.int32)
    rank = np.zeros(face_count, dtype=np.uint8)
    parity_to_parent = np.zeros(face_count, dtype=np.uint8)
    contradictory = np.zeros(face_count, dtype=np.uint8)

    def find(face: int) -> tuple[int, int]:
        direct_parent = int(parent[face])
        if direct_parent == face:
            return face, 0
        root, parent_parity = find(direct_parent)
        parity_to_parent[face] ^= parent_parity
        parent[face] = root
        return root, int(parity_to_parent[face])

    for left, right, constraint in zip(
        face_a,
        face_b,
        required_xor,
        strict=True,
    ):
        root_left, parity_left = find(int(left))
        root_right, parity_right = find(int(right))
        constraint = int(constraint)
        if root_left == root_right:
            if (parity_left ^ parity_right) != constraint:
                contradictory[root_left] = 1
            continue
        if rank[root_left] < rank[root_right]:
            root_left, root_right = root_right, root_left
            parity_left, parity_right = parity_right, parity_left
        parent[root_right] = root_left
        parity_to_parent[root_right] = (
            parity_left ^ parity_right ^ constraint
        )
        contradictory[root_left] |= contradictory[root_right]
        if rank[root_left] == rank[root_right]:
            rank[root_left] += 1

    roots = np.empty(face_count, dtype=np.int32)
    parity = np.empty(face_count, dtype=np.uint8)
    for face in range(face_count):
        roots[face], parity[face] = find(face)
    root_contradictory = np.zeros(face_count, dtype=np.uint8)
    for root in np.flatnonzero(contradictory):
        resolved_root, _ = find(int(root))
        root_contradictory[resolved_root] = 1
    sizes = np.bincount(roots, minlength=face_count)
    component_roots = np.flatnonzero(sizes)
    contradictory_faces = root_contradictory[roots].astype(bool)
    return {
        "component_count": int(len(component_roots)),
        "contradictory_component_count": int(
            root_contradictory[component_roots].sum()
        ),
        "contradictory_face_count": int(contradictory_faces.sum()),
        "orientable_component_count": int(
            len(component_roots)
            - root_contradictory[component_roots].sum()
        ),
        "orientable_face_count": int(
            face_count - contradictory_faces.sum()
        ),
        "_roots": roots,
        "_parity": parity,
        "_sizes": sizes,
        "_contradictory_faces": contradictory_faces,
        "_root_contradictory": root_contradictory,
    }


def _output_flips(
    input_faces: np.ndarray,
    output_faces: np.ndarray,
) -> tuple[np.ndarray, bool, int]:
    same_shape = input_faces.shape == output_faces.shape
    if not same_shape:
        return np.zeros(len(input_faces), dtype=np.uint8), False, len(
            input_faces
        )
    unchanged = np.all(output_faces == input_faces, axis=1)
    reversed_winding = (
        (output_faces[:, 0] == input_faces[:, 0])
        & (output_faces[:, 1] == input_faces[:, 2])
        & (output_faces[:, 2] == input_faces[:, 1])
    )
    valid = unchanged | reversed_winding
    canonical_exact = bool(
        np.array_equal(
            np.sort(input_faces, axis=1),
            np.sort(output_faces, axis=1),
        )
    )
    return (
        reversed_winding.astype(np.uint8),
        canonical_exact and bool(np.all(valid)),
        int(np.count_nonzero(~valid)),
    )


def analyze_orientation_topology(faces: np.ndarray) -> dict[str, int]:
    """Summarize whether manifold-edge winding constraints are satisfiable."""
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    face_a, face_b, required_xor, edge_group_count = (
        _manifold_constraints(faces)
    )
    solved = _solve_parity_components(
        len(faces),
        face_a,
        face_b,
        required_xor,
    )
    return {
        "faces": int(len(faces)),
        "edge_groups": edge_group_count,
        "manifold_edges": int(len(face_a)),
        "components": solved["component_count"],
        "orientable_components": solved["orientable_component_count"],
        "contradictory_components": solved[
            "contradictory_component_count"
        ],
        "orientable_faces": solved["orientable_face_count"],
        "contradictory_faces": solved["contradictory_face_count"],
    }


def _component_non_global_choices(
    values: np.ndarray,
    roots: np.ndarray,
    sizes: np.ndarray,
    root_contradictory: np.ndarray,
) -> tuple[int, int]:
    minimum = np.full(len(sizes), 2, dtype=np.uint8)
    maximum = np.zeros(len(sizes), dtype=np.uint8)
    np.minimum.at(minimum, roots, values)
    np.maximum.at(maximum, roots, values)
    inconsistent = (
        (minimum != maximum)
        & (sizes > 0)
        & (root_contradictory == 0)
    )
    return int(inconsistent.sum()), int(sizes[inconsistent].sum())


def analyze_face_orientations(
    input_faces: np.ndarray,
    reference_faces: np.ndarray,
    candidate_faces: np.ndarray,
) -> dict[str, Any]:
    input_faces = np.ascontiguousarray(input_faces, dtype=np.int32)
    reference_faces = np.ascontiguousarray(reference_faces, dtype=np.int32)
    candidate_faces = np.ascontiguousarray(candidate_faces, dtype=np.int32)
    face_a, face_b, required_xor, edge_group_count = (
        _manifold_constraints(input_faces)
    )
    solved = _solve_parity_components(
        len(input_faces),
        face_a,
        face_b,
        required_xor,
    )
    roots = solved["_roots"]
    parity = solved["_parity"]
    sizes = solved["_sizes"]
    contradictory_faces = solved["_contradictory_faces"]
    root_contradictory = solved["_root_contradictory"]

    output_reports = {}
    output_flips = {}
    for name, faces in (
        ("reference", reference_faces),
        ("candidate", candidate_faces),
    ):
        flips, canonical_exact, invalid_rows = _output_flips(
            input_faces,
            faces,
        )
        output_flips[name] = flips
        violated = (flips[face_a] ^ flips[face_b]) != required_xor
        non_global_components, non_global_faces = (
            _component_non_global_choices(
                flips ^ parity,
                roots,
                sizes,
                root_contradictory,
            )
        )
        output_reports[name] = {
            "canonical_face_rows_exact": canonical_exact,
            "invalid_face_rows": invalid_rows,
            "edge_violations": int(violated.sum()),
            "orientable_edge_violations": int(
                np.count_nonzero(violated & ~contradictory_faces[face_a])
            ),
            "contradictory_edge_violations": int(
                np.count_nonzero(violated & contradictory_faces[face_a])
            ),
            "orientable_non_global_choice_components": (
                non_global_components
            ),
            "orientable_non_global_choice_faces": non_global_faces,
        }

    differing = output_flips["reference"] ^ output_flips["candidate"]
    comparison_non_global, comparison_non_global_faces = (
        _component_non_global_choices(
            differing,
            roots,
            sizes,
            root_contradictory,
        )
    )
    comparison = {
        "differing_face_rows": int(differing.sum()),
        "orientable_differing_face_rows": int(
            np.count_nonzero(differing & ~contradictory_faces)
        ),
        "contradictory_differing_face_rows": int(
            np.count_nonzero(differing & contradictory_faces)
        ),
        "orientable_non_global_choice_components": comparison_non_global,
        "orientable_non_global_choice_faces": (
            comparison_non_global_faces
        ),
    }
    semantic_parity = bool(
        all(
            output_reports[name]["canonical_face_rows_exact"]
            and output_reports[name]["orientable_edge_violations"] == 0
            and output_reports[name][
                "orientable_non_global_choice_components"
            ]
            == 0
            for name in ("reference", "candidate")
        )
        and comparison_non_global == 0
    )
    return {
        "schema": "trellis2mlx.orientation_semantics.v1",
        "semantic_parity": semantic_parity,
        "topology": analyze_orientation_topology(input_faces),
        **output_reports,
        "comparison": comparison,
    }


def main() -> int:
    args = build_parser().parse_args()
    requested_output_json = args.output_json
    protected_inputs = (
        args.input_ply,
        args.reference_ply,
        args.candidate_ply,
    )
    output_aliases_input = any(
        requested_output_json.resolve(strict=False)
        == path.resolve(strict=False)
        for path in protected_inputs
    )
    effective_output_json = (
        requested_output_json.with_name(
            requested_output_json.name + ".failure.json"
        )
        if output_aliases_input
        else requested_output_json
    )
    report: dict[str, Any] = {
        "schema": "trellis2mlx.orientation_semantics.v1",
        "status": "running",
        "failure_phase": None,
        "last_trustworthy_phase": "request_validated",
        "requested_output_json": str(requested_output_json),
        "effective_output_json": str(effective_output_json),
        "report_rerouted": output_aliases_input,
        "artifacts": {
            name: {
                "path": str(path),
                "expected_sha256": expected,
                "status": "unchecked",
            }
            for name, path, expected in (
                ("input", args.input_ply, args.expected_input_sha256),
                (
                    "reference",
                    args.reference_ply,
                    args.expected_reference_sha256,
                ),
                (
                    "candidate",
                    args.candidate_ply,
                    args.expected_candidate_sha256,
                ),
            )
        },
    }

    def write_report() -> None:
        effective_output_json.parent.mkdir(parents=True, exist_ok=True)
        effective_output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )

    phase = "request_validation"
    try:
        if output_aliases_input:
            raise ValueError(
                "output JSON aliases a protected input; failure report "
                f"rerouted to {effective_output_json}"
            )

        phase = "identity_validation"
        for name, path, expected in (
            ("input", args.input_ply, args.expected_input_sha256),
            ("reference", args.reference_ply, args.expected_reference_sha256),
            ("candidate", args.candidate_ply, args.expected_candidate_sha256),
        ):
            actual = sha256_file(path)
            artifact = report["artifacts"][name]
            artifact["sha256"] = actual
            if actual != expected:
                artifact["status"] = "hash_mismatch"
                raise ValueError(
                    f"{name} PLY SHA256 mismatch: expected {expected}, "
                    f"got {actual}"
                )
            artifact["status"] = "validated"
        report["last_trustworthy_phase"] = "identities_validated"

        phase = "ply_read"
        input_vertices, input_faces = read_binary_ply(args.input_ply)
        reference_vertices, reference_faces = read_binary_ply(
            args.reference_ply
        )
        candidate_vertices, candidate_faces = read_binary_ply(
            args.candidate_ply
        )
        report["last_trustworthy_phase"] = "ply_arrays_read"

        phase = "semantic_analysis"
        analysis = analyze_face_orientations(
            input_faces,
            reference_faces,
            candidate_faces,
        )
        report.update(analysis)
        report["vertices"] = {
            "reference_exact": bool(
                np.array_equal(input_vertices, reference_vertices)
            ),
            "candidate_exact": bool(
                np.array_equal(input_vertices, candidate_vertices)
            ),
        }
        report["semantic_parity"] = bool(
            report["semantic_parity"]
            and report["vertices"]["reference_exact"]
            and report["vertices"]["candidate_exact"]
        )
        report["status"] = "done"
        report["last_trustworthy_phase"] = "semantic_analysis_complete"
        write_report()
    except Exception as exc:
        report["status"] = "failed"
        report["failure_phase"] = phase
        report["error"] = f"{type(exc).__name__}: {exc}"
        write_report()
        print(json.dumps(report, sort_keys=True))
        return 1

    print(json.dumps(report, sort_keys=True))
    return 0 if report["semantic_parity"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
