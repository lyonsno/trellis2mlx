"""Measure topology and orientability across a captured mesh stage sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from scripts.analyze_orientation_semantics import (
    _manifold_constraints,
    _solve_parity_components,
)
from scripts.source_cuda_cumesh_postprocess_witness import (
    read_binary_ply,
    sha256_file,
)
from scripts.source_cuda_cumesh_orientation_repeats_witness import (
    _orientation_delta,
    _same_direction_conflicts,
)


SCHEMA = "trellis2mlx.mesh_stage_topology.v1"
_STAGE_PATTERN = re.compile(r"^(?P<index>\d+?)_(?P<operation>.+)\.ply$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", required=True)
    parser.add_argument("--raw-ply", required=True, type=Path)
    parser.add_argument("--stages-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def analyze_stage(path: Path, *, index: int, operation: str) -> dict[str, Any]:
    vertices, faces = read_binary_ply(path)
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    edge_report = _same_direction_conflicts(faces)
    face_a, face_b, required_xor, edge_groups = _manifold_constraints(faces)
    solved = _solve_parity_components(
        len(faces),
        face_a,
        face_b,
        required_xor,
    )
    contradictory_faces = solved["contradictory_face_count"]
    return {
        "index": index,
        "operation": operation,
        "path": str(path),
        "sha256": sha256_file(path),
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "edge_groups": int(edge_groups),
        "boundary_edges": edge_report["boundary_edges"],
        "manifold_edges": edge_report["manifold_edges"],
        "nonmanifold_edges": edge_report["nonmanifold_edges"],
        "same_direction_manifold_edges": edge_report[
            "same_direction_conflicts"
        ],
        "opposite_direction_manifold_edges": edge_report[
            "opposite_direction_manifold_edges"
        ],
        "components": solved["component_count"],
        "orientable_components": solved["orientable_component_count"],
        "contradictory_components": solved[
            "contradictory_component_count"
        ],
        "orientable_faces": solved["orientable_face_count"],
        "contradictory_faces": contradictory_faces,
        "contradictory_face_fraction": (
            float(contradictory_faces / len(faces)) if len(faces) else 0.0
        ),
    }


def _stage_files(stages_dir: Path) -> list[tuple[int, str, Path]]:
    stages = []
    for path in stages_dir.glob("*.ply"):
        match = _STAGE_PATTERN.match(path.name)
        if match is None:
            continue
        stages.append(
            (
                int(match.group("index")),
                match.group("operation"),
                path,
            )
        )
    return sorted(stages)


def _validate_stage_files(
    stage_files: list[tuple[int, str, Path]],
    stages_dir: Path,
) -> None:
    indices = [index for index, _, _ in stage_files]
    expected = list(range(1, 13))
    if indices != expected:
        raise ValueError(
            f"expected exactly numbered stage PLYs 1..12 in {stages_dir}, "
            f"found indices {indices}"
        )


def analyze_orientation_transition(
    before_path: Path,
    after_path: Path,
) -> dict[str, Any]:
    before_vertices, before_faces = read_binary_ply(before_path)
    after_vertices, after_faces = read_binary_ply(after_path)
    delta = _orientation_delta(before_faces, after_faces)
    return {
        "before_path": str(before_path),
        "after_path": str(after_path),
        "vertices_exact": bool(np.array_equal(before_vertices, after_vertices)),
        "face_rows": int(len(before_faces)),
        "same_orientation_rows": delta["same"],
        "reversed_orientation_rows": delta["reversed"],
        "noncorresponding_rows": delta["neither"],
        "topology_preserved_row_for_row": bool(
            delta["neither"] == 0
            and delta["same"] + delta["reversed"] == len(before_faces)
        ),
    }


def run(
    *,
    route: str,
    raw_ply: Path,
    stages_dir: Path,
    output_json: Path,
) -> dict[str, Any]:
    raw_ply = raw_ply.resolve()
    stages_dir = stages_dir.resolve()
    stage_files = _stage_files(stages_dir)
    _validate_stage_files(stage_files, stages_dir)
    records = [analyze_stage(raw_ply, index=0, operation="raw")]
    records.extend(
        analyze_stage(path, index=index, operation=operation)
        for index, operation, path in stage_files
    )
    stage_paths = {index: path for index, _, path in stage_files}
    report = {
        "schema": SCHEMA,
        "route": route,
        "raw_ply": str(raw_ply),
        "stages_dir": str(stages_dir),
        "stages": records,
        "final_orientation_transition": analyze_orientation_transition(
            stage_paths[11],
            stage_paths[12],
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    args = build_parser().parse_args()
    run(
        route=args.route,
        raw_ply=args.raw_ply,
        stages_dir=args.stages_dir,
        output_json=args.output_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
