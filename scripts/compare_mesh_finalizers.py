#!/usr/bin/env python
"""Compare two topology-changing mesh finalizers on a fixed raw specimen."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trellmlx.mesh_lineage import approximate_surface_transition, mesh_topology_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--before-label", default="before")
    parser.add_argument("--after-label", default="after")
    parser.add_argument("--fixed-raw-sha256")
    parser.add_argument("--max-faces-per-side", type=int, default=500_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one triangle mesh without applying geometry processing."""
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            vertices = np.asarray(data["vertices"])
            faces = np.asarray(data["faces"])
    else:
        import trimesh

        loaded = trimesh.load(path, process=False)
        if isinstance(loaded, trimesh.Scene):
            geometries = list(loaded.geometry.values())
            if len(geometries) != 1:
                raise ValueError(
                    f"expected exactly one geometry in {path}, found {len(geometries)}"
                )
            loaded = geometries[0]
        vertices = np.asarray(loaded.vertices)
        faces = np.asarray(loaded.faces)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape [V, 3], got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape [F, 3], got {faces.shape}")
    return vertices, faces


def _scalar_delta(
    before: float | int,
    after: float | int,
) -> dict[str, float | None]:
    absolute = float(after) - float(before)
    percent = None if float(before) == 0 else absolute / float(before) * 100.0
    return {"absolute": absolute, "percent_of_before": percent}


def build_report(
    before_path: Path,
    after_path: Path,
    *,
    before_label: str,
    after_label: str,
    fixed_raw_sha256: str | None,
    max_faces_per_side: int,
) -> dict[str, Any]:
    if max_faces_per_side <= 0:
        raise ValueError("max_faces_per_side must be positive")
    before_vertices, before_faces = load_mesh(before_path)
    after_vertices, after_faces = load_mesh(after_path)
    before_summary = mesh_topology_summary(before_vertices, before_faces)
    after_summary = mesh_topology_summary(after_vertices, after_faces)
    delta_fields = (
        "vertices",
        "faces",
        "surface_area",
        "degenerate_faces",
        "boundary_edges",
        "manifold_edges",
        "nonmanifold_edges",
        "same_direction_manifold_conflicts",
    )
    return {
        "schema": "trellis2mlx.fixed-raw-finalizer-comparison.v1",
        "claim_scope": {
            "fixed_raw_specimen_sha256": fixed_raw_sha256,
            "topology": "exact scalar summaries of each final mesh",
            "surface_comparison": "nearest-face-centroid proximity, not ancestry",
        },
        "before": {
            "label": before_label,
            "path": str(before_path),
            "sha256": _sha256_file(before_path),
            "topology": before_summary,
        },
        "after": {
            "label": after_label,
            "path": str(after_path),
            "sha256": _sha256_file(after_path),
            "topology": after_summary,
        },
        "delta_after_minus_before": {
            field: _scalar_delta(before_summary[field], after_summary[field])
            for field in delta_fields
        },
        "surface_proximity": approximate_surface_transition(
            before_vertices,
            before_faces,
            after_vertices,
            after_faces,
            max_faces_per_side=max_faces_per_side,
        ),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {args.output}; pass --overwrite")
    report = build_report(
        args.before,
        args.after,
        before_label=args.before_label,
        after_label=args.after_label,
        fixed_raw_sha256=args.fixed_raw_sha256,
        max_faces_per_side=args.max_faces_per_side,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
