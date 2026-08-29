#!/usr/bin/env python
"""Replay reference cleanup and report stage-aware mesh lineage evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trellmlx.mesh_cleanup import (
    fill_small_holes,
    orient_faces_by_adjacency,
    remove_duplicate_faces,
    remove_small_components,
    repair_non_manifold_edges,
)
from trellmlx.mesh_lineage import (
    approximate_surface_transition,
    attest_uv_mapping,
    exact_face_transition,
    mesh_topology_summary,
)
from scripts.analyze_orientation_semantics import (
    analyze_face_orientations,
    analyze_orientation_topology,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-mesh", type=Path, required=True)
    parser.add_argument("--expected-clean", type=Path)
    parser.add_argument("--expected-uv", type=Path)
    parser.add_argument("--expected-glb", type=Path)
    parser.add_argument("--target-faces", type=int, default=100_000)
    parser.add_argument("--proximity-max-faces", type=int, default=500_000)
    parser.add_argument("--orientation-topology-max-faces", type=int, default=500_000)
    parser.add_argument("--orientation-artifact-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["vertices"]), np.asarray(data["faces"])


def _array_match(
    actual_vertices: np.ndarray,
    actual_faces: np.ndarray,
    expected_vertices: np.ndarray,
    expected_faces: np.ndarray,
) -> dict[str, Any]:
    vertices_shape_equal = actual_vertices.shape == expected_vertices.shape
    faces_shape_equal = actual_faces.shape == expected_faces.shape
    vertices_exact = vertices_shape_equal and np.array_equal(actual_vertices, expected_vertices)
    faces_exact = faces_shape_equal and np.array_equal(actual_faces, expected_faces)
    return {
        "vertices_shape_equal": vertices_shape_equal,
        "faces_shape_equal": faces_shape_equal,
        "vertices_exact": bool(vertices_exact),
        "faces_exact": bool(faces_exact),
        "exact": bool(vertices_exact and faces_exact),
        "actual_vertices_dtype": str(actual_vertices.dtype),
        "expected_vertices_dtype": str(expected_vertices.dtype),
        "actual_faces_dtype": str(actual_faces.dtype),
        "expected_faces_dtype": str(expected_faces.dtype),
    }


def _glb_attestation(
    uv_vertices: np.ndarray,
    uv_faces: np.ndarray,
    path: Path,
) -> dict[str, Any]:
    import trimesh

    scene = trimesh.load(path, force="scene", process=False)
    geometries = list(scene.geometry.values())
    if len(geometries) != 1:
        return {
            "claim": "export-axis-transform-and-index-preservation",
            "geometry_count": len(geometries),
            "attested": False,
        }
    geometry = geometries[0]
    expected_vertices = np.asarray(uv_vertices, dtype=np.float64).copy()
    expected_vertices[:, 1] = uv_vertices[:, 2]
    expected_vertices[:, 2] = -uv_vertices[:, 1]
    loaded_vertices = np.asarray(geometry.vertices)
    loaded_faces = np.asarray(geometry.faces)
    vertex_delta = (
        np.abs(loaded_vertices - expected_vertices)
        if loaded_vertices.shape == expected_vertices.shape
        else np.empty(0)
    )
    vertices_float32_exact = bool(
        loaded_vertices.shape == expected_vertices.shape
        and np.array_equal(
            loaded_vertices.astype(np.float32), expected_vertices.astype(np.float32)
        )
    )
    faces_exact = bool(
        loaded_faces.shape == uv_faces.shape
        and np.array_equal(loaded_faces.astype(uv_faces.dtype), uv_faces)
    )
    return {
        "claim": "export-axis-transform-and-index-preservation",
        "geometry_count": 1,
        "vertices_float32_exact_after_declared_axis_transform": vertices_float32_exact,
        "faces_exact": faces_exact,
        "max_vertex_abs_delta": float(vertex_delta.max()) if vertex_delta.size else None,
        "attested": bool(vertices_float32_exact and faces_exact),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot encode {type(value).__name__}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_npz_atomic(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, vertices=vertices, faces=faces)
    temporary.replace(path)
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {args.output}; pass --overwrite")
    if args.target_faces <= 0:
        raise ValueError("--target-faces must be positive")
    if args.proximity_max_faces <= 0:
        raise ValueError("--proximity-max-faces must be positive")
    if args.orientation_topology_max_faces <= 0:
        raise ValueError("--orientation-topology-max-faces must be positive")

    import fast_simplification

    started = time.time()
    vertices, faces = _load_mesh(args.raw_mesh)
    report: dict[str, Any] = {
        "schema": "trellis2mlx.mesh-postprocess-lineage.v1",
        "route": "reference-cleanup-fast-simplification",
        "raw_mesh": str(args.raw_mesh),
        "target_faces": args.target_faces,
        "stages": [],
        "transitions": [],
    }

    def record_stage(name: str, stage_vertices: np.ndarray, stage_faces: np.ndarray) -> None:
        print(f"  {name}: {len(stage_vertices):,}V {len(stage_faces):,}F", flush=True)
        stage_report = {
            "name": name,
            **mesh_topology_summary(stage_vertices, stage_faces),
        }
        if len(stage_faces) <= args.orientation_topology_max_faces:
            stage_report["orientation_topology"] = analyze_orientation_topology(
                stage_faces
            )
        report["stages"].append(stage_report)

    def exact_step(
        name: str,
        operation: Callable[..., tuple[np.ndarray, np.ndarray]],
        **kwargs: Any,
    ) -> None:
        nonlocal vertices, faces
        before_vertices, before_faces = vertices, faces
        t0 = time.perf_counter()
        vertices, faces = operation(vertices, faces, verbose=False, **kwargs)
        report["transitions"].append(
            {
                "name": name,
                "kind": "exact",
                "seconds": time.perf_counter() - t0,
                **exact_face_transition(before_vertices, before_faces, vertices, faces),
            }
        )
        record_stage(name, vertices, faces)

    record_stage("raw", vertices, faces)
    coarse_target = args.target_faces * 3
    if len(faces) > coarse_target:
        before_vertices, before_faces = vertices, faces
        t0 = time.perf_counter()
        vertices, faces = fast_simplification.simplify(
            vertices, faces, target_count=coarse_target
        )
        report["transitions"].append(
            {
                "name": "coarse_simplify",
                "kind": "approximate",
                "seconds": time.perf_counter() - t0,
                **approximate_surface_transition(
                    before_vertices,
                    before_faces,
                    vertices,
                    faces,
                    max_faces_per_side=args.proximity_max_faces,
                ),
            }
        )
        record_stage("coarse_simplify", vertices, faces)

    exact_step("initial_remove_duplicate_faces", remove_duplicate_faces)
    exact_step("initial_repair_non_manifold_edges", repair_non_manifold_edges)
    exact_step("initial_remove_small_components", remove_small_components, min_area=1e-5)
    before_hole_faces = len(faces)
    exact_step(
        "initial_fill_small_holes", fill_small_holes, max_hole_perimeter=3e-2
    )
    report["transitions"][-1]["synthetic_hole_fill_faces"] = max(
        len(faces) - before_hole_faces, 0
    )

    if len(faces) > args.target_faces:
        before_vertices, before_faces = vertices, faces
        t0 = time.perf_counter()
        vertices, faces = fast_simplification.simplify(
            vertices, faces, target_count=args.target_faces
        )
        report["transitions"].append(
            {
                "name": "final_simplify",
                "kind": "approximate",
                "seconds": time.perf_counter() - t0,
                **approximate_surface_transition(
                    before_vertices,
                    before_faces,
                    vertices,
                    faces,
                    max_faces_per_side=args.proximity_max_faces,
                ),
            }
        )
        record_stage("final_simplify", vertices, faces)

    exact_step("final_remove_duplicate_faces", remove_duplicate_faces)
    exact_step("final_repair_non_manifold_edges", repair_non_manifold_edges)
    exact_step("final_remove_small_components", remove_small_components, min_area=1e-5)
    before_hole_faces = len(faces)
    exact_step("final_fill_small_holes", fill_small_holes, max_hole_perimeter=3e-2)
    report["transitions"][-1]["synthetic_hole_fill_faces"] = max(
        len(faces) - before_hole_faces, 0
    )
    preorientation_vertices = vertices.copy()
    preorientation_faces = faces.copy()
    exact_step("orient_faces_by_adjacency", orient_faces_by_adjacency)
    report["orientation_semantics"] = {
        "roles": {
            "input": "preorientation topology and face order",
            "reference": "local adjacency-oriented output",
            "candidate": "no-orientation baseline",
        },
        **analyze_face_orientations(
            preorientation_faces,
            faces,
            preorientation_faces,
        ),
    }
    if args.orientation_artifact_dir:
        report["orientation_artifacts"] = {
            "preorientation": _write_npz_atomic(
                args.orientation_artifact_dir / "preorientation.npz",
                preorientation_vertices,
                preorientation_faces,
            ),
            "local_adjacency_oriented": _write_npz_atomic(
                args.orientation_artifact_dir / "local-adjacency-oriented.npz",
                vertices,
                faces,
            ),
        }

    if args.expected_clean:
        expected_vertices, expected_faces = _load_mesh(args.expected_clean)
        report["expected_clean"] = {
            "path": str(args.expected_clean),
            **_array_match(vertices, faces, expected_vertices, expected_faces),
        }

    if args.expected_uv:
        with np.load(args.expected_uv, allow_pickle=False) as uv_data:
            uv_vertices = np.asarray(uv_data["vertices"])
            uv_faces = np.asarray(uv_data["faces"])
            vmapping = np.asarray(uv_data["vmapping"])
        report["uv_attestation"] = {
            "path": str(args.expected_uv),
            **attest_uv_mapping(vertices, faces, uv_vertices, uv_faces, vmapping),
        }
        if args.expected_glb:
            report["glb_attestation"] = {
                "path": str(args.expected_glb),
                **_glb_attestation(uv_vertices, uv_faces, args.expected_glb),
            }
    elif args.expected_glb:
        raise ValueError("--expected-glb requires --expected-uv")

    report["elapsed_seconds"] = time.time() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    temporary.replace(args.output)
    print(f"  report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
