"""Replay Trellis Mac's fallback mesh path without running inference.

The diagnostic Trellis Mac wrapper writes its decoder mesh directly to OBJ,
then builds the textured GLB through ``fast_simplification`` and ``xatlas``.
This witness snapshots those geometry boundaries and compares them with the
already-exported GLB so final-output damage is not attributed to inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import trimesh


SCHEMA = "trellis2mlx.trellis_mac_fallback_postprocess_witness.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-obj", required=True, type=Path)
    parser.add_argument("--reference-glb", type=Path)
    parser.add_argument("--clean-checkpoint", type=Path)
    parser.add_argument("--uv-checkpoint", type=Path)
    parser.add_argument("--final-glb", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-faces", type=int, default=200_000)
    parser.add_argument(
        "--raw-orientation-only",
        action="store_true",
        help="Audit the raw OBJ's shared-edge orientation and stop before simplification.",
    )
    parser.add_argument(
        "--local-cleanup-only",
        action="store_true",
        help="Apply Trellis2MLX local cleanup to input geometry and stop.",
    )
    parser.add_argument(
        "--checkpoint-hourglass",
        action="store_true",
        help="Audit and export raw, clean, UV, and final pipeline mesh waists.",
    )
    parser.add_argument(
        "--product-route",
        choices=("simplify-first", "cleanup-first", "reference-fast"),
        help="Replay one exact Trellis2MLX cleanup/simplification branch.",
    )
    parser.add_argument(
        "--component-sign-only",
        action="store_true",
        help="Flip coherently inward connected components without moving vertices.",
    )
    parser.add_argument(
        "--reference-stage-sign-ladder",
        action="store_true",
        help=(
            "Capture component sign after every simplification, cleanup, and "
            "adjacency-orientation operation in the reference-fast route."
        ),
    )
    return parser


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as checkpoint:
            missing = {"vertices", "faces"} - set(checkpoint.files)
            if missing:
                raise ValueError(
                    f"mesh checkpoint is missing {sorted(missing)}: {path}"
                )
            return (
                np.ascontiguousarray(checkpoint["vertices"], dtype=np.float32),
                np.ascontiguousarray(checkpoint["faces"], dtype=np.int32),
            )
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"mesh scene is empty: {path}")
        mesh = loaded.to_geometry()
    else:
        mesh = loaded
    return (
        np.ascontiguousarray(mesh.vertices, dtype=np.float32),
        np.ascontiguousarray(mesh.faces, dtype=np.int32),
    )


def oriented_edge_summary(faces: np.ndarray) -> dict[str, Any]:
    """Count shared-edge orientation conflicts without Python edge objects."""
    faces = np.asarray(faces, dtype=np.int64)
    directed = np.concatenate(
        (faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]), axis=0
    )
    lo = np.minimum(directed[:, 0], directed[:, 1]).astype(np.uint64)
    hi = np.maximum(directed[:, 0], directed[:, 1]).astype(np.uint64)
    keys = (lo << np.uint64(32)) | hi
    signs = np.where(directed[:, 0] <= directed[:, 1], 1, -1).astype(np.int8)
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    signs = signs[order]
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    counts = np.diff(np.r_[starts, len(keys)])
    sign_sums = np.add.reduceat(signs.astype(np.int32), starts)
    manifold = counts == 2
    inconsistent = manifold & (sign_sums != 0)
    return {
        "unique_edges": int(len(starts)),
        "boundary_edges": int(np.count_nonzero(counts == 1)),
        "manifold_shared_edges": int(np.count_nonzero(manifold)),
        "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
        "inconsistently_oriented_shared_edges": int(np.count_nonzero(inconsistent)),
        "inconsistent_fraction_of_manifold_shared_edges": (
            float(np.count_nonzero(inconsistent) / np.count_nonzero(manifold))
            if np.any(manifold)
            else None
        ),
    }


def mesh_summary(vertices: np.ndarray, faces: np.ndarray, *, edges: bool) -> dict[str, Any]:
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    report: dict[str, Any] = {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "finite_vertices": bool(np.isfinite(vertices).all()),
        "index_degenerate_faces": int(
            np.count_nonzero(
                (faces[:, 0] == faces[:, 1])
                | (faces[:, 1] == faces[:, 2])
                | (faces[:, 2] == faces[:, 0])
            )
        ),
        "zero_area_faces": int(np.count_nonzero(area2 <= 1e-12)),
        "bounds_min": vertices.min(axis=0).astype(float).tolist(),
        "bounds_max": vertices.max(axis=0).astype(float).tolist(),
    }
    report["oriented_edges"] = (
        oriented_edge_summary(faces)
        if edges
        else {"status": "skipped", "reason": "raw mesh exceeds staged edge-audit budget"}
    )
    return report


def face_normal_comparison(
    left_vertices: np.ndarray,
    left_faces: np.ndarray,
    right_vertices: np.ndarray,
    right_faces: np.ndarray,
) -> dict[str, Any]:
    if len(left_faces) != len(right_faces):
        return {"comparable": False, "reason": "face counts differ"}
    left_tri = left_vertices[left_faces]
    right_tri = right_vertices[right_faces]
    left = np.cross(left_tri[:, 1] - left_tri[:, 0], left_tri[:, 2] - left_tri[:, 0])
    right = np.cross(right_tri[:, 1] - right_tri[:, 0], right_tri[:, 2] - right_tri[:, 0])
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    valid = (left_norm > 1e-12) & (right_norm > 1e-12)
    dots = np.einsum("ij,ij->i", left[valid], right[valid]) / (
        left_norm[valid] * right_norm[valid]
    )
    return {
        "comparable": True,
        "valid_faces": int(np.count_nonzero(valid)),
        "same_direction_faces": int(np.count_nonzero(dots > 0.9999)),
        "opposite_direction_faces": int(np.count_nonzero(dots < -0.9999)),
        "minimum_dot": float(dots.min()) if len(dots) else None,
        "mean_dot": float(dots.mean()) if len(dots) else None,
    }


def export_flat_glb(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual.face_colors = np.tile(
        np.array([[166, 171, 168, 255]], dtype=np.uint8), (len(faces), 1)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


def undo_glb_axis_transform(vertices: np.ndarray) -> np.ndarray:
    """Map exported ``[x, z, -y]`` vertices back to pipeline coordinates."""
    vertices = np.asarray(vertices)
    restored = vertices.copy()
    restored[:, 1] = -vertices[:, 2]
    restored[:, 2] = vertices[:, 1]
    return restored


def orient_components_outward(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Choose an outward sign per face-adjacent component.

    The radial score is the area-weighted flux of each component's face
    normals away from its own vertex centroid. It remains usable for open
    shells where signed-volume-only repair intentionally declines to act.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    components = trimesh.graph.connected_components(
        mesh.face_adjacency,
        nodes=np.arange(len(faces), dtype=np.int64),
        min_len=1,
    )
    oriented = faces.copy()
    records: list[dict[str, Any]] = []
    for component in components:
        component = np.asarray(component, dtype=np.int64)
        component_faces = faces[component]
        vertex_ids = np.unique(component_faces)
        center = vertices[vertex_ids].mean(axis=0)
        triangles = vertices[component_faces]
        crosses = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        radial = triangles.mean(axis=1) - center
        contributions = np.einsum("ij,ij->i", crosses, radial)
        score = float(contributions.sum())
        absolute_flux = float(np.abs(contributions).sum())
        confidence = abs(score) / absolute_flux if absolute_flux > 0.0 else 0.0
        flipped = score < 0.0
        if flipped:
            oriented[component] = oriented[component][:, ::-1]
        records.append(
            {
                "faces": int(len(component)),
                "vertices": int(len(vertex_ids)),
                "radial_score": score,
                "radial_confidence": confidence,
                "flipped": flipped,
            }
        )
    records.sort(key=lambda item: item["faces"], reverse=True)
    return np.ascontiguousarray(oriented, dtype=np.int32), records


def component_sign_summary(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> dict[str, Any]:
    """Measure coherent component sign without mutating the input mesh."""
    _, components = orient_components_outward(vertices, faces)
    flipped_faces = sum(
        item["faces"] for item in components if item["flipped"]
    )
    return {
        "component_count": len(components),
        "inward_component_count": sum(item["flipped"] for item in components),
        "inward_face_count": flipped_faces,
        "inward_face_fraction": flipped_faces / len(faces) if len(faces) else 0.0,
        "largest_components": components[:32],
    }


def radial_flux_summary(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    chunk_size: int = 250_000,
) -> dict[str, Any]:
    """Measure aggregate normal flux without constructing face adjacency."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    center = vertices[np.unique(faces)].mean(axis=0)
    score = 0.0
    absolute_flux = 0.0
    for start in range(0, len(faces), chunk_size):
        triangles = vertices[faces[start : start + chunk_size]]
        crosses = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        radial = triangles.mean(axis=1) - center
        contributions = np.einsum("ij,ij->i", crosses, radial)
        score += float(contributions.sum())
        absolute_flux += float(np.abs(contributions).sum())
    return {
        "radial_score": score,
        "radial_confidence": abs(score) / absolute_flux if absolute_flux else 0.0,
        "predominant_sign": "outward" if score >= 0.0 else "inward",
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_witness(
    *, input_obj: Path, reference_glb: Path, output_dir: Path, target_faces: int
) -> dict[str, Any]:
    import fast_simplification
    import xatlas

    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "phase": "load_raw_obj",
        "input_obj": str(input_obj),
        "input_obj_sha256": sha256(input_obj),
        "reference_glb": str(reference_glb),
        "reference_glb_sha256": sha256(reference_glb),
        "target_faces": int(target_faces),
        "stages": {},
        "comparisons": {},
    }
    report_path = output_dir / "report.json"
    try:
        raw_vertices, raw_faces = load_mesh(input_obj)
        report["stages"]["raw_obj"] = mesh_summary(
            raw_vertices, raw_faces, edges=False
        )

        report["phase"] = "fast_simplification"
        ratio = 1.0 - min(int(target_faces), len(raw_faces)) / len(raw_faces)
        simplified_vertices, simplified_faces = fast_simplification.simplify(
            raw_vertices, raw_faces, ratio
        )
        simplified_vertices = np.ascontiguousarray(simplified_vertices, dtype=np.float32)
        simplified_faces = np.ascontiguousarray(simplified_faces, dtype=np.int32)
        simplified_path = output_dir / "01-fast-simplification-flat.glb"
        export_flat_glb(simplified_path, simplified_vertices, simplified_faces)
        report["stages"]["fast_simplification"] = {
            **mesh_summary(simplified_vertices, simplified_faces, edges=True),
            "artifact": str(simplified_path),
            "artifact_sha256": sha256(simplified_path),
        }

        report["phase"] = "xatlas"
        vmapping, indices, uvs = xatlas.parametrize(
            np.ascontiguousarray(simplified_vertices, dtype=np.float32),
            np.ascontiguousarray(simplified_faces, dtype=np.uint32),
        )
        atlas_vertices = np.ascontiguousarray(simplified_vertices[vmapping], dtype=np.float32)
        atlas_faces = np.ascontiguousarray(indices.reshape(-1, 3), dtype=np.int32)
        atlas_path = output_dir / "02-xatlas-flat.glb"
        export_flat_glb(atlas_path, atlas_vertices, atlas_faces)
        report["stages"]["xatlas"] = {
            **mesh_summary(atlas_vertices, atlas_faces, edges=True),
            "uv_vertices": int(len(uvs)),
            "artifact": str(atlas_path),
            "artifact_sha256": sha256(atlas_path),
        }
        report["comparisons"]["simplified_vs_xatlas_face_normals"] = (
            face_normal_comparison(
                simplified_vertices,
                simplified_faces,
                atlas_vertices,
                atlas_faces,
            )
        )

        report["phase"] = "load_reference_glb"
        reference_vertices, reference_faces = load_mesh(reference_glb)
        reference_flat_path = output_dir / "03-reference-glb-geometry-flat.glb"
        export_flat_glb(reference_flat_path, reference_vertices, reference_faces)
        report["stages"]["reference_glb_geometry"] = {
            **mesh_summary(reference_vertices, reference_faces, edges=True),
            "artifact": str(reference_flat_path),
            "artifact_sha256": sha256(reference_flat_path),
        }
        report["comparisons"]["xatlas_vs_reference_glb_face_normals"] = (
            face_normal_comparison(
                atlas_vertices,
                atlas_faces,
                reference_vertices,
                reference_faces,
            )
        )
        same_shapes = (
            atlas_vertices.shape == reference_vertices.shape
            and atlas_faces.shape == reference_faces.shape
        )
        report["comparisons"]["xatlas_vs_reference_glb_direct"] = {
            "same_shapes": bool(same_shapes),
            "vertices_allclose": bool(
                same_shapes and np.allclose(atlas_vertices, reference_vertices, atol=1e-7)
            ),
            "faces_exact": bool(
                same_shapes and np.array_equal(atlas_faces, reference_faces)
            ),
        }

        report.update(
            {
                "status": "done",
                "phase": "done",
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    except Exception as exc:
        report.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def run_checkpoint_hourglass(
    *,
    raw_checkpoint: Path,
    clean_checkpoint: Path,
    uv_checkpoint: Path,
    final_glb: Path,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        "raw": raw_checkpoint,
        "clean": clean_checkpoint,
        "uv": uv_checkpoint,
        "final": final_glb,
    }
    report: dict[str, Any] = {
        "schema": "trellis2mlx.checkpoint_mesh_hourglass.v1",
        "status": "failed",
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in inputs.items()
        },
        "stages": {},
        "comparisons": {},
    }
    report_path = output_dir / "checkpoint-hourglass-report.json"
    try:
        loaded: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for index, (name, source) in enumerate(inputs.items(), start=1):
            vertices, faces = load_mesh(source)
            loaded[name] = (vertices, faces)
            artifact = output_dir / f"{index:02d}-{name}-geometry-flat.glb"
            export_flat_glb(artifact, vertices, faces)
            report["stages"][name] = {
                **mesh_summary(vertices, faces, edges=True),
                "artifact": str(artifact),
                "artifact_sha256": sha256(artifact),
            }

        clean_vertices, clean_faces = loaded["clean"]
        uv_vertices, uv_faces = loaded["uv"]
        final_vertices, final_faces = loaded["final"]
        report["comparisons"]["clean_vs_uv_face_normals"] = face_normal_comparison(
            clean_vertices,
            clean_faces,
            uv_vertices,
            uv_faces,
        )
        restored_final_vertices = undo_glb_axis_transform(final_vertices)
        report["comparisons"]["uv_vs_final_face_normals"] = face_normal_comparison(
            uv_vertices,
            uv_faces,
            restored_final_vertices,
            final_faces,
        )
        same_shapes = (
            uv_vertices.shape == restored_final_vertices.shape
            and uv_faces.shape == final_faces.shape
        )
        report["comparisons"]["uv_vs_final_direct"] = {
            "same_shapes": bool(same_shapes),
            "vertices_allclose_after_axis_restore": bool(
                same_shapes
                and np.allclose(uv_vertices, restored_final_vertices, atol=1e-7)
            ),
            "faces_exact": bool(
                same_shapes and np.array_equal(uv_faces, final_faces)
            ),
        }
        report.update(
            {
                "status": "done",
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    except Exception as exc:
        report.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def run_product_route(
    *,
    input_mesh: Path,
    output_dir: Path,
    target_faces: int,
    route: str,
    postprocess=None,
) -> dict[str, Any]:
    if route not in {"simplify-first", "cleanup-first", "reference-fast"}:
        raise ValueError(f"unknown product route: {route}")
    if postprocess is None:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from generate import _cleanup_and_simplify_mesh as postprocess

    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    vertices, faces = load_mesh(input_mesh)
    operation_trace: list[dict[str, Any]] = []
    out_vertices, out_faces = postprocess(
        vertices,
        faces,
        target_faces=target_faces,
        no_cleanup=False,
        keep_largest=False,
        simplify_first=route == "simplify-first",
        reference_cleanup=route == "reference-fast",
        qem_simplify=False,
        operation_trace=operation_trace,
        log=print,
    )
    artifact = output_dir / f"product-{route}-flat.glb"
    export_flat_glb(artifact, out_vertices, out_faces)
    report = {
        "schema": "trellis2mlx.product_mesh_postprocess_replay.v1",
        "status": "done",
        "route": route,
        "target_faces": int(target_faces),
        "input": {
            "path": str(input_mesh),
            "sha256": sha256(input_mesh),
            **mesh_summary(vertices, faces, edges=True),
        },
        "output": {
            **mesh_summary(out_vertices, out_faces, edges=True),
            "artifact": str(artifact),
            "artifact_sha256": sha256(artifact),
        },
        "operation_trace": operation_trace,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / f"product-{route}-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def run_component_sign_witness(
    *, input_mesh: Path, output_dir: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    vertices, faces = load_mesh(input_mesh)
    oriented_faces, components = orient_components_outward(vertices, faces)
    artifact = output_dir / "component-outward-sign-flat.glb"
    export_flat_glb(artifact, vertices, oriented_faces)
    report = {
        "schema": "trellis2mlx.component_outward_sign_witness.v1",
        "status": "done",
        "input": {
            "path": str(input_mesh),
            "sha256": sha256(input_mesh),
            **mesh_summary(vertices, faces, edges=True),
        },
        "output": {
            **mesh_summary(vertices, oriented_faces, edges=True),
            "artifact": str(artifact),
            "artifact_sha256": sha256(artifact),
        },
        "component_count": len(components),
        "flipped_component_count": sum(item["flipped"] for item in components),
        "flipped_face_count": sum(
            item["faces"] for item in components if item["flipped"]
        ),
        "largest_components": components[:32],
        "vertices_exact": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "component-outward-sign-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def run_reference_stage_sign_ladder(
    *,
    input_mesh: Path,
    output_dir: Path,
    target_faces: int,
    postprocess=None,
    simplify=None,
    cleanup=None,
    orient=None,
) -> dict[str, Any]:
    """Locate the operation where reference-fast component sign changes."""
    if postprocess is None:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from generate import _cleanup_and_simplify_mesh as postprocess
    if simplify is None:
        import fast_simplification

        simplify = fast_simplification.simplify
    if cleanup is None or orient is None:
        from trellmlx.mesh_cleanup import cleanup_mesh, orient_faces_by_adjacency

        cleanup = cleanup or cleanup_mesh
        orient = orient or orient_faces_by_adjacency

    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    vertices, faces = load_mesh(input_mesh)
    captured: list[tuple[str, np.ndarray, np.ndarray]] = []
    simplify_count = 0
    cleanup_count = 0

    def capture(name: str, stage_vertices: np.ndarray, stage_faces: np.ndarray) -> None:
        captured.append(
            (
                name,
                np.array(stage_vertices, copy=True, order="C"),
                np.array(stage_faces, copy=True, order="C"),
            )
        )

    def capture_simplify(stage_vertices, stage_faces, *args, **kwargs):
        nonlocal simplify_count
        out_vertices, out_faces = simplify(
            stage_vertices, stage_faces, *args, **kwargs
        )
        simplify_count += 1
        name = "01-coarse-simplify" if simplify_count == 1 else "03-final-simplify"
        capture(name, out_vertices, out_faces)
        return out_vertices, out_faces

    def capture_cleanup(stage_vertices, stage_faces, **kwargs):
        nonlocal cleanup_count
        out_vertices, out_faces = cleanup(stage_vertices, stage_faces, **kwargs)
        cleanup_count += 1
        name = "02-initial-cleanup" if cleanup_count == 1 else "04-final-cleanup"
        capture(name, out_vertices, out_faces)
        return out_vertices, out_faces

    def capture_orient(stage_vertices, stage_faces, **kwargs):
        out_vertices, out_faces = orient(stage_vertices, stage_faces, **kwargs)
        capture("05-adjacency-orientation", out_vertices, out_faces)
        return out_vertices, out_faces

    operation_trace: list[dict[str, Any]] = []
    out_vertices, out_faces = postprocess(
        vertices,
        faces,
        target_faces=target_faces,
        no_cleanup=False,
        keep_largest=False,
        simplify_first=False,
        reference_cleanup=True,
        qem_simplify=False,
        simplify=capture_simplify,
        cleanup_mesh=capture_cleanup,
        orient_faces_by_adjacency=capture_orient,
        operation_trace=operation_trace,
        log=print,
    )

    stages: dict[str, Any] = {}
    for name, stage_vertices, stage_faces in captured:
        artifact = output_dir / f"{name}-flat.glb"
        export_flat_glb(artifact, stage_vertices, stage_faces)
        stages[name] = {
            **mesh_summary(stage_vertices, stage_faces, edges=True),
            **component_sign_summary(stage_vertices, stage_faces),
            "artifact": str(artifact),
            "artifact_sha256": sha256(artifact),
        }

    report = {
        "schema": "trellis2mlx.reference_fast_stage_sign_ladder.v1",
        "status": "done",
        "input": {
            "path": str(input_mesh),
            "sha256": sha256(input_mesh),
            **mesh_summary(vertices, faces, edges=False),
            "aggregate_radial_flux": radial_flux_summary(vertices, faces),
        },
        "target_faces": int(target_faces),
        "stages": stages,
        "operation_trace": operation_trace,
        "final_matches_last_capture": {
            "vertices_exact": bool(
                captured and np.array_equal(out_vertices, captured[-1][1])
            ),
            "faces_exact": bool(
                captured and np.array_equal(out_faces, captured[-1][2])
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "reference-stage-sign-ladder-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected_modes = sum(
        bool(value)
        for value in (
            args.raw_orientation_only,
            args.local_cleanup_only,
            args.checkpoint_hourglass,
            args.product_route is not None,
            args.component_sign_only,
            args.reference_stage_sign_ladder,
        )
    )
    if selected_modes > 1:
        raise SystemExit("orientation-only modes are mutually exclusive")
    if args.component_sign_only:
        run_component_sign_witness(
            input_mesh=args.input_obj,
            output_dir=args.output_dir,
        )
        return 0
    if args.reference_stage_sign_ladder:
        run_reference_stage_sign_ladder(
            input_mesh=args.input_obj,
            output_dir=args.output_dir,
            target_faces=args.target_faces,
        )
        return 0
    if args.product_route is not None:
        run_product_route(
            input_mesh=args.input_obj,
            output_dir=args.output_dir,
            target_faces=args.target_faces,
            route=args.product_route,
        )
        return 0
    if args.checkpoint_hourglass:
        missing = [
            flag
            for flag, value in (
                ("--clean-checkpoint", args.clean_checkpoint),
                ("--uv-checkpoint", args.uv_checkpoint),
                ("--final-glb", args.final_glb),
            )
            if value is None
        ]
        if missing:
            raise SystemExit(
                "checkpoint hourglass requires " + ", ".join(missing)
            )
        run_checkpoint_hourglass(
            raw_checkpoint=args.input_obj,
            clean_checkpoint=args.clean_checkpoint,
            uv_checkpoint=args.uv_checkpoint,
            final_glb=args.final_glb,
            output_dir=args.output_dir,
        )
        return 0
    if args.raw_orientation_only:
        started = time.perf_counter()
        vertices, faces = load_mesh(args.input_obj)
        report = {
            "schema": "trellis2mlx.raw_mesh_orientation_witness.v1",
            "status": "done",
            "input_obj": str(args.input_obj),
            "input_obj_sha256": sha256(args.input_obj),
            "mesh": mesh_summary(vertices, faces, edges=True),
            "elapsed_seconds": time.perf_counter() - started,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "raw-orientation-report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        return 0
    if args.local_cleanup_only:
        from trellmlx.mesh_cleanup import cleanup_mesh, orient_faces_by_adjacency

        started = time.perf_counter()
        vertices, faces = load_mesh(args.input_obj)
        input_summary = mesh_summary(vertices, faces, edges=True)
        cleaned_vertices, cleaned_faces = cleanup_mesh(
            vertices,
            faces,
            keep_largest=False,
            do_fix_normals=True,
            verbose=True,
        )
        cleaned_path = args.output_dir / "04-trellis2mlx-local-cleanup-flat.glb"
        export_flat_glb(cleaned_path, cleaned_vertices, cleaned_faces)
        oriented_vertices, oriented_faces = orient_faces_by_adjacency(
            cleaned_vertices,
            cleaned_faces,
            verbose=True,
        )
        oriented_path = args.output_dir / "05-adjacency-oriented-flat.glb"
        export_flat_glb(oriented_path, oriented_vertices, oriented_faces)
        report = {
            "schema": "trellis2mlx.local_mesh_cleanup_replay.v1",
            "status": "done",
            "input_mesh": str(args.input_obj),
            "input_mesh_sha256": sha256(args.input_obj),
            "input": input_summary,
            "output": {
                **mesh_summary(cleaned_vertices, cleaned_faces, edges=True),
                "artifact": str(cleaned_path),
                "artifact_sha256": sha256(cleaned_path),
            },
            "adjacency_oriented_output": {
                **mesh_summary(oriented_vertices, oriented_faces, edges=True),
                "artifact": str(oriented_path),
                "artifact_sha256": sha256(oriented_path),
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "local-cleanup-report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        return 0
    if args.reference_glb is None:
        raise SystemExit("--reference-glb is required unless --raw-orientation-only is set")
    run_witness(
        input_obj=args.input_obj,
        reference_glb=args.reference_glb,
        output_dir=args.output_dir,
        target_faces=args.target_faces,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
