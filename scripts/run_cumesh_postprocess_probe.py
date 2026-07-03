#!/usr/bin/env python3
"""Run a no-generation cumesh postprocess probe from a saved raw mesh."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROUTE = "trellis2mlx_cumesh_postprocess_probe"
FORBIDDEN_TO_PROVE = [
    "full_trellis2_parity",
    "texture_bake_parity",
    "production_winding_closure",
    "image_conditioning_or_sampling_equivalence",
]
REPORT_NAME = "postprocess_probe_report.json"


class ProbeError(RuntimeError):
    def __init__(self, phase: str, message: str):
        super().__init__(message)
        self.phase = phase


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n")


def _git_identity() -> dict[str, Any]:
    def run_git(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    return {
        "cwd": str(Path.cwd()),
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "dirty_short": run_git("status", "--short"),
    }


def _last_trustworthy_evidence(raw_mesh: Path, output_dir: Path) -> dict[str, Any]:
    output_glb = output_dir / "output.glb"
    report = {
        "raw_mesh_exists": raw_mesh.exists(),
        "raw_mesh_size_bytes": raw_mesh.stat().st_size if raw_mesh.exists() else None,
        "output_dir_exists": output_dir.exists(),
        "primary_output_exists": output_glb.exists(),
        "primary_output_size_bytes": output_glb.stat().st_size if output_glb.exists() else None,
    }
    checkpoints = output_dir / "checkpoints"
    if checkpoints.exists():
        report["checkpoint_files"] = sorted(path.name for path in checkpoints.iterdir())
    return report


def _base_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "route": ROUTE,
        "raw_mesh": str(args.raw_mesh),
        "output_dir": str(args.output_dir),
        "target_faces": int(args.target_faces),
        "device": args.device,
        "primary_output": str(args.output_dir / "output.glb"),
        "primary_output_status": "not_produced",
        "matched_variables": {
            "raw_mesh": str(args.raw_mesh),
            "target_faces": int(args.target_faces),
            "geometry_cleanup_sequence": "official_cumesh_postprocess_order",
        },
        "intentional_differences": [
            "no_image_conditioning",
            "no_sampling",
            "no_texture_decode",
            "no_texture_bake",
            "untextured_glb_export_for_geometry_culling_witness",
        ],
        "forbidden_to_prove": FORBIDDEN_TO_PROVE,
        "route_identity": {
            "script": str(Path(__file__).resolve()),
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "pid": os.getpid(),
            "git": _git_identity(),
        },
    }


def _failure_report(args: argparse.Namespace, phase: str, error: str) -> dict[str, Any]:
    report = _base_report(args)
    report.update(
        {
            "status": "error",
            "phase": phase,
            "error": error,
            "last_trustworthy_evidence": _last_trustworthy_evidence(args.raw_mesh, args.output_dir),
        }
    )
    return report


def _load_raw_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise ProbeError("load_inputs", f"raw mesh does not exist: {path}")
    try:
        data = np.load(path)
    except Exception as exc:
        raise ProbeError("load_inputs", f"failed to load raw mesh npz: {exc}") from exc

    if "vertices" not in data or "faces" not in data:
        raise ProbeError("load_inputs", "raw mesh npz must contain vertices and faces arrays")
    vertices = np.asarray(data["vertices"], dtype=np.float32)
    faces = np.asarray(data["faces"], dtype=np.int64)

    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        raise ProbeError("validate_inputs", f"invalid vertices shape: {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
        raise ProbeError("validate_inputs", f"invalid faces shape: {faces.shape}")
    if not np.isfinite(vertices).all():
        raise ProbeError("validate_inputs", "vertices contain non-finite values")
    if faces.min(initial=0) < 0 or faces.max(initial=-1) >= len(vertices):
        raise ProbeError("validate_inputs", "faces reference vertices outside the vertex array")
    return vertices, faces


def _as_numpy(tensor: Any) -> np.ndarray:
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    return np.asarray(tensor)


def _read_mesh(mesh: Any) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = mesh.read()
    return _as_numpy(vertices).astype(np.float32), _as_numpy(faces).astype(np.int64)


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _time_stage(
    *,
    name: str,
    mesh: Any,
    stages: list[dict[str, Any]],
    fn: Callable[[], Any],
    save_path: Path | None = None,
) -> Any:
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    vertices, faces = _read_mesh(mesh)
    record: dict[str, Any] = {
        "name": name,
        "elapsed_seconds": elapsed,
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
    }
    if save_path is not None:
        _save_npz(save_path, vertices=vertices, faces=faces)
        record["checkpoint"] = str(save_path)
    stages.append(record)
    print(
        f"[{name}] {len(vertices):,}V {len(faces):,}F in {elapsed:.3f}s",
        flush=True,
    )
    return result


def _export_glb(
    *,
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray | None = None,
    vertex_normals: np.ndarray | None = None,
) -> None:
    import trimesh
    from trimesh.visual.material import PBRMaterial

    export_vertices = vertices.astype(np.float32).copy()
    export_vertices[:, 1], export_vertices[:, 2] = vertices[:, 2].copy(), -vertices[:, 1].copy()
    export_uvs = None
    if uvs is not None and len(uvs) == len(vertices):
        export_uvs = uvs.astype(np.float32).copy()
        export_uvs[:, 1] = 1.0 - export_uvs[:, 1]

    export_normals = None
    if vertex_normals is not None and len(vertex_normals) == len(vertices):
        export_normals = vertex_normals.astype(np.float32).copy()
        export_normals[:, 1], export_normals[:, 2] = vertex_normals[:, 2].copy(), -vertex_normals[:, 1].copy()

    visual = None
    if export_uvs is not None:
        material = PBRMaterial(
            baseColorFactor=np.array([186, 188, 190, 255], dtype=np.uint8),
            metallicFactor=0.0,
            roughnessFactor=0.7,
            alphaMode="OPAQUE",
            doubleSided=True,
        )
        visual = trimesh.visual.TextureVisuals(uv=export_uvs, material=material)

    mesh = trimesh.Trimesh(
        vertices=export_vertices,
        faces=faces.astype(np.int64),
        vertex_normals=export_normals,
        process=False,
        visual=visual,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    total_start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    vertices, faces = _load_raw_mesh(args.raw_mesh)
    print(f"[load_inputs] {len(vertices):,}V {len(faces):,}F from {args.raw_mesh}", flush=True)

    try:
        import cumesh
        import torch
    except Exception as exc:
        raise ProbeError("import_dependencies", f"failed to import cumesh/torch: {exc}") from exc

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ProbeError("init_cumesh", "requested device=mps but torch.backends.mps is unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ProbeError("init_cumesh", "requested device=cuda but torch.cuda is unavailable")

    device = torch.device(args.device)
    mesh = cumesh.CuMesh()
    mesh.init(
        torch.as_tensor(vertices, dtype=torch.float32, device=device),
        torch.as_tensor(faces, dtype=torch.int64, device=device),
    )
    stages: list[dict[str, Any]] = []
    _save_npz(checkpoints / "mesh_raw.npz", vertices=vertices, faces=faces)
    stages.append(
        {
            "name": "mesh_raw_input",
            "elapsed_seconds": 0.0,
            "vertices": int(len(vertices)),
            "faces": int(len(faces)),
            "checkpoint": str(checkpoints / "mesh_raw.npz"),
        }
    )

    def cleanup_pass(prefix: str) -> None:
        _time_stage(
            name=f"{prefix}_remove_duplicate_faces",
            mesh=mesh,
            stages=stages,
            fn=mesh.remove_duplicate_faces,
        )
        _time_stage(
            name=f"{prefix}_repair_non_manifold_edges",
            mesh=mesh,
            stages=stages,
            fn=mesh.repair_non_manifold_edges,
        )
        _time_stage(
            name=f"{prefix}_remove_small_connected_components",
            mesh=mesh,
            stages=stages,
            fn=lambda: mesh.remove_small_connected_components(1e-5),
        )
        _time_stage(
            name=f"{prefix}_fill_holes",
            mesh=mesh,
            stages=stages,
            fn=lambda: mesh.fill_holes(max_hole_perimeter=3e-2),
        )

    _time_stage(
        name="initial_fill_holes",
        mesh=mesh,
        stages=stages,
        fn=lambda: mesh.fill_holes(max_hole_perimeter=3e-2),
        save_path=checkpoints / "mesh_after_initial_fill.npz",
    )
    _time_stage(
        name="coarse_simplify_3x",
        mesh=mesh,
        stages=stages,
        fn=lambda: mesh.simplify(int(args.target_faces) * 3, verbose=True),
        save_path=checkpoints / "mesh_after_coarse_simplify.npz",
    )
    cleanup_pass("cleanup_pass1")
    _time_stage(
        name="after_cleanup_pass1",
        mesh=mesh,
        stages=stages,
        fn=lambda: None,
        save_path=checkpoints / "mesh_after_cleanup_pass1.npz",
    )
    _time_stage(
        name="final_simplify",
        mesh=mesh,
        stages=stages,
        fn=lambda: mesh.simplify(int(args.target_faces), verbose=True),
        save_path=checkpoints / "mesh_after_final_simplify.npz",
    )
    cleanup_pass("cleanup_pass2")
    _time_stage(
        name="unify_face_orientations",
        mesh=mesh,
        stages=stages,
        fn=mesh.unify_face_orientations,
        save_path=checkpoints / "mesh_clean.npz",
    )

    clean_vertices, clean_faces = _read_mesh(mesh)
    unwrap_start = time.perf_counter()
    uv_vertices_t, uv_faces_t, uvs_t, vmapping_t = mesh.uv_unwrap(return_vmaps=True, verbose=True)
    uv_elapsed = time.perf_counter() - unwrap_start
    uv_vertices = _as_numpy(uv_vertices_t).astype(np.float32)
    uv_faces = _as_numpy(uv_faces_t).astype(np.int64)
    uvs = _as_numpy(uvs_t).astype(np.float32)
    vmapping = _as_numpy(vmapping_t).astype(np.int64)
    stages.append(
        {
            "name": "uv_unwrap",
            "elapsed_seconds": uv_elapsed,
            "vertices": int(len(uv_vertices)),
            "faces": int(len(uv_faces)),
            "checkpoint": str(checkpoints / "mesh_uv.npz"),
        }
    )
    print(f"[uv_unwrap] {len(uv_vertices):,}V {len(uv_faces):,}F in {uv_elapsed:.3f}s", flush=True)

    normals_start = time.perf_counter()
    mesh.compute_vertex_normals()
    clean_normals = _as_numpy(mesh.read_vertex_normals()).astype(np.float32)
    uv_normals = clean_normals[vmapping]
    normals_elapsed = time.perf_counter() - normals_start
    stages.append(
        {
            "name": "compute_vertex_normals_mapped_to_uv",
            "elapsed_seconds": normals_elapsed,
            "vertices": int(len(uv_normals)),
            "faces": int(len(uv_faces)),
        }
    )

    _save_npz(
        checkpoints / "mesh_uv.npz",
        vertices=uv_vertices,
        faces=uv_faces,
        uvs=uvs,
        vmapping=vmapping,
        vertex_normals=uv_normals,
        mesh_coord_space="normalized_world_aabb_-0.5_0.5",
    )
    _save_npz(
        checkpoints / "mesh_clean_with_normals.npz",
        vertices=clean_vertices,
        faces=clean_faces,
        vertex_normals=clean_normals,
        mesh_coord_space="normalized_world_aabb_-0.5_0.5",
    )

    geometry_glb = args.output_dir / "output_geometry_clean.glb"
    _export_glb(path=geometry_glb, vertices=clean_vertices, faces=clean_faces)
    output_glb = args.output_dir / "output.glb"
    _export_glb(path=output_glb, vertices=uv_vertices, faces=uv_faces, uvs=uvs, vertex_normals=uv_normals)

    total_elapsed = time.perf_counter() - total_start
    report = _base_report(args)
    report.update(
        {
            "status": "ok",
            "phase": "complete",
            "primary_output_status": "produced",
            "total_elapsed_seconds": total_elapsed,
            "dependency_identity": {
                "torch_version": torch.__version__,
                "torch_mps_available": bool(torch.backends.mps.is_available()),
                "torch_cuda_available": bool(torch.cuda.is_available()),
                "cumesh_module": getattr(cumesh, "__file__", None),
            },
            "source_artifacts": {
                "raw_mesh": str(args.raw_mesh),
                "output_glb": str(output_glb),
                "geometry_clean_glb": str(geometry_glb),
                "mesh_raw_npz": str(checkpoints / "mesh_raw.npz"),
                "mesh_clean_npz": str(checkpoints / "mesh_clean.npz"),
                "mesh_uv_npz": str(checkpoints / "mesh_uv.npz"),
                "report_json": str(args.output_dir / REPORT_NAME),
            },
            "counts": {
                "raw_vertices": int(len(vertices)),
                "raw_faces": int(len(faces)),
                "clean_vertices": int(len(clean_vertices)),
                "clean_faces": int(len(clean_faces)),
                "uv_vertices": int(len(uv_vertices)),
                "uv_faces": int(len(uv_faces)),
            },
            "stages": stages,
            "last_trustworthy_evidence": _last_trustworthy_evidence(args.raw_mesh, args.output_dir),
        }
    )
    _write_json(args.output_dir / REPORT_NAME, report)
    print(f"[complete] wrote {output_glb} in {total_elapsed:.3f}s", flush=True)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-faces", type=int, default=350000)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    args = parser.parse_args(argv)
    if args.target_faces <= 0:
        parser.error("--target-faces must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_probe(args)
        return 0
    except ProbeError as exc:
        report = _failure_report(args, exc.phase, str(exc))
        _write_json(args.output_dir / REPORT_NAME, report)
        print(f"{exc.phase}: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        report = _failure_report(args, "unexpected_error", repr(exc))
        _write_json(args.output_dir / REPORT_NAME, report)
        print(f"unexpected_error: {exc!r}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
