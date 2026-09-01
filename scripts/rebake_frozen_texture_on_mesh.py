#!/usr/bin/env python3
"""Bake a saved TRELLIS texture field onto an already-finalized mesh.

This assay intentionally bypasses mesh cleanup and simplification. It is for
fixed-product comparisons where inference and the decoded texture field must
remain frozen while only the finalized mesh surface changes.
"""

from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str | None) -> str:
    actual = sha256_file(path)
    if expected is not None and actual != expected:
        raise ValueError(
            f"SHA256 mismatch for {path}: expected {expected}, got {actual}"
        )
    return actual


def snapshot_implementation_identity(
    *,
    repo_root: Path,
    files: tuple[Path, ...],
    git_head: str | None = None,
) -> dict:
    """Bind the run to local source bytes before output-producing work begins."""
    repo_root = repo_root.resolve()
    resolved_files = tuple(path.resolve() for path in files)
    if git_head is None:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    file_records = []
    for path in resolved_files:
        try:
            recorded_path = str(path.relative_to(repo_root))
        except ValueError:
            recorded_path = str(path)
        file_records.append(
            {
                "path": recorded_path,
                "sha256": sha256_file(path),
            }
        )

    status = subprocess.run(
        [
            "git",
            "status",
            "--short",
            "--untracked-files=all",
            "--",
            *(record["path"] for record in file_records),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines() if (repo_root / ".git").exists() else []

    manifest = {
        "git_head": git_head,
        "git_status": status,
        "files": file_records,
    }
    manifest_bytes = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        **manifest,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "snapshotted_before_execution": True,
    }


def select_unwrap(method: str, *, xatlas_fix_winding: bool = False):
    from trellmlx.texture_bake import uv_unwrap, uv_unwrap_cube, uv_unwrap_lscm

    if method == "xatlas":
        return partial(uv_unwrap, fix_winding=xatlas_fix_winding)
    if xatlas_fix_winding:
        raise ValueError("--xatlas-fix-winding only applies to xatlas UV unwrap")
    if method == "cube":
        return uv_unwrap_cube
    if method == "lscm":
        return uv_unwrap_lscm
    raise ValueError(f"unsupported UV method: {method}")


def orient_connected_components_outward(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    min_confidence: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """Reverse confidently inward connected face sheets without moving them.

    The score is the signed area-vector projection away from the mesh bounding
    box center, accumulated independently for each edge-connected face group.
    Confidence measures how much the signed votes agree rather than cancel.
    """
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("orientation confidence must be between 0 and 1")

    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    oriented = np.array(faces, copy=True)
    if len(oriented) == 0:
        return oriented, {
            "components": 0,
            "flipped_components": 0,
            "flipped_faces": 0,
            "min_confidence": float(min_confidence),
        }

    center = np.asarray(mesh.bounding_box.centroid)
    triangles = np.asarray(mesh.vertices)[oriented]
    area_vectors = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    centroids = triangles.mean(axis=1)
    outward_scores = np.einsum("ij,ij->i", area_vectors, centroids - center)
    components = trimesh.graph.connected_components(
        mesh.face_adjacency,
        nodes=np.arange(len(oriented)),
        min_len=1,
    )

    flipped_components = 0
    flipped_faces = 0
    for component in components:
        score = float(outward_scores[component].sum())
        magnitude = float(np.abs(outward_scores[component]).sum())
        confidence = abs(score) / magnitude if magnitude else 0.0
        if score < 0.0 and confidence >= min_confidence:
            oriented[component] = oriented[component][:, [0, 2, 1]]
            flipped_components += 1
            flipped_faces += int(len(component))

    return oriented.astype(faces.dtype, copy=False), {
        "components": int(len(components)),
        "flipped_components": flipped_components,
        "flipped_faces": flipped_faces,
        "min_confidence": float(min_confidence),
    }


def prepare_faces_for_bake(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    orient_connected_components: bool,
    orientation_confidence: float,
) -> tuple[np.ndarray, dict]:
    """Apply the explicit pre-bake orientation route and attest its effect."""
    confidence = float(orientation_confidence)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("orientation confidence must be between 0 and 1")

    if not orient_connected_components:
        return np.array(faces, copy=True), {
            "requested": False,
            "applied": False,
            "components": None,
            "flipped_components": 0,
            "flipped_faces": 0,
            "min_confidence": confidence,
        }

    prepared, receipt = orient_connected_components_outward(
        vertices,
        faces,
        min_confidence=confidence,
    )
    return prepared, {
        "requested": True,
        "applied": True,
        **receipt,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--texture-checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--mesh-grid-size", required=True, type=int)
    parser.add_argument("--texture-size", type=int, default=512)
    parser.add_argument(
        "--uv-method", choices=("xatlas", "cube", "lscm"), default="xatlas"
    )
    parser.add_argument(
        "--xatlas-fix-winding",
        action="store_true",
        help=(
            "Ask xatlas to account for inconsistent input winding while "
            "generating UV coordinates; valid only with --uv-method xatlas."
        ),
    )
    parser.add_argument(
        "--orient-connected-components-outward",
        action="store_true",
        help=(
            "Before UV unwrap and texture bake, reverse edge-connected face "
            "groups whose signed area vectors confidently point inward."
        ),
    )
    parser.add_argument(
        "--orientation-confidence",
        type=float,
        default=0.5,
        help="Minimum signed-vote confidence for pre-bake component reversal.",
    )
    parser.add_argument(
        "--texture-backend",
        choices=("cpu", "gpu"),
        default="gpu",
        help="Production texture baker backend; 'gpu' selects Metal on MLX.",
    )
    parser.add_argument("--expected-mesh-sha256")
    parser.add_argument("--expected-texture-npz-sha256")
    parser.add_argument("--expected-texture-json-sha256")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_output_paths(
    *,
    mesh_path: Path,
    texture_npz: Path,
    texture_json: Path,
    output_path: Path,
    report_path: Path,
    overwrite: bool,
) -> None:
    protected = {mesh_path, texture_npz, texture_json}
    if output_path in protected or report_path in protected:
        raise ValueError("output or report aliases a protected input")
    if output_path == report_path:
        raise ValueError("output GLB and report JSON must be distinct paths")
    if output_path.suffix.lower() != ".glb":
        raise ValueError(f"output must use .glb extension: {output_path}")
    existing = [path for path in (output_path, report_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"output paths already exist: {existing}; pass --overwrite"
        )


def main() -> None:
    args = parse_args()
    started = time.time()
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    implementation_identity = snapshot_implementation_identity(
        repo_root=repo_root,
        files=(
            script_path,
            repo_root / "trellmlx" / "texture_bake.py",
            repo_root / "trellmlx" / "checkpoint.py",
            repo_root / "pyproject.toml",
            repo_root / "uv.lock",
        ),
    )

    mesh_path = args.mesh.expanduser().resolve()
    checkpoint_dir = args.texture_checkpoint_dir.expanduser().resolve()
    texture_npz = checkpoint_dir / "texture.npz"
    texture_json = checkpoint_dir / "texture.json"
    output_path = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve()

    validate_output_paths(
        mesh_path=mesh_path,
        texture_npz=texture_npz,
        texture_json=texture_json,
        output_path=output_path,
        report_path=report_path,
        overwrite=args.overwrite,
    )

    mesh_sha256 = require_sha256(mesh_path, args.expected_mesh_sha256)
    texture_npz_sha256 = require_sha256(
        texture_npz, args.expected_texture_npz_sha256
    )
    texture_json_sha256 = require_sha256(
        texture_json, args.expected_texture_json_sha256
    )

    import trimesh
    from PIL import Image
    from trimesh.visual.material import PBRMaterial

    loaded = trimesh.load(mesh_path, process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"expected one Trimesh in {mesh_path}, got {type(loaded)}")
    vertices = np.asarray(loaded.vertices)
    faces = np.asarray(loaded.faces)
    faces, orientation_receipt = prepare_faces_for_bake(
        vertices,
        faces,
        orient_connected_components=args.orient_connected_components_outward,
        orientation_confidence=args.orientation_confidence,
    )

    from trellmlx.checkpoint import load_checkpoint

    texture = load_checkpoint(str(checkpoint_dir), "texture")
    tex_np = np.asarray(texture["tex_np"])
    tex_coords_spatial = np.asarray(texture["tex_coords_spatial"])
    checkpoint_grid_size = int(texture["mesh_grid_size"])
    if checkpoint_grid_size != args.mesh_grid_size:
        raise ValueError(
            "texture checkpoint grid size differs from requested mesh grid size: "
            f"{checkpoint_grid_size} != {args.mesh_grid_size}"
        )

    unwrap = select_unwrap(
        args.uv_method,
        xatlas_fix_winding=args.xatlas_fix_winding,
    )
    unwrap_started = time.perf_counter()
    uv_vertices, uv_faces, uvs, vmapping = unwrap(vertices, faces)
    unwrap_seconds = time.perf_counter() - unwrap_started

    from trellmlx.texture_bake import bake_texture

    bake_started = time.perf_counter()
    base_color, metallic_roughness, alpha_mode = bake_texture(
        uv_vertices,
        uv_faces,
        uvs,
        vmapping,
        tex_coords_spatial,
        tex_np,
        args.mesh_grid_size,
        texture_size=args.texture_size,
        backend=args.texture_backend,
    )
    bake_seconds = time.perf_counter() - bake_started

    export_vertices = uv_vertices.copy()
    export_vertices[:, 1], export_vertices[:, 2] = (
        uv_vertices[:, 2].copy(),
        -uv_vertices[:, 1].copy(),
    )
    export_uvs = uvs.copy()
    export_uvs[:, 1] = 1 - export_uvs[:, 1]

    normal_mesh = trimesh.Trimesh(
        vertices=export_vertices, faces=uv_faces, process=False
    )
    material = PBRMaterial(
        baseColorTexture=Image.fromarray(base_color),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(metallic_roughness),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode=alpha_mode,
        doubleSided=True,
    )
    textured_mesh = trimesh.Trimesh(
        vertices=export_vertices,
        faces=uv_faces,
        vertex_normals=normal_mesh.vertex_normals,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=export_uvs, material=material),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_temporary = output_path.with_name(output_path.name + ".tmp")
    output_temporary.write_bytes(textured_mesh.export(file_type="glb"))
    output_temporary.replace(output_path)

    report = {
        "schema": "trellis2mlx.frozen-texture-rebake.v1",
        "implementation": implementation_identity,
        "mesh": {
            "path": str(mesh_path),
            "sha256": mesh_sha256,
            "vertices": int(len(vertices)),
            "faces": int(len(faces)),
        },
        "texture_checkpoint": {
            "directory": str(checkpoint_dir),
            "texture_npz_sha256": texture_npz_sha256,
            "texture_json_sha256": texture_json_sha256,
            "voxels": int(tex_np.shape[0]),
            "channels": int(tex_np.shape[1]),
            "mesh_grid_size": args.mesh_grid_size,
        },
        "route": {
            "mesh_cleanup": False,
            "mesh_simplification": False,
            "orient_connected_components_outward": bool(
                args.orient_connected_components_outward
            ),
            "orientation_confidence": float(args.orientation_confidence),
            "uv_method": args.uv_method,
            "xatlas_fix_winding": bool(args.xatlas_fix_winding),
            "texture_backend": args.texture_backend,
            "texture_size": args.texture_size,
        },
        "connected_component_orientation": orientation_receipt,
        "uv_mesh": {
            "vertices": int(len(uv_vertices)),
            "faces": int(len(uv_faces)),
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "size_bytes": output_path.stat().st_size,
            "alpha_mode": str(alpha_mode),
        },
        "timing_seconds": {
            "unwrap": unwrap_seconds,
            "bake": bake_seconds,
            "total": time.time() - started,
        },
        "pid": os.getpid(),
    }
    report_temporary = report_path.with_name(report_path.name + ".tmp")
    report_temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    report_temporary.replace(report_path)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
