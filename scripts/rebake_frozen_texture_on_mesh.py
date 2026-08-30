#!/usr/bin/env python3
"""Bake a saved TRELLIS texture field onto an already-finalized mesh.

This assay intentionally bypasses mesh cleanup and simplification. It is for
fixed-product comparisons where inference and the decoded texture field must
remain frozen while only the finalized mesh surface changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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


def select_unwrap(method: str):
    from trellmlx.texture_bake import uv_unwrap, uv_unwrap_cube, uv_unwrap_lscm

    if method == "xatlas":
        return uv_unwrap
    if method == "cube":
        return uv_unwrap_cube
    if method == "lscm":
        return uv_unwrap_lscm
    raise ValueError(f"unsupported UV method: {method}")


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

    unwrap = select_unwrap(args.uv_method)
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
            "uv_method": args.uv_method,
            "texture_backend": args.texture_backend,
            "texture_size": args.texture_size,
        },
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
