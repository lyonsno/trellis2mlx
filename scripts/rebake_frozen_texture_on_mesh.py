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
import uuid
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def apply_face_reversal_manifest(
    faces: np.ndarray,
    *,
    manifest_path: Path | None,
    source_mesh_sha256: str,
    failure_evidence: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Reverse an explicit source-bound face selection before UV unwrap."""
    prepared = np.array(faces, copy=True)
    if manifest_path is None:
        return prepared, {
            "requested": False,
            "applied": False,
            "manifest_path": None,
            "manifest_sha256": None,
            "manifest_schema": None,
            "semantic_name": None,
            "source_mesh_sha256": source_mesh_sha256,
            "source_mesh_faces": int(len(faces)),
            "reversed_faces": 0,
        }

    manifest_path = manifest_path.expanduser().resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = sha256_bytes(manifest_bytes)
    if failure_evidence is not None:
        failure_evidence.update(
            {
                "requested": True,
                "applied": False,
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest_sha256,
                "source_mesh_sha256": source_mesh_sha256,
                "source_mesh_faces": int(len(faces)),
                "reversed_faces": 0,
            }
        )
    try:
        payload = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid face reversal manifest JSON: {manifest_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("face reversal manifest must contain a JSON object")

    schema = payload.get("schema")
    expected_schema = "trellis2mlx.face-reversal-manifest.v1"
    if schema != expected_schema:
        raise ValueError(
            f"unsupported face reversal manifest schema: {schema!r}"
        )

    semantic_name = payload.get("semantic_name")
    if not isinstance(semantic_name, str) or not semantic_name.strip():
        raise ValueError("semantic_name must be a non-empty string")

    source_mesh = payload.get("source_mesh")
    if not isinstance(source_mesh, dict):
        raise ValueError("source_mesh must be an object")
    manifest_mesh_sha256 = source_mesh.get("sha256")
    if manifest_mesh_sha256 != source_mesh_sha256:
        raise ValueError(
            "source mesh SHA256 mismatch for face reversal manifest: "
            f"expected {manifest_mesh_sha256}, got {source_mesh_sha256}"
        )
    manifest_face_count = source_mesh.get("faces")
    if (
        isinstance(manifest_face_count, bool)
        or not isinstance(manifest_face_count, int)
        or manifest_face_count != len(faces)
    ):
        raise ValueError(
            "source mesh face count mismatch for face reversal manifest: "
            f"expected {manifest_face_count}, got {len(faces)}"
        )

    face_indices = payload.get("face_indices")
    if not isinstance(face_indices, list) or not face_indices:
        raise ValueError("face_indices must be a non-empty list")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in face_indices):
        raise ValueError("face_indices must contain integers")
    if len(set(face_indices)) != len(face_indices):
        raise ValueError("face_indices must be unique")
    if min(face_indices) < 0 or max(face_indices) >= len(faces):
        raise ValueError(
            f"face index out of range for source mesh with {len(faces)} faces"
        )

    selected = np.asarray(face_indices, dtype=np.int64)
    prepared[selected] = prepared[selected][:, [0, 2, 1]]
    return prepared.astype(faces.dtype, copy=False), {
        "requested": True,
        "applied": True,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "manifest_schema": schema,
        "semantic_name": semantic_name,
        "source_mesh_sha256": source_mesh_sha256,
        "source_mesh_faces": int(len(faces)),
        "reversed_faces": int(len(face_indices)),
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
        "--face-reversal-manifest",
        type=Path,
        help=(
            "Reverse the exact source face indices named by a manifest whose "
            "mesh SHA256 and face count match the loaded mesh."
        ),
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
    face_reversal_manifest: Path | None = None,
    output_path: Path,
    report_path: Path,
    overwrite: bool,
) -> None:
    protected = {mesh_path, texture_npz, texture_json}
    if face_reversal_manifest is not None:
        protected.add(face_reversal_manifest)
    output_temporary = output_path.with_name(output_path.name + ".tmp")
    report_temporary = report_path.with_name(report_path.name + ".tmp")
    write_paths = {
        output_path,
        report_path,
        output_temporary,
        report_temporary,
    }
    if write_paths & protected:
        raise ValueError("output or report aliases a protected input")
    if len(write_paths) != 4:
        raise ValueError("output/report final or staging write paths collide")
    if output_path.suffix.lower() != ".glb":
        raise ValueError(f"output must use .glb extension: {output_path}")
    existing = [path for path in (output_path, report_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"output paths already exist: {existing}; pass --overwrite"
        )


def open_exclusive_sibling(
    path: Path,
    *,
    prefix: str,
    suffix: str,
) -> tuple[int, Path]:
    """Create an exclusive sibling with ordinary 0666-and-umask permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    while True:
        candidate = path.parent / f"{prefix}{uuid.uuid4().hex}{suffix}"
        try:
            descriptor = os.open(candidate, flags, 0o666)
        except FileExistsError:
            continue
        return descriptor, candidate


def write_json_atomically(path: Path, payload: dict) -> None:
    """Publish JSON atomically with ordinary creation-mode semantics."""
    descriptor, temporary_path = open_exclusive_sibling(
        path,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_bytes_atomically(path: Path, payload: bytes) -> None:
    """Publish bytes atomically with ordinary creation-mode semantics."""
    descriptor, temporary_path = open_exclusive_sibling(
        path,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def publish_success_report(path: Path, payload: dict) -> None:
    """Publish the success report without caller-precreatable staging names."""
    write_json_atomically(path, payload)


def write_json_exclusively_beside(path: Path, payload: dict) -> Path:
    """Atomically expose failure JSON at a distinct no-replace coordinate."""
    descriptor, temporary_path = open_exclusive_sibling(
        path,
        prefix=f".{path.stem}.failure-",
        suffix=".tmp",
    )
    token = temporary_path.name.removeprefix(
        f".{path.stem}.failure-"
    ).removesuffix(".tmp")
    exclusive_path = path.parent / f"{path.stem}.failure-{token}.json"
    try:
        payload["publication"]["effective_failure_report_path"] = str(
            exclusive_path.resolve()
        )
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, exclusive_path)
        return exclusive_path
    finally:
        temporary_path.unlink(missing_ok=True)


def make_failure_state(args: argparse.Namespace) -> dict:
    mesh_path = args.mesh.expanduser().resolve()
    checkpoint_dir = args.texture_checkpoint_dir.expanduser().resolve()
    manifest_path = (
        args.face_reversal_manifest.expanduser().resolve()
        if args.face_reversal_manifest is not None
        else None
    )
    output_path = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    return {
        "phase": "initialization",
        "started": time.time(),
        "requested": {
            "mesh_path": str(mesh_path),
            "texture_checkpoint_dir": str(checkpoint_dir),
            "output_path": str(output_path),
            "report_path": str(report_path),
            "route": {
                "mesh_cleanup": False,
                "mesh_simplification": False,
                "orient_connected_components_outward": bool(
                    args.orient_connected_components_outward
                ),
                "orientation_confidence": float(args.orientation_confidence),
                "face_reversal_manifest": (
                    str(manifest_path) if manifest_path is not None else None
                ),
                "uv_method": args.uv_method,
                "xatlas_fix_winding": bool(args.xatlas_fix_winding),
                "texture_backend": args.texture_backend,
                "texture_size": args.texture_size,
            },
        },
        "effective": {
            "implementation": None,
            "mesh": None,
            "texture_checkpoint": {
                "directory": str(checkpoint_dir),
                "texture_npz_sha256": None,
                "texture_json_sha256": None,
            },
            "face_reversal": {
                "requested": manifest_path is not None,
                "applied": False,
                "manifest_path": (
                    str(manifest_path) if manifest_path is not None else None
                ),
                "manifest_sha256": None,
                "source_mesh_sha256": None,
                "source_mesh_faces": None,
                "reversed_faces": 0,
            },
        },
        "primary_output": {
            "path": str(output_path),
            "existed_before": output_path.exists(),
            "produced_by_attempt": False,
        },
    }


def write_failure_report(
    *,
    args: argparse.Namespace,
    state: dict,
    error: Exception,
) -> Path:
    report_path = args.report.expanduser().resolve()
    protected = {
        args.mesh.expanduser().resolve(),
        (args.texture_checkpoint_dir.expanduser().resolve() / "texture.npz"),
        (args.texture_checkpoint_dir.expanduser().resolve() / "texture.json"),
    }
    if args.face_reversal_manifest is not None:
        protected.add(args.face_reversal_manifest.expanduser().resolve())
    if report_path in protected or report_path == args.output.expanduser().resolve():
        raise ValueError("failure report path aliases a protected input or output")

    primary_output = dict(state["primary_output"])
    primary_output["exists_after"] = Path(primary_output["path"]).exists()
    requested_report_preexisting = report_path.exists()
    no_clobber = requested_report_preexisting and not args.overwrite
    report = {
        "schema": "trellis2mlx.frozen-texture-rebake.failure.v1",
        "status": "failed",
        "failure": {
            "phase": state["phase"],
            "type": type(error).__name__,
            "message": str(error),
        },
        "requested": state["requested"],
        "effective": state["effective"],
        "primary_output": primary_output,
        "publication": {
            "overwrite_authorized": bool(args.overwrite),
            "requested_report_path": str(report_path),
            "requested_report_preexisting": requested_report_preexisting,
            "effective_failure_report_path": None,
            "mode": (
                "exclusive_sibling_no_clobber"
                if no_clobber
                else "requested_report_atomic_replace"
            ),
        },
        "timing_seconds": {"total": time.time() - state["started"]},
        "pid": os.getpid(),
    }
    if no_clobber:
        return write_json_exclusively_beside(report_path, report)
    report["publication"]["effective_failure_report_path"] = str(report_path)
    write_json_atomically(report_path, report)
    return report_path


def run(args: argparse.Namespace, state: dict) -> None:
    started = time.time()
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    state["phase"] = "implementation_identity"
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
    state["effective"]["implementation"] = implementation_identity

    mesh_path = args.mesh.expanduser().resolve()
    checkpoint_dir = args.texture_checkpoint_dir.expanduser().resolve()
    texture_npz = checkpoint_dir / "texture.npz"
    texture_json = checkpoint_dir / "texture.json"
    face_reversal_manifest = (
        args.face_reversal_manifest.expanduser().resolve()
        if args.face_reversal_manifest is not None
        else None
    )
    output_path = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve()

    state["phase"] = "output_path_validation"
    validate_output_paths(
        mesh_path=mesh_path,
        texture_npz=texture_npz,
        texture_json=texture_json,
        face_reversal_manifest=face_reversal_manifest,
        output_path=output_path,
        report_path=report_path,
        overwrite=args.overwrite,
    )

    state["phase"] = "input_identity"
    mesh_sha256 = require_sha256(mesh_path, args.expected_mesh_sha256)
    texture_npz_sha256 = require_sha256(
        texture_npz, args.expected_texture_npz_sha256
    )
    texture_json_sha256 = require_sha256(
        texture_json, args.expected_texture_json_sha256
    )
    state["effective"]["texture_checkpoint"].update(
        {
            "texture_npz_sha256": texture_npz_sha256,
            "texture_json_sha256": texture_json_sha256,
        }
    )

    import trimesh
    from PIL import Image
    from trimesh.visual.material import PBRMaterial

    state["phase"] = "mesh_load"
    loaded = trimesh.load(mesh_path, process=False)
    state["effective"]["mesh"] = {
        "path": str(mesh_path),
        "sha256": mesh_sha256,
        "carrier_type": type(loaded).__name__,
        "vertices": None,
        "faces": None,
    }
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"expected one Trimesh in {mesh_path}, got {type(loaded)}")
    state["effective"]["mesh"].pop("carrier_type")
    vertices = np.asarray(loaded.vertices)
    faces = np.asarray(loaded.faces)
    state["effective"]["mesh"].update(
        {
            "vertices": int(len(vertices)),
            "faces": int(len(faces)),
        }
    )
    state["phase"] = "connected_component_orientation"
    faces, orientation_receipt = prepare_faces_for_bake(
        vertices,
        faces,
        orient_connected_components=args.orient_connected_components_outward,
        orientation_confidence=args.orientation_confidence,
    )
    state["phase"] = "face_reversal_manifest"
    faces, face_reversal_receipt = apply_face_reversal_manifest(
        faces,
        manifest_path=face_reversal_manifest,
        source_mesh_sha256=mesh_sha256,
        failure_evidence=state["effective"]["face_reversal"],
    )
    state["effective"]["face_reversal"] = face_reversal_receipt

    state["phase"] = "texture_checkpoint_load"
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

    state["phase"] = "uv_unwrap"
    unwrap = select_unwrap(
        args.uv_method,
        xatlas_fix_winding=args.xatlas_fix_winding,
    )
    unwrap_started = time.perf_counter()
    uv_vertices, uv_faces, uvs, vmapping = unwrap(vertices, faces)
    unwrap_seconds = time.perf_counter() - unwrap_started

    state["phase"] = "texture_bake"
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

    state["phase"] = "output_export"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_bytes_atomically(output_path, textured_mesh.export(file_type="glb"))
    state["primary_output"]["produced_by_attempt"] = True

    state["phase"] = "success_report"
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
            "face_reversal_manifest": (
                str(face_reversal_manifest)
                if face_reversal_manifest is not None
                else None
            ),
            "uv_method": args.uv_method,
            "xatlas_fix_winding": bool(args.xatlas_fix_winding),
            "texture_backend": args.texture_backend,
            "texture_size": args.texture_size,
        },
        "connected_component_orientation": orientation_receipt,
        "face_reversal": face_reversal_receipt,
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
    publish_success_report(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def main() -> None:
    args = parse_args()
    state = make_failure_state(args)
    try:
        run(args, state)
    except Exception as error:
        try:
            failure_report_path = write_failure_report(
                args=args,
                state=state,
                error=error,
            )
            error.add_note(f"failure report written to: {failure_report_path}")
        except Exception as report_error:
            error.add_note(f"failure report could not be written: {report_error}")
        raise


if __name__ == "__main__":
    main()
