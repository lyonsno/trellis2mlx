"""Optional source-native mtlmesh simplification route."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from trellmlx.source_route_identity import (
    SourceRouteIdentityError,
    probe_cumesh_route_identity,
    validate_source_route_identity,
)


def _external_source_native_code() -> str:
    return r'''
import sys

import numpy as np

from trellmlx.source_mtlmesh import simplify_source_native


verbose = sys.argv[1] == "1"
input_npz, output_npz, target_faces_text, expected_root = sys.argv[2:6]
data = np.load(input_npz)
vertices, faces = simplify_source_native(
    np.asarray(data["vertices"], dtype=np.float32),
    np.asarray(data["faces"], dtype=np.int32),
    int(target_faces_text),
    verbose=verbose,
    expected_source_root=expected_root or None,
)
np.savez_compressed(output_npz, vertices=vertices, faces=faces)
'''


def _run_source_native_subprocess(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
    *,
    reference_python: str | Path,
    verbose: bool,
    expected_source_root: str | Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(repo_root)
        if not existing_pythonpath
        else str(repo_root) + os.pathsep + existing_pythonpath
    )

    with tempfile.TemporaryDirectory(prefix="trellis2mlx-source-native-") as tmp:
        tmp_path = Path(tmp)
        input_npz = tmp_path / "input_mesh.npz"
        output_npz = tmp_path / "output_mesh.npz"
        np.savez_compressed(
            input_npz,
            vertices=np.asarray(vertices, dtype=np.float32),
            faces=np.asarray(faces, dtype=np.int32),
        )
        cmd = [
            str(reference_python),
            "-c",
            _external_source_native_code(),
            "1" if verbose else "0",
            str(input_npz),
            str(output_npz),
            str(int(target_faces)),
            str(expected_source_root or ""),
        ]
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if completed.returncode != 0:
            detail = completed.stderr or completed.stdout or "no subprocess output"
            raise RuntimeError(
                f"qem_backend='source-native' reference_python {reference_python} "
                f"failed with exit code {completed.returncode}: {detail}"
            )
        if not output_npz.exists():
            raise RuntimeError(
                f"qem_backend='source-native' reference_python {reference_python} "
                "completed without writing output mesh"
            )
        output = np.load(output_npz)
        return (
            np.asarray(output["vertices"], dtype=np.float32),
            np.asarray(output["faces"], dtype=np.int32),
        )


def _load_source_mesh_class(*, expected_source_root: str | Path | None = None):
    try:
        cumesh = importlib.import_module("cumesh")
    except ImportError as exc:
        raise RuntimeError(
            "qem_backend='source-native' requires the reference mtlmesh/cumesh "
            "package on PYTHONPATH"
        ) from exc

    mesh_cls = getattr(cumesh, "CuMesh", None)
    if mesh_cls is not None:
        try:
            validate_source_route_identity(
                probe_cumesh_route_identity(),
                expected_root=expected_source_root,
            )
        except SourceRouteIdentityError as exc:
            raise RuntimeError(f"qem_backend='source-native' rejected route: {exc}") from exc
        return mesh_cls

    try:
        metal_backend = importlib.import_module("cumesh.metal_backend")
    except ImportError as exc:
        raise RuntimeError(
            "qem_backend='source-native' found cumesh but not cumesh.metal_backend"
        ) from exc

    mesh_cls = getattr(metal_backend, "MtlMesh", None)
    if mesh_cls is None:
        raise RuntimeError(
            "qem_backend='source-native' requires cumesh.CuMesh or "
            "cumesh.metal_backend.MtlMesh"
        )
    try:
        validate_source_route_identity(
            probe_cumesh_route_identity(),
            expected_root=expected_source_root,
        )
    except SourceRouteIdentityError as exc:
        raise RuntimeError(f"qem_backend='source-native' rejected route: {exc}") from exc
    return mesh_cls


def simplify_source_native(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
    *,
    verbose: bool = True,
    lambda_edge_length: float = 1e-2,
    lambda_skinny: float = 1e-3,
    thresh: float = 1e-8,
    expected_source_root: str | Path | None = None,
    reference_python: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simplify with the reference mtlmesh/cumesh backend when installed.

    This is intentionally not a fallback path: callers select it when they want
    source-native QEM behavior rather than the local MLX parity probe.
    """
    if reference_python is not None:
        return _run_source_native_subprocess(
            vertices,
            faces,
            target_faces,
            reference_python=reference_python,
            verbose=verbose,
            expected_source_root=expected_source_root,
        )

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "qem_backend='source-native' requires torch for the reference "
            "mtlmesh/cumesh backend"
        ) from exc

    mesh_cls = _load_source_mesh_class(expected_source_root=expected_source_root)
    mesh = mesh_cls()
    verts_t = torch.from_numpy(np.asarray(vertices, dtype=np.float32)).contiguous()
    faces_t = torch.from_numpy(np.asarray(faces, dtype=np.int32)).contiguous()
    mesh.init(verts_t, faces_t)

    options = {
        "lambda_edge_length": float(lambda_edge_length),
        "lambda_skinny": float(lambda_skinny),
        "thresh": float(thresh),
    }
    try:
        mesh.simplify(int(target_faces), verbose=verbose, options=options)
    except TypeError:
        mesh.simplify(int(target_faces), verbose=verbose)

    out_vertices, out_faces = mesh.read()
    if hasattr(out_vertices, "detach"):
        out_vertices = out_vertices.detach().cpu().numpy()
    if hasattr(out_faces, "detach"):
        out_faces = out_faces.detach().cpu().numpy()
    return (
        np.asarray(out_vertices, dtype=np.float32),
        np.asarray(out_faces, dtype=np.int32),
    )
