"""Optional source-native mtlmesh simplification route."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from trellmlx.source_route_identity import (
    SourceRouteIdentityError,
    probe_cumesh_route_identity,
    validate_source_route_identity,
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
) -> tuple[np.ndarray, np.ndarray]:
    """Simplify with the reference mtlmesh/cumesh backend when installed.

    This is intentionally not a fallback path: callers select it when they want
    source-native QEM behavior rather than the local MLX parity probe.
    """
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
