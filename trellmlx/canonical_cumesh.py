"""Deterministic simplification controls shared by CUDA and Metal witnesses."""

from __future__ import annotations

import hashlib
from typing import Any, Callable

import numpy as np


def _ordered_array_digest(array: Any) -> tuple[np.ndarray, str]:
    if hasattr(array, "detach"):
        array = array.detach()
    if hasattr(array, "cpu"):
        array = array.cpu()
    if hasattr(array, "numpy"):
        array = array.numpy()
    normalized = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256(normalized.tobytes()).hexdigest()
    return normalized, digest


def mesh_state_digest_observer(
    mesh: Any,
    step: dict[str, object],
) -> dict[str, object]:
    vertices, faces = mesh.read()
    vertices, vertices_sha256 = _ordered_array_digest(vertices)
    faces, faces_sha256 = _ordered_array_digest(faces)
    if vertices.shape != (int(step["output_vertices"]), 3):
        raise RuntimeError(
            "post-step vertex readback shape does not match simplifier result"
        )
    if faces.shape != (int(step["output_faces"]), 3):
        raise RuntimeError(
            "post-step face readback shape does not match simplifier result"
        )
    return {
        "vertices_shape": list(vertices.shape),
        "vertices_dtype": str(vertices.dtype),
        "vertices_sha256": vertices_sha256,
        "faces_shape": list(faces.shape),
        "faces_dtype": str(faces.dtype),
        "faces_sha256": faces_sha256,
    }


def _as_int(value) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    if callable(value):
        return int(value())
    return int(value)


def _mesh_face_count(mesh) -> int:
    return _as_int(getattr(mesh, "num_faces"))


def simplify_with_canonical_adjacency_step_loop(
    mesh,
    target_faces: int,
    *,
    lambda_edge_length: float = 1e-2,
    lambda_skinny: float = 1e-3,
    thresh: float = 1e-8,
    rsqrt_lut=None,
    step_observer: Callable[[Any, dict[str, object]], dict[str, object]]
    | None = None,
    max_steps: int | None = None,
) -> list[dict[str, object]]:
    """Simplify while sorting the exact adjacency consumed by every step."""
    if max_steps is not None and int(max_steps) <= 0:
        raise ValueError("max_steps must be positive when provided")

    step_backend = getattr(mesh, "cu_mesh", mesh)
    required = (
        "get_vertex_face_adjacency",
        "sort_vertex_face_adjacency",
    )
    missing = [name for name in required if not hasattr(step_backend, name)]
    step_method = (
        "simplify_step_turing" if rsqrt_lut is not None else "simplify_step"
    )
    if not hasattr(step_backend, step_method):
        missing.append(step_method)
    if missing:
        raise RuntimeError(
            "canonical adjacency simplification backend is missing: "
            + ", ".join(missing)
        )

    step_trace: list[dict[str, object]] = []
    if _mesh_face_count(mesh) <= int(target_faces):
        return step_trace

    iteration = 0
    while True:
        before = _mesh_face_count(mesh)
        step_backend.get_vertex_face_adjacency()
        step_backend.sort_vertex_face_adjacency()
        args = (
            float(lambda_edge_length),
            float(lambda_skinny),
            float(thresh),
            False,
            True,
        )
        if rsqrt_lut is None:
            new_num_vert, new_num_face = step_backend.simplify_step(*args)
        else:
            new_num_vert, new_num_face = step_backend.simplify_step_turing(
                rsqrt_lut,
                *args,
            )
        iteration += 1
        new_num_vert = _as_int(new_num_vert)
        new_num_face = _as_int(new_num_face)
        removed = before - new_num_face
        step_record: dict[str, object] = {
            "iteration": iteration,
            "threshold": float(thresh),
            "input_faces": int(before),
            "output_faces": int(new_num_face),
            "output_vertices": int(new_num_vert),
            "removed_faces": int(removed),
            "adjacency_order": "ascending-face-id-per-vertex",
        }
        if step_observer is not None:
            observation = step_observer(mesh, dict(step_record))
            if not isinstance(observation, dict):
                raise RuntimeError(
                    "canonical simplification step observer must return a dict"
                )
            step_record["observation"] = observation
        step_trace.append(step_record)
        if max_steps is not None and iteration >= int(max_steps):
            break
        if new_num_face <= int(target_faces):
            break
        if removed / max(before, 1) < 1e-2:
            thresh *= 10
    return step_trace
