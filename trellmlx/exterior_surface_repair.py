"""Camera-independent repair for one dominant exterior-inverted sheet.

The repair has no asset, component-rank, camera, or face-id input. It finds
coherent sheets whose radial orientation is negative and whose decisive
outside rays reach only the back of the triangles, then acts only when one
candidate has more surface area than every eligible alternative combined.
It changes triangle winding only; it never adds, removes, or welds geometry.
"""

from __future__ import annotations

import math

import igl
import numpy as np
import trimesh


NORMAL_SIDE_ONLY = 1
BACK_SIDE_ONLY = 2
BOTH_SIDES = 3
NEITHER_SIDE = 4


def _topology_counts(faces: np.ndarray) -> dict[str, int]:
    if len(faces) == 0:
        return {
            "boundary_edges": 0,
            "nonmanifold_edges": 0,
            "same_direction_shared_edges": 0,
        }
    directed = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
    )
    undirected = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(
        undirected, axis=0, return_inverse=True, return_counts=True
    )
    direction = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    direction_sum = np.bincount(inverse, weights=direction, minlength=len(counts))
    same_direction = int(
        np.count_nonzero((counts == 2) & (np.abs(direction_sum) == 2))
    )
    return {
        "boundary_edges": int(np.count_nonzero(counts == 1)),
        "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
        "same_direction_shared_edges": same_direction,
    }


def _orient_face_components_outward(
    vertices: np.ndarray, faces: np.ndarray, *, min_confidence: float = 0.5
) -> tuple[np.ndarray, dict]:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    oriented = np.array(faces, copy=True)
    if len(faces) == 0:
        return oriented, {
            "components": 0,
            "flipped_components": 0,
            "flipped_faces": 0,
            "min_confidence": float(min_confidence),
        }
    triangles = vertices[faces]
    area_vectors = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    centroids = triangles.mean(axis=1)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    scores = np.einsum("ij,ij->i", area_vectors, centroids - center)
    components = trimesh.graph.connected_components(
        mesh.face_adjacency, nodes=np.arange(len(faces)), min_len=1
    )
    flipped_components = 0
    flipped_faces = 0
    for component in components:
        score = float(scores[component].sum())
        magnitude = float(np.abs(scores[component]).sum())
        confidence = abs(score) / magnitude if magnitude else 0.0
        if score < 0.0 and confidence >= min_confidence:
            oriented[component] = oriented[component][:, [0, 2, 1]]
            flipped_components += 1
            flipped_faces += int(len(component))
    return oriented, {
        "components": int(len(components)),
        "flipped_components": flipped_components,
        "flipped_faces": flipped_faces,
        "min_confidence": float(min_confidence),
    }


def _exterior_side_classes(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    centroids = triangles.mean(axis=1)
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    magnitudes = np.linalg.norm(normals, axis=1)
    valid = magnitudes > 0.0
    normals[valid] /= magnitudes[valid, None]
    diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    reach = max(diagonal * 2.0, np.finfo(np.float64).eps * 1024.0)

    classes = np.full(len(faces), NEITHER_SIDE, dtype=np.int8)
    if not np.any(valid):
        return classes
    tree = igl.AABB()
    int_faces = faces.astype(np.int64, copy=False)
    tree.init(vertices, int_faces)
    rows = np.flatnonzero(valid)
    normal_hits, _, _ = tree.intersect_ray_first(
        vertices,
        int_faces,
        centroids[rows] + normals[rows] * reach,
        -normals[rows],
        reach * 1.1,
    )
    back_hits, _, _ = tree.intersect_ray_first(
        vertices,
        int_faces,
        centroids[rows] - normals[rows] * reach,
        normals[rows],
        reach * 1.1,
    )
    normal_visible = normal_hits == rows
    back_visible = back_hits == rows
    classes[rows[normal_visible & ~back_visible]] = NORMAL_SIDE_ONLY
    classes[rows[~normal_visible & back_visible]] = BACK_SIDE_ONLY
    classes[rows[normal_visible & back_visible]] = BOTH_SIDES
    return classes


def _negative_sheet_clusters(
    vertices: np.ndarray,
    faces: np.ndarray,
    components: list[np.ndarray],
    face_adjacency: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    triangles = vertices[faces]
    area_vectors = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    centroids = triangles.mean(axis=1)
    negative_mask = np.zeros(len(faces), dtype=bool)
    for component in components:
        component = np.asarray(component, dtype=np.int64)
        component_vertices = np.unique(faces[component].reshape(-1))
        component_points = vertices[component_vertices]
        center = (
            component_points.min(axis=0) + component_points.max(axis=0)
        ) * 0.5
        scores = np.einsum(
            "ij,ij->i",
            area_vectors[component],
            centroids[component] - center,
        )
        negative_mask[component] = scores < 0.0
    negative = np.flatnonzero(negative_mask)
    if len(negative) == 0:
        return [], area_vectors
    edges = face_adjacency[
        negative_mask[face_adjacency[:, 0]]
        & negative_mask[face_adjacency[:, 1]]
    ]
    if len(edges) == 0:
        return (
            [np.asarray([face_id], dtype=np.int64) for face_id in negative],
            area_vectors,
        )
    clusters = trimesh.graph.connected_components(edges, nodes=negative, min_len=1)
    return [np.asarray(cluster, dtype=np.int64) for cluster in clusters], area_vectors


def _find_orientation_patch(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray | None, dict]:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    components = trimesh.graph.connected_components(
        mesh.face_adjacency, nodes=np.arange(len(faces)), min_len=1
    )
    if not components:
        return None, {
            "candidate_count": 0,
            "selection_status": "none",
            "alternative_area": 0.0,
            "candidates": [],
        }
    clusters, area_vectors = _negative_sheet_clusters(
        vertices,
        faces,
        components,
        mesh.face_adjacency,
    )
    areas = 0.5 * np.linalg.norm(area_vectors, axis=1)
    classes = _exterior_side_classes(vertices, faces)
    candidates = []
    for cluster in clusters:
        back = cluster[classes[cluster] == BACK_SIDE_ONLY]
        normal = cluster[classes[cluster] == NORMAL_SIDE_ONLY]
        neither = cluster[classes[cluster] == NEITHER_SIDE]
        both = cluster[classes[cluster] == BOTH_SIDES]
        back_area = float(areas[back].sum())
        normal_area = float(areas[normal].sum())
        neither_area = float(areas[neither].sum())
        both_area = float(areas[both].sum())
        ambiguous_area = neither_area + both_area
        eligible = (
            len(back) > 0
            and normal_area == 0.0
            and back_area > ambiguous_area
        )
        if eligible:
            candidates.append(
                {
                    "faces": cluster,
                    "face_count": int(len(cluster)),
                    "area": float(areas[cluster].sum()),
                    "back_side_faces": int(len(back)),
                    "neither_side_faces": int(len(neither)),
                }
            )
    candidates.sort(key=lambda row: (-row["area"], -row["face_count"]))
    selected = None
    alternative_area = float(sum(row["area"] for row in candidates[1:]))
    if (
        candidates
        and candidates[0]["area"] > alternative_area
        and not np.isclose(
            candidates[0]["area"], alternative_area, rtol=1e-12, atol=0.0
        )
    ):
        selected = candidates[0]["faces"]
    receipt = {
        "candidate_count": len(candidates),
        "selection_status": (
            "selected"
            if selected is not None
            else "ambiguous"
            if candidates
            else "none"
        ),
        "alternative_area": alternative_area,
        "candidates": [
            {key: value for key, value in candidate.items() if key != "faces"}
            for candidate in candidates
        ],
    }
    if selected is not None:
        receipt["selected_face_count"] = int(len(selected))
        receipt["selected_area"] = float(candidates[0]["area"])
    return selected, receipt


def validate_exterior_surface_repair_receipt(receipt: dict) -> None:
    """Reject receipts that cannot describe this winding-only repair."""

    def require_dict(container: dict, name: str) -> dict:
        value = container.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"exterior repair receipt {name} must be an object")
        return value

    def require_int(container: dict, name: str, *, minimum: int = 0) -> int:
        value = container.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(
                f"exterior repair receipt {name} must be an integer >= {minimum}"
            )
        return value

    def require_number(
        container: dict, name: str, *, minimum: float = 0.0
    ) -> float:
        value = container.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < minimum
        ):
            raise ValueError(
                f"exterior repair receipt {name} must be finite and >= {minimum}"
            )
        return float(value)

    if not isinstance(receipt, dict):
        raise ValueError("exterior repair receipt must be an object")
    if receipt.get("schema") != "trellis2mlx.exterior_surface_repair.v1":
        raise ValueError("exterior repair receipt schema is missing or unsupported")

    input_mesh = require_dict(receipt, "input_mesh")
    output_mesh = require_dict(receipt, "output_mesh")
    input_vertices = require_int(input_mesh, "vertices")
    input_faces = require_int(input_mesh, "faces")
    output_vertices = require_int(output_mesh, "vertices")
    output_faces = require_int(output_mesh, "faces")
    if (input_vertices, input_faces) != (output_vertices, output_faces):
        raise ValueError(
            "exterior repair receipt violates winding-only mesh cardinality"
        )
    for mesh in (input_mesh, output_mesh):
        topology = require_dict(mesh, "topology")
        for field in (
            "boundary_edges",
            "nonmanifold_edges",
            "same_direction_shared_edges",
        ):
            require_int(topology, field)

    component = require_dict(receipt, "component_orientation")
    component_count = require_int(component, "components")
    flipped_components = require_int(component, "flipped_components")
    flipped_faces = require_int(component, "flipped_faces")
    confidence = require_number(component, "min_confidence")
    if confidence > 1.0:
        raise ValueError("exterior repair receipt min_confidence must be <= 1")
    if flipped_components > component_count or flipped_faces > input_faces:
        raise ValueError("exterior repair receipt component counts are impossible")

    orientation = require_dict(receipt, "orientation")
    candidate_count = require_int(orientation, "candidate_count")
    status = orientation.get("selection_status")
    if status not in {"selected", "ambiguous", "none"}:
        raise ValueError("exterior repair receipt selection_status is invalid")
    require_number(orientation, "alternative_area")
    reversed_faces = require_int(orientation, "reversed_faces")
    candidates = orientation.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != candidate_count:
        raise ValueError("exterior repair receipt candidates do not match count")
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("exterior repair receipt candidate must be an object")
        face_count = require_int(candidate, "face_count", minimum=1)
        require_number(candidate, "area", minimum=np.finfo(np.float64).tiny)
        back_faces = require_int(candidate, "back_side_faces")
        neither_faces = require_int(candidate, "neither_side_faces")
        if back_faces + neither_faces > face_count:
            raise ValueError("exterior repair receipt candidate counts are impossible")

    has_selected_fields = (
        "selected_face_count" in orientation or "selected_area" in orientation
    )
    if status == "selected":
        if candidate_count == 0:
            raise ValueError("exterior repair receipt selected without a candidate")
        selected_faces = require_int(
            orientation, "selected_face_count", minimum=1
        )
        selected_area = require_number(
            orientation,
            "selected_area",
            minimum=np.finfo(np.float64).tiny,
        )
        if (
            reversed_faces != selected_faces
            or selected_faces != candidates[0]["face_count"]
            or selected_area != float(candidates[0]["area"])
        ):
            raise ValueError("exterior repair receipt selected counts disagree")
    else:
        if reversed_faces != 0 or has_selected_fields:
            raise ValueError(
                "exterior repair receipt refusal cannot reverse or select faces"
            )
        if status == "none" and candidate_count != 0:
            raise ValueError("exterior repair receipt none status has candidates")
        if status == "ambiguous" and candidate_count == 0:
            raise ValueError("exterior repair receipt ambiguous status lacks candidates")


def repair_exterior_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    component_orientation_confidence: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """Reverse one uniquely dominant exterior-inverted sheet, if present."""
    vertices = np.asarray(vertices, dtype=np.float64)
    source_faces = np.asarray(faces)
    if not np.issubdtype(source_faces.dtype, np.integer):
        raise ValueError("faces must use an integer dtype")
    source_face_dtype = source_faces.dtype
    faces = np.asarray(source_faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3)")
    if not np.isfinite(vertices).all():
        raise ValueError("vertices must be finite")
    if not (
        np.isfinite(component_orientation_confidence)
        and 0.0 <= component_orientation_confidence <= 1.0
    ):
        raise ValueError("component orientation confidence must be between 0 and 1")

    input_topology = _topology_counts(faces)
    prepared, component_receipt = _orient_face_components_outward(
        vertices, faces, min_confidence=component_orientation_confidence
    )
    patch, orientation_receipt = _find_orientation_patch(vertices, prepared)
    reversed_faces = 0
    if patch is not None:
        prepared[patch] = prepared[patch][:, [0, 2, 1]]
        reversed_faces = int(len(patch))
    output_topology = _topology_counts(prepared)
    receipt = {
        "schema": "trellis2mlx.exterior_surface_repair.v1",
        "input_mesh": {
            "vertices": int(len(vertices)),
            "faces": int(len(faces)),
            "topology": input_topology,
        },
        "output_mesh": {
            "vertices": int(len(vertices)),
            "faces": int(len(prepared)),
            "topology": output_topology,
        },
        "component_orientation": component_receipt,
        "orientation": {
            **orientation_receipt,
            "reversed_faces": reversed_faces,
        },
    }
    validate_exterior_surface_repair_receipt(receipt)
    return prepared.astype(source_face_dtype, copy=False), receipt
