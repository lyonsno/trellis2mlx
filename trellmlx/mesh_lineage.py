"""Stage-aware mesh lineage and topology analysis helpers.

Exact lineage is only claimed across operations that preserve triangle geometry.
Topology-changing simplifiers are compared geometrically and reported as such.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree


def _mesh_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape [V, 3], got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape [F, 3], got {faces.shape}")
    if faces.size and (int(faces.min()) < 0 or int(faces.max()) >= len(vertices)):
        raise ValueError("faces contain out-of-range vertex indices")
    return vertices, faces


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values)
    if values.size == 0:
        return {key: None for key in ("min", "p50", "p90", "p95", "p99", "max")}
    result = np.quantile(values, [0.0, 0.5, 0.9, 0.95, 0.99, 1.0])
    return {
        key: float(value)
        for key, value in zip(("min", "p50", "p90", "p95", "p99", "max"), result)
    }


def _face_geometry(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangles = np.asarray(vertices[faces], dtype=np.float64)
    centroids = triangles.mean(axis=1)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    normals = np.zeros_like(cross)
    valid = double_area > 0
    normals[valid] = cross[valid] / double_area[valid, None]
    return centroids, normals, double_area * 0.5


def _edge_groups(faces: np.ndarray) -> dict[str, np.ndarray]:
    face_count = len(faces)
    directed = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]),
        axis=0,
    ).astype(np.int64, copy=False)
    face_indices = np.tile(np.arange(face_count, dtype=np.int64), 3)
    undirected = np.sort(directed, axis=1)
    order = np.lexsort((undirected[:, 1], undirected[:, 0]))
    sorted_edges = undirected[order]
    sorted_directed = directed[order]
    sorted_faces = face_indices[order]
    if len(sorted_edges) == 0:
        empty = np.empty(0, dtype=np.int64)
        return {
            "starts": empty,
            "counts": empty,
            "directed": sorted_directed,
            "faces": sorted_faces,
        }
    starts_mask = np.empty(len(sorted_edges), dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = np.any(sorted_edges[1:] != sorted_edges[:-1], axis=1)
    starts = np.flatnonzero(starts_mask)
    ends = np.concatenate((starts[1:], np.array([len(sorted_edges)], dtype=np.int64)))
    return {
        "starts": starts,
        "counts": ends - starts,
        "directed": sorted_directed,
        "faces": sorted_faces,
    }


def mesh_topology_summary(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    """Return geometry, edge-incidence, and local winding scalars."""
    vertices, faces = _mesh_arrays(vertices, faces)
    centroids, _normals, areas = _face_geometry(vertices, faces)
    edge_groups = _edge_groups(faces)
    starts = edge_groups["starts"]
    counts = edge_groups["counts"]
    manifold_starts = starts[counts == 2]
    directed = edge_groups["directed"]
    same_direction = np.zeros(len(manifold_starts), dtype=bool)
    if len(manifold_starts):
        same_direction = np.all(
            directed[manifold_starts] == directed[manifold_starts + 1], axis=1
        )

    bounds_min = vertices.min(axis=0) if len(vertices) else np.zeros(3)
    bounds_max = vertices.max(axis=0) if len(vertices) else np.zeros(3)
    diagonal = float(np.linalg.norm(bounds_max.astype(np.float64) - bounds_min))
    area_epsilon = max(diagonal * diagonal * 1e-16, np.finfo(np.float64).tiny)
    return {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "bounds_min": [float(value) for value in bounds_min],
        "bounds_max": [float(value) for value in bounds_max],
        "bounds_diagonal": diagonal,
        "surface_area": float(areas.sum()),
        "face_area": _quantiles(areas),
        "degenerate_faces": int((areas <= area_epsilon).sum()),
        "centroid_finite": bool(np.isfinite(centroids).all()),
        "unique_edges": int(len(starts)),
        "boundary_edges": int((counts == 1).sum()),
        "manifold_edges": int((counts == 2).sum()),
        "nonmanifold_edges": int((counts > 2).sum()),
        "same_direction_manifold_conflicts": int(same_direction.sum()),
        "opposite_direction_manifold_edges": int((~same_direction).sum()),
    }


def _canonical_face_keys(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    triangles = np.asarray(vertices[faces], dtype=np.float64)
    triangles[triangles == 0] = 0.0
    order = np.lexsort(
        (triangles[:, :, 2], triangles[:, :, 1], triangles[:, :, 0]),
        axis=1,
    )
    canonical = np.take_along_axis(triangles, order[:, :, None], axis=1)
    canonical = np.ascontiguousarray(canonical.reshape(len(canonical), 9))
    key_dtype = np.dtype((np.void, canonical.dtype.itemsize * canonical.shape[1]))
    return canonical.view(key_dtype).reshape(-1), triangles


def _rowwise_equal(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.all(left == right, axis=(1, 2))


def exact_face_transition(
    before_vertices: np.ndarray,
    before_faces: np.ndarray,
    after_vertices: np.ndarray,
    after_faces: np.ndarray,
) -> dict[str, Any]:
    """Compare exact coordinate triangles and classify orientation changes.

    Vertex indices may be compacted or split. Duplicate coordinate-triangle
    keys are counted but excluded from orientation classification.
    """
    before_vertices, before_faces = _mesh_arrays(before_vertices, before_faces)
    after_vertices, after_faces = _mesh_arrays(after_vertices, after_faces)
    before_keys, before_triangles = _canonical_face_keys(before_vertices, before_faces)
    after_keys, after_triangles = _canonical_face_keys(after_vertices, after_faces)
    before_unique, before_first, before_counts = np.unique(
        before_keys, return_index=True, return_counts=True
    )
    after_unique, after_first, after_counts = np.unique(
        after_keys, return_index=True, return_counts=True
    )
    common, before_common, after_common = np.intersect1d(
        before_unique, after_unique, assume_unique=True, return_indices=True
    )
    unique_pairs = (before_counts[before_common] == 1) & (after_counts[after_common] == 1)
    before_rows = before_triangles[before_first[before_common[unique_pairs]]]
    after_rows = after_triangles[after_first[after_common[unique_pairs]]]

    same = np.zeros(len(before_rows), dtype=bool)
    reversed_rows = np.zeros(len(before_rows), dtype=bool)
    for shift in range(3):
        same |= _rowwise_equal(before_rows, np.roll(after_rows, shift, axis=1))
        reversed_rows |= _rowwise_equal(
            before_rows,
            np.roll(after_rows[:, [0, 2, 1]], shift, axis=1),
        )

    return {
        "claim": "exact-coordinate-triangle-identity",
        "before_faces": int(len(before_faces)),
        "after_faces": int(len(after_faces)),
        "before_unique_face_keys": int(len(before_unique)),
        "after_unique_face_keys": int(len(after_unique)),
        "common_unique_face_keys": int(len(common)),
        "before_only_unique_face_keys": int(len(before_unique) - len(common)),
        "after_only_unique_face_keys": int(len(after_unique) - len(common)),
        "before_duplicate_face_rows": int((before_counts - 1).clip(min=0).sum()),
        "after_duplicate_face_rows": int((after_counts - 1).clip(min=0).sum()),
        "orientation_classifiable_unique_pairs": int(len(before_rows)),
        "same_orientation": int(same.sum()),
        "reversed_orientation": int((reversed_rows & ~same).sum()),
        "orientation_ambiguous": int((same & reversed_rows).sum()),
        "orientation_unmatched": int((~(same | reversed_rows)).sum()),
    }


def approximate_surface_transition(
    before_vertices: np.ndarray,
    before_faces: np.ndarray,
    after_vertices: np.ndarray,
    after_faces: np.ndarray,
    *,
    max_faces_per_side: int = 500_000,
) -> dict[str, Any]:
    """Compare topology-changing stages using nearest face centroids.

    This is a bounded geometric witness, not an ancestry claim. Distances are
    normalized by the union bounding-box diagonal; normal agreement is sampled
    at the nearest face centroid in each direction.
    """
    before_vertices, before_faces = _mesh_arrays(before_vertices, before_faces)
    after_vertices, after_faces = _mesh_arrays(after_vertices, after_faces)
    if max_faces_per_side <= 0:
        raise ValueError("max_faces_per_side must be positive")

    def sample(face_rows: np.ndarray) -> np.ndarray:
        if len(face_rows) <= max_faces_per_side:
            return face_rows
        indices = np.linspace(
            0,
            len(face_rows) - 1,
            num=max_faces_per_side,
            dtype=np.int64,
        )
        return face_rows[indices]

    before_sample = sample(before_faces)
    after_sample = sample(after_faces)
    before_centroids, before_normals, _before_areas = _face_geometry(
        before_vertices, before_sample
    )
    after_centroids, after_normals, _after_areas = _face_geometry(
        after_vertices, after_sample
    )
    bounds_min = np.minimum(before_vertices.min(axis=0), after_vertices.min(axis=0))
    bounds_max = np.maximum(before_vertices.max(axis=0), after_vertices.max(axis=0))
    diagonal = float(np.linalg.norm(bounds_max.astype(np.float64) - bounds_min))
    normalizer = diagonal if diagonal > 0 else 1.0

    def direction(
        source_centroids: np.ndarray,
        source_normals: np.ndarray,
        target_centroids: np.ndarray,
        target_normals: np.ndarray,
    ) -> dict[str, Any]:
        if len(source_centroids) == 0 or len(target_centroids) == 0:
            return {
                "normalized_centroid_distance": _quantiles(np.empty(0)),
                "absolute_normal_dot": _quantiles(np.empty(0)),
                "opposed_normal_fraction": None,
            }
        distances, nearest = cKDTree(target_centroids).query(source_centroids, workers=-1)
        dots = np.einsum("ij,ij->i", source_normals, target_normals[nearest])
        return {
            "normalized_centroid_distance": _quantiles(distances / normalizer),
            "absolute_normal_dot": _quantiles(np.abs(dots)),
            "opposed_normal_fraction": float((dots < 0).mean()),
        }

    return {
        "claim": "nearest-face-centroid-proximity-not-lineage",
        "sampling": "deterministic-evenly-spaced-face-rows",
        "max_faces_per_side": int(max_faces_per_side),
        "before_faces": int(len(before_faces)),
        "before_sampled_faces": int(len(before_sample)),
        "after_faces": int(len(after_faces)),
        "after_sampled_faces": int(len(after_sample)),
        "bounds_diagonal": diagonal,
        "before_to_after": direction(
            before_centroids, before_normals, after_centroids, after_normals
        ),
        "after_to_before": direction(
            after_centroids, after_normals, before_centroids, before_normals
        ),
    }


def attest_uv_mapping(
    clean_vertices: np.ndarray,
    clean_faces: np.ndarray,
    uv_vertices: np.ndarray,
    uv_faces: np.ndarray,
    vmapping: np.ndarray,
) -> dict[str, Any]:
    """Attest whether UV seam splitting preserves exact clean geometry."""
    clean_vertices, clean_faces = _mesh_arrays(clean_vertices, clean_faces)
    uv_vertices, uv_faces = _mesh_arrays(uv_vertices, uv_faces)
    vmapping = np.asarray(vmapping)
    valid_mapping = bool(
        vmapping.ndim == 1
        and len(vmapping) == len(uv_vertices)
        and (not len(vmapping) or (int(vmapping.min()) >= 0 and int(vmapping.max()) < len(clean_vertices)))
    )
    vertices_exact = valid_mapping and np.array_equal(uv_vertices, clean_vertices[vmapping])
    faces_exact = False
    if valid_mapping and len(uv_faces) == len(clean_faces):
        faces_exact = np.array_equal(
            vmapping[uv_faces].astype(clean_faces.dtype, copy=False), clean_faces
        )
    return {
        "claim": "exact-uv-seam-split-mapping",
        "mapping_valid": valid_mapping,
        "uv_vertices_equal_clean_vertices_at_vmapping": bool(vertices_exact),
        "uv_faces_map_exactly_to_clean_faces": bool(faces_exact),
        "geometry_preserved_exactly": bool(vertices_exact and faces_exact),
        "clean_vertices": int(len(clean_vertices)),
        "uv_vertices": int(len(uv_vertices)),
        "clean_faces": int(len(clean_faces)),
        "uv_faces": int(len(uv_faces)),
    }
