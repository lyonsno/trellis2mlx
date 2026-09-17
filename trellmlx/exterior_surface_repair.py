"""Camera-independent repair for exposed inward sheets and touching contour gaps.

The repair deliberately has no asset, component-rank, camera, or face-id input.
It finds coherent sheets whose radial orientation is negative and whose
decisive outside rays reach only the back of the triangles, then acts only when
one candidate has more surface area than every eligible alternative combined.
After reversing that sheet, it closes only detached boundary loops which are
local to the corrected sheet and have one exact-contact mate on the main shell.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import igl
import numpy as np
import trimesh
from scipy.spatial import cKDTree


NORMAL_SIDE_ONLY = 1
BACK_SIDE_ONLY = 2
BOTH_SIDES = 3
NEITHER_SIDE = 4
MIN_CONTOUR_PLANE_ALIGNMENT = 0.9
MIN_CONTACT_TANGENT_ALIGNMENT = 0.9
MAX_HAUSDORFF_SPAN_RATIO = 0.5
MAX_MEAN_SEPARATION_SPAN_RATIO = 0.2


class BridgeRefused(ValueError):
    """The proposed contour closure has no geometrically safe realization."""


@dataclass(frozen=True)
class _BoundaryLoop:
    component: int
    component_faces: int
    vertices: np.ndarray
    perimeter: float


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
    same_direction = int(np.count_nonzero((counts == 2) & (np.abs(direction_sum) == 2)))
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
    tree.init(vertices, faces.astype(np.int64, copy=False))
    rows = np.flatnonzero(valid)
    normal_hits, _, _ = tree.intersect_ray_first(
        vertices,
        faces.astype(np.int64, copy=False),
        centroids[rows] + normals[rows] * reach,
        -normals[rows],
        reach * 1.1,
    )
    back_hits, _, _ = tree.intersect_ray_first(
        vertices,
        faces.astype(np.int64, copy=False),
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
    vertices: np.ndarray, faces: np.ndarray, main_component: np.ndarray
) -> tuple[list[np.ndarray], np.ndarray]:
    triangles = vertices[faces]
    area_vectors = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    centroids = triangles.mean(axis=1)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    scores = np.einsum("ij,ij->i", area_vectors, centroids - center)
    negative = np.asarray(
        [face_id for face_id in main_component if scores[face_id] < 0.0],
        dtype=np.int64,
    )
    if len(negative) == 0:
        return [], area_vectors
    negative_set = set(int(face_id) for face_id in negative)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    edges = np.asarray(
        [
            pair
            for pair in mesh.face_adjacency
            if int(pair[0]) in negative_set and int(pair[1]) in negative_set
        ],
        dtype=np.int64,
    )
    if len(edges) == 0:
        return [np.asarray([face_id], dtype=np.int64) for face_id in negative], area_vectors
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
    main = np.asarray(max(components, key=len), dtype=np.int64)
    clusters, area_vectors = _negative_sheet_clusters(vertices, faces, main)
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
        and not np.isclose(candidates[0]["area"], alternative_area, rtol=1e-12, atol=0.0)
    ):
        selected = candidates[0]["faces"]
    receipt = {
        "candidate_count": len(candidates),
        "selection_status": (
            "selected" if selected is not None else "ambiguous" if candidates else "none"
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


def _boundary_loops(vertices: np.ndarray, faces: np.ndarray) -> list[_BoundaryLoop]:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    components = sorted(
        trimesh.graph.connected_components(
            mesh.face_adjacency, nodes=np.arange(len(faces)), min_len=1
        ),
        key=len,
        reverse=True,
    )
    loops = []
    for component_index, component in enumerate(components):
        component_faces = faces[np.asarray(component, dtype=np.int64)]
        directed = np.concatenate(
            (
                component_faces[:, [0, 1]],
                component_faces[:, [1, 2]],
                component_faces[:, [2, 0]],
            ),
            axis=0,
        )
        undirected = np.sort(directed, axis=1)
        unique, counts = np.unique(undirected, axis=0, return_counts=True)
        boundary = unique[counts == 1]
        graph: dict[int, set[int]] = defaultdict(set)
        for left, right in boundary:
            graph[int(left)].add(int(right))
            graph[int(right)].add(int(left))
        unseen = set(graph)
        while unseen:
            seed = min(unseen)
            stack = [seed]
            group = set()
            while stack:
                vertex = stack.pop()
                if vertex in group:
                    continue
                group.add(vertex)
                stack.extend(graph[vertex] - group)
            unseen -= group
            if Counter(len(graph[vertex]) for vertex in group) != Counter({2: len(group)}):
                continue
            ordered = [min(group)]
            previous = None
            current = ordered[0]
            while True:
                choices = sorted(
                    graph[current] - ({previous} if previous is not None else set())
                )
                following = choices[0]
                if following == ordered[0]:
                    break
                ordered.append(following)
                previous, current = current, following
            ordered_array = np.asarray(ordered, dtype=np.int64)
            points = vertices[ordered_array]
            perimeter = float(
                np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).sum()
            )
            loops.append(
                _BoundaryLoop(
                    component=component_index,
                    component_faces=int(len(component)),
                    vertices=ordered_array,
                    perimeter=perimeter,
                )
            )
    return loops


def _zipper_faces(
    vertices: np.ndarray, main_loop: np.ndarray, detached_loop: np.ndarray
) -> np.ndarray:
    main_points = vertices[main_loop]
    detached_points = vertices[detached_loop]
    main_lengths = np.linalg.norm(
        np.roll(main_points, -1, axis=0) - main_points, axis=1
    )
    detached_lengths = np.linalg.norm(
        np.roll(detached_points, -1, axis=0) - detached_points, axis=1
    )
    main_progress = np.cumsum(main_lengths) / max(float(main_lengths.sum()), 1e-30)
    detached_progress = np.cumsum(detached_lengths) / max(
        float(detached_lengths.sum()), 1e-30
    )
    main_index = 0
    detached_index = 0
    faces = []
    while main_index < len(main_loop) or detached_index < len(detached_loop):
        next_main = main_progress[main_index] if main_index < len(main_loop) else np.inf
        next_detached = (
            detached_progress[detached_index]
            if detached_index < len(detached_loop)
            else np.inf
        )
        current_main = int(main_loop[main_index % len(main_loop)])
        current_detached = int(detached_loop[detached_index % len(detached_loop)])
        if next_main <= next_detached:
            following_main = int(main_loop[(main_index + 1) % len(main_loop)])
            faces.append((current_main, current_detached, following_main))
            main_index += 1
        else:
            following_detached = int(
                detached_loop[(detached_index + 1) % len(detached_loop)]
            )
            faces.append((current_main, current_detached, following_detached))
            detached_index += 1
    return np.asarray(faces, dtype=np.int64)


def _rotate(values: np.ndarray, start: int, *, reverse: bool) -> np.ndarray:
    ordered = values[::-1] if reverse else values
    positions = np.flatnonzero(ordered == start)
    if len(positions) != 1:
        raise ValueError("loop start must appear exactly once")
    return np.roll(ordered, -int(positions[0]))


def _loop_plane_normal(points: np.ndarray) -> np.ndarray | None:
    if len(points) < 3:
        return None
    centered = points - points.mean(axis=0)
    _, singular_values, axes = np.linalg.svd(centered, full_matrices=False)
    if len(singular_values) < 2 or singular_values[1] <= np.finfo(np.float64).eps:
        return None
    normal = axes[-1]
    magnitude = float(np.linalg.norm(normal))
    return normal / magnitude if magnitude else None


def _loop_tangent(points: np.ndarray, index: int) -> np.ndarray | None:
    tangent = points[(index + 1) % len(points)] - points[(index - 1) % len(points)]
    magnitude = float(np.linalg.norm(tangent))
    return tangent / magnitude if magnitude else None


def _contour_frames_are_compatible(
    main_points: np.ndarray,
    detached_points: np.ndarray,
    *,
    contact_tolerance: float,
) -> bool:
    main_normal = _loop_plane_normal(main_points)
    detached_normal = _loop_plane_normal(detached_points)
    if main_normal is None or detached_normal is None:
        return False
    if abs(float(np.dot(main_normal, detached_normal))) < MIN_CONTOUR_PLANE_ALIGNMENT:
        return False

    main_tree = cKDTree(main_points)
    detached_tree = cKDTree(detached_points)
    main_distances = detached_tree.query(main_points, k=1)[0]
    detached_distances = main_tree.query(detached_points, k=1)[0]
    span = max(
        float(np.linalg.norm(np.ptp(main_points, axis=0))),
        float(np.linalg.norm(np.ptp(detached_points, axis=0))),
        np.finfo(np.float64).tiny,
    )
    hausdorff = max(float(main_distances.max()), float(detached_distances.max()))
    mean_separation = max(
        float(main_distances.mean()), float(detached_distances.mean())
    )
    if hausdorff / span > MAX_HAUSDORFF_SPAN_RATIO:
        return False
    if mean_separation / span > MAX_MEAN_SEPARATION_SPAN_RATIO:
        return False

    contacts = []
    for detached_index, point in enumerate(detached_points):
        matches = main_tree.query_ball_point(point, r=contact_tolerance)
        if len(matches) > 1:
            return False
        if len(matches) == 1:
            contacts.append((matches[0], detached_index))
    if not contacts:
        return False
    for main_index, detached_index in contacts:
        main_tangent = _loop_tangent(main_points, main_index)
        detached_tangent = _loop_tangent(detached_points, detached_index)
        if main_tangent is None or detached_tangent is None:
            return False
        if (
            abs(float(np.dot(main_tangent, detached_tangent)))
            < MIN_CONTACT_TANGENT_ALIGNMENT
        ):
            return False
    return True


def _triangles_intersect(
    vertices: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> bool:
    shared = sorted(
        set(int(value) for value in left) & set(int(value) for value in right)
    )
    left_points = vertices[left]
    right_points = vertices[right]
    if np.any(left_points.max(axis=0) < right_points.min(axis=0)):
        return False
    if np.any(right_points.max(axis=0) < left_points.min(axis=0)):
        return False
    scale = max(
        float(np.linalg.norm(np.ptp(left_points, axis=0))),
        float(np.linalg.norm(np.ptp(right_points, axis=0))),
        np.finfo(np.float64).tiny,
    )
    origin = np.concatenate((left_points, right_points), axis=0).mean(axis=0)
    normalized_left = (left_points - origin) / scale
    normalized_right = (right_points - origin) / scale
    hit, coplanar, source, target = igl.tri_tri_intersection_test_3d(
        normalized_left[[0]],
        normalized_left[[1]],
        normalized_left[[2]],
        normalized_right[[0]],
        normalized_right[[1]],
        normalized_right[[2]],
    )
    if not hit:
        return False
    if not shared:
        return True
    if len(shared) == 3:
        return True

    relative_tolerance = 1e-9
    linear_tolerance = relative_tolerance
    area_tolerance = relative_tolerance
    if not coplanar:
        endpoints = (
            np.asarray(source).reshape(-1, 3)[0],
            np.asarray(target).reshape(-1, 3)[0],
        )
        if len(shared) == 1:
            permitted = (vertices[shared[0]] - origin) / scale
            return any(
                np.linalg.norm(endpoint - permitted) > linear_tolerance
                for endpoint in endpoints
            )
        edge_start, edge_end = (vertices[shared] - origin) / scale
        edge = edge_end - edge_start
        edge_length_squared = float(np.dot(edge, edge))
        if edge_length_squared == 0.0:
            return True
        for endpoint in endpoints:
            progress = float(np.dot(endpoint - edge_start, edge) / edge_length_squared)
            closest = edge_start + np.clip(progress, 0.0, 1.0) * edge
            if not (
                -relative_tolerance <= progress <= 1.0 + relative_tolerance
            ):
                return True
            if np.linalg.norm(endpoint - closest) > linear_tolerance:
                return True
        return False

    normal = np.cross(
        normalized_left[1] - normalized_left[0],
        normalized_left[2] - normalized_left[0],
    )
    if np.linalg.norm(normal) <= area_tolerance:
        normal = np.cross(
            normalized_right[1] - normalized_right[0],
            normalized_right[2] - normalized_right[0],
        )
    drop_axis = int(np.argmax(np.abs(normal)))
    left_2d = np.delete(normalized_left, drop_axis, axis=1)
    right_2d = np.delete(normalized_right, drop_axis, axis=1)

    def orient(a: np.ndarray, b: np.ndarray, point: np.ndarray) -> float:
        edge = b - a
        offset = point - a
        return float(edge[0] * offset[1] - edge[1] * offset[0])

    if len(shared) == 2:
        shared_left = [int(np.flatnonzero(left == vertex)[0]) for vertex in shared]
        shared_right = [int(np.flatnonzero(right == vertex)[0]) for vertex in shared]
        left_other = next(index for index in range(3) if index not in shared_left)
        right_other = next(index for index in range(3) if index not in shared_right)
        edge_start = np.delete((vertices[shared[0]] - origin) / scale, drop_axis)
        edge_end = np.delete((vertices[shared[1]] - origin) / scale, drop_axis)
        left_side = orient(edge_start, edge_end, left_2d[left_other])
        right_side = orient(edge_start, edge_end, right_2d[right_other])
        return left_side * right_side >= -(area_tolerance * area_tolerance)

    shared_point = np.delete((vertices[shared[0]] - origin) / scale, drop_axis)

    def close_to_shared(point: np.ndarray) -> bool:
        return bool(np.linalg.norm(point - shared_point) <= linear_tolerance)

    def on_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> bool:
        if abs(orient(start, end, point)) > area_tolerance:
            return False
        return bool(
            np.dot(point - start, point - end)
            <= linear_tolerance * linear_tolerance
        )

    def inside_triangle(point: np.ndarray, triangle: np.ndarray) -> bool:
        signs = np.asarray(
            [
                orient(triangle[0], triangle[1], point),
                orient(triangle[1], triangle[2], point),
                orient(triangle[2], triangle[0], point),
            ]
        )
        return bool(
            np.all(signs >= -area_tolerance)
            or np.all(signs <= area_tolerance)
        )

    for point in left_2d:
        if not close_to_shared(point) and inside_triangle(point, right_2d):
            return True
    for point in right_2d:
        if not close_to_shared(point) and inside_triangle(point, left_2d):
            return True
    left_edges = list(zip(left_2d, np.roll(left_2d, -1, axis=0)))
    right_edges = list(zip(right_2d, np.roll(right_2d, -1, axis=0)))
    for left_start, left_end in left_edges:
        for right_start, right_end in right_edges:
            candidates = []
            for point in (left_start, left_end):
                if on_segment(point, right_start, right_end):
                    candidates.append(point)
            for point in (right_start, right_end):
                if on_segment(point, left_start, left_end):
                    candidates.append(point)
            if any(not close_to_shared(point) for point in candidates):
                return True
            left_a = orient(left_start, left_end, right_start)
            left_b = orient(left_start, left_end, right_end)
            right_a = orient(right_start, right_end, left_start)
            right_b = orient(right_start, right_end, left_end)
            if (
                left_a * left_b < -(area_tolerance * area_tolerance)
                and right_a * right_b < -(area_tolerance * area_tolerance)
            ):
                return True
    return False


def _has_forbidden_bridge_intersection(
    vertices: np.ndarray,
    base_faces: np.ndarray,
    bridge_faces: np.ndarray,
) -> bool:
    for left_index, left in enumerate(bridge_faces):
        for right in bridge_faces[left_index + 1 :]:
            if _triangles_intersect(vertices, left, right):
                return True
    if len(base_faces) == 0 or len(bridge_faces) == 0:
        return False
    base_points = vertices[base_faces]
    base_centers = base_points.mean(axis=1)
    base_radii = np.linalg.norm(base_points - base_centers[:, None], axis=2).max(
        axis=1
    )
    base_tree = cKDTree(base_centers)
    max_base_radius = float(base_radii.max())
    for bridge in bridge_faces:
        points = vertices[bridge]
        center = points.mean(axis=0)
        radius = float(np.linalg.norm(points - center, axis=1).max())
        for base_index in base_tree.query_ball_point(
            center, r=radius + max_base_radius
        ):
            if np.linalg.norm(base_centers[base_index] - center) > (
                radius + base_radii[base_index]
            ):
                continue
            if _triangles_intersect(vertices, bridge, base_faces[base_index]):
                return True
    return False


def bridge_loop_pair(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    main_loop: np.ndarray,
    detached_loop: np.ndarray,
    coincidence_tolerance: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Bridge one admitted contour pair with local-only exact-contact welding."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    main_loop = np.asarray(main_loop, dtype=np.int64)
    detached_loop = np.asarray(detached_loop, dtype=np.int64)
    diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0))) if len(vertices) else 0.0
    tolerance = (
        max(diagonal * 1e-9, np.finfo(np.float64).eps * 1024.0)
        if coincidence_tolerance is None
        else float(coincidence_tolerance)
    )
    main_tree = cKDTree(vertices[main_loop])
    weld = {}
    for detached_vertex in detached_loop:
        matching = main_tree.query_ball_point(vertices[detached_vertex], r=tolerance)
        if len(matching) > 1:
            raise ValueError("ambiguous cross-contour contact")
        if len(matching) == 1:
            weld[int(detached_vertex)] = int(main_loop[matching[0]])
    for loop in (main_loop, detached_loop):
        for left, right in zip(loop, np.roll(loop, -1)):
            left = int(left)
            right = int(right)
            if np.linalg.norm(vertices[left] - vertices[right]) <= tolerance:
                canonical = min(left, right)
                weld[max(left, right)] = canonical

    input_topology = _topology_counts(faces)
    welded_vertices = np.asarray(
        sorted(set(weld) | set(weld.values())), dtype=np.int64
    )
    affected_base = (
        np.any(np.isin(faces, welded_vertices), axis=1)
        if len(welded_vertices)
        else np.zeros(len(faces), dtype=bool)
    )
    remapped_base = np.array(faces, copy=True)
    for source, target in weld.items():
        remapped_base[remapped_base == source] = target
    original_unique = np.asarray([len(set(row)) == 3 for row in faces], dtype=bool)
    remapped_unique = np.asarray(
        [len(set(row)) == 3 for row in remapped_base], dtype=bool
    )
    collapsed_base = original_unique & ~remapped_unique
    remapped_base = remapped_base[~collapsed_base]
    affected_base = affected_base[~collapsed_base]
    post_weld_topology = _topology_counts(remapped_base)
    post_weld_intersection_safe = not _has_forbidden_bridge_intersection(
        vertices,
        remapped_base[~affected_base],
        remapped_base[affected_base],
    )

    distance_matrix = np.linalg.norm(
        vertices[main_loop][:, None, :] - vertices[detached_loop][None, :, :], axis=2
    )
    minimum = float(distance_matrix.min())
    starts = np.argwhere(distance_matrix <= minimum + tolerance)
    area_tolerance = max(diagonal * diagonal * 1e-18, np.finfo(np.float64).tiny)
    candidates = []
    for main_position, detached_position in starts:
        main_start = int(main_loop[main_position])
        detached_start = int(detached_loop[detached_position])
        for reverse_main in (False, True):
            for reverse_detached in (False, True):
                ordered_main = _rotate(main_loop, main_start, reverse=reverse_main)
                ordered_detached = _rotate(
                    detached_loop, detached_start, reverse=reverse_detached
                )
                bridge = _zipper_faces(vertices, ordered_main, ordered_detached)
                for source, target in weld.items():
                    bridge[bridge == source] = target
                unique = np.asarray([len(set(row)) == 3 for row in bridge], dtype=bool)
                areas = np.zeros(len(bridge), dtype=np.float64)
                if np.any(unique):
                    triangles = vertices[bridge[unique]]
                    areas[unique] = 0.5 * np.linalg.norm(
                        np.cross(
                            triangles[:, 1] - triangles[:, 0],
                            triangles[:, 2] - triangles[:, 0],
                        ),
                        axis=1,
                    )
                kept = bridge[unique & (areas > area_tolerance)]
                combined = np.concatenate((remapped_base, kept), axis=0)
                topology = _topology_counts(combined)
                topology_safe = (
                    topology["nonmanifold_edges"]
                    <= input_topology["nonmanifold_edges"]
                    and topology["same_direction_shared_edges"]
                    <= input_topology["same_direction_shared_edges"]
                    and (
                        input_topology["boundary_edges"] == 0
                        or topology["boundary_edges"]
                        < input_topology["boundary_edges"]
                    )
                )
                intersection_safe = (
                    post_weld_intersection_safe
                    and not _has_forbidden_bridge_intersection(
                        vertices, remapped_base, kept
                    )
                )
                connector_cost = (
                    float(
                        np.linalg.norm(
                            vertices[kept][:, 0] - vertices[kept][:, 1], axis=1
                        ).sum()
                    )
                    if len(kept)
                    else float("inf")
                )
                candidates.append(
                    {
                        "faces": combined,
                        "kept_bridge_faces": len(kept),
                        "dropped_bridge_faces": int(len(bridge) - len(kept)),
                        "topology": topology,
                        "topology_safe": topology_safe,
                        "intersection_safe": intersection_safe,
                        "connector_cost": connector_cost,
                    }
                )
    candidates = [
        row
        for row in candidates
        if row["topology_safe"] and row["intersection_safe"]
    ]
    if not candidates:
        raise BridgeRefused("contour pair has no topology-safe alignment")
    candidates.sort(
        key=lambda row: (
            row["topology"]["nonmanifold_edges"],
            row["topology"]["same_direction_shared_edges"],
            row["topology"]["boundary_edges"],
            row["connector_cost"],
        )
    )
    chosen = candidates[0]
    after = chosen["topology"]
    return chosen["faces"], {
        "welded_vertex_pairs": len(weld),
        "dropped_source_faces": int(np.count_nonzero(collapsed_base)),
        "dropped_bridge_faces": chosen["dropped_bridge_faces"],
        "appended_bridge_faces": chosen["kept_bridge_faces"],
        "new_zero_area_faces": 0,
        "new_nonmanifold_edges": max(
            0,
            after["nonmanifold_edges"] - input_topology["nonmanifold_edges"],
        ),
        "new_same_direction_shared_edges": max(
            0,
            after["same_direction_shared_edges"]
            - input_topology["same_direction_shared_edges"],
        ),
        "new_self_intersections": 0,
        "boundary_edge_delta": (
            after["boundary_edges"] - input_topology["boundary_edges"]
        ),
        "before": input_topology,
        "post_weld": post_weld_topology,
        "after": after,
    }


def _select_bridge_pairs(
    vertices: np.ndarray, faces: np.ndarray, patch_faces: np.ndarray
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    loops = _boundary_loops(vertices, faces)
    main_loops = [loop for loop in loops if loop.component == 0]
    detached_loops = [loop for loop in loops if loop.component != 0]
    if not main_loops or not detached_loops:
        return [], {
            "candidate_detached_loops": 0,
            "rejected_ambiguous": 0,
            "rejected_incompatible": 0,
        }
    patch_vertices = np.unique(faces[patch_faces].reshape(-1))
    patch_tree = cKDTree(vertices[patch_vertices])
    diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    contact_tolerance = max(diagonal * 1e-9, np.finfo(np.float64).eps * 1024.0)
    pairs = []
    rejected_ambiguous = 0
    rejected_incompatible = 0
    candidate_count = 0
    used_main: set[int] = set()
    for detached in detached_loops:
        edge_scale = detached.perimeter / max(len(detached.vertices), 1)
        patch_distance = float(
            patch_tree.query(vertices[detached.vertices], k=1)[0].min()
        )
        if patch_distance > 2.0 * edge_scale:
            continue
        candidate_count += 1
        contacts = []
        detached_tree = cKDTree(vertices[detached.vertices])
        for main_index, main in enumerate(main_loops):
            distance = float(
                detached_tree.query(vertices[main.vertices], k=1)[0].min()
            )
            if distance <= contact_tolerance:
                contacts.append((main_index, main))
        if len(contacts) != 1 or contacts[0][0] in used_main:
            rejected_ambiguous += 1
            continue
        main_index, main = contacts[0]
        main_points = vertices[main.vertices]
        detached_points = vertices[detached.vertices]
        main_span = float(np.linalg.norm(np.ptp(main_points, axis=0)))
        detached_span = float(np.linalg.norm(np.ptp(detached_points, axis=0)))
        span_ratio = min(main_span, detached_span) / max(
            main_span, detached_span, np.finfo(np.float64).tiny
        )
        perimeter_ratio = min(main.perimeter, detached.perimeter) / max(
            main.perimeter, detached.perimeter, np.finfo(np.float64).tiny
        )
        if span_ratio < 0.75 or perimeter_ratio < 0.25:
            rejected_incompatible += 1
            continue
        if not _contour_frames_are_compatible(
            main_points,
            detached_points,
            contact_tolerance=contact_tolerance,
        ):
            rejected_incompatible += 1
            continue
        used_main.add(main_index)
        pairs.append((main.vertices, detached.vertices))
    return pairs, {
        "candidate_detached_loops": candidate_count,
        "rejected_ambiguous": rejected_ambiguous,
        "rejected_incompatible": rejected_incompatible,
    }


def repair_exterior_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    component_orientation_confidence: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """Repair one dominant exterior-inverted sheet and its touching contour gaps."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
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
    if patch is None:
        output_topology = _topology_counts(prepared)
        return prepared, {
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
                "reversed_faces": 0,
            },
            "bridges": {
                "candidate_detached_loops": 0,
                "rejected_ambiguous": 0,
                "rejected_incompatible": 0,
                "rejected_postcondition": 0,
                "accepted_pairs": 0,
                "pairs": [],
            },
        }

    prepared[patch] = prepared[patch][:, [0, 2, 1]]
    pairs, selection_receipt = _select_bridge_pairs(vertices, prepared, patch)
    bridge_receipts = []
    rejected_postcondition = 0
    for main_loop, detached_loop in pairs:
        try:
            prepared, bridge_receipt = bridge_loop_pair(
                vertices,
                prepared,
                main_loop=main_loop,
                detached_loop=detached_loop,
            )
        except BridgeRefused:
            rejected_postcondition += 1
            continue
        bridge_receipts.append(bridge_receipt)
    output_topology = _topology_counts(prepared)
    return prepared, {
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
            "reversed_faces": int(len(patch)),
        },
        "bridges": {
            **selection_receipt,
            "rejected_postcondition": rejected_postcondition,
            "accepted_pairs": len(bridge_receipts),
            "pairs": bridge_receipts,
        },
    }
