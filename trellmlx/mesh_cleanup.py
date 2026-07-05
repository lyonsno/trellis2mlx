"""Mesh post-processing: remove floaters, fill holes, clean floors.

The TRELLIS.2 decoder produces raw meshes with:
- Small disconnected components (floaters)
- Holes from single-view occlusion
- A large flat "floor" component from the surface the object sits on

This module cleans all three without cumesh (which segfaults on Metal).
"""

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def cleanup_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    max_hole_perimeter: float = 3e-2,
    keep_largest: bool = False,
    min_component_ratio: float = 1e-5,
    do_fix_normals: bool = True,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Clean up a raw decoder mesh.

    Pipeline:
    1. Remove duplicate faces
    2. Repair non-manifold edges
    3. Remove small connected components (surface-area threshold, matching
       reference ``cumesh.remove_small_connected_components(1e-5)``), or
       keep only the largest if ``keep_largest=True``
    4. Fill small holes
    5. Fix normals/winding consistency (optional, skip for intermediate passes)

    Args:
        vertices: [V, 3] float32
        faces: [F, 3] int
        max_hole_perimeter: Fill holes with perimeter smaller than this (world-space
            units). Matches reference cumesh ``fill_holes(max_hole_perimeter=3e-2)``.
        keep_largest: If True, discard all components except the largest.
            Overrides min_component_ratio.
        min_component_ratio: Backward-compatible name for the absolute
            minimum component surface area to keep. Default 1e-5 matches the
            reference ``cumesh.remove_small_connected_components(1e-5)``.
        do_fix_normals: Run winding unification. The reference only does this once
            at the end, so callers can skip it for intermediate cleanup passes
            before simplification.
        verbose: Print progress.

    Returns:
        cleaned_vertices: [V', 3]
        cleaned_faces: [F', 3]
    """
    original_faces = len(faces)
    original_verts = len(vertices)

    # Step 1: Remove duplicate faces
    vertices, faces = remove_duplicate_faces(vertices, faces, verbose)

    # Step 2: Repair non-manifold edges
    vertices, faces = repair_non_manifold_edges(vertices, faces, verbose)

    # Step 3: Component removal
    if keep_largest:
        vertices, faces = keep_largest_component(vertices, faces, verbose)
    else:
        vertices, faces = remove_small_components(
            vertices, faces, min_ratio=min_component_ratio, verbose=verbose,
        )

    # Step 4: Fill small holes on the remaining mesh
    vertices, faces = fill_small_holes(
        vertices,
        faces,
        max_hole_perimeter=max_hole_perimeter,
        verbose=verbose,
    )

    # Step 5: Fix normals/winding consistency (skip for intermediate passes)
    if do_fix_normals:
        vertices, faces = fix_normals(vertices, faces, verbose)

    if verbose:
        print(f"  Cleanup: {original_verts:,}V {original_faces:,}F → "
              f"{len(vertices):,}V {len(faces):,}F", flush=True)

    return vertices, faces


def keep_largest_component(
    vertices: np.ndarray,
    faces: np.ndarray,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the largest connected component, remove everything else."""
    if len(faces) == 0:
        return vertices, faces

    n_faces = len(faces)
    n_components, labels = _face_connected_components(faces, n_faces)

    if n_components <= 1:
        return vertices, faces

    component_sizes = np.bincount(labels)
    largest_idx = component_sizes.argmax()
    keep_mask = labels == largest_idx

    kept = keep_mask.sum()
    removed = n_faces - kept
    if verbose:
        print(f"  Kept largest component ({kept:,} faces), "
              f"removed {n_components - 1} others ({removed:,} faces, "
              f"{removed/n_faces*100:.1f}%)", flush=True)

    return _reindex_mesh(vertices, faces[keep_mask])


def remove_small_components(
    vertices: np.ndarray,
    faces: np.ndarray,
    min_ratio: float = 1e-5,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove connected components below an absolute surface-area cutoff.

    Matches the reference ``cumesh.remove_small_connected_components(1e-5)``.
    Pure surface-area thresholding only — no shape heuristics.
    """
    if len(faces) == 0:
        return vertices, faces

    n_faces = len(faces)
    n_components, labels = _face_connected_components(faces, n_faces)

    if n_components <= 1:
        return vertices, faces

    component_areas = _component_surface_areas(vertices, faces, labels, n_components)
    threshold = float(min_ratio)

    keep_components = component_areas >= threshold
    keep_mask = keep_components[labels]

    kept_faces = faces[keep_mask]
    removed = n_faces - len(kept_faces)
    removed_count = (~keep_components).sum()

    if verbose and removed > 0:
        print(f"  Removed {removed_count} small components "
              f"({removed:,} faces, {removed/n_faces*100:.1f}%)", flush=True)

    return _reindex_mesh(vertices, kept_faces)


def _component_surface_areas(
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """Compute total triangle area for each face-connected component."""
    tri = vertices[np.asarray(faces, dtype=np.int64)]
    face_areas = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
        axis=1,
    )
    component_areas = np.zeros(n_components, dtype=np.float64)
    np.add.at(component_areas, labels, face_areas)
    return component_areas


def _face_connected_components(faces, n_faces):
    """Compute connected components via face adjacency (shared edges)."""
    edges = {}
    for fi, face in enumerate(faces):
        for i in range(3):
            e = tuple(sorted((face[i], face[(i + 1) % 3])))
            edges.setdefault(e, []).append(fi)

    rows, cols = [], []
    for face_list in edges.values():
        for i in range(len(face_list)):
            for j in range(i + 1, len(face_list)):
                rows.extend([face_list[i], face_list[j]])
                cols.extend([face_list[j], face_list[i]])

    if not rows:
        return 1, np.zeros(n_faces, dtype=np.int32)

    adj = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(n_faces, n_faces),
    )
    return connected_components(adj, directed=False)


def fill_small_holes(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    max_hole_perimeter: float = 3e-2,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill boundary loops (holes) with fan triangulation.

    Uses a perimeter-based threshold in world-space units, matching the
    reference cumesh ``fill_holes(max_hole_perimeter=3e-2)``.
    """
    if len(faces) == 0:
        return vertices, faces

    # Find boundary edges (edges that appear in only one face). Keep their
    # original directed traversal so filled caps oppose the adjacent face.
    edge_count = {}
    directed_edges = []
    for face in faces:
        for i in range(3):
            v0 = int(face[i])
            v1 = int(face[(i + 1) % 3])
            e = tuple(sorted((v0, v1)))
            edge_count[e] = edge_count.get(e, 0) + 1
            directed_edges.append((v0, v1, e))

    boundary_edges = [(v0, v1) for v0, v1, e in directed_edges if edge_count[e] == 1]
    if not boundary_edges:
        return vertices, faces

    # Build directed adjacency for boundary vertices.
    boundary_adj = {}
    for v0, v1 in boundary_edges:
        boundary_adj.setdefault(v0, []).append(v1)

    # Trace boundary loops
    visited_edges = set()
    loops = []
    for start_edge in boundary_edges:
        if start_edge in visited_edges:
            continue
        loop = [start_edge[0], start_edge[1]]
        visited_edges.add(start_edge)
        closed = False
        while True:
            current = loop[-1]
            neighbors = boundary_adj.get(current, [])
            next_vertex = None
            for n in neighbors:
                e = (current, n)
                if e not in visited_edges:
                    next_vertex = n
                    visited_edges.add(e)
                    break
            if next_vertex is None:
                break
            if next_vertex == loop[0] and len(loop) >= 3:
                closed = True
                break
            if next_vertex in loop:
                break
            loop.append(next_vertex)
        if closed:
            loops.append(loop)

    if not loops:
        return vertices, faces

    # Fill loops whose perimeter is below the threshold
    new_faces = list(faces)
    filled = 0
    skipped = 0
    for loop in loops:
        # Compute perimeter in world-space units
        loop_verts = vertices[loop]  # [L, 3]
        edge_vecs = np.roll(loop_verts, -1, axis=0) - loop_verts  # [L, 3]
        perimeter = np.sqrt((edge_vecs ** 2).sum(axis=1)).sum()

        if perimeter > max_hole_perimeter:
            skipped += 1
            continue

        # Fan triangulation from the centroid
        centroid = loop_verts.mean(axis=0)
        centroid_idx = len(vertices)
        vertices = np.vstack([vertices, centroid[None]])
        for i in range(len(loop)):
            v0 = loop[i]
            v1 = loop[(i + 1) % len(loop)]
            new_faces.append([v1, v0, centroid_idx])
        filled += 1

    if verbose and filled > 0:
        print(f"  Filled {filled} holes ({skipped} too large, "
              f"max_perimeter={max_hole_perimeter})", flush=True)

    return vertices, np.array(new_faces, dtype=faces.dtype)


def remove_duplicate_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove duplicate faces (including permutations of the same triangle)."""
    if len(faces) == 0:
        return vertices, faces

    # Canonical form: sort vertex indices within each face
    canonical = np.sort(faces, axis=1)
    _, unique_idx = np.unique(canonical, axis=0, return_index=True)

    if len(unique_idx) == len(faces):
        return vertices, faces

    removed = len(faces) - len(unique_idx)
    if verbose:
        print(f"  Removed {removed} duplicate faces", flush=True)

    unique_idx.sort()  # preserve original order
    return vertices, faces[unique_idx]


def repair_non_manifold_edges(
    vertices: np.ndarray,
    faces: np.ndarray,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Split face-corner vertices across non-manifold edges.

    This mirrors reference cumesh ``repair_non_manifold_edges``. It starts from
    one logical vertex per face corner, unions only corners connected by edges
    shared by exactly two faces, then rebuilds vertices/faces from those union
    components. Faces are preserved; non-manifold sharing is repaired by vertex
    splitting rather than by deleting incident faces.
    """
    if len(faces) == 0:
        return vertices, faces

    n_faces = len(faces)
    faces_i64 = np.asarray(faces, dtype=np.int64)

    edge_faces: dict[tuple[int, int], list[int]] = {}
    edge_order: list[tuple[int, int]] = []
    for face_index, face in enumerate(faces_i64):
        for corner in range(3):
            edge = tuple(sorted((int(face[corner]), int(face[(corner + 1) % 3]))))
            if edge not in edge_faces:
                edge_order.append(edge)
            edge_faces.setdefault(edge, []).append(face_index)

    manifold_adjacency: list[tuple[int, int]] = []
    for edge in edge_order:
        incident = edge_faces[edge]
        if len(incident) == 2:
            manifold_adjacency.append((incident[0], incident[1]))

    if not manifold_adjacency:
        return vertices, faces

    parent = np.arange(3 * n_faces, dtype=np.int64)

    def find(idx: int) -> int:
        root = idx
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[idx]) != idx:
            next_idx = int(parent[idx])
            parent[idx] = root
            idx = next_idx
        return root

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return
        if root_a < root_b:
            parent[root_b] = root_a
        else:
            parent[root_a] = root_b

    for face_a, face_b in manifold_adjacency:
        shared_a: list[int] = []
        shared_b: list[int] = []
        face1 = faces_i64[face_a]
        face2 = faces_i64[face_b]
        for corner_a, vertex_a in enumerate(face1):
            for corner_b, vertex_b in enumerate(face2):
                if int(vertex_a) == int(vertex_b):
                    shared_a.append(corner_a)
                    shared_b.append(corner_b)
                    break
            if len(shared_a) == 2:
                break
        if len(shared_a) != 2:
            continue
        union(3 * face_a + shared_a[0], 3 * face_b + shared_b[0])
        union(3 * face_a + shared_a[1], 3 * face_b + shared_b[1])

    roots = np.array([find(i) for i in range(3 * n_faces)], dtype=np.int64)
    unique_roots, inverse = np.unique(roots, return_inverse=True)
    new_faces = inverse.reshape(n_faces, 3).astype(faces.dtype, copy=False)

    if verbose:
        duplicated = len(unique_roots) - len(vertices)
        if duplicated > 0:
            print(
                f"  Split non-manifold vertices into {len(unique_roots):,} face-corner components "
                f"(+{duplicated:,})",
                flush=True,
            )

    representative_corner = np.empty(len(unique_roots), dtype=np.int64)
    representative_corner[inverse] = np.arange(3 * n_faces, dtype=np.int64)
    representative_faces = representative_corner // 3
    representative_local = representative_corner % 3
    representative_vertices = faces_i64[representative_faces, representative_local]
    return vertices[representative_vertices].copy(), new_faces


def remove_same_direction_manifold_conflicts(
    vertices: np.ndarray,
    faces: np.ndarray,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove the smaller face from residual same-directed manifold edges.

    ``trimesh.repair.fix_normals`` can leave conflicts on complex open meshes:
    an edge shared by exactly two faces is still traversed in the same direction
    by both faces. That cannot render as a locally consistent oriented surface.
    The reference cumesh route runs a final orientation unifier; this fallback
    conservatively prunes the smaller adjacent triangle only for conflicts that
    remain after the normal repair pass.
    """
    if len(faces) == 0:
        return vertices, faces

    faces_i64 = faces.astype(np.int64, copy=False)
    tri = vertices[faces_i64]
    face_areas = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
        axis=1,
    )

    edge_dirs: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for face_index, face in enumerate(faces_i64):
        for corner in range(3):
            a = int(face[corner])
            b = int(face[(corner + 1) % 3])
            edge_dirs.setdefault(tuple(sorted((a, b))), []).append((face_index, a, b))

    faces_to_remove: set[int] = set()
    for incident in edge_dirs.values():
        if len(incident) != 2:
            continue
        (face_a, a0, b0), (face_b, a1, b1) = incident
        if a0 != a1 or b0 != b1:
            continue
        if face_areas[face_a] <= face_areas[face_b]:
            faces_to_remove.add(face_a)
        else:
            faces_to_remove.add(face_b)

    if not faces_to_remove:
        return vertices, faces

    if verbose:
        print(
            f"  Removed {len(faces_to_remove)} residual winding-conflict faces",
            flush=True,
        )

    keep_mask = np.ones(len(faces), dtype=bool)
    keep_mask[list(faces_to_remove)] = False
    return _reindex_mesh(vertices, faces[keep_mask])


def orient_faces_by_adjacency(
    vertices: np.ndarray,
    faces: np.ndarray,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate face winding across manifold adjacency without deleting faces.

    This mirrors reference cumesh ``unify_face_orientations``: compute whether
    each manifold edge requires a flip, then solve face flips with an oriented
    union-find that hooks higher roots to lower roots. On contradictory open
    components, this preserves cumesh's spanning-choice behavior instead of
    imposing a DFS traversal or deleting faces.
    """
    if len(faces) == 0:
        return vertices, faces

    faces_i64 = np.asarray(faces, dtype=np.int64)
    edge_faces: dict[tuple[int, int], list[int]] = {}
    edge_order: list[tuple[int, int]] = []
    for face_index, face in enumerate(faces_i64):
        for corner in range(3):
            a = int(face[corner])
            b = int(face[(corner + 1) % 3])
            edge = tuple(sorted((a, b)))
            if edge not in edge_faces:
                edge_order.append(edge)
            edge_faces.setdefault(edge, []).append(face_index)

    adjacency: list[tuple[int, int]] = []
    flip_required: list[bool] = []
    for edge in edge_order:
        incident = edge_faces[edge]
        if len(incident) != 2:
            continue
        face_a, face_b = incident
        face1 = faces_i64[face_a]
        face2 = faces_i64[face_b]
        shared_in_face1: list[int] = []
        shared_in_face2: list[int] = []
        for i, vertex_a in enumerate(face1):
            for j, vertex_b in enumerate(face2):
                if int(vertex_a) == int(vertex_b):
                    shared_in_face1.append(i)
                    shared_in_face2.append(j)
                    break
            if len(shared_in_face1) == 2:
                break
        if len(shared_in_face1) != 2:
            continue
        direction1 = (shared_in_face1[1] - shared_in_face1[0] + 3) % 3
        direction2 = (shared_in_face2[1] - shared_in_face2[0] + 3) % 3
        adjacency.append((face_a, face_b))
        flip_required.append(direction1 == direction2)

    component_with_flip = np.arange(len(faces_i64), dtype=np.int64) << 1

    def root_with_flip(face_index: int) -> tuple[int, int]:
        value = int(component_with_flip[face_index])
        root = value >> 1
        flip = value & 1
        while True:
            parent_value = int(component_with_flip[root])
            parent_root = parent_value >> 1
            if parent_root == root:
                return root, flip
            flip ^= parent_value & 1
            root = parent_root

    iterations = 0
    for iterations in range(1, 65):
        changed = False
        for (face_a, face_b), edge_needs_flip in zip(adjacency, flip_required):
            root_a, flip_a = root_with_flip(face_a)
            root_b, flip_b = root_with_flip(face_b)
            if root_a == root_b:
                continue
            high = max(root_a, root_b)
            low = min(root_a, root_b)
            component_with_flip[high] = (
                low << 1
            ) | (int(edge_needs_flip) ^ flip_a ^ flip_b)
            changed = True

        for face_index in range(len(faces_i64)):
            root, flip = root_with_flip(face_index)
            component_with_flip[face_index] = (root << 1) | flip

        if not changed:
            break

    flip = (component_with_flip & 1).astype(bool)

    if not flip.any():
        return vertices, faces

    oriented = np.array(faces, copy=True)
    oriented[flip] = oriented[flip][:, ::-1]
    if verbose:
        print(
            f"  Oriented {int(flip.sum())} faces by cumesh-style adjacency union"
            + (f" ({iterations} union iterations)" if iterations else ""),
            flush=True,
        )
    return vertices, oriented.astype(faces.dtype, copy=False)


def orient_components_outward_by_radial_heuristic(
    vertices: np.ndarray,
    faces: np.ndarray,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Flip consistently wound open components whose normals point inward.

    Open decoder meshes do not have a reliable signed volume, but each connected
    exterior patch still has a useful local radial cue. For each face-connected
    component, compare area-weighted normals against vectors from the component
    center to face centers. If the component's aggregate score is inward, flip
    the whole component. Whole-component flips preserve adjacency consistency.
    """
    if len(faces) == 0:
        return vertices, faces

    faces_i64 = np.asarray(faces, dtype=np.int64)
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces_i64):
        for corner in range(3):
            edge = tuple(sorted((int(face[corner]), int(face[(corner + 1) % 3]))))
            edge_faces.setdefault(edge, []).append(face_index)

    adjacency: list[list[int]] = [[] for _ in range(len(faces_i64))]
    for incident in edge_faces.values():
        if len(incident) < 2:
            continue
        for i, face_a in enumerate(incident):
            for face_b in incident[i + 1:]:
                adjacency[face_a].append(face_b)
                adjacency[face_b].append(face_a)

    tri = vertices[faces_i64]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal_len = np.linalg.norm(normals, axis=1)
    face_centers = tri.mean(axis=1)

    oriented = np.array(faces, copy=True)
    seen = np.zeros(len(faces_i64), dtype=bool)
    components_flipped = 0
    faces_flipped = 0
    for start in range(len(faces_i64)):
        if seen[start]:
            continue
        seen[start] = True
        stack = [start]
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)

        comp_idx = np.asarray(component, dtype=np.int64)
        used_vertices = np.unique(faces_i64[comp_idx].reshape(-1))
        if len(used_vertices) == 0:
            continue
        component_center = vertices[used_vertices].mean(axis=0)
        radial = face_centers[comp_idx] - component_center
        radial_len = np.linalg.norm(radial, axis=1)
        usable = (normal_len[comp_idx] > 1e-12) & (radial_len > 1e-12)
        if not usable.any():
            continue
        score = np.sum(
            np.sum(normals[comp_idx][usable] * radial[usable], axis=1)
            / radial_len[usable]
        )
        if score < 0:
            oriented[comp_idx] = oriented[comp_idx][:, ::-1]
            components_flipped += 1
            faces_flipped += int(len(comp_idx))

    if verbose and faces_flipped:
        print(
            f"  Flipped {faces_flipped} faces across {components_flipped} inward components",
            flush=True,
        )
    return vertices, oriented.astype(faces.dtype, copy=False)


_VISIBLE_EXTERIOR_VIEWS = {
    "+X": (0, 1),
    "-X": (0, -1),
    "+Y": (1, 1),
    "-Y": (1, -1),
    "+Z": (2, 1),
    "-Z": (2, -1),
}


def _edge_function_2d(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    return float((p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0]))


def orient_uv_islands_by_visible_exterior(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    image_size: int = 128,
    min_visible_pixels: int = 1,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Flip UV/export islands whose visible exterior pixels are mostly backfacing.

    UV unwrap can split a still-problematic open mesh into many render islands.
    At that point adjacency and simple radial cues no longer identify every
    exterior-facing patch, but the export-facing culling contract is explicit:
    across six orthographic exterior views, an island should not be mostly
    visible only through its back side. This helper flips whole face-connected
    islands only, preserving local winding consistency inside each island.
    """
    if len(faces) == 0:
        return vertices, faces
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if min_visible_pixels < 0:
        raise ValueError("min_visible_pixels must be non-negative")

    vertices64 = np.asarray(vertices, dtype=np.float64)
    faces_i64 = np.asarray(faces, dtype=np.int64)
    n_faces = len(faces_i64)
    n_components, labels = _face_connected_components(faces_i64, n_faces)

    tri = vertices64[faces_i64]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal_len = np.linalg.norm(normals, axis=1)
    unit_normals = np.zeros_like(normals)
    valid_normals = normal_len > 1e-12
    unit_normals[valid_normals] = normals[valid_normals] / normal_len[valid_normals, None]

    component_visible = np.zeros(n_components, dtype=np.int64)
    component_backfacing = np.zeros(n_components, dtype=np.int64)

    for depth_axis, sign in _VISIBLE_EXTERIOR_VIEWS.values():
        axes = [axis for axis in range(3) if axis != depth_axis]
        outward_axis = np.zeros(3, dtype=np.float64)
        outward_axis[depth_axis] = float(sign)

        coords = vertices64[:, axes]
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        span = maxs - mins
        span[span == 0.0] = 1.0

        margin = max(2.0, image_size * 0.05)
        scale = min(
            (image_size - 2.0 * margin) / span[0],
            (image_size - 2.0 * margin) / span[1],
        )
        offset = np.array(
            [
                (image_size - span[0] * scale) / 2.0 - mins[0] * scale,
                (image_size - span[1] * scale) / 2.0 - mins[1] * scale,
            ],
            dtype=np.float64,
        )
        projected = coords * scale + offset
        depths = vertices64[:, depth_axis] * float(sign)

        z_buffer = np.full((image_size, image_size), -np.inf, dtype=np.float64)
        face_buffer = np.full((image_size, image_size), -1, dtype=np.int64)

        for face_index, face in enumerate(faces_i64):
            tri_2d = projected[face]
            if not np.isfinite(tri_2d).all():
                continue
            area2 = _edge_function_2d(tri_2d[0], tri_2d[1], tri_2d[2])
            if abs(area2) < 1e-8:
                continue

            min_xy = np.floor(tri_2d.min(axis=0)).astype(int)
            max_xy = np.ceil(tri_2d.max(axis=0)).astype(int)
            x0 = max(0, int(min_xy[0]))
            y0 = max(0, int(min_xy[1]))
            x1 = min(image_size - 1, int(max_xy[0]))
            y1 = min(image_size - 1, int(max_xy[1]))
            if x0 > x1 or y0 > y1:
                continue

            tri_depths = depths[face]
            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    sample = np.array([x + 0.5, y + 0.5], dtype=np.float64)
                    w0 = _edge_function_2d(tri_2d[1], tri_2d[2], sample) / area2
                    w1 = _edge_function_2d(tri_2d[2], tri_2d[0], sample) / area2
                    w2 = _edge_function_2d(tri_2d[0], tri_2d[1], sample) / area2
                    if w0 < -1e-8 or w1 < -1e-8 or w2 < -1e-8:
                        continue
                    depth = w0 * tri_depths[0] + w1 * tri_depths[1] + w2 * tri_depths[2]
                    if depth > z_buffer[y, x]:
                        z_buffer[y, x] = depth
                        face_buffer[y, x] = face_index

        visible_faces = face_buffer[face_buffer >= 0]
        if len(visible_faces) == 0:
            continue
        visible_components = labels[visible_faces]
        np.add.at(component_visible, visible_components, 1)

        backfacing = (unit_normals[visible_faces] @ outward_axis) < -1e-6
        if backfacing.any():
            np.add.at(component_backfacing, visible_components[backfacing], 1)

    flip_components = (
        (component_visible >= min_visible_pixels)
        & (component_backfacing * 2 > component_visible)
    )
    if not flip_components.any():
        return vertices, faces

    flip_faces = flip_components[labels]
    oriented = np.array(faces, copy=True)
    oriented[flip_faces] = oriented[flip_faces][:, ::-1]

    if verbose:
        print(
            f"  Flipped {int(flip_faces.sum()):,} faces across "
            f"{int(flip_components.sum()):,} visible-backface UV islands",
            flush=True,
        )
    return vertices, oriented.astype(faces.dtype, copy=False)


def fix_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Unify face winding so normals are consistent.

    Uses trimesh's fix_normals which handles both winding consistency
    and outward orientation.
    """
    if len(faces) == 0:
        return vertices, faces

    import trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if not mesh.is_watertight:
        oriented_vertices, oriented_faces = orient_faces_by_adjacency(
            vertices,
            faces,
            verbose=verbose,
        )
        return remove_same_direction_manifold_conflicts(
            oriented_vertices,
            oriented_faces,
            verbose=verbose,
        )

    trimesh.repair.fix_normals(mesh, multibody=True)
    # np.array() to avoid returning trimesh TrackedArray (carries refs to Trimesh)
    fixed_vertices = np.array(mesh.vertices, dtype=vertices.dtype)
    fixed_faces = np.array(mesh.faces, dtype=faces.dtype)
    return remove_same_direction_manifold_conflicts(
        fixed_vertices,
        fixed_faces,
        verbose=verbose,
    )


def _reindex_mesh(vertices, faces):
    """Remove unreferenced vertices and reindex faces."""
    used = np.unique(faces)
    new_idx = np.full(len(vertices), -1, dtype=np.int64)
    new_idx[used] = np.arange(len(used))
    return vertices[used], new_idx[faces].astype(faces.dtype)
