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
    min_component_area: float = 1e-5,
    do_fix_normals: bool = True,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Clean up a raw decoder mesh.

    Pipeline:
    1. Remove duplicate faces
    2. Repair non-manifold edges
    3. Remove small connected components (area threshold, matching
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
            Overrides min_component_area.
        min_component_area: Remove components whose summed face area is below this
            absolute threshold. Default 1e-5 matches the reference pipeline.
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
            vertices, faces, min_area=min_component_area, verbose=verbose,
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
    min_area: float = 1e-5,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove connected components whose summed face area is below min_area.

    Matches the reference ``cumesh.remove_small_connected_components(1e-5)``.
    Pure summed-area thresholding only — no shape heuristics.
    """
    if len(faces) == 0:
        return vertices, faces

    n_faces = len(faces)
    n_components, labels = _face_connected_components(faces, n_faces)

    if n_components <= 1:
        return vertices, faces

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    face_areas = 0.5 * np.sqrt((cross ** 2).sum(axis=1))
    component_areas = np.bincount(labels, weights=face_areas, minlength=n_components)

    keep_mask = component_areas[labels] >= min_area

    kept_faces = faces[keep_mask]
    removed = n_faces - len(kept_faces)
    removed_count = int((component_areas < min_area).sum())

    if verbose and removed > 0:
        print(f"  Removed {removed_count} small components "
              f"({removed:,} faces, {removed/n_faces*100:.1f}%)", flush=True)

    return _reindex_mesh(vertices, kept_faces)


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
        return n_faces, np.arange(n_faces, dtype=np.int32)

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

    # Find boundary edges (edges that appear in only one face)
    edge_count = {}
    for face in faces:
        for i in range(3):
            e = tuple(sorted((face[i], face[(i + 1) % 3])))
            edge_count[e] = edge_count.get(e, 0) + 1

    boundary_edges = {e for e, c in edge_count.items() if c == 1}
    if not boundary_edges:
        return vertices, faces

    # Build adjacency for boundary vertices
    boundary_adj = {}
    for v0, v1 in boundary_edges:
        boundary_adj.setdefault(v0, []).append(v1)
        boundary_adj.setdefault(v1, []).append(v0)

    # Trace boundary loops
    visited_edges = set()
    loops = []
    for start_edge in boundary_edges:
        if start_edge in visited_edges:
            continue
        # Trace from start_edge[0]
        loop = [start_edge[0], start_edge[1]]
        visited_edges.add(start_edge)
        while True:
            current = loop[-1]
            neighbors = boundary_adj.get(current, [])
            found_next = False
            for n in neighbors:
                e = tuple(sorted((current, n)))
                if e not in visited_edges and n != loop[-2] if len(loop) > 1 else True:
                    visited_edges.add(e)
                    if n == loop[0] and len(loop) >= 3:
                        # Closed loop
                        loops.append(loop)
                        found_next = True
                        break
                    loop.append(n)
                    found_next = True
                    break
            if not found_next:
                break

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
        loop_edges = sorted(
            tuple(sorted((int(loop[i]), int(loop[(i + 1) % len(loop)]))))
            for i in range(len(loop))
        )
        for v0, v1 in loop_edges:
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
    """Split vertices across non-manifold edges without deleting faces.

    This mirrors cumesh's repair path: make one ID per face corner, union corner
    IDs only across manifold edges (edges with exactly two incident faces), then
    rebuild faces from the compressed corner IDs. Edges shared by 3+ faces are
    not unioned, so the adjacent faces keep separate vertex instances.
    """
    if len(faces) == 0:
        return vertices, faces

    n_faces = len(faces)
    faces_i64 = faces.astype(np.int64)

    # Guard: packed int64 edge keys overflow at vertex index >= 2^31
    max_vi = faces_i64.max()
    if max_vi >= 2**31:
        raise ValueError(
            f"Vertex index {max_vi} exceeds 2^31-1; "
            "packed int64 edge keys would overflow"
        )

    # Build all edges: 3 edges per face → [3F, 2] sorted pairs
    e0 = np.stack([faces_i64[:, 0], faces_i64[:, 1]], axis=1)
    e1 = np.stack([faces_i64[:, 1], faces_i64[:, 2]], axis=1)
    e2 = np.stack([faces_i64[:, 2], faces_i64[:, 0]], axis=1)
    all_edges = np.concatenate([e0, e1, e2], axis=0)  # [3F, 2]
    all_edges.sort(axis=1)

    # Face and local-corner indices for each edge occurrence. The local-corner
    # arrays are ordered to match the sorted edge endpoints.
    face_idx = np.tile(np.arange(n_faces, dtype=np.int64), 3)
    edge_local_pairs = [(0, 1), (1, 2), (2, 0)]
    first_corners = []
    second_corners = []
    for local_a, local_b in edge_local_pairs:
        va = faces_i64[:, local_a]
        vb = faces_i64[:, local_b]
        a_first = va <= vb
        first_corners.append(np.where(a_first, local_a, local_b))
        second_corners.append(np.where(a_first, local_b, local_a))
    edge_first_corner = np.concatenate(first_corners).astype(np.int64, copy=False)
    edge_second_corner = np.concatenate(second_corners).astype(np.int64, copy=False)

    # Pack edge pairs into single int64 for fast grouping
    edge_keys = all_edges[:, 0] * (2**32) + all_edges[:, 1]

    # Group by edge key
    sort_order = np.argsort(edge_keys)
    sorted_keys = edge_keys[sort_order]
    sorted_face_idx = face_idx[sort_order]
    sorted_first_corner = edge_first_corner[sort_order]
    sorted_second_corner = edge_second_corner[sort_order]

    # Find group boundaries
    breaks = np.concatenate([
        [0],
        np.where(sorted_keys[1:] != sorted_keys[:-1])[0] + 1,
        [len(sorted_keys)],
    ])

    # Only manifold edges contribute vertex-adjacency pairs in cumesh.
    group_sizes = np.diff(breaks)
    manifold_mask = group_sizes == 2
    if not manifold_mask.any():
        return vertices, faces
    if manifold_mask.all():
        return vertices, faces

    parent = np.arange(n_faces * 3, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            if root_a < root_b:
                parent[root_b] = root_a
            else:
                parent[root_a] = root_b

    for gi in np.where(manifold_mask)[0]:
        start, end = breaks[gi], breaks[gi + 1]
        face_a = int(sorted_face_idx[start])
        face_b = int(sorted_face_idx[start + 1])
        union(face_a * 3 + int(sorted_first_corner[start]), face_b * 3 + int(sorted_first_corner[start + 1]))
        union(face_a * 3 + int(sorted_second_corner[start]), face_b * 3 + int(sorted_second_corner[start + 1]))

    roots = np.array([find(i) for i in range(n_faces * 3)], dtype=np.int64)
    unique_roots, inverse = np.unique(roots, return_inverse=True)
    corner_vertices = faces.reshape(-1)
    representative_corners = unique_roots
    new_vertices = vertices[corner_vertices[representative_corners]]
    new_faces = inverse.reshape(n_faces, 3).astype(faces.dtype, copy=False)

    non_manifold_edges = int((group_sizes > 2).sum())
    split_vertices = len(new_vertices) - len(vertices)

    if verbose:
        print(
            f"  Split {split_vertices} vertices across {non_manifold_edges} non-manifold edges",
            flush=True,
        )

    return new_vertices.astype(vertices.dtype, copy=False), new_faces


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
    trimesh.repair.fix_normals(mesh)
    # np.array() to avoid returning trimesh TrackedArray (carries refs to Trimesh)
    return np.array(mesh.vertices, dtype=vertices.dtype), np.array(mesh.faces, dtype=faces.dtype)


def orient_faces_by_adjacency(
    vertices: np.ndarray,
    faces: np.ndarray,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Unify adjacent face winding without choosing outward orientation.

    Reference TRELLIS runs final cleanup before orientation, then separately
    unifies face orientation. This helper preserves the first face orientation
    in each connected component and only flips neighbors that share an edge in
    the same directed order.
    """
    if len(faces) == 0:
        return vertices, faces

    oriented = np.array(faces, copy=True)
    edge_faces: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]] = {}
    for face_index, face in enumerate(oriented):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            ia = int(a)
            ib = int(b)
            key = (min(ia, ib), max(ia, ib))
            edge_faces.setdefault(key, []).append((face_index, (ia, ib)))

    adjacency: list[list[tuple[int, bool]]] = [[] for _ in range(len(oriented))]
    for entries in edge_faces.values():
        if len(entries) != 2:
            continue
        (face_a, edge_a), (face_b, edge_b) = entries
        same_direction = edge_a == edge_b
        adjacency[face_a].append((face_b, same_direction))
        adjacency[face_b].append((face_a, same_direction))

    flips = np.zeros(len(oriented), dtype=np.bool_)
    seen = np.zeros(len(oriented), dtype=np.bool_)
    for root in range(len(oriented)):
        if seen[root]:
            continue
        seen[root] = True
        stack = [root]
        while stack:
            current = stack.pop()
            for neighbor, same_direction in adjacency[current]:
                if seen[neighbor]:
                    continue
                flips[neighbor] = bool(flips[current] ^ same_direction)
                seen[neighbor] = True
                stack.append(neighbor)

    if flips.any():
        oriented[flips] = oriented[flips][:, [0, 2, 1]]
        if verbose:
            print(f"  Oriented {int(flips.sum()):,} faces by adjacency", flush=True)

    return vertices, oriented.astype(faces.dtype, copy=False)


def _reindex_mesh(vertices, faces):
    """Remove unreferenced vertices and reindex faces."""
    used = np.unique(faces)
    new_idx = np.full(len(vertices), -1, dtype=np.int64)
    new_idx[used] = np.arange(len(used))
    return vertices[used], new_idx[faces].astype(faces.dtype)
