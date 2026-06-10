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
    3. Remove small connected components (fractional threshold, matching
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
        min_component_ratio: Remove components with fewer faces than this
            fraction of the largest component. Default 1e-5 matches the
            reference pipeline.
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
    """Remove connected components smaller than min_ratio of the largest.

    Matches the reference ``cumesh.remove_small_connected_components(1e-5)``.
    Pure size thresholding only — no shape heuristics.
    """
    if len(faces) == 0:
        return vertices, faces

    n_faces = len(faces)
    n_components, labels = _face_connected_components(faces, n_faces)

    if n_components <= 1:
        return vertices, faces

    component_sizes = np.bincount(labels)
    largest = component_sizes.max()
    threshold = int(largest * min_ratio)

    keep_mask = component_sizes[labels] >= threshold

    kept_faces = faces[keep_mask]
    removed = n_faces - len(kept_faces)
    removed_count = (component_sizes < threshold).sum()

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
        for i in range(len(loop)):
            v0 = loop[i]
            v1 = loop[(i + 1) % len(loop)]
            new_faces.append([v0, v1, centroid_idx])
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
    """Remove faces that create non-manifold edges (edges shared by 3+ faces).

    For each non-manifold edge, keeps the two largest adjacent faces and
    removes the rest. Vectorized: uses packed int64 edge keys, np.unique
    for grouping, and batch area computation.
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

    # Face index for each edge
    face_idx = np.tile(np.arange(n_faces, dtype=np.int64), 3)

    # Pack edge pairs into single int64 for fast grouping
    edge_keys = all_edges[:, 0] * (2**32) + all_edges[:, 1]

    # Group by edge key
    sort_order = np.argsort(edge_keys)
    sorted_keys = edge_keys[sort_order]
    sorted_face_idx = face_idx[sort_order]

    # Find group boundaries
    breaks = np.concatenate([
        [0],
        np.where(sorted_keys[1:] != sorted_keys[:-1])[0] + 1,
        [len(sorted_keys)],
    ])

    # Find non-manifold edges (group size > 2)
    group_sizes = np.diff(breaks)
    non_manifold_mask = group_sizes > 2
    if not non_manifold_mask.any():
        return vertices, faces

    # Precompute all face areas
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    face_areas = 0.5 * np.sqrt((cross ** 2).sum(axis=1))

    # For each non-manifold edge group, keep the 2 largest-area faces
    faces_to_remove = set()
    nm_indices = np.where(non_manifold_mask)[0]
    for gi in nm_indices:
        start, end = breaks[gi], breaks[gi + 1]
        group_faces = sorted_face_idx[start:end]
        group_areas = face_areas[group_faces]
        keep_idx = np.argsort(group_areas)[-2:]
        for i in range(len(group_faces)):
            if i not in keep_idx:
                faces_to_remove.add(group_faces[i])

    if not faces_to_remove:
        return vertices, faces

    if verbose:
        print(f"  Removed {len(faces_to_remove)} non-manifold faces", flush=True)

    keep_mask = np.ones(n_faces, dtype=bool)
    keep_mask[list(faces_to_remove)] = False
    return _reindex_mesh(vertices, faces[keep_mask])


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


def _reindex_mesh(vertices, faces):
    """Remove unreferenced vertices and reindex faces."""
    used = np.unique(faces)
    new_idx = np.full(len(vertices), -1, dtype=np.int64)
    new_idx[used] = np.arange(len(used))
    return vertices[used], new_idx[faces].astype(faces.dtype)
