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
    min_component_ratio: float = 0.01,
    fill_max_hole_edges: int = 100,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Clean up a raw decoder mesh.

    Args:
        vertices: [V, 3] float32
        faces: [F, 3] int
        min_component_ratio: Remove components smaller than this fraction of the largest.
        fill_max_hole_edges: Fill boundary loops with fewer edges than this.
        verbose: Print progress.

    Returns:
        cleaned_vertices: [V', 3]
        cleaned_faces: [F', 3]
    """
    original_faces = len(faces)
    original_verts = len(vertices)

    # Step 1: Keep only the largest connected component
    vertices, faces = keep_largest_component(vertices, faces, verbose)

    # Step 2: Fill small holes on the remaining mesh
    vertices, faces = fill_small_holes(vertices, faces, fill_max_hole_edges, verbose)

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
    min_ratio: float = 0.01,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove connected components smaller than min_ratio of the largest,
    plus flat 'floor' components regardless of size."""
    if len(faces) == 0:
        return vertices, faces

    n_faces = len(faces)
    n_components, labels = _face_connected_components(faces, n_faces)

    if n_components <= 1:
        return vertices, faces

    # Count faces per component
    component_sizes = np.bincount(labels)
    largest_idx = component_sizes.argmax()
    largest = component_sizes[largest_idx]
    threshold = int(largest * min_ratio)

    # Identify flat "floor" components: thin bounding box in at least one axis
    floor_components = set()
    for comp_id in range(n_components):
        if comp_id == largest_idx:
            continue  # never remove the largest component
        if component_sizes[comp_id] < threshold:
            continue  # already removed by size threshold
        # Check if this component is flat
        comp_face_mask = labels == comp_id
        comp_verts = vertices[np.unique(faces[comp_face_mask])]
        bbox = comp_verts.max(axis=0) - comp_verts.min(axis=0)
        bbox_sorted = np.sort(bbox)
        # Flat = thinnest dimension is < 10% of the thickest
        if bbox_sorted[0] < bbox_sorted[2] * 0.10:
            floor_components.add(comp_id)

    # Keep: largest component + non-floor components above threshold
    keep_mask = np.zeros(n_faces, dtype=bool)
    for comp_id in range(n_components):
        if component_sizes[comp_id] >= threshold and comp_id not in floor_components:
            keep_mask[labels == comp_id] = True

    kept_faces = faces[keep_mask]
    removed = n_faces - len(kept_faces)
    removed_small = sum(1 for s in component_sizes if s < threshold)
    removed_floors = len(floor_components)

    if verbose and removed > 0:
        parts = []
        if removed_small:
            parts.append(f"{removed_small} small")
        if removed_floors:
            parts.append(f"{removed_floors} flat/floor")
        print(f"  Removed {' + '.join(parts)} components "
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
    max_edges: int = 100,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill boundary loops (holes) with fan triangulation."""
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

    # Fill loops that are small enough
    new_faces = list(faces)
    filled = 0
    for loop in loops:
        if len(loop) > max_edges:
            continue
        # Fan triangulation from the centroid
        centroid = vertices[loop].mean(axis=0)
        centroid_idx = len(vertices)
        vertices = np.vstack([vertices, centroid[None]])
        for i in range(len(loop)):
            v0 = loop[i]
            v1 = loop[(i + 1) % len(loop)]
            new_faces.append([v0, v1, centroid_idx])
        filled += 1

    if verbose and filled > 0:
        print(f"  Filled {filled} holes ({len(loops) - filled} too large, "
              f"max_edges={max_edges})", flush=True)

    return vertices, np.array(new_faces, dtype=faces.dtype)


def _reindex_mesh(vertices, faces):
    """Remove unreferenced vertices and reindex faces."""
    used = np.unique(faces)
    new_idx = np.full(len(vertices), -1, dtype=np.int64)
    new_idx[used] = np.arange(len(used))
    return vertices[used], new_idx[faces].astype(faces.dtype)
