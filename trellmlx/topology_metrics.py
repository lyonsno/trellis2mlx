"""Topology metrics for mesh quality assessment.

Computes boundary edges/loops, non-manifold edges, connected components,
degenerate faces, and vertex/face counts. Works on raw numpy arrays or
loaded GLB files.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class TopologyReport:
    """Topology metrics for a single mesh."""
    # Identity
    source_path: str = ""
    route: str = ""  # e.g. "raw", "cleanup-first", "remesh-candidate"
    config: dict = field(default_factory=dict)

    # Counts
    n_vertices: int = 0
    n_faces: int = 0

    # Topology
    n_boundary_edges: int = 0
    n_boundary_loops: int = 0
    boundary_loop_sizes: list = field(default_factory=list)
    n_non_manifold_edges: int = 0
    n_connected_components: int = 0
    component_face_counts: list = field(default_factory=list)
    n_degenerate_faces: int = 0

    # Timing
    compute_time_s: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Topology Report: {self.source_path}",
            f"  Route: {self.route}",
            f"  Config: {self.config}",
            f"  Vertices: {self.n_vertices:,}",
            f"  Faces: {self.n_faces:,}",
            f"  Boundary edges: {self.n_boundary_edges:,}",
            f"  Boundary loops: {self.n_boundary_loops:,}",
        ]
        if self.boundary_loop_sizes:
            lines.append(f"    Loop sizes: {self.boundary_loop_sizes[:20]}"
                         + (" ..." if len(self.boundary_loop_sizes) > 20 else ""))
        lines += [
            f"  Non-manifold edges: {self.n_non_manifold_edges:,}",
            f"  Connected components: {self.n_connected_components:,}",
        ]
        if self.component_face_counts:
            lines.append(f"    Component sizes: {self.component_face_counts[:10]}"
                         + (" ..." if len(self.component_face_counts) > 10 else ""))
        lines += [
            f"  Degenerate faces: {self.n_degenerate_faces:,}",
            f"  Compute time: {self.compute_time_s:.2f}s",
        ]
        return "\n".join(lines)


def compute_topology_metrics(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    source_path: str = "",
    route: str = "",
    config: Optional[dict] = None,
) -> TopologyReport:
    """Compute topology metrics for a mesh.

    Args:
        vertices: [V, 3] float32
        faces: [F, 3] int
        source_path: path to the source file (for report identity)
        route: processing route name (for report identity)
        config: processing config dict (for report identity)

    Returns:
        TopologyReport with all metrics filled.
    """
    t0 = time.perf_counter()

    report = TopologyReport(
        source_path=source_path,
        route=route,
        config=config or {},
        n_vertices=len(vertices),
        n_faces=len(faces),
    )

    if len(faces) == 0:
        report.compute_time_s = time.perf_counter() - t0
        return report

    # Edge counts
    edge_count = _count_edges(faces)

    # Boundary edges: appear in exactly 1 face
    boundary_edges = {e for e, c in edge_count.items() if c == 1}
    report.n_boundary_edges = len(boundary_edges)

    # Boundary loops
    if boundary_edges:
        loops = _trace_boundary_loops(boundary_edges)
        report.n_boundary_loops = len(loops)
        report.boundary_loop_sizes = sorted([len(l) for l in loops], reverse=True)

    # Non-manifold edges: appear in 3+ faces
    report.n_non_manifold_edges = sum(1 for c in edge_count.values() if c > 2)

    # Connected components
    report.n_connected_components, report.component_face_counts = (
        _connected_components(faces)
    )

    # Degenerate faces (zero area)
    report.n_degenerate_faces = _count_degenerate_faces(vertices, faces)

    report.compute_time_s = time.perf_counter() - t0
    return report


def load_mesh_from_glb(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load vertices and faces from a GLB file."""
    import trimesh
    scene = trimesh.load(path, process=False)
    if isinstance(scene, trimesh.Scene):
        meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No triangle meshes found in {path}")
        if len(meshes) == 1:
            mesh = meshes[0]
        else:
            mesh = trimesh.util.concatenate(meshes)
    elif isinstance(scene, trimesh.Trimesh):
        mesh = scene
    else:
        raise ValueError(f"Unexpected type from trimesh.load: {type(scene)}")
    return np.array(mesh.vertices, dtype=np.float32), np.array(mesh.faces, dtype=np.int64)


def load_mesh_from_checkpoint(checkpoint_dir: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load raw mesh from a trellis2mlx checkpoint directory.

    Returns (vertices, faces, metadata) where metadata includes mesh_grid_size.
    """
    import json
    import os

    npz_path = os.path.join(checkpoint_dir, "mesh_raw.npz")
    json_path = os.path.join(checkpoint_dir, "mesh_raw.json")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"No mesh_raw.npz in {checkpoint_dir}")

    data = np.load(npz_path)
    vertices = data["vertices"]
    faces = data["faces"]

    metadata = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            metadata = json.load(f)

    return vertices, faces, metadata


def _count_edges(faces: np.ndarray) -> dict:
    """Count how many faces each edge belongs to."""
    edge_count = {}
    for face in faces:
        for i in range(3):
            e = (min(face[i], face[(i + 1) % 3]), max(face[i], face[(i + 1) % 3]))
            edge_count[e] = edge_count.get(e, 0) + 1
    return edge_count


def _trace_boundary_loops(boundary_edges: set) -> list:
    """Trace closed loops from boundary edges."""
    adj = {}
    for v0, v1 in boundary_edges:
        adj.setdefault(v0, []).append(v1)
        adj.setdefault(v1, []).append(v0)

    visited_edges = set()
    loops = []
    for edge in boundary_edges:
        if edge in visited_edges:
            continue
        loop = [edge[0], edge[1]]
        visited_edges.add(edge)
        while True:
            current = loop[-1]
            neighbors = adj.get(current, [])
            found_next = False
            for n in neighbors:
                e = (min(current, n), max(current, n))
                if e not in visited_edges:
                    visited_edges.add(e)
                    if n == loop[0] and len(loop) >= 3:
                        loops.append(loop)
                        found_next = True
                        break
                    loop.append(n)
                    found_next = True
                    break
            if not found_next:
                break
    return loops


def _connected_components(faces: np.ndarray) -> tuple[int, list]:
    """Compute face-connected components. Returns (count, sorted sizes)."""
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    n_faces = len(faces)
    edges = {}
    for fi, face in enumerate(faces):
        for i in range(3):
            e = (min(face[i], face[(i + 1) % 3]), max(face[i], face[(i + 1) % 3]))
            edges.setdefault(e, []).append(fi)

    rows, cols = [], []
    for face_list in edges.values():
        for i in range(len(face_list)):
            for j in range(i + 1, len(face_list)):
                rows.extend([face_list[i], face_list[j]])
                cols.extend([face_list[j], face_list[i]])

    if not rows:
        return n_faces, [1] * n_faces

    adj = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(n_faces, n_faces),
    )
    n_comp, labels = connected_components(adj, directed=False)
    sizes = sorted(np.bincount(labels).tolist(), reverse=True)
    return n_comp, sizes


def _count_degenerate_faces(vertices: np.ndarray, faces: np.ndarray) -> int:
    """Count faces with zero (or near-zero) area."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.sqrt((cross ** 2).sum(axis=1))
    return int((areas < 1e-12).sum())
