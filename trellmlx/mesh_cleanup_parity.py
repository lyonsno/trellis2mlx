"""Reference cleanup parity contract and scalar mesh witnesses."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


REFERENCE_CLEANUP_CONTRACT: dict[str, Any] = {
    "schema": "trellis2mlx.reference_cleanup_contract.v1",
    "postprocess_source": {
        "path": "/Users/noahlyons/dev/trellis-mac/TRELLIS.2/o-voxel/o_voxel/postprocess.py",
        "line_range": [133, 162],
    },
    "cumesh_simplify_source": {
        "path": "/Users/noahlyons/dev/trellis-mac/deps/mtlmesh/cumesh/cumesh.py",
        "line_range": [320, 355],
        "stop_condition": "break when new_num_face <= target_num_faces",
    },
    "operations": [
        "simplify_coarse",
        "cleanup_initial",
        "simplify_final",
        "cleanup_final",
        "unify_face_orientations",
    ],
    "local_equivalent_operations": {
        "unify_face_orientations": "orient_faces_by_adjacency",
    },
    "source_native_equivalent_operations": {
        "cleanup_initial": "cleanup_source_native",
        "cleanup_final": "cleanup_source_native",
        "unify_face_orientations": "orient_source_native",
    },
    "qem_status": "primitive_choice_only_not_reference_equivalent",
    "qem_probe_status": "probe_only_not_reference_equivalent",
}


def compute_mesh_cleanup_scalars(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> dict[str, int]:
    """Return JSONable topology and orientation scalars for cleanup comparison."""
    face_count = int(len(faces))
    vertex_count = int(len(vertices))
    if face_count == 0:
        return {
            "vertex_count": vertex_count,
            "face_count": 0,
            "component_count": 0,
            "boundary_edge_count": 0,
            "nonmanifold_edge_count": 0,
            "same_direction_shared_edge_count": 0,
        }

    edge_faces: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face in np.asarray(faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            ia = int(a)
            ib = int(b)
            key = (min(ia, ib), max(ia, ib))
            edge_faces.setdefault(key, []).append((ia, ib))

    boundary_edges = 0
    nonmanifold_edges = 0
    same_direction_shared_edges = 0
    adjacency_rows = []
    adjacency_cols = []
    for key, directed_edges in edge_faces.items():
        if len(directed_edges) == 1:
            boundary_edges += 1
        elif len(directed_edges) > 2:
            nonmanifold_edges += 1
        elif directed_edges[0] == directed_edges[1]:
            same_direction_shared_edges += 1
        a, b = key
        adjacency_rows.extend([a, b])
        adjacency_cols.extend([b, a])

    if adjacency_rows:
        graph = sparse.csr_matrix(
            (np.ones(len(adjacency_rows), dtype=np.int8), (adjacency_rows, adjacency_cols)),
            shape=(vertex_count, vertex_count),
        )
        used_vertices = np.unique(np.asarray(faces).reshape(-1))
        subgraph = graph[used_vertices][:, used_vertices]
        component_count, _ = connected_components(subgraph, directed=False)
    else:
        component_count = int(len(np.unique(np.asarray(faces).reshape(-1))))

    return {
        "vertex_count": vertex_count,
        "face_count": face_count,
        "component_count": int(component_count),
        "boundary_edge_count": int(boundary_edges),
        "nonmanifold_edge_count": int(nonmanifold_edges),
        "same_direction_shared_edge_count": int(same_direction_shared_edges),
    }


def build_mesh_cleanup_parity_report(
    *,
    requested_route: str,
    effective_route: str,
    input_vertices: np.ndarray,
    input_faces: np.ndarray,
    output_vertices: np.ndarray,
    output_faces: np.ndarray,
    operation_trace: list[dict[str, Any]],
    reference_backend: dict[str, Any],
    reference_vertices: np.ndarray | None = None,
    reference_faces: np.ndarray | None = None,
    reference_operation_trace: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a durable report payload for cleanup parity experiments."""
    if not requested_route:
        raise ValueError("requested_route must be nonempty")
    if not effective_route:
        raise ValueError("effective_route must be nonempty")
    if "status" not in reference_backend:
        raise ValueError("reference_backend must include status")

    output_scalars = compute_mesh_cleanup_scalars(output_vertices, output_faces)
    report = {
        "schema": "trellis2mlx.mesh_cleanup_parity_report.v1",
        "requested_route": requested_route,
        "effective_route": effective_route,
        "source_contract": REFERENCE_CLEANUP_CONTRACT,
        "reference_backend": dict(reference_backend),
        "input_scalars": compute_mesh_cleanup_scalars(input_vertices, input_faces),
        "output_scalars": output_scalars,
        "operation_trace": [dict(entry) for entry in operation_trace],
    }
    if reference_vertices is not None and reference_faces is not None:
        reference_scalars = compute_mesh_cleanup_scalars(reference_vertices, reference_faces)
        report["reference_scalars"] = reference_scalars
        report["reference_operation_trace"] = [
            dict(entry) for entry in (reference_operation_trace or [])
        ]
        report["comparison"] = {
            "vertex_count_delta_local_minus_reference": (
                output_scalars["vertex_count"] - reference_scalars["vertex_count"]
            ),
            "face_count_delta_local_minus_reference": (
                output_scalars["face_count"] - reference_scalars["face_count"]
            ),
            "component_count_delta_local_minus_reference": (
                output_scalars["component_count"] - reference_scalars["component_count"]
            ),
            "boundary_edge_delta_local_minus_reference": (
                output_scalars["boundary_edge_count"] - reference_scalars["boundary_edge_count"]
            ),
            "nonmanifold_edge_delta_local_minus_reference": (
                output_scalars["nonmanifold_edge_count"] - reference_scalars["nonmanifold_edge_count"]
            ),
            "same_direction_shared_edge_delta_local_minus_reference": (
                output_scalars["same_direction_shared_edge_count"]
                - reference_scalars["same_direction_shared_edge_count"]
            ),
        }
    return report
