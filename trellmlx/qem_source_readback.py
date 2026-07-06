"""Compare local QEM first-step state against a direct source readback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from trellmlx.simplify_qem_metal import (
    HAS_MLX,
    _build_adjacency,
    _compute_base_costs,
    _compute_qem,
    _topology_check_cpu,
)

if HAS_MLX:
    from trellmlx.simplify_qem_metal import _topology_check_metal


REPORT_SCHEMA = "trellis2mlx.qem_source_readback_compare.v1"
LOCAL_READBACK_SCHEMA = "trellis2mlx.qem_local_step_readback.v1"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def load_mesh_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "vertices" not in data or "faces" not in data:
        raise ValueError(f"mesh NPZ must contain vertices and faces arrays: {path}")
    return (
        np.asarray(data["vertices"], dtype=np.float32),
        np.asarray(data["faces"], dtype=np.int32),
    )


def _as_uint64_props(props: np.ndarray) -> np.ndarray:
    props = np.asarray(props)
    if props.dtype == np.uint64:
        return props.copy()
    if props.dtype == np.int64:
        return props.view(np.uint64).copy()
    return props.astype(np.uint64, copy=False)


def load_source_readback_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    required = {"edges", "costs", "props"}
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"source readback NPZ missing arrays {missing}: {path}")

    source = {
        "edges": np.asarray(data["edges"], dtype=np.int32),
        "costs": np.asarray(data["costs"], dtype=np.float32),
        "props": _as_uint64_props(data["props"]),
    }
    if "qems" in data.files:
        source["qems"] = np.asarray(data["qems"], dtype=np.float32)
    if "boundaries" in data.files:
        source["boundaries"] = np.asarray(data["boundaries"], dtype=np.int32)
    if "terms" in data.files:
        source["terms"] = np.asarray(data["terms"], dtype=np.float32)
    if "status" in data.files:
        source["status"] = np.asarray(data["status"], dtype=np.int32)
    return source


def _pack_cost_keys(costs: np.ndarray) -> np.ndarray:
    cost_bits = np.asarray(costs, dtype=np.float32).view(np.uint32)
    edge_ids = np.arange(len(cost_bits), dtype=np.uint64)
    return (cost_bits.astype(np.uint64) << np.uint64(32)) | edge_ids


def _propagate_cost_keys(
    *,
    edges: np.ndarray,
    vf_offset: np.ndarray,
    vf_data: np.ndarray,
    costs: np.ndarray,
    face_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    props = np.full(face_count, np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
    face_min_edge = np.full(face_count, -1, dtype=np.int32)
    packs = _pack_cost_keys(costs)

    for ei, pack in enumerate(packs):
        v0i, v1i = edges[ei]
        for vi in (v0i, v1i):
            for idx in range(vf_offset[vi], vf_offset[vi + 1]):
                fi = vf_data[idx]
                if pack < props[fi]:
                    props[fi] = pack
                    face_min_edge[fi] = ei

    return props, face_min_edge


def _face_min_edges_from_props(props: np.ndarray) -> np.ndarray:
    props = _as_uint64_props(props)
    face_min_edge = (props & np.uint64(0xFFFFFFFF)).astype(np.int64)
    face_min_edge[props == np.uint64(0xFFFFFFFFFFFFFFFF)] = -1
    return face_min_edge.astype(np.int32)


def _collapsed_edge_ids(
    *,
    edges: np.ndarray,
    vf_offset: np.ndarray,
    vf_data: np.ndarray,
    costs: np.ndarray,
    face_min_edge: np.ndarray,
    collapse_thresh: np.float32,
) -> list[int]:
    collapsed: list[int] = []
    eligible = np.where(np.asarray(costs, dtype=np.float32) <= np.float32(collapse_thresh))[0]

    for ei in eligible:
        v0i, v1i = edges[ei]
        agreed = True
        for vi in (v0i, v1i):
            for idx in range(vf_offset[vi], vf_offset[vi + 1]):
                fi = vf_data[idx]
                if face_min_edge[fi] != ei:
                    agreed = False
                    break
            if not agreed:
                break
        if agreed:
            collapsed.append(int(ei))

    return collapsed


def _removed_faces_count(
    *,
    faces: np.ndarray,
    edges: np.ndarray,
    vf_offset: np.ndarray,
    vf_data: np.ndarray,
    collapsed_edges: list[int],
) -> int:
    faces_kept = np.ones(len(faces), dtype=np.bool_)
    for ei in collapsed_edges:
        v0i, v1i = edges[ei]
        for vi, other in ((v0i, v1i), (v1i, v0i)):
            for idx in range(vf_offset[vi], vf_offset[vi + 1]):
                fi = vf_data[idx]
                if other in faces[fi]:
                    faces_kept[fi] = False
    return int((~faces_kept).sum())


def _collapse_counts(costs: np.ndarray, collapsed_edges: list[int], removed_faces: int, collapse_thresh: np.float32) -> dict[str, int]:
    finite_costs = np.asarray(costs, dtype=np.float32)
    eligible = finite_costs <= np.float32(collapse_thresh)
    return {
        "eligible": int(eligible.sum()),
        "negative": int((finite_costs < 0).sum()),
        "positive_le_threshold": int(((finite_costs >= 0) & eligible).sum()),
        "collapsed_edges": len(collapsed_edges),
        "removed_faces": removed_faces,
    }


def _topology_terms_cpu(
    vertices: np.ndarray,
    faces: np.ndarray,
    edges: np.ndarray,
    v_new: np.ndarray,
    vf_offset: np.ndarray,
    vf_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    skinny_avgs = np.zeros(len(edges), dtype=np.float32)
    status = np.zeros((len(edges), 2), dtype=np.int32)

    for ei, (v0i, v1i) in enumerate(edges):
        vn = v_new[ei]
        skinny_cost = np.float32(0.0)
        num_tri = 0
        flipped = False

        for vi, other in ((v0i, v1i), (v1i, v0i)):
            for idx in range(vf_offset[vi], vf_offset[vi + 1]):
                fi = vf_data[idx]
                fv = faces[fi]
                if other in fv:
                    continue

                fa, fb, fc = vertices[fv[0]], vertices[fv[1]], vertices[fv[2]]
                na = vn if fv[0] == vi else fa
                nb = vn if fv[1] == vi else fb
                nc = vn if fv[2] == vi else fc

                old_n = np.cross(fb - fa, fc - fa)
                new_e1, new_e2 = nb - na, nc - na
                new_n = np.cross(new_e1, new_e2)

                if np.dot(old_n, new_n) < 0.0:
                    flipped = True
                    break

                new_area = np.float32(0.5) * np.float32(np.linalg.norm(new_n))
                new_e0 = nc - nb
                denom = np.float32(np.dot(new_e0, new_e0) + np.dot(new_e1, new_e1) + np.dot(new_e2, new_e2))
                if denom < np.float32(1e-12):
                    denom = np.float32(1e-12)
                shape = np.float32(4.0) * np.float32(np.sqrt(np.float32(3.0))) * new_area / denom
                skinny_cost = np.float32(skinny_cost + (np.float32(1.0) - np.clip(shape, np.float32(0.0), np.float32(1.0))))
                num_tri += 1
            if flipped:
                break

        status[ei, 0] = num_tri
        status[ei, 1] = int(flipped)
        if flipped:
            skinny_avgs[ei] = np.inf
        elif num_tri > 0:
            skinny_avgs[ei] = np.float32(skinny_cost / np.float32(num_tri))

    return skinny_avgs, status


def _qem_cost_from_qems(qems: np.ndarray, edges: np.ndarray, v_new: np.ndarray) -> np.ndarray:
    ev0, ev1 = edges[:, 0], edges[:, 1]
    q = (qems[ev0] + qems[ev1]).astype(np.float32)
    vx, vy, vz = v_new[:, 0], v_new[:, 1], v_new[:, 2]
    return (
        q[:, 0] * vx * vx + 2 * q[:, 1] * vx * vy + 2 * q[:, 2] * vx * vz + 2 * q[:, 3] * vx +
        q[:, 4] * vy * vy + 2 * q[:, 5] * vy * vz + 2 * q[:, 6] * vy +
        q[:, 7] * vz * vz + 2 * q[:, 8] * vz + q[:, 9]
    ).astype(np.float32)


def local_simplify_step_readback(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    lambda_edge_length: float = 1e-2,
    lambda_skinny: float = 1e-3,
    collapse_thresh: np.float32 = np.float32(1e-8),
) -> dict[str, Any]:
    vertices = np.asarray(vertices, dtype=np.float32).copy()
    faces = np.asarray(faces, dtype=np.int32).copy()
    vertex_count = len(vertices)
    face_count = len(faces)

    edges, _edge_count, is_boundary, vf_offset, vf_data = _build_adjacency(faces, vertex_count)
    qems = _compute_qem(vertices, faces, vertex_count)
    base_costs, v_new, edge_len2 = _compute_base_costs(
        vertices,
        edges,
        qems,
        is_boundary,
        lambda_edge_length,
    )

    if HAS_MLX:
        costs = _topology_check_metal(
            vertices,
            faces,
            edges,
            v_new,
            base_costs,
            edge_len2,
            vf_offset,
            vf_data,
            lambda_skinny,
        )
        topology_backend = "mlx-metal"
    else:
        costs = _topology_check_cpu(
            vertices,
            faces,
            edges,
            v_new,
            base_costs,
            edge_len2,
            vf_offset,
            vf_data,
            lambda_skinny,
        )
        topology_backend = "cpu"

    costs = np.asarray(costs, dtype=np.float32)
    qem_costs = _qem_cost_from_qems(qems, edges, v_new)
    skinny_avgs, status = _topology_terms_cpu(vertices, faces, edges, v_new, vf_offset, vf_data)
    skinny_terms = (costs - base_costs.astype(np.float32)).astype(np.float32)
    terms = np.column_stack([
        v_new.astype(np.float32),
        qem_costs,
        edge_len2.astype(np.float32),
        skinny_avgs,
        skinny_terms,
    ]).astype(np.float32)
    props, face_min_edge = _propagate_cost_keys(
        edges=edges,
        vf_offset=vf_offset,
        vf_data=vf_data,
        costs=costs,
        face_count=face_count,
    )
    collapsed = _collapsed_edge_ids(
        edges=edges,
        vf_offset=vf_offset,
        vf_data=vf_data,
        costs=costs,
        face_min_edge=face_min_edge,
        collapse_thresh=np.float32(collapse_thresh),
    )
    removed_faces = _removed_faces_count(
        faces=faces,
        edges=edges,
        vf_offset=vf_offset,
        vf_data=vf_data,
        collapsed_edges=collapsed,
    )

    return {
        "schema": LOCAL_READBACK_SCHEMA,
        "settings": {
            "lambda_edge_length": float(lambda_edge_length),
            "lambda_skinny": float(lambda_skinny),
            "collapse_thresh": float(collapse_thresh),
            "topology_backend": topology_backend,
        },
        "edges": edges.astype(np.int32, copy=False),
        "costs": costs,
        "props": props,
        "qems": qems.astype(np.float32, copy=False),
        "terms": terms,
        "status": status,
        "face_min_edge": face_min_edge,
        "collapsed_edge_ids": collapsed,
        "collapse_counts": _collapse_counts(costs, collapsed, removed_faces, np.float32(collapse_thresh)),
    }


def _source_collapse_state(
    *,
    source: dict[str, np.ndarray],
    faces: np.ndarray,
    vf_offset: np.ndarray,
    vf_data: np.ndarray,
    collapse_thresh: np.float32,
) -> tuple[list[int], dict[str, int]]:
    face_min_edge = _face_min_edges_from_props(source["props"])
    collapsed = _collapsed_edge_ids(
        edges=source["edges"],
        vf_offset=vf_offset,
        vf_data=vf_data,
        costs=source["costs"],
        face_min_edge=face_min_edge,
        collapse_thresh=collapse_thresh,
    )
    removed_faces = _removed_faces_count(
        faces=faces,
        edges=source["edges"],
        vf_offset=vf_offset,
        vf_data=vf_data,
        collapsed_edges=collapsed,
    )
    return collapsed, _collapse_counts(source["costs"], collapsed, removed_faces, collapse_thresh)


def _bit_exact_count(a: np.ndarray, b: np.ndarray) -> int:
    if a.shape != b.shape:
        return 0
    a_bytes = np.ascontiguousarray(a).view(np.uint8).reshape(a.shape + (-1,))
    b_bytes = np.ascontiguousarray(b).view(np.uint8).reshape(b.shape + (-1,))
    return int((a_bytes == b_bytes).all(axis=-1).sum())


def _float_summary(local: np.ndarray, source: np.ndarray) -> dict[str, Any]:
    local = np.asarray(local, dtype=np.float32)
    source = np.asarray(source, dtype=np.float32)
    same_shape = local.shape == source.shape
    paired = min(local.size, source.size)
    local_flat = local.reshape(-1)[:paired]
    source_flat = source.reshape(-1)[:paired]
    finite_pairs = np.isfinite(local_flat) & np.isfinite(source_flat)
    if finite_pairs.any():
        diff = np.abs(source_flat[finite_pairs] - local_flat[finite_pairs])
        max_abs = float(diff.max())
        mean_abs = float(diff.mean())
    else:
        max_abs = 0.0
        mean_abs = 0.0
    nonfinite_pairs = paired - int(finite_pairs.sum())
    nonfinite_exact = int(((local_flat == source_flat) & ~finite_pairs).sum()) if paired else 0
    return {
        "same_shape": same_shape,
        "local_count": int(local.size),
        "source_count": int(source.size),
        "source_vs_local_bit_exact_edges": _bit_exact_count(local, source) if same_shape else 0,
        "finite_pair_count": int(finite_pairs.sum()),
        "nonfinite_pair_count": nonfinite_pairs,
        "nonfinite_exact_pair_count": nonfinite_exact,
        "source_vs_local_max_abs_finite_diff": max_abs,
        "source_vs_local_mean_abs_finite_diff": mean_abs,
    }


def _float_entry_summary(local: np.ndarray, source: np.ndarray) -> dict[str, Any]:
    local = np.asarray(local, dtype=np.float32)
    source = np.asarray(source, dtype=np.float32)
    same_shape = local.shape == source.shape
    paired = min(local.size, source.size)
    local_flat = local.reshape(-1)[:paired]
    source_flat = source.reshape(-1)[:paired]
    finite_pairs = np.isfinite(local_flat) & np.isfinite(source_flat)
    if finite_pairs.any():
        diff = np.abs(source_flat[finite_pairs] - local_flat[finite_pairs])
        max_abs = float(diff.max())
        mean_abs = float(diff.mean())
    else:
        max_abs = 0.0
        mean_abs = 0.0
    exact = int((local_flat.view(np.uint32) == source_flat.view(np.uint32)).sum()) if same_shape else 0
    return {
        "same_shape": same_shape,
        "local_entries": int(local.size),
        "source_entries": int(source.size),
        "bit_exact_entries": exact,
        "finite_pair_count": int(finite_pairs.sum()),
        "nonfinite_pair_count": int(paired - finite_pairs.sum()),
        "max_abs_finite_diff": max_abs,
        "mean_abs_finite_diff": mean_abs,
    }


def _term_summary(local: dict[str, Any], source: dict[str, np.ndarray], lambda_edge_length: float) -> dict[str, Any]:
    if "terms" not in source:
        return {"available": False}

    local_terms = np.asarray(local["terms"], dtype=np.float32)
    source_terms = np.asarray(source["terms"], dtype=np.float32)
    if local_terms.shape != source_terms.shape or local_terms.ndim != 2 or local_terms.shape[1] < 7:
        return {
            "available": True,
            "same_shape": False,
            "local_shape": list(local_terms.shape),
            "source_shape": list(source_terms.shape),
        }

    local_base = (local_terms[:, 3] + np.float32(lambda_edge_length) * local_terms[:, 4]).astype(np.float32)
    source_base = (source_terms[:, 3] + np.float32(lambda_edge_length) * source_terms[:, 4]).astype(np.float32)
    local_skinny = local_terms[:, 6]
    source_skinny = source_terms[:, 6]
    finite = (
        np.isfinite(local_base)
        & np.isfinite(source_base)
        & np.isfinite(local_skinny)
        & np.isfinite(source_skinny)
    )
    base_diff = np.zeros_like(local_base, dtype=np.float32)
    skinny_diff = np.zeros_like(local_skinny, dtype=np.float32)
    base_diff[finite] = np.abs(source_base[finite] - local_base[finite])
    skinny_diff[finite] = np.abs(source_skinny[finite] - local_skinny[finite])

    qem_eval_split: dict[str, Any] = {
        "source_qem_local_eval_vs_source_qem_cost": {"available": False},
    }
    if "qems" in source:
        source_qem_eval = _qem_cost_from_qems(
            np.asarray(source["qems"], dtype=np.float32),
            np.asarray(source["edges"], dtype=np.int32),
            source_terms[:, :3],
        )
        qem_eval_split["source_qem_local_eval_vs_source_qem_cost"] = {
            "available": True,
            **_float_summary(source_qem_eval, source_terms[:, 3]),
        }

    return {
        "available": True,
        "same_shape": True,
        "collapse_position": _float_entry_summary(local_terms[:, :3], source_terms[:, :3]),
        "qem_cost": _float_summary(local_terms[:, 3], source_terms[:, 3]),
        "edge_length2": _float_summary(local_terms[:, 4], source_terms[:, 4]),
        "skinny_avg": _float_summary(local_terms[:, 5], source_terms[:, 5]),
        "skinny_term": _float_summary(local_terms[:, 6], source_terms[:, 6]),
        "attribution": {
            "finite_all_edges": int(finite.sum()),
            "base_diff_ge_skinny_diff_edges": int((base_diff[finite] >= skinny_diff[finite]).sum()),
            "skinny_diff_gt_base_diff_edges": int((skinny_diff[finite] > base_diff[finite]).sum()),
            "base_diff_mean": float(base_diff[finite].mean()) if finite.any() else 0.0,
            "skinny_diff_mean": float(skinny_diff[finite].mean()) if finite.any() else 0.0,
            "base_diff_max": float(base_diff[finite].max()) if finite.any() else 0.0,
            "skinny_diff_max": float(skinny_diff[finite].max()) if finite.any() else 0.0,
        },
        "qem_eval_split": qem_eval_split,
    }


def _qem_summary(local_qems: np.ndarray, source_qems: np.ndarray | None) -> dict[str, Any]:
    if source_qems is None:
        return {"available": False}
    local_qems = np.asarray(local_qems, dtype=np.float32)
    source_qems = np.asarray(source_qems, dtype=np.float32)
    if local_qems.shape != source_qems.shape:
        return {
            "available": True,
            "same_shape": False,
            "local_shape": list(local_qems.shape),
            "source_shape": list(source_qems.shape),
        }
    diff = np.abs(source_qems - local_qems)
    return {
        "available": True,
        "same_shape": True,
        "total_entries": int(local_qems.size),
        "bit_exact_entries": _bit_exact_count(local_qems.reshape(-1), source_qems.reshape(-1)),
        "vertices_with_any_qem_bit_diff": int((diff != 0).any(axis=1).sum()) if local_qems.ndim == 2 else None,
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
    }


def build_qem_source_readback_report(
    *,
    requested_route: str,
    effective_route: str,
    mesh_path: Path,
    source_readback_path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    source: dict[str, np.ndarray],
    lambda_edge_length: float = 1e-2,
    lambda_skinny: float = 1e-3,
    collapse_thresh: np.float32 = np.float32(1e-8),
) -> dict[str, Any]:
    if not requested_route:
        raise ValueError("requested_route must be non-empty")
    if not effective_route:
        raise ValueError("effective_route must be non-empty")

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    local = local_simplify_step_readback(
        vertices,
        faces,
        lambda_edge_length=lambda_edge_length,
        lambda_skinny=lambda_skinny,
        collapse_thresh=collapse_thresh,
    )
    _edges, _edge_count, _is_boundary, vf_offset, vf_data = _build_adjacency(faces, len(vertices))

    source_collapsed, source_counts = _source_collapse_state(
        source=source,
        faces=faces,
        vf_offset=vf_offset,
        vf_data=vf_data,
        collapse_thresh=np.float32(collapse_thresh),
    )
    local_collapsed = local["collapsed_edge_ids"]

    edge_order_exact = (
        local["edges"].shape == source["edges"].shape
        and bool(np.array_equal(local["edges"], source["edges"]))
    )
    props_exact = (
        local["props"].shape == source["props"].shape
        and bool(np.array_equal(local["props"], source["props"]))
    )

    return {
        "schema": REPORT_SCHEMA,
        "status": "ok",
        "requested_route": requested_route,
        "effective_route": effective_route,
        "asset": {
            "mesh_path": str(mesh_path),
            "source_readback_path": str(source_readback_path),
            "input_vertices": int(len(vertices)),
            "input_faces": int(len(faces)),
        },
        "settings": {
            "lambda_edge_length": float(lambda_edge_length),
            "lambda_skinny": float(lambda_skinny),
            "collapse_thresh": float(collapse_thresh),
            "local_topology_backend": local["settings"]["topology_backend"],
        },
        "identity": {
            "edge_order_exact": edge_order_exact,
            "local_edge_count": int(len(local["edges"])),
            "source_edge_count": int(len(source["edges"])),
            "propagated_face_keys_exact": props_exact,
            "local_prop_count": int(len(local["props"])),
            "source_prop_count": int(len(source["props"])),
        },
        "collapse_counts": {
            "local_costs": local["collapse_counts"],
            "source_costs": source_counts,
        },
        "collapse_identity": {
            "collapsed_edge_ids_exact": local_collapsed == source_collapsed,
            "local_collapsed_edge_ids": local_collapsed,
            "source_collapsed_edge_ids": source_collapsed,
            "local_only_collapsed_edge_ids": sorted(set(local_collapsed) - set(source_collapsed)),
            "source_only_collapsed_edge_ids": sorted(set(source_collapsed) - set(local_collapsed)),
        },
        "cost_summary": _float_summary(local["costs"], source["costs"]),
        "term_summary": _term_summary(local, source, lambda_edge_length),
        "qem_summary": _qem_summary(local["qems"], source.get("qems")),
    }


def failure_report(
    *,
    requested_route: str,
    effective_route: str,
    failure_phase: str,
    error: Exception,
    mesh_path: Path | None,
    source_readback_path: Path | None,
) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "status": "failed",
        "requested_route": requested_route,
        "effective_route": effective_route,
        "failure_phase": failure_phase,
        "error": f"{type(error).__name__}: {error}",
        "asset": {
            "mesh_path": str(mesh_path) if mesh_path is not None else None,
            "source_readback_path": str(source_readback_path) if source_readback_path is not None else None,
        },
        "last_trustworthy_evidence": {
            "report_written": True,
            "route_identity_known": bool(requested_route and effective_route),
        },
    }


__all__ = [
    "REPORT_SCHEMA",
    "LOCAL_READBACK_SCHEMA",
    "_jsonable",
    "build_qem_source_readback_report",
    "failure_report",
    "load_mesh_npz",
    "load_source_readback_npz",
    "local_simplify_step_readback",
]
