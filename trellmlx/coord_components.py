"""Connected-component filtering for sparse coordinate supports."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


_NEIGHBORS_3D_6 = np.array(
    [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ],
    dtype=np.int32,
)


def filter_sparse_coordinate_components(
    coords: np.ndarray,
    features: np.ndarray | None = None,
    *,
    mode: str = "none",
    min_component_ratio: float = 1e-5,
    include_row_indices: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Filter disconnected sparse coordinate support.

    Args:
        coords: ``[N, 3]`` spatial coords or ``[N, 4]`` batch+spatial coords.
        features: Optional row-aligned feature/latent tensor with first dim ``N``.
        mode: ``"none"`` reports components without dropping rows,
            ``"largest"`` keeps only the largest component, and ``"min_ratio"``
            keeps components whose size is at least ``largest * min_component_ratio``.
        min_component_ratio: Fractional threshold for ``mode="min_ratio"``.

    Returns:
        ``(filtered_coords, filtered_features, report)``. Row order is preserved.
    """
    coords_np = np.asarray(coords)
    if coords_np.ndim != 2 or coords_np.shape[1] not in (3, 4):
        raise ValueError(f"coords must have shape [N,3] or [N,4], got {list(coords_np.shape)}")
    if mode not in {"none", "largest", "min_ratio"}:
        raise ValueError(f"unknown component filter mode: {mode!r}")
    if min_component_ratio < 0:
        raise ValueError("min_component_ratio must be non-negative")

    features_np = None if features is None else np.asarray(features)
    if features_np is not None and len(features_np) != len(coords_np):
        raise ValueError(
            f"feature row count {len(features_np)} does not match coord count {len(coords_np)}"
        )

    components = _coordinate_components(coords_np.astype(np.int32, copy=False))
    component_sizes = [len(component["rows"]) for component in components]
    keep_component_ids = _select_components(
        component_sizes,
        mode=mode,
        min_component_ratio=min_component_ratio,
    )
    keep_mask = np.ones(len(coords_np), dtype=bool)
    if mode != "none":
        keep_mask[:] = False
        for component_id in keep_component_ids:
            keep_mask[components[component_id]["rows"]] = True

    filtered_coords = coords_np[keep_mask]
    filtered_features = None if features_np is None else features_np[keep_mask]
    kept_row_indices = np.where(keep_mask)[0].astype(np.int64).tolist()
    report = {
        "mode": mode,
        "applied": mode != "none",
        "connectivity": 6,
        "coord_width": int(coords_np.shape[1]),
        "input_count": int(len(coords_np)),
        "kept_count": int(keep_mask.sum()),
        "dropped_count": int(len(coords_np) - keep_mask.sum()),
        "component_count": int(len(components)),
        "component_sizes": component_sizes,
        "kept_component_ids": keep_component_ids,
        "kept_row_indices_count": len(kept_row_indices),
        "largest_fraction": (
            float(component_sizes[0] / len(coords_np))
            if len(coords_np) and component_sizes
            else 0.0
        ),
        "min_component_ratio": float(min_component_ratio),
        "component_bounds": [component["bounds"] for component in components],
    }
    if include_row_indices:
        report["kept_row_indices"] = kept_row_indices
    return filtered_coords, filtered_features, report


def _select_components(
    component_sizes: list[int],
    *,
    mode: str,
    min_component_ratio: float,
) -> list[int]:
    if not component_sizes:
        return []
    if mode == "none":
        return list(range(len(component_sizes)))
    if mode == "largest":
        return [0]
    largest = component_sizes[0]
    threshold = largest * min_component_ratio
    return [
        component_id
        for component_id, size in enumerate(component_sizes)
        if size >= threshold
    ]


def _coordinate_components(coords: np.ndarray) -> list[dict[str, Any]]:
    if len(coords) == 0:
        return []

    point_to_rows: dict[tuple[int, ...], list[int]] = {}
    for row_index, coord in enumerate(coords):
        point_to_rows.setdefault(tuple(int(value) for value in coord.tolist()), []).append(row_index)

    points = set(point_to_rows)
    visited: set[tuple[int, ...]] = set()
    components = []
    for point in sorted(points):
        if point in visited:
            continue
        queue = deque([point])
        visited.add(point)
        component_points = []
        component_rows = []
        while queue:
            current = queue.popleft()
            component_points.append(current)
            component_rows.extend(point_to_rows[current])
            for neighbor in _neighbors(current):
                if neighbor in points and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        component_rows.sort()
        spatial = np.array(
            [point[-3:] for point in component_points],
            dtype=np.int32,
        )
        components.append(
            {
                "rows": np.array(component_rows, dtype=np.int64),
                "first_row": component_rows[0],
                "bounds": [
                    spatial.min(axis=0).tolist(),
                    spatial.max(axis=0).tolist(),
                ],
            }
        )

    components.sort(key=lambda item: (-len(item["rows"]), item["first_row"]))
    return components


def _neighbors(point: tuple[int, ...]) -> list[tuple[int, ...]]:
    prefix = point[:-3]
    spatial = np.array(point[-3:], dtype=np.int32)
    return [
        prefix + tuple((spatial + delta).tolist())
        for delta in _NEIGHBORS_3D_6
    ]


__all__ = ["filter_sparse_coordinate_components"]
