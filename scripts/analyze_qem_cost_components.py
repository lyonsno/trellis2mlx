"""Attribute CUDA/Metal edge-cost deltas on backend-self-consistent edges."""

from __future__ import annotations

from typing import Any

import numpy as np


COMPONENT_NAMES = (
    "qem_costs",
    "edge_length2",
    "skinny_avgs",
    "skinny_terms",
)
REQUIRED_NAMES = {
    "qems",
    "edge_collapse_costs",
    "component_edge_collapse_costs",
    *COMPONENT_NAMES,
}


def _bit_mismatch(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return left.view(np.uint32) != right.view(np.uint32)


def _normalize(
    arrays: dict[str, np.ndarray],
    *,
    backend: str,
) -> dict[str, np.ndarray]:
    missing = REQUIRED_NAMES - set(arrays)
    if missing:
        raise ValueError(f"{backend} arrays lack {sorted(missing)}")
    normalized = {
        name: np.ascontiguousarray(np.asarray(arrays[name]))
        for name in REQUIRED_NAMES
    }
    for name, array in normalized.items():
        if array.dtype != np.float32:
            raise ValueError(f"{backend} {name} must be float32")
    edge_shape = normalized["edge_collapse_costs"].shape
    if len(edge_shape) != 1 or not edge_shape[0]:
        raise ValueError(f"{backend} edge costs must be a nonempty vector")
    for name in REQUIRED_NAMES - {"qems"}:
        if normalized[name].shape != edge_shape:
            raise ValueError(f"{backend} {name} shape differs from edge costs")
    if normalized["qems"].ndim != 2 or normalized["qems"].shape[1] != 10:
        raise ValueError(f"{backend} qems must have shape [V, 10]")
    return normalized


def analyze_component_arrays(
    cuda_arrays: dict[str, np.ndarray],
    metal_arrays: dict[str, np.ndarray],
    *,
    collapse_threshold: float = 1e-7,
) -> dict[str, Any]:
    collapse_threshold = float(collapse_threshold)
    if not np.isfinite(collapse_threshold) or collapse_threshold <= 0:
        raise ValueError("collapse_threshold must be finite and positive")
    cuda = _normalize(cuda_arrays, backend="CUDA")
    metal = _normalize(metal_arrays, backend="Metal")
    if cuda["qems"].shape != metal["qems"].shape:
        raise ValueError("CUDA and Metal QEM shapes differ")
    if (
        cuda["edge_collapse_costs"].shape
        != metal["edge_collapse_costs"].shape
    ):
        raise ValueError("CUDA and Metal edge-cost shapes differ")

    cuda_rejected = _bit_mismatch(
        cuda["edge_collapse_costs"],
        cuda["component_edge_collapse_costs"],
    )
    metal_rejected = _bit_mismatch(
        metal["edge_collapse_costs"],
        metal["component_edge_collapse_costs"],
    )
    admitted = ~(cuda_rejected | metal_rejected)
    final_mismatch = _bit_mismatch(
        cuda["edge_collapse_costs"],
        metal["edge_collapse_costs"],
    )
    admitted_final_mismatch = admitted & final_mismatch

    components: dict[str, dict[str, int]] = {}
    component_union = np.zeros(admitted.shape, dtype=bool)
    for name in COMPONENT_NAMES:
        mismatch = admitted & _bit_mismatch(cuda[name], metal[name])
        component_union |= mismatch
        components[name] = {
            "joint_mismatch_count": int(np.count_nonzero(mismatch)),
            "final_mismatch_overlap": int(
                np.count_nonzero(mismatch & final_mismatch)
            ),
            "final_exact_overlap": int(
                np.count_nonzero(mismatch & ~final_mismatch)
            ),
        }

    qem_mismatch = _bit_mismatch(cuda["qems"], metal["qems"])
    cuda_eligible = admitted & (
        cuda["edge_collapse_costs"] <= collapse_threshold
    )
    metal_eligible = admitted & (
        metal["edge_collapse_costs"] <= collapse_threshold
    )
    eligibility_crossings = cuda_eligible ^ metal_eligible
    covered = admitted_final_mismatch & component_union
    uncovered = admitted_final_mismatch & ~component_union
    return {
        "schema": "trellis2mlx.qem_cost_component_attribution.v1",
        "edges": int(admitted.size),
        "admission": {
            "cuda_rejected": int(np.count_nonzero(cuda_rejected)),
            "metal_rejected": int(np.count_nonzero(metal_rejected)),
            "joint_admitted": int(np.count_nonzero(admitted)),
        },
        "qems": {
            "coefficients": int(qem_mismatch.size),
            "bit_mismatch_count": int(np.count_nonzero(qem_mismatch)),
        },
        "collapse_threshold": {
            "value": collapse_threshold,
            "cuda_eligible": int(np.count_nonzero(cuda_eligible)),
            "metal_eligible": int(np.count_nonzero(metal_eligible)),
            "joint_eligible": int(
                np.count_nonzero(cuda_eligible & metal_eligible)
            ),
            "eligibility_crossings": int(
                np.count_nonzero(eligibility_crossings)
            ),
            "cuda_only_eligible": int(
                np.count_nonzero(cuda_eligible & ~metal_eligible)
            ),
            "metal_only_eligible": int(
                np.count_nonzero(metal_eligible & ~cuda_eligible)
            ),
        },
        "final_cost": {
            "all_mismatch_count": int(np.count_nonzero(final_mismatch)),
            "joint_mismatch_count": int(
                np.count_nonzero(admitted_final_mismatch)
            ),
            "excluded_mismatch_count": int(
                np.count_nonzero(final_mismatch & ~admitted)
            ),
        },
        "components": components,
        "component_union": {
            "joint_mismatch_count": int(np.count_nonzero(component_union)),
            "final_mismatch_covered": int(np.count_nonzero(covered)),
            "final_mismatch_uncovered": int(np.count_nonzero(uncovered)),
        },
    }
