"""Trace raw dual-grid triangles back to decoder edge/quads.

This assay distinguishes triangulation and cleanup defects from topology that is
already requested by the decoder field.  It reconstructs the exact extractor
face rows, removes each quad's internal diagonal, and solves orientation parity
on the resulting quad-boundary graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_orientation_semantics import _solve_parity_components
from trellmlx.mesh_extract import (
    _EDGE_NEIGHBOR_OFFSETS,
    _QUAD_SPLIT_1,
    _QUAD_SPLIT_2,
    _softplus,
)


SCHEMA = "trellis2mlx.dual_grid_quad_provenance.v1"
_PACK_BITS = 21
_PACK_LIMIT = 1 << _PACK_BITS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pack_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape [N, 3], got {coords.shape}")
    if coords.size and (
        int(coords.min()) < 0 or int(coords.max()) >= _PACK_LIMIT
    ):
        raise ValueError(
            f"coords must be in [0, {_PACK_LIMIT}), got "
            f"[{int(coords.min())}, {int(coords.max())}]"
        )
    return (
        (coords[:, 0] << (2 * _PACK_BITS))
        | (coords[:, 1] << _PACK_BITS)
        | coords[:, 2]
    )


def reconstruct_quads(
    feats: np.ndarray,
    coords: np.ndarray,
    *,
    lookup_chunk_size: int = 250_000,
) -> dict[str, np.ndarray | int]:
    """Reconstruct valid source quads and their raw triangle rows."""
    feats = np.asarray(feats, dtype=np.float32)
    coords = np.asarray(coords)
    if feats.ndim != 2 or feats.shape[1] != 7:
        raise ValueError(f"feats must have shape [N, 7], got {feats.shape}")
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"coords must have shape [N, 4], got {coords.shape}")
    if len(feats) != len(coords):
        raise ValueError("feats and coords row counts differ")
    if lookup_chunk_size <= 0:
        raise ValueError("lookup_chunk_size must be positive")

    spatial = np.asarray(coords[:, 1:4], dtype=np.int64)
    packed = _pack_coords(spatial)
    order = np.argsort(packed, kind="stable")
    sorted_keys = packed[order]
    duplicate_coords = int(np.count_nonzero(sorted_keys[1:] == sorted_keys[:-1]))
    if duplicate_coords:
        raise ValueError(
            f"decoder coordinates are not unique ({duplicate_coords} duplicates)"
        )

    active_voxels, active_axes = np.nonzero(feats[:, 3:6] > 0)
    active_voxels = active_voxels.astype(np.int64, copy=False)
    active_axes = active_axes.astype(np.int8, copy=False)
    quad_chunks: list[np.ndarray] = []
    voxel_chunks: list[np.ndarray] = []
    axis_chunks: list[np.ndarray] = []
    missing_sentinel = np.int64(-1)

    for start in range(0, len(active_voxels), lookup_chunk_size):
        stop = min(start + lookup_chunk_size, len(active_voxels))
        voxels = active_voxels[start:stop]
        axes = active_axes[start:stop]
        neighbor_coords = (
            spatial[voxels, None, :]
            + _EDGE_NEIGHBOR_OFFSETS[axes.astype(np.int64)]
        )
        neighbor_keys = _pack_coords(neighbor_coords.reshape(-1, 3))
        positions = np.searchsorted(sorted_keys, neighbor_keys)
        in_range = positions < len(sorted_keys)
        found = np.zeros(len(positions), dtype=bool)
        found[in_range] = sorted_keys[positions[in_range]] == neighbor_keys[in_range]
        indices = np.full(len(positions), missing_sentinel, dtype=np.int64)
        indices[found] = order[positions[found]]
        indices = indices.reshape(-1, 4)
        valid = np.all(indices != missing_sentinel, axis=1)
        if np.any(valid):
            quad_chunks.append(indices[valid])
            voxel_chunks.append(voxels[valid])
            axis_chunks.append(axes[valid])

    if quad_chunks:
        quads = np.concatenate(quad_chunks)
        source_voxels = np.concatenate(voxel_chunks)
        source_axes = np.concatenate(axis_chunks)
    else:
        quads = np.empty((0, 4), dtype=np.int64)
        source_voxels = np.empty(0, dtype=np.int64)
        source_axes = np.empty(0, dtype=np.int8)

    weights = _softplus(feats[:, 6])
    split_02 = (
        weights[quads[:, 0]] * weights[quads[:, 2]]
        > weights[quads[:, 1]] * weights[quads[:, 3]]
    )
    templates = np.where(
        split_02[:, None],
        _QUAD_SPLIT_1[None, :],
        _QUAD_SPLIT_2[None, :],
    )
    faces = np.take_along_axis(quads, templates, axis=1).reshape(-1, 3)
    return {
        "quads": quads,
        "source_voxels": source_voxels,
        "source_axes": source_axes,
        "source_coords": spatial[source_voxels],
        "split_02": split_02,
        "faces": faces,
        "active_edges": int(len(active_voxels)),
        "invalid_active_edges": int(len(active_voxels) - len(quads)),
    }


def _axis_pair_histogram(
    axis_a: np.ndarray,
    axis_b: np.ndarray,
) -> dict[str, int]:
    if len(axis_a) == 0:
        return {}
    low = np.minimum(axis_a, axis_b).astype(np.int64)
    high = np.maximum(axis_a, axis_b).astype(np.int64)
    values = low * 3 + high
    counts = np.bincount(values, minlength=9)
    return {
        f"{index // 3}-{index % 3}": int(count)
        for index, count in enumerate(counts)
        if count
    }


def _duplicate_quad_report(
    quads: np.ndarray,
    source_axes: np.ndarray,
    source_coords: np.ndarray,
    max_witnesses: int,
) -> dict[str, Any]:
    if len(quads) == 0:
        return {"groups": 0, "quads": 0, "witnesses": []}
    canonical = np.ascontiguousarray(np.sort(quads, axis=1))
    key_dtype = np.dtype((np.void, canonical.dtype.itemsize * 4))
    keys = canonical.view(key_dtype).reshape(-1)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    duplicate_group_ids = np.flatnonzero((ends - starts) > 1)
    witnesses = []
    for group_id in duplicate_group_ids[:max_witnesses]:
        start = int(starts[group_id])
        end = int(ends[group_id])
        quad_ids = order[start:end]
        witnesses.append(
            {
                "canonical_vertices": [int(value) for value in canonical[quad_ids[0]]],
                "quad_ids": [int(value) for value in quad_ids],
                "source_axes": [int(value) for value in source_axes[quad_ids]],
                "source_coords": source_coords[quad_ids].astype(int).tolist(),
            }
        )
    counts = ends - starts
    return {
        "groups": int(np.count_nonzero(counts > 1)),
        "quads": int(counts[counts > 1].sum()),
        "witnesses": witnesses,
    }


def analyze_quad_topology(
    quads: np.ndarray,
    source_axes: np.ndarray,
    source_coords: np.ndarray,
    *,
    max_witnesses: int = 64,
) -> dict[str, Any]:
    """Analyze topology after removing triangulation-internal diagonals."""
    quads = np.asarray(quads, dtype=np.int64)
    source_axes = np.asarray(source_axes, dtype=np.int8)
    source_coords = np.asarray(source_coords, dtype=np.int64)
    if quads.ndim != 2 or quads.shape[1] != 4:
        raise ValueError(f"quads must have shape [Q, 4], got {quads.shape}")
    if source_axes.shape != (len(quads),):
        raise ValueError("source_axes must have one value per quad")
    if source_coords.shape != (len(quads), 3):
        raise ValueError("source_coords must have shape [Q, 3]")
    if max_witnesses < 0:
        raise ValueError("max_witnesses must be nonnegative")
    if len(quads) == 0:
        return {
            "quads": 0,
            "edge_groups": 0,
            "boundary_edges": 0,
            "manifold_edges": 0,
            "nonmanifold_edges": 0,
            "same_direction_manifold_edges": 0,
            "opposite_direction_manifold_edges": 0,
            "orientation": {
                "components": 0,
                "contradictory_components": 0,
                "contradictory_quads": 0,
            },
            "duplicate_quads": {"groups": 0, "quads": 0, "witnesses": []},
            "witnesses": {"nonmanifold_edges": [], "parity_violations": []},
        }

    max_vertex = int(quads.max()) + 1
    sources = np.concatenate((quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]))
    targets = np.concatenate((quads[:, 1], quads[:, 2], quads[:, 3], quads[:, 0]))
    quad_ids = np.tile(np.arange(len(quads), dtype=np.int64), 4)
    low = np.minimum(sources, targets)
    high = np.maximum(sources, targets)
    edge_keys = low * np.int64(max_vertex) + high
    order = np.argsort(edge_keys, kind="stable")
    sorted_keys = edge_keys[order]
    sorted_quads = quad_ids[order]
    sorted_directions = (sources < targets)[order]
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    counts = ends - starts
    boundary_starts = starts[counts == 1]
    manifold_starts = starts[counts == 2]
    nonmanifold_starts = starts[counts > 2]

    quad_a = sorted_quads[manifold_starts]
    quad_b = sorted_quads[manifold_starts + 1]
    same_direction = (
        sorted_directions[manifold_starts]
        == sorted_directions[manifold_starts + 1]
    )
    required_xor = same_direction.astype(np.uint8)
    solved = _solve_parity_components(
        len(quads), quad_a, quad_b, required_xor
    )
    parity = solved["_parity"]
    violated = (parity[quad_a] ^ parity[quad_b]) != required_xor

    nonmanifold_witnesses = []
    for start in nonmanifold_starts[:max_witnesses]:
        end = ends[np.searchsorted(starts, start)]
        incident = sorted_quads[start:end]
        key = int(sorted_keys[start])
        nonmanifold_witnesses.append(
            {
                "vertices": [key // max_vertex, key % max_vertex],
                "quad_ids": [int(value) for value in incident],
                "source_axes": [int(value) for value in source_axes[incident]],
                "source_coords": source_coords[incident].astype(int).tolist(),
                "directions_low_to_high": [
                    bool(value) for value in sorted_directions[start:end]
                ],
            }
        )

    violation_ids = np.flatnonzero(violated)[:max_witnesses]
    parity_witnesses = []
    for constraint_id in violation_ids:
        start = int(manifold_starts[constraint_id])
        left = int(quad_a[constraint_id])
        right = int(quad_b[constraint_id])
        key = int(sorted_keys[start])
        parity_witnesses.append(
            {
                "vertices": [key // max_vertex, key % max_vertex],
                "quad_ids": [left, right],
                "source_axes": [int(source_axes[left]), int(source_axes[right])],
                "source_coords": [
                    source_coords[left].astype(int).tolist(),
                    source_coords[right].astype(int).tolist(),
                ],
                "same_directed_traversal": bool(same_direction[constraint_id]),
                "required_xor": int(required_xor[constraint_id]),
                "assigned_xor": int(parity[left] ^ parity[right]),
            }
        )

    return {
        "quads": int(len(quads)),
        "edge_groups": int(len(starts)),
        "boundary_edges": int(len(boundary_starts)),
        "manifold_edges": int(len(manifold_starts)),
        "nonmanifold_edges": int(len(nonmanifold_starts)),
        "same_direction_manifold_edges": int(same_direction.sum()),
        "opposite_direction_manifold_edges": int((~same_direction).sum()),
        "manifold_axis_pairs": _axis_pair_histogram(
            source_axes[quad_a], source_axes[quad_b]
        ),
        "same_direction_axis_pairs": _axis_pair_histogram(
            source_axes[quad_a[same_direction]],
            source_axes[quad_b[same_direction]],
        ),
        "orientation": {
            "components": solved["component_count"],
            "orientable_components": solved["orientable_component_count"],
            "contradictory_components": solved["contradictory_component_count"],
            "orientable_quads": solved["orientable_face_count"],
            "contradictory_quads": solved["contradictory_face_count"],
            "violated_constraints_in_canonical_assignment": int(violated.sum()),
            "violated_axis_pairs": _axis_pair_histogram(
                source_axes[quad_a[violated]], source_axes[quad_b[violated]]
            ),
        },
        "duplicate_quads": _duplicate_quad_report(
            quads, source_axes, source_coords, max_witnesses
        ),
        "witnesses": {
            "nonmanifold_edges": nonmanifold_witnesses,
            "parity_violations": parity_witnesses,
        },
    }


def _verify_internal_diagonals(faces: np.ndarray) -> dict[str, Any]:
    if len(faces) % 2:
        return {"exactly_two_faces_per_quad": False, "opposite_directed": False}
    first = faces[0::2]
    second = faces[1::2]
    common = first[:, :, None] == second[:, None, :]
    shared_counts = np.count_nonzero(np.any(common, axis=2), axis=1)
    return {
        "exactly_two_faces_per_quad": True,
        "all_pairs_share_one_diagonal": bool(np.all(shared_counts == 2)),
        "pairs_not_sharing_one_diagonal": int(np.count_nonzero(shared_counts != 2)),
    }


def _compare_face_rows(
    reconstructed_faces: np.ndarray,
    raw_faces: np.ndarray,
) -> dict[str, Any]:
    reconstructed_faces = np.asarray(reconstructed_faces)
    raw_faces = np.asarray(raw_faces)
    shapes_exact = reconstructed_faces.shape == raw_faces.shape
    rows_exact = bool(shapes_exact and np.array_equal(reconstructed_faces, raw_faces))
    first_mismatch = None
    if shapes_exact and not rows_exact:
        mismatch_rows = np.flatnonzero(
            np.any(reconstructed_faces != raw_faces, axis=1)
        )
        if len(mismatch_rows):
            first_mismatch = int(mismatch_rows[0])
    return {
        "raw_faces_shape": list(raw_faces.shape),
        "reconstructed_faces_shape": list(reconstructed_faces.shape),
        "shapes_exact": shapes_exact,
        "rows_exact": rows_exact,
        "first_mismatching_face_row": first_mismatch,
    }


def _triangulation_exonerated(
    reconstruction: dict[str, Any],
    quad_topology: dict[str, Any],
) -> bool:
    return bool(
        reconstruction["raw_faces_row_exact"]
        and reconstruction["internal_diagonals"].get(
            "all_pairs_share_one_diagonal"
        )
        and quad_topology["orientation"]["contradictory_components"] > 0
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--raw-mesh", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-decoder-sha256", required=True)
    parser.add_argument("--expected-raw-mesh-sha256", required=True)
    parser.add_argument("--lookup-chunk-size", type=int, default=250_000)
    parser.add_argument("--max-witnesses", type=int, default=64)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    protected = (args.decoder_checkpoint, args.raw_mesh)
    if any(
        args.output_json.resolve(strict=False) == path.resolve(strict=False)
        for path in protected
    ):
        raise ValueError("output JSON aliases a protected input")

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "decoder_checkpoint": str(args.decoder_checkpoint),
        "raw_mesh": str(args.raw_mesh),
    }

    def write_report() -> None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )

    phase = "identity_validation"
    try:
        decoder_sha = sha256_file(args.decoder_checkpoint)
        raw_sha = sha256_file(args.raw_mesh)
        report["identity"] = {
            "decoder_sha256": decoder_sha,
            "raw_mesh_sha256": raw_sha,
        }
        if decoder_sha != args.expected_decoder_sha256:
            raise ValueError(
                f"decoder SHA256 mismatch: expected {args.expected_decoder_sha256}, "
                f"got {decoder_sha}"
            )
        if raw_sha != args.expected_raw_mesh_sha256:
            raise ValueError(
                f"raw mesh SHA256 mismatch: expected {args.expected_raw_mesh_sha256}, "
                f"got {raw_sha}"
            )

        phase = "checkpoint_read"
        with np.load(args.decoder_checkpoint, allow_pickle=False) as checkpoint:
            feats = np.array(checkpoint["feats"], copy=True)
            coords = np.array(checkpoint["coords"], copy=True)
        with np.load(args.raw_mesh, allow_pickle=False) as raw_mesh:
            raw_vertices = np.array(raw_mesh["vertices"], copy=False)
            raw_faces = np.array(raw_mesh["faces"], copy=True)
        report["arrays"] = {
            "decoder_rows": int(len(feats)),
            "raw_vertices": int(len(raw_vertices)),
            "raw_faces": int(len(raw_faces)),
        }

        phase = "quad_reconstruction"
        reconstructed = reconstruct_quads(
            feats, coords, lookup_chunk_size=args.lookup_chunk_size
        )
        faces = reconstructed["faces"]
        face_comparison = _compare_face_rows(faces, raw_faces)
        exact_faces = face_comparison["rows_exact"]
        report["reconstruction"] = {
            "active_edges": reconstructed["active_edges"],
            "invalid_active_edges": reconstructed["invalid_active_edges"],
            "valid_quads": int(len(reconstructed["quads"])),
            "split_02": int(np.count_nonzero(reconstructed["split_02"])),
            "split_13": int(
                len(reconstructed["split_02"])
                - np.count_nonzero(reconstructed["split_02"])
            ),
            "raw_faces_row_exact": exact_faces,
            "face_row_comparison": face_comparison,
            "first_mismatching_face_row": face_comparison[
                "first_mismatching_face_row"
            ],
            "internal_diagonals": _verify_internal_diagonals(faces),
        }
        if not exact_faces:
            raise ValueError("reconstructed face rows differ from saved raw mesh")

        phase = "quad_topology"
        report["quad_topology"] = analyze_quad_topology(
            reconstructed["quads"],
            reconstructed["source_axes"],
            reconstructed["source_coords"],
            max_witnesses=args.max_witnesses,
        )
        report["status"] = "complete"
        report["conclusion"] = {
            "triangulation_exonerated": _triangulation_exonerated(
                report["reconstruction"], report["quad_topology"]
            ),
            "decoded_quad_complex_orientable": (
                report["quad_topology"]["orientation"][
                    "contradictory_components"
                ]
                == 0
            ),
            "decoded_quad_complex_manifold": (
                report["quad_topology"]["nonmanifold_edges"] == 0
            ),
        }
    except Exception as exc:
        report["status"] = "failed"
        report["failure_phase"] = phase
        report["error"] = f"{type(exc).__name__}: {exc}"
        write_report()
        raise

    write_report()
    print(json.dumps(report["conclusion"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
