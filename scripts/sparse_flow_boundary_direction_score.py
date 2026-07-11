"""Score whether a sparse-flow residual direction separates boundary cell sets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-steps", required=True, type=Path)
    parser.add_argument("--candidate-step", required=True, type=Path)
    parser.add_argument("--negative-coords", required=True, type=Path)
    parser.add_argument("--positive-coords", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--step-index", type=int, required=True)
    parser.add_argument("--state-resolution", type=int, default=16)
    parser.add_argument("--support-resolution", type=int, default=32)
    parser.add_argument("--fields", default="pred_final,sample_next,pred_pos,pred_neg,pred_cfg")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fields = [name.strip() for name in args.fields.split(",") if name.strip()]
    if not fields:
        raise SystemExit("--fields did not name any arrays")

    negative_cells_all = _project_coords(
        _coord_set(args.negative_coords),
        state_resolution=args.state_resolution,
        support_resolution=args.support_resolution,
    )
    positive_cells_all = _project_coords(
        _coord_set(args.positive_coords),
        state_resolution=args.state_resolution,
        support_resolution=args.support_resolution,
    )
    negative_cells = sorted(negative_cells_all - positive_cells_all)
    positive_cells = sorted(positive_cells_all - negative_cells_all)
    overlap_cells = sorted(negative_cells_all & positive_cells_all)

    source_fields = _load_source_fields(args.source_steps, args.step_index, fields)
    candidate_fields = _load_candidate_fields(args.candidate_step, fields)
    common_fields = [name for name in fields if name in source_fields and name in candidate_fields]
    if not common_fields:
        raise SystemExit("no comparable fields found in source/candidate inputs")

    report = {
        "schema": "trellis2mlx.sparse_flow_boundary_direction_score.v1",
        "inputs": {
            "source_steps": _artifact_identity(args.source_steps),
            "candidate_step": _artifact_identity(args.candidate_step),
            "negative_coords": _artifact_identity(args.negative_coords),
            "positive_coords": _artifact_identity(args.positive_coords),
        },
        "comparison": {
            "step_index": int(args.step_index),
            "state_resolution": int(args.state_resolution),
            "support_resolution": int(args.support_resolution),
            "support_to_state_divisor": float(args.support_resolution / args.state_resolution),
            "fields_requested": fields,
            "fields_compared": common_fields,
        },
        "labels": {
            "projected_negative_count": len(negative_cells_all),
            "projected_positive_count": len(positive_cells_all),
            "overlap_count": len(overlap_cells),
            "clean_negative_count": len(negative_cells),
            "clean_positive_count": len(positive_cells),
        },
        "fields": {},
    }
    for name in common_fields:
        delta = candidate_fields[name] - source_fields[name]
        report["fields"][name] = _field_score(delta, negative_cells, positive_cells)
    _write_json(args.output, report)
    print(json.dumps(_console_summary(report), sort_keys=True))
    return 0


def _field_score(
    delta: np.ndarray,
    negative_cells: list[tuple[int, int, int]],
    positive_cells: list[tuple[int, int, int]],
) -> dict[str, Any]:
    if delta.ndim != 4:
        raise ValueError(f"selected field must be [C,Z,Y,X], got {delta.shape}")
    neg = _gather_cells(delta, negative_cells)
    pos = _gather_cells(delta, positive_cells)
    if neg.size == 0 or pos.size == 0:
        return {
            "negative_count": int(neg.shape[0]),
            "positive_count": int(pos.shape[0]),
            "negative_mean": None,
            "positive_mean": None,
            "centroid_delta": None,
            "auc_positive_gt_negative": None,
            "top_channels": [],
        }

    neg_mean = neg.mean(axis=0)
    pos_mean = pos.mean(axis=0)
    centroid_delta = pos_mean - neg_mean
    neg_scores = neg @ centroid_delta
    pos_scores = pos @ centroid_delta
    top_channels = sorted(
        [
            {
                "channel": int(ch),
                "centroid_delta": float(centroid_delta[ch]),
                "abs_centroid_delta": float(abs(centroid_delta[ch])),
                "negative_mean": float(neg_mean[ch]),
                "positive_mean": float(pos_mean[ch]),
            }
            for ch in range(delta.shape[0])
        ],
        key=lambda row: row["abs_centroid_delta"],
        reverse=True,
    )
    return {
        "negative_count": int(neg.shape[0]),
        "positive_count": int(pos.shape[0]),
        "negative_mean": [float(v) for v in neg_mean],
        "positive_mean": [float(v) for v in pos_mean],
        "centroid_delta": [float(v) for v in centroid_delta],
        "negative_score_summary": _value_summary(neg_scores),
        "positive_score_summary": _value_summary(pos_scores),
        "auc_positive_gt_negative": _auc(pos_scores, neg_scores),
        "top_channels": top_channels,
    }


def _gather_cells(delta: np.ndarray, cells: list[tuple[int, int, int]]) -> np.ndarray:
    rows = []
    zyx_shape = delta.shape[1:]
    for z, y, x in cells:
        if 0 <= z < zyx_shape[0] and 0 <= y < zyx_shape[1] and 0 <= x < zyx_shape[2]:
            rows.append(delta[:, z, y, x])
    if not rows:
        return np.empty((0, delta.shape[0]), dtype=np.float32)
    return np.stack(rows, axis=0).astype(np.float32)


def _auc(positive: np.ndarray, negative: np.ndarray) -> float:
    wins = 0.0
    total = 0
    for pos in positive:
        for neg in negative:
            total += 1
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return float(wins / total) if total else 0.5


def _value_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "std": float(values.std()),
    }


def _load_source_fields(path: Path, step_index: int, field_names: list[str]) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {}
    with np.load(path) as data:
        for name in field_names:
            if name not in data:
                continue
            fields[name] = _select_source_step(np.asarray(data[name], dtype=np.float32), step_index, name=name)
    return fields


def _load_candidate_fields(path: Path, field_names: list[str]) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {}
    with np.load(path) as data:
        for name in field_names:
            if name not in data:
                continue
            fields[name] = _select_candidate_step(np.asarray(data[name], dtype=np.float32), name=name)
    return fields


def _select_source_step(array: np.ndarray, step_index: int, *, name: str) -> np.ndarray:
    if array.ndim == 6:
        sample = array[step_index]
    elif array.ndim == 5:
        if step_index not in (0, -1):
            raise IndexError(f"{name} has no source step axis; only 0/-1 is valid")
        sample = array
    else:
        raise ValueError(f"{name} must be [S,B,C,Z,Y,X] or [B,C,Z,Y,X], got {array.shape}")
    return _squeeze_bczxy(sample, name=name)


def _select_candidate_step(array: np.ndarray, *, name: str) -> np.ndarray:
    if array.ndim == 6:
        if array.shape[0] != 1:
            raise ValueError(f"{name} candidate has multiple steps; pass a one-step capture, got {array.shape}")
        sample = array[0]
    elif array.ndim == 5:
        sample = array
    else:
        raise ValueError(f"{name} must be [B,C,Z,Y,X] or one-step [1,B,C,Z,Y,X], got {array.shape}")
    return _squeeze_bczxy(sample, name=name)


def _squeeze_bczxy(sample: np.ndarray, *, name: str) -> np.ndarray:
    if sample.ndim != 5 or sample.shape[0] != 1:
        raise ValueError(f"{name} selected sample must be [1,C,Z,Y,X], got {sample.shape}")
    return sample[0]


def _project_coords(
    coords: set[tuple[int, int, int]],
    *,
    state_resolution: int,
    support_resolution: int,
) -> set[tuple[int, int, int]]:
    divisor = support_resolution / state_resolution
    cells = set()
    for coord in coords:
        cell = tuple(int(np.floor(axis / divisor)) for axis in coord)
        if all(0 <= axis < state_resolution for axis in cell):
            cells.add(cell)
    return cells


def _coord_set(path: Path) -> set[tuple[int, int, int]]:
    with np.load(path) as data:
        if "coords_3d" in data:
            coords = np.asarray(data["coords_3d"], dtype=np.int32)
        elif "coords" in data:
            coords = np.asarray(data["coords"], dtype=np.int32)[:, -3:]
        else:
            raise KeyError(f"{path} missing coords or coords_3d")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{path} coords must have shape [N,3], got {coords.shape}")
    return {tuple(int(v) for v in row) for row in coords.tolist()}


def _artifact_identity(path: Path) -> dict[str, Any]:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return {
        "path": str(path),
        "sha256": h.hexdigest(),
        "size": path.stat().st_size,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": report["schema"],
        "labels": report["labels"],
        "fields": {
            name: {
                "auc": field["auc_positive_gt_negative"],
                "top_channel": field["top_channels"][0]["channel"] if field["top_channels"] else None,
            }
            for name, field in report["fields"].items()
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
