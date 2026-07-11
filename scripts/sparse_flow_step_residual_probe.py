"""Map a sparse-flow step residual onto changed sparse-support categories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_FIELDS = ("pred_pos", "pred_neg", "pred_cfg", "pred_final", "sample_next")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-steps", required=True, type=Path)
    parser.add_argument("--candidate-step", required=True, type=Path)
    parser.add_argument("--source-coords", required=True, type=Path)
    parser.add_argument("--old-coords", required=True, type=Path)
    parser.add_argument("--current-coords", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--step-index", type=int, required=True)
    parser.add_argument("--state-resolution", type=int, default=16)
    parser.add_argument("--support-resolution", type=int, default=32)
    parser.add_argument("--fields", default=",".join(DEFAULT_FIELDS))
    parser.add_argument("--top-k", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    field_names = [name.strip() for name in args.fields.split(",") if name.strip()]
    if not field_names:
        raise SystemExit("--fields did not name any arrays")

    source_fields = _load_source_fields(args.source_steps, args.step_index, field_names)
    candidate_fields = _load_candidate_fields(args.candidate_step, field_names)
    common_fields = [name for name in field_names if name in source_fields and name in candidate_fields]
    if not common_fields:
        raise SystemExit("no comparable fields found in source/candidate inputs")

    support_sets = {
        "source": _coord_set(args.source_coords),
        "old": _coord_set(args.old_coords),
        "current": _coord_set(args.current_coords),
    }
    support_projection = _support_projection(
        support_sets,
        state_resolution=args.state_resolution,
        support_resolution=args.support_resolution,
    )

    fields = {}
    for name in common_fields:
        fields[name] = _field_report(
            source_fields[name],
            candidate_fields[name],
            support_projection=support_projection,
            top_k=args.top_k,
        )

    old = support_sets["old"]
    current = support_sets["current"]
    source = support_sets["source"]
    report = {
        "schema": "trellis2mlx.sparse_flow_step_residual_probe.v1",
        "inputs": {
            "source_steps": _artifact_identity(args.source_steps),
            "candidate_step": _artifact_identity(args.candidate_step),
            "source_coords": _artifact_identity(args.source_coords),
            "old_coords": _artifact_identity(args.old_coords),
            "current_coords": _artifact_identity(args.current_coords),
        },
        "comparison": {
            "step_index": int(args.step_index),
            "state_resolution": int(args.state_resolution),
            "support_resolution": int(args.support_resolution),
            "support_to_state_divisor": float(args.support_resolution / args.state_resolution),
            "fields_requested": field_names,
            "fields_compared": common_fields,
            "top_k": int(args.top_k),
        },
        "support": {
            "source_count": len(source),
            "old_count": len(old),
            "current_count": len(current),
            "old_current_common_count": len(old & current),
            "old_only_count": len(old - current),
            "current_only_count": len(current - old),
            "source_only_vs_old_current_count": len(source - old - current),
            "source_old_common_not_current_count": len((source & old) - current),
            "source_current_common_not_old_count": len((source & current) - old),
            "source_old_current_common_count": len(source & old & current),
        },
        "support_projection": support_projection,
        "fields": fields,
    }
    _write_json(args.output, report)
    print(json.dumps(_console_summary(report), sort_keys=True))
    return 0


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


def _field_report(
    source: np.ndarray,
    candidate: np.ndarray,
    *,
    support_projection: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    if source.shape != candidate.shape:
        raise ValueError(f"source/candidate shapes differ: {source.shape} vs {candidate.shape}")
    if source.ndim != 4:
        raise ValueError(f"selected field must be [C,Z,Y,X], got {source.shape}")

    delta = candidate - source
    all_mask = np.ones(source.shape[1:], dtype=bool)
    report = {
        "shape_czyx": [int(v) for v in source.shape],
        "all": _masked_summary(delta, all_mask),
        "support_categories": {},
        "per_channel": {
            "all": _channel_summary(delta, all_mask),
            "support_categories": {},
        },
        "top_cells": _top_cells(delta, top_k, support_projection),
    }
    for category, projection in support_projection.items():
        mask = _mask_from_state_cells(projection["state_cells"], source.shape[1:])
        report["support_categories"][category] = _masked_summary(delta, mask)
        report["per_channel"]["support_categories"][category] = _channel_summary(delta, mask)
    return report


def _masked_summary(delta: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    if mask.shape != delta.shape[1:]:
        raise ValueError(f"mask shape {mask.shape} does not match delta field {delta.shape}")
    count = int(mask.sum())
    if count == 0:
        return {
            "count": 0,
            "mean_abs": None,
            "max_abs": None,
            "rms": None,
            "mean_delta_l2": None,
            "max_delta_l2": None,
        }
    vals = delta[:, mask]
    per_cell = delta[:, mask].T
    l2 = np.linalg.norm(per_cell, axis=1)
    return {
        "count": count,
        "mean_abs": float(np.mean(np.abs(vals))),
        "max_abs": float(np.max(np.abs(vals))),
        "rms": float(np.sqrt(np.mean(vals * vals))),
        "mean_delta_l2": float(np.mean(l2)),
        "max_delta_l2": float(np.max(l2)),
    }


def _channel_summary(delta: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    if mask.shape != delta.shape[1:]:
        raise ValueError(f"mask shape {mask.shape} does not match delta field {delta.shape}")
    count = int(mask.sum())
    if count == 0:
        return {
            "count": 0,
            "mean_delta": None,
            "mean_abs_delta": None,
            "signed_rank": [],
        }
    vals = delta[:, mask]
    mean_delta = vals.mean(axis=1)
    mean_abs_delta = np.abs(vals).mean(axis=1)
    signed_rank = sorted(
        [
            {
                "channel": int(ch),
                "mean_delta": float(mean_delta[ch]),
                "mean_abs_delta": float(mean_abs_delta[ch]),
                "abs_mean_delta": float(abs(mean_delta[ch])),
            }
            for ch in range(delta.shape[0])
        ],
        key=lambda row: row["abs_mean_delta"],
        reverse=True,
    )
    return {
        "count": count,
        "mean_delta": [float(v) for v in mean_delta],
        "mean_abs_delta": [float(v) for v in mean_abs_delta],
        "signed_rank": signed_rank,
    }


def _top_cells(
    delta: np.ndarray,
    top_k: int,
    support_projection: dict[str, Any],
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    l2 = np.linalg.norm(np.moveaxis(delta, 0, -1), axis=-1)
    flat = l2.reshape(-1)
    if flat.size == 0:
        return []
    count = min(top_k, flat.size)
    indices = np.argpartition(flat, -count)[-count:]
    ordered = sorted(indices.tolist(), key=lambda idx: float(flat[idx]), reverse=True)
    rows = []
    for idx in ordered:
        zyx = np.unravel_index(idx, l2.shape)
        vec = delta[(slice(None),) + zyx]
        abs_vec = np.abs(vec)
        categories = _categories_for_cell(zyx, support_projection)
        rows.append(
            {
                "state_zyx": [int(v) for v in zyx],
                "support_categories": categories,
                "delta_l2": float(flat[idx]),
                "max_abs": float(abs_vec.max()),
                "max_abs_channel": int(abs_vec.argmax()),
                "delta_by_channel": [float(v) for v in vec],
            }
        )
    return rows


def _categories_for_cell(
    zyx: tuple[int, int, int],
    support_projection: dict[str, Any],
) -> list[str]:
    wanted = [int(v) for v in zyx]
    return [
        name
        for name, projection in support_projection.items()
        if wanted in projection["state_cells"]
    ]


def _support_projection(
    support_sets: dict[str, set[tuple[int, int, int]]],
    *,
    state_resolution: int,
    support_resolution: int,
) -> dict[str, Any]:
    old = support_sets["old"]
    current = support_sets["current"]
    source = support_sets["source"]
    categories = {
        "old_only": old - current,
        "current_only": current - old,
        "source_only_vs_old_current": source - old - current,
        "source_old_common_not_current": (source & old) - current,
        "source_current_common_not_old": (source & current) - old,
        "source_old_current_common": source & old & current,
    }
    return {
        name: _project_coords(
            coords,
            state_resolution=state_resolution,
            support_resolution=support_resolution,
        )
        for name, coords in categories.items()
    }


def _project_coords(
    coords: set[tuple[int, int, int]],
    *,
    state_resolution: int,
    support_resolution: int,
) -> dict[str, Any]:
    divisor = support_resolution / state_resolution
    cells: list[tuple[int, int, int]] = []
    inside_count = 0
    for coord in sorted(coords):
        cell = tuple(int(np.floor(axis / divisor)) for axis in coord)
        if all(0 <= axis < state_resolution for axis in cell):
            inside_count += 1
            cells.append(cell)
    unique_cells = sorted(set(cells))
    return {
        "count": len(coords),
        "inside_count": inside_count,
        "state_cell_count": len(unique_cells),
        "state_cells": [[int(v) for v in cell] for cell in unique_cells],
    }


def _mask_from_state_cells(cells: list[list[int]], shape_zyx: tuple[int, int, int]) -> np.ndarray:
    mask = np.zeros(shape_zyx, dtype=bool)
    for z, y, x in cells:
        if 0 <= z < shape_zyx[0] and 0 <= y < shape_zyx[1] and 0 <= x < shape_zyx[2]:
            mask[z, y, x] = True
    return mask


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
    fields = {}
    for name, field in report["fields"].items():
        old_only = field["support_categories"].get("old_only", {})
        current_only = field["support_categories"].get("current_only", {})
        fields[name] = {
            "all_mean_abs": field["all"]["mean_abs"],
            "all_max_abs": field["all"]["max_abs"],
            "old_only_mean_l2": old_only.get("mean_delta_l2"),
            "current_only_mean_l2": current_only.get("mean_delta_l2"),
        }
    return {
        "schema": report["schema"],
        "step_index": report["comparison"]["step_index"],
        "fields": fields,
    }


if __name__ == "__main__":
    raise SystemExit(main())
