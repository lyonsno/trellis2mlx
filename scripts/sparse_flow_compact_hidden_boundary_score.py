"""Score compact sparse-flow hidden states against projected boundary cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


STAGE_SUFFIXES = ("input", "after_self", "after_cross", "after_mlp")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", required=True, type=Path)
    parser.add_argument("--negative-coords", required=True, type=Path)
    parser.add_argument("--positive-coords", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--block-index", type=int)
    parser.add_argument("--state-resolution", type=int, default=16)
    parser.add_argument("--support-resolution", type=int, default=32)
    parser.add_argument("--source-steps", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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

    inputs: dict[str, Any] = {
        "negative_coords": _artifact_identity(args.negative_coords),
        "positive_coords": _artifact_identity(args.positive_coords),
    }
    if args.source_steps is not None:
        inputs["source_steps"] = _artifact_identity(args.source_steps)

    report = {
        "schema": "trellis2mlx.sparse_flow_compact_hidden_boundary_score.v1",
        "inputs": inputs,
        "labels": {
            "projected_negative_count": len(negative_cells_all),
            "projected_positive_count": len(positive_cells_all),
            "overlap_count": len(overlap_cells),
            "clean_negative_count": len(negative_cells),
            "clean_positive_count": len(positive_cells),
        },
        "blocks": {},
    }
    for trace in args.trace:
        block_index, arrays = _load_trace_arrays(trace, requested_block_index=args.block_index)
        input_array = arrays.get("input")
        if input_array is None:
            raise SystemExit(f"{trace} missing required input stage for block {block_index}")
        raw_scores = {}
        delta_scores = {}
        for stage in STAGE_SUFFIXES:
            if stage not in arrays:
                continue
            raw_scores[stage] = _field_score(
                arrays[stage],
                negative_cells,
                positive_cells,
                state_resolution=args.state_resolution,
            )
            if stage != "input":
                delta_scores[stage] = _field_score(
                    arrays[stage] - input_array,
                    negative_cells,
                    positive_cells,
                    state_resolution=args.state_resolution,
                )
        report["blocks"][str(block_index)] = {
            "trace_path": str(trace),
            "trace_sha256": _sha256(trace),
            "raw_stage_scores": raw_scores,
            "delta_from_input_scores": delta_scores,
        }
    _write_json(args.output, report)
    print(json.dumps(_console_summary(report), sort_keys=True))
    return 0


def _load_trace_arrays(
    path: Path,
    *,
    requested_block_index: int | None,
) -> tuple[int, dict[str, np.ndarray]]:
    arrays: dict[str, np.ndarray] = {}
    with np.load(path) as data:
        block_index = _resolve_block_index(data, requested_block_index)
        prefix = f"pos_block{block_index}_"
        for stage in STAGE_SUFFIXES:
            key = prefix + stage
            if key in data:
                arrays[stage] = _select_hidden(np.asarray(data[key], dtype=np.float32), name=key)
    return block_index, arrays


def _resolve_block_index(data: np.lib.npyio.NpzFile, requested: int | None) -> int:
    if requested is not None:
        return requested
    if "trace_block_index" in data:
        return int(np.asarray(data["trace_block_index"]).item())
    for key in data.files:
        if key.startswith("pos_block"):
            tail = key[len("pos_block") :]
            digits = []
            for char in tail:
                if not char.isdigit():
                    break
                digits.append(char)
            if digits:
                return int("".join(digits))
    raise ValueError("could not infer block index; pass --block-index")


def _select_hidden(array: np.ndarray, *, name: str) -> np.ndarray:
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0]
    if array.ndim == 2:
        return array
    raise ValueError(f"{name} must be [1,T,C] or [T,C], got {array.shape}")


def _field_score(
    hidden: np.ndarray,
    negative_cells: list[tuple[int, int, int]],
    positive_cells: list[tuple[int, int, int]],
    *,
    state_resolution: int,
) -> dict[str, Any]:
    if hidden.ndim != 2:
        raise ValueError(f"selected hidden state must be [T,C], got {hidden.shape}")
    negative = _gather_cells(hidden, negative_cells, state_resolution=state_resolution)
    positive = _gather_cells(hidden, positive_cells, state_resolution=state_resolution)
    if negative.size == 0 or positive.size == 0:
        return {
            "negative_count": int(negative.shape[0]),
            "positive_count": int(positive.shape[0]),
            "negative_mean": None,
            "positive_mean": None,
            "centroid_delta": None,
            "centroid_delta_l2": None,
            "centroid_delta_mean_abs": None,
            "auc_positive_gt_negative": None,
            "top_channels": [],
        }

    negative_mean = negative.mean(axis=0)
    positive_mean = positive.mean(axis=0)
    centroid_delta = positive_mean - negative_mean
    negative_scores = negative @ centroid_delta
    positive_scores = positive @ centroid_delta
    top_channels = sorted(
        [
            {
                "channel": int(channel),
                "centroid_delta": float(centroid_delta[channel]),
                "abs_centroid_delta": float(abs(centroid_delta[channel])),
                "negative_mean": float(negative_mean[channel]),
                "positive_mean": float(positive_mean[channel]),
            }
            for channel in range(hidden.shape[1])
        ],
        key=lambda row: row["abs_centroid_delta"],
        reverse=True,
    )
    return {
        "negative_count": int(negative.shape[0]),
        "positive_count": int(positive.shape[0]),
        "negative_mean": [float(v) for v in negative_mean],
        "positive_mean": [float(v) for v in positive_mean],
        "centroid_delta": [float(v) for v in centroid_delta],
        "centroid_delta_l2": float(np.linalg.norm(centroid_delta)),
        "centroid_delta_mean_abs": float(np.mean(np.abs(centroid_delta))),
        "negative_score_summary": _value_summary(negative_scores),
        "positive_score_summary": _value_summary(positive_scores),
        "auc_positive_gt_negative": _auc(positive_scores, negative_scores),
        "top_channels": top_channels,
    }


def _gather_cells(
    hidden: np.ndarray,
    cells: list[tuple[int, int, int]],
    *,
    state_resolution: int,
) -> np.ndarray:
    rows = []
    for cell in cells:
        token_index = _flatten_zyx(cell, state_resolution=state_resolution)
        if 0 <= token_index < hidden.shape[0]:
            rows.append(hidden[token_index])
    if not rows:
        return np.empty((0, hidden.shape[1]), dtype=np.float32)
    return np.stack(rows, axis=0).astype(np.float32)


def _flatten_zyx(cell: tuple[int, int, int], *, state_resolution: int) -> int:
    z, y, x = cell
    return z * state_resolution * state_resolution + y * state_resolution + x


def _auc(positive: np.ndarray, negative: np.ndarray) -> float:
    wins = 0.0
    total = 0
    for positive_value in positive:
        for negative_value in negative:
            total += 1
            if positive_value > negative_value:
                wins += 1.0
            elif positive_value == negative_value:
                wins += 0.5
    return float(wins / total) if total else 0.5


def _value_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "std": float(values.std()),
    }


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
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size": path.stat().st_size,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = {"blocks": {}, "labels": report["labels"], "schema": report["schema"]}
    for block, payload in report["blocks"].items():
        summary["blocks"][block] = {
            "raw_stage_auc": {
                stage: score["auc_positive_gt_negative"]
                for stage, score in payload["raw_stage_scores"].items()
            },
            "delta_stage_auc": {
                stage: score["auc_positive_gt_negative"]
                for stage, score in payload["delta_from_input_scores"].items()
            },
        }
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
