"""Probe sparse-support basin flips against occupancy logits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-coords", required=True, type=Path)
    parser.add_argument("--current-coords", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--details-output", type=Path)
    parser.add_argument("--old-logits", type=Path)
    parser.add_argument("--current-logits", type=Path)
    parser.add_argument("--old-steps", type=Path)
    parser.add_argument("--current-steps", type=Path)
    parser.add_argument("--decoder-checkpoint", type=Path)
    parser.add_argument("--decode-step", type=int, default=-1)
    parser.add_argument("--decode-array", default="sample_next")
    parser.add_argument(
        "--logit-grid",
        choices=["auto", "raw", "block-max"],
        default="auto",
        help="Map decoder logits to support coords. block-max matches production 64->32 any(logit > 0).",
    )
    parser.add_argument("--lr-resolution", type=int, default=32)
    parser.add_argument("--surface-band", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    old_coords = _load_coords(args.old_coords)
    current_coords = _load_coords(args.current_coords)
    old_set = {tuple(row) for row in old_coords.tolist()}
    current_set = {tuple(row) for row in current_coords.tolist()}

    old_logits, current_logits, logit_source = _load_or_decode_logits(args)
    old_logits, current_logits, logit_grid = _effective_logit_grids(
        old_logits,
        current_logits,
        old_coords,
        current_coords,
        args,
    )
    report = build_report(
        old_coords=old_coords,
        current_coords=current_coords,
        old_set=old_set,
        current_set=current_set,
        old_logits=old_logits,
        current_logits=current_logits,
        logit_source=logit_source,
        logit_grid=logit_grid,
        args=args,
    )
    _write_json(args.output, report)
    if args.details_output:
        _write_details(
            args.details_output,
            sorted(old_set - current_set),
            sorted(current_set - old_set),
            old_logits,
            current_logits,
        )
    print(json.dumps(_console_summary(report), sort_keys=True))
    return 0


def build_report(
    *,
    old_coords: np.ndarray,
    current_coords: np.ndarray,
    old_set: set[tuple[int, int, int]],
    current_set: set[tuple[int, int, int]],
    old_logits: np.ndarray,
    current_logits: np.ndarray,
    logit_source: dict[str, Any],
    logit_grid: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    old_only = sorted(old_set - current_set)
    current_only = sorted(current_set - old_set)
    common = old_set & current_set
    union_count = len(old_set | current_set)
    shape_zyx = _logit_shape_zyx(old_logits)
    margin_sets = {
        "old_only": old_only,
        "current_only": current_only,
        "common": sorted(common),
    }
    margins = {
        name: {
            "old_logits": _value_summary(_gather_logits(old_logits, coords)),
            "current_logits": _value_summary(_gather_logits(current_logits, coords)),
            "delta_current_minus_old": _value_summary(
                _gather_logits(current_logits, coords) - _gather_logits(old_logits, coords)
            ),
        }
        for name, coords in margin_sets.items()
    }
    return {
        "schema": "trellis2mlx.sparse_support_basin_probe.v1",
        "inputs": {
            "old_coords": _artifact_identity(args.old_coords),
            "current_coords": _artifact_identity(args.current_coords),
        },
        "logit_source": logit_source,
        "logit_grid": logit_grid,
        "support": {
            "old_count": len(old_set),
            "current_count": len(current_set),
            "common_count": len(common),
            "old_only_count": len(old_only),
            "current_only_count": len(current_only),
            "union_count": union_count,
            "jaccard": (len(common) / union_count) if union_count else 1.0,
            "old_bounds_zyx": _bounds(old_coords),
            "current_bounds_zyx": _bounds(current_coords),
            "old_only_bounds_zyx": _bounds(np.asarray(old_only, dtype=np.int32)),
            "current_only_bounds_zyx": _bounds(np.asarray(current_only, dtype=np.int32)),
        },
        "logit_shape_zyx": list(shape_zyx),
        "surface_band": int(args.surface_band),
        "surface_bands": {
            "old_only": _surface_bands(old_only, shape_zyx, args.surface_band),
            "current_only": _surface_bands(current_only, shape_zyx, args.surface_band),
            "common": _surface_bands(common, shape_zyx, args.surface_band),
        },
        "axis_histograms": {
            "old_only": _axis_histograms(old_only, shape_zyx),
            "current_only": _axis_histograms(current_only, shape_zyx),
        },
        "parity_histograms": {
            "old_only": _parity_histogram(old_only),
            "current_only": _parity_histogram(current_only),
        },
        "nearest_changed_distance": {
            "old_only_to_current_only": _nearest_manhattan_summary(old_only, current_only),
            "current_only_to_old_only": _nearest_manhattan_summary(current_only, old_only),
        },
        "logit_margins": margins,
        "top_cells": _top_cells(old_only, current_only, old_logits, current_logits, args.top_k),
    }


def _load_or_decode_logits(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if args.old_logits and args.current_logits:
        old_logits = _load_logits(args.old_logits)
        current_logits = _load_logits(args.current_logits)
        return old_logits, current_logits, {
            "kind": "provided_logits",
            "old_logits": _artifact_identity(args.old_logits),
            "current_logits": _artifact_identity(args.current_logits),
        }
    required = [args.old_steps, args.current_steps, args.decoder_checkpoint]
    if not all(required):
        raise SystemExit(
            "provide either --old-logits/--current-logits or "
            "--old-steps/--current-steps/--decoder-checkpoint"
        )
    old_logits = _decode_step_logits(args.old_steps, args.decoder_checkpoint, args.decode_array, args.decode_step)
    current_logits = _decode_step_logits(args.current_steps, args.decoder_checkpoint, args.decode_array, args.decode_step)
    return old_logits, current_logits, {
        "kind": "decoded_sparse_flow_steps",
        "old_steps": _artifact_identity(args.old_steps),
        "current_steps": _artifact_identity(args.current_steps),
        "decoder_checkpoint": _artifact_identity(args.decoder_checkpoint),
        "decode_array": args.decode_array,
        "decode_step": int(args.decode_step),
    }


def _effective_logit_grids(
    old_logits: np.ndarray,
    current_logits: np.ndarray,
    old_coords: np.ndarray,
    current_coords: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    old_grid = _squeeze_logits(old_logits)
    current_grid = _squeeze_logits(current_logits)
    if old_grid.shape != current_grid.shape:
        raise ValueError(f"old/current logits shapes differ: {old_grid.shape} vs {current_grid.shape}")
    coord_max = 0
    if old_coords.size:
        coord_max = max(coord_max, int(old_coords.max()))
    if current_coords.size:
        coord_max = max(coord_max, int(current_coords.max()))
    mode = args.logit_grid
    if mode == "auto":
        mode = "block-max" if old_grid.shape[0] > coord_max + 1 else "raw"
    if mode == "raw":
        effective_old = old_grid
        effective_current = current_grid
    else:
        effective_old = _block_max_grid(old_grid, args.lr_resolution)
        effective_current = _block_max_grid(current_grid, args.lr_resolution)
    return effective_old, effective_current, {
        "mode": mode,
        "requested_mode": args.logit_grid,
        "old_logits_shape_zyx": [int(v) for v in old_grid.shape],
        "current_logits_shape_zyx": [int(v) for v in current_grid.shape],
        "effective_shape_zyx": [int(v) for v in effective_old.shape],
        "lr_resolution": int(args.lr_resolution),
    }


def _block_max_grid(grid: np.ndarray, lr_resolution: int) -> np.ndarray:
    if grid.ndim != 3:
        raise ValueError(f"block-max logits must be rank 3; got {grid.shape}")
    if any(dim % lr_resolution for dim in grid.shape):
        raise ValueError(f"logit shape {grid.shape} is not divisible by lr_resolution={lr_resolution}")
    factors = [dim // lr_resolution for dim in grid.shape]
    reshaped = grid.reshape(
        lr_resolution,
        factors[0],
        lr_resolution,
        factors[1],
        lr_resolution,
        factors[2],
    )
    return reshaped.max(axis=(1, 3, 5))


def _decode_step_logits(
    steps_path: Path,
    checkpoint_path: Path,
    array_name: str,
    step_index: int,
) -> np.ndarray:
    import mlx.core as mx

    from trellmlx.models.sparse_structure_decoder import SparseStructureDecoder
    from trellmlx.weight_loader import load_weights

    with np.load(steps_path) as steps:
        if array_name not in steps:
            raise KeyError(f"{steps_path} missing {array_name!r}")
        sample = np.asarray(steps[array_name][step_index], dtype=np.float32)
    model = SparseStructureDecoder()
    load_weights(model, str(checkpoint_path), verbose=False)
    logits = model(mx.array(sample))
    mx.eval(logits)
    return np.asarray(logits, dtype=np.float32)


def _load_coords(path: Path) -> np.ndarray:
    with np.load(path) as data:
        if "coords_3d" in data:
            coords = np.asarray(data["coords_3d"], dtype=np.int32)
        elif "coords" in data:
            coords = np.asarray(data["coords"], dtype=np.int32)[:, -3:]
        else:
            raise KeyError(f"{path} missing coords or coords_3d")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{path} coords must have shape [N, 3]")
    return coords


def _load_logits(path: Path) -> np.ndarray:
    with np.load(path) as data:
        for name in ("logits", "decoded", "occupancy_logits"):
            if name in data:
                return np.asarray(data[name], dtype=np.float32)
        if len(data.files) == 1:
            return np.asarray(data[data.files[0]], dtype=np.float32)
    raise KeyError(f"{path} missing logits array")


def _logit_shape_zyx(logits: np.ndarray) -> tuple[int, int, int]:
    if logits.ndim == 5:
        return tuple(int(v) for v in logits.shape[-3:])
    if logits.ndim == 4:
        return tuple(int(v) for v in logits.shape[-3:])
    if logits.ndim == 3:
        return tuple(int(v) for v in logits.shape)
    raise ValueError(f"logits must be rank 3, 4, or 5; got {logits.shape}")


def _squeeze_logits(logits: np.ndarray) -> np.ndarray:
    if logits.ndim == 5:
        if logits.shape[0] != 1 or logits.shape[1] != 1:
            raise ValueError(f"rank-5 logits must be [1,1,Z,Y,X]; got {logits.shape}")
        return logits[0, 0]
    if logits.ndim == 4:
        if logits.shape[0] != 1:
            raise ValueError(f"rank-4 logits must be [1,Z,Y,X]; got {logits.shape}")
        return logits[0]
    return logits


def _gather_logits(logits: np.ndarray, coords: list[tuple[int, int, int]] | set[tuple[int, int, int]]) -> np.ndarray:
    grid = _squeeze_logits(logits)
    if not coords:
        return np.asarray([], dtype=np.float64)
    arr = np.asarray(list(coords), dtype=np.int64)
    return grid[arr[:, 0], arr[:, 1], arr[:, 2]].astype(np.float64)


def _value_summary(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
    }


def _bounds(coords: np.ndarray) -> dict[str, list[int] | None]:
    if coords.size == 0:
        return {"min": None, "max": None}
    return {
        "min": [int(v) for v in coords.min(axis=0).tolist()],
        "max": [int(v) for v in coords.max(axis=0).tolist()],
    }


def _surface_bands(
    coords: list[tuple[int, int, int]] | set[tuple[int, int, int]],
    shape_zyx: tuple[int, int, int],
    band: int,
) -> dict[str, int]:
    result = {
        "z_low": 0,
        "z_high": 0,
        "y_low": 0,
        "y_high": 0,
        "x_low": 0,
        "x_high": 0,
    }
    for z, y, x in coords:
        if z < band:
            result["z_low"] += 1
        if z >= shape_zyx[0] - band:
            result["z_high"] += 1
        if y < band:
            result["y_low"] += 1
        if y >= shape_zyx[1] - band:
            result["y_high"] += 1
        if x < band:
            result["x_low"] += 1
        if x >= shape_zyx[2] - band:
            result["x_high"] += 1
    return result


def _axis_histograms(
    coords: list[tuple[int, int, int]] | set[tuple[int, int, int]],
    shape_zyx: tuple[int, int, int],
) -> dict[str, list[int]]:
    arr = np.asarray(list(coords), dtype=np.int32)
    if arr.size == 0:
        return {axis: [0] * size for axis, size in zip(("z", "y", "x"), shape_zyx)}
    return {
        "z": np.bincount(arr[:, 0], minlength=shape_zyx[0]).astype(int).tolist(),
        "y": np.bincount(arr[:, 1], minlength=shape_zyx[1]).astype(int).tolist(),
        "x": np.bincount(arr[:, 2], minlength=shape_zyx[2]).astype(int).tolist(),
    }


def _parity_histogram(coords: list[tuple[int, int, int]] | set[tuple[int, int, int]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for z, y, x in coords:
        key = f"{z % 2}{y % 2}{x % 2}"
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def _nearest_manhattan_summary(
    source: list[tuple[int, int, int]],
    target: list[tuple[int, int, int]],
) -> dict[str, Any]:
    if not source or not target:
        return {"count": len(source), "min": None, "max": None, "mean": None, "histogram": {}}
    target_arr = np.asarray(target, dtype=np.int32)
    distances = []
    for coord in source:
        diff = np.abs(target_arr - np.asarray(coord, dtype=np.int32))
        distances.append(int(np.min(diff.sum(axis=1))))
    hist: dict[str, int] = {}
    for distance in distances:
        hist[str(distance)] = hist.get(str(distance), 0) + 1
    arr = np.asarray(distances, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "mean": float(np.mean(arr)),
        "histogram": dict(sorted(hist.items(), key=lambda item: int(item[0]))),
    }


def _top_cells(
    old_only: list[tuple[int, int, int]],
    current_only: list[tuple[int, int, int]],
    old_logits: np.ndarray,
    current_logits: np.ndarray,
    top_k: int,
) -> dict[str, list[dict[str, Any]]]:
    old_rows = _cell_rows(old_only, old_logits, current_logits)
    current_rows = _cell_rows(current_only, old_logits, current_logits)
    return {
        "old_only_by_logit_drop": sorted(
            old_rows,
            key=lambda row: row["old_minus_current_logit"],
            reverse=True,
        )[:top_k],
        "current_only_by_logit_gain": sorted(
            current_rows,
            key=lambda row: row["current_minus_old_logit"],
            reverse=True,
        )[:top_k],
    }


def _cell_rows(
    coords: list[tuple[int, int, int]],
    old_logits: np.ndarray,
    current_logits: np.ndarray,
) -> list[dict[str, Any]]:
    old_values = _gather_logits(old_logits, coords)
    current_values = _gather_logits(current_logits, coords)
    rows = []
    for coord, old_value, current_value in zip(coords, old_values, current_values):
        rows.append(
            {
                "coord_zyx": [int(v) for v in coord],
                "old_logit": float(old_value),
                "current_logit": float(current_value),
                "current_minus_old_logit": float(current_value - old_value),
                "old_minus_current_logit": float(old_value - current_value),
            }
        )
    return rows


def _write_details(
    output: Path,
    old_only: list[tuple[int, int, int]],
    current_only: list[tuple[int, int, int]],
    old_logits: np.ndarray,
    current_logits: np.ndarray,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        old_only_coords=np.asarray(old_only, dtype=np.int32),
        current_only_coords=np.asarray(current_only, dtype=np.int32),
        old_logits=_squeeze_logits(old_logits),
        current_logits=_squeeze_logits(current_logits),
    )


def _artifact_identity(path: Path) -> dict[str, Any]:
    path = Path(path)
    identity = {
        "path": str(path),
        "exists": path.exists(),
    }
    if path.exists():
        stat = path.stat()
        identity.update(
            {
                "size": stat.st_size,
                "sha256": _sha256(path),
            }
        )
    return identity


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
    support = report["support"]
    return {
        "schema": report["schema"],
        "old_only": support["old_only_count"],
        "current_only": support["current_only_count"],
        "jaccard": support["jaccard"],
        "logit_source": report["logit_source"]["kind"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
