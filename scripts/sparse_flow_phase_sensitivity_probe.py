"""Probe ch0/ch2 sparse-flow phase deltas against changed support cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-steps", required=True, type=Path)
    parser.add_argument("--current-steps", required=True, type=Path)
    parser.add_argument("--old-coords", required=True, type=Path)
    parser.add_argument("--current-coords", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument("--state-resolution", type=int, default=16)
    parser.add_argument("--support-resolution", type=int, default=32)
    parser.add_argument("--x-gate", default="0:4")
    parser.add_argument("--y-gate", default="8:12")
    parser.add_argument("--channels", default="0,2")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    channels = _parse_channels(args.channels)
    if len(channels) != 2:
        raise SystemExit("--channels must name exactly two channels, e.g. 0,2")
    x_gate = _parse_slice(args.x_gate, name="x-gate")
    y_gate = _parse_slice(args.y_gate, name="y-gate")

    old_steps = _load_step_fields(args.old_steps, args.step_index)
    current_steps = _load_step_fields(args.current_steps, args.step_index)
    gate = _xy_gate(args.state_resolution, x_gate=x_gate, y_gate=y_gate)

    old_coords = _load_coords(args.old_coords)
    current_coords = _load_coords(args.current_coords)
    old_set = {tuple(row) for row in old_coords.tolist()}
    current_set = {tuple(row) for row in current_coords.tolist()}
    support_projection = _support_projection(
        old_only=sorted(old_set - current_set),
        current_only=sorted(current_set - old_set),
        common=sorted(old_set & current_set),
        gate=gate,
        state_resolution=args.state_resolution,
        support_resolution=args.support_resolution,
    )

    fields = {}
    for field in ("sample_next", "pred_final"):
        if field not in old_steps or field not in current_steps:
            continue
        fields[field] = _field_report(
            old_steps[field],
            current_steps[field],
            channels=channels,
            gate=gate,
            support_projection=support_projection,
        )
    if not fields:
        raise SystemExit("no comparable fields found; expected sample_next and/or pred_final")

    report = {
        "schema": "trellis2mlx.sparse_flow_phase_sensitivity_probe.v1",
        "inputs": {
            "old_steps": _artifact_identity(args.old_steps),
            "current_steps": _artifact_identity(args.current_steps),
            "old_coords": _artifact_identity(args.old_coords),
            "current_coords": _artifact_identity(args.current_coords),
        },
        "transition": {
            "step_index": int(args.step_index),
            "state_resolution": int(args.state_resolution),
            "support_resolution": int(args.support_resolution),
            "support_to_state_divisor": float(args.support_resolution / args.state_resolution),
            "channels": [int(v) for v in channels],
            "x_gate": {"start": x_gate.start, "stop": x_gate.stop},
            "y_gate": {"start": y_gate.start, "stop": y_gate.stop},
            "gate_cell_count": int(gate.sum()),
            "total_cell_count": int(gate.size),
            "gate_fraction": float(gate.mean()),
        },
        "support": {
            "old_count": len(old_set),
            "current_count": len(current_set),
            "common_count": len(old_set & current_set),
            "old_only_count": len(old_set - current_set),
            "current_only_count": len(current_set - old_set),
            "jaccard": (len(old_set & current_set) / len(old_set | current_set)) if old_set or current_set else 1.0,
        },
        "support_projection": support_projection,
        "fields": fields,
    }
    _write_json(args.output, report)
    print(json.dumps(_console_summary(report), sort_keys=True))
    return 0


def _load_step_fields(path: Path, step_index: int) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {}
    with np.load(path) as data:
        for name in ("sample_next", "pred_final"):
            if name not in data:
                continue
            fields[name] = _select_step(np.asarray(data[name], dtype=np.float32), step_index, name=name)
    return fields


def _select_step(array: np.ndarray, step_index: int, *, name: str) -> np.ndarray:
    if array.ndim == 6:
        sample = array[step_index]
    elif array.ndim == 5:
        if step_index not in (0, -1):
            raise IndexError(f"{name} has no step axis; only 0/-1 is valid")
        sample = array
    else:
        raise ValueError(f"{name} must be [S,B,C,Z,Y,X] or [B,C,Z,Y,X], got {array.shape}")
    if sample.ndim != 5 or sample.shape[0] != 1:
        raise ValueError(f"{name} selected sample must be [1,C,Z,Y,X], got {sample.shape}")
    return sample[0]


def _field_report(
    old: np.ndarray,
    current: np.ndarray,
    *,
    channels: tuple[int, int],
    gate: np.ndarray,
    support_projection: dict[str, Any],
) -> dict[str, Any]:
    if old.shape != current.shape:
        raise ValueError(f"old/current field shapes differ: {old.shape} vs {current.shape}")
    if old.ndim != 4:
        raise ValueError(f"selected field must be [C,Z,Y,X], got {old.shape}")
    for ch in channels:
        if ch < 0 or ch >= old.shape[0]:
            raise ValueError(f"channel {ch} outside field channel count {old.shape[0]}")
    if old.shape[1:] != gate.shape:
        raise ValueError(f"field spatial shape {old.shape[1:]} does not match gate {gate.shape}")

    old_vec = np.stack([old[channels[0]], old[channels[1]]], axis=-1)
    current_vec = np.stack([current[channels[0]], current[channels[1]]], axis=-1)
    delta = current_vec - old_vec
    phase_delta = _phase_delta(old_vec, current_vec)

    report = {
        "all": _vector_summary(old_vec, current_vec, delta, phase_delta, np.ones(gate.shape, dtype=bool)),
        "xy_gate": _vector_summary(old_vec, current_vec, delta, phase_delta, gate),
        "outside_gate": _vector_summary(old_vec, current_vec, delta, phase_delta, ~gate),
        "support_categories": {},
        "per_channel": {
            "all": _channel_summary(current - old, np.ones(gate.shape, dtype=bool)),
            "xy_gate": _channel_summary(current - old, gate),
            "outside_gate": _channel_summary(current - old, ~gate),
            "support_categories": {},
        },
    }
    for name, projection in support_projection.items():
        mask = _mask_from_state_cells(projection["state_cells"], gate.shape)
        report["support_categories"][name] = _vector_summary(old_vec, current_vec, delta, phase_delta, mask)
        report["per_channel"]["support_categories"][name] = _channel_summary(current - old, mask)
    return report


def _phase_delta(old_vec: np.ndarray, current_vec: np.ndarray) -> np.ndarray:
    old_phase = np.arctan2(old_vec[..., 1], old_vec[..., 0])
    current_phase = np.arctan2(current_vec[..., 1], current_vec[..., 0])
    return np.arctan2(np.sin(current_phase - old_phase), np.cos(current_phase - old_phase))


def _vector_summary(
    old_vec: np.ndarray,
    current_vec: np.ndarray,
    delta: np.ndarray,
    phase_delta: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    if mask.shape != delta.shape[:-1]:
        raise ValueError(f"mask shape {mask.shape} does not match delta field {delta.shape}")
    count = int(mask.sum())
    if count == 0:
        return {
            "count": 0,
            "active_phase_count": 0,
            "mean_delta_l2": None,
            "mean_active_delta_l2": None,
            "max_delta_l2": None,
            "mean_abs_phase_delta_rad": None,
            "max_abs_phase_delta_rad": None,
            "mean_delta": None,
        }
    d = delta[mask]
    l2 = np.linalg.norm(d, axis=-1)
    old_mag = np.linalg.norm(old_vec[mask], axis=-1)
    current_mag = np.linalg.norm(current_vec[mask], axis=-1)
    active = (old_mag > 0) | (current_mag > 0)
    p = np.abs(phase_delta[mask][active])
    active_l2 = l2[active]
    return {
        "count": count,
        "active_phase_count": int(active.sum()),
        "mean_delta_l2": float(np.mean(l2)),
        "mean_active_delta_l2": float(np.mean(active_l2)) if active_l2.size else None,
        "max_delta_l2": float(np.max(l2)),
        "mean_abs_phase_delta_rad": float(np.mean(p)) if p.size else None,
        "max_abs_phase_delta_rad": float(np.max(p)) if p.size else None,
        "mean_delta": [float(v) for v in np.mean(d, axis=0)],
    }


def _channel_summary(delta: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    if delta.ndim != 4:
        raise ValueError(f"delta must be [C,Z,Y,X], got {delta.shape}")
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
    rank = sorted(
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
        "signed_rank": rank,
    }


def _support_projection(
    *,
    old_only: list[tuple[int, int, int]],
    current_only: list[tuple[int, int, int]],
    common: list[tuple[int, int, int]],
    gate: np.ndarray,
    state_resolution: int,
    support_resolution: int,
) -> dict[str, Any]:
    divisor = support_resolution / state_resolution
    if divisor <= 0:
        raise ValueError("support/state resolution ratio must be positive")
    return {
        "old_only": _project_coord_set(old_only, gate, divisor),
        "current_only": _project_coord_set(current_only, gate, divisor),
        "common": _project_coord_set(common, gate, divisor),
    }


def _project_coord_set(coords: list[tuple[int, int, int]], gate: np.ndarray, divisor: float) -> dict[str, Any]:
    state_cells: set[tuple[int, int, int]] = set()
    inside = 0
    for coord in coords:
        cell = tuple(int(np.floor(v / divisor)) for v in coord)
        cell = tuple(max(0, min(gate.shape[i] - 1, cell[i])) for i in range(3))
        state_cells.add(cell)
        if bool(gate[cell]):
            inside += 1
    sorted_cells = sorted(state_cells)
    return {
        "count": len(coords),
        "inside_gate_count": inside,
        "outside_gate_count": len(coords) - inside,
        "inside_gate_fraction": float(inside / len(coords)) if coords else None,
        "state_cell_count": len(sorted_cells),
        "state_cells": [[int(v) for v in cell] for cell in sorted_cells],
    }


def _mask_from_state_cells(cells: list[list[int]], shape: tuple[int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for cell in cells:
        z, y, x = (int(v) for v in cell)
        if 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]:
            mask[z, y, x] = True
    return mask


def _xy_gate(resolution: int, *, x_gate: slice, y_gate: slice) -> np.ndarray:
    gate = np.zeros((resolution, resolution, resolution), dtype=bool)
    gate[:, :, x_gate] = True
    gate[:, y_gate, :] = True
    return gate


def _load_coords(path: Path) -> np.ndarray:
    with np.load(path) as data:
        if "coords_3d" in data:
            coords = np.asarray(data["coords_3d"], dtype=np.int32)
        elif "coords" in data:
            coords = np.asarray(data["coords"], dtype=np.int32)
            if coords.ndim == 2 and coords.shape[1] == 4:
                coords = coords[:, 1:]
        else:
            raise KeyError(f"{path} missing coords_3d or coords")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{path} coords must be [N,3], got {coords.shape}")
    return coords


def _parse_channels(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def _parse_slice(text: str, *, name: str) -> slice:
    if ":" not in text:
        value = int(text)
        return slice(value, value + 1)
    start_text, stop_text = text.split(":", 1)
    if not start_text or not stop_text:
        raise ValueError(f"{name} must be explicit start:stop, got {text!r}")
    start = int(start_text)
    stop = int(stop_text)
    if stop <= start:
        raise ValueError(f"{name} stop must be > start, got {text!r}")
    return slice(start, stop)


def _artifact_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    fields = report["fields"]
    summary = {
        "schema": report["schema"],
        "step_index": report["transition"]["step_index"],
        "support_old_only": report["support"]["old_only_count"],
        "support_current_only": report["support"]["current_only_count"],
    }
    for name, field in fields.items():
        summary[f"{name}_xy_mean_delta_l2"] = field["xy_gate"]["mean_delta_l2"]
        summary[f"{name}_xy_mean_abs_phase_delta_rad"] = field["xy_gate"]["mean_abs_phase_delta_rad"]
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
