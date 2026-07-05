#!/usr/bin/env python3
"""Summarize residual visible-backface evidence by UV island.

This is a diagnostic join over an existing culling face table. It does not
generate meshes, mutate geometry, or prove visual closure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "trellis2mlx.uv_island_residuals.v1"
REQUIRED_COLUMNS = (
    "uv_island",
    "source_orientation",
    "visible_pixels",
    "backface_pixels",
    "projected_missing_pixels",
)
ORIENTATION_ORDER = ("same", "reversed", "unmatched", "ambiguous", "out_of_range")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _require_columns(table: dict[str, np.ndarray]) -> None:
    missing = [name for name in REQUIRED_COLUMNS if name not in table]
    if missing:
        raise ValueError(f"face table missing required columns: {', '.join(missing)}")
    lengths = {name: len(np.asarray(table[name])) for name in REQUIRED_COLUMNS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"face table columns have inconsistent lengths: {lengths}")


def _empty_orientation_totals() -> dict[str, dict[str, int]]:
    return {
        name: {"faces": 0, "visible_pixels": 0, "backface_pixels": 0, "projected_missing_pixels": 0}
        for name in ORIENTATION_ORDER
    }


def _dominant_orientation(pixels: dict[str, int]) -> tuple[str, int, int]:
    ordered = sorted(pixels.items(), key=lambda item: (-item[1], item[0]))
    if not ordered:
        return "none", 0, 0
    first_name, first_pixels = ordered[0]
    second_pixels = ordered[1][1] if len(ordered) > 1 else 0
    return first_name, first_pixels, second_pixels


def _orientation_class(
    pixels: dict[str, int],
    *,
    visible_pixels: int,
    near_tie_margin: float,
) -> str:
    nonzero = {name: count for name, count in pixels.items() if count > 0}
    if not nonzero:
        return "no_visible_pixels"
    if len(nonzero) == 1:
        return f"{next(iter(nonzero))}_only"
    dominant, first, second = _dominant_orientation(nonzero)
    if visible_pixels > 0 and (first - second) / visible_pixels <= near_tie_margin:
        return "mixed_near_tie"
    return f"mixed_{dominant}_dominant"


def _residual_class(
    *,
    backface_pixels: int,
    visible_pixels: int,
    orientation_pixels: dict[str, int],
    backface_by_orientation: dict[str, int],
    high_backface_ratio: float,
) -> str:
    if backface_pixels == 0:
        return "no_backface_residual"
    ratio = backface_pixels / visible_pixels if visible_pixels else 0.0
    prefix = "high" if ratio >= high_backface_ratio else "low"
    same_back = backface_by_orientation.get("same", 0)
    reversed_back = backface_by_orientation.get("reversed", 0)
    if same_back and reversed_back:
        return f"{prefix}_mixed_orientation_residual"
    if reversed_back > same_back:
        return f"{prefix}_reversed_source_residual"
    if same_back > reversed_back:
        return f"{prefix}_same_source_residual"
    dominant, _, _ = _dominant_orientation(orientation_pixels)
    return f"{prefix}_{dominant}_source_residual"


def _sorted_dict(d: dict[str, int]) -> dict[str, int]:
    return {name: int(d[name]) for name in sorted(d)}


def summarize_face_table(
    table: dict[str, np.ndarray],
    *,
    top_n: int = 25,
    high_backface_ratio: float = 0.25,
    near_tie_margin: float = 0.10,
) -> dict[str, Any]:
    """Aggregate per-face culling evidence into UV-island residual buckets."""
    _require_columns(table)
    if top_n < 0:
        raise ValueError("top_n must be non-negative")
    if high_backface_ratio < 0.0:
        raise ValueError("high_backface_ratio must be non-negative")
    if near_tie_margin < 0.0:
        raise ValueError("near_tie_margin must be non-negative")

    uv_islands = np.asarray(table["uv_island"], dtype=np.int64)
    orientations = np.asarray(table["source_orientation"]).astype(str)
    visible = np.asarray(table["visible_pixels"], dtype=np.int64)
    backface = np.asarray(table["backface_pixels"], dtype=np.int64)
    projected_missing = np.asarray(table["projected_missing_pixels"], dtype=np.int64)

    orientation_totals = _empty_orientation_totals()
    islands: dict[int, dict[str, Any]] = {}
    for island, orientation, visible_pixels, backface_pixels, missing_pixels in zip(
        uv_islands,
        orientations,
        visible,
        backface,
        projected_missing,
    ):
        orientation_name = orientation if orientation in orientation_totals else "out_of_range"
        if int(island) < 0:
            island_key = -1
        else:
            island_key = int(island)

        orientation_totals[orientation_name]["faces"] += 1
        orientation_totals[orientation_name]["visible_pixels"] += int(visible_pixels)
        orientation_totals[orientation_name]["backface_pixels"] += int(backface_pixels)
        orientation_totals[orientation_name]["projected_missing_pixels"] += int(missing_pixels)

        entry = islands.setdefault(
            island_key,
            {
                "uv_island": island_key,
                "faces": 0,
                "visible_pixels": 0,
                "backface_pixels": 0,
                "projected_missing_pixels": 0,
                "source_orientation_pixels": {},
                "backface_pixels_by_source_orientation": {},
                "projected_missing_pixels_by_source_orientation": {},
            },
        )
        entry["faces"] += 1
        entry["visible_pixels"] += int(visible_pixels)
        entry["backface_pixels"] += int(backface_pixels)
        entry["projected_missing_pixels"] += int(missing_pixels)
        entry["source_orientation_pixels"][orientation_name] = (
            entry["source_orientation_pixels"].get(orientation_name, 0) + int(visible_pixels)
        )
        entry["backface_pixels_by_source_orientation"][orientation_name] = (
            entry["backface_pixels_by_source_orientation"].get(orientation_name, 0)
            + int(backface_pixels)
        )
        entry["projected_missing_pixels_by_source_orientation"][orientation_name] = (
            entry["projected_missing_pixels_by_source_orientation"].get(orientation_name, 0)
            + int(missing_pixels)
        )

    for entry in islands.values():
        visible_pixels = int(entry["visible_pixels"])
        backface_pixels = int(entry["backface_pixels"])
        entry["backface_pixel_ratio"] = (
            float(backface_pixels / visible_pixels) if visible_pixels else 0.0
        )
        entry["orientation_class"] = _orientation_class(
            entry["source_orientation_pixels"],
            visible_pixels=visible_pixels,
            near_tie_margin=near_tie_margin,
        )
        entry["residual_class"] = _residual_class(
            backface_pixels=backface_pixels,
            visible_pixels=visible_pixels,
            orientation_pixels=entry["source_orientation_pixels"],
            backface_by_orientation=entry["backface_pixels_by_source_orientation"],
            high_backface_ratio=high_backface_ratio,
        )
        entry["source_orientation_pixels"] = _sorted_dict(entry["source_orientation_pixels"])
        entry["backface_pixels_by_source_orientation"] = _sorted_dict(
            entry["backface_pixels_by_source_orientation"]
        )
        entry["projected_missing_pixels_by_source_orientation"] = _sorted_dict(
            entry["projected_missing_pixels_by_source_orientation"]
        )

    top_backface = sorted(
        islands.values(),
        key=lambda item: (-int(item["backface_pixels"]), -float(item["backface_pixel_ratio"]), int(item["uv_island"])),
    )[:top_n]
    top_projected_missing = sorted(
        islands.values(),
        key=lambda item: (-int(item["projected_missing_pixels"]), int(item["uv_island"])),
    )[:top_n]

    residual_class_totals: dict[str, dict[str, int]] = {}
    orientation_class_totals: dict[str, dict[str, int]] = {}
    for entry in islands.values():
        for field, totals in (
            ("residual_class", residual_class_totals),
            ("orientation_class", orientation_class_totals),
        ):
            key = str(entry[field])
            bucket = totals.setdefault(
                key,
                {"islands": 0, "visible_pixels": 0, "backface_pixels": 0, "projected_missing_pixels": 0},
            )
            bucket["islands"] += 1
            bucket["visible_pixels"] += int(entry["visible_pixels"])
            bucket["backface_pixels"] += int(entry["backface_pixels"])
            bucket["projected_missing_pixels"] += int(entry["projected_missing_pixels"])

    total_visible = int(visible.sum())
    total_backface = int(backface.sum())
    return {
        "schema": SCHEMA,
        "route": "cpu_uv_island_residual_face_table_summary",
        "evidence_use_class": "diagnostic_face_table_aggregation",
        "parameters": {
            "top_n": int(top_n),
            "high_backface_ratio": float(high_backface_ratio),
            "near_tie_margin": float(near_tie_margin),
        },
        "totals": {
            "islands": int(len(islands)),
            "visible_pixels": total_visible,
            "backface_pixels": total_backface,
            "projected_missing_pixels": int(projected_missing.sum()),
            "backface_pixel_ratio": float(total_backface / total_visible) if total_visible else 0.0,
        },
        "orientation_totals": orientation_totals,
        "orientation_class_totals": orientation_class_totals,
        "residual_class_totals": residual_class_totals,
        "top_backface_islands": top_backface,
        "top_projected_missing_islands": top_projected_missing,
        "forbidden_to_prove": [
            "not visual closure",
            "not renderer-ground-truth culling",
            "not source-route parity by itself",
            "post-UV source_orientation is bounded evidence, not a standalone flip log",
        ],
    }


def load_face_table(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: np.array(data[name]) for name in data.files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-table", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--high-backface-ratio", type=float, default=0.25)
    parser.add_argument("--near-tie-margin", type=float, default=0.10)
    args = parser.parse_args()

    report = summarize_face_table(
        load_face_table(args.face_table),
        top_n=args.top_n,
        high_backface_ratio=args.high_backface_ratio,
        near_tie_margin=args.near_tie_margin,
    )
    report["face_table"] = str(args.face_table)
    report["report_json"] = str(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n")
    print(f"wrote UV island residual summary: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

