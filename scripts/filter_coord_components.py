#!/usr/bin/env python3
"""Filter disconnected sparse coordinate components in an NPZ artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from trellmlx.coord_components import filter_sparse_coordinate_components


SCHEMA = "trellis2mlx.coord_component_filter.v1"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n")


def filter_npz_artifact(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    coords_key: str,
    features_key: str | None,
    aligned_keys: list[str] | None = None,
    mode: str,
    min_component_ratio: float,
    overwrite: bool = False,
    allow_existing_report: bool = False,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"input artifact not found: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output artifact already exists: {output_path}")
    if report_path.exists() and not overwrite and not allow_existing_report:
        raise FileExistsError(f"report already exists: {report_path}")

    with np.load(input_path, allow_pickle=False) as data:
        keys = list(data.keys())
        if coords_key not in data:
            raise KeyError(f"coords key {coords_key!r} not found; available keys: {keys}")
        row_aligned_keys = []
        if features_key is not None:
            row_aligned_keys.append(features_key)
        row_aligned_keys.extend(aligned_keys or [])
        for key in row_aligned_keys:
            if key not in data:
                raise KeyError(f"aligned key {key!r} not found; available keys: {keys}")

        payload = {key: data[key] for key in keys}
        coords = payload[coords_key]
        features = payload[features_key] if features_key is not None else None
        filtered_coords, filtered_features, filter_report = filter_sparse_coordinate_components(
            coords,
            features,
            mode=mode,
            min_component_ratio=min_component_ratio,
            include_row_indices=True,
        )
        kept_row_indices = filter_report.pop("kept_row_indices")
        payload[coords_key] = filtered_coords
        if features_key is not None:
            payload[features_key] = filtered_features
        filtered_keys = [coords_key]
        if features_key is not None:
            filtered_keys.append(features_key)
        for key in aligned_keys or []:
            value = payload[key]
            if len(value) != len(coords):
                raise ValueError(
                    f"aligned key {key!r} row count {len(value)} does not match coord count {len(coords)}"
                )
            payload[key] = value[kept_row_indices]
            filtered_keys.append(key)
        payload["component_filter_report_json"] = np.array(
            json.dumps(filter_report, sort_keys=True),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)

    report = {
        "schema": SCHEMA,
        "status": "ok",
        "route": "sparse-coordinate-component-filter",
        "input": {
            "path": str(input_path),
            "keys": keys,
            "coords_key": coords_key,
            "features_key": features_key,
        },
        "output": {
            "path": str(output_path),
            "exists": output_path.exists(),
            "size_bytes": int(output_path.stat().st_size),
            "component_filter_report_key": "component_filter_report_json",
            "filtered_keys": filtered_keys,
        },
        "filter": filter_report,
    }
    _write_json(report_path, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input NPZ artifact.")
    parser.add_argument("--output", required=True, type=Path, help="Filtered output NPZ artifact.")
    parser.add_argument("--report", required=True, type=Path, help="JSON report path.")
    parser.add_argument("--coords-key", required=True, help="Coordinate array key to filter.")
    parser.add_argument("--features-key", help="Optional row-aligned feature/latent key to filter with coords.")
    parser.add_argument(
        "--aligned-key",
        action="append",
        default=[],
        help="Additional row-aligned array key to filter with coords. May be repeated.",
    )
    parser.add_argument("--mode", choices=("none", "largest", "min_ratio"), default="largest")
    parser.add_argument("--min-component-ratio", type=float, default=1e-5)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output/report paths.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = {
        "schema": SCHEMA,
        "status": "running",
        "route": "sparse-coordinate-component-filter",
        "input": {
            "path": str(args.input),
            "coords_key": args.coords_key,
            "features_key": args.features_key,
        },
        "output": {
            "path": str(args.output),
        },
    }
    try:
        _write_json(args.report, report)
        filter_npz_artifact(
            args.input,
            args.output,
            args.report,
            coords_key=args.coords_key,
            features_key=args.features_key,
            aligned_keys=args.aligned_key,
            mode=args.mode,
            min_component_ratio=args.min_component_ratio,
            overwrite=args.overwrite,
            allow_existing_report=True,
        )
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "phase": "filter_npz_artifact",
                "error": f"{type(exc).__name__}: {exc}",
                "last_trustworthy_evidence": {
                    "input_exists": args.input.exists(),
                    "output_exists": args.output.exists(),
                    "report_exists": args.report.exists(),
                },
            }
        )
        _write_json(args.report, report)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
