#!/usr/bin/env python3
"""Decode a saved HR shape SLat artifact to a shape-only GLB."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generate import SHAPE_SLAT_MEAN, SHAPE_SLAT_STD
from trellmlx.coord_components import filter_sparse_coordinate_components


SCHEMA = "trellis2mlx.shape_artifact_decode.v1"


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


def _array_shape(value: np.ndarray) -> list[int]:
    return [int(dim) for dim in value.shape]


def _bounds(value: np.ndarray) -> dict[str, list[float]] | None:
    if len(value) == 0:
        return None
    return {
        "min": np.min(value, axis=0).astype(float).tolist(),
        "max": np.max(value, axis=0).astype(float).tolist(),
    }


def _require_key(data: Any, key: str, *, role: str) -> np.ndarray:
    keys = list(data.keys())
    if key not in data:
        raise KeyError(f"{role} key {key!r} not found; available keys: {keys}")
    return data[key]


def _prepare_decoder_slat(slat: np.ndarray, normalization: str) -> np.ndarray:
    if normalization == "normalized":
        return (slat * SHAPE_SLAT_STD[None, :]) + SHAPE_SLAT_MEAN[None, :]
    if normalization == "denormalized":
        return slat
    raise ValueError(f"unknown SLat normalization mode: {normalization!r}")


def _export_glb(vertices: np.ndarray, faces: np.ndarray, output_path: Path) -> dict[str, Any]:
    import trimesh

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_vertices = np.asarray(vertices, dtype=np.float32).copy()
    if len(export_vertices):
        export_vertices[:, 1], export_vertices[:, 2] = (
            export_vertices[:, 2].copy(),
            -export_vertices[:, 1].copy(),
        )
    mesh = trimesh.Trimesh(
        vertices=export_vertices,
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    mesh.export(output_path)
    return {
        "path": str(output_path),
        "glb_exists": output_path.exists(),
        "glb_size_bytes": int(output_path.stat().st_size) if output_path.exists() else 0,
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "bounds": _bounds(export_vertices),
        "axis_transform": "swap_yz_invert_y",
    }


def _default_decoder_factory():
    from trellmlx.models.shape_slat_decoder import SLatDecoder

    return SLatDecoder(out_channels=7, pred_subdiv=True)


def _default_weight_loader(decoder: Any, checkpoint: Path, *, verbose: bool = False) -> None:
    from trellmlx.weight_loader import load_weights

    load_weights(decoder, str(checkpoint), verbose=verbose)


def _default_mesh_converter(dec_out: np.ndarray, dec_coords: np.ndarray, resolution: int):
    from trellmlx.mesh_extract import decoder_output_to_mesh

    return decoder_output_to_mesh(dec_out, dec_coords, resolution=resolution)


def decode_shape_artifact(
    input_path: Path,
    output_glb_path: Path,
    report_path: Path,
    *,
    output_artifact_path: Path | None = None,
    slat_key: str = "hr_slat",
    coords_key: str = "hr_coords_quantized_1024",
    spatial_coords_key: str = "hr_coords_3d_1024",
    component_filter_mode: str = "largest",
    component_filter_min_ratio: float = 1e-5,
    slat_normalization: str = "normalized",
    decoder_checkpoint: Path,
    resolution: int = 1024,
    overwrite: bool = False,
    allow_existing_report: bool = False,
    decoder_factory: Callable[[], Any] | None = None,
    weight_loader: Callable[..., Any] | None = None,
    mesh_converter: Callable[[np.ndarray, np.ndarray, int], tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Decode saved HR shape latents to a shape-only GLB and JSON report."""
    input_path = Path(input_path)
    output_glb_path = Path(output_glb_path)
    report_path = Path(report_path)
    output_artifact_path = None if output_artifact_path is None else Path(output_artifact_path)
    decoder_checkpoint = Path(decoder_checkpoint)

    if not input_path.exists():
        raise FileNotFoundError(f"input artifact not found: {input_path}")
    if output_glb_path.exists() and not overwrite:
        raise FileExistsError(f"output GLB already exists: {output_glb_path}")
    if output_artifact_path is not None and output_artifact_path.exists() and not overwrite:
        raise FileExistsError(f"output artifact already exists: {output_artifact_path}")
    if report_path.exists() and not overwrite and not allow_existing_report:
        raise FileExistsError(f"report already exists: {report_path}")
    decoder_factory = decoder_factory or _default_decoder_factory
    weight_loader = weight_loader or _default_weight_loader
    mesh_converter = mesh_converter or _default_mesh_converter

    timings: dict[str, float] = {}
    started = time.perf_counter()
    with np.load(input_path, allow_pickle=False) as data:
        keys = list(data.keys())
        slat = np.asarray(_require_key(data, slat_key, role="SLat"), dtype=np.float32)
        coords = np.asarray(_require_key(data, coords_key, role="decoder coordinate"), dtype=np.int32)
        spatial_coords = np.asarray(_require_key(data, spatial_coords_key, role="spatial coordinate"), dtype=np.int32)

    if len(coords) != len(slat):
        raise ValueError(
            f"decoder coordinate row count {len(coords)} does not match SLat row count {len(slat)}"
        )
    if len(spatial_coords) != len(slat):
        raise ValueError(
            f"spatial coordinate row count {len(spatial_coords)} does not match SLat row count {len(slat)}"
        )
    if slat.shape[1] != len(SHAPE_SLAT_MEAN):
        raise ValueError(
            f"SLat channel count {slat.shape[1]} does not match shape latent stats width {len(SHAPE_SLAT_MEAN)}"
        )
    if not decoder_checkpoint.exists() and decoder_factory is _default_decoder_factory:
        raise FileNotFoundError(f"decoder checkpoint not found: {decoder_checkpoint}")

    t0 = time.perf_counter()
    filtered_spatial, filtered_slat, component_report = filter_sparse_coordinate_components(
        spatial_coords,
        slat,
        mode=component_filter_mode,
        min_component_ratio=component_filter_min_ratio,
        include_row_indices=True,
    )
    kept_row_indices = np.asarray(component_report.pop("kept_row_indices"), dtype=np.int64)
    filtered_coords = coords[kept_row_indices]
    timings["component_filter_seconds"] = time.perf_counter() - t0
    decoder_slat = _prepare_decoder_slat(filtered_slat, slat_normalization).astype(np.float32, copy=False)

    t0 = time.perf_counter()
    decoder = decoder_factory()
    weight_loader(decoder, decoder_checkpoint, verbose=False)
    timings["decoder_load_seconds"] = time.perf_counter() - t0

    import mlx.core as mx

    t0 = time.perf_counter()
    dec_out, dec_coords, shape_subs = decoder(
        mx.array(decoder_slat),
        mx.array(filtered_coords),
        return_subs=True,
    )
    mx.eval(dec_out, dec_coords)
    dec_out_np = np.asarray(dec_out, dtype=np.float32)
    dec_coords_np = np.asarray(dec_coords, dtype=np.int32)
    timings["decode_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    vertices, faces = mesh_converter(dec_out_np, dec_coords_np, resolution)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    timings["mesh_extract_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    output = _export_glb(vertices, faces, output_glb_path)
    timings["glb_export_seconds"] = time.perf_counter() - t0

    artifact_output = None
    if output_artifact_path is not None:
        output_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_payload = {
            "filtered_" + slat_key: filtered_slat,
            "filtered_" + coords_key: filtered_coords,
            "filtered_" + spatial_coords_key: filtered_spatial,
            "decoder_" + slat_key: decoder_slat,
            "dec_out": dec_out_np,
            "dec_coords": dec_coords_np,
            "vertices": vertices,
            "faces": faces,
            "component_filter_report_json": np.array(json.dumps(component_report, sort_keys=True)),
        }
        for index, sub in enumerate(shape_subs):
            artifact_payload[f"shape_sub_{index}"] = np.asarray(sub)
        np.savez_compressed(output_artifact_path, **artifact_payload)
        artifact_output = {
            "path": str(output_artifact_path),
            "artifact_exists": output_artifact_path.exists(),
            "artifact_size_bytes": (
                int(output_artifact_path.stat().st_size) if output_artifact_path.exists() else 0
            ),
            "component_filter_report_key": "component_filter_report_json",
        }

    report = {
        "schema": SCHEMA,
        "status": "ok",
        "route": "shape-artifact-decode",
        "input": {
            "path": str(input_path),
            "keys": keys,
            "slat_key": slat_key,
            "coords_key": coords_key,
            "spatial_coords_key": spatial_coords_key,
            "slat_shape": _array_shape(slat),
            "coords_shape": _array_shape(coords),
            "spatial_coords_shape": _array_shape(spatial_coords),
        },
        "decoder": {
            "checkpoint": str(decoder_checkpoint),
            "class": type(decoder).__name__,
            "out_channels": 7,
            "pred_subdiv": True,
            "resolution": int(resolution),
        },
        "component_filter": component_report,
        "decode": {
            "input_slat_shape": _array_shape(filtered_slat),
            "input_coords_shape": _array_shape(filtered_coords),
            "slat_normalization": slat_normalization,
            "decoder_slat_shape": _array_shape(decoder_slat),
            "output_shape": _array_shape(dec_out_np),
            "output_coords_shape": _array_shape(dec_coords_np),
            "subdivision_level_count": int(len(shape_subs)),
        },
        "mesh": {
            "vertex_count": int(len(vertices)),
            "face_count": int(len(faces)),
            "bounds": _bounds(vertices),
        },
        "output": output | {
            "artifact_exists": bool(
                artifact_output is not None and artifact_output["artifact_exists"]
            ),
            "artifact": artifact_output,
        },
        "timings": timings | {"total_seconds": time.perf_counter() - started},
        "last_trustworthy_evidence": {
            "phase": "glb_export",
            "input_rows": int(len(slat)),
            "filtered_rows": int(len(filtered_slat)),
            "glb_exists": output_glb_path.exists(),
        },
    }
    _write_json(report_path, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input NPZ artifact with HR SLat arrays.")
    parser.add_argument("--output-glb", required=True, type=Path, help="Output shape-only GLB path.")
    parser.add_argument("--output-artifact", type=Path, help="Optional decoded NPZ artifact path.")
    parser.add_argument("--report", required=True, type=Path, help="JSON report path.")
    parser.add_argument("--decoder-checkpoint", required=True, type=Path, help="Shape decoder safetensors path.")
    parser.add_argument("--slat-key", default="hr_slat")
    parser.add_argument("--coords-key", default="hr_coords_quantized_1024")
    parser.add_argument("--spatial-coords-key", default="hr_coords_3d_1024")
    parser.add_argument("--component-filter", choices=("none", "largest", "min_ratio"), default="largest")
    parser.add_argument("--component-filter-min-ratio", type=float, default=1e-5)
    parser.add_argument(
        "--slat-normalization",
        choices=("normalized", "denormalized"),
        default="normalized",
        help="Whether the input SLat key is normalized sampler output or already denormalized decoder input.",
    )
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = {
        "schema": SCHEMA,
        "status": "running",
        "route": "shape-artifact-decode",
        "input": {
            "path": str(args.input),
            "slat_key": args.slat_key,
            "coords_key": args.coords_key,
            "spatial_coords_key": args.spatial_coords_key,
        },
        "output": {
            "glb_path": str(args.output_glb),
            "artifact_path": str(args.output_artifact) if args.output_artifact else None,
        },
    }
    try:
        _write_json(args.report, report)
        decode_shape_artifact(
            args.input,
            args.output_glb,
            args.report,
            output_artifact_path=args.output_artifact,
            slat_key=args.slat_key,
            coords_key=args.coords_key,
            spatial_coords_key=args.spatial_coords_key,
            component_filter_mode=args.component_filter,
            component_filter_min_ratio=args.component_filter_min_ratio,
            slat_normalization=args.slat_normalization,
            decoder_checkpoint=args.decoder_checkpoint,
            resolution=args.resolution,
            overwrite=args.overwrite,
            allow_existing_report=True,
        )
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "phase": "decode_shape_artifact",
                "error": f"{type(exc).__name__}: {exc}",
                "last_trustworthy_evidence": {
                    "input_exists": args.input.exists(),
                    "glb_exists": args.output_glb.exists(),
                    "artifact_exists": args.output_artifact.exists() if args.output_artifact else False,
                    "report_exists": args.report.exists(),
                },
            }
        )
        _write_json(args.report, report)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
