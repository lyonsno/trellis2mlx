#!/usr/bin/env python3
"""Compare MLX Pixal3D projection rows against a NumPy port of the reference formula."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx
import numpy as np

import generate_pixal3d as route


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--overwrite-report", action="store_true")
    parser.add_argument("--feature-source", choices=("synthetic", "mlx-dinov3"), default="synthetic")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--grid-resolution", type=_positive_int, default=32)
    parser.add_argument("--image-size", type=_positive_int, default=512)
    parser.add_argument("--patch-size", type=_positive_int, default=16)
    parser.add_argument("--channels", type=_positive_int, default=1024)
    parser.add_argument("--projection-mode", choices=("native", "bilinear_hr_concat"), default="native")
    parser.add_argument("--hr-feature-size", type=_positive_int, default=512)
    parser.add_argument("--coords-npz", type=Path)
    parser.add_argument("--coords-key", default="lr_coords")
    parser.add_argument("--coord-count", type=_positive_int, default=16)
    parser.add_argument("--camera-angle-x", type=float, default=route.PIXAL3D_DEFAULT_CAMERA_ANGLE_X)
    parser.add_argument("--camera-distance", type=float, default=route.PIXAL3D_DEFAULT_CAMERA_DISTANCE)
    parser.add_argument("--mesh-scale", type=float, default=route.PIXAL3D_DEFAULT_MESH_SCALE)
    return parser.parse_args(argv)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n")


def _run_text(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def _repo_identity() -> dict[str, Any]:
    return {
        "root": str(REPO_ROOT),
        "branch": _run_text(["git", "branch", "--show-current"], cwd=REPO_ROOT),
        "head": _run_text(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT),
        "status_short": _run_text(["git", "status", "--short"], cwd=REPO_ROOT),
    }


def _metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    diff = actual.astype(np.float64) - expected.astype(np.float64)
    abs_diff = np.abs(diff)
    expected_abs = np.abs(expected.astype(np.float64))
    denom = float(expected_abs.max()) if expected_abs.size else 0.0
    return {
        "shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "max_abs": float(abs_diff.max()) if abs_diff.size else 0.0,
        "mean_abs": float(abs_diff.mean()) if abs_diff.size else 0.0,
        "rmse": float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0,
        "max_abs_over_expected_abs_max": float(abs_diff.max() / denom) if denom else 0.0,
        "actual_nonfinite_count": int(actual.size - np.isfinite(actual).sum()),
        "expected_nonfinite_count": int(expected.size - np.isfinite(expected).sum()),
    }


def _synthetic_features(*, image_size: int, patch_size: int, channels: int) -> np.ndarray:
    patch_grid = image_size // patch_size
    if patch_grid * patch_size != image_size:
        raise ValueError("image-size must be divisible by patch-size")
    prefix = np.zeros((1, 5, channels), dtype=np.float32)
    y, x, c = np.meshgrid(
        np.arange(patch_grid, dtype=np.float32),
        np.arange(patch_grid, dtype=np.float32),
        np.arange(channels, dtype=np.float32),
        indexing="ij",
    )
    patches = (1000.0 * y + 100.0 * x + c + 0.125 * y * x).reshape(1, patch_grid * patch_grid, channels)
    return np.concatenate([prefix, patches.astype(np.float32)], axis=1)


def _load_features(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    if args.feature_source == "synthetic":
        features = _synthetic_features(
            image_size=args.image_size,
            patch_size=args.patch_size,
            channels=args.channels,
        )
        return features, {
            "source": "synthetic",
            "shape": list(features.shape),
            "channels": args.channels,
        }

    if args.image is None:
        raise ValueError("--feature-source mlx-dinov3 requires --image")
    features_mx, feature_report = route.extract_pixal3d_image_features(args.image, image_size=args.image_size)
    mx.eval(features_mx)
    features = np.array(features_mx, dtype=np.float32)
    return features, {
        **feature_report,
        "source": "native_mlx_dinov3",
        "shape": list(features.shape),
    }


def _split_features(features: np.ndarray, *, image_size: int, patch_size: int) -> tuple[np.ndarray, np.ndarray]:
    patch_grid = image_size // patch_size
    prefix = features[:, :5, :]
    patches = features[:, 5:, :]
    if patches.shape[1] != patch_grid * patch_grid:
        raise ValueError(f"patch token count {patches.shape[1]} does not match patch grid {patch_grid}x{patch_grid}")
    return prefix, patches.reshape(features.shape[0], patch_grid, patch_grid, features.shape[2])


def _reference_project_points(
    points_3d: np.ndarray,
    transform_matrix: np.ndarray,
    camera_angle_x: np.ndarray,
    *,
    resolution: int,
) -> np.ndarray:
    if points_3d.ndim == 2:
        points_3d = np.broadcast_to(points_3d[None, :, :], (transform_matrix.shape[0], points_3d.shape[0], 3))
    bsz, count, _ = points_3d.shape
    ones = np.ones((bsz, count, 1), dtype=np.float32)
    points_h = np.concatenate([points_3d.astype(np.float32), ones], axis=-1)
    world_to_camera = np.linalg.inv(transform_matrix.astype(np.float32))
    points_camera = np.matmul(points_h, np.swapaxes(world_to_camera, -1, -2))[..., :3]
    x_cam = points_camera[..., 0]
    y_cam = points_camera[..., 1]
    z_cam = points_camera[..., 2]
    focal_length = 16.0 / np.tan(camera_angle_x.astype(np.float32) / 2.0)
    focal_pixels = (focal_length * resolution / 32.0)[:, None]
    x_ndc = focal_pixels * x_cam / (-z_cam + 1e-8)
    y_ndc = focal_pixels * y_cam / (-z_cam + 1e-8)
    x_pixel = x_ndc + resolution / 2.0
    y_pixel = -y_ndc + resolution / 2.0
    return np.stack([x_pixel, y_pixel], axis=-1).astype(np.float32)


def _reference_sample_features(feature_map: np.ndarray, queries_ndc: np.ndarray, *, bhwc: bool = True) -> np.ndarray:
    if not bhwc:
        feature_map = np.transpose(feature_map, (0, 2, 3, 1))
    bsz, height, width, channels = feature_map.shape
    x = ((queries_ndc[..., 0] + 1.0) * width - 1.0) / 2.0
    y = ((queries_ndc[..., 1] + 1.0) * height - 1.0) / 2.0
    x = np.clip(x, 0.0, width - 1.0)
    y = np.clip(y, 0.0, height - 1.0)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    batch = np.arange(bsz)[:, None]
    top_left = feature_map[batch, y0, x0]
    top_right = feature_map[batch, y0, x1]
    bottom_left = feature_map[batch, y1, x0]
    bottom_right = feature_map[batch, y1, x1]
    dx = (x - x0)[..., None]
    dy = (y - y0)[..., None]
    return (
        top_left * (1.0 - dx) * (1.0 - dy)
        + top_right * dx * (1.0 - dy)
        + bottom_left * (1.0 - dx) * dy
        + bottom_right * dx * dy
    ).astype(np.float32).reshape(bsz, queries_ndc.shape[1], channels)


def _reference_resize_bhwc(feature_map: np.ndarray, target_size: int) -> np.ndarray:
    bsz, _, _, channels = feature_map.shape
    xs = (np.arange(target_size, dtype=np.float32) + 0.5) / target_size * 2.0 - 1.0
    ys = (np.arange(target_size, dtype=np.float32) + 0.5) / target_size * 2.0 - 1.0
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    query = np.stack([grid_x, grid_y], axis=-1).reshape(1, target_size * target_size, 2)
    query = np.broadcast_to(query, (bsz, target_size * target_size, 2))
    resized = _reference_sample_features(feature_map, query, bhwc=True)
    return resized.reshape(bsz, target_size, target_size, channels)


def _reference_proj_grid(
    feature_map: np.ndarray,
    *,
    grid_resolution: int,
    image_resolution: int,
    camera_angle_x: float,
    distance: float,
    mesh_scale: float,
    bhwc: bool = True,
) -> np.ndarray:
    if not bhwc:
        bsz = feature_map.shape[0]
    else:
        bsz = feature_map.shape[0]
    one_dim = np.linspace(-1.0, 1.0, grid_resolution, dtype=np.float32)
    x, y, z = np.meshgrid(one_dim, one_dim, one_dim, indexing="ij")
    grid_points = np.stack((x, y, z), axis=-1)
    rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    grid_points = (grid_points @ rotation.T).reshape(-1, 3)
    grid_points = np.broadcast_to(grid_points[None, :, :], (bsz, grid_points.shape[0], 3)).copy()
    grid_points = grid_points / mesh_scale / 2.0
    transform = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, -2.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    transform = np.broadcast_to(transform[None, :, :], (bsz, 4, 4)).copy()
    transform[:, 1, 3] = -distance
    image_points = _reference_project_points(
        grid_points,
        transform,
        np.full((bsz,), camera_angle_x, dtype=np.float32),
        resolution=image_resolution,
    )
    image_points_norm = (image_points + 0.5) / image_resolution * 2.0 - 1.0
    return _reference_sample_features(feature_map, image_points_norm, bhwc=bhwc)


def _reference_context(features: np.ndarray, args: argparse.Namespace) -> dict[str, np.ndarray]:
    global_features, patch_map = _split_features(features, image_size=args.image_size, patch_size=args.patch_size)
    projected_lr = _reference_proj_grid(
        patch_map,
        grid_resolution=args.grid_resolution,
        image_resolution=args.image_size,
        camera_angle_x=args.camera_angle_x,
        distance=args.camera_distance,
        mesh_scale=args.mesh_scale,
        bhwc=True,
    )
    if args.projection_mode == "native":
        projected = projected_lr
    else:
        hr_patch_map = _reference_resize_bhwc(patch_map, args.hr_feature_size)
        projected_hr = _reference_proj_grid(
            hr_patch_map,
            grid_resolution=args.grid_resolution,
            image_resolution=args.image_size,
            camera_angle_x=args.camera_angle_x,
            distance=args.camera_distance,
            mesh_scale=args.mesh_scale,
            bhwc=True,
        )
        projected = np.concatenate([projected_lr, projected_hr], axis=-1)
    return {"global": global_features.astype(np.float32), "proj": projected.astype(np.float32)}


def _flat_indices(coords: np.ndarray, grid_resolution: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape [N,3], got {list(coords.shape)}")
    if len(coords) and ((coords < 0).any() or (coords >= grid_resolution).any()):
        raise ValueError(f"coords are outside grid resolution {grid_resolution}")
    return coords[:, 0] * grid_resolution * grid_resolution + coords[:, 1] * grid_resolution + coords[:, 2]


def _load_coords(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    if args.coords_npz is None:
        grid = args.grid_resolution
        coords = np.array(
            [
                [0, 1 % grid, 2 % grid],
                [1 % grid, 2 % grid, 3 % grid],
                [max(grid - 1, 0), 1 % grid, 0],
                [2 % grid, max(grid - 1, 0), 1 % grid],
            ],
            dtype=np.int32,
        )
        return coords, {"source": "synthetic-default", "key": None, "path": None, "available_keys": None}

    if not args.coords_npz.exists():
        raise FileNotFoundError(f"coords npz not found: {args.coords_npz}")
    with np.load(args.coords_npz) as data:
        keys = list(data.files)
        if args.coords_key not in data:
            raise KeyError(f"coords key {args.coords_key!r} not found in {args.coords_npz}; available keys: {keys}")
        coords = np.asarray(data[args.coords_key], dtype=np.int32)
    return coords, {"source": "npz", "path": str(args.coords_npz), "key": args.coords_key, "available_keys": keys}


def _select_coords(coords: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    if len(coords) == 0:
        raise ValueError("coordinate artifact contains no coords")
    if len(coords) <= count:
        indices = np.arange(len(coords), dtype=np.int32)
    else:
        indices = np.linspace(0, len(coords) - 1, count, dtype=np.int32)
    return coords[indices], indices


def _permutation_sweep(reference_full: np.ndarray, selected_coords: np.ndarray, grid_resolution: int) -> dict[str, Any]:
    permutations = {
        "xyz": (0, 1, 2),
        "xzy": (0, 2, 1),
        "yxz": (1, 0, 2),
        "yzx": (1, 2, 0),
        "zxy": (2, 0, 1),
        "zyx": (2, 1, 0),
    }
    canonical = reference_full[:, _flat_indices(selected_coords, grid_resolution), :]
    report = {}
    for name, order in permutations.items():
        coords = selected_coords[:, order]
        rows = reference_full[:, _flat_indices(coords, grid_resolution), :]
        report[name] = _metrics(rows, canonical)
    return {"canonical_order": "xyz", "permutations": report}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.report.exists() and not args.overwrite_report:
        raise FileExistsError(f"report exists: {args.report}; pass --overwrite-report")

    report: dict[str, Any] = {
        "schema": "trellis2mlx.pixal3d_projection_parity.v1",
        "status": "running",
        "started_at_unix": time.time(),
        "command_line": sys.argv,
        "repo": _repo_identity(),
        "projection": {
            "mode": args.projection_mode,
            "grid_resolution": args.grid_resolution,
            "image_size": args.image_size,
            "patch_size": args.patch_size,
            "hr_feature_size": args.hr_feature_size if args.projection_mode == "bilinear_hr_concat" else None,
            "camera_angle_x": args.camera_angle_x,
            "camera_distance": args.camera_distance,
            "mesh_scale": args.mesh_scale,
            "reference_route": "numpy-port-of-Pixal3D-ProjGrid-and-torch-grid_sample-align_corners_false",
            "naf_parity": False,
        },
        "last_trustworthy_evidence": {"phase": "initialized"},
    }
    write_report(args.report, report)

    try:
        try:
            coords, coord_source = _load_coords(args)
        except Exception as exc:
            report["status"] = "failed"
            report["failure"] = {"phase": "load_coords", "error": str(exc)}
            report["last_trustworthy_evidence"] = {"phase": "initialized"}
            write_report(args.report, report)
            raise
        selected_coords, selected_indices = _select_coords(coords, args.coord_count)
        _flat_indices(selected_coords, args.grid_resolution)
        report["coords"] = {
            **coord_source,
            "total_count": int(len(coords)),
            "selected_count": int(len(selected_coords)),
            "selected_indices": selected_indices.tolist(),
            "selected_coords": selected_coords.tolist(),
            "assumed_order": "xyz",
        }
        report["last_trustworthy_evidence"] = {"phase": "coords_loaded"}
        write_report(args.report, report)

        features, feature_report = _load_features(args)
        report["features"] = feature_report
        report["last_trustworthy_evidence"] = {"phase": "features_loaded"}
        write_report(args.report, report)

        config = {"resolution": args.grid_resolution}
        mlx_context = route.pixal3d_context_from_features(
            mx.array(features),
            config,
            image_size=args.image_size,
            patch_size=args.patch_size,
            projection_mode=args.projection_mode,
            hr_feature_size=args.hr_feature_size,
            camera_angle_x=args.camera_angle_x,
            distance=args.camera_distance,
            mesh_scale=args.mesh_scale,
        )
        mx.eval(mlx_context["global"], mlx_context["proj"])
        mlx_np = {key: np.array(value, dtype=np.float32) for key, value in mlx_context.items()}
        ref_np = _reference_context(features, args)
        selected_flat = _flat_indices(selected_coords, args.grid_resolution)
        report["metrics"] = {
            "global": _metrics(mlx_np["global"], ref_np["global"]),
            "full_projection": _metrics(mlx_np["proj"], ref_np["proj"]),
            "selected_rows": _metrics(mlx_np["proj"][:, selected_flat, :], ref_np["proj"][:, selected_flat, :]),
        }
        report["coordinate_order_sweep"] = _permutation_sweep(ref_np["proj"], selected_coords, args.grid_resolution)
        report["status"] = "ok"
        report["last_trustworthy_evidence"] = {"phase": "parity_metrics"}
        report["updated_at_unix"] = time.time()
        write_report(args.report, report)
        return report
    except Exception as exc:
        if report.get("status") != "failed":
            report["status"] = "failed"
            report["failure"] = {"phase": "unexpected", "error": f"{type(exc).__name__}: {exc}"}
            report["updated_at_unix"] = time.time()
            write_report(args.report, report)
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        print(f"projection parity failed: {exc}", file=sys.stderr)
        return 1
    print(f"wrote report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
