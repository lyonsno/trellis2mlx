#!/usr/bin/env python3
"""Pixal3D-MLX route-control harness.

This is not yet the full image-to-GLB pipeline. It proves that the effective
route is the Pixal3D projected-conditioning route, writes a JSON report, and
fails loud if the requested route silently collapses to vanilla TRELLIS global
conditioning.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.utils
import numpy as np


SCHEMA = "trellis2mlx.pixal3d_route.v1"
PIXAL3D_DEFAULT_CAMERA_ANGLE_X = 0.8575560450553894
PIXAL3D_DEFAULT_CAMERA_DISTANCE = 2.0
PIXAL3D_DEFAULT_MESH_SCALE = 1.0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def default_report_path() -> Path:
    run_id = time.strftime("%Y%m%d-%H%M%S")
    unique = f"{os.getpid()}-{time.time_ns()}"
    return Path("/tmp") / f"pixal3d-mlx-route-{run_id}-{unique}" / "report.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-route", action="store_true", help="Run the projected-conditioning route smoke")
    parser.add_argument("--report", type=Path, default=None, help="JSON report path")
    parser.add_argument("--overwrite-report", action="store_true", help="Overwrite an existing JSON report")
    parser.add_argument("--repo-root", type=Path, default=None, help="Effective pixal3d-mlx repo root")
    parser.add_argument("--checkpoint", type=Path, help="Optional Pixal3D denoiser checkpoint to inventory")
    parser.add_argument("--checkpoint-config", type=Path, help="Optional Pixal3D checkpoint JSON config for no-allocation shape profiling")
    parser.add_argument(
        "--chunked-quantize-sparse-structure",
        action="store_true",
        help="Quantize eligible sparse-structure checkpoint weights one tensor at a time and report packed payload sizes",
    )
    parser.add_argument("--quantize-bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--quantize-group-size", type=_positive_int, default=64)
    parser.add_argument(
        "--export-packed-sparse-structure",
        type=Path,
        help="Write a packed sparse-structure quantized artifact directory",
    )
    parser.add_argument("--overwrite-artifact", action="store_true", help="Replace an existing packed artifact directory")
    parser.add_argument("--packed-sparse-artifact", type=Path, help="Packed sparse-structure artifact directory for stage smoke")
    parser.add_argument("--run-packed-sparse-stage", action="store_true", help="Run a packed sparse-structure sampler smoke")
    parser.add_argument("--run-image-packed-sparse-stage", action="store_true", help="Run packed sparse stage using image-derived Pixal3D projected conditioning")
    parser.add_argument("--packed-flow-artifact", type=Path, help="Packed generic flow artifact directory for SLat diagnostics")
    parser.add_argument("--packed-flow-stage", default="shape-lr-slat", help="Expected packed generic flow stage")
    parser.add_argument("--run-packed-slat-width-diagnostic", action="store_true", help="Run packed SLat projected-context width diagnostic")
    parser.add_argument(
        "--component-filter-artifact",
        type=Path,
        help="Saved Pixal3D HR SLat artifact to component-filter under the projected route smoke",
    )
    parser.add_argument("--component-filter", choices=("none", "largest", "min_ratio"), default="none")
    parser.add_argument("--component-filter-min-ratio", type=float, default=1e-5)
    parser.add_argument("--component-filter-slat-key", default="hr_slat")
    parser.add_argument("--component-filter-coords-key", default="hr_coords_quantized_1024")
    parser.add_argument("--component-filter-spatial-coords-key", default="hr_coords_3d_1024")
    parser.add_argument("--projection-mode", choices=("native", "bilinear_hr_concat"), default="native", help="Projected feature mode for SLat diagnostics")
    parser.add_argument("--hr-feature-size", type=_positive_int, default=None, help="Bilinear HR feature side length for SLat diagnostics")
    parser.add_argument("--camera-angle-x", type=float, default=PIXAL3D_DEFAULT_CAMERA_ANGLE_X, help="Pixal3D projection camera horizontal FOV/ray angle")
    parser.add_argument("--camera-distance", type=float, default=PIXAL3D_DEFAULT_CAMERA_DISTANCE, help="Pixal3D projection camera distance")
    parser.add_argument("--mesh-scale", type=float, default=PIXAL3D_DEFAULT_MESH_SCALE, help="Pixal3D projection mesh scale")
    parser.add_argument("--image", type=Path, help="Image path for DINOv3/Pixal3D projected conditioning")
    parser.add_argument("--sparse-stage-steps", type=_positive_int, default=1)
    parser.add_argument("--grid-resolution", type=_positive_int, default=2)
    parser.add_argument("--image-size", type=_positive_int, default=32)
    parser.add_argument("--patch-size", type=_positive_int, default=16)
    parser.add_argument("--context-channels", type=_positive_int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.repo_root is None:
        args.repo_root = Path.cwd()
    if args.report is None:
        args.report = default_report_path()
    return args


def ensure_report_writable(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"report already exists: {path}; pass --overwrite-report to replace it")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n")


def _run_text(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def _repo_identity(repo_root: Path) -> dict[str, Any]:
    return {
        "root": str(repo_root),
        "head": _run_text(["git", "rev-parse", "HEAD"], cwd=repo_root),
        "branch": _run_text(["git", "branch", "--show-current"], cwd=repo_root),
        "status_short": _run_text(["git", "status", "--short"], cwd=repo_root),
    }


def _host_identity() -> dict[str, Any]:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "default_device": str(mx.default_device()),
    }


def _model_keys(model) -> set[str]:
    return {key for key, _ in mlx.utils.tree_flatten(model.parameters())}


def _model_shapes(models: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    shapes = {}
    for model in models.values():
        for key, value in mlx.utils.tree_flatten(model.parameters()):
            shapes[key] = tuple(value.shape)
    return shapes


def inspect_checkpoint(checkpoint: Path | None, models: dict[str, Any]) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "path": str(checkpoint) if checkpoint is not None else None,
        "exists": checkpoint.exists() if checkpoint is not None else None,
        "size_bytes": checkpoint.stat().st_size if checkpoint is not None and checkpoint.exists() else None,
        "loaded": 0,
        "missing": None,
        "skipped": 0,
        "projection_keys_present": None,
        "inventory_mode": "header-only",
    }
    if checkpoint is None:
        return identity
    if not checkpoint.exists():
        identity["error"] = "checkpoint does not exist"
        return identity

    from safetensors import safe_open
    from trellmlx.weight_loader import _remap_key

    model_shapes = _model_shapes(models)
    model_keys = set(model_shapes)
    checkpoint_keys = []
    checkpoint_shapes = {}
    with safe_open(str(checkpoint), framework="numpy") as sf:
        checkpoint_keys = list(sf.keys())
        for key in checkpoint_keys:
            checkpoint_shapes[key] = tuple(sf.get_slice(key).get_shape())

    remapped = [_remap_key(key) for key in checkpoint_keys]
    matched_by_name = [key for key in remapped if key in model_keys]
    matched_by_shape = [
        remapped_key
        for raw_key, remapped_key in zip(checkpoint_keys, remapped)
        if remapped_key in model_shapes and checkpoint_shapes[raw_key] == model_shapes[remapped_key]
    ]
    projection_keys = [key for key in remapped if ".cross_attn.proj_linear." in key]
    wrapped_cross_attn_keys = [key for key in checkpoint_keys if ".cross_attn.cross_attn_block." in key]
    plain_cross_attn_keys = [
        key
        for key in checkpoint_keys
        if ".cross_attn." in key
        and ".cross_attn.cross_attn_block." not in key
        and ".cross_attn.proj_linear." not in key
    ]
    plain_cross_attn_remapped = [
        remapped_key
        for raw_key, remapped_key in zip(checkpoint_keys, remapped)
        if raw_key in plain_cross_attn_keys and remapped_key != raw_key
    ]
    shape_mismatch = [
        {
            "key": raw_key,
            "remapped_key": remapped_key,
            "checkpoint_shape": list(checkpoint_shapes[raw_key]),
            "model_shape": list(model_shapes[remapped_key]),
        }
        for raw_key, remapped_key in zip(checkpoint_keys, remapped)
        if remapped_key in model_shapes and checkpoint_shapes[raw_key] != model_shapes[remapped_key]
    ]
    identity.update({
        "total_keys": len(checkpoint_keys),
        "loaded": len(matched_by_shape),
        "missing": len(model_keys - set(matched_by_name)),
        "skipped": len(checkpoint_keys) - len(matched_by_name),
        "matched_by_name": len(matched_by_name),
        "matched_by_shape": len(matched_by_shape),
        "shape_mismatch_count": len(shape_mismatch),
        "shape_mismatch_samples": shape_mismatch[:10],
        "projection_keys_present": bool(projection_keys),
        "projection_key_count": len(projection_keys),
        "wrapped_cross_attn_key_count": len(wrapped_cross_attn_keys),
        "plain_cross_attn_key_count": len(plain_cross_attn_keys),
        "plain_cross_attn_remapped_count": len(plain_cross_attn_remapped),
        "sample_projection_keys": projection_keys[:10],
    })
    return identity


def _flow_config_profile(config_path: Path) -> dict[str, Any]:
    payload = json.loads(config_path.read_text())
    args = payload.get("args", {})
    channels = int(args["model_channels"])
    num_heads = int(args["num_heads"])
    head_dim = channels // num_heads
    cond_channels = int(args.get("cond_channels", args.get("context_channels", channels)))
    proj_in_channels = int(args.get("proj_in_channels", cond_channels))
    mlp_hidden = int(round(float(args["mlp_ratio"]) * channels))
    return {
        "path": str(config_path),
        "name": payload.get("name"),
        "resolution": int(args.get("resolution", 0)),
        "in_channels": int(args["in_channels"]),
        "out_channels": int(args["out_channels"]),
        "model_channels": channels,
        "cond_channels": cond_channels,
        "proj_in_channels": proj_in_channels,
        "num_blocks": int(args["num_blocks"]),
        "num_heads": num_heads,
        "head_dim": head_dim,
        "mlp_hidden": mlp_hidden,
        "image_attn_mode": args.get("image_attn_mode"),
    }


def _linear_shapes(prefix: str, out_features: int, in_features: int) -> dict[str, tuple[int, ...]]:
    return {
        f"{prefix}.weight": (out_features, in_features),
        f"{prefix}.bias": (out_features,),
    }


def expected_flow_shapes_from_config(config_path: Path) -> tuple[dict[str, Any], dict[str, tuple[int, ...]]]:
    config = _flow_config_profile(config_path)
    C = config["model_channels"]
    in_C = config["in_channels"]
    out_C = config["out_channels"]
    ctx_C = config["cond_channels"]
    proj_C = config["proj_in_channels"]
    H = config["num_heads"]
    D = config["head_dim"]
    mlp_hidden = config["mlp_hidden"]
    shapes: dict[str, tuple[int, ...]] = {}

    shapes.update(_linear_shapes("t_embedder.mlp_0", C, 256))
    shapes.update(_linear_shapes("t_embedder.mlp_2", C, C))
    shapes.update(_linear_shapes("input_layer", C, in_C))
    shapes.update(_linear_shapes("out_layer", out_C, C))
    shapes.update(_linear_shapes("adaLN_modulation.layers.1", 6 * C, C))

    for i in range(config["num_blocks"]):
        prefix = f"blocks.{i}"
        shapes[f"{prefix}.modulation"] = (6 * C,)
        shapes[f"{prefix}.norm2.weight"] = (C,)
        shapes[f"{prefix}.norm2.bias"] = (C,)

        shapes.update(_linear_shapes(f"{prefix}.self_attn.to_qkv", 3 * C, C))
        shapes.update(_linear_shapes(f"{prefix}.self_attn.to_out", C, C))
        shapes[f"{prefix}.self_attn.q_rms_norm.gamma"] = (H, D)
        shapes[f"{prefix}.self_attn.k_rms_norm.gamma"] = (H, D)

        cross = f"{prefix}.cross_attn.cross_attn_block"
        shapes.update(_linear_shapes(f"{cross}.to_q", C, C))
        shapes.update(_linear_shapes(f"{cross}.to_kv", 2 * C, ctx_C))
        shapes.update(_linear_shapes(f"{cross}.to_out", C, C))
        shapes[f"{cross}.q_rms_norm.gamma"] = (H, D)
        shapes[f"{cross}.k_rms_norm.gamma"] = (H, D)
        shapes.update(_linear_shapes(f"{prefix}.cross_attn.proj_linear", C, proj_C))

        shapes.update(_linear_shapes(f"{prefix}.mlp.mlp_0", mlp_hidden, C))
        shapes.update(_linear_shapes(f"{prefix}.mlp.mlp_2", C, mlp_hidden))

    return config, shapes


def profile_checkpoint_architecture(checkpoint: Path, config_path: Path) -> dict[str, Any]:
    from safetensors import safe_open
    from trellmlx.weight_loader import _remap_key

    config, expected_shapes = expected_flow_shapes_from_config(config_path)
    checkpoint_shapes = {}
    with safe_open(str(checkpoint), framework="numpy") as sf:
        for key in sf.keys():
            checkpoint_shapes[key] = tuple(sf.get_slice(key).get_shape())

    remapped_shapes = {_remap_key(key): shape for key, shape in checkpoint_shapes.items()}
    matched = []
    shape_mismatch = []
    for key, expected in expected_shapes.items():
        if key not in remapped_shapes:
            continue
        actual = remapped_shapes[key]
        if actual == expected:
            matched.append(key)
        else:
            shape_mismatch.append({
                "key": key,
                "expected_shape": list(expected),
                "checkpoint_shape": list(actual),
            })

    projection_expected = [key for key in expected_shapes if ".cross_attn.proj_linear." in key]
    wrapped_cross_expected = [key for key in expected_shapes if ".cross_attn.cross_attn_block." in key]
    return {
        "mode": "config-header-no-allocation",
        "allocates_model": False,
        "config": config,
        "checkpoint": {
            "path": str(checkpoint),
            "exists": checkpoint.exists(),
            "size_bytes": checkpoint.stat().st_size if checkpoint.exists() else None,
        },
        "expected_shape_count": len(expected_shapes),
        "checkpoint_key_count": len(checkpoint_shapes),
        "matched_shape_count": len(matched),
        "missing_expected_count": len(set(expected_shapes) - set(remapped_shapes)),
        "extra_checkpoint_key_count": len(set(remapped_shapes) - set(expected_shapes)),
        "extra_checkpoint_key_samples": sorted(set(remapped_shapes) - set(expected_shapes))[:10],
        "shape_mismatch_count": len(shape_mismatch),
        "shape_mismatch_samples": shape_mismatch[:10],
        "projection_expected_count": len(projection_expected),
        "projection_shape_match_count": sum(1 for key in projection_expected if key in matched),
        "wrapped_cross_attn_expected_count": len(wrapped_cross_expected),
        "wrapped_cross_attn_shape_match_count": sum(1 for key in wrapped_cross_expected if key in matched),
    }


def _dtype_name(value: Any) -> str:
    return str(value).replace("mlx.core.", "").replace("numpy.", "")


def _array_nbytes(array: Any) -> int:
    if hasattr(array, "nbytes"):
        return int(array.nbytes)
    return int(np.prod(array.shape, dtype=np.int64)) * int(array.itemsize)


def _quantized_linear_weight_skip_reason(shape: tuple[int, ...], group_size: int) -> str | None:
    if len(shape) != 2:
        return "not_2d_linear_weight"
    out_features, in_features = shape
    if out_features < group_size or in_features < group_size:
        return "below_group_size"
    if in_features % group_size != 0:
        return "input_dim_not_divisible_by_group_size"
    return None


def sparse_structure_model_from_config(config_path: Path):
    from trellmlx.models.pixal3d_flow import Pixal3DSparseStructureFlowModel

    config = _flow_config_profile(config_path)
    return Pixal3DSparseStructureFlowModel(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        model_channels=config["model_channels"],
        num_heads=config["num_heads"],
        num_blocks=config["num_blocks"],
        mlp_hidden=config["mlp_hidden"],
        context_channels=config["cond_channels"],
        proj_in_channels=config["proj_in_channels"],
        resolution=config["resolution"],
    )


def flow_model_from_config(config_path: Path):
    from trellmlx.models.pixal3d_flow import Pixal3DSLatFlowModel, Pixal3DSparseStructureFlowModel

    config = _flow_config_profile(config_path)
    if config["name"] == "SparseStructureFlowModel":
        return Pixal3DSparseStructureFlowModel(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            model_channels=config["model_channels"],
            num_heads=config["num_heads"],
            num_blocks=config["num_blocks"],
            mlp_hidden=config["mlp_hidden"],
            context_channels=config["cond_channels"],
            proj_in_channels=config["proj_in_channels"],
            resolution=config["resolution"],
        )
    if config["name"] in {"ElasticSLatFlowModel", "SLatFlowModel"}:
        return Pixal3DSLatFlowModel(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            model_channels=config["model_channels"],
            num_heads=config["num_heads"],
            num_blocks=config["num_blocks"],
            mlp_hidden=config["mlp_hidden"],
            context_channels=config["cond_channels"],
            proj_in_channels=config["proj_in_channels"],
        )
    raise ValueError(f"unsupported Pixal3D flow config name: {config['name']}")


def chunked_quantize_sparse_structure_checkpoint(
    checkpoint: Path,
    config_path: Path,
    *,
    bits: int = 4,
    group_size: int = 64,
) -> dict[str, Any]:
    from safetensors import safe_open
    from trellmlx.weight_loader import _remap_key

    if bits not in (4, 8):
        raise ValueError("chunked sparse-structure quantization requires bits=4 or bits=8")
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    if not config_path.exists():
        raise FileNotFoundError(f"checkpoint config does not exist: {config_path}")

    config, expected_shapes = expected_flow_shapes_from_config(config_path)
    profile = profile_checkpoint_architecture(checkpoint, config_path)
    if profile["shape_mismatch_count"]:
        raise RuntimeError(
            "checkpoint/config shape mismatch blocks chunked quantization: "
            f"{profile['shape_mismatch_samples'][:3]}"
        )

    checkpoint_entries: list[dict[str, Any]] = []
    with safe_open(str(checkpoint), framework="numpy") as sf:
        for raw_key in sf.keys():
            tensor_slice = sf.get_slice(raw_key)
            shape = tuple(tensor_slice.get_shape())
            checkpoint_entries.append({
                "raw_key": raw_key,
                "key": _remap_key(raw_key),
                "shape": shape,
                "dtype": _dtype_name(tensor_slice.get_dtype()),
            })

    checkpoint_entries.sort(key=lambda item: item["key"])
    quantized_weights = []
    skipped_weights = []
    nonweight_key_count = 0
    extra_keys = []
    original_weight_bytes = 0
    packed_weight_bytes = 0
    scale_bytes = 0
    biases_bytes = 0
    peak_materialized_tensor_bytes = 0
    projection_quantized_count = 0

    with safe_open(str(checkpoint), framework="numpy") as sf:
        for entry in checkpoint_entries:
            key = entry["key"]
            raw_key = entry["raw_key"]
            shape = entry["shape"]
            if key not in expected_shapes:
                extra_keys.append(key)
                continue
            if not key.endswith(".weight"):
                nonweight_key_count += 1
                continue

            reason = _quantized_linear_weight_skip_reason(shape, group_size)
            if reason is not None:
                skipped_weights.append({
                    "key": key,
                    "raw_key": raw_key,
                    "shape": list(shape),
                    "dtype": entry["dtype"],
                    "reason": reason,
                })
                continue

            tensor = sf.get_tensor(raw_key)
            peak_materialized_tensor_bytes = max(
                peak_materialized_tensor_bytes,
                _array_nbytes(tensor),
            )
            source = mx.array(tensor)
            packed, scales, biases = mx.quantize(source, group_size=group_size, bits=bits)
            mx.eval(packed, scales, biases)
            source_bytes = _array_nbytes(tensor)
            packed_bytes = _array_nbytes(packed)
            scales_bytes = _array_nbytes(scales)
            per_weight_biases_bytes = _array_nbytes(biases)
            original_weight_bytes += source_bytes
            packed_weight_bytes += packed_bytes
            scale_bytes += scales_bytes
            biases_bytes += per_weight_biases_bytes
            if ".cross_attn.proj_linear.weight" in key:
                projection_quantized_count += 1
            quantized_weights.append({
                "key": key,
                "raw_key": raw_key,
                "shape": list(shape),
                "source_dtype": entry["dtype"],
                "source_bytes": source_bytes,
                "packed_shape": list(packed.shape),
                "packed_dtype": _dtype_name(packed.dtype),
                "packed_bytes": packed_bytes,
                "scale_shape": list(scales.shape),
                "scale_dtype": _dtype_name(scales.dtype),
                "scale_bytes": scales_bytes,
                "biases_shape": list(biases.shape),
                "biases_dtype": _dtype_name(biases.dtype),
                "biases_bytes": per_weight_biases_bytes,
            })
            del source, packed, scales, biases, tensor
            mx.clear_cache()

    quantized_payload_bytes = packed_weight_bytes + scale_bytes + biases_bytes
    return {
        "mode": "chunked-header-guarded-mx-quantize",
        "stage": "sparse-structure",
        "allocates_full_model": False,
        "loads_all_tensors_at_once": False,
        "tensor_materialization": "one safetensors tensor per quantized weight",
        "bits": bits,
        "group_size": group_size,
        "config": config,
        "checkpoint": {
            "path": str(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "key_count": len(checkpoint_entries),
        },
        "profile": profile,
        "eligible_weight_count": len(quantized_weights),
        "quantized_weight_count": len(quantized_weights),
        "projection_weight_quantized_count": projection_quantized_count,
        "skipped_weight_count": len(skipped_weights),
        "nonweight_key_count": nonweight_key_count,
        "extra_checkpoint_key_count": len(extra_keys),
        "extra_checkpoint_key_samples": extra_keys[:10],
        "original_weight_bytes": original_weight_bytes,
        "packed_weight_bytes": packed_weight_bytes,
        "scale_bytes": scale_bytes,
        "biases_bytes": biases_bytes,
        "quantized_payload_bytes": quantized_payload_bytes,
        "compression_ratio_vs_quantized_weights": (
            original_weight_bytes / quantized_payload_bytes
            if quantized_payload_bytes
            else None
        ),
        "peak_materialized_tensor_bytes": peak_materialized_tensor_bytes,
        "sample_quantized_weights": quantized_weights[:10],
        "sample_skipped_weights": skipped_weights[:10],
    }


PACKED_SPARSE_ARTIFACT_SCHEMA = "trellis2mlx.pixal3d_sparse_quant_artifact.v1"
PACKED_FLOW_ARTIFACT_SCHEMA = "trellis2mlx.pixal3d_packed_flow_artifact.v1"


def _prepare_artifact_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"artifact directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def _np_from_mx(array: mx.array) -> np.ndarray:
    mx.eval(array)
    return np.array(array)


def export_packed_sparse_structure_artifact(
    checkpoint: Path,
    config_path: Path,
    output_dir: Path,
    *,
    bits: int = 4,
    group_size: int = 64,
    overwrite: bool = False,
) -> dict[str, Any]:
    from safetensors import safe_open
    from safetensors.numpy import save_file
    from trellmlx.weight_loader import _remap_key

    if bits not in (4, 8):
        raise ValueError("packed sparse-structure export requires bits=4 or bits=8")
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    if not config_path.exists():
        raise FileNotFoundError(f"checkpoint config does not exist: {config_path}")

    _prepare_artifact_dir(output_dir, overwrite=overwrite)
    tensor_path = output_dir / "model.safetensors"
    manifest_path = output_dir / "manifest.json"

    config, expected_shapes = expected_flow_shapes_from_config(config_path)
    profile = profile_checkpoint_architecture(checkpoint, config_path)
    if profile["shape_mismatch_count"]:
        raise RuntimeError(
            "checkpoint/config shape mismatch blocks packed artifact export: "
            f"{profile['shape_mismatch_samples'][:3]}"
        )

    entries = []
    with safe_open(str(checkpoint), framework="numpy") as sf:
        for raw_key in sf.keys():
            tensor_slice = sf.get_slice(raw_key)
            entries.append({
                "raw_key": raw_key,
                "key": _remap_key(raw_key),
                "shape": tuple(tensor_slice.get_shape()),
                "dtype": _dtype_name(tensor_slice.get_dtype()),
            })
    entries.sort(key=lambda item: item["key"])

    tensors: dict[str, np.ndarray] = {}
    quantized_modules = []
    skipped_weights = []
    extra_keys = []
    fp_remainder_count = 0
    original_weight_bytes = 0
    packed_weight_bytes = 0
    scale_bytes = 0
    biases_bytes = 0
    peak_materialized_tensor_bytes = 0

    with safe_open(str(checkpoint), framework="numpy") as sf:
        for entry in entries:
            key = entry["key"]
            raw_key = entry["raw_key"]
            shape = entry["shape"]
            if key not in expected_shapes:
                extra_keys.append(key)
                continue

            if key.endswith(".weight"):
                reason = _quantized_linear_weight_skip_reason(shape, group_size)
            else:
                reason = "not_weight"

            if key.endswith(".weight") and reason is None:
                tensor = sf.get_tensor(raw_key)
                peak_materialized_tensor_bytes = max(peak_materialized_tensor_bytes, _array_nbytes(tensor))
                source = mx.array(tensor)
                packed, scales, biases = mx.quantize(source, group_size=group_size, bits=bits)
                mx.eval(packed, scales, biases)
                module_key = key.removesuffix(".weight")
                packed_np = _np_from_mx(packed)
                scales_np = _np_from_mx(scales)
                biases_np = _np_from_mx(biases)
                tensors[f"{module_key}.weight"] = packed_np
                tensors[f"{module_key}.scales"] = scales_np
                tensors[f"{module_key}.biases"] = biases_np
                original_weight_bytes += _array_nbytes(tensor)
                packed_weight_bytes += packed_np.nbytes
                scale_bytes += scales_np.nbytes
                biases_bytes += biases_np.nbytes
                quantized_modules.append({
                    "name": module_key,
                    "raw_key": raw_key,
                    "source_shape": list(shape),
                    "source_dtype": entry["dtype"],
                    "packed_shape": list(packed_np.shape),
                    "packed_dtype": str(packed_np.dtype),
                    "scale_shape": list(scales_np.shape),
                    "biases_shape": list(biases_np.shape),
                })
                del tensor, source, packed, scales, biases, packed_np, scales_np, biases_np
                mx.clear_cache()
                continue

            tensor = sf.get_tensor(raw_key)
            peak_materialized_tensor_bytes = max(peak_materialized_tensor_bytes, _array_nbytes(tensor))
            tensors[key] = np.array(tensor)
            fp_remainder_count += 1
            if key.endswith(".weight"):
                skipped_weights.append({
                    "key": key,
                    "raw_key": raw_key,
                    "shape": list(shape),
                    "dtype": entry["dtype"],
                    "reason": reason,
                })

    if not quantized_modules:
        raise RuntimeError("packed sparse-structure export produced no quantized modules")

    save_file(
        tensors,
        tensor_path,
        metadata={
            "schema": PACKED_SPARSE_ARTIFACT_SCHEMA,
            "stage": "sparse-structure",
            "bits": str(bits),
            "group_size": str(group_size),
        },
    )

    quantized_payload_bytes = packed_weight_bytes + scale_bytes + biases_bytes
    manifest = {
        "schema": PACKED_SPARSE_ARTIFACT_SCHEMA,
        "stage": "sparse-structure",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": {
            "checkpoint": str(checkpoint),
            "checkpoint_config": str(config_path),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
        },
        "artifact": {
            "manifest": str(manifest_path),
            "tensor_file": str(tensor_path),
            "tensor_file_size_bytes": tensor_path.stat().st_size,
            "saved_tensor_count": len(tensors),
        },
        "config": config,
        "profile": profile,
        "quantization": {
            "bits": bits,
            "group_size": group_size,
            "mode": "mlx.core.quantize",
        },
        "quantized_weight_count": len(quantized_modules),
        "quantized_module_names": [item["name"] for item in quantized_modules],
        "fp_remainder_tensor_count": fp_remainder_count,
        "skipped_weight_count": len(skipped_weights),
        "extra_checkpoint_key_count": len(extra_keys),
        "extra_checkpoint_key_samples": extra_keys[:10],
        "original_weight_bytes": original_weight_bytes,
        "packed_weight_bytes": packed_weight_bytes,
        "scale_bytes": scale_bytes,
        "biases_bytes": biases_bytes,
        "quantized_payload_bytes": quantized_payload_bytes,
        "compression_ratio_vs_quantized_weights": (
            original_weight_bytes / quantized_payload_bytes
            if quantized_payload_bytes
            else None
        ),
        "peak_materialized_tensor_bytes": peak_materialized_tensor_bytes,
        "sample_quantized_modules": quantized_modules[:10],
        "sample_skipped_weights": skipped_weights[:10],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=_jsonable) + "\n")
    return manifest


def export_packed_flow_artifact(
    checkpoint: Path,
    config_path: Path,
    output_dir: Path,
    *,
    stage: str,
    bits: int = 4,
    group_size: int = 64,
    overwrite: bool = False,
) -> dict[str, Any]:
    from safetensors import safe_open
    from safetensors.numpy import save_file
    from trellmlx.weight_loader import _remap_key

    if bits not in (4, 8):
        raise ValueError("packed flow export requires bits=4 or bits=8")
    if not stage:
        raise ValueError("packed flow export requires a non-empty stage")
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    if not config_path.exists():
        raise FileNotFoundError(f"checkpoint config does not exist: {config_path}")

    _prepare_artifact_dir(output_dir, overwrite=overwrite)
    tensor_path = output_dir / "model.safetensors"
    manifest_path = output_dir / "manifest.json"

    config, expected_shapes = expected_flow_shapes_from_config(config_path)
    profile = profile_checkpoint_architecture(checkpoint, config_path)
    if profile["shape_mismatch_count"]:
        raise RuntimeError(
            "checkpoint/config shape mismatch blocks packed flow export: "
            f"{profile['shape_mismatch_samples'][:3]}"
        )

    entries = []
    with safe_open(str(checkpoint), framework="numpy") as sf:
        for raw_key in sf.keys():
            tensor_slice = sf.get_slice(raw_key)
            entries.append({
                "raw_key": raw_key,
                "key": _remap_key(raw_key),
                "shape": tuple(tensor_slice.get_shape()),
                "dtype": _dtype_name(tensor_slice.get_dtype()),
            })
    entries.sort(key=lambda item: item["key"])

    tensors: dict[str, np.ndarray] = {}
    quantized_modules = []
    skipped_weights = []
    extra_keys = []
    fp_remainder_count = 0
    original_weight_bytes = 0
    packed_weight_bytes = 0
    scale_bytes = 0
    biases_bytes = 0
    peak_materialized_tensor_bytes = 0

    with safe_open(str(checkpoint), framework="numpy") as sf:
        for entry in entries:
            key = entry["key"]
            raw_key = entry["raw_key"]
            shape = entry["shape"]
            if key not in expected_shapes:
                extra_keys.append(key)
                continue

            reason = _quantized_linear_weight_skip_reason(shape, group_size) if key.endswith(".weight") else "not_weight"
            if key.endswith(".weight") and reason is None:
                tensor = sf.get_tensor(raw_key)
                peak_materialized_tensor_bytes = max(peak_materialized_tensor_bytes, _array_nbytes(tensor))
                source = mx.array(tensor)
                packed, scales, biases = mx.quantize(source, group_size=group_size, bits=bits)
                mx.eval(packed, scales, biases)
                module_key = key.removesuffix(".weight")
                packed_np = _np_from_mx(packed)
                scales_np = _np_from_mx(scales)
                biases_np = _np_from_mx(biases)
                tensors[f"{module_key}.weight"] = packed_np
                tensors[f"{module_key}.scales"] = scales_np
                tensors[f"{module_key}.biases"] = biases_np
                original_weight_bytes += _array_nbytes(tensor)
                packed_weight_bytes += packed_np.nbytes
                scale_bytes += scales_np.nbytes
                biases_bytes += biases_np.nbytes
                quantized_modules.append({
                    "name": module_key,
                    "raw_key": raw_key,
                    "source_shape": list(shape),
                    "source_dtype": entry["dtype"],
                    "packed_shape": list(packed_np.shape),
                    "packed_dtype": str(packed_np.dtype),
                    "scale_shape": list(scales_np.shape),
                    "biases_shape": list(biases_np.shape),
                })
                del tensor, source, packed, scales, biases, packed_np, scales_np, biases_np
                mx.clear_cache()
                continue

            tensor = sf.get_tensor(raw_key)
            peak_materialized_tensor_bytes = max(peak_materialized_tensor_bytes, _array_nbytes(tensor))
            tensors[key] = np.array(tensor)
            fp_remainder_count += 1
            if key.endswith(".weight"):
                skipped_weights.append({
                    "key": key,
                    "raw_key": raw_key,
                    "shape": list(shape),
                    "dtype": entry["dtype"],
                    "reason": reason,
                })

    if not quantized_modules:
        raise RuntimeError("packed flow export produced no quantized modules")

    save_file(
        tensors,
        tensor_path,
        metadata={
            "schema": PACKED_FLOW_ARTIFACT_SCHEMA,
            "stage": stage,
            "bits": str(bits),
            "group_size": str(group_size),
        },
    )

    quantized_payload_bytes = packed_weight_bytes + scale_bytes + biases_bytes
    manifest = {
        "schema": PACKED_FLOW_ARTIFACT_SCHEMA,
        "stage": stage,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": {
            "checkpoint": str(checkpoint),
            "checkpoint_config": str(config_path),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
        },
        "artifact": {
            "manifest": str(manifest_path),
            "tensor_file": str(tensor_path),
            "tensor_file_size_bytes": tensor_path.stat().st_size,
            "saved_tensor_count": len(tensors),
        },
        "config": config,
        "profile": profile,
        "quantization": {
            "bits": bits,
            "group_size": group_size,
            "mode": "mlx.core.quantize",
        },
        "quantized_weight_count": len(quantized_modules),
        "quantized_module_names": [item["name"] for item in quantized_modules],
        "fp_remainder_tensor_count": fp_remainder_count,
        "skipped_weight_count": len(skipped_weights),
        "extra_checkpoint_key_count": len(extra_keys),
        "extra_checkpoint_key_samples": extra_keys[:10],
        "original_weight_bytes": original_weight_bytes,
        "packed_weight_bytes": packed_weight_bytes,
        "scale_bytes": scale_bytes,
        "biases_bytes": biases_bytes,
        "quantized_payload_bytes": quantized_payload_bytes,
        "compression_ratio_vs_quantized_weights": (
            original_weight_bytes / quantized_payload_bytes
            if quantized_payload_bytes
            else None
        ),
        "peak_materialized_tensor_bytes": peak_materialized_tensor_bytes,
        "sample_quantized_modules": quantized_modules[:10],
        "sample_skipped_weights": skipped_weights[:10],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=_jsonable) + "\n")
    return manifest


def load_packed_flow_artifact(
    model: Any,
    artifact_dir: Path,
    *,
    expected_stage: str | None = None,
) -> dict[str, Any]:
    import mlx.nn as nn
    from safetensors import safe_open
    from trellmlx.quantize import quantize_model

    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"packed artifact manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != PACKED_FLOW_ARTIFACT_SCHEMA:
        raise RuntimeError(f"unsupported packed flow artifact schema: {manifest.get('schema')}")
    stage = manifest.get("stage")
    if expected_stage is not None and stage != expected_stage:
        raise RuntimeError(f"packed flow artifact stage mismatch: expected {expected_stage}, got {stage}")
    quantized_names = manifest.get("quantized_module_names") or []
    if not quantized_names:
        raise RuntimeError("packed artifact contains no quantized module names")

    tensor_path = Path(manifest["artifact"]["tensor_file"])
    if not tensor_path.exists():
        tensor_path = artifact_dir / "model.safetensors"
    if not tensor_path.exists():
        raise FileNotFoundError(f"packed artifact tensor file does not exist: {tensor_path}")

    bits = int(manifest["quantization"]["bits"])
    group_size = int(manifest["quantization"]["group_size"])
    quantize_model(model, bits=bits, group_size=group_size)

    weights = []
    with safe_open(str(tensor_path), framework="numpy") as sf:
        keys = list(sf.keys())
        for key in keys:
            weights.append((key, mx.array(sf.get_tensor(key))))
    model.load_weights(weights, strict=False)

    loaded = {}
    for name in quantized_names:
        module = _nested_getattr(model, name)
        if not isinstance(module, nn.QuantizedLinear):
            raise RuntimeError(f"packed artifact expected QuantizedLinear module at {name}")
        dtype = _dtype_name(module.weight.dtype)
        if dtype != "uint32":
            raise RuntimeError(f"packed artifact module {name} did not load uint32 packed weight: {dtype}")
        loaded[name] = {
            "weight_shape": list(module.weight.shape),
            "weight_dtype": dtype,
            "scales_shape": list(module.scales.shape),
            "biases_shape": list(module.biases.shape),
        }

    if not loaded:
        raise RuntimeError("requested packed runtime loaded no quantized modules")

    return {
        "schema": PACKED_FLOW_ARTIFACT_SCHEMA,
        "stage": stage,
        "effective_route": f"packed-quantized-{stage}",
        "artifact": {
            "manifest": str(manifest_path),
            "tensor_file": str(tensor_path),
        },
        "quantization": {
            "bits": bits,
            "group_size": group_size,
        },
        "loaded_tensor_count": len(weights),
        "loaded_quantized_module_count": len(loaded),
        "loaded_quantized_modules": loaded,
    }


def load_packed_sparse_structure_artifact(model: Any, artifact_dir: Path) -> dict[str, Any]:
    import mlx.nn as nn
    from safetensors import safe_open
    from trellmlx.quantize import quantize_model

    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"packed artifact manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != PACKED_SPARSE_ARTIFACT_SCHEMA:
        raise RuntimeError(f"unsupported packed artifact schema: {manifest.get('schema')}")
    quantized_names = manifest.get("quantized_module_names") or []
    if not quantized_names:
        raise RuntimeError("packed artifact contains no quantized module names")

    tensor_path = Path(manifest["artifact"]["tensor_file"])
    if not tensor_path.exists():
        tensor_path = artifact_dir / "model.safetensors"
    if not tensor_path.exists():
        raise FileNotFoundError(f"packed artifact tensor file does not exist: {tensor_path}")

    bits = int(manifest["quantization"]["bits"])
    group_size = int(manifest["quantization"]["group_size"])
    quantize_model(model, bits=bits, group_size=group_size)

    weights = []
    with safe_open(str(tensor_path), framework="numpy") as sf:
        keys = list(sf.keys())
        for key in keys:
            weights.append((key, mx.array(sf.get_tensor(key))))
    model.load_weights(weights, strict=False)

    loaded = {}
    for name in quantized_names:
        module = _nested_getattr(model, name)
        if not isinstance(module, nn.QuantizedLinear):
            raise RuntimeError(f"packed artifact expected QuantizedLinear module at {name}")
        dtype = _dtype_name(module.weight.dtype)
        if dtype != "uint32":
            raise RuntimeError(f"packed artifact module {name} did not load uint32 packed weight: {dtype}")
        loaded[name] = {
            "weight_shape": list(module.weight.shape),
            "weight_dtype": dtype,
            "scales_shape": list(module.scales.shape),
            "biases_shape": list(module.biases.shape),
        }

    if not loaded:
        raise RuntimeError("requested packed runtime loaded no quantized modules")

    return {
        "schema": PACKED_SPARSE_ARTIFACT_SCHEMA,
        "effective_route": "packed-quantized-sparse-structure",
        "artifact": {
            "manifest": str(manifest_path),
            "tensor_file": str(tensor_path),
        },
        "quantization": {
            "bits": bits,
            "group_size": group_size,
        },
        "loaded_tensor_count": len(weights),
        "loaded_quantized_module_count": len(loaded),
        "loaded_quantized_modules": loaded,
    }


def _sample_module_routes(model: Any, names: list[str]) -> dict[str, Any]:
    samples = {}
    for name in names:
        module = _nested_getattr(model, name)
        samples[name] = {
            "class": module.__class__.__name__,
            "weight_shape": list(module.weight.shape),
            "weight_dtype": _dtype_name(module.weight.dtype),
        }
    return samples


def _mlx_peak_memory_bytes() -> int | None:
    if hasattr(mx, "get_peak_memory"):
        return int(mx.get_peak_memory())
    metal = getattr(mx, "metal", None)
    if metal is not None and hasattr(metal, "get_peak_memory"):
        return int(metal.get_peak_memory())
    return None


def _synthetic_sparse_context(config: dict[str, Any]) -> dict[str, mx.array]:
    token_count = config["resolution"] ** 3
    context_channels = config["cond_channels"]
    return {
        "global": mx.zeros((1, 5, context_channels), dtype=mx.float32),
        "proj": mx.zeros((1, token_count, context_channels), dtype=mx.float32),
    }


def pixal3d_context_from_features(
    features: mx.array | np.ndarray,
    config: dict[str, Any],
    *,
    image_size: int = 512,
    patch_size: int = 16,
    projection_mode: str = "native",
    hr_feature_size: int | tuple[int, int] | None = None,
    camera_angle_x: float = PIXAL3D_DEFAULT_CAMERA_ANGLE_X,
    distance: float = PIXAL3D_DEFAULT_CAMERA_DISTANCE,
    mesh_scale: float = PIXAL3D_DEFAULT_MESH_SCALE,
) -> dict[str, mx.array]:
    from trellmlx.models.dinov3_proj import DINOv3ProjectionAdapter

    feature_array = mx.array(features)
    adapter = DINOv3ProjectionAdapter(
        image_size=image_size,
        patch_size=patch_size,
        grid_resolution=config["resolution"],
        num_prefix_tokens=5,
        projection_mode=projection_mode,
        hr_feature_size=hr_feature_size,
    )
    context = adapter(
        feature_array,
        camera_angle_x=mx.array([camera_angle_x], dtype=mx.float32),
        distance=mx.array([distance], dtype=mx.float32),
        mesh_scale=mx.array([mesh_scale], dtype=mx.float32),
    )
    mx.eval(context["global"], context["proj"])
    return context


def diagnose_packed_slat_projection_width(
    artifact_dir: Path,
    config_path: Path,
    *,
    expected_stage: str = "shape-lr-slat",
    image_size: int = 512,
    patch_size: int = 16,
    projection_mode: str = "native",
    hr_feature_size: int | tuple[int, int] | None = None,
    camera_angle_x: float = PIXAL3D_DEFAULT_CAMERA_ANGLE_X,
    distance: float = PIXAL3D_DEFAULT_CAMERA_DISTANCE,
    mesh_scale: float = PIXAL3D_DEFAULT_MESH_SCALE,
    token_count: int = 8,
    seed: int = 42,
    run_zero_augmented: bool = True,
) -> dict[str, Any]:
    """Distinguish packed SLat load viability from Pixal3D projection width mismatch."""
    if token_count <= 0:
        raise ValueError("packed SLat width diagnostic requires a positive token_count")

    config = _flow_config_profile(config_path)
    patch_grid = image_size // patch_size
    feature_shape = (1, 5 + patch_grid * patch_grid, config["cond_channels"])
    features = mx.zeros(feature_shape, dtype=mx.float32)
    adapter_context = pixal3d_context_from_features(
        features,
        config,
        image_size=image_size,
        patch_size=patch_size,
        projection_mode=projection_mode,
        hr_feature_size=hr_feature_size,
        camera_angle_x=camera_angle_x,
        distance=distance,
        mesh_scale=mesh_scale,
    )
    if adapter_context["proj"].shape[1] < token_count:
        raise ValueError(
            f"packed SLat width diagnostic token_count {token_count} exceeds projected token count "
            f"{adapter_context['proj'].shape[1]}"
        )

    model = flow_model_from_config(config_path)
    load_report = load_packed_flow_artifact(model, artifact_dir, expected_stage=expected_stage)

    mx.random.seed(seed)
    x = mx.random.normal((token_count, config["in_channels"])).astype(mx.float32)
    t = mx.array([1000.0], dtype=mx.float32)
    coords = mx.array(
        [
            [
                i % config["resolution"],
                (i * 3) % config["resolution"],
                (i * 5) % config["resolution"],
            ]
            for i in range(token_count)
        ],
        dtype=mx.int32,
    )
    native_context = {
        "global": adapter_context["global"],
        "proj": adapter_context["proj"][:, :token_count, :],
    }

    report: dict[str, Any] = {
        "schema": "trellis2mlx.pixal3d_packed_slat_width_diagnostic.v1",
        "stage": expected_stage,
        "route": {
            "requested": f"packed-quantized-{expected_stage}",
            "effective": load_report["effective_route"],
            "packed_artifact": str(artifact_dir),
            "checkpoint_config": str(config_path),
            "model_class": model.__class__.__name__,
            "fp_checkpoint_loaded": False,
        },
        "config": config,
        "load": {
            "loaded_tensor_count": load_report["loaded_tensor_count"],
            "loaded_quantized_module_count": load_report["loaded_quantized_module_count"],
            "quantization": load_report["quantization"],
        },
        "conditioning": {
            "current_mlx_projection_adapter": {
                "input_feature_shape": list(feature_shape),
                "global_shape": list(adapter_context["global"].shape),
                "proj_shape": list(adapter_context["proj"].shape),
                "projection_mode": projection_mode,
                "hr_feature_size": hr_feature_size,
                "camera_angle_x": camera_angle_x,
                "distance": distance,
                "mesh_scale": mesh_scale,
                "expected_slat_proj_in_channels": config["proj_in_channels"],
                "matches_slat_projection_width": adapter_context["proj"].shape[-1] == config["proj_in_channels"],
            },
            "upstream_contract": "Pixal3D projected SLat configs may require concat(lr 1024, hr 1024) NAF-upsampled features.",
        },
        "inputs": {
            "token_count": token_count,
            "x_shape": list(x.shape),
            "coords_shape": list(coords.shape),
            "native_global_shape": list(native_context["global"].shape),
            "native_proj_shape": list(native_context["proj"].shape),
        },
        "sample_modules": _sample_module_routes(
            model,
            [
                "blocks.0.cross_attn.proj_linear",
                "blocks.0.self_attn.to_qkv",
                "out_layer",
            ],
        ),
    }

    try:
        native_out = model(x, t, native_context, coords=coords)
        mx.eval(native_out)
        report["native_projection_forward"] = {
            "status": (
                "ok"
                if native_context["proj"].shape[-1] == config["proj_in_channels"]
                else "unexpected_ok"
            ),
            "output_shape": list(native_out.shape),
            "output_mean": float(mx.mean(native_out).item()),
            "output_std": float(mx.std(native_out).item()),
        }
    except Exception as exc:
        report["native_projection_forward"] = {
            "status": (
                "expected_failure"
                if native_context["proj"].shape[-1] != config["proj_in_channels"]
                else "failure"
            ),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    if not run_zero_augmented:
        report["zero_augmented_projection_forward"] = {"status": "skipped"}
        return report

    width_delta = config["proj_in_channels"] - native_context["proj"].shape[-1]
    if width_delta < 0:
        report["zero_augmented_projection_forward"] = {
            "status": "skipped",
            "reason": "native projection width exceeds expected SLat projection width",
        }
        return report
    if width_delta == 0:
        augmented_context = native_context
        zero_augmented = False
    else:
        pad = mx.zeros((*native_context["proj"].shape[:-1], width_delta), dtype=native_context["proj"].dtype)
        augmented_context = {
            "global": native_context["global"],
            "proj": mx.concatenate([native_context["proj"], pad], axis=-1),
        }
        zero_augmented = True

    start = time.time()
    try:
        augmented_out = model(x, t, augmented_context, coords=coords)
        mx.eval(augmented_out)
        report["zero_augmented_projection_forward"] = {
            "status": "ok",
            "zero_augmented_projection_for_diagnostic_only": zero_augmented,
            "proj_shape": list(augmented_context["proj"].shape),
            "output_shape": list(augmented_out.shape),
            "output_mean": float(mx.mean(augmented_out).item()),
            "output_std": float(mx.std(augmented_out).item()),
            "elapsed_seconds": round(time.time() - start, 3),
            "peak_mlx_memory_bytes": _mlx_peak_memory_bytes(),
        }
    except Exception as exc:
        report["zero_augmented_projection_forward"] = {
            "status": "failure",
            "zero_augmented_projection_for_diagnostic_only": zero_augmented,
            "proj_shape": list(augmented_context["proj"].shape),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": round(time.time() - start, 3),
            "peak_mlx_memory_bytes": _mlx_peak_memory_bytes(),
        }

    return report


def _spatial_coords_array(coords: mx.array | np.ndarray) -> np.ndarray:
    coords_np = np.array(coords, dtype=np.int32)
    if coords_np.ndim != 2 or coords_np.shape[1] not in (3, 4):
        raise ValueError(f"coords must have shape [N,3] or [N,4], got {list(coords_np.shape)}")
    if coords_np.shape[1] == 4:
        if np.any(coords_np[:, 0] != 0):
            raise ValueError("only batch index 0 is supported for packed LR SLat stage smoke")
        coords_np = coords_np[:, 1:4]
    return coords_np


def gather_projected_context_for_coords(
    context: dict[str, mx.array],
    coords: mx.array | np.ndarray,
    *,
    grid_resolution: int,
) -> dict[str, mx.array]:
    """Gather full-grid projected context rows at sparse SLat coordinates."""
    if "global" not in context or "proj" not in context:
        raise ValueError("projected context must contain 'global' and 'proj'")
    coords_np = _spatial_coords_array(coords)
    if np.any(coords_np < 0) or np.any(coords_np >= grid_resolution):
        raise ValueError(f"coords must be within [0,{grid_resolution})")

    expected_tokens = grid_resolution ** 3
    if context["proj"].shape[1] != expected_tokens:
        raise ValueError(
            f"projected context token count {context['proj'].shape[1]} does not match "
            f"grid_resolution^3={expected_tokens}"
        )
    flat_indices = (
        coords_np[:, 0] * grid_resolution * grid_resolution
        + coords_np[:, 1] * grid_resolution
        + coords_np[:, 2]
    ).astype(np.int32)
    gathered = mx.take(context["proj"][0], mx.array(flat_indices), axis=0)[None, :, :]
    mx.eval(gathered)
    return {
        "global": context["global"],
        "proj": gathered,
    }


def run_packed_lr_slat_stage_smoke(
    artifact_dir: Path,
    config_path: Path,
    context: dict[str, mx.array],
    coords: mx.array | np.ndarray,
    *,
    steps: int = 1,
    seed: int = 42,
    conditioning_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from trellmlx.samplers import flow_euler_sample

    if steps <= 0:
        raise ValueError("packed LR SLat stage smoke requires positive steps")

    config = _flow_config_profile(config_path)
    coords_np = _spatial_coords_array(coords)
    if len(coords_np) == 0:
        raise ValueError("packed LR SLat stage smoke requires at least one coordinate")
    gathered_context = gather_projected_context_for_coords(
        context,
        coords_np,
        grid_resolution=config["resolution"],
    )

    mx.random.seed(seed)
    model = flow_model_from_config(config_path)
    load_report = load_packed_flow_artifact(model, artifact_dir, expected_stage="shape-lr-slat")
    if load_report["loaded_quantized_module_count"] <= 0:
        raise RuntimeError("requested packed LR SLat stage loaded no quantized modules")

    noise = mx.random.normal((len(coords_np), config["in_channels"])).astype(mx.float32)
    coords_mx = mx.array(coords_np, dtype=mx.int32)
    neg_context = {key: mx.zeros_like(value) for key, value in gathered_context.items()}
    mx.eval(noise, coords_mx, gathered_context["global"], gathered_context["proj"])
    out = flow_euler_sample(
        model,
        noise,
        gathered_context,
        neg_context,
        steps=steps,
        guidance_strength=1.0,
        guidance_rescale=0.0,
        verbose=False,
        coords=coords_mx,
    )
    mx.eval(out)

    sample_names = [
        "blocks.0.cross_attn.proj_linear",
        "blocks.0.self_attn.to_qkv",
        "out_layer",
    ]
    return {
        "schema": "trellis2mlx.pixal3d_packed_lr_slat_stage_smoke.v1",
        "stage": "shape-lr-slat",
        "route": {
            "requested": "packed-quantized-shape-lr-slat",
            "effective": load_report["effective_route"],
            "fp_checkpoint_loaded": False,
            "packed_artifact": str(artifact_dir),
            "checkpoint_config": str(config_path),
            "model_class": model.__class__.__name__,
        },
        "conditioning": conditioning_report or {},
        "config": config,
        "load": load_report,
        "inputs": {
            "coords_shape": list(coords_np.shape),
            "noise_shape": list(noise.shape),
            "full_context_global_shape": list(context["global"].shape),
            "full_context_proj_shape": list(context["proj"].shape),
            "context_global_shape": list(gathered_context["global"].shape),
            "context_proj_shape": list(gathered_context["proj"].shape),
        },
        "sampler": {
            "steps": steps,
            "guidance_strength": 1.0,
            "guidance_rescale": 0.0,
        },
        "sample_modules": _sample_module_routes(model, sample_names),
        "output": {
            "shape": list(out.shape),
            "dtype": _dtype_name(out.dtype),
            "mean": float(mx.mean(out).item()),
            "std": float(mx.std(out).item()),
            "abs_max": float(mx.max(mx.abs(out)).item()),
            "peak_mlx_memory_bytes": _mlx_peak_memory_bytes(),
        },
    }


def extract_pixal3d_image_features(image_path: Path, *, image_size: int = 512) -> tuple[mx.array, dict[str, Any]]:
    from trellmlx.models.dinov3 import extract_features

    features = extract_features(str(image_path), image_size=image_size)
    mx.eval(features)
    return features, {
        "source": "native_mlx_dinov3",
        "image_path": str(image_path),
        "image_size": image_size,
        "feature_shape": list(features.shape),
    }


def _run_packed_sparse_stage_with_context(
    artifact_dir: Path,
    config_path: Path,
    context: dict[str, mx.array],
    *,
    steps: int,
    seed: int,
    conditioning_report: dict[str, Any],
) -> dict[str, Any]:
    from trellmlx.samplers import flow_euler_sample

    if steps <= 0:
        raise ValueError("packed sparse stage smoke requires positive steps")

    mx.random.seed(seed)
    config = _flow_config_profile(config_path)
    model = sparse_structure_model_from_config(config_path)
    load_report = load_packed_sparse_structure_artifact(model, artifact_dir)
    if load_report["loaded_quantized_module_count"] <= 0:
        raise RuntimeError("requested packed sparse stage loaded no quantized modules")

    neg_context = {key: mx.zeros_like(value) for key, value in context.items()}
    noise = mx.random.normal(
        (
            1,
            config["in_channels"],
            config["resolution"],
            config["resolution"],
            config["resolution"],
        )
    )
    mx.eval(noise, context["global"], context["proj"])
    out = flow_euler_sample(
        model,
        noise,
        context,
        neg_context,
        steps=steps,
        guidance_strength=1.0,
        guidance_rescale=0.0,
        verbose=False,
    )
    mx.eval(out)

    sample_names = [
        "blocks.0.cross_attn.proj_linear",
        "blocks.0.self_attn.to_qkv",
        "adaLN_modulation.layers.1",
        "out_layer",
    ]
    sample_names = [name for name in sample_names if name.split(".")[0] != "blocks" or config["num_blocks"] > 0]

    return {
        "schema": "trellis2mlx.pixal3d_sparse_stage_smoke.v1",
        "stage": "sparse-structure",
        "route": {
            "requested": "packed-quantized-sparse-structure",
            "effective": "packed-quantized-sparse-structure",
            "fp_checkpoint_loaded": False,
            "packed_artifact": str(artifact_dir),
            "checkpoint_config": str(config_path),
            "model_class": model.__class__.__name__,
        },
        "conditioning": conditioning_report,
        "config": config,
        "load": load_report,
        "inputs": {
            "noise_shape": list(noise.shape),
            "context_keys": sorted(context),
            "context_global_shape": list(context["global"].shape),
            "context_proj_shape": list(context["proj"].shape),
        },
        "sampler": {
            "steps": steps,
            "guidance_strength": 1.0,
            "guidance_rescale": 0.0,
        },
        "sample_modules": _sample_module_routes(model, sample_names),
        "output": {
            "shape": list(out.shape),
            "dtype": _dtype_name(out.dtype),
            "mean": float(mx.mean(out).item()),
            "std": float(mx.std(out).item()),
            "abs_max": float(mx.max(mx.abs(out)).item()),
        },
    }


def run_packed_sparse_structure_stage_smoke(
    artifact_dir: Path,
    config_path: Path,
    *,
    steps: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    config = _flow_config_profile(config_path)
    context = _synthetic_sparse_context(config)
    return _run_packed_sparse_stage_with_context(
        artifact_dir,
        config_path,
        context,
        steps=steps,
        seed=seed,
        conditioning_report={
            "source": "synthetic",
            "synthetic_fallback_used": True,
            "image_projected_conditioning": False,
        },
    )


def run_image_conditioned_packed_sparse_structure_stage_smoke(
    artifact_dir: Path,
    config_path: Path,
    *,
    image_path: Path | None = None,
    features: mx.array | np.ndarray | None = None,
    feature_source: str | None = None,
    image_size: int = 512,
    patch_size: int = 16,
    camera_angle_x: float = PIXAL3D_DEFAULT_CAMERA_ANGLE_X,
    distance: float = PIXAL3D_DEFAULT_CAMERA_DISTANCE,
    mesh_scale: float = PIXAL3D_DEFAULT_MESH_SCALE,
    steps: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    if features is None and image_path is None:
        raise RuntimeError("image conditioning requested but no image path or feature tensor was supplied")

    config = _flow_config_profile(config_path)
    if features is None:
        assert image_path is not None
        features, feature_report = extract_pixal3d_image_features(image_path, image_size=image_size)
    else:
        features = mx.array(features)
        mx.eval(features)
        feature_report = {
            "source": feature_source or "provided_features",
            "image_path": str(image_path) if image_path is not None else None,
            "image_size": image_size,
            "feature_shape": list(features.shape),
        }

    context = pixal3d_context_from_features(
        features,
        config,
        image_size=image_size,
        patch_size=patch_size,
        camera_angle_x=camera_angle_x,
        distance=distance,
        mesh_scale=mesh_scale,
    )
    conditioning_report = {
        **feature_report,
        "synthetic_fallback_used": False,
        "image_projected_conditioning": True,
        "patch_size": patch_size,
        "camera_angle_x": camera_angle_x,
        "distance": distance,
        "mesh_scale": mesh_scale,
        "context_global_shape": list(context["global"].shape),
        "context_proj_shape": list(context["proj"].shape),
    }
    return _run_packed_sparse_stage_with_context(
        artifact_dir,
        config_path,
        context,
        steps=steps,
        seed=seed,
        conditioning_report=conditioning_report,
    )


def _require_image_projected_conditioning(conditioning_report: dict[str, Any]) -> None:
    if conditioning_report.get("synthetic_fallback_used") is True or not conditioning_report.get("image_projected_conditioning"):
        raise RuntimeError("sparse decoder boundary requires real image-projected conditioning, not synthetic fallback")


def _require_packed_sparse_route(route_report: dict[str, Any]) -> None:
    if route_report.get("effective") != "packed-quantized-sparse-structure":
        raise RuntimeError("sparse decoder boundary requires effective packed-quantized-sparse-structure route")
    if route_report.get("fp_checkpoint_loaded") is not False:
        raise RuntimeError("sparse decoder boundary refuses fp sparse checkpoint fallback")


def decode_sparse_structure_occupancy_boundary(
    sparse_sample: mx.array | np.ndarray,
    *,
    route_report: dict[str, Any],
    conditioning_report: dict[str, Any],
    decoder_checkpoint: Path | None = None,
    decoder: Any | None = None,
    threshold: float = 0.0,
) -> dict[str, Any]:
    _require_packed_sparse_route(route_report)
    _require_image_projected_conditioning(conditioning_report)

    sample = mx.array(sparse_sample)
    if len(sample.shape) != 5:
        raise ValueError(f"sparse decoder boundary expects rank-5 sparse sample, got shape {list(sample.shape)}")
    if sample.shape[1] != 8:
        raise ValueError(f"sparse decoder boundary expects 8 latent channels, got shape {list(sample.shape)}")

    checkpoint_loaded = False
    if decoder is None:
        if decoder_checkpoint is None:
            raise ValueError("decoder_checkpoint is required when decoder is not supplied")
        from trellmlx.models.sparse_structure_decoder import SparseStructureDecoder
        from trellmlx.weight_loader import load_weights

        decoder = SparseStructureDecoder()
        load_weights(decoder, str(decoder_checkpoint), verbose=False)
        checkpoint_loaded = True
    elif decoder_checkpoint is not None:
        checkpoint_loaded = True

    logits = decoder(sample.astype(mx.float32))
    mx.eval(logits)
    if len(logits.shape) != 5:
        raise RuntimeError(f"sparse decoder produced non-rank-5 logits: {list(logits.shape)}")
    if logits.shape[1] != 1:
        raise RuntimeError(f"sparse decoder expected one occupancy logit channel, got {list(logits.shape)}")

    occupancy = logits[:, 0] > threshold
    mx.eval(occupancy)
    occupied_count = int(mx.sum(occupancy).item())
    total_count = int(np.prod(occupancy.shape))

    return {
        "schema": "trellis2mlx.pixal3d_sparse_decoder_boundary.v1",
        "stage": "sparse-structure-decoder",
        "route": {
            "effective": route_report.get("effective"),
            "fp_checkpoint_loaded": route_report.get("fp_checkpoint_loaded"),
            "packed_artifact": route_report.get("packed_artifact"),
        },
        "conditioning": {
            "source": conditioning_report.get("source"),
            "synthetic_fallback_used": conditioning_report.get("synthetic_fallback_used"),
            "image_projected_conditioning": conditioning_report.get("image_projected_conditioning"),
            "image_path": conditioning_report.get("image_path"),
            "feature_shape": conditioning_report.get("feature_shape"),
            "context_global_shape": conditioning_report.get("context_global_shape"),
            "context_proj_shape": conditioning_report.get("context_proj_shape"),
        },
        "decoder": {
            "class": decoder.__class__.__name__,
            "checkpoint": str(decoder_checkpoint) if decoder_checkpoint is not None else None,
            "checkpoint_loaded": checkpoint_loaded,
        },
        "input": {
            "shape": list(sample.shape),
            "dtype": _dtype_name(sample.dtype),
        },
        "logits": {
            "shape": list(logits.shape),
            "dtype": _dtype_name(logits.dtype),
            "min": float(mx.min(logits).item()),
            "max": float(mx.max(logits).item()),
            "mean": float(mx.mean(logits).item()),
        },
        "occupancy": {
            "shape": list(occupancy.shape),
            "threshold": float(threshold),
            "occupied_count": occupied_count,
            "total_count": total_count,
            "occupied_ratio": occupied_count / total_count if total_count else None,
        },
        "output": {
            "mesh_or_glb_exported": False,
        },
    }


def export_occupancy_voxel_glb_boundary(
    occupancy: mx.array | np.ndarray,
    output_path: Path,
    *,
    route_report: dict[str, Any],
    decoder_report: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    _require_packed_sparse_route(route_report)
    if decoder_report.get("stage") != "sparse-structure-decoder":
        raise RuntimeError("occupancy GLB export requires sparse-structure-decoder report")

    occ = np.asarray(occupancy)
    if occ.ndim == 4 and occ.shape[0] == 1:
        occ_grid = occ[0]
    elif occ.ndim == 3:
        occ_grid = occ
    else:
        raise ValueError(f"occupancy GLB export expects [1,D,H,W] or [D,H,W], got {list(occ.shape)}")
    occ_grid = occ_grid.astype(bool)
    occupied_count = int(occ_grid.sum())
    if occupied_count <= 0:
        raise RuntimeError("occupancy GLB export refuses empty occupancy")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}")

    import trimesh

    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolution = int(occ_grid.shape[0])
    voxel_grid = trimesh.voxel.VoxelGrid(
        trimesh.voxel.encoding.DenseEncoding(occ_grid)
    )
    mesh = voxel_grid.marching_cubes
    voxel_size = 1.0 / resolution
    mesh.apply_scale(voxel_size)
    mesh.apply_translation([-0.5, -0.5, -0.5])
    mesh.export(output_path)

    loaded = trimesh.load(output_path, force="mesh")
    bounds = loaded.bounds.tolist() if loaded.bounds is not None else None
    size_bytes = output_path.stat().st_size if output_path.exists() else 0
    if len(loaded.vertices) <= 0 or len(loaded.faces) <= 0:
        raise RuntimeError("occupancy GLB export wrote empty mesh geometry")

    return {
        "schema": "trellis2mlx.pixal3d_occupancy_voxel_glb_boundary.v1",
        "stage": "occupancy-voxel-glb-export",
        "route": {
            "effective": route_report.get("effective"),
            "fp_checkpoint_loaded": route_report.get("fp_checkpoint_loaded"),
            "packed_artifact": route_report.get("packed_artifact"),
        },
        "decoder": {
            "stage": decoder_report.get("stage"),
            "checkpoint": (decoder_report.get("decoder") or {}).get("checkpoint"),
            "logits_shape": (decoder_report.get("logits") or {}).get("shape"),
        },
        "occupancy": {
            "input_shape": list(occ.shape),
            "grid_shape": list(occ_grid.shape),
            "occupied_count": occupied_count,
            "total_count": int(occ_grid.size),
            "occupied_ratio": occupied_count / int(occ_grid.size),
        },
        "mesh": {
            "vertices": int(len(loaded.vertices)),
            "faces": int(len(loaded.faces)),
            "bounds": bounds,
        },
        "output": {
            "path": str(output_path),
            "exists": output_path.exists(),
            "size_bytes": int(size_bytes),
            "visually_inspected": False,
        },
    }


def validate_checkpoint_inventory(inventory: dict[str, Any]) -> None:
    if inventory.get("path") is None:
        return
    if not inventory.get("exists"):
        raise RuntimeError(f"checkpoint does not exist: {inventory.get('path')}")
    if not inventory.get("projection_keys_present"):
        raise RuntimeError(
            f"checkpoint lacks Pixal3D projection keys: {inventory.get('path')}"
        )


def build_synthetic_context(args: argparse.Namespace) -> dict[str, mx.array]:
    from trellmlx.models.dinov3_proj import DINOv3ProjectionAdapter

    patch_grid = args.image_size // args.patch_size
    if patch_grid * args.patch_size != args.image_size:
        raise ValueError("image-size must be divisible by patch-size")

    prefix = mx.zeros((1, 5, args.context_channels), dtype=mx.float32)
    patches = mx.ones((1, patch_grid * patch_grid, args.context_channels), dtype=mx.float32)
    features = mx.concatenate([prefix, patches], axis=1)

    adapter = DINOv3ProjectionAdapter(
        image_size=args.image_size,
        patch_size=args.patch_size,
        grid_resolution=args.grid_resolution,
        num_prefix_tokens=5,
    )
    return adapter(
        features,
        camera_angle_x=mx.array([getattr(args, "camera_angle_x", PIXAL3D_DEFAULT_CAMERA_ANGLE_X)], dtype=mx.float32),
        distance=mx.array([getattr(args, "camera_distance", PIXAL3D_DEFAULT_CAMERA_DISTANCE)], dtype=mx.float32),
        mesh_scale=mx.array([getattr(args, "mesh_scale", PIXAL3D_DEFAULT_MESH_SCALE)], dtype=mx.float32),
    )


def build_smoke_models(args: argparse.Namespace) -> dict[str, Any]:
    from trellmlx.models.pixal3d_flow import Pixal3DSLatFlowModel, Pixal3DSparseStructureFlowModel

    return {
        "ss_flow": Pixal3DSparseStructureFlowModel(
            in_channels=8,
            out_channels=8,
            model_channels=64,
            num_heads=4,
            num_blocks=1,
            mlp_hidden=128,
            context_channels=args.context_channels,
            proj_in_channels=args.context_channels,
            resolution=args.grid_resolution,
        ),
        "slat_flow": Pixal3DSLatFlowModel(
            in_channels=32,
            out_channels=32,
            model_channels=64,
            num_heads=4,
            num_blocks=1,
            mlp_hidden=128,
            context_channels=args.context_channels,
            proj_in_channels=args.context_channels,
        ),
    }


def _tiny_sparse_structure_model(*, grid_resolution: int, context_channels: int):
    from trellmlx.models.pixal3d_flow import Pixal3DSparseStructureFlowModel

    return Pixal3DSparseStructureFlowModel(
        in_channels=8,
        out_channels=8,
        model_channels=64,
        num_heads=4,
        num_blocks=1,
        mlp_hidden=128,
        context_channels=context_channels,
        proj_in_channels=context_channels,
        resolution=grid_resolution,
    )


def _nested_getattr(root: Any, path: str) -> Any:
    value = root
    for part in path.split("."):
        if part.isdigit() and isinstance(value, list):
            value = value[int(part)]
        else:
            value = getattr(value, part)
    return value


def _tensor_for_key(model: Any, key: str) -> mx.array:
    module_path, attr = key.rsplit(".", 1)
    return getattr(_nested_getattr(model, module_path), attr)


def _quantized_module_report(model: Any) -> dict[str, Any]:
    import mlx.nn as nn

    quantized = {}
    packed_dtypes = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            quantized[name] = {
                "bits": module.bits,
                "group_size": module.group_size,
                "weight_shape": list(module.weight.shape),
                "scale_shape": list(module.scales.shape),
                "biases_shape": list(module.biases.shape),
            }
            packed_dtypes[name] = str(module.weight.dtype).replace("mlx.core.", "")
    return {
        "quantized_module_count": len(quantized),
        "quantized_module_names": sorted(quantized),
        "quantized_modules": quantized,
        "packed_weight_dtypes": packed_dtypes,
    }


def quantized_sparse_structure_assignment_smoke(
    checkpoint: Path,
    *,
    bits: int = 4,
    group_size: int = 64,
    grid_resolution: int = 2,
    context_channels: int = 8,
    sentinel_key: str | None = None,
    sentinel_expected: np.ndarray | None = None,
) -> dict[str, Any]:
    from safetensors import safe_open
    from trellmlx.quantize import quantize_model
    from trellmlx.weight_loader import load_weights

    if bits not in (4, 8):
        raise ValueError("quantized sparse-structure smoke requires bits=4 or bits=8")

    model = _tiny_sparse_structure_model(
        grid_resolution=grid_resolution,
        context_channels=context_channels,
    )
    inventory = inspect_checkpoint(checkpoint, {"ss_flow": model})
    validate_checkpoint_inventory(inventory)

    with safe_open(str(checkpoint), framework="numpy") as sf:
        checkpoint_keys = set(sf.keys())
    if sentinel_key is not None and sentinel_key not in checkpoint_keys:
        raise RuntimeError(f"sentinel key missing from checkpoint: {sentinel_key}")

    load_weights(model, str(checkpoint), verbose=False, strict=False)

    sentinel_assigned = None
    sentinel_max_abs_diff = None
    if sentinel_key is not None and sentinel_expected is not None:
        loaded = np.array(_tensor_for_key(model, sentinel_key))
        sentinel_max_abs_diff = float(np.max(np.abs(loaded - sentinel_expected)))
        sentinel_assigned = sentinel_max_abs_diff < 1e-2
        if not sentinel_assigned:
            raise RuntimeError(
                f"sentinel assignment mismatch for {sentinel_key}: max_abs_diff={sentinel_max_abs_diff}"
            )

    quantize_model(model, bits=bits, group_size=group_size)
    quant_report = _quantized_module_report(model)
    if quant_report["quantized_module_count"] == 0:
        raise RuntimeError("quantization requested but no QuantizedLinear modules were installed")

    context_args = argparse.Namespace(
        image_size=32,
        patch_size=16,
        grid_resolution=grid_resolution,
        context_channels=context_channels,
    )
    context = build_synthetic_context(context_args)
    x_ss = mx.random.normal((1, 8, grid_resolution, grid_resolution, grid_resolution))
    out = model(x_ss, mx.array([500.0]), context)
    mx.eval(out)

    return {
        "stage": "sparse-structure",
        "checkpoint": inventory,
        "assignment": {
            "mode": "shape-matched-load",
            "matched_by_shape": inventory.get("matched_by_shape"),
            "sentinel_key": sentinel_key,
            "sentinel_assigned_before_quantize": sentinel_assigned,
            "sentinel_max_abs_diff": sentinel_max_abs_diff,
        },
        "quantization": {
            "requested_bits": bits,
            "effective_bits": bits,
            "group_size": group_size,
            "order": "load_fp_then_quantize_in_memory",
            **quant_report,
        },
        "smoke": {
            "ss_flow_output_shape": list(out.shape),
            "ss_flow_output_std": float(mx.std(out).item()),
        },
    }


def validate_effective_route(route: dict[str, Any]) -> None:
    errors = []
    if route.get("requested") == "pixal3d-proj" and route.get("effective") != "pixal3d-proj":
        errors.append("effective route is not pixal3d-proj")
    if "proj" not in route.get("context_keys", []):
        errors.append("projected conditioning key 'proj' is missing")
    if route.get("projected_shape") is None:
        errors.append("projected conditioning shape is missing")
    for role, class_name in route.get("model_classes", {}).items():
        if not class_name.startswith("Pixal3D"):
            errors.append(f"{role} model is not a Pixal3D model: {class_name}")
    if errors:
        route["fallback_detected"] = True
        raise RuntimeError("Pixal3D projected conditioning route invalid: " + "; ".join(errors))
    route["fallback_detected"] = False
    route["pixal3d_projected_conditioning"] = True


def _checkpoint_family(path: Path | None) -> str | None:
    if path is None:
        return None
    normalized = str(path)
    if "TencentARC" in normalized or "Pixal3D" in normalized:
        return "TencentARC/Pixal3D"
    if "microsoft" in normalized or "TRELLIS" in normalized:
        return "microsoft/TRELLIS"
    return "unknown"


def _projection_width(route: dict[str, Any]) -> int | None:
    projected_shape = route.get("projected_shape")
    if not projected_shape:
        return None
    return int(projected_shape[-1])


def _require_pixal3d_projected_route(route: dict[str, Any], *, phase: str) -> None:
    errors = []
    if route.get("effective") != "pixal3d-proj":
        errors.append(f"effective route is {route.get('effective')!r}")
    if route.get("fallback_detected") is True:
        errors.append("fallback_detected is true")
    if "proj" not in route.get("context_keys", []):
        errors.append("projected conditioning key 'proj' is missing")
    if route.get("projected_shape") is None:
        errors.append("projected conditioning shape is missing")
    for role, class_name in route.get("model_classes", {}).items():
        if not str(class_name).startswith("Pixal3D"):
            errors.append(f"{role} model is not Pixal3D: {class_name}")
    if errors:
        raise RuntimeError(f"{phase} is not Pixal3D projected: " + "; ".join(errors))


def run_component_filter_artifact_smoke(
    artifact_path: Path,
    route: dict[str, Any],
    *,
    mode: str,
    min_component_ratio: float,
    slat_key: str = "hr_slat",
    coords_key: str = "hr_coords_quantized_1024",
    spatial_coords_key: str = "hr_coords_3d_1024",
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    _require_pixal3d_projected_route(route, phase="component filter artifact smoke")
    if mode not in {"none", "largest", "min_ratio"}:
        raise ValueError(f"unknown component filter mode: {mode!r}")
    if min_component_ratio < 0:
        raise ValueError("component filter min ratio must be non-negative")
    if not artifact_path.exists():
        raise FileNotFoundError(f"component filter artifact does not exist: {artifact_path}")

    from trellmlx.coord_components import filter_sparse_coordinate_components

    with np.load(artifact_path) as data:
        missing = [
            key
            for key in (slat_key, coords_key, spatial_coords_key)
            if key not in data.files
        ]
        if missing:
            raise KeyError(f"component filter artifact missing keys: {missing}")
        slat = np.asarray(data[slat_key])
        quant_coords = np.asarray(data[coords_key])
        spatial_coords = np.asarray(data[spatial_coords_key])
        keys = list(data.files)

    if slat.ndim != 2:
        raise ValueError(f"{slat_key} must be rank 2, got {list(slat.shape)}")
    if quant_coords.ndim != 2 or quant_coords.shape[1] != 4:
        raise ValueError(f"{coords_key} must have shape [N,4], got {list(quant_coords.shape)}")
    if spatial_coords.ndim != 2 or spatial_coords.shape[1] != 3:
        raise ValueError(f"{spatial_coords_key} must have shape [N,3], got {list(spatial_coords.shape)}")
    row_count = slat.shape[0]
    if quant_coords.shape[0] != row_count or spatial_coords.shape[0] != row_count:
        raise ValueError(
            "component filter artifact row mismatch: "
            f"{slat_key}={slat.shape[0]}, {coords_key}={quant_coords.shape[0]}, "
            f"{spatial_coords_key}={spatial_coords.shape[0]}"
        )

    filtered_spatial, filtered_slat, component_report = filter_sparse_coordinate_components(
        spatial_coords,
        slat,
        mode=mode,
        min_component_ratio=min_component_ratio,
        include_row_indices=True,
    )
    kept_row_indices = np.asarray(component_report.pop("kept_row_indices"), dtype=np.int64)
    filtered_quant_coords = quant_coords[kept_row_indices]
    return {
        "schema": "trellis2mlx.pixal3d_component_filter_artifact_smoke.v1",
        "route": "pixal3d-hr-support-component-filter-artifact-smoke",
        "status": "ok",
        "effective_generation_route": route.get("effective"),
        "not_vanilla_trellis": True,
        "fallback_detected": route.get("fallback_detected"),
        "context_keys": route.get("context_keys"),
        "projected_shape": route.get("projected_shape"),
        "projection_width": _projection_width(route),
        "model_classes": route.get("model_classes"),
        "checkpoint_family": _checkpoint_family(checkpoint),
        "slat_normalization": "normalized",
        "input": {
            "path": artifact_path,
            "keys": keys,
            "slat_key": slat_key,
            "slat_shape": list(slat.shape),
            "coords_key": coords_key,
            "coords_shape": list(quant_coords.shape),
            "spatial_coords_key": spatial_coords_key,
            "spatial_coords_shape": list(spatial_coords.shape),
        },
        "component_filter": component_report,
        "filtered_keys": [slat_key, coords_key, spatial_coords_key],
        "filtered_shapes": {
            slat_key: list(filtered_slat.shape),
            coords_key: list(filtered_quant_coords.shape),
            spatial_coords_key: list(filtered_spatial.shape),
        },
    }


def mark_failure(
    report: dict[str, Any],
    report_path: Path,
    *,
    phase: str,
    error: str,
    extra: dict[str, Any] | None = None,
) -> None:
    report["status"] = "failed"
    report["phase"] = phase
    report["error"] = error
    report["last_trustworthy_evidence"] = {
        "phase": phase,
        "report_existed_before_write": report_path.exists(),
    }
    if extra:
        report.update(extra)
    write_report(report_path, report)


def run_smoke_route(args: argparse.Namespace, command_line: list[str] | None = None) -> dict[str, Any]:
    ensure_report_writable(args.report, overwrite=args.overwrite_report)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command_line": command_line or sys.argv,
        "repo": _repo_identity(args.repo_root),
        "host": _host_identity(),
        "settings": {
            "seed": args.seed,
            "grid_resolution": args.grid_resolution,
            "image_size": args.image_size,
            "patch_size": args.patch_size,
            "context_channels": args.context_channels,
            "checkpoint_config": str(args.checkpoint_config) if args.checkpoint_config else None,
            "chunked_quantize_sparse_structure": args.chunked_quantize_sparse_structure,
            "quantize_bits": args.quantize_bits,
            "quantize_group_size": args.quantize_group_size,
            "export_packed_sparse_structure": str(args.export_packed_sparse_structure) if args.export_packed_sparse_structure else None,
            "packed_sparse_artifact": str(args.packed_sparse_artifact) if args.packed_sparse_artifact else None,
            "run_packed_sparse_stage": args.run_packed_sparse_stage,
            "run_image_packed_sparse_stage": args.run_image_packed_sparse_stage,
            "packed_flow_artifact": str(args.packed_flow_artifact) if args.packed_flow_artifact else None,
            "packed_flow_stage": args.packed_flow_stage,
            "run_packed_slat_width_diagnostic": args.run_packed_slat_width_diagnostic,
            "component_filter_artifact": str(args.component_filter_artifact) if args.component_filter_artifact else None,
            "component_filter": args.component_filter,
            "component_filter_min_ratio": args.component_filter_min_ratio,
            "component_filter_slat_key": args.component_filter_slat_key,
            "component_filter_coords_key": args.component_filter_coords_key,
            "component_filter_spatial_coords_key": args.component_filter_spatial_coords_key,
            "projection_mode": args.projection_mode,
            "hr_feature_size": args.hr_feature_size,
            "camera_angle_x": args.camera_angle_x,
            "camera_distance": args.camera_distance,
            "mesh_scale": args.mesh_scale,
            "image": str(args.image) if args.image else None,
            "sparse_stage_steps": args.sparse_stage_steps,
        },
        "last_trustworthy_evidence": {"phase": "initialized"},
    }
    write_report(args.report, report)

    try:
        mx.random.seed(args.seed)
        context = build_synthetic_context(args)
        models = build_smoke_models(args)
        route = {
            "requested": "pixal3d-proj",
            "effective": "pixal3d-proj",
            "context_keys": sorted(context.keys()),
            "projected_shape": list(context["proj"].shape) if "proj" in context else None,
            "global_shape": list(context["global"].shape) if "global" in context else None,
            "model_classes": {name: model.__class__.__name__ for name, model in models.items()},
        }
        validate_effective_route(route)
        checkpoint_inventory = inspect_checkpoint(args.checkpoint, models)
        try:
            validate_checkpoint_inventory(checkpoint_inventory)
        except Exception as exc:
            report["checkpoint"] = checkpoint_inventory
            mark_failure(report, args.report, phase="checkpoint_inventory", error=str(exc))
            raise

        x_ss = mx.random.normal((1, 8, args.grid_resolution, args.grid_resolution, args.grid_resolution))
        t = mx.array([500.0])
        ss_out = models["ss_flow"](x_ss, t, context)

        token_count = args.grid_resolution ** 3
        x_slat = mx.random.normal((token_count, 32))
        coords = mx.array(np.indices((args.grid_resolution, args.grid_resolution, args.grid_resolution)).reshape(3, -1).T)
        slat_out = models["slat_flow"](x_slat, t, context, coords=coords)
        mx.eval(ss_out, slat_out)

        report["checkpoint"] = checkpoint_inventory
        if args.checkpoint and args.checkpoint_config:
            report["architecture_profile"] = profile_checkpoint_architecture(
                args.checkpoint, args.checkpoint_config
            )
        if args.component_filter_artifact:
            try:
                report["component_filter_artifact_smoke"] = run_component_filter_artifact_smoke(
                    args.component_filter_artifact,
                    route,
                    mode=args.component_filter,
                    min_component_ratio=args.component_filter_min_ratio,
                    slat_key=args.component_filter_slat_key,
                    coords_key=args.component_filter_coords_key,
                    spatial_coords_key=args.component_filter_spatial_coords_key,
                    checkpoint=args.checkpoint,
                )
                report["last_trustworthy_evidence"] = {"phase": "component_filter_artifact_smoke"}
                write_report(args.report, report)
            except Exception as exc:
                mark_failure(report, args.report, phase="component_filter_artifact_smoke", error=str(exc))
                raise
        if args.chunked_quantize_sparse_structure:
            if not args.checkpoint or not args.checkpoint_config:
                exc = ValueError("--chunked-quantize-sparse-structure requires --checkpoint and --checkpoint-config")
                mark_failure(report, args.report, phase="chunked_quantization", error=str(exc))
                raise exc
            try:
                report["chunked_quantization"] = chunked_quantize_sparse_structure_checkpoint(
                    args.checkpoint,
                    args.checkpoint_config,
                    bits=args.quantize_bits,
                    group_size=args.quantize_group_size,
                )
                report["last_trustworthy_evidence"] = {"phase": "chunked_quantization"}
                write_report(args.report, report)
            except Exception as exc:
                mark_failure(report, args.report, phase="chunked_quantization", error=str(exc))
                raise
        if args.export_packed_sparse_structure:
            if not args.checkpoint or not args.checkpoint_config:
                exc = ValueError("--export-packed-sparse-structure requires --checkpoint and --checkpoint-config")
                mark_failure(report, args.report, phase="packed_artifact_export", error=str(exc))
                raise exc
            try:
                report["packed_artifact_export"] = export_packed_sparse_structure_artifact(
                    args.checkpoint,
                    args.checkpoint_config,
                    args.export_packed_sparse_structure,
                    bits=args.quantize_bits,
                    group_size=args.quantize_group_size,
                    overwrite=args.overwrite_artifact,
                )
                report["last_trustworthy_evidence"] = {"phase": "packed_artifact_export"}
                write_report(args.report, report)
            except Exception as exc:
                mark_failure(report, args.report, phase="packed_artifact_export", error=str(exc))
                raise
        if args.run_packed_sparse_stage:
            if not args.packed_sparse_artifact or not args.checkpoint_config:
                exc = ValueError("--run-packed-sparse-stage requires --packed-sparse-artifact and --checkpoint-config")
                mark_failure(report, args.report, phase="packed_sparse_stage", error=str(exc))
                raise exc
            try:
                report["packed_sparse_stage"] = run_packed_sparse_structure_stage_smoke(
                    args.packed_sparse_artifact,
                    args.checkpoint_config,
                    steps=args.sparse_stage_steps,
                    seed=args.seed,
                )
                report["last_trustworthy_evidence"] = {"phase": "packed_sparse_stage"}
                write_report(args.report, report)
            except Exception as exc:
                mark_failure(report, args.report, phase="packed_sparse_stage", error=str(exc))
                raise
        if args.run_image_packed_sparse_stage:
            if not args.packed_sparse_artifact or not args.checkpoint_config or not args.image:
                exc = ValueError("--run-image-packed-sparse-stage requires --packed-sparse-artifact, --checkpoint-config, and --image")
                mark_failure(report, args.report, phase="image_packed_sparse_stage", error=str(exc))
                raise exc
            try:
                report["image_packed_sparse_stage"] = run_image_conditioned_packed_sparse_structure_stage_smoke(
                    args.packed_sparse_artifact,
                    args.checkpoint_config,
                    image_path=args.image,
                    image_size=args.image_size,
                    patch_size=args.patch_size,
                    camera_angle_x=args.camera_angle_x,
                    distance=args.camera_distance,
                    mesh_scale=args.mesh_scale,
                    steps=args.sparse_stage_steps,
                    seed=args.seed,
                )
                report["last_trustworthy_evidence"] = {"phase": "image_packed_sparse_stage"}
                write_report(args.report, report)
            except Exception as exc:
                mark_failure(report, args.report, phase="image_packed_sparse_stage", error=str(exc))
                raise
        if args.run_packed_slat_width_diagnostic:
            if not args.packed_flow_artifact or not args.checkpoint_config:
                exc = ValueError("--run-packed-slat-width-diagnostic requires --packed-flow-artifact and --checkpoint-config")
                mark_failure(report, args.report, phase="packed_slat_width_diagnostic", error=str(exc))
                raise exc
            try:
                report["packed_slat_width_diagnostic"] = diagnose_packed_slat_projection_width(
                    args.packed_flow_artifact,
                    args.checkpoint_config,
                    expected_stage=args.packed_flow_stage,
                    image_size=args.image_size,
                    patch_size=args.patch_size,
                    projection_mode=args.projection_mode,
                    hr_feature_size=args.hr_feature_size,
                    camera_angle_x=args.camera_angle_x,
                    distance=args.camera_distance,
                    mesh_scale=args.mesh_scale,
                    seed=args.seed,
                    run_zero_augmented=True,
                )
                report["last_trustworthy_evidence"] = {"phase": "packed_slat_width_diagnostic"}
                write_report(args.report, report)
            except Exception as exc:
                mark_failure(report, args.report, phase="packed_slat_width_diagnostic", error=str(exc))
                raise
        report["route"] = route
        report["smoke"] = {
            "ss_flow_output_shape": list(ss_out.shape),
            "slat_flow_output_shape": list(slat_out.shape),
            "ss_flow_output_std": float(mx.std(ss_out).item()),
            "slat_flow_output_std": float(mx.std(slat_out).item()),
        }
        report["status"] = "ok"
        if args.run_image_packed_sparse_stage:
            final_phase = "image_packed_sparse_stage"
        elif args.run_packed_slat_width_diagnostic:
            final_phase = "packed_slat_width_diagnostic"
        elif args.run_packed_sparse_stage:
            final_phase = "packed_sparse_stage"
        elif args.export_packed_sparse_structure:
            final_phase = "packed_artifact_export"
        elif args.chunked_quantize_sparse_structure:
            final_phase = "chunked_quantization"
        elif args.component_filter_artifact:
            final_phase = "component_filter_artifact_smoke"
        else:
            final_phase = "smoke_route"
        report["last_trustworthy_evidence"] = {"phase": final_phase}
        write_report(args.report, report)
        return report
    except Exception as exc:
        if report.get("status") == "failed":
            raise
        mark_failure(report, args.report, phase="smoke_route", error=str(exc))
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.smoke_route:
        raise SystemExit("Only --smoke-route is implemented in this checkpoint")
    run_smoke_route(args, command_line=["generate_pixal3d.py", *(argv or sys.argv[1:])])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
