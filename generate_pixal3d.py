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


def _synthetic_sparse_context(config: dict[str, Any]) -> dict[str, mx.array]:
    token_count = config["resolution"] ** 3
    context_channels = config["cond_channels"]
    return {
        "global": mx.zeros((1, 5, context_channels), dtype=mx.float32),
        "proj": mx.zeros((1, token_count, context_channels), dtype=mx.float32),
    }


def run_packed_sparse_structure_stage_smoke(
    artifact_dir: Path,
    config_path: Path,
    *,
    steps: int = 1,
    seed: int = 42,
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

    context = _synthetic_sparse_context(config)
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
        camera_angle_x=mx.array([np.pi / 2], dtype=mx.float32),
        distance=mx.array([2.0], dtype=mx.float32),
        mesh_scale=mx.array([1.0], dtype=mx.float32),
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
        report["route"] = route
        report["smoke"] = {
            "ss_flow_output_shape": list(ss_out.shape),
            "slat_flow_output_shape": list(slat_out.shape),
            "ss_flow_output_std": float(mx.std(ss_out).item()),
            "slat_flow_output_std": float(mx.std(slat_out).item()),
        }
        report["status"] = "ok"
        if args.run_packed_sparse_stage:
            final_phase = "packed_sparse_stage"
        elif args.export_packed_sparse_structure:
            final_phase = "packed_artifact_export"
        elif args.chunked_quantize_sparse_structure:
            final_phase = "chunked_quantization"
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
