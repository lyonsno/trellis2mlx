#!/usr/bin/env python3
"""Probe CUDA FP16 GEMM policy at the first shape-decoder convolution fork."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import traceback
from typing import Any
import zipfile

import numpy as np


SCHEMA = "trellis2mlx.cuda_decoder_block0_gemm_witness.v1"
EXPECTED_TORCH = "2.10.0+cu128"
EXPECTED_DEVICE = "Tesla T4"
VARIANT_NAMES = (
    "cuda_default_full",
    "cuda_default_m1",
    "cuda_no_reduced_full",
    "cuda_fp32_full",
)
ROUTE_SCHEMA = "trellis2mlx.decoder_block0_gemm_input.v1"
ROUTE_OPERATION = "shape_decoder.level0.block0.conv.center_gemm"
SOURCE_ROUTE = "official-source-cuda-shape-decoder-level0-trace"
LOCAL_ROUTE = "mlx-shape-decoder-level0-trace-fp16"
CENTER_KERNEL_INDEX = 13


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: str) -> bool:
    return (
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _scalar_text(value: np.ndarray, *, name: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{name} must be a scalar string")
    result = str(array.item())
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _require_array(
    arrays: dict[str, np.ndarray],
    name: str,
    *,
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> np.ndarray:
    if name not in arrays:
        raise ValueError(f"witness missing required array {name!r}")
    value = np.asarray(arrays[name])
    if value.dtype != np.dtype(dtype):
        raise ValueError(
            f"{name} must have dtype {np.dtype(dtype)}, got {value.dtype}"
        )
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(value)


def _require_route_value(
    payload: dict[str, Any],
    name: str,
    expected: Any,
    *,
    parent: str = "route_identity_json",
) -> None:
    actual = payload.get(name)
    if actual != expected:
        raise ValueError(
            f"{parent}.{name} must be {expected!r}, got {actual!r}"
        )


def _validate_route_identity(
    route_identity: Any,
    *,
    expected_rows: int,
    channels: int,
) -> dict[str, Any]:
    if not isinstance(route_identity, dict) or not route_identity:
        raise ValueError("route_identity_json must decode to a nonempty object")
    exact_values = {
        "schema": ROUTE_SCHEMA,
        "operation": ROUTE_OPERATION,
        "source_control": "alpha-1_beta-1",
        "kernel_index": CENTER_KERNEL_INDEX,
        "logical_matrix_shape": [expected_rows, channels, channels],
    }
    for name, expected in exact_values.items():
        _require_route_value(route_identity, name, expected)
    for name in (
        "source_trace_sha256",
        "local_fp16_trace_sha256",
        "checkpoint_sha256",
        "comparison_sha256",
        "input_tensor_sha256",
    ):
        value = route_identity.get(name)
        if not isinstance(value, str) or not _canonical_sha256(value):
            raise ValueError(
                f"route_identity_json.{name} must be canonical lowercase sha256"
            )

    source_route = route_identity.get("source_effective_route")
    if not isinstance(source_route, dict):
        raise ValueError(
            "route_identity_json.source_effective_route must be an object"
        )
    for name, expected in {
        "route": SOURCE_ROUTE,
        "device_type": "cuda",
        "cuda_device": EXPECTED_DEVICE,
        "sparse_conv_backend": "none",
        "decoder_level0_trace": True,
        "mesh_conversion": False,
        "raw_meshes": False,
    }.items():
        _require_route_value(
            source_route,
            name,
            expected,
            parent="route_identity_json.source_effective_route",
        )

    local_route = route_identity.get("local_effective_route")
    if not isinstance(local_route, dict):
        raise ValueError(
            "route_identity_json.local_effective_route must be an object"
        )
    for name, expected in {
        "route": LOCAL_ROUTE,
        "device_type": "metal",
        "decoder_precision": "fp16",
    }.items():
        _require_route_value(
            local_route,
            name,
            expected,
            parent="route_identity_json.local_effective_route",
        )
    return route_identity


def validate_witness_arrays(
    arrays: dict[str, np.ndarray],
    *,
    expected_rows: int = 7697,
    channels: int = 1024,
    expected_row: int = 7693,
) -> dict[str, Any]:
    coords = _require_array(
        arrays,
        "coords",
        dtype=np.int32,
        shape=(expected_rows, 4),
    )
    torso_input = _require_array(
        arrays,
        "torso_input",
        dtype=np.float16,
        shape=(expected_rows, channels),
    )
    center_weight = _require_array(
        arrays,
        "center_weight",
        dtype=np.float16,
        shape=(channels, channels),
    )
    bias = _require_array(
        arrays,
        "bias",
        dtype=np.float16,
        shape=(channels,),
    )
    source_trace_row = _require_array(
        arrays,
        "source_trace_row",
        dtype=np.float16,
        shape=(channels,),
    )
    local_trace_row = _require_array(
        arrays,
        "local_trace_row",
        dtype=np.float16,
        shape=(channels,),
    )
    row_index = np.asarray(arrays.get("row_index"))
    if (
        row_index.dtype != np.int32
        or row_index.shape != ()
        or int(row_index) != expected_row
    ):
        raise ValueError(
            f"row_index must be scalar int32 {expected_row}, got {row_index!r}"
        )
    route_identity = _validate_route_identity(
        json.loads(
            _scalar_text(
                arrays.get("route_identity_json"),
                name="route_identity_json",
            )
        ),
        expected_rows=expected_rows,
        channels=channels,
    )

    coordinate_lookup = {tuple(row) for row in coords.tolist()}
    if len(coordinate_lookup) != expected_rows:
        raise ValueError("coords contains duplicate coordinates")
    batch, z, y, x = (int(value) for value in coords[expected_row])
    neighbor_count = sum(
        (batch, z + dz, y + dy, x + dx) in coordinate_lookup
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
    )
    if neighbor_count != 1:
        raise ValueError(
            "selected target must have exactly one active neighbor, "
            f"got {neighbor_count}"
        )
    return {
        "coords": coords,
        "torso_input": torso_input,
        "center_weight": center_weight,
        "bias": bias,
        "source_trace_row": source_trace_row,
        "local_trace_row": local_trace_row,
        "row_index": expected_row,
        "route_identity": route_identity,
        "neighbor_count": neighbor_count,
    }


def _metric(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    candidate = np.asarray(candidate)
    reference = np.asarray(reference)
    if candidate.shape != reference.shape:
        raise ValueError(
            f"metric shape mismatch: {candidate.shape} versus {reference.shape}"
        )
    difference = np.abs(
        candidate.astype(np.float64) - reference.astype(np.float64)
    )
    return {
        "exact": bool(np.array_equal(candidate, reference)),
        "nonzero": int(np.count_nonzero(candidate != reference)),
        "mean_abs": float(difference.mean()) if difference.size else 0.0,
        "max_abs": float(difference.max(initial=0.0)),
    }


def analyze_outputs(
    *,
    outputs: dict[str, np.ndarray],
    source_trace_row: np.ndarray,
    local_trace_row: np.ndarray,
) -> dict[str, Any]:
    missing = set(VARIANT_NAMES) - outputs.keys()
    if missing:
        raise ValueError(f"CUDA output missing variants: {sorted(missing)}")
    source = np.asarray(source_trace_row)
    local = np.asarray(local_trace_row)
    if source.dtype != np.float16 or local.dtype != np.float16:
        raise ValueError("source and local trace rows must have dtype float16")
    normalized = {name: np.asarray(outputs[name]) for name in VARIANT_NAMES}
    for name, value in normalized.items():
        if value.dtype != np.float16:
            raise ValueError(f"{name} must have dtype float16, got {value.dtype}")
        if value.shape != source.shape:
            raise ValueError(
                f"{name} shape {value.shape} does not match {source.shape}"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains non-finite values")
    if not np.array_equal(normalized["cuda_default_full"], source):
        raise ValueError(
            "isolated CUDA default full GEMM does not reproduce source trace"
        )
    return {
        "self_authentication": {
            "default_full_exact_source": True,
        },
        "source_vs_local": _metric(local, source),
        "variants": {
            name: {
                "vs_source": _metric(value, source),
                "vs_local": _metric(value, local),
            }
            for name, value in normalized.items()
        },
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(path) as archive:
        logical = [
            name[:-4] if name.endswith(".npy") else name
            for name in archive.namelist()
        ]
        if len(logical) != len(set(logical)):
            raise ValueError("witness NPZ contains duplicate logical members")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _fp32_gemm_bias_row(
    x,
    weight,
    bias,
    *,
    row_index: int,
    float32_dtype,
    float16_dtype,
):
    product = x.to(dtype=float32_dtype) @ weight.to(dtype=float32_dtype)
    return (
        product[row_index] + bias.to(dtype=float32_dtype)
    ).to(dtype=float16_dtype)


def _to_cpu_numpy_preserve_dtype(value) -> np.ndarray:
    return value.detach().to(device="cpu").numpy()


def _run_cuda(
    witness: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch

    if torch.__version__ != EXPECTED_TORCH:
        raise ValueError(
            f"Torch route must be {EXPECTED_TORCH}, got {torch.__version__}"
        )
    if not torch.cuda.is_available():
        raise ValueError("CUDA route is unavailable")
    device = torch.cuda.get_device_name(0)
    if device != EXPECTED_DEVICE:
        raise ValueError(
            f"CUDA device route must be {EXPECTED_DEVICE}, got {device}"
        )
    matmul_backend = torch.backends.cuda.matmul
    original_reduction = bool(
        matmul_backend.allow_fp16_reduced_precision_reduction
    )
    if original_reduction is not True:
        raise ValueError(
            "default CUDA route disabled FP16 reduced-precision reduction"
        )

    x = torch.from_numpy(witness["torso_input"].copy()).to(
        device="cuda",
        dtype=torch.float16,
    )
    weight = torch.from_numpy(witness["center_weight"].copy()).to(
        device="cuda",
        dtype=torch.float16,
    )
    bias = torch.from_numpy(witness["bias"].copy()).to(
        device="cuda",
        dtype=torch.float16,
    )
    row_index = witness["row_index"]
    outputs: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}

    def execute(name: str, fn) -> None:
        torch.cuda.synchronize()
        started = time.perf_counter()
        value = fn()
        torch.cuda.synchronize()
        timings[name] = time.perf_counter() - started
        outputs[name] = _to_cpu_numpy_preserve_dtype(value)

    try:
        with torch.no_grad():
            matmul_backend.allow_fp16_reduced_precision_reduction = True
            execute(
                "cuda_default_full",
                lambda: (x @ weight)[row_index] + bias,
            )
            execute(
                "cuda_default_m1",
                lambda: (x[row_index : row_index + 1] @ weight)[0] + bias,
            )
            matmul_backend.allow_fp16_reduced_precision_reduction = False
            execute(
                "cuda_no_reduced_full",
                lambda: (x @ weight)[row_index] + bias,
            )
            execute(
                "cuda_fp32_full",
                lambda: _fp32_gemm_bias_row(
                    x,
                    weight,
                    bias,
                    row_index=row_index,
                    float32_dtype=torch.float32,
                    float16_dtype=torch.float16,
                ),
            )
    finally:
        matmul_backend.allow_fp16_reduced_precision_reduction = (
            original_reduction
        )

    return outputs, {
        "torch": torch.__version__,
        "cuda_device": device,
        "default_allow_fp16_reduced_precision_reduction": original_reduction,
        "restored_allow_fp16_reduced_precision_reduction": bool(
            matmul_backend.allow_fp16_reduced_precision_reduction
        ),
        "timings_seconds": timings,
    }


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as output:
            np.savez(output, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witness", required=True, type=Path)
    parser.add_argument("--expected-witness-sha256", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int, default=7697)
    parser.add_argument("--channels", type=int, default=1024)
    parser.add_argument("--expected-row", type=int, default=7693)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "failure_phase": "request_validation",
        "last_trustworthy_phase": None,
        "requested_witness": {
            "path": str(args.witness),
            "sha256": args.expected_witness_sha256,
        },
        "primary_output": {
            "path": str(args.output_npz),
            "exists": False,
        },
    }
    try:
        resolved = [
            args.witness.resolve(),
            args.output_json.resolve(),
            args.output_npz.resolve(),
        ]
        if len(set(resolved)) != len(resolved):
            raise ValueError("witness and output paths must be distinct")
        args.output_npz.unlink(missing_ok=True)
        report["last_trustworthy_phase"] = "output_paths_validated"
        report["failure_phase"] = "input_validation"
        if not _canonical_sha256(args.expected_witness_sha256):
            raise ValueError(
                "expected witness sha256 must be canonical lowercase hex"
            )
        actual_digest = sha256_file(args.witness)
        if actual_digest != args.expected_witness_sha256:
            raise ValueError(
                "witness sha256 mismatch: "
                f"expected {args.expected_witness_sha256}, got {actual_digest}"
            )
        witness = validate_witness_arrays(
            _load_npz(args.witness),
            expected_rows=args.expected_rows,
            channels=args.channels,
            expected_row=args.expected_row,
        )
        report["witness"] = {
            "path": str(args.witness),
            "sha256": actual_digest,
            "size_bytes": args.witness.stat().st_size,
            "route_identity": witness["route_identity"],
            "rows": args.expected_rows,
            "channels": args.channels,
            "row_index": args.expected_row,
            "neighbor_count": witness["neighbor_count"],
        }
        report["last_trustworthy_phase"] = "input_validated"
        report["failure_phase"] = "cuda_execution"
        outputs, runtime = _run_cuda(witness)
        analysis = analyze_outputs(
            outputs=outputs,
            source_trace_row=witness["source_trace_row"],
            local_trace_row=witness["local_trace_row"],
        )
        report["last_trustworthy_phase"] = "cuda_authenticated"
        report["failure_phase"] = "output_publication"
        _write_npz_atomic(args.output_npz, outputs)
        reopened = _load_npz(args.output_npz)
        if set(reopened) != set(outputs) or any(
            not np.array_equal(reopened[name], outputs[name])
            for name in outputs
        ):
            raise ValueError("published NPZ does not reopen exactly")
        report.update(
            {
                "status": "done",
                "failure_phase": None,
                "last_trustworthy_phase": "output_reopened_exact",
                "runtime": runtime,
                "analysis": analysis,
                "primary_output": {
                    "path": str(args.output_npz),
                    "exists": True,
                    "sha256": sha256_file(args.output_npz),
                    "size_bytes": args.output_npz.stat().st_size,
                    "keys": sorted(outputs),
                    "reopened_exact": True,
                },
            }
        )
    except Exception as exc:
        args.output_npz.unlink(missing_ok=True)
        report.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "primary_output": {
                    "path": str(args.output_npz),
                    "exists": False,
                },
            }
        )
        _write_json(args.output_json, report)
        return 1
    _write_json(args.output_json, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
