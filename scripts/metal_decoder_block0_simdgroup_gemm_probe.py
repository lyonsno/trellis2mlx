#!/usr/bin/env python3
"""Probe Apple's direct SIMD-group matrix arithmetic against the SM75 witness."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
import traceback
from typing import Any, Callable
import zipfile

import numpy as np


SCHEMA = "trellis2mlx.metal_decoder_block0_simdgroup_gemm_probe.v1"
EXPECTED_DEVICE = "Apple M4 Max"
SELECTED_WINDOW_ROW = 13
ROWS = 16
CHANNELS = 1024
REDUCTION = 1024
KERNEL_IDENTITY = "direct_simdgroup_matrix_8x8x8"

METAL_SOURCE = r"""
    const ushort lane = thread_index_in_simdgroup;
    const uint tile_col = threadgroup_position_in_grid.x * 8;
    const uint tile_row = threadgroup_position_in_grid.y * 8;
    const short qid = lane / 4;
    const short frag_row = (qid & 4) + ((lane / 2) % 4);
    const short frag_col = (qid & 2) * 2 + (lane % 2) * 2;
    float2 accumulator = float2(0.0f);

    for (uint k = 0; k < 1024; k += 8) {
        metal::simdgroup_matrix<half, 8, 8> a_fragment;
        metal::simdgroup_matrix<half, 8, 8> b_fragment;
        metal::simdgroup_matrix<float, 8, 8> c_fragment;
        metal::simdgroup_matrix<float, 8, 8> d_fragment;

        a_fragment.thread_elements()[0] =
            x[(tile_row + frag_row) * 1024 + k + frag_col];
        a_fragment.thread_elements()[1] =
            x[(tile_row + frag_row) * 1024 + k + frag_col + 1];
        b_fragment.thread_elements()[0] =
            weight[(k + frag_row) * 1024 + tile_col + frag_col];
        b_fragment.thread_elements()[1] =
            weight[(k + frag_row) * 1024 + tile_col + frag_col + 1];
        c_fragment.thread_elements()[0] = accumulator[0];
        c_fragment.thread_elements()[1] = accumulator[1];

        simdgroup_multiply_accumulate(
            d_fragment, a_fragment, b_fragment, c_fragment);
        accumulator[0] = d_fragment.thread_elements()[0];
        accumulator[1] = d_fragment.thread_elements()[1];
    }

    output[(tile_row + frag_row) * 1024 + tile_col + frag_col] =
        accumulator[0];
    output[(tile_row + frag_row) * 1024 + tile_col + frag_col + 1] =
        accumulator[1];
"""
METAL_SOURCE_SHA256 = hashlib.sha256(METAL_SOURCE.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(path) as archive:
        logical = [
            name[:-4] if name.endswith(".npy") else name
            for name in archive.namelist()
        ]
        if len(logical) != len(set(logical)):
            raise ValueError(f"{path.name} contains duplicate logical members")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _require_array(
    arrays: dict[str, np.ndarray],
    name: str,
    *,
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> np.ndarray:
    if name not in arrays:
        raise ValueError(f"archive missing required array {name!r}")
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


def _verify_digest(path: Path, expected: str, *, label: str) -> str:
    if not _canonical_sha256(expected):
        raise ValueError(f"expected {label} sha256 must be canonical lowercase")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{label} sha256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def load_probe_inputs(
    witness_path: Path,
    cuda_result_path: Path,
    *,
    expected_witness_sha256: str,
    expected_cuda_result_sha256: str,
) -> dict[str, Any]:
    witness_path = Path(witness_path)
    cuda_result_path = Path(cuda_result_path)
    witness_sha256 = _verify_digest(
        witness_path,
        expected_witness_sha256,
        label="witness",
    )
    cuda_sha256 = _verify_digest(
        cuda_result_path,
        expected_cuda_result_sha256,
        label="CUDA result",
    )
    witness = _load_npz(witness_path)
    cuda = _load_npz(cuda_result_path)
    result: dict[str, Any] = {
        "witness_sha256": witness_sha256,
        "cuda_result_sha256": cuda_sha256,
        "center_weight": _require_array(
            witness,
            "center_weight",
            dtype=np.float16,
            shape=(REDUCTION, CHANNELS),
        ),
        "wmma_input_window": _require_array(
            cuda,
            "wmma_input_window",
            dtype=np.float16,
            shape=(ROWS, REDUCTION),
        ),
    }
    for name, dtype in (
        ("cublas_tensor_fp16_unbiased_row", np.float16),
        ("cublas_regular_fp16_unbiased_row", np.float16),
        ("cublas_tensor_fp32_row", np.float32),
        ("cublas_regular_fp32_row", np.float32),
        ("wmma_fp32_row", np.float32),
    ):
        result[name] = _require_array(
            cuda,
            name,
            dtype=dtype,
            shape=(CHANNELS,),
        )

    if not np.array_equal(
        result["wmma_fp32_row"],
        result["cublas_tensor_fp32_row"],
    ):
        raise ValueError(
            "WMMA FP32 row does not exactly authenticate the cuBLAS tensor row"
        )
    if not np.array_equal(
        result["wmma_fp32_row"].astype(np.float16),
        result["cublas_tensor_fp16_unbiased_row"],
    ):
        raise ValueError(
            "WMMA FP32 cast does not exactly authenticate the tensor FP16 row"
        )
    if np.array_equal(
        result["cublas_tensor_fp32_row"],
        result["cublas_regular_fp32_row"],
    ):
        raise ValueError("CUDA tensor and regular FP32 anchors are not distinct")
    return result


def _metric(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    return {
        "exact": bool(np.array_equal(actual, expected)),
        "nonzero": int(np.count_nonzero(actual != expected)),
        "max_abs": float(np.max(np.abs(difference), initial=0.0)),
        "mean_abs": float(np.mean(np.abs(difference))),
    }


def _validate_metal_output(value: np.ndarray) -> np.ndarray:
    output = np.asarray(value)
    if output.dtype != np.float32:
        raise ValueError(
            f"Metal output must have dtype float32, got {output.dtype}"
        )
    if output.shape != (ROWS, CHANNELS):
        raise ValueError(
            f"Metal output must have shape {(ROWS, CHANNELS)}, got {output.shape}"
        )
    if not np.all(np.isfinite(output)):
        raise ValueError("Metal output contains non-finite values")
    return np.ascontiguousarray(output)


def analyze_metal_output(
    metal_fp32_full: np.ndarray,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    metal_fp32_full = _validate_metal_output(metal_fp32_full)
    metal_fp32 = metal_fp32_full[SELECTED_WINDOW_ROW]
    metal_fp16 = metal_fp32.astype(np.float16)
    tensor_fp32 = inputs["cublas_tensor_fp32_row"]
    regular_fp32 = inputs["cublas_regular_fp32_row"]
    tensor_fp16 = inputs["cublas_tensor_fp16_unbiased_row"]
    regular_fp16 = inputs["cublas_regular_fp16_unbiased_row"]
    tensor32 = _metric(metal_fp32, tensor_fp32)
    regular32 = _metric(metal_fp32, regular_fp32)
    tensor16 = _metric(metal_fp16, tensor_fp16)
    regular16 = _metric(metal_fp16, regular_fp16)
    if tensor32["exact"] and tensor16["exact"]:
        classification = "sm75_tensor_exact"
    elif regular32["exact"] and regular16["exact"]:
        classification = "regular_exact"
    else:
        classification = "third_island"
    return {
        "classification": classification,
        "selected_window_row": SELECTED_WINDOW_ROW,
        "metal_fp32_vs_sm75_tensor_fp32": tensor32,
        "metal_fp32_vs_regular_fp32": regular32,
        "metal_fp16_vs_sm75_tensor_fp16": tensor16,
        "metal_fp16_vs_regular_fp16": regular16,
    }


def _validate_effective_route(route: Any) -> dict[str, str]:
    if not isinstance(route, dict):
        raise ValueError("backend effective route must be an object")
    expected = {
        "backend": "metal",
        "device": EXPECTED_DEVICE,
        "kernel": KERNEL_IDENTITY,
        "metal_source_sha256": METAL_SOURCE_SHA256,
    }
    normalized: dict[str, str] = {}
    for name, expected_value in expected.items():
        value = route.get(name)
        if value != expected_value:
            raise ValueError(
                f"effective route {name} must be {expected_value!r}, "
                f"got {value!r}"
            )
        normalized[name] = value
    return normalized


def run_metal_backend(
    window: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, dict[str, str]]:
    import mlx.core as mx

    kernel = mx.fast.metal_kernel(
        name="decoder_block0_direct_simdgroup_gemm",
        input_names=["x", "weight"],
        output_names=["output"],
        source=METAL_SOURCE,
        ensure_row_contiguous=True,
    )
    x_mx = mx.array(window)
    weight_mx = mx.array(weight)
    started = time.perf_counter()
    output = kernel(
        inputs=[x_mx, weight_mx],
        grid=(32 * (CHANNELS // 8), ROWS // 8, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(ROWS, CHANNELS)],
        output_dtypes=[mx.float32],
    )[0]
    mx.eval(output)
    elapsed = time.perf_counter() - started
    device_info = mx.device_info()
    device = str(
        device_info.get("device_name")
        or device_info.get("name")
        or device_info
    )
    return np.asarray(output), {
        "backend": "metal",
        "device": device,
        "kernel": KERNEL_IDENTITY,
        "metal_source_sha256": METAL_SOURCE_SHA256,
        "elapsed_seconds": str(elapsed),
    }


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        np.savez(handle, **arrays)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_probe(
    *,
    witness_path: Path,
    cuda_result_path: Path,
    expected_witness_sha256: str,
    expected_cuda_result_sha256: str,
    output_json: Path,
    output_npz: Path,
    backend: Callable[
        [np.ndarray, np.ndarray],
        tuple[np.ndarray, dict[str, str]],
    ] = run_metal_backend,
) -> dict[str, Any]:
    output_json = Path(output_json)
    output_npz = Path(output_npz)
    if output_json.resolve() == output_npz.resolve():
        output_json.unlink(missing_ok=True)
        raise ValueError("output_json and output_npz must be distinct paths")
    output_json.unlink(missing_ok=True)
    output_npz.unlink(missing_ok=True)
    phase = "input_validation"
    inputs: dict[str, Any] | None = None
    try:
        inputs = load_probe_inputs(
            Path(witness_path),
            Path(cuda_result_path),
            expected_witness_sha256=expected_witness_sha256,
            expected_cuda_result_sha256=expected_cuda_result_sha256,
        )
        phase = "backend_execution"
        metal_output, route = backend(
            inputs["wmma_input_window"],
            inputs["center_weight"],
        )
        phase = "backend_output_validation"
        metal_fp32 = _validate_metal_output(metal_output)
        route = _validate_effective_route(route)
        analysis = analyze_metal_output(metal_fp32, inputs)
        metal_fp16 = metal_fp32.astype(np.float16)
        phase = "primary_publication"
        _write_npz_atomic(
            output_npz,
            metal_fp32_full=metal_fp32,
            metal_fp16_full=metal_fp16,
        )
        report: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "done",
            "failure_phase": None,
            "effective_route": route,
            "artifacts": {
                "witness_path": str(Path(witness_path).resolve()),
                "witness_sha256": inputs["witness_sha256"],
                "cuda_result_path": str(Path(cuda_result_path).resolve()),
                "cuda_result_sha256": inputs["cuda_result_sha256"],
                "output_npz": str(output_npz.resolve()),
                "output_npz_sha256": sha256_file(output_npz),
            },
            "analysis": analysis,
        }
        _write_json_atomic(output_json, report)
        return report
    except Exception as error:
        output_npz.unlink(missing_ok=True)
        failure = {
            "schema": SCHEMA,
            "status": "failed",
            "failure_phase": phase,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
        }
        _write_json_atomic(output_json, failure)
        raise
    except BaseException:
        output_json.unlink(missing_ok=True)
        output_npz.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witness", type=Path, required=True)
    parser.add_argument("--witness-sha256", required=True)
    parser.add_argument("--cuda-result", type=Path, required=True)
    parser.add_argument("--cuda-result-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = run_probe(
        witness_path=args.witness,
        cuda_result_path=args.cuda_result,
        expected_witness_sha256=args.witness_sha256,
        expected_cuda_result_sha256=args.cuda_result_sha256,
        output_json=args.output_json,
        output_npz=args.output_npz,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
