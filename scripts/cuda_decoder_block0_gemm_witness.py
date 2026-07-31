#!/usr/bin/env python3
"""Probe CUDA FP16 GEMM policy at the first shape-decoder convolution fork."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import traceback
from typing import Any
import zipfile

import numpy as np


SCHEMA = "trellis2mlx.cuda_decoder_block0_gemm_witness.v3"
EXPECTED_TORCH = "2.10.0+cu128"
EXPECTED_DEVICE = "Tesla T4"
VARIANT_NAMES = (
    "cuda_default_full",
    "cuda_default_m1",
    "cuda_no_reduced_full",
    "cublas_default_tensor_op_full",
    "cuda_fp32_full",
)
CUBLAS_STATUS_SUCCESS = 0
CUBLAS_STATUS_NOT_SUPPORTED = 15
CUBLAS_OP_N = 0
CUDA_R_32F = 0
CUDA_R_16F = 2
CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99
CUBLAS_GEMM_REGULAR_REFERENCE = 2
CUBLAS_GEMM_TENSOR_REFERENCE = 100
LEGACY_CUBLAS_EXPLICIT_ALGORITHM_IDS = (
    *range(0, 24),
    *range(100, 116),
)
ROUTE_SCHEMA = "trellis2mlx.decoder_block0_gemm_input.v1"
ROUTE_OPERATION = "shape_decoder.level0.block0.conv.center_gemm"
SOURCE_ROUTE = "official-source-cuda-shape-decoder-level0-trace"
LOCAL_ROUTE = "mlx-shape-decoder-level0-trace-fp16"
CENTER_KERNEL_INDEX = 13

SM75_WMMA_CPP_SOURCE = """
torch::Tensor sm75_wmma_gemm_cuda(
    torch::Tensor input,
    torch::Tensor weight);
"""

SM75_WMMA_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;

__global__ void sm75_wmma_gemm_kernel(
    const half* input,
    const half* weight,
    float* output,
    int rows,
    int channels,
    int reduction) {
  const int tile_row = static_cast<int>(blockIdx.y) * 16;
  const int tile_col = static_cast<int>(blockIdx.x) * 16;
  if (tile_row >= rows || tile_col >= channels) {
    return;
  }

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
      input_fragment;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
      weight_fragment;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float>
      accumulator_fragment;
  wmma::fill_fragment(accumulator_fragment, 0.0f);

  for (int offset = 0; offset < reduction; offset += 16) {
    wmma::load_matrix_sync(
        input_fragment,
        input + tile_row * reduction + offset,
        reduction);
    wmma::load_matrix_sync(
        weight_fragment,
        weight + offset * channels + tile_col,
        channels);
    wmma::mma_sync(
        accumulator_fragment,
        input_fragment,
        weight_fragment,
        accumulator_fragment);
  }
  wmma::store_matrix_sync(
      output + tile_row * channels + tile_col,
      accumulator_fragment,
      channels,
      wmma::mem_row_major);
}

torch::Tensor sm75_wmma_gemm_cuda(
    torch::Tensor input,
    torch::Tensor weight) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf &&
          weight.scalar_type() == at::kHalf,
      "inputs must be float16");
  TORCH_CHECK(
      input.is_contiguous() && weight.is_contiguous(),
      "inputs must be contiguous");
  TORCH_CHECK(
      input.dim() == 2 && weight.dim() == 2,
      "inputs must be matrices");
  const int rows = static_cast<int>(input.size(0));
  const int reduction = static_cast<int>(input.size(1));
  TORCH_CHECK(
      weight.size(0) == reduction,
      "weight reduction dimension mismatch");
  const int channels = static_cast<int>(weight.size(1));
  TORCH_CHECK(
      rows % 16 == 0 && channels % 16 == 0 && reduction % 16 == 0,
      "WMMA dimensions must be multiples of 16");

  auto output = torch::empty(
      {rows, channels},
      input.options().dtype(torch::kFloat32));
  const dim3 grid(channels / 16, rows / 16);
  sm75_wmma_gemm_kernel<<<
      grid,
      32,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
          output.data_ptr<float>(),
          rows,
          channels,
          reduction);
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(
      error == cudaSuccess,
      "SM75 WMMA kernel launch failed: ",
      cudaGetErrorString(error));
  return output;
}
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
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
    if not np.array_equal(
        normalized["cublas_default_tensor_op_full"],
        normalized["cuda_default_full"],
    ):
        raise ValueError(
            "direct legacy cuBLAS default tensor-op GEMM does not reproduce "
            "the PyTorch default full GEMM row"
        )
    return {
        "self_authentication": {
            "default_full_exact_source": True,
            "cublas_default_tensor_op_exact_default_full": True,
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


def _ptx_target_function(ptx: str) -> tuple[str, str, list[str]]:
    lines = ptx.splitlines()
    architecture: str | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(".target "):
            architecture = stripped.removeprefix(".target ").split(",", 1)[0]
        if (
            ".entry" not in stripped
            or "sm75_wmma_gemm_kernel" not in stripped
            or architecture != "sm_75"
        ):
            continue
        symbol = stripped.split(".entry", 1)[1].strip().split("(", 1)[0]
        section: list[str] = []
        brace_depth = 0
        body_started = False
        for section_line in lines[index:]:
            section.append(section_line)
            brace_depth += section_line.count("{")
            if "{" in section_line:
                body_started = True
            brace_depth -= section_line.count("}")
            if body_started and brace_depth == 0:
                return architecture, symbol, section
        raise ValueError("target SM75 WMMA PTX function body is incomplete")
    raise ValueError("compiler evidence lacks target SM75 WMMA kernel PTX")


def _sass_target_function(sass: str) -> tuple[str, str, list[str]]:
    lines = sass.splitlines()
    architecture: str | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("code for "):
            architecture = stripped.removeprefix("code for ").split()[0]
        if (
            not stripped.startswith("Function :")
            or "sm75_wmma_gemm_kernel" not in stripped
            or architecture != "sm_75"
        ):
            continue
        symbol = stripped.split(":", 1)[1].strip()
        section = [line]
        for section_line in lines[index + 1 :]:
            section_stripped = section_line.strip()
            if (
                section_stripped.startswith("Function :")
                or section_stripped.startswith("code for ")
                or section_stripped.startswith("Fatbin ")
            ):
                break
            section.append(section_line)
        return architecture, symbol, section
    raise ValueError("compiler evidence lacks sm_75 target-kernel SASS")


def classify_sm75_compiler_evidence(
    *, ptx: str, sass: str
) -> dict[str, Any]:
    ptx_arch, ptx_symbol, ptx_section = _ptx_target_function(ptx)
    sass_arch, sass_symbol, sass_section = _sass_target_function(sass)
    if ptx_symbol != sass_symbol:
        raise ValueError(
            "PTX and SASS target-kernel symbols differ: "
            f"{ptx_symbol} versus {sass_symbol}"
        )
    ptx_matching_lines = [
        line.strip()
        for line in ptx_section
        if (
            "wmma.mma.sync.aligned" in line
            and "m16n16k16" in line
            and "f32" in line
        )
    ]
    sass_matching_lines = [
        line.strip() for line in sass_section if "HMMA.1688" in line
    ]
    ptx_count = len(ptx_matching_lines)
    sass_count = len(sass_matching_lines)
    if not ptx_count:
        raise ValueError(
            "target-kernel PTX lacks WMMA m16n16k16 FP32 accumulation"
        )
    if not sass_count:
        raise ValueError(
            "target-kernel sm_75 SASS lacks HMMA.1688 instructions"
        )
    return {
        "effective_ptx_architecture": ptx_arch,
        "effective_cubin_architecture": sass_arch,
        "ptx_target_symbol": ptx_symbol,
        "sass_target_symbol": sass_symbol,
        "ptx_wmma_m16n16k16_count": ptx_count,
        "sass_hmma_1688_count": sass_count,
        "ptx_matching_lines": ptx_matching_lines,
        "sass_matching_lines": sass_matching_lines,
    }


def validate_sm75_mma_probe_arrays(
    arrays: dict[str, np.ndarray],
    *,
    reduction: int,
    channels: int,
) -> dict[str, np.ndarray]:
    required = {
        "wmma_input_window": (np.dtype(np.float16), (16, reduction)),
        "cublas_tensor_fp16_unbiased_row": (
            np.dtype(np.float16),
            (channels,),
        ),
        "cublas_regular_fp16_unbiased_row": (
            np.dtype(np.float16),
            (channels,),
        ),
        "cublas_tensor_fp32_row": (np.dtype(np.float32), (channels,)),
        "cublas_regular_fp32_row": (np.dtype(np.float32), (channels,)),
        "wmma_fp32_row": (np.dtype(np.float32), (channels,)),
    }
    normalized: dict[str, np.ndarray] = {}
    for name, (dtype, shape) in required.items():
        if name not in arrays:
            raise ValueError(f"SM75 MMA output missing {name}")
        value = np.asarray(arrays[name])
        if value.dtype != dtype:
            raise ValueError(
                f"{name} must have dtype {dtype}, got {value.dtype}"
            )
        if value.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {value.shape}"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains non-finite values")
        normalized[name] = value
    return normalized


def analyze_sm75_mma_outputs(
    *,
    outputs: dict[str, np.ndarray],
    bias: np.ndarray,
    source_trace_row: np.ndarray,
    local_trace_row: np.ndarray,
) -> dict[str, Any]:
    required = {
        "cublas_tensor_fp16_unbiased_row": np.dtype(np.float16),
        "cublas_regular_fp16_unbiased_row": np.dtype(np.float16),
        "cublas_tensor_fp32_row": np.dtype(np.float32),
        "cublas_regular_fp32_row": np.dtype(np.float32),
        "wmma_fp32_row": np.dtype(np.float32),
    }
    source = np.asarray(source_trace_row)
    local = np.asarray(local_trace_row)
    bias_row = np.asarray(bias)
    if source.dtype != np.float16 or local.dtype != np.float16:
        raise ValueError("source and local trace rows must have dtype float16")
    if bias_row.dtype != np.float16 or bias_row.shape != source.shape:
        raise ValueError(
            "SM75 MMA bias must be a float16 row matching the trace"
        )

    normalized: dict[str, np.ndarray] = {}
    for name, dtype in required.items():
        if name not in outputs:
            raise ValueError(f"SM75 MMA output missing {name}")
        value = np.asarray(outputs[name])
        if value.dtype != dtype:
            raise ValueError(
                f"{name} must have dtype {dtype}, got {value.dtype}"
            )
        if value.shape != source.shape:
            raise ValueError(
                f"{name} shape {value.shape} does not match {source.shape}"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains non-finite values")
        normalized[name] = value

    tensor_fp16 = normalized["cublas_tensor_fp16_unbiased_row"]
    regular_fp16 = normalized["cublas_regular_fp16_unbiased_row"]
    tensor_biased = np.add(tensor_fp16, bias_row, dtype=np.float16)
    regular_biased = np.add(regular_fp16, bias_row, dtype=np.float16)
    if not np.array_equal(tensor_biased, source):
        raise ValueError(
            "cuBLAS tensor FP16 pre-bias row does not reproduce source "
            "after bias"
        )
    if not np.array_equal(regular_biased, local):
        raise ValueError(
            "cuBLAS regular FP16 pre-bias row does not reproduce local "
            "MLX after bias"
        )

    wmma_fp32 = normalized["wmma_fp32_row"]
    tensor_fp32 = normalized["cublas_tensor_fp32_row"]
    regular_fp32 = normalized["cublas_regular_fp32_row"]
    wmma_fp16 = wmma_fp32.astype(np.float16)
    wmma_tensor_fp16 = _metric(wmma_fp16, tensor_fp16)
    wmma_regular_fp16 = _metric(wmma_fp16, regular_fp16)
    wmma_tensor_fp32 = _metric(wmma_fp32, tensor_fp32)
    localization = (
        "sm75_wmma_chain_exact_cublas_tensor"
        if (
            wmma_tensor_fp16["exact"]
            and wmma_tensor_fp32["exact"]
        )
        else "cublas_mainloop_or_epilogue_required"
    )
    return {
        "self_authentication": {
            "tensor_fp16_plus_bias_exact_source": True,
            "regular_fp16_plus_bias_exact_local": True,
        },
        "localization": localization,
        "wmma_fp16_vs_cublas_tensor_fp16": wmma_tensor_fp16,
        "wmma_fp16_vs_cublas_regular_fp16": wmma_regular_fp16,
        "wmma_fp32_vs_cublas_tensor_fp32": wmma_tensor_fp32,
        "wmma_fp32_vs_cublas_regular_fp32": _metric(
            wmma_fp32, regular_fp32
        ),
        "cublas_tensor_fp32_cast_vs_tensor_fp16": _metric(
            tensor_fp32.astype(np.float16), tensor_fp16
        ),
        "cublas_regular_fp32_cast_vs_regular_fp16": _metric(
            regular_fp32.astype(np.float16), regular_fp16
        ),
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


def _invoke_cublas_gemm_ex(
    gemm_ex,
    *,
    handle: int,
    x,
    weight,
    output,
    algorithm_id: int,
    output_cuda_type: int = CUDA_R_16F,
    beta: float = 0.0,
) -> int:
    if len(x.shape) != 2 or len(weight.shape) != 2 or len(output.shape) != 2:
        raise ValueError("cuBLAS GEMM tensors must all be two-dimensional")
    rows, reduction = (int(value) for value in x.shape)
    weight_reduction, channels = (int(value) for value in weight.shape)
    if weight_reduction != reduction:
        raise ValueError(
            f"cuBLAS GEMM reduction mismatch: {reduction} versus "
            f"{weight_reduction}"
        )
    if tuple(output.shape) != (rows, channels):
        raise ValueError(
            f"cuBLAS GEMM output must have shape {(rows, channels)}, "
            f"got {tuple(output.shape)}"
        )
    for name, value in (("x", x), ("weight", weight), ("output", output)):
        if not value.is_contiguous():
            raise ValueError(f"cuBLAS GEMM {name} tensor must be contiguous")
    if output_cuda_type not in {CUDA_R_16F, CUDA_R_32F}:
        raise ValueError(
            f"unsupported cuBLAS output CUDA type {output_cuda_type}"
        )

    alpha = ctypes.c_float(1.0)
    beta_value = ctypes.c_float(beta)
    status = gemm_ex(
        ctypes.c_void_p(handle),
        ctypes.c_int(CUBLAS_OP_N),
        ctypes.c_int(CUBLAS_OP_N),
        ctypes.c_int(channels),
        ctypes.c_int(rows),
        ctypes.c_int(reduction),
        ctypes.byref(alpha),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_int(CUDA_R_16F),
        ctypes.c_int(channels),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(CUDA_R_16F),
        ctypes.c_int(reduction),
        ctypes.byref(beta_value),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int(output_cuda_type),
        ctypes.c_int(channels),
        ctypes.c_int(CUDA_R_32F),
        ctypes.c_int(algorithm_id),
    )
    return int(status)


def _collect_cublas_algorithm_results(
    *,
    algorithm_ids,
    run_algorithm,
    summarize_success,
    channels: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    requested = tuple(int(value) for value in algorithm_ids)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("explicit cuBLAS algorithm IDs must be nonempty and unique")

    statuses: list[int] = []
    supported_ids: list[int] = []
    supported_rows: list[np.ndarray] = []
    supported_nonzero: list[int] = []
    supported_nonzero_local: list[int] = []
    algorithms: list[dict[str, Any]] = []
    unsupported_ids: list[int] = []
    exact_match_ids: list[int] = []
    exact_local_match_ids: list[int] = []
    for algorithm_id in requested:
        status, candidate = run_algorithm(algorithm_id)
        status = int(status)
        statuses.append(status)
        entry: dict[str, Any] = {
            "algorithm_id": algorithm_id,
            "status": status,
        }
        if status == CUBLAS_STATUS_NOT_SUPPORTED:
            if candidate is not None:
                raise ValueError(
                    f"unsupported cuBLAS algorithm {algorithm_id} returned output"
                )
            entry["supported"] = False
            unsupported_ids.append(algorithm_id)
        elif status == CUBLAS_STATUS_SUCCESS:
            if candidate is None:
                raise ValueError(
                    f"successful cuBLAS algorithm {algorithm_id} returned no output"
                )
            metric, selected_row = summarize_success(candidate)
            selected_row = np.asarray(selected_row)
            if selected_row.dtype != np.float16 or selected_row.shape != (channels,):
                raise ValueError(
                    f"cuBLAS algorithm {algorithm_id} selected row must be "
                    f"float16[{channels}], got {selected_row.dtype}"
                    f"{selected_row.shape}"
                )
            if not np.all(np.isfinite(selected_row)):
                raise ValueError(
                    f"cuBLAS algorithm {algorithm_id} selected row is non-finite"
                )
            nonzero = int(metric["nonzero_vs_default_full"])
            max_abs = float(metric["max_abs_vs_default_full"])
            local_nonzero = int(metric["nonzero_vs_local_full"])
            local_max_abs = float(metric["max_abs_vs_local_full"])
            if (
                nonzero < 0
                or local_nonzero < 0
                or not np.isfinite(max_abs)
                or max_abs < 0
                or not np.isfinite(local_max_abs)
                or local_max_abs < 0
            ):
                raise ValueError(
                    f"cuBLAS algorithm {algorithm_id} returned invalid metric"
                )
            entry.update(
                {
                    "supported": True,
                    "nonzero_vs_default_full": nonzero,
                    "max_abs_vs_default_full": max_abs,
                    "nonzero_vs_local_full": local_nonzero,
                    "max_abs_vs_local_full": local_max_abs,
                }
            )
            supported_ids.append(algorithm_id)
            supported_rows.append(np.ascontiguousarray(selected_row))
            supported_nonzero.append(nonzero)
            supported_nonzero_local.append(local_nonzero)
            if nonzero == 0:
                exact_match_ids.append(algorithm_id)
            if local_nonzero == 0:
                exact_local_match_ids.append(algorithm_id)
        else:
            raise RuntimeError(
                f"cuBLAS algorithm {algorithm_id} failed with status {status}"
            )
        algorithms.append(entry)

    rows = (
        np.stack(supported_rows).astype(np.float16, copy=False)
        if supported_rows
        else np.empty((0, channels), dtype=np.float16)
    )
    arrays = {
        "cublas_explicit_algorithm_ids": np.asarray(requested, dtype=np.int32),
        "cublas_explicit_statuses": np.asarray(statuses, dtype=np.int32),
        "cublas_supported_algorithm_ids": np.asarray(
            supported_ids,
            dtype=np.int32,
        ),
        "cublas_supported_rows": rows,
        "cublas_supported_nonzero_vs_default_full": np.asarray(
            supported_nonzero,
            dtype=np.int64,
        ),
        "cublas_supported_nonzero_vs_local_full": np.asarray(
            supported_nonzero_local,
            dtype=np.int64,
        ),
    }
    report = {
        "requested_algorithm_ids": list(requested),
        "supported_algorithm_ids": supported_ids,
        "unsupported_algorithm_ids": unsupported_ids,
        "exact_match_algorithm_ids": exact_match_ids,
        "exact_local_match_algorithm_ids": exact_local_match_ids,
        "algorithms": algorithms,
    }
    return arrays, report


def _load_cublas_gemm_ex():
    failures = []
    for library_name in ("libcublas.so.12", "libcublas.so"):
        try:
            library = ctypes.CDLL(library_name)
            break
        except OSError as exc:
            failures.append(f"{library_name}: {exc}")
    else:
        raise RuntimeError(
            "unable to load the process cuBLAS library: " + "; ".join(failures)
        )
    gemm_ex = library.cublasGemmEx
    gemm_ex.restype = ctypes.c_int
    gemm_ex.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    return library, gemm_ex


def _run_cublas_algorithm_sweep(
    *,
    torch,
    x,
    weight,
    bias,
    default_product,
    local_product,
    row_index: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    library, gemm_ex = _load_cublas_gemm_ex()
    handle = int(torch.cuda.current_blas_handle())

    direct_default = torch.empty_like(default_product)
    default_status = _invoke_cublas_gemm_ex(
        gemm_ex,
        handle=handle,
        x=x,
        weight=weight,
        output=direct_default,
        algorithm_id=CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    )
    torch.cuda.synchronize()
    if default_status != CUBLAS_STATUS_SUCCESS:
        raise RuntimeError(
            "direct legacy cuBLAS default tensor-op route failed with status "
            f"{default_status}"
        )
    default_nonzero = int(
        torch.count_nonzero(direct_default != default_product).item()
    )
    if default_nonzero:
        raise ValueError(
            "direct legacy cuBLAS default tensor-op route differs from the "
            f"PyTorch full product at {default_nonzero} elements"
        )

    def run_algorithm(algorithm_id: int):
        candidate = torch.empty_like(default_product)
        status = _invoke_cublas_gemm_ex(
            gemm_ex,
            handle=handle,
            x=x,
            weight=weight,
            output=candidate,
            algorithm_id=algorithm_id,
        )
        torch.cuda.synchronize()
        return status, candidate if status == CUBLAS_STATUS_SUCCESS else None

    def summarize_success(candidate):
        difference = torch.abs(
            candidate.to(dtype=torch.float32)
            - default_product.to(dtype=torch.float32)
        )
        metric = {
            "nonzero_vs_default_full": int(
                torch.count_nonzero(candidate != default_product).item()
            ),
            "max_abs_vs_default_full": float(difference.max().item()),
            "nonzero_vs_local_full": int(
                torch.count_nonzero(candidate != local_product).item()
            ),
            "max_abs_vs_local_full": float(
                torch.abs(
                    candidate.to(dtype=torch.float32)
                    - local_product.to(dtype=torch.float32)
                )
                .max()
                .item()
            ),
        }
        selected_row = _to_cpu_numpy_preserve_dtype(
            candidate[row_index] + bias
        )
        return metric, selected_row

    arrays, report = _collect_cublas_algorithm_results(
        algorithm_ids=LEGACY_CUBLAS_EXPLICIT_ALGORITHM_IDS,
        run_algorithm=run_algorithm,
        summarize_success=summarize_success,
        channels=int(default_product.shape[1]),
    )
    arrays["cublas_default_tensor_op_full"] = _to_cpu_numpy_preserve_dtype(
        direct_default[row_index] + bias
    )
    report.update(
        {
            "route": "legacy_cublasGemmEx",
            "default_algorithm_id": CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            "default_tensor_op_status": default_status,
            "default_tensor_op_nonzero_vs_pytorch_full": default_nonzero,
            "default_tensor_op_exact_pytorch_full": True,
            "algorithm_inventory_complete": list(
                LEGACY_CUBLAS_EXPLICIT_ALGORITHM_IDS
            )
            == [*range(0, 24), *range(100, 116)],
        }
    )
    del library
    return arrays, report


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _build_sm75_wmma_extension():
    from torch.utils.cpp_extension import load_inline

    previous_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5+PTX"
    try:
        return load_inline(
            name="trellis2mlx_sm75_wmma_gemm_v1",
            cpp_sources=SM75_WMMA_CPP_SOURCE,
            cuda_sources=SM75_WMMA_CUDA_SOURCE,
            functions=["sm75_wmma_gemm_cuda"],
            extra_cuda_cflags=["-O3", "--fmad=true"],
            with_cuda=True,
            verbose=True,
        )
    finally:
        if previous_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch_list


def _collect_sm75_compiler_evidence(module) -> dict[str, Any]:
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        raise RuntimeError("cuobjdump is unavailable")
    module_path = Path(module.__file__)
    if not module_path.is_file():
        raise RuntimeError(
            f"compiled WMMA extension is missing: {module_path}"
        )

    outputs: dict[str, str] = {}
    commands = {
        "ptx": [cuobjdump, "--dump-ptx", str(module_path)],
        "sass": [cuobjdump, "--dump-sass", str(module_path)],
    }
    for name, command in commands.items():
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"cuobjdump {name} failed: "
                + (completed.stdout + completed.stderr).strip()
            )
        outputs[name] = completed.stdout

    classification = classify_sm75_compiler_evidence(
        ptx=outputs["ptx"],
        sass=outputs["sass"],
    )
    return {
        "requested_architecture": "sm_75",
        "effective_ptx_architecture": classification[
            "effective_ptx_architecture"
        ],
        "effective_cubin_architecture": classification[
            "effective_cubin_architecture"
        ],
        "ptx_target_symbol": classification["ptx_target_symbol"],
        "sass_target_symbol": classification["sass_target_symbol"],
        "torch_cuda_arch_list": "7.5+PTX",
        "classification": {
            "ptx_wmma_m16n16k16_count": classification[
                "ptx_wmma_m16n16k16_count"
            ],
            "sass_hmma_1688_count": classification[
                "sass_hmma_1688_count"
            ],
        },
        "extension_sha256": sha256_file(module_path),
        "ptx_sha256": _sha256_text(outputs["ptx"]),
        "sass_sha256": _sha256_text(outputs["sass"]),
        "ptx_matching_lines": classification["ptx_matching_lines"],
        "sass_matching_lines": classification["sass_matching_lines"],
        "cuobjdump_commands": {
            name: ["cuobjdump", *command[1:-1], "<extension>"]
            for name, command in commands.items()
        },
    }


def _run_sm75_mma_probe(
    *,
    torch,
    x,
    weight,
    default_product,
    local_product,
    row_index: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    library, gemm_ex = _load_cublas_gemm_ex()
    handle = int(torch.cuda.current_blas_handle())
    rows, channels = (int(value) for value in default_product.shape)

    def run_cublas(*, algorithm_id: int, output_dtype, output_cuda_type: int):
        candidate = torch.empty(
            (rows, channels),
            device="cuda",
            dtype=output_dtype,
        )
        status = _invoke_cublas_gemm_ex(
            gemm_ex,
            handle=handle,
            x=x,
            weight=weight,
            output=candidate,
            output_cuda_type=output_cuda_type,
            algorithm_id=algorithm_id,
        )
        torch.cuda.synchronize()
        if status != CUBLAS_STATUS_SUCCESS:
            raise RuntimeError(
                f"cuBLAS algorithm {algorithm_id} with output type "
                f"{output_cuda_type} failed with status {status}"
            )
        return candidate

    regular_fp16 = run_cublas(
        algorithm_id=CUBLAS_GEMM_REGULAR_REFERENCE,
        output_dtype=torch.float16,
        output_cuda_type=CUDA_R_16F,
    )
    tensor_fp16 = run_cublas(
        algorithm_id=CUBLAS_GEMM_TENSOR_REFERENCE,
        output_dtype=torch.float16,
        output_cuda_type=CUDA_R_16F,
    )
    regular_nonzero = int(
        torch.count_nonzero(regular_fp16 != local_product).item()
    )
    tensor_nonzero = int(
        torch.count_nonzero(tensor_fp16 != default_product).item()
    )
    if regular_nonzero:
        raise ValueError(
            "regular cuBLAS reference no longer reproduces complete local "
            f"product: {regular_nonzero} mismatches"
        )
    if tensor_nonzero:
        raise ValueError(
            "tensor cuBLAS reference no longer reproduces complete source "
            f"product: {tensor_nonzero} mismatches"
        )

    regular_fp32 = run_cublas(
        algorithm_id=CUBLAS_GEMM_REGULAR_REFERENCE,
        output_dtype=torch.float32,
        output_cuda_type=CUDA_R_32F,
    )
    tensor_fp32 = run_cublas(
        algorithm_id=CUBLAS_GEMM_TENSOR_REFERENCE,
        output_dtype=torch.float32,
        output_cuda_type=CUDA_R_32F,
    )

    window_start = (row_index // 16) * 16
    if window_start + 16 > rows:
        window_start = rows - 16
    if window_start < 0 or not (window_start <= row_index < window_start + 16):
        raise ValueError("unable to construct a 16-row WMMA target window")
    input_window = x[window_start : window_start + 16].contiguous()
    if tuple(input_window.shape) != (16, int(x.shape[1])):
        raise ValueError(
            f"WMMA input window has invalid shape {tuple(input_window.shape)}"
        )

    module = _build_sm75_wmma_extension()
    compiler_evidence = _collect_sm75_compiler_evidence(module)
    torch.cuda.synchronize()
    started = time.perf_counter()
    wmma_output = module.sm75_wmma_gemm_cuda(input_window, weight)
    torch.cuda.synchronize()
    wmma_seconds = time.perf_counter() - started
    if wmma_output.dtype != torch.float32:
        raise ValueError(
            f"WMMA output must be float32, got {wmma_output.dtype}"
        )
    if tuple(wmma_output.shape) != (16, channels):
        raise ValueError(
            f"WMMA output must have shape {(16, channels)}, "
            f"got {tuple(wmma_output.shape)}"
        )
    target_offset = row_index - window_start

    arrays = {
        "wmma_input_window": _to_cpu_numpy_preserve_dtype(input_window),
        "cublas_tensor_fp16_unbiased_row": _to_cpu_numpy_preserve_dtype(
            tensor_fp16[row_index]
        ),
        "cublas_regular_fp16_unbiased_row": _to_cpu_numpy_preserve_dtype(
            regular_fp16[row_index]
        ),
        "cublas_tensor_fp32_row": _to_cpu_numpy_preserve_dtype(
            tensor_fp32[row_index]
        ),
        "cublas_regular_fp32_row": _to_cpu_numpy_preserve_dtype(
            regular_fp32[row_index]
        ),
        "wmma_fp32_row": _to_cpu_numpy_preserve_dtype(
            wmma_output[target_offset]
        ),
    }
    arrays = validate_sm75_mma_probe_arrays(
        arrays,
        reduction=int(x.shape[1]),
        channels=channels,
    )
    report = {
        "route": "direct-sm75-wmma-versus-legacy-cublas",
        "regular_algorithm_id": CUBLAS_GEMM_REGULAR_REFERENCE,
        "tensor_algorithm_id": CUBLAS_GEMM_TENSOR_REFERENCE,
        "regular_fp16_exact_complete_local_product": True,
        "tensor_fp16_exact_complete_pytorch_product": True,
        "wmma_input_window": {
            "start": window_start,
            "stop": window_start + 16,
            "target_offset": target_offset,
            "shape": [16, int(x.shape[1])],
            "dtype": "float16",
            "sha256": sha256_array(arrays["wmma_input_window"]),
        },
        "wmma_weight_matrix_sha256": sha256_array(
            _to_cpu_numpy_preserve_dtype(weight)
        ),
        "wmma_seconds": wmma_seconds,
        "compiler_evidence": compiler_evidence,
    }
    del library
    return arrays, report


def _reduction_policy(matmul_backend) -> dict[str, bool]:
    return {
        "allow_reduced_precision": bool(
            matmul_backend.allow_fp16_reduced_precision_reduction
        ),
        "allow_splitk": bool(
            matmul_backend.allow_fp16_reduced_precision_reduction_split_k
        ),
    }


def _set_reduction_policy(
    matmul_backend,
    *,
    allow_reduced_precision: bool,
    allow_splitk: bool,
) -> dict[str, dict[str, bool]]:
    requested = {
        "allow_reduced_precision": allow_reduced_precision,
        "allow_splitk": allow_splitk,
    }
    matmul_backend.allow_fp16_reduced_precision_reduction = (
        allow_reduced_precision,
        allow_splitk,
    )
    effective = _reduction_policy(matmul_backend)
    if effective != requested:
        raise ValueError(
            f"effective FP16 reduction policy {effective} "
            f"does not match requested {requested}"
        )
    return {
        "requested": requested,
        "effective": effective,
    }


def _run_cuda(
    witness: dict[str, Any],
    local_full_product: np.ndarray,
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
    original_policy = _reduction_policy(matmul_backend)
    if original_policy != {
        "allow_reduced_precision": True,
        "allow_splitk": True,
    }:
        raise ValueError(
            f"default CUDA reduction policy is not fully enabled: "
            f"{original_policy}"
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
    local_product = torch.from_numpy(local_full_product.copy()).to(
        device="cuda",
        dtype=torch.float16,
    )
    row_index = witness["row_index"]
    outputs: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    variant_policies: dict[str, dict[str, dict[str, bool]]] = {}
    active_policy_identity: dict[str, dict[str, bool]] | None = None
    cublas_report: dict[str, Any] | None = None
    sm75_mma_report: dict[str, Any] | None = None

    def execute(name: str, fn) -> None:
        if active_policy_identity is None:
            raise ValueError(f"variant {name} has no requested reduction policy")
        effective = _reduction_policy(matmul_backend)
        if effective != active_policy_identity["effective"]:
            raise ValueError(
                f"variant {name} effective reduction policy {effective} "
                f"changed after assignment {active_policy_identity}"
            )
        variant_policies[name] = {
            "requested": dict(active_policy_identity["requested"]),
            "effective": effective,
        }
        torch.cuda.synchronize()
        started = time.perf_counter()
        value = fn()
        torch.cuda.synchronize()
        timings[name] = time.perf_counter() - started
        outputs[name] = _to_cpu_numpy_preserve_dtype(value)

    try:
        with torch.no_grad():
            active_policy_identity = _set_reduction_policy(
                matmul_backend,
                allow_reduced_precision=True,
                allow_splitk=True,
            )
            torch.cuda.synchronize()
            started = time.perf_counter()
            default_product = x @ weight
            torch.cuda.synchronize()
            timings["cuda_default_full_product"] = time.perf_counter() - started
            execute(
                "cuda_default_full",
                lambda: default_product[row_index] + bias,
            )
            execute(
                "cuda_default_m1",
                lambda: (x[row_index : row_index + 1] @ weight)[0] + bias,
            )
            active_policy_identity = _set_reduction_policy(
                matmul_backend,
                allow_reduced_precision=False,
                allow_splitk=True,
            )
            execute(
                "cuda_no_reduced_full",
                lambda: (x @ weight)[row_index] + bias,
            )
            active_policy_identity = _set_reduction_policy(
                matmul_backend,
                allow_reduced_precision=True,
                allow_splitk=True,
            )
            torch.cuda.synchronize()
            started = time.perf_counter()
            cublas_arrays, cublas_report = _run_cublas_algorithm_sweep(
                torch=torch,
                x=x,
                weight=weight,
                bias=bias,
                default_product=default_product,
                local_product=local_product,
                row_index=row_index,
            )
            torch.cuda.synchronize()
            timings["cublas_explicit_algorithm_sweep"] = (
                time.perf_counter() - started
            )
            outputs.update(cublas_arrays)
            variant_policies["cublas_default_tensor_op_full"] = {
                "requested": {
                    "route": "legacy_cublasGemmEx",
                    "algorithm_id": CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                },
                "effective": {
                    "route": "legacy_cublasGemmEx",
                    "algorithm_id": CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                    "exact_pytorch_default_full": True,
                },
            }
            torch.cuda.synchronize()
            started = time.perf_counter()
            sm75_arrays, sm75_mma_report = _run_sm75_mma_probe(
                torch=torch,
                x=x,
                weight=weight,
                default_product=default_product,
                local_product=local_product,
                row_index=row_index,
            )
            torch.cuda.synchronize()
            timings["sm75_mma_probe"] = time.perf_counter() - started
            outputs.update(sm75_arrays)
            active_policy_identity = _set_reduction_policy(
                matmul_backend,
                allow_reduced_precision=True,
                allow_splitk=True,
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
            original_policy["allow_reduced_precision"],
            original_policy["allow_splitk"],
        )

    restored_policy = _reduction_policy(matmul_backend)
    if restored_policy != original_policy:
        raise ValueError(
            f"restored FP16 reduction policy {restored_policy} "
            f"does not match original {original_policy}"
        )
    return outputs, {
        "torch": torch.__version__,
        "cuda_device": device,
        "default_reduction_policy": original_policy,
        "variant_reduction_policies": variant_policies,
        "cublas_explicit_algorithm_sweep": cublas_report,
        "sm75_mma_probe": sm75_mma_report,
        "restored_reduction_policy": restored_policy,
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
    parser.add_argument("--local-full-product", required=True, type=Path)
    parser.add_argument("--expected-local-full-product-sha256", required=True)
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
        "requested_local_full_product": {
            "path": str(args.local_full_product),
            "sha256": args.expected_local_full_product_sha256,
        },
        "primary_output": {
            "path": str(args.output_npz),
            "exists": False,
        },
    }
    try:
        resolved = [
            args.witness.resolve(),
            args.local_full_product.resolve(),
            args.output_json.resolve(),
            args.output_npz.resolve(),
        ]
        if len(set(resolved)) != len(resolved):
            raise ValueError("witness and output paths must be distinct")
        args.output_json.unlink(missing_ok=True)
        args.output_npz.unlink(missing_ok=True)
        report["last_trustworthy_phase"] = "output_paths_validated"
        report["failure_phase"] = "input_validation"
        if not _canonical_sha256(args.expected_witness_sha256):
            raise ValueError(
                "expected witness sha256 must be canonical lowercase hex"
            )
        if not _canonical_sha256(args.expected_local_full_product_sha256):
            raise ValueError(
                "expected local full product sha256 must be canonical "
                "lowercase hex"
            )
        actual_digest = sha256_file(args.witness)
        if actual_digest != args.expected_witness_sha256:
            raise ValueError(
                "witness sha256 mismatch: "
                f"expected {args.expected_witness_sha256}, got {actual_digest}"
            )
        actual_local_digest = sha256_file(args.local_full_product)
        if actual_local_digest != args.expected_local_full_product_sha256:
            raise ValueError(
                "local full product sha256 mismatch: expected "
                f"{args.expected_local_full_product_sha256}, "
                f"got {actual_local_digest}"
            )
        witness = validate_witness_arrays(
            _load_npz(args.witness),
            expected_rows=args.expected_rows,
            channels=args.channels,
            expected_row=args.expected_row,
        )
        local_arrays = _load_npz(args.local_full_product)
        if set(local_arrays) != {"best_product"}:
            raise ValueError(
                "local full product NPZ must contain exactly 'best_product'"
            )
        local_full_product = _require_array(
            local_arrays,
            "best_product",
            dtype=np.float16,
            shape=(args.expected_rows, args.channels),
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
        report["local_full_product"] = {
            "path": str(args.local_full_product),
            "sha256": actual_local_digest,
            "size_bytes": args.local_full_product.stat().st_size,
            "key": "best_product",
            "shape": [args.expected_rows, args.channels],
            "dtype": "float16",
        }
        report["last_trustworthy_phase"] = "input_validated"
        report["failure_phase"] = "cuda_execution"
        outputs, runtime = _run_cuda(witness, local_full_product)
        validate_sm75_mma_probe_arrays(
            outputs,
            reduction=int(witness["torso_input"].shape[1]),
            channels=args.channels,
        )
        analysis = analyze_outputs(
            outputs=outputs,
            source_trace_row=witness["source_trace_row"],
            local_trace_row=witness["local_trace_row"],
        )
        analysis["sm75_mma"] = analyze_sm75_mma_outputs(
            outputs=outputs,
            bias=witness["bias"],
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
        _write_json(args.output_json, report)
        return 0
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
    except BaseException:
        args.output_json.unlink(missing_ok=True)
        args.output_npz.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
