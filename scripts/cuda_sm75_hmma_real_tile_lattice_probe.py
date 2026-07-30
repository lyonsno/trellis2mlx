#!/usr/bin/env python3
"""Compare direct SM75 HMMA composition with WMMA over one real tile lattice."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import traceback
from typing import Any, Callable
import zipfile

import numpy as np


SCHEMA = "trellis2mlx.cuda_sm75_hmma_real_tile_lattice_probe.v1"
TEST_SCHEMA = "trellis2mlx.cuda_sm75_hmma_real_tile_lattice_probe.test.v1"
EXPECTED_TORCH = "2.10.0+cu128"
EXPECTED_DEVICE = "Tesla T4"
CASE_COUNT = 1 << 16
MATRIX_SIZE = 16
TILE_COLUMN = 16
KERNEL_IDENTITY = "direct_sm75_hmma_real_tile_lattice_m16n8k8"

PARENT_WITNESS_SHA256 = (
    "9fb030c521b0489bbdf7e0ee7eed29bd775d3f886894137da7757c2a38e0c105"
)
PARENT_DIRECT_WMMA_SHA256 = (
    "beb81530139d62dcc7f1e8690e0879b3d0ef8653cd6f09ab64baf69bb56206d5"
)
PARENT_PREFIX_SHA256 = (
    "329cd27cf3e90a3db74aa0b66c1a255aa41f1b9e55105416c1ab1552245aea94"
)

CPP_SOURCE = """
#include <vector>

std::vector<torch::Tensor> sm75_hmma_real_tile_lattice_cuda(
    torch::Tensor matrix_a,
    torch::Tensor matrix_b);
"""

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <vector>

namespace wmma = nvcuda::wmma;

__device__ __forceinline__ unsigned pack_halves(half low, half high) {
  union {
    half2 values;
    unsigned bits;
  } packed;
  packed.values = __halves2half2(low, high);
  return packed.bits;
}

__global__ void sm75_hmma_m16n8k8_stage_kernel(
    const half* matrix_a,
    const half* matrix_b,
    const float* input_c,
    float* output,
    int k_offset,
    int cases,
    int matrix_stride) {
  const int case_index = static_cast<int>(blockIdx.x);
  if (case_index >= cases) {
    return;
  }
  const int n_offset = 8 * static_cast<int>(blockIdx.y);
  const int lane = static_cast<int>(threadIdx.x);
  const int group = lane / 4;
  const int pair = lane % 4;
  const int case_offset = case_index * matrix_stride;
  const half* a = matrix_a + case_offset;
  const half* b = matrix_b + case_offset;
  const float* c = input_c + case_offset;
  float* d = output + case_offset;

  const int k0 = k_offset + 2 * pair;
  const int k1 = k0 + 1;
  const unsigned a0 = pack_halves(
      a[group * 16 + k0],
      a[group * 16 + k1]);
  const unsigned a1 = pack_halves(
      a[(group + 8) * 16 + k0],
      a[(group + 8) * 16 + k1]);
  const unsigned b0 = pack_halves(
      b[k0 * 16 + n_offset + group],
      b[k1 * 16 + n_offset + group]);

  const int column0 = n_offset + 2 * pair;
  const int column1 = column0 + 1;
  float d0 = c[group * 16 + column0];
  float d1 = c[group * 16 + column1];
  float d2 = c[(group + 8) * 16 + column0];
  float d3 = c[(group + 8) * 16 + column1];

  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(b0));

  d[group * 16 + column0] = d0;
  d[group * 16 + column1] = d1;
  d[(group + 8) * 16 + column0] = d2;
  d[(group + 8) * 16 + column1] = d3;
}

__global__ void sm75_wmma_real_lattice_kernel(
    const half* matrix_a,
    const half* matrix_b,
    float* output,
    int cases) {
  const int case_index = static_cast<int>(blockIdx.x);
  if (case_index >= cases) {
    return;
  }
  const int case_offset = case_index * 16 * 16;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
      matrix_a_fragment;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
      matrix_b_fragment;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float>
      accumulator_fragment;
  wmma::load_matrix_sync(
      matrix_a_fragment,
      matrix_a + case_offset,
      16);
  wmma::load_matrix_sync(
      matrix_b_fragment,
      matrix_b + case_offset,
      16);
  wmma::fill_fragment(accumulator_fragment, 0.0f);
  wmma::mma_sync(
      accumulator_fragment,
      matrix_a_fragment,
      matrix_b_fragment,
      accumulator_fragment);
  wmma::store_matrix_sync(
      output + case_offset,
      accumulator_fragment,
      16,
      wmma::mem_row_major);
}

std::vector<torch::Tensor> sm75_hmma_real_tile_lattice_cuda(
    torch::Tensor matrix_a,
    torch::Tensor matrix_b) {
  TORCH_CHECK(
      matrix_a.is_cuda() && matrix_b.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(
      matrix_a.scalar_type() == at::kHalf &&
          matrix_b.scalar_type() == at::kHalf,
      "inputs must be float16");
  TORCH_CHECK(
      matrix_a.is_contiguous() && matrix_b.is_contiguous(),
      "inputs must be contiguous");
  TORCH_CHECK(
      matrix_a.sizes() == matrix_b.sizes(),
      "input shapes must match");
  TORCH_CHECK(
      matrix_a.dim() == 3 &&
          matrix_a.size(0) > 0 &&
          matrix_a.size(1) == 16 &&
          matrix_a.size(2) == 16,
      "inputs must have shape [cases, 16, 16]");

  const int cases = static_cast<int>(matrix_a.size(0));
  auto options = matrix_a.options().dtype(torch::kFloat32);
  auto stage_zero = torch::zeros({cases, 16, 16}, options);
  auto stage_one = torch::zeros({cases, 16, 16}, options);
  auto wmma_output = torch::empty({cases, 16, 16}, options);
  dim3 direct_grid(cases, 2);
  auto stream = at::cuda::getCurrentCUDAStream();

  sm75_hmma_m16n8k8_stage_kernel<<<direct_grid, 32, 0, stream>>>(
      reinterpret_cast<const half*>(matrix_a.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(matrix_b.data_ptr<at::Half>()),
      stage_zero.data_ptr<float>(),
      stage_zero.data_ptr<float>(),
      0,
      cases,
      16 * 16);
  TORCH_CHECK(
      cudaGetLastError() == cudaSuccess,
      "SM75 direct first-stage launch failed");

  sm75_hmma_m16n8k8_stage_kernel<<<direct_grid, 32, 0, stream>>>(
      reinterpret_cast<const half*>(matrix_a.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(matrix_b.data_ptr<at::Half>()),
      stage_zero.data_ptr<float>(),
      stage_one.data_ptr<float>(),
      8,
      cases,
      16 * 16);
  TORCH_CHECK(
      cudaGetLastError() == cudaSuccess,
      "SM75 direct second-stage launch failed");

  sm75_wmma_real_lattice_kernel<<<cases, 32, 0, stream>>>(
      reinterpret_cast<const half*>(matrix_a.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(matrix_b.data_ptr<at::Half>()),
      wmma_output.data_ptr<float>(),
      cases);
  TORCH_CHECK(
      cudaGetLastError() == cudaSuccess,
      "SM75 WMMA reference launch failed");
  return {stage_zero, stage_one, wmma_output};
}
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _canonical_sha256(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_digest(path: Path, expected: str, *, label: str) -> str:
    if not _canonical_sha256(expected):
        raise ValueError(f"expected {label} sha256 must be canonical lowercase")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{label} sha256 mismatch: expected {expected}, got {actual}"
        )
    return actual


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


def load_probe_inputs(
    witness_path: Path,
    cuda_result_path: Path,
    prefix_result_path: Path,
    *,
    expected_witness_sha256: str,
    expected_cuda_result_sha256: str,
    expected_prefix_result_sha256: str,
) -> dict[str, Any]:
    witness_path = Path(witness_path)
    cuda_result_path = Path(cuda_result_path)
    prefix_result_path = Path(prefix_result_path)
    witness = _load_npz(witness_path)
    cuda = _load_npz(cuda_result_path)
    prefix = _load_npz(prefix_result_path)
    weight = _require_array(
        witness,
        "center_weight",
        dtype=np.float16,
        shape=(1024, 1024),
    )
    window = _require_array(
        cuda,
        "wmma_input_window",
        dtype=np.float16,
        shape=(16, 1024),
    )
    prefixes = _require_array(
        prefix,
        "wmma_prefix_fp32",
        dtype=np.float32,
        shape=(64, 16, 16),
    )
    return {
        "witness_sha256": _verify_digest(
            witness_path,
            expected_witness_sha256,
            label="witness",
        ),
        "cuda_result_sha256": _verify_digest(
            cuda_result_path,
            expected_cuda_result_sha256,
            label="CUDA result",
        ),
        "prefix_result_sha256": _verify_digest(
            prefix_result_path,
            expected_prefix_result_sha256,
            label="prefix result",
        ),
        "base_a": np.ascontiguousarray(window[:, :16]),
        "base_b": np.ascontiguousarray(
            weight[:16, TILE_COLUMN : TILE_COLUMN + 16]
        ),
        "expected_full_wmma": np.ascontiguousarray(prefixes[0]),
    }


def complete_masks() -> np.ndarray:
    return np.arange(CASE_COUNT, dtype=np.uint16)


def _validate_masks(value: np.ndarray) -> np.ndarray:
    masks = np.asarray(value)
    if masks.dtype != np.uint16 or masks.ndim != 1 or masks.size == 0:
        raise ValueError("subset masks must be a nonempty uint16 vector")
    return np.ascontiguousarray(masks)


def _validate_complete_masks(value: np.ndarray) -> np.ndarray:
    masks = _validate_masks(value)
    expected = complete_masks()
    if masks.shape != expected.shape or not np.array_equal(masks, expected):
        raise ValueError(
            "authoritative subset masks must be the complete ordered "
            "uint16[65536] lattice"
        )
    return masks


def generate_cases(
    base_a: np.ndarray,
    base_b: np.ndarray,
    masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    base_a = np.asarray(base_a)
    base_b = np.asarray(base_b)
    if (
        base_a.dtype != np.float16
        or base_b.dtype != np.float16
        or base_a.shape != (16, 16)
        or base_b.shape != (16, 16)
    ):
        raise ValueError("real tile operands must be float16[16,16]")
    masks = _validate_masks(masks)
    positions = np.arange(16, dtype=np.uint32)
    enabled = (
        (masks.astype(np.uint32)[:, None] >> positions[None, :]) & 1
    ).astype(bool)
    matrix_a = np.where(
        enabled[:, None, :],
        base_a[None, :, :],
        np.float16(0),
    )
    matrix_b = np.where(
        enabled[:, :, None],
        base_b[None, :, :],
        np.float16(0),
    )
    return (
        np.ascontiguousarray(matrix_a),
        np.ascontiguousarray(matrix_b),
    )


def validate_outputs(
    value: np.ndarray,
    *,
    case_count: int,
    label: str,
) -> np.ndarray:
    outputs = np.asarray(value)
    if outputs.dtype != np.float32:
        raise ValueError(f"{label} outputs must have dtype float32")
    if outputs.shape != (case_count, 16, 16):
        raise ValueError(
            f"{label} outputs must have shape ({case_count}, 16, 16), "
            f"got {outputs.shape}"
        )
    if not np.all(np.isfinite(outputs)):
        raise ValueError(f"{label} outputs contain non-finite values")
    if not np.any(outputs):
        raise ValueError(f"{label} outputs are blank")
    return np.ascontiguousarray(outputs)


def _metric(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    actual_bits = np.asarray(actual, dtype=np.float32).view(np.uint32)
    expected_bits = np.asarray(expected, dtype=np.float32).view(np.uint32)
    difference = (
        np.asarray(actual, dtype=np.float64)
        - np.asarray(expected, dtype=np.float64)
    )
    return {
        "exact": bool(np.array_equal(actual_bits, expected_bits)),
        "nonzero": int(np.count_nonzero(actual_bits != expected_bits)),
        "max_abs": float(np.max(np.abs(difference), initial=0.0)),
        "mean_abs": float(np.mean(np.abs(difference))),
    }


def analyze_outputs(
    masks: np.ndarray,
    direct_stage0: np.ndarray,
    direct_stage1: np.ndarray,
    wmma: np.ndarray,
    *,
    expected_full_wmma: np.ndarray,
) -> dict[str, Any]:
    masks = _validate_masks(masks)
    case_count = int(masks.size)
    direct_stage0 = validate_outputs(
        direct_stage0,
        case_count=case_count,
        label="direct stage 0",
    )
    direct_stage1 = validate_outputs(
        direct_stage1,
        case_count=case_count,
        label="direct stage 1",
    )
    wmma = validate_outputs(wmma, case_count=case_count, label="WMMA")
    full_indices = np.flatnonzero(masks == np.uint16(0xFFFF))
    if full_indices.size != 1:
        raise ValueError("subset masks must contain exactly one full mask")
    full_index = int(full_indices[0])
    expected_full_wmma = np.asarray(expected_full_wmma)
    if (
        expected_full_wmma.dtype != np.float32
        or expected_full_wmma.shape != (16, 16)
    ):
        raise ValueError("expected full WMMA must be float32[16,16]")
    full_metric = _metric(wmma[full_index], expected_full_wmma)
    if not full_metric["exact"]:
        raise ValueError(
            "full-mask WMMA does not reproduce the authenticated prefix: "
            f"{full_metric['nonzero']} mismatches"
        )
    composition = _metric(direct_stage1, wmma)
    return {
        "classification": (
            "register_visible_composition_exact"
            if composition["exact"]
            else "hidden_cross_instruction_state"
        ),
        "case_count": case_count,
        "direct_vs_wmma": composition,
        "full_mask_wmma_vs_prefix": full_metric,
        "stage0_nonzero": int(np.count_nonzero(direct_stage0)),
    }


def _target_section(
    text: str,
    *,
    kind: str,
    target_name: str,
) -> tuple[str, str, list[str]]:
    lines = text.splitlines()
    architecture: str | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if kind == "ptx" and stripped.startswith(".target "):
            architecture = stripped.removeprefix(".target ").split(",", 1)[0]
        elif kind == "sass" and stripped.startswith("code for "):
            architecture = stripped.removeprefix("code for ").split()[0]
        target = (
            kind == "ptx"
            and ".entry" in stripped
            and target_name in stripped
        ) or (
            kind == "sass"
            and stripped.startswith("Function :")
            and target_name in stripped
        )
        if not target or architecture != "sm_75":
            continue
        symbol = (
            stripped.split(".entry", 1)[1].strip().split("(", 1)[0]
            if kind == "ptx"
            else stripped.split(":", 1)[1].strip()
        )
        section = [line]
        if kind == "ptx":
            brace_depth = 0
            body_started = False
            for section_line in lines[index + 1 :]:
                section.append(section_line)
                brace_depth += section_line.count("{")
                body_started = body_started or "{" in section_line
                brace_depth -= section_line.count("}")
                if body_started and brace_depth == 0:
                    return architecture, symbol, section
            raise ValueError(f"target {target_name} PTX is incomplete")
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
    raise ValueError(f"compiler evidence lacks target SM75 {target_name} {kind}")


def classify_compiler_evidence(*, ptx: str, sass: str) -> dict[str, Any]:
    direct_name = "sm75_hmma_m16n8k8_stage_kernel"
    wmma_name = "sm75_wmma_real_lattice_kernel"
    dp_arch, dp_symbol, dp_section = _target_section(
        ptx,
        kind="ptx",
        target_name=direct_name,
    )
    ds_arch, ds_symbol, ds_section = _target_section(
        sass,
        kind="sass",
        target_name=direct_name,
    )
    wp_arch, wp_symbol, wp_section = _target_section(
        ptx,
        kind="ptx",
        target_name=wmma_name,
    )
    ws_arch, ws_symbol, ws_section = _target_section(
        sass,
        kind="sass",
        target_name=wmma_name,
    )
    direct_ptx = [
        line.strip()
        for line in dp_section
        if "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32" in line
    ]
    direct_sass = [
        line.strip() for line in ds_section if "HMMA.1688.F32" in line
    ]
    wmma_ptx = [
        line.strip()
        for line in wp_section
        if "wmma.mma.sync.aligned" in line and "m16n16k16" in line
    ]
    wmma_sass = [
        line.strip() for line in ws_section if "HMMA.1688.F32" in line
    ]
    return {
        "effective_ptx_architecture": dp_arch,
        "effective_cubin_architecture": ds_arch,
        "direct_ptx_target_symbol": dp_symbol,
        "direct_sass_target_symbol": ds_symbol,
        "direct_ptx_m16n8k8_count": len(direct_ptx),
        "direct_sass_hmma_1688_count": len(direct_sass),
        "direct_ptx_matching_lines": direct_ptx,
        "direct_sass_matching_lines": direct_sass,
        "wmma_ptx_target_symbol": wp_symbol,
        "wmma_sass_target_symbol": ws_symbol,
        "wmma_ptx_m16n16k16_count": len(wmma_ptx),
        "wmma_sass_hmma_1688_count": len(wmma_sass),
        "wmma_ptx_matching_lines": wmma_ptx,
        "wmma_sass_matching_lines": wmma_sass,
        "wmma_effective_ptx_architecture": wp_arch,
        "wmma_effective_cubin_architecture": ws_arch,
    }


def validate_effective_route(route: Any) -> dict[str, Any]:
    if not isinstance(route, dict):
        raise ValueError("backend effective route must be an object")
    expected = {
        "backend": "cuda",
        "device": EXPECTED_DEVICE,
        "kernel": KERNEL_IDENTITY,
        "effective_compute_capability": "7.5",
    }
    for name, value in expected.items():
        if route.get(name) != value:
            raise ValueError(
                f"effective route {name} must be {value!r}, "
                f"got {route.get(name)!r}"
            )
    evidence = route.get("compiler_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("effective route compiler_evidence must be an object")
    for name in (
        "effective_ptx_architecture",
        "effective_cubin_architecture",
        "wmma_effective_ptx_architecture",
        "wmma_effective_cubin_architecture",
    ):
        if evidence.get(name) != "sm_75":
            raise ValueError(f"compiler evidence {name} must be sm_75")
    direct_ptx = evidence.get("direct_ptx_target_symbol")
    direct_sass = evidence.get("direct_sass_target_symbol")
    if (
        not isinstance(direct_ptx, str)
        or "sm75_hmma_m16n8k8_stage_kernel" not in direct_ptx
        or direct_ptx != direct_sass
    ):
        raise ValueError("direct PTX and SASS target symbols are invalid")
    if (
        evidence.get("direct_ptx_m16n8k8_count") != 1
        or evidence.get("direct_sass_hmma_1688_count") != 1
    ):
        raise ValueError("direct target must contain exactly one HMMA.1688")
    wmma_ptx = evidence.get("wmma_ptx_target_symbol")
    wmma_sass = evidence.get("wmma_sass_target_symbol")
    if (
        not isinstance(wmma_ptx, str)
        or "sm75_wmma_real_lattice_kernel" not in wmma_ptx
        or wmma_ptx != wmma_sass
    ):
        raise ValueError("WMMA PTX and SASS target symbols are invalid")
    if (
        evidence.get("wmma_ptx_m16n16k16_count") != 1
        or evidence.get("wmma_sass_hmma_1688_count") != 4
    ):
        raise ValueError("WMMA target must lower to exactly four HMMA.1688")
    return route


def _build_extension():
    from torch.utils.cpp_extension import load_inline

    previous_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5+PTX"
    try:
        return load_inline(
            name="trellis2mlx_sm75_hmma_real_tile_lattice_v1",
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=["sm75_hmma_real_tile_lattice_cuda"],
            extra_cuda_cflags=["-O3", "--fmad=true"],
            with_cuda=True,
            verbose=True,
        )
    finally:
        if previous_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch_list


def _collect_compiler_evidence(module: Any) -> dict[str, Any]:
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        raise RuntimeError("cuobjdump is unavailable")
    module_path = Path(module.__file__)
    outputs: dict[str, str] = {}
    for name, flag in (("ptx", "--dump-ptx"), ("sass", "--dump-sass")):
        completed = subprocess.run(
            [cuobjdump, flag, str(module_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"cuobjdump {name} failed: "
                + (completed.stdout + completed.stderr).strip()
            )
        outputs[name] = completed.stdout
    result = classify_compiler_evidence(
        ptx=outputs["ptx"],
        sass=outputs["sass"],
    )
    result.update(
        {
            "requested_architecture": "sm_75",
            "torch_cuda_arch_list": "7.5+PTX",
            "extension_sha256": sha256_file(module_path),
            "ptx_sha256": _sha256_text(outputs["ptx"]),
            "sass_sha256": _sha256_text(outputs["sass"]),
        }
    )
    return result


def run_cuda_backend(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    import torch

    if torch.__version__ != EXPECTED_TORCH:
        raise RuntimeError(
            f"expected Torch {EXPECTED_TORCH}, got {torch.__version__}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.cuda.get_device_name(0)
    if device != EXPECTED_DEVICE:
        raise RuntimeError(f"expected {EXPECTED_DEVICE}, got {device}")
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
    if capability != (7, 5):
        raise RuntimeError(
            f"expected compute capability (7, 5), got {capability}"
        )
    module = _build_extension()
    compiler_evidence = _collect_compiler_evidence(module)
    matrix_a_cuda = torch.from_numpy(matrix_a).to(device="cuda").contiguous()
    matrix_b_cuda = torch.from_numpy(matrix_b).to(device="cuda").contiguous()
    torch.cuda.synchronize()
    started = time.perf_counter()
    stage_zero, stage_one, wmma_output = (
        module.sm75_hmma_real_tile_lattice_cuda(
            matrix_a_cuda,
            matrix_b_cuda,
        )
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return (
        stage_zero.detach().cpu().numpy(),
        stage_one.detach().cpu().numpy(),
        wmma_output.detach().cpu().numpy(),
        {
            "backend": "cuda",
            "device": device,
            "kernel": KERNEL_IDENTITY,
            "effective_compute_capability": "7.5",
            "elapsed_seconds": elapsed,
            "compiler_evidence": compiler_evidence,
        },
    )


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
    prefix_result_path: Path,
    expected_witness_sha256: str,
    expected_cuda_result_sha256: str,
    expected_prefix_result_sha256: str,
    output_json: Path,
    output_npz: Path,
    masks: np.ndarray | None = None,
    input_loader: Callable[..., dict[str, Any]] = load_probe_inputs,
    backend: Callable[
        [np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]],
    ] = run_cuda_backend,
    authoritative: bool = True,
) -> dict[str, Any]:
    witness_path = Path(witness_path)
    cuda_result_path = Path(cuda_result_path)
    prefix_result_path = Path(prefix_result_path)
    output_json = Path(output_json)
    output_npz = Path(output_npz)
    named_paths = {
        "witness_path": witness_path.resolve(),
        "cuda_result_path": cuda_result_path.resolve(),
        "prefix_result_path": prefix_result_path.resolve(),
        "output_json": output_json.resolve(),
        "output_npz": output_npz.resolve(),
    }
    collisions: dict[Path, list[str]] = {}
    for name, path in named_paths.items():
        collisions.setdefault(path, []).append(name)
    duplicates = {
        str(path): names
        for path, names in collisions.items()
        if len(names) > 1
    }
    if duplicates:
        raise ValueError(f"input and output paths collide: {duplicates}")
    output_json.unlink(missing_ok=True)
    output_npz.unlink(missing_ok=True)
    schema = SCHEMA if authoritative else TEST_SCHEMA
    phase = "input_validation"
    try:
        masks = complete_masks() if masks is None else _validate_masks(masks)
        if authoritative:
            masks = _validate_complete_masks(masks)
        inputs = input_loader(
            witness_path,
            cuda_result_path,
            prefix_result_path,
            expected_witness_sha256=expected_witness_sha256,
            expected_cuda_result_sha256=expected_cuda_result_sha256,
            expected_prefix_result_sha256=expected_prefix_result_sha256,
        )
        matrix_a, matrix_b = generate_cases(
            inputs["base_a"],
            inputs["base_b"],
            masks,
        )
        phase = "backend_execution"
        direct_stage0, direct_stage1, wmma_output, route = backend(
            matrix_a,
            matrix_b,
        )
        phase = "backend_output_validation"
        route = validate_effective_route(route)
        analysis = analyze_outputs(
            masks,
            direct_stage0,
            direct_stage1,
            wmma_output,
            expected_full_wmma=inputs["expected_full_wmma"],
        )
        phase = "primary_publication"
        _write_npz_atomic(
            output_npz,
            subset_masks=masks,
            direct_stage0_fp32=direct_stage0,
            direct_stage1_fp32=direct_stage1,
            wmma_fp32=wmma_output,
        )
        report: dict[str, Any] = {
            "schema": schema,
            "status": "done" if authoritative else "test_done",
            "failure_phase": None,
            "effective_route": route,
            "input_custody": {
                "witness_sha256": inputs["witness_sha256"],
                "cuda_result_sha256": inputs["cuda_result_sha256"],
                "prefix_result_sha256": inputs["prefix_result_sha256"],
                "tile_column": TILE_COLUMN,
            },
            "artifacts": {
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
            "schema": schema,
            "status": "failed",
            "failure_phase": phase,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
        }
        _write_json_atomic(output_json, failure)
        raise
    except BaseException as error:
        output_npz.unlink(missing_ok=True)
        interruption = {
            "schema": schema,
            "status": "failed",
            "failure_phase": phase,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "hard_interruption": True,
        }
        try:
            _write_json_atomic(output_json, interruption)
        except BaseException:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witness", type=Path, required=True)
    parser.add_argument("--cuda-result", type=Path, required=True)
    parser.add_argument("--prefix-result", type=Path, required=True)
    parser.add_argument("--expected-witness-sha256", required=True)
    parser.add_argument("--expected-cuda-result-sha256", required=True)
    parser.add_argument("--expected-prefix-result-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = run_probe(
        witness_path=args.witness,
        cuda_result_path=args.cuda_result,
        prefix_result_path=args.prefix_result,
        expected_witness_sha256=args.expected_witness_sha256,
        expected_cuda_result_sha256=args.expected_cuda_result_sha256,
        expected_prefix_result_sha256=args.expected_prefix_result_sha256,
        output_json=args.output_json,
        output_npz=args.output_npz,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
