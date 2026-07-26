#!/usr/bin/env python3
"""Expose intermediates from the authenticated PyTorch CUDA softmax schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback
from typing import Any

import numpy as np


SCHEMA = "trellis2mlx.cuda_softmax_intermediate_witness.v1"
EXPECTED_TORCH = "2.10.0+cu128"
EXPECTED_DEVICE = "Tesla T4"
EXPECTED_ROWS = 3822
EXPECTED_WIDTH = 7697
THREADS = 1024
WARPS = THREADS // 32


CPP_SOURCE = r"""
#include <torch/extension.h>

std::vector<torch::Tensor> softmax_intermediates_cuda(torch::Tensor scores);
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cmath>
#include <limits>

namespace {

constexpr int kClasses = 7697;
constexpr int kThreads = 1024;
constexpr int kWarps = 32;
constexpr int kRegisters = 8;

__device__ __forceinline__ float max_op(float a, float b) {
  return a < b ? b : a;
}

__device__ __forceinline__ float warp_reduce_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = max_op(
        value, __shfl_down_sync(0xffffffff, value, offset, 32));
  }
  return value;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  return value;
}

__device__ __forceinline__ float block_reduce_max(
    float value, float* shared) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  value = warp_reduce_max(value);
  __syncthreads();
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();
  value = threadIdx.x < kWarps
      ? shared[lane]
      : -std::numeric_limits<float>::max();
  if (warp == 0) {
    value = warp_reduce_max(value);
  }
  if (threadIdx.x == 0) {
    shared[0] = value;
  }
  __syncthreads();
  return shared[0];
}

__device__ __forceinline__ float block_reduce_sum(
    float value, float* shared, float* warp_sums, int row) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  value = warp_reduce_sum(value);
  __syncthreads();
  if (lane == 0) {
    shared[warp] = value;
    warp_sums[row * kWarps + warp] = value;
  }
  __syncthreads();
  value = threadIdx.x < kWarps ? shared[lane] : 0.0f;
  if (warp == 0) {
    value = warp_reduce_sum(value);
  }
  if (threadIdx.x == 0) {
    shared[0] = value;
  }
  __syncthreads();
  return shared[0];
}

__global__ void softmax_intermediates_kernel(
    const float* __restrict__ scores,
    float* __restrict__ custom_probs,
    float* __restrict__ captured_probs,
    float* __restrict__ exponents,
    float* __restrict__ thread_sums,
    float* __restrict__ warp_sums,
    float* __restrict__ row_maxes,
    float* __restrict__ row_sums) {
  extern __shared__ float shared[];
  const int row = blockIdx.x;
  const int row_offset = row * kClasses;
  float registers[kRegisters];
  float thread_max = -std::numeric_limits<float>::max();

#pragma unroll
  for (int reg = 0; reg < kRegisters; ++reg) {
    const int offset = threadIdx.x + reg * blockDim.x;
    if (offset < kClasses) {
      registers[reg] = scores[row_offset + offset];
      thread_max = max_op(thread_max, registers[reg]);
    }
  }

  const float row_max = block_reduce_max(thread_max, shared);
  float thread_sum = 0.0f;
#pragma unroll
  for (int reg = 0; reg < kRegisters; ++reg) {
    const int offset = threadIdx.x + reg * blockDim.x;
    if (offset < kClasses) {
      const float exponent = std::exp(registers[reg] - row_max);
      exponents[row_offset + offset] = exponent;
      thread_sum = thread_sum + exponent;
    }
  }
  thread_sums[row * kThreads + threadIdx.x] = thread_sum;
  const float row_sum = block_reduce_sum(
      thread_sum, shared, warp_sums, row);

  __syncthreads();
#pragma unroll
  for (int reg = 0; reg < kRegisters; ++reg) {
    const int offset = threadIdx.x + reg * blockDim.x;
    if (offset < kClasses) {
      custom_probs[row_offset + offset] =
          std::exp(registers[reg] - row_max) / row_sum;
      volatile float* captured_exponents = exponents;
      captured_probs[row_offset + offset] =
          captured_exponents[row_offset + offset] / row_sum;
    }
  }
  if (threadIdx.x == 0) {
    row_maxes[row] = row_max;
    row_sums[row] = row_sum;
  }
}

}  // namespace

std::vector<torch::Tensor> softmax_intermediates_cuda(
    torch::Tensor scores) {
  TORCH_CHECK(scores.is_cuda(), "scores must be a CUDA tensor");
  TORCH_CHECK(
      scores.scalar_type() == torch::kFloat32,
      "scores must use float32");
  TORCH_CHECK(scores.is_contiguous(), "scores must be contiguous");
  TORCH_CHECK(
      scores.dim() == 2 && scores.size(0) == 3822
          && scores.size(1) == kClasses,
      "scores must have shape [3822, 7697]");

  auto custom_probs = torch::empty_like(scores);
  auto captured_probs = torch::empty_like(scores);
  auto exponents = torch::empty_like(scores);
  auto thread_sums = torch::empty(
      {scores.size(0), kThreads}, scores.options());
  auto warp_sums = torch::empty(
      {scores.size(0), kWarps}, scores.options());
  auto row_maxes = torch::empty({scores.size(0)}, scores.options());
  auto row_sums = torch::empty({scores.size(0)}, scores.options());

  auto stream = at::cuda::getCurrentCUDAStream();
  softmax_intermediates_kernel<<<
      static_cast<unsigned int>(scores.size(0)),
      kThreads,
      kWarps * sizeof(float),
      stream>>>(
      scores.data_ptr<float>(),
      custom_probs.data_ptr<float>(),
      captured_probs.data_ptr<float>(),
      exponents.data_ptr<float>(),
      thread_sums.data_ptr<float>(),
      warp_sums.data_ptr<float>(),
      row_maxes.data_ptr<float>(),
      row_sums.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {
      custom_probs,
      captured_probs,
      exponents,
      thread_sums,
      warp_sums,
      row_maxes,
      row_sums};
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


def requested_stage_identity(expected_sha256: str | None) -> dict[str, Any]:
    if (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise ValueError(
            "expected stage sha256 must be 64 lowercase hexadecimal characters"
        )
    return {
        "sha256": expected_sha256,
        "rows": EXPECTED_ROWS,
        "width": EXPECTED_WIDTH,
        "dtype": "float32",
    }


def _require_array(
    name: str,
    value: np.ndarray,
    shape: tuple[int, ...],
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(
            f"{name} shape must be {shape}, got {array.shape}"
        )
    if array.dtype != np.float32:
        raise ValueError(f"{name} must use float32, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _warp_reduce_sum(values: np.ndarray) -> np.ndarray:
    work = np.asarray(values, dtype=np.float32).copy()
    if work.shape[-1] != 32:
        raise ValueError("warp reduction requires 32 lanes")
    for offset in (16, 8, 4, 2, 1):
        work[..., : 32 - offset] = np.float32(
            work[..., : 32 - offset] + work[..., offset:]
        )
    return work[..., 0]


def analyze_softmax_oracle(
    *,
    scores: np.ndarray,
    persisted_probs: np.ndarray,
    live_native_probs: np.ndarray,
    custom_probs: np.ndarray,
    captured_probs: np.ndarray,
    exponents: np.ndarray,
    thread_sums: np.ndarray,
    warp_sums: np.ndarray,
    row_maxes: np.ndarray,
    row_sums: np.ndarray,
    expected_shape: tuple[int, int] = (EXPECTED_ROWS, EXPECTED_WIDTH),
    expected_threads: int = THREADS,
    expected_warps: int = WARPS,
) -> dict[str, Any]:
    rows, width = expected_shape
    arrays = {
        "scores": _require_array("scores", scores, expected_shape),
        "persisted_probs": _require_array(
            "persisted_probs", persisted_probs, expected_shape
        ),
        "live_native_probs": _require_array(
            "live_native_probs", live_native_probs, expected_shape
        ),
        "custom_probs": _require_array(
            "custom_probs", custom_probs, expected_shape
        ),
        "captured_probs": _require_array(
            "captured_probs", captured_probs, expected_shape
        ),
        "exponents": _require_array(
            "exponents", exponents, expected_shape
        ),
        "thread_sums": _require_array(
            "thread_sums", thread_sums, (rows, expected_threads)
        ),
        "warp_sums": _require_array(
            "warp_sums", warp_sums, (rows, expected_warps)
        ),
        "row_maxes": _require_array(
            "row_maxes", row_maxes, (rows,)
        ),
        "row_sums": _require_array("row_sums", row_sums, (rows,)),
    }
    if expected_threads != expected_warps * 32:
        raise ValueError("thread and warp geometry are inconsistent")
    if not np.array_equal(
        arrays["persisted_probs"], arrays["live_native_probs"]
    ):
        raise ValueError(
            "persisted probabilities do not reproduce live native softmax"
        )
    if not np.array_equal(
        arrays["custom_probs"], arrays["live_native_probs"]
    ):
        raise ValueError(
            "custom softmax does not reproduce live native softmax"
        )
    if not np.array_equal(
        arrays["captured_probs"], arrays["custom_probs"]
    ):
        raise ValueError(
            "captured exponent positions do not reproduce probabilities"
        )

    expected_maxes = np.max(arrays["scores"], axis=1)
    if not np.array_equal(arrays["row_maxes"], expected_maxes):
        raise ValueError("captured row maxima do not match scores")
    maxima = np.argmax(arrays["scores"], axis=1)
    if not np.array_equal(
        arrays["exponents"][np.arange(rows), maxima],
        np.ones((rows,), dtype=np.float32),
    ):
        raise ValueError("maximum-score exponent is not exactly one")

    reconstructed_threads = np.zeros(
        (rows, expected_threads), dtype=np.float32
    )
    for register in range((width + expected_threads - 1) // expected_threads):
        start = register * expected_threads
        count = min(expected_threads, width - start)
        reconstructed_threads[:, :count] = np.float32(
            reconstructed_threads[:, :count]
            + arrays["exponents"][:, start : start + count]
        )
    if not np.array_equal(
        arrays["thread_sums"], reconstructed_threads
    ):
        raise ValueError("thread sums do not match captured exponents")

    reconstructed_warps = _warp_reduce_sum(
        reconstructed_threads.reshape(rows, expected_warps, 32)
    )
    if not np.array_equal(arrays["warp_sums"], reconstructed_warps):
        raise ValueError("warp sums do not match source reduction tree")
    second_level_lanes = np.zeros((rows, 32), dtype=np.float32)
    second_level_lanes[:, :expected_warps] = reconstructed_warps
    reconstructed_rows = _warp_reduce_sum(second_level_lanes)
    if not np.array_equal(arrays["row_sums"], reconstructed_rows):
        raise ValueError("row sums do not match source reduction tree")
    if np.any(arrays["row_sums"] <= 0.0):
        raise ValueError("row sums must be positive")
    reconstructed_probs = np.float32(
        arrays["exponents"] / arrays["row_sums"][:, None]
    )
    if not np.array_equal(
        reconstructed_probs, arrays["captured_probs"]
    ):
        raise ValueError(
            "captured exponent positions do not reproduce probabilities"
        )

    return {
        "self_authentication": {
            "persisted_native_exact": True,
            "custom_native_exact": True,
            "captured_exponent_positions_exact": True,
            "row_max_exact": True,
            "thread_sum_tree_exact": True,
            "warp_sum_tree_exact": True,
            "row_sum_tree_exact": True,
        },
        "shape": list(expected_shape),
        "threads": expected_threads,
        "warps": expected_warps,
        "array_sha256": {
            name: sha256_array(value) for name, value in arrays.items()
        },
    }


def _build_extension() -> Any:
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="trellis2mlx_cuda_softmax_intermediates_v1",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=["softmax_intermediates_cuda"],
        extra_cuda_cflags=["-O3", "--fmad=true"],
        with_cuda=True,
        verbose=True,
    )


def _nvcc_version() -> str:
    try:
        completed = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"
    text = (completed.stdout + completed.stderr).strip()
    return text or f"nvcc exited {completed.returncode} without output"


def _write_report(path: Path, report: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _same_path(first: Path, second: Path) -> bool:
    if first.resolve() == second.resolve():
        return True
    try:
        return os.path.samefile(first, second)
    except FileNotFoundError:
        return False
    except OSError:
        # Refuse the write when existing-path identity cannot be established.
        return True


def _safe_failure_report_path(
    requested: Path,
    protected: tuple[Path, ...],
) -> Path:
    if not any(_same_path(requested, path) for path in protected):
        return requested
    candidate = protected[0].with_name(
        protected[0].name + ".softmax-oracle-failure.json"
    )
    while any(_same_path(candidate, path) for path in protected):
        candidate = candidate.with_name(candidate.name + ".failure.json")
    return candidate


def _validate_written_primary(
    path: Path,
    *,
    expected_arrays: dict[str, np.ndarray],
    expected_route_identity_json: str,
) -> None:
    if not path.is_file():
        raise ValueError("written primary is missing")
    with np.load(path, allow_pickle=False) as loaded:
        expected_keys = {
            *expected_arrays,
            "route_identity_json",
        }
        if set(loaded.files) != expected_keys:
            raise ValueError(
                "written primary keys mismatch: "
                f"expected {sorted(expected_keys)}, got {sorted(loaded.files)}"
            )
        for name, expected in expected_arrays.items():
            actual = np.asarray(loaded[name])
            if actual.dtype != expected.dtype or actual.shape != expected.shape:
                raise ValueError(f"written primary {name} metadata mismatch")
            if not np.array_equal(actual, expected):
                raise ValueError(f"written primary {name} values mismatch")
        route_identity_json = str(loaded["route_identity_json"].item())
        if route_identity_json != expected_route_identity_json:
            raise ValueError("written primary route identity mismatch")
        json.loads(route_identity_json)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-npz", required=True)
    parser.add_argument("--expected-stage-sha256", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-npz", required=True)
    args = parser.parse_args(argv)

    started = time.time()
    stage_path = Path(args.stage_npz)
    requested_output_json = Path(args.output_json)
    output_npz = Path(args.output_npz)
    output_json = _safe_failure_report_path(
        requested_output_json,
        (stage_path, output_npz),
    )
    path_collisions = []
    if _same_path(stage_path, requested_output_json):
        path_collisions.append("output JSON aliases protected stage NPZ")
    if _same_path(stage_path, output_npz):
        path_collisions.append("output NPZ aliases protected stage NPZ")
    if _same_path(requested_output_json, output_npz):
        path_collisions.append("output JSON aliases output NPZ")
    primary_output_status = (
        "protected_input"
        if _same_path(stage_path, output_npz)
        else ("preexisting_untrusted" if output_npz.exists() else "missing")
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "failure_phase": "request_validation",
        "last_trustworthy_phase": "request_received",
        "primary_output_status": primary_output_status,
        "stage_npz": str(stage_path),
        "output_json_requested": str(requested_output_json),
        "output_json_effective": str(output_json),
        "output_npz": str(output_npz),
    }
    try:
        if path_collisions:
            raise ValueError("; ".join(path_collisions))
        requested = requested_stage_identity(args.expected_stage_sha256)
        report["stage_identity_requested"] = requested
        if output_npz.exists():
            output_npz.unlink()
        report["primary_output_status"] = "missing"
        report["last_trustworthy_phase"] = "request_validated"

        report["failure_phase"] = "stage_load"
        if not stage_path.is_file():
            raise FileNotFoundError(f"stage NPZ does not exist: {stage_path}")
        effective_sha = sha256_file(stage_path)
        if effective_sha != requested["sha256"]:
            raise ValueError(
                "stage sha256 mismatch: "
                f"expected {requested['sha256']}, got {effective_sha}"
            )
        with np.load(stage_path, allow_pickle=False) as loaded:
            required = {
                "row_tokens",
                "row_heads",
                "scores_fp32",
                "probs_fp32",
                "route_identity_json",
            }
            missing = sorted(required - set(loaded.files))
            if missing:
                raise ValueError(f"stage NPZ is missing arrays: {missing}")
            row_tokens = np.asarray(loaded["row_tokens"], dtype=np.int32)
            row_heads = np.asarray(loaded["row_heads"], dtype=np.int32)
            scores = _require_array(
                "scores_fp32",
                loaded["scores_fp32"],
                (EXPECTED_ROWS, EXPECTED_WIDTH),
            ).copy()
            persisted_probs = _require_array(
                "probs_fp32",
                loaded["probs_fp32"],
                (EXPECTED_ROWS, EXPECTED_WIDTH),
            ).copy()
            route_identity_json = str(loaded["route_identity_json"].item())
        if row_tokens.shape != (EXPECTED_ROWS,):
            raise ValueError("row_tokens shape mismatch")
        if row_heads.shape != (EXPECTED_ROWS,):
            raise ValueError("row_heads shape mismatch")
        json.loads(route_identity_json)
        report["stage_identity_effective"] = {
            "sha256": effective_sha,
            "rows": int(scores.shape[0]),
            "width": int(scores.shape[1]),
            "dtype": str(scores.dtype),
            "route_identity_json_sha256": hashlib.sha256(
                route_identity_json.encode("utf-8")
            ).hexdigest(),
        }
        report["last_trustworthy_phase"] = "stage_loaded"

        report["failure_phase"] = "runtime_validation"
        import torch

        if torch.__version__ != EXPECTED_TORCH:
            raise RuntimeError(
                f"expected Torch {EXPECTED_TORCH}, got {torch.__version__}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        device_name = torch.cuda.get_device_name(0)
        if device_name != EXPECTED_DEVICE:
            raise RuntimeError(
                f"expected CUDA device {EXPECTED_DEVICE}, got {device_name}"
            )
        report["runtime"] = {
            "torch": torch.__version__,
            "device": device_name,
            "cuda": torch.version.cuda,
            "nvcc": _nvcc_version(),
        }
        report["implementation_identity"] = {
            "script_sha256": sha256_file(Path(__file__)),
            "cuda_source_sha256": hashlib.sha256(
                CUDA_SOURCE.encode("utf-8")
            ).hexdigest(),
            "compile_flags": ["-O3", "--fmad=true"],
        }
        report["last_trustworthy_phase"] = "runtime_validated"

        report["failure_phase"] = "extension_build"
        extension = _build_extension()
        report["last_trustworthy_phase"] = "extension_built"

        report["failure_phase"] = "oracle_execution"
        scores_cuda = torch.from_numpy(scores).to(
            device="cuda", dtype=torch.float32
        )
        live_native_tensor = torch.softmax(scores_cuda, dim=-1)
        outputs = extension.softmax_intermediates_cuda(scores_cuda)
        torch.cuda.synchronize()
        arrays = [
            tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            for tensor in outputs
        ]
        live_native = (
            live_native_tensor.detach().cpu().numpy().astype(
                np.float32, copy=False
            )
        )
        (
            custom_probs,
            captured_probs,
            exponents,
            thread_sums,
            warp_sums,
            row_maxes,
            row_sums,
        ) = arrays
        report["last_trustworthy_phase"] = "oracle_executed"

        report["failure_phase"] = "self_authentication"
        analysis = analyze_softmax_oracle(
            scores=scores,
            persisted_probs=persisted_probs,
            live_native_probs=live_native,
            custom_probs=custom_probs,
            captured_probs=captured_probs,
            exponents=exponents,
            thread_sums=thread_sums,
            warp_sums=warp_sums,
            row_maxes=row_maxes,
            row_sums=row_sums,
        )
        report["analysis"] = analysis
        report["last_trustworthy_phase"] = "self_authenticated"

        report["failure_phase"] = "primary_write"
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        route_identity_json = json.dumps(
            {
                "schema": SCHEMA,
                "stage_identity": report["stage_identity_effective"],
                "runtime": report["runtime"],
                "implementation_identity": report[
                    "implementation_identity"
                ],
                "schedule": {
                    "classes": EXPECTED_WIDTH,
                    "threads": THREADS,
                    "warps": WARPS,
                    "registers": 8,
                    "exp": "std::exp",
                    "reduction": "two-level-warp-shuffle-down",
                    "normalization": "division",
                },
                "self_authentication": analysis["self_authentication"],
            },
            sort_keys=True,
        )
        primary_arrays = {
            "row_tokens": row_tokens,
            "row_heads": row_heads,
            "exponents_fp32": exponents,
            "thread_sums_fp32": thread_sums,
            "warp_sums_fp32": warp_sums,
            "row_maxes_fp32": row_maxes,
            "row_sums_fp32": row_sums,
        }
        np.savez_compressed(
            output_npz,
            **primary_arrays,
            route_identity_json=np.array(route_identity_json),
        )
        report["primary_output_status"] = "written_unverified"
        _validate_written_primary(
            output_npz,
            expected_arrays=primary_arrays,
            expected_route_identity_json=route_identity_json,
        )
        report["primary_output_status"] = "written"
        report["primary_output"] = {
            "sha256": sha256_file(output_npz),
            "size": output_npz.stat().st_size,
        }
        report["status"] = "done"
        report["failure_phase"] = None
        report["last_trustworthy_phase"] = "primary_rehashed"
        return 0
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        if (
            output_npz.exists()
            and report["primary_output_status"] != "protected_input"
        ):
            report["primary_output_status"] = "partial_unverified"
        return 1
    finally:
        report["elapsed_seconds"] = time.time() - started
        _write_report(output_json, report)


if __name__ == "__main__":
    raise SystemExit(main())
