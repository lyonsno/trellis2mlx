#!/usr/bin/env python3
"""Expose the variance inside an authenticated CUDA LayerNorm schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import traceback
from typing import Any

import numpy as np


SCHEMA = "trellis2mlx.cuda_welford_variance_witness.v1"
EXPECTED_TORCH = "2.10.0+cu128"
EXPECTED_DEVICE = "Tesla T4"
WIDTH = 1536
EXPECTED_WITNESS_SHAPE = (1, 4096, WIDTH)
EXPECTED_EPS_FLOAT32_BITS = "0x358637bd"


CPP_SOURCE = r"""
#include <torch/extension.h>

std::vector<torch::Tensor> welford_variance_cuda(
    torch::Tensor input,
    double eps);
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kWidth = 1536;
constexpr int kVectorWidth = 4;

struct WelfordDataLN {
  float mean;
  float sigma2;
  float count;
};

__device__ __forceinline__ WelfordDataLN welford_online(
    float value,
    const WelfordDataLN& current) {
  float delta = value - current.mean;
  float new_count = current.count + 1.0f;
  auto reciprocal_multiply = [](float a, float b) {
    return a * (1.0f / b);
  };
  float new_mean =
      current.mean + reciprocal_multiply(delta, new_count);
  return {
      new_mean,
      current.sigma2 + delta * (value - new_mean),
      new_count};
}

__device__ __forceinline__ WelfordDataLN welford_combine(
    WelfordDataLN data_b,
    WelfordDataLN data_a) {
  float delta = data_b.mean - data_a.mean;
  float count = data_a.count + data_b.count;
  float mean;
  float sigma2;
  if (count > 0.0f) {
    float coefficient = 1.0f / count;
    float n_a = data_a.count * coefficient;
    float n_b = data_b.count * coefficient;
    mean = n_a * data_a.mean + n_b * data_b.mean;
    sigma2 = data_a.sigma2 + data_b.sigma2
        + delta * delta * data_a.count * n_b;
  } else {
    mean = 0.0f;
    sigma2 = 0.0f;
  }
  return {mean, sigma2, count};
}

__global__ void welford_variance_kernel(
    const __nv_bfloat16* __restrict__ input,
    float eps,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ mean,
    float* __restrict__ variance,
    float* __restrict__ rstd) {
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int thread_index = lane + warp * blockDim.x;
  const int row = blockIdx.x;
  const int row_offset = row * kWidth;

  WelfordDataLN wd{0.0f, 0.0f, 0.0f};
  for (int vector_index = thread_index;
       vector_index < kWidth / kVectorWidth;
       vector_index += blockDim.x * blockDim.y) {
    const int offset = row_offset + vector_index * kVectorWidth;
#pragma unroll
    for (int element = 0; element < kVectorWidth; ++element) {
      wd = welford_online(
          __bfloat162float(input[offset + element]), wd);
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    WelfordDataLN upper{
        __shfl_down_sync(0xffffffff, wd.mean, offset),
        __shfl_down_sync(0xffffffff, wd.sigma2, offset),
        __shfl_down_sync(0xffffffff, wd.count, offset)};
    wd = welford_combine(wd, upper);
  }

  extern __shared__ float shared[];
  float* mean_sigma = shared;
  float* counts = shared + blockDim.y;
  for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
    if (lane == 0 && warp >= offset && warp < 2 * offset) {
      const int slot = warp - offset;
      mean_sigma[2 * slot] = wd.mean;
      mean_sigma[2 * slot + 1] = wd.sigma2;
      counts[slot] = wd.count;
    }
    __syncthreads();
    if (lane == 0 && warp < offset) {
      WelfordDataLN upper{
          mean_sigma[2 * warp],
          mean_sigma[2 * warp + 1],
          counts[warp]};
      wd = welford_combine(wd, upper);
    }
    __syncthreads();
  }
  if (lane == 0 && warp == 0) {
    mean_sigma[0] = wd.mean;
    mean_sigma[1] = wd.sigma2 / static_cast<float>(kWidth);
  }
  __syncthreads();

  const float row_mean = mean_sigma[0];
  const float row_variance = mean_sigma[1];
  const float row_rstd = rsqrtf(row_variance + eps);
  for (int vector_index = thread_index;
       vector_index < kWidth / kVectorWidth;
       vector_index += blockDim.x * blockDim.y) {
    const int offset = row_offset + vector_index * kVectorWidth;
#pragma unroll
    for (int element = 0; element < kVectorWidth; ++element) {
      const float value = __bfloat162float(input[offset + element]);
      output[offset + element] =
          __float2bfloat16(row_rstd * (value - row_mean));
    }
  }
  if (thread_index == 0) {
    mean[row] = row_mean;
    variance[row] = row_variance;
    rstd[row] = row_rstd;
  }
}

}  // namespace

std::vector<torch::Tensor> welford_variance_cuda(
    torch::Tensor input,
    double eps) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16,
              "input must be bfloat16");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.dim() >= 1 && input.size(-1) == kWidth,
              "input hidden width must be 1536");

  const auto rows = input.numel() / kWidth;
  auto output = torch::empty_like(input);
  std::vector<int64_t> stats_shape(input.sizes().begin(), input.sizes().end());
  stats_shape.back() = 1;
  auto stats_options = input.options().dtype(torch::kFloat32);
  auto mean = torch::empty(stats_shape, stats_options);
  auto variance = torch::empty(stats_shape, stats_options);
  auto rstd = torch::empty(stats_shape, stats_options);

  const dim3 threads(32, 4, 1);
  const dim3 blocks(static_cast<unsigned int>(rows), 1, 1);
  const size_t shared_bytes = 6 * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();
  welford_variance_kernel<<<blocks, threads, shared_bytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(
          input.data_ptr<at::BFloat16>()),
      static_cast<float>(eps),
      reinterpret_cast<__nv_bfloat16*>(
          output.data_ptr<at::BFloat16>()),
      mean.data_ptr<float>(),
      variance.data_ptr<float>(),
      rstd.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, mean, variance, rstd};
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


def _float32_bits(value: np.float32) -> str:
    bits = int(np.asarray(np.float32(value)).view(np.uint32))
    return f"0x{bits:08x}"


def requested_witness_identity(expected_sha256: str | None) -> dict[str, Any]:
    if (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise ValueError(
            "expected witness sha256 must be 64 lowercase hexadecimal "
            "characters"
        )
    return {
        "sha256": expected_sha256,
        "input_shape": list(EXPECTED_WITNESS_SHAPE),
        "reference_shape": list(EXPECTED_WITNESS_SHAPE),
        "eps_float32_bits": EXPECTED_EPS_FLOAT32_BITS,
    }


def effective_witness_identity(
    witness_path: Path,
    input_array: np.ndarray,
    reference_array: np.ndarray,
    eps: np.float32,
) -> dict[str, Any]:
    return {
        "sha256": sha256_file(witness_path),
        "input_shape": list(np.asarray(input_array).shape),
        "reference_shape": list(np.asarray(reference_array).shape),
        "eps_float32_bits": _float32_bits(eps),
    }


def validate_witness_identity(
    requested: dict[str, Any], effective: dict[str, Any]
) -> None:
    comparisons = (
        ("sha256", "witness sha256 mismatch"),
        ("input_shape", "witness input shape mismatch"),
        ("reference_shape", "witness reference shape mismatch"),
        ("eps_float32_bits", "witness epsilon mismatch"),
    )
    for field, message in comparisons:
        if effective.get(field) != requested.get(field):
            raise ValueError(
                f"{message}: expected {requested.get(field)!r}, "
                f"got {effective.get(field)!r}"
            )


def _exact_metric(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    candidate = np.asarray(candidate, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if candidate.shape != reference.shape:
        return {
            "shape_match": False,
            "candidate_shape": list(candidate.shape),
            "reference_shape": list(reference.shape),
            "exact": False,
        }
    difference = np.abs(
        candidate.astype(np.float64) - reference.astype(np.float64)
    )
    return {
        "shape_match": True,
        "shape": list(candidate.shape),
        "mean_abs": float(difference.mean(dtype=np.float64)),
        "max_abs": float(difference.max(initial=0.0)),
        "nonzero": int(np.count_nonzero(candidate != reference)),
        "exact": bool(np.array_equal(candidate, reference)),
    }


def _positive_ulp_histogram(
    candidate: np.ndarray, reference: np.ndarray
) -> dict[str, int]:
    candidate = np.asarray(candidate, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if np.any(candidate <= 0.0) or np.any(reference <= 0.0):
        raise ValueError("rstd ULP census requires positive finite values")
    delta = (
        candidate.reshape(-1).view(np.uint32).astype(np.int64)
        - reference.reshape(-1).view(np.uint32).astype(np.int64)
    )
    values, counts = np.unique(delta, return_counts=True)
    return {
        str(int(value)): int(count)
        for value, count in zip(values, counts, strict=True)
    }


def analyze_oracle(
    *,
    custom_out: np.ndarray,
    custom_mean: np.ndarray,
    custom_variance: np.ndarray,
    custom_rstd: np.ndarray,
    native_out: np.ndarray,
    native_mean: np.ndarray,
    native_rstd: np.ndarray,
    eps: np.float32,
) -> dict[str, Any]:
    arrays = {
        "custom_out": np.asarray(custom_out, dtype=np.float32),
        "custom_mean": np.asarray(custom_mean, dtype=np.float32),
        "custom_variance": np.asarray(custom_variance, dtype=np.float32),
        "custom_rstd": np.asarray(custom_rstd, dtype=np.float32),
        "native_out": np.asarray(native_out, dtype=np.float32),
        "native_mean": np.asarray(native_mean, dtype=np.float32),
        "native_rstd": np.asarray(native_rstd, dtype=np.float32),
    }
    for name, array in arrays.items():
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")
    if arrays["custom_out"].shape != arrays["native_out"].shape:
        raise ValueError("custom output shape does not match native")
    stats_shape = arrays["native_mean"].shape
    for name in ("custom_mean", "custom_variance", "custom_rstd", "native_rstd"):
        if arrays[name].shape != stats_shape:
            raise ValueError(f"{name} shape does not match native mean")
    if not np.array_equal(arrays["custom_out"], arrays["native_out"]):
        raise ValueError("custom output does not reproduce native")
    if not np.array_equal(arrays["custom_mean"], arrays["native_mean"]):
        raise ValueError("custom mean does not reproduce native")
    if not np.array_equal(arrays["custom_rstd"], arrays["native_rstd"]):
        raise ValueError("custom rstd does not reproduce native")
    if np.any(arrays["custom_variance"] < 0.0):
        raise ValueError("custom variance contains negative values")

    variance_plus_eps = np.float32(
        arrays["custom_variance"] + np.float32(eps)
    )
    correctly_rounded = np.float32(
        1.0 / np.sqrt(variance_plus_eps.astype(np.float64))
    )
    rstd_metric = _exact_metric(arrays["native_rstd"], correctly_rounded)
    rstd_metric["ulp_histogram"] = _positive_ulp_histogram(
        arrays["native_rstd"], correctly_rounded
    )
    return {
        "self_authentication": {
            "output_exact": True,
            "mean_exact": True,
            "rstd_exact": True,
        },
        "native_rstd_vs_correctly_rounded": rstd_metric,
        "array_sha256": {
            name: sha256_array(array) for name, array in arrays.items()
        },
        "correctly_rounded_rstd_sha256": sha256_array(correctly_rounded),
    }


def _native_layer_norm(
    input_tensor: Any, eps: float
) -> tuple[Any, Any, Any]:
    import torch

    shape = [int(input_tensor.shape[-1])]
    if hasattr(torch, "native_layer_norm"):
        return torch.native_layer_norm(
            input_tensor, shape, None, None, eps
        )
    return torch.ops.aten.native_layer_norm.default(
        input_tensor, shape, None, None, eps
    )


def _build_extension() -> Any:
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="trellis2mlx_cuda_welford_variance_v1",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=["welford_variance_cuda"],
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
    Path(path).write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--witness", default="witness.npz")
    parser.add_argument("--expected-witness-sha256")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-npz", required=True)
    args = parser.parse_args()

    started = time.time()
    output_json = Path(args.output_json)
    output_npz = Path(args.output_npz)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "failure_phase": "request_validation",
        "last_trustworthy_phase": "request_received",
        "witness": str(args.witness),
        "output_json": str(output_json),
        "output_npz": str(output_npz),
    }
    try:
        if output_json.resolve() == output_npz.resolve():
            raise ValueError("output JSON and NPZ paths must be distinct")
        if output_npz.exists():
            output_npz.unlink()
        report["last_trustworthy_phase"] = "output_path_validated"
        report["witness_identity_requested_raw"] = {
            "sha256": args.expected_witness_sha256,
        }
        requested_identity = requested_witness_identity(
            args.expected_witness_sha256
        )
        report["witness_identity_requested"] = requested_identity
        report["last_trustworthy_phase"] = "request_validated"
        report["failure_phase"] = "input_validation"

        witness_path = Path(args.witness)
        with np.load(witness_path, allow_pickle=False) as witness:
            required = {
                "layernorm_input",
                "reference_norm",
                "eps",
            }
            missing = sorted(required - set(witness.files))
            if missing:
                raise ValueError(f"witness missing required arrays: {missing}")
            input_np = np.asarray(
                witness["layernorm_input"], dtype=np.float32
            )
            reference_np = np.asarray(
                witness["reference_norm"], dtype=np.float32
            )
            eps = np.float32(np.asarray(witness["eps"]).item())
        effective_identity = effective_witness_identity(
            witness_path, input_np, reference_np, eps
        )
        report["witness_identity_effective"] = effective_identity
        validate_witness_identity(requested_identity, effective_identity)
        if not np.isfinite(input_np).all() or not np.isfinite(reference_np).all():
            raise ValueError("witness contains non-finite values")
        report["witness_sha256"] = effective_identity["sha256"]
        report["input_shape"] = list(input_np.shape)
        report["eps"] = float(eps)
        report["last_trustworthy_phase"] = "input_validated"
        report["failure_phase"] = "cuda_route_validation"

        import torch

        report["torch"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_device"] = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        )
        report["nvcc"] = _nvcc_version()
        if report["torch"] != EXPECTED_TORCH:
            raise RuntimeError(
                f"expected Torch {EXPECTED_TORCH}, got {report['torch']}"
            )
        if report["cuda_device"] != EXPECTED_DEVICE:
            raise RuntimeError(
                f"expected CUDA device {EXPECTED_DEVICE}, "
                f"got {report['cuda_device']!r}"
            )
        report["last_trustworthy_phase"] = "cuda_route_validated"
        report["failure_phase"] = "extension_compile"

        compile_started = time.time()
        extension = _build_extension()
        report["compile_seconds"] = time.time() - compile_started
        report["last_trustworthy_phase"] = "extension_compiled"
        report["failure_phase"] = "oracle_execution"

        input_tensor = torch.from_numpy(input_np).to(
            device="cuda", dtype=torch.bfloat16
        )
        native_out, native_mean, native_rstd = _native_layer_norm(
            input_tensor, float(eps)
        )
        custom_out, custom_mean, custom_variance, custom_rstd = (
            extension.welford_variance_cuda(input_tensor, float(eps))
        )
        torch.cuda.synchronize()

        def as_float32(value: Any) -> np.ndarray:
            return (
                value.detach()
                .to(dtype=torch.float32, device="cpu")
                .numpy()
                .astype(np.float32, copy=False)
            )

        native_out_np = as_float32(native_out)
        native_mean_np = as_float32(native_mean)
        native_rstd_np = as_float32(native_rstd)
        custom_out_np = as_float32(custom_out)
        custom_mean_np = as_float32(custom_mean)
        custom_variance_np = as_float32(custom_variance)
        custom_rstd_np = as_float32(custom_rstd)
        report["last_trustworthy_phase"] = "oracle_executed"
        report["failure_phase"] = "self_authentication"
        analysis = analyze_oracle(
            custom_out=custom_out_np,
            custom_mean=custom_mean_np,
            custom_variance=custom_variance_np,
            custom_rstd=custom_rstd_np,
            native_out=native_out_np,
            native_mean=native_mean_np,
            native_rstd=native_rstd_np,
            eps=eps,
        )
        reference_metric = _exact_metric(custom_out_np, reference_np)
        if not reference_metric["exact"]:
            raise ValueError(
                "authenticated custom output does not reproduce the "
                "bound source-CUDA reference"
            )
        report.update(analysis)
        report["custom_output_vs_reference"] = reference_metric
        report["last_trustworthy_phase"] = "self_authenticated"
        report["failure_phase"] = "output_write"

        correctly_rounded = np.float32(
            1.0
            / np.sqrt(
                np.float32(custom_variance_np + eps).astype(np.float64)
            )
        )
        np.savez_compressed(
            output_npz,
            custom_out=custom_out_np,
            custom_mean=custom_mean_np,
            custom_variance=custom_variance_np,
            custom_rstd=custom_rstd_np,
            native_mean=native_mean_np,
            native_rstd=native_rstd_np,
            correctly_rounded_rstd=correctly_rounded,
            eps=np.asarray(eps, dtype=np.float32),
        )
        report["output_npz_sha256"] = sha256_file(output_npz)
        report["output_npz_size_bytes"] = output_npz.stat().st_size
        report["status"] = "done"
        report["failure_phase"] = None
        report["last_trustworthy_phase"] = "output_validated"
        report["elapsed_seconds"] = time.time() - started
        _write_report(output_json, report)
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        report["elapsed_seconds"] = time.time() - started
        report["primary_output"] = {
            "exists": output_npz.is_file(),
            "sha256": sha256_file(output_npz)
            if output_npz.is_file()
            else None,
            "size_bytes": output_npz.stat().st_size
            if output_npz.is_file()
            else None,
        }
        _write_report(output_json, report)
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
