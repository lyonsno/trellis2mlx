#!/usr/bin/env python3
"""Identify the CUDA/Turing instruction semantics behind native rsqrtf."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import traceback
from typing import Any

import numpy as np


SCHEMA = "trellis2mlx.cuda_rsqrt_instruction_witness.v1"
EXPECTED_TORCH = "2.10.0+cu128"
EXPECTED_DEVICE = "Tesla T4"
EXPECTED_COUNT = 4096
NORMALIZED_SWEEP_COUNT = 1 << 24
VARIANT_NAMES = (
    "rsqrtf",
    "frsqrt_rn",
    "one_over_sqrtf",
    "inline_ptx_rsqrt_approx",
)


CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor rsqrt_variants_cuda(torch::Tensor input);
torch::Tensor rsqrt_normalized_delta_cuda(torch::Tensor input);
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ float inline_ptx_rsqrt_approx(float value) {
  float result;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

__global__ void rsqrt_variants_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t count) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const float value = input[index];
  output[index] = rsqrtf(value);
  output[count + index] = __frsqrt_rn(value);
  output[2 * count + index] = 1.0f / sqrtf(value);
  output[3 * count + index] = inline_ptx_rsqrt_approx(value);
}

__global__ void rsqrt_normalized_delta_kernel(
    int8_t* __restrict__ output) {
  constexpr int kHalfDomain = 1 << 23;
  constexpr int kDomain = 1 << 24;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kDomain) {
    return;
  }
  const uint32_t bits = index < kHalfDomain
      ? 0x3f800000u + static_cast<uint32_t>(index)
      : 0x40000000u + static_cast<uint32_t>(index - kHalfDomain);
  const float value = __uint_as_float(bits);
  const int32_t approximate_bits =
      static_cast<int32_t>(__float_as_uint(rsqrtf(value)));
  const int32_t rounded_bits =
      static_cast<int32_t>(__float_as_uint(__frsqrt_rn(value)));
  output[index] = static_cast<int8_t>(approximate_bits - rounded_bits);
}

}  // namespace

torch::Tensor rsqrt_variants_cuda(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32,
              "input must be float32");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.dim() == 1, "input must be one-dimensional");

  const int64_t count = input.numel();
  auto output = torch::empty({4, count}, input.options());
  constexpr int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  rsqrt_variants_kernel<<<
      blocks,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<float>(),
      output.data_ptr<float>(),
      count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor rsqrt_normalized_delta_cuda(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  constexpr int count = 1 << 24;
  auto output = torch::empty(
      {count}, input.options().dtype(torch::kInt8));
  constexpr int threads = 256;
  constexpr int blocks = (count + threads - 1) / threads;
  rsqrt_normalized_delta_kernel<<<
      blocks,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<int8_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
"""


STANDALONE_CUDA_SOURCE = r"""
#include <cuda_runtime.h>

__device__ __forceinline__ float inline_ptx_rsqrt_approx(float value) {
  float result;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

extern "C" __global__ void rsqrt_variants(
    const float* __restrict__ input,
    float* __restrict__ output,
    int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const float value = input[index];
  output[index] = rsqrtf(value);
  output[count + index] = __frsqrt_rn(value);
  output[2 * count + index] = 1.0f / sqrtf(value);
  output[3 * count + index] = inline_ptx_rsqrt_approx(value);
}
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _exact_metric(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    shape_match = actual.shape == expected.shape
    if not shape_match:
        return {
            "shape_match": False,
            "shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "exact": False,
            "nonzero": None,
            "max_abs": None,
            "mean_abs": None,
        }
    difference = np.abs(
        actual.astype(np.float64) - expected.astype(np.float64)
    )
    return {
        "shape_match": True,
        "shape": list(actual.shape),
        "exact": bool(np.array_equal(actual, expected)),
        "nonzero": int(np.count_nonzero(actual != expected)),
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()) if difference.size else 0.0,
    }


def _positive_ulp_histogram(
    actual: np.ndarray, expected: np.ndarray
) -> dict[str, int]:
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    if actual.shape != expected.shape:
        raise ValueError("ULP inputs must have matching shapes")
    if np.any(actual <= 0) or np.any(expected <= 0):
        raise ValueError("ULP inputs must be positive finite float32 values")
    delta = (
        actual.view(np.uint32).astype(np.int64)
        - expected.view(np.uint32).astype(np.int64)
    )
    values, counts = np.unique(delta, return_counts=True)
    return {
        str(int(value)): int(count)
        for value, count in zip(values, counts, strict=True)
    }


def analyze_variants(
    *,
    native_rstd: np.ndarray,
    correctly_rounded_rstd: np.ndarray,
    variants: dict[str, np.ndarray],
) -> dict[str, Any]:
    native = np.asarray(native_rstd, dtype=np.float32)
    correctly_rounded = np.asarray(
        correctly_rounded_rstd, dtype=np.float32
    )
    if native.shape != correctly_rounded.shape:
        raise ValueError("native and correctly rounded arrays differ in shape")
    missing = {"rsqrtf", "inline_ptx_rsqrt_approx"} - variants.keys()
    if missing:
        raise ValueError(f"missing required variants: {sorted(missing)}")

    normalized = {
        name: np.asarray(value, dtype=np.float32)
        for name, value in variants.items()
    }
    for name, value in normalized.items():
        if value.shape != native.shape:
            raise ValueError(
                f"{name} shape {value.shape} does not match {native.shape}"
            )
    if not np.array_equal(normalized["rsqrtf"], native):
        raise ValueError(
            "rsqrtf does not reproduce authenticated native rstd"
        )
    if not np.array_equal(
        normalized["inline_ptx_rsqrt_approx"], native
    ):
        raise ValueError(
            "inline PTX rsqrt.approx does not reproduce authenticated native "
            "rstd"
        )

    variant_report: dict[str, Any] = {}
    for name, value in normalized.items():
        variant_report[name] = {
            "sha256": sha256_array(value),
            "vs_native": _exact_metric(value, native),
            "vs_correctly_rounded": _exact_metric(
                value, correctly_rounded
            ),
            "ulp_vs_correctly_rounded": _positive_ulp_histogram(
                value, correctly_rounded
            ),
        }
    native_metric = _exact_metric(native, correctly_rounded)
    native_metric["ulp_histogram"] = _positive_ulp_histogram(
        native, correctly_rounded
    )
    return {
        "self_authentication": {
            "rsqrtf_exact_native": True,
            "inline_ptx_exact_native": True,
        },
        "native_vs_correctly_rounded": native_metric,
        "variants": variant_report,
    }


def classify_compiler_evidence(*, ptx: str, sass: str) -> dict[str, int]:
    return {
        "ptx_rsqrt_approx_count": len(
            re.findall(r"\brsqrt\.approx\.f32\b", ptx)
        ),
        "ptx_rsqrt_approx_ftz_count": len(
            re.findall(r"\brsqrt\.approx\.ftz\.f32\b", ptx)
        ),
        "sass_mufu_rsq_count": len(
            re.findall(r"\bMUFU\.RSQ\b", sass, flags=re.IGNORECASE)
        ),
    }


def validate_compiler_evidence(evidence: dict[str, Any]) -> None:
    classification = evidence.get("classification")
    if not isinstance(classification, dict):
        raise ValueError("compiler evidence is missing its classification")
    if classification.get("ptx_rsqrt_approx_count", 0) < 1:
        raise ValueError("PTX omits rsqrt.approx.f32 instruction evidence")
    if classification.get("ptx_rsqrt_approx_ftz_count", 0) < 1:
        raise ValueError("PTX omits rsqrt.approx.ftz.f32 instruction evidence")
    if classification.get("sass_mufu_rsq_count", 0) < 1:
        raise ValueError("SASS omits MUFU.RSQ instruction evidence")


def requested_witness_identity(expected_sha256: str | None) -> dict[str, str]:
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError(
            "expected witness sha256 must be 64 lowercase hexadecimal characters"
        )
    return {"sha256": expected_sha256}


def normalized_rsqrt_coordinate(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(
            "normalized rsqrt coordinates require positive finite values"
        )
    bits = values.view(np.uint32)
    biased_exponent = ((bits >> 23) & np.uint32(0xFF)).astype(np.int32)
    if np.any(biased_exponent == 0):
        raise ValueError("normalized rsqrt coordinates exclude subnormals")
    exponent_parity = (biased_exponent - 127) & 1
    mantissa = bits & np.uint32(0x7FFFFF)
    return (
        mantissa
        + exponent_parity.astype(np.uint32) * np.uint32(1 << 23)
    )


def analyze_normalized_sweep(
    *,
    normalized_delta: np.ndarray,
    witness_coordinates: np.ndarray,
    witness_delta: np.ndarray,
    expected_count: int = NORMALIZED_SWEEP_COUNT,
) -> dict[str, Any]:
    delta = np.asarray(normalized_delta)
    if delta.dtype != np.int8:
        raise ValueError(
            f"normalized sweep must have dtype int8, got {delta.dtype}"
        )
    if delta.shape != (expected_count,):
        raise ValueError(
            "normalized sweep must have shape "
            f"({expected_count},), got {delta.shape}"
        )
    coordinates = np.asarray(witness_coordinates, dtype=np.uint32)
    observed = np.asarray(witness_delta, dtype=np.int64)
    if coordinates.shape != observed.shape:
        raise ValueError("witness coordinates and deltas differ in shape")
    if np.any(coordinates >= expected_count):
        raise ValueError("witness coordinate escapes normalized sweep")
    if not np.array_equal(
        delta[coordinates].astype(np.int64), observed
    ):
        raise ValueError(
            "normalized sweep does not reproduce witness deltas"
        )
    values, counts = np.unique(delta, return_counts=True)
    return {
        "count": int(delta.size),
        "dtype": str(delta.dtype),
        "minimum_delta": int(delta.min()),
        "maximum_delta": int(delta.max()),
        "histogram": {
            str(int(value)): int(count)
            for value, count in zip(values, counts, strict=True)
        },
        "run_count": int(
            1 + np.count_nonzero(delta[1:] != delta[:-1])
            if delta.size
            else 0
        ),
        "witness_exact": True,
    }


def _compile_instruction_evidence(
    *, output_ptx: Path, output_sass: Path
) -> dict[str, Any]:
    nvcc = shutil.which("nvcc")
    cuobjdump = shutil.which("cuobjdump")
    if nvcc is None:
        raise RuntimeError("nvcc is unavailable")
    if cuobjdump is None:
        raise RuntimeError("cuobjdump is unavailable")

    with tempfile.TemporaryDirectory(prefix="t2mlx-rsqrt-") as temp:
        root = Path(temp)
        source = root / "rsqrt_variants.cu"
        cubin = root / "rsqrt_variants.cubin"
        source.write_text(STANDALONE_CUDA_SOURCE)
        common = [
            nvcc,
            "-O3",
            "--fmad=true",
            "-arch=sm_75",
            str(source),
        ]
        ptx_run = subprocess.run(
            [*common, "--ptx", "-o", str(output_ptx)],
            capture_output=True,
            text=True,
            check=False,
        )
        if ptx_run.returncode != 0:
            raise RuntimeError(
                "nvcc PTX compilation failed: "
                + (ptx_run.stdout + ptx_run.stderr).strip()
            )
        cubin_run = subprocess.run(
            [*common, "--cubin", "-o", str(cubin)],
            capture_output=True,
            text=True,
            check=False,
        )
        if cubin_run.returncode != 0:
            raise RuntimeError(
                "nvcc cubin compilation failed: "
                + (cubin_run.stdout + cubin_run.stderr).strip()
            )
        sass_run = subprocess.run(
            [cuobjdump, "--dump-sass", str(cubin)],
            capture_output=True,
            text=True,
            check=False,
        )
        if sass_run.returncode != 0:
            raise RuntimeError(
                "cuobjdump failed: "
                + (sass_run.stdout + sass_run.stderr).strip()
            )
        output_sass.write_text(sass_run.stdout)

    ptx = output_ptx.read_text()
    sass = output_sass.read_text()
    evidence = {
        "classification": classify_compiler_evidence(ptx=ptx, sass=sass),
        "ptx_sha256": sha256_file(output_ptx),
        "sass_sha256": sha256_file(output_sass),
        "nvcc_command": [
            "nvcc",
            "-O3",
            "--fmad=true",
            "-arch=sm_75",
        ],
        "cuobjdump_command": ["cuobjdump", "--dump-sass"],
    }
    validate_compiler_evidence(evidence)
    return evidence


def _build_extension() -> Any:
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="trellis2mlx_cuda_rsqrt_instruction_v1",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=[
            "rsqrt_variants_cuda",
            "rsqrt_normalized_delta_cuda",
        ],
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
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witness", type=Path, default=Path("rsqrt_witness.npz"))
    parser.add_argument("--expected-witness-sha256")
    parser.add_argument(
        "--output-json", type=Path, default=Path("cuda_rsqrt_result.json")
    )
    parser.add_argument(
        "--output-npz", type=Path, default=Path("cuda_rsqrt_result.npz")
    )
    parser.add_argument(
        "--output-ptx", type=Path, default=Path("cuda_rsqrt_variants.ptx")
    )
    parser.add_argument(
        "--output-sass", type=Path, default=Path("cuda_rsqrt_variants.sass")
    )
    parser.add_argument("--normalized-sweep", action="store_true")
    return parser.parse_args()


def main() -> int:
    started = time.time()
    phase = "argument_parsing"
    last_trustworthy = "process_started"
    args: argparse.Namespace | None = None
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "failure_phase": phase,
        "last_trustworthy_phase": last_trustworthy,
    }
    try:
        args = _parse_args()
        for path in (
            args.output_json,
            args.output_npz,
            args.output_ptx,
            args.output_sass,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
        for stale in (args.output_npz, args.output_ptx, args.output_sass):
            stale.unlink(missing_ok=True)
        phase = "request_validation"
        last_trustworthy = "output_paths_validated"
        report["witness_identity_requested_raw"] = {
            "sha256": args.expected_witness_sha256
        }
        requested_identity = requested_witness_identity(
            args.expected_witness_sha256
        )
        report["witness_identity_requested"] = requested_identity

        import torch

        report.update(
            {
                "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
                "nvcc": _nvcc_version(),
            }
        )
        if torch.__version__ != EXPECTED_TORCH:
            raise RuntimeError(
                f"expected Torch {EXPECTED_TORCH}, got {torch.__version__}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if torch.cuda.get_device_name(0) != EXPECTED_DEVICE:
            raise RuntimeError(
                f"expected {EXPECTED_DEVICE}, got "
                f"{torch.cuda.get_device_name(0)}"
            )
        if not args.witness.is_file():
            raise FileNotFoundError(f"witness does not exist: {args.witness}")
        witness_sha256 = sha256_file(args.witness)
        effective_identity = {"sha256": witness_sha256}
        report["witness_identity_effective"] = effective_identity
        if effective_identity != requested_identity:
            raise ValueError(
                "witness sha256 mismatch: expected "
                f"{requested_identity['sha256']}, got {witness_sha256}"
            )
        phase = "witness_loading"
        last_trustworthy = "request_validated"

        with np.load(args.witness, allow_pickle=False) as loaded:
            required = {
                "variance_plus_eps",
                "native_rstd",
                "correctly_rounded_rstd",
            }
            missing = required - set(loaded.files)
            if missing:
                raise ValueError(f"witness missing arrays: {sorted(missing)}")
            variance_plus_eps = np.asarray(
                loaded["variance_plus_eps"], dtype=np.float32
            ).reshape(-1)
            native_rstd = np.asarray(
                loaded["native_rstd"], dtype=np.float32
            ).reshape(-1)
            correctly_rounded = np.asarray(
                loaded["correctly_rounded_rstd"], dtype=np.float32
            ).reshape(-1)
        if not (
            variance_plus_eps.shape
            == native_rstd.shape
            == correctly_rounded.shape
            == (EXPECTED_COUNT,)
        ):
            raise ValueError(
                "witness arrays must all have shape "
                f"({EXPECTED_COUNT},), got {variance_plus_eps.shape}, "
                f"{native_rstd.shape}, {correctly_rounded.shape}"
            )
        if not np.all(np.isfinite(variance_plus_eps)):
            raise ValueError("variance_plus_eps contains non-finite values")
        if np.any(variance_plus_eps <= 0):
            raise ValueError("variance_plus_eps must be positive")
        phase = "compiler_evidence"
        last_trustworthy = "witness_loaded"

        compiler_started = time.time()
        compiler_evidence = _compile_instruction_evidence(
            output_ptx=args.output_ptx,
            output_sass=args.output_sass,
        )
        compiler_seconds = time.time() - compiler_started
        phase = "extension_compilation"
        last_trustworthy = "compiler_evidence_written"

        extension_started = time.time()
        extension = _build_extension()
        extension_seconds = time.time() - extension_started
        phase = "variant_execution"
        last_trustworthy = "extension_compiled"

        input_tensor = torch.from_numpy(variance_plus_eps).cuda()
        output_tensor = extension.rsqrt_variants_cuda(input_tensor)
        torch.cuda.synchronize()
        output = output_tensor.cpu().numpy()
        variants = {
            name: np.asarray(output[index], dtype=np.float32)
            for index, name in enumerate(VARIANT_NAMES)
        }
        normalized_delta = None
        if args.normalized_sweep:
            phase = "normalized_sweep_execution"
            normalized_delta = (
                extension.rsqrt_normalized_delta_cuda(input_tensor)
                .cpu()
                .numpy()
            )
        phase = "result_validation"
        last_trustworthy = "variants_executed"

        analysis = analyze_variants(
            native_rstd=native_rstd,
            correctly_rounded_rstd=correctly_rounded,
            variants=variants,
        )
        normalized_analysis = None
        if normalized_delta is not None:
            normalized_analysis = analyze_normalized_sweep(
                normalized_delta=normalized_delta,
                witness_coordinates=normalized_rsqrt_coordinate(
                    variance_plus_eps
                ),
                witness_delta=(
                    variants["rsqrtf"].view(np.uint32).astype(np.int64)
                    - correctly_rounded.view(np.uint32).astype(np.int64)
                ),
            )
            np.savez_compressed(
                args.output_npz,
                variance_plus_eps=variance_plus_eps,
                native_rstd=native_rstd,
                correctly_rounded_rstd=correctly_rounded,
                normalized_delta=normalized_delta,
                **variants,
            )
        else:
            np.savez(
                args.output_npz,
                variance_plus_eps=variance_plus_eps,
                native_rstd=native_rstd,
                correctly_rounded_rstd=correctly_rounded,
                **variants,
            )
        report.update(
            {
                "status": "done",
                "failure_phase": None,
                "last_trustworthy_phase": "output_validated",
                "witness": str(args.witness),
                "witness_sha256": witness_sha256,
                "witness_sha256_requested": args.expected_witness_sha256,
                "witness_count": EXPECTED_COUNT,
                "input_sha256": sha256_array(variance_plus_eps),
                "native_rstd_sha256": sha256_array(native_rstd),
                "correctly_rounded_rstd_sha256": sha256_array(
                    correctly_rounded
                ),
                "compiler_evidence": compiler_evidence,
                "compiler_seconds": compiler_seconds,
                "extension_compile_seconds": extension_seconds,
                "elapsed_seconds": time.time() - started,
                "output_npz": str(args.output_npz),
                "output_npz_sha256": sha256_file(args.output_npz),
                "output_ptx": str(args.output_ptx),
                "output_sass": str(args.output_sass),
                "normalized_sweep_requested": args.normalized_sweep,
                "normalized_sweep": normalized_analysis,
                **analysis,
            }
        )
        _write_report(args.output_json, report)
        print(json.dumps(report, sort_keys=True))
        return 0
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "failure_phase": phase,
                "last_trustworthy_phase": last_trustworthy,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.time() - started,
            }
        )
        if args is not None:
            args.output_npz.unlink(missing_ok=True)
            report["primary_output"] = {
                "path": str(args.output_npz),
                "exists": args.output_npz.exists(),
            }
            _write_report(args.output_json, report)
        print(json.dumps(report, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
