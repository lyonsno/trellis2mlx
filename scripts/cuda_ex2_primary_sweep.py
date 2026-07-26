#!/usr/bin/env python3
"""Harvest and authenticate the Turing EX2 primary response surface."""

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


SCHEMA = "trellis2mlx.cuda_ex2_primary_sweep.v1"
EXPECTED_TORCH = "2.10.0+cu128"
EXPECTED_CUDA = "12.8"
EXPECTED_DEVICE = "Tesla T4"
DEFAULT_GRID_FRACTION_BITS = 24
PRIMARY_GRID_POINTS = 1 << DEFAULT_GRID_FRACTION_BITS
DEFAULT_PROBE_POINTS = 1 << 20
EXPECTED_FIXTURE_ROWS = 4
EXPECTED_FIXTURE_WIDTH = 7697
EXPECTED_FIXTURE_POINTS = EXPECTED_FIXTURE_ROWS * EXPECTED_FIXTURE_WIDTH
EXPECTED_FIXTURE_SCHEMA = "trellis2mlx.source_cuda_softmax_fixture.v1"
EXPECTED_FIXTURE_SELECTED_ROWS = [182, 1059, 1261, 3821]
EXPECTED_FIXTURE_SELECTION = [
    "max_abs",
    "max_nonzero",
    "min_nonzero",
    "last_control",
]
EX2_INSTRUCTION = "ex2.approx.ftz.f32"
PROBE_GENERATOR = "float32-bit-stratified-v1"


CPP_SOURCE = r"""
#include <torch/extension.h>

std::vector<torch::Tensor> ex2_primary_sweep_cuda(
    int64_t grid_fraction_bits,
    torch::Tensor probe_input_bits,
    torch::Tensor fixture_args);
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

__device__ __forceinline__ float direct_ex2(float value) {
  float result;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

__device__ __forceinline__ float manual_libdevice_expf(float value) {
  const float log2e_hi = __int_as_float(1069066811);
  const float log2e_lo = __int_as_float(849703008);
  const float magic = __int_as_float(1262485504);
  const float bucket = __saturatef(
      __fmaf_rn(value, log2e_hi / 252.0f, 0.5f));
  const float exponent_magic = __fmaf_rd(
      bucket,
      252.0f,
      (magic - 126.0f) + 127.0f);
  const float exponent = exponent_magic - (magic + 127.0f);
  float reduced = __fmaf_rn(value, log2e_hi, -exponent);
  reduced = __fmaf_rn(value, log2e_lo, reduced);
  const float scale = __uint_as_float(
      __float_as_uint(exponent_magic) << 23);
  return direct_ex2(reduced) * scale;
}

__global__ void grid_kernel(
    uint32_t* output_bits,
    uint64_t points,
    int fraction_bits) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < points) {
    const float input = ldexpf(static_cast<float>(index), -fraction_bits);
    output_bits[index] = __float_as_uint(direct_ex2(input));
  }
}

__global__ void probe_kernel(
    const uint32_t* input_bits,
    uint32_t* output_bits,
    int64_t count) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    output_bits[index] = __float_as_uint(
        direct_ex2(__uint_as_float(input_bits[index])));
  }
}

__global__ void fixture_kernel(
    const float* args,
    uint32_t* expf_bits,
    uint32_t* manual_bits,
    int64_t count) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    expf_bits[index] = __float_as_uint(expf(args[index]));
    manual_bits[index] = __float_as_uint(
        manual_libdevice_expf(args[index]));
  }
}

}  // namespace

std::vector<torch::Tensor> ex2_primary_sweep_cuda(
    int64_t grid_fraction_bits,
    torch::Tensor probe_input_bits,
    torch::Tensor fixture_args) {
  TORCH_CHECK(
      grid_fraction_bits == 24,
      "primary grid_fraction_bits must be exactly 24");
  TORCH_CHECK(
      probe_input_bits.is_cuda()
          && probe_input_bits.scalar_type() == torch::kInt32
          && probe_input_bits.dim() == 1
          && probe_input_bits.is_contiguous(),
      "probe_input_bits must be a contiguous CUDA int32 vector");
  TORCH_CHECK(
      fixture_args.is_cuda()
          && fixture_args.scalar_type() == torch::kFloat32
          && fixture_args.dim() == 1
          && fixture_args.is_contiguous(),
      "fixture_args must be a contiguous CUDA float32 vector");

  const uint64_t grid_points = uint64_t{1} << grid_fraction_bits;
  auto int_options = probe_input_bits.options();
  auto grid_output_bits = torch::empty(
      {static_cast<int64_t>(grid_points)}, int_options);
  auto probe_output_bits = torch::empty_like(probe_input_bits);
  auto probe_repeat_bits = torch::empty_like(probe_input_bits);
  auto fixture_expf_bits = torch::empty(
      {fixture_args.size(0)}, int_options);
  auto fixture_manual_bits = torch::empty(
      {fixture_args.size(0)}, int_options);

  constexpr int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  grid_kernel<<<
      static_cast<unsigned int>((grid_points + threads - 1) / threads),
      threads,
      0,
      stream>>>(
      reinterpret_cast<uint32_t*>(grid_output_bits.data_ptr<int32_t>()),
      grid_points,
      static_cast<int>(grid_fraction_bits));
  probe_kernel<<<
      static_cast<unsigned int>(
          (probe_input_bits.size(0) + threads - 1) / threads),
      threads,
      0,
      stream>>>(
      reinterpret_cast<const uint32_t*>(
          probe_input_bits.data_ptr<int32_t>()),
      reinterpret_cast<uint32_t*>(
          probe_output_bits.data_ptr<int32_t>()),
      probe_input_bits.size(0));
  probe_kernel<<<
      static_cast<unsigned int>(
          (probe_input_bits.size(0) + threads - 1) / threads),
      threads,
      0,
      stream>>>(
      reinterpret_cast<const uint32_t*>(
          probe_input_bits.data_ptr<int32_t>()),
      reinterpret_cast<uint32_t*>(
          probe_repeat_bits.data_ptr<int32_t>()),
      probe_input_bits.size(0));
  fixture_kernel<<<
      static_cast<unsigned int>(
          (fixture_args.size(0) + threads - 1) / threads),
      threads,
      0,
      stream>>>(
      fixture_args.data_ptr<float>(),
      reinterpret_cast<uint32_t*>(
          fixture_expf_bits.data_ptr<int32_t>()),
      reinterpret_cast<uint32_t*>(
          fixture_manual_bits.data_ptr<int32_t>()),
      fixture_args.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {
      grid_output_bits,
      probe_output_bits,
      probe_repeat_bits,
      fixture_expf_bits,
      fixture_manual_bits};
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


def _require_u32_vector(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.uint32 or array.ndim != 1:
        raise ValueError(f"{name} must be a uint32 vector")
    return array


def requested_fixture_identity(expected_sha256: str | None) -> str:
    if (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise ValueError(
            "expected fixture sha256 must be 64 lowercase hexadecimal characters"
        )
    return expected_sha256


def deterministic_probe_bits(count: int) -> np.ndarray:
    if count < 16:
        raise ValueError("probe count must be at least 16")
    index = np.arange(count, dtype=np.uint64)
    # This odd affine permutation spreads probes over every positive float32
    # exponent band below one without depending on a library RNG.
    bits = (
        index * np.uint64(2654435761) + np.uint64(2246822519)
    ) % np.uint64(0x3F800000)
    bits = bits.astype(np.uint32)
    anchors = np.array(
        [
            0x00000000,
            0x00000001,
            0x007FFFFF,
            0x00800000,
            0x00800001,
            0x33800000,
            0x33800001,
            0x3E800000,
            0x3F000000,
            0x3F7FFFFD,
            0x3F7FFFFE,
            0x3F7FFFFF,
        ],
        dtype=np.uint32,
    )
    bits[: anchors.size] = anchors
    return bits


def _analyze_ex2_sweep(
    *,
    grid_output_bits: np.ndarray,
    probe_input_bits: np.ndarray,
    probe_output_bits: np.ndarray,
    probe_repeat_bits: np.ndarray,
    fixture_expected_bits: np.ndarray,
    fixture_expf_bits: np.ndarray,
    fixture_manual_bits: np.ndarray,
    runtime: dict[str, Any],
    implementation: dict[str, Any],
    expected_grid_points: int,
) -> dict[str, Any]:
    arrays = {
        name: _require_u32_vector(name, value)
        for name, value in {
            "grid_output_bits": grid_output_bits,
            "probe_input_bits": probe_input_bits,
            "probe_output_bits": probe_output_bits,
            "probe_repeat_bits": probe_repeat_bits,
            "fixture_expected_bits": fixture_expected_bits,
            "fixture_expf_bits": fixture_expf_bits,
            "fixture_manual_bits": fixture_manual_bits,
        }.items()
    }
    if arrays["grid_output_bits"].size != expected_grid_points:
        raise ValueError(
            "grid is partial: "
            f"expected {expected_grid_points}, "
            f"got {arrays['grid_output_bits'].size}"
        )
    probe_size = arrays["probe_input_bits"].size
    if any(
        arrays[name].size != probe_size
        for name in ("probe_output_bits", "probe_repeat_bits")
    ):
        raise ValueError("probe arrays have inconsistent lengths")
    fixture_size = arrays["fixture_expected_bits"].size
    if any(
        arrays[name].size != fixture_size
        for name in ("fixture_expf_bits", "fixture_manual_bits")
    ):
        raise ValueError("fixture arrays have inconsistent lengths")
    if runtime.get("torch") != EXPECTED_TORCH:
        raise ValueError(
            f"expected Torch {EXPECTED_TORCH}, got {runtime.get('torch')}"
        )
    if runtime.get("cuda") != EXPECTED_CUDA:
        raise ValueError(
            f"expected CUDA {EXPECTED_CUDA}, got {runtime.get('cuda')}"
        )
    if runtime.get("device") != EXPECTED_DEVICE:
        raise ValueError(
            f"expected CUDA device {EXPECTED_DEVICE}, "
            f"got {runtime.get('device')}"
        )
    if runtime.get("device_ordinal") != 0:
        raise ValueError(
            "effective CUDA device ordinal must be 0, "
            f"got {runtime.get('device_ordinal')}"
        )
    nvcc = runtime.get("nvcc")
    if not isinstance(nvcc, dict) or nvcc.get("release") != EXPECTED_CUDA:
        release = nvcc.get("release") if isinstance(nvcc, dict) else None
        raise ValueError(
            f"expected NVCC release {EXPECTED_CUDA}, got {release}"
        )
    nvcc_path = nvcc.get("path")
    if (
        not isinstance(nvcc_path, str)
        or not Path(nvcc_path).is_absolute()
    ):
        raise ValueError("effective NVCC path must be absolute")
    if implementation.get("instruction") != EX2_INSTRUCTION:
        raise ValueError(
            f"inline PTX instruction must be {EX2_INSTRUCTION}"
        )
    fraction_bits = implementation.get("grid_fraction_bits")
    if not isinstance(fraction_bits, int):
        raise ValueError("grid_fraction_bits is missing")
    if expected_grid_points != 1 << fraction_bits:
        raise ValueError("grid geometry is inconsistent")
    if implementation.get("grid_points") != expected_grid_points:
        raise ValueError("effective grid point count is inconsistent")
    if implementation.get("probe_generator") != PROBE_GENERATOR:
        raise ValueError("probe generator identity is inconsistent")
    if not np.array_equal(
        arrays["probe_output_bits"], arrays["probe_repeat_bits"]
    ):
        raise ValueError("probe replay is not deterministic")
    if not np.array_equal(
        arrays["fixture_expected_bits"], arrays["fixture_expf_bits"]
    ):
        raise ValueError("CUDA expf does not reproduce the Trellis fixture")
    if not np.array_equal(
        arrays["fixture_expected_bits"], arrays["fixture_manual_bits"]
    ):
        raise ValueError(
            "inline EX2 composition does not reproduce the Trellis fixture"
        )

    grid_bits = arrays["grid_output_bits"]
    if grid_bits[0] != np.float32(1.0).view(np.uint32):
        raise ValueError("EX2 grid does not begin at exactly one")
    if not np.all(grid_bits[1:] >= grid_bits[:-1]):
        raise ValueError("EX2 grid is not monotonic")
    grid_values = grid_bits.view(np.float32)
    if not np.isfinite(grid_values).all():
        raise ValueError("EX2 grid contains non-finite values")
    if np.any(grid_values < 1.0) or np.any(grid_values >= 2.0):
        raise ValueError("EX2 grid leaves the primary [1, 2) range")

    return {
        "self_authentication": {
            "runtime_exact": True,
            "instruction_exact": True,
            "probe_replay_exact": True,
            "fixture_expf_exact": True,
            "fixture_manual_ex2_exact": True,
        },
        "grid": {
            "fraction_bits": fraction_bits,
            "points": expected_grid_points,
            "monotonic": True,
            "first_bits": int(grid_bits[0]),
            "last_bits": int(grid_bits[-1]),
        },
        "probe_points": probe_size,
        "fixture_points": fixture_size,
        "array_sha256": {
            name: sha256_array(array) for name, array in arrays.items()
        },
    }


def analyze_ex2_sweep_unit(**kwargs: Any) -> dict[str, Any]:
    """Analyze reduced synthetic arrays without primary-census authority."""

    analysis = _analyze_ex2_sweep(**kwargs)
    analysis["grid"]["coverage"] = "unit-reduced"
    return analysis


def analyze_ex2_sweep(**kwargs: Any) -> dict[str, Any]:
    """Authenticate only the fixed complete 2^24 primary census."""

    implementation = kwargs.get("implementation")
    expected_grid_points = kwargs.get("expected_grid_points")
    if expected_grid_points != PRIMARY_GRID_POINTS:
        raise ValueError(
            "primary grid must contain exactly "
            f"{PRIMARY_GRID_POINTS} points"
        )
    if (
        not isinstance(implementation, dict)
        or implementation.get("grid_fraction_bits")
        != DEFAULT_GRID_FRACTION_BITS
        or implementation.get("grid_points") != PRIMARY_GRID_POINTS
    ):
        raise ValueError("primary grid identity must be exactly 24 bits")
    analysis = _analyze_ex2_sweep(**kwargs)
    analysis["self_authentication"]["complete_grid"] = True
    analysis["grid"]["coverage"] = "complete-primary"
    return analysis


def _build_extension() -> Any:
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="trellis2mlx_cuda_ex2_primary_sweep_v1",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=["ex2_primary_sweep_cuda"],
        extra_cuda_cflags=["-O3", "--fmad=true"],
        with_cuda=True,
        verbose=True,
    )


def validated_nvcc_identity(cuda_home: str | Path | None) -> dict[str, str]:
    if cuda_home is None:
        raise RuntimeError("PyTorch CUDA_HOME is unavailable")
    nvcc_path = (Path(cuda_home) / "bin" / "nvcc").resolve()
    if not nvcc_path.is_file():
        raise RuntimeError(
            f"PyTorch extension NVCC does not exist: {nvcc_path}"
        )
    try:
        completed = subprocess.run(
            [str(nvcc_path), "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(
            f"unable to execute PyTorch extension NVCC {nvcc_path}: {exc}"
        ) from exc
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        raise RuntimeError(
            f"PyTorch extension NVCC exited {completed.returncode}: {output}"
        )
    match = re.search(r"\brelease\s+(\d+\.\d+)\b", output)
    if match is None:
        raise RuntimeError(
            f"unable to parse PyTorch extension NVCC version: {output}"
        )
    release = match.group(1)
    if release != EXPECTED_CUDA:
        raise RuntimeError(
            f"expected NVCC release {EXPECTED_CUDA}, got {release}"
        )
    return {
        "path": str(nvcc_path),
        "release": release,
        "version_output": output,
    }


def _same_path(first: Path, second: Path) -> bool:
    if first.resolve() == second.resolve():
        return True
    try:
        return os.path.samefile(first, second)
    except FileNotFoundError:
        return False
    except OSError:
        return True


def _safe_failure_report_path(
    requested: Path,
    protected: tuple[Path, ...],
) -> Path:
    if not any(_same_path(requested, path) for path in protected):
        return requested
    candidate = protected[0].with_name(
        protected[0].name + ".ex2-sweep-failure.json"
    )
    while any(_same_path(candidate, path) for path in protected):
        candidate = candidate.with_name(candidate.name + ".failure.json")
    return candidate


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def validate_written_primary(
    path: Path,
    *,
    expected_arrays: dict[str, np.ndarray],
    expected_route_identity_json: str,
) -> None:
    if not path.is_file():
        raise ValueError("written primary is missing")
    with np.load(path, allow_pickle=False) as loaded:
        expected_keys = {*expected_arrays, "route_identity_json"}
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
        actual_identity = str(loaded["route_identity_json"].item())
        if actual_identity != expected_route_identity_json:
            raise ValueError("written primary route identity mismatch")
        json.loads(actual_identity)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-npz", required=True)
    parser.add_argument("--expected-fixture-sha256", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument(
        "--grid-fraction-bits",
        type=int,
        default=DEFAULT_GRID_FRACTION_BITS,
    )
    parser.add_argument(
        "--probe-points",
        type=int,
        default=DEFAULT_PROBE_POINTS,
    )
    args = parser.parse_args(argv)

    started = time.time()
    fixture_path = Path(args.fixture_npz)
    requested_output_json = Path(args.output_json)
    output_npz = Path(args.output_npz)
    output_json = _safe_failure_report_path(
        requested_output_json,
        (fixture_path, output_npz),
    )
    path_collisions = []
    if _same_path(fixture_path, requested_output_json):
        path_collisions.append("output JSON aliases protected fixture NPZ")
    if _same_path(fixture_path, output_npz):
        path_collisions.append("output NPZ aliases protected fixture NPZ")
    if _same_path(requested_output_json, output_npz):
        path_collisions.append("output JSON aliases output NPZ")
    primary_output_status = (
        "protected_input"
        if _same_path(fixture_path, output_npz)
        else ("preexisting_untrusted" if output_npz.exists() else "missing")
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "failure_phase": "request_validation",
        "last_trustworthy_phase": "request_received",
        "primary_output_status": primary_output_status,
        "fixture_npz": str(fixture_path),
        "output_json_requested": str(requested_output_json),
        "output_json_effective": str(output_json),
        "output_npz": str(output_npz),
        "requested_config": {
            "grid_fraction_bits": args.grid_fraction_bits,
            "probe_points": args.probe_points,
        },
    }
    try:
        if path_collisions:
            raise ValueError("; ".join(path_collisions))
        expected_fixture_sha = requested_fixture_identity(
            args.expected_fixture_sha256
        )
        if args.grid_fraction_bits != DEFAULT_GRID_FRACTION_BITS:
            raise ValueError(
                "primary grid_fraction_bits must be exactly "
                f"{DEFAULT_GRID_FRACTION_BITS}"
            )
        deterministic_probe_bits(args.probe_points)
        if output_npz.exists():
            output_npz.unlink()
        report["primary_output_status"] = "missing"
        report["last_trustworthy_phase"] = "request_validated"

        report["failure_phase"] = "fixture_load"
        if not fixture_path.is_file():
            raise FileNotFoundError(
                f"fixture NPZ does not exist: {fixture_path}"
            )
        effective_fixture_sha = sha256_file(fixture_path)
        if effective_fixture_sha != expected_fixture_sha:
            raise ValueError(
                "fixture sha256 mismatch: "
                f"expected {expected_fixture_sha}, "
                f"got {effective_fixture_sha}"
            )
        with np.load(fixture_path, allow_pickle=False) as loaded:
            required = {
                "scores_fp32",
                "exponents_fp32",
                "row_maxes_fp32",
                "route_identity_json",
            }
            missing = sorted(required - set(loaded.files))
            if missing:
                raise ValueError(f"fixture is missing arrays: {missing}")
            scores = np.asarray(loaded["scores_fp32"])
            expected = np.asarray(loaded["exponents_fp32"])
            row_maxes = np.asarray(loaded["row_maxes_fp32"])
            fixture_route_identity_json = str(
                loaded["route_identity_json"].item()
            )
        if (
            scores.dtype != np.float32
            or expected.dtype != np.float32
            or row_maxes.dtype != np.float32
            or scores.ndim != 2
            or expected.shape != scores.shape
            or row_maxes.shape != (scores.shape[0],)
            or not np.isfinite(scores).all()
            or not np.isfinite(expected).all()
            or not np.isfinite(row_maxes).all()
        ):
            raise ValueError("fixture arrays have invalid metadata or values")
        if scores.shape != (
            EXPECTED_FIXTURE_ROWS,
            EXPECTED_FIXTURE_WIDTH,
        ):
            raise ValueError(
                "fixture scores/exponents must have exact shape "
                f"[{EXPECTED_FIXTURE_ROWS},{EXPECTED_FIXTURE_WIDTH}]"
            )
        if scores.size != EXPECTED_FIXTURE_POINTS:
            raise ValueError(
                f"fixture must contain exactly {EXPECTED_FIXTURE_POINTS} points"
            )
        try:
            fixture_route_identity = json.loads(
                fixture_route_identity_json
            )
        except json.JSONDecodeError as exc:
            raise ValueError("fixture route identity is invalid JSON") from exc
        expected_route_fields = {
            "schema": EXPECTED_FIXTURE_SCHEMA,
            "selected_rows": EXPECTED_FIXTURE_SELECTED_ROWS,
            "selection": EXPECTED_FIXTURE_SELECTION,
            "width": EXPECTED_FIXTURE_WIDTH,
        }
        for field, required_value in expected_route_fields.items():
            if fixture_route_identity.get(field) != required_value:
                raise ValueError(
                    f"fixture route identity {field} mismatch"
                )
        for field in (
            "source_oracle_script_sha256",
            "source_oracle_sha256",
            "source_stage_sha256",
        ):
            value = fixture_route_identity.get(field)
            if (
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{64}", value) is None
            ):
                raise ValueError(
                    f"fixture route identity {field} is invalid"
                )
        fixture_args = np.float32(scores - row_maxes[:, None]).reshape(-1)
        fixture_expected_bits = expected.reshape(-1).view(np.uint32).copy()
        report["fixture_identity_effective"] = {
            "sha256": effective_fixture_sha,
            "rows": int(scores.shape[0]),
            "width": int(scores.shape[1]),
            "points": int(scores.size),
            "route_identity_json_sha256": hashlib.sha256(
                fixture_route_identity_json.encode("utf-8")
            ).hexdigest(),
        }
        report["last_trustworthy_phase"] = "fixture_loaded"

        report["failure_phase"] = "runtime_validation"
        import torch

        if torch.__version__ != EXPECTED_TORCH:
            raise RuntimeError(
                f"expected Torch {EXPECTED_TORCH}, got {torch.__version__}"
            )
        if torch.version.cuda != EXPECTED_CUDA:
            raise RuntimeError(
                f"expected CUDA {EXPECTED_CUDA}, got {torch.version.cuda}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        device_ordinal = int(torch.cuda.current_device())
        if device_ordinal != 0:
            raise RuntimeError(
                "current CUDA device must be ordinal 0, "
                f"got {device_ordinal}"
            )
        device_name = torch.cuda.get_device_name(device_ordinal)
        if device_name != EXPECTED_DEVICE:
            raise RuntimeError(
                f"expected CUDA device {EXPECTED_DEVICE}, got {device_name}"
            )
        from torch.utils.cpp_extension import CUDA_HOME

        nvcc_identity = validated_nvcc_identity(CUDA_HOME)
        runtime = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": device_name,
            "device_ordinal": device_ordinal,
            "nvcc": nvcc_identity,
        }
        implementation = {
            "instruction": EX2_INSTRUCTION,
            "grid_fraction_bits": args.grid_fraction_bits,
            "grid_points": PRIMARY_GRID_POINTS,
            "probe_generator": PROBE_GENERATOR,
            "script_sha256": sha256_file(Path(__file__)),
            "cuda_source_sha256": hashlib.sha256(
                CUDA_SOURCE.encode("utf-8")
            ).hexdigest(),
            "compile_flags": ["-O3", "--fmad=true"],
        }
        report["runtime"] = runtime
        report["implementation_identity"] = implementation
        report["effective_config"] = {
            "grid_fraction_bits": args.grid_fraction_bits,
            "grid_points": PRIMARY_GRID_POINTS,
            "probe_points": args.probe_points,
        }
        report["last_trustworthy_phase"] = "runtime_validated"

        report["failure_phase"] = "extension_build"
        extension = _build_extension()
        report["last_trustworthy_phase"] = "extension_built"

        report["failure_phase"] = "sweep_execution"
        probe_input_bits = deterministic_probe_bits(args.probe_points)
        device = torch.device("cuda", 0)
        probe_tensor = torch.from_numpy(
            probe_input_bits.view(np.int32)
        ).to(device=device)
        fixture_tensor = torch.from_numpy(fixture_args).to(device=device)
        outputs = extension.ex2_primary_sweep_cuda(
            args.grid_fraction_bits,
            probe_tensor,
            fixture_tensor,
        )
        torch.cuda.synchronize(device=0)
        (
            grid_output_bits,
            probe_output_bits,
            probe_repeat_bits,
            fixture_expf_bits,
            fixture_manual_bits,
        ) = [
            tensor.detach().cpu().numpy().view(np.uint32).copy()
            for tensor in outputs
        ]
        report["last_trustworthy_phase"] = "sweep_executed"

        report["failure_phase"] = "self_authentication"
        analysis = analyze_ex2_sweep(
            grid_output_bits=grid_output_bits,
            probe_input_bits=probe_input_bits,
            probe_output_bits=probe_output_bits,
            probe_repeat_bits=probe_repeat_bits,
            fixture_expected_bits=fixture_expected_bits,
            fixture_expf_bits=fixture_expf_bits,
            fixture_manual_bits=fixture_manual_bits,
            runtime=runtime,
            implementation=implementation,
            expected_grid_points=PRIMARY_GRID_POINTS,
        )
        report["analysis"] = analysis
        report["last_trustworthy_phase"] = "self_authenticated"

        report["failure_phase"] = "primary_write"
        route_identity_json = json.dumps(
            {
                "schema": SCHEMA,
                "fixture_identity": report["fixture_identity_effective"],
                "runtime": runtime,
                "implementation_identity": implementation,
                "effective_config": report["effective_config"],
                "self_authentication": analysis["self_authentication"],
            },
            sort_keys=True,
        )
        primary_arrays = {
            "grid_output_bits": grid_output_bits,
            "probe_input_bits": probe_input_bits,
            "probe_output_bits": probe_output_bits,
            "probe_repeat_bits": probe_repeat_bits,
            "fixture_expected_bits": fixture_expected_bits,
            "fixture_expf_bits": fixture_expf_bits,
            "fixture_manual_bits": fixture_manual_bits,
        }
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_npz,
            **primary_arrays,
            route_identity_json=np.array(route_identity_json),
        )
        report["primary_output_status"] = "written_unverified"
        validate_written_primary(
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
