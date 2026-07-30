#!/usr/bin/env python3
"""Capture every K=16 prefix of the authenticated SM75 WMMA dot product."""

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


SCHEMA = "trellis2mlx.cuda_decoder_block0_wmma_prefix_probe.v1"
EXPECTED_TORCH = "2.10.0+cu128"
EXPECTED_DEVICE = "Tesla T4"
ROWS = 16
CHANNELS = 1024
REDUCTION = 1024
TILE_WIDTH = 16
PREFIX_COUNT = REDUCTION // 16
SELECTED_WINDOW_ROW = 13
DEFAULT_TILE_COL = 16
KERNEL_IDENTITY = "direct_sm75_wmma_prefix_m16n16k16"

CPP_SOURCE = """
torch::Tensor sm75_wmma_prefix_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int64_t tile_col);
"""

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;

__global__ void sm75_wmma_prefix_kernel(
    const half* input,
    const half* weight,
    float* prefixes,
    int channels,
    int reduction,
    int tile_col) {
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
        input + offset,
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
    wmma::store_matrix_sync(
        prefixes + (offset / 16) * 16 * 16,
        accumulator_fragment,
        16,
        wmma::mem_row_major);
  }
}

torch::Tensor sm75_wmma_prefix_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int64_t tile_col) {
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
  TORCH_CHECK(
      input.size(0) == 16,
      "input must contain exactly one 16-row WMMA tile");
  const int reduction = static_cast<int>(input.size(1));
  TORCH_CHECK(
      weight.size(0) == reduction,
      "weight reduction dimension mismatch");
  const int channels = static_cast<int>(weight.size(1));
  TORCH_CHECK(
      reduction % 16 == 0 && channels % 16 == 0,
      "WMMA dimensions must be multiples of 16");
  TORCH_CHECK(
      tile_col >= 0 && tile_col + 16 <= channels && tile_col % 16 == 0,
      "tile_col must identify an aligned in-range 16-column tile");

  auto output = torch::empty(
      {reduction / 16, 16, 16},
      input.options().dtype(torch::kFloat32));
  sm75_wmma_prefix_kernel<<<
      1,
      32,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
          output.data_ptr<float>(),
          channels,
          reduction,
          static_cast<int>(tile_col));
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(
      error == cudaSuccess,
      "SM75 WMMA prefix kernel launch failed: ",
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
    *,
    expected_witness_sha256: str,
    expected_cuda_result_sha256: str,
) -> dict[str, Any]:
    witness_path = Path(witness_path)
    cuda_result_path = Path(cuda_result_path)
    result: dict[str, Any] = {
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
    }
    witness = _load_npz(witness_path)
    cuda = _load_npz(cuda_result_path)
    result["center_weight"] = _require_array(
        witness,
        "center_weight",
        dtype=np.float16,
        shape=(REDUCTION, CHANNELS),
    )
    result["wmma_input_window"] = _require_array(
        cuda,
        "wmma_input_window",
        dtype=np.float16,
        shape=(ROWS, REDUCTION),
    )
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


def validate_tile_col(tile_col: int) -> int:
    if (
        not isinstance(tile_col, int)
        or tile_col < 0
        or tile_col % TILE_WIDTH
        or tile_col + TILE_WIDTH > CHANNELS
    ):
        raise ValueError(
            "tile_col must be a 16-aligned integer selecting an in-range tile"
        )
    return tile_col


def validate_prefixes(value: np.ndarray) -> np.ndarray:
    prefixes = np.asarray(value)
    expected_shape = (PREFIX_COUNT, ROWS, TILE_WIDTH)
    if prefixes.dtype != np.float32:
        raise ValueError(
            f"WMMA prefixes must have dtype float32, got {prefixes.dtype}"
        )
    if prefixes.shape != expected_shape:
        raise ValueError(
            f"WMMA prefixes must have shape {expected_shape}, "
            f"got {prefixes.shape}"
        )
    if not np.all(np.isfinite(prefixes)):
        raise ValueError("WMMA prefixes contain non-finite values")
    return np.ascontiguousarray(prefixes)


def _metric(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    return {
        "exact": bool(np.array_equal(actual, expected)),
        "nonzero": int(np.count_nonzero(actual != expected)),
        "max_abs": float(np.max(np.abs(difference), initial=0.0)),
        "mean_abs": float(np.mean(np.abs(difference))),
    }


def analyze_prefixes(
    prefixes: np.ndarray,
    inputs: dict[str, Any],
    *,
    tile_col: int,
) -> dict[str, Any]:
    tile_col = validate_tile_col(tile_col)
    prefixes = validate_prefixes(prefixes)
    tile = slice(tile_col, tile_col + TILE_WIDTH)
    final_row = prefixes[-1, SELECTED_WINDOW_ROW]
    admitted_wmma = inputs["wmma_fp32_row"][tile]
    final_metric = _metric(final_row, admitted_wmma)
    if not final_metric["exact"]:
        raise ValueError(
            "final WMMA prefix does not exactly authenticate the admitted "
            f"WMMA row tile: {final_metric['nonzero']} mismatches"
        )
    regular_fp32 = inputs["cublas_regular_fp32_row"][tile]
    admitted_fp16 = inputs["cublas_tensor_fp16_unbiased_row"][tile]
    regular_fp16 = inputs["cublas_regular_fp16_unbiased_row"][tile]
    return {
        "tile_col": tile_col,
        "tile_stop": tile_col + TILE_WIDTH,
        "selected_window_row": SELECTED_WINDOW_ROW,
        "k_prefixes": list(range(16, REDUCTION + 1, 16)),
        "final_prefix_exact_admitted_wmma_row": True,
        "final_fp32_vs_regular": _metric(final_row, regular_fp32),
        "final_fp16_vs_admitted_wmma": _metric(
            final_row.astype(np.float16),
            admitted_fp16,
        ),
        "final_fp16_vs_regular": _metric(
            final_row.astype(np.float16),
            regular_fp16,
        ),
    }


def _target_section(
    text: str,
    *,
    kind: str,
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
            and "sm75_wmma_prefix_kernel" in stripped
        ) or (
            kind == "sass"
            and stripped.startswith("Function :")
            and "sm75_wmma_prefix_kernel" in stripped
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
                if "{" in section_line:
                    body_started = True
                brace_depth -= section_line.count("}")
                if body_started and brace_depth == 0:
                    return architecture, symbol, section
            raise ValueError("target SM75 prefix-kernel PTX is incomplete")
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
    raise ValueError(f"compiler evidence lacks target SM75 prefix-kernel {kind}")


def classify_compiler_evidence(*, ptx: str, sass: str) -> dict[str, Any]:
    ptx_arch, ptx_symbol, ptx_section = _target_section(ptx, kind="ptx")
    sass_arch, sass_symbol, sass_section = _target_section(sass, kind="sass")
    if ptx_symbol != sass_symbol:
        raise ValueError("PTX and SASS prefix-kernel symbols differ")
    ptx_lines = [
        line.strip()
        for line in ptx_section
        if (
            "wmma.mma.sync.aligned" in line
            and "m16n16k16" in line
            and "f32" in line
        )
    ]
    sass_lines = [
        line.strip() for line in sass_section if "HMMA.1688" in line
    ]
    if not ptx_lines:
        raise ValueError("target prefix-kernel PTX lacks WMMA m16n16k16")
    if not sass_lines:
        raise ValueError("target prefix-kernel SASS lacks HMMA.1688")
    return {
        "effective_ptx_architecture": ptx_arch,
        "effective_cubin_architecture": sass_arch,
        "ptx_target_symbol": ptx_symbol,
        "sass_target_symbol": sass_symbol,
        "ptx_wmma_m16n16k16_count": len(ptx_lines),
        "sass_hmma_1688_count": len(sass_lines),
        "ptx_matching_lines": ptx_lines,
        "sass_matching_lines": sass_lines,
    }


def _build_extension():
    from torch.utils.cpp_extension import load_inline

    previous_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5+PTX"
    try:
        return load_inline(
            name="trellis2mlx_sm75_wmma_prefix_v1",
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=["sm75_wmma_prefix_cuda"],
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
    if not module_path.is_file():
        raise RuntimeError(f"compiled extension is missing: {module_path}")
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
    window: np.ndarray,
    weight: np.ndarray,
    *,
    tile_col: int,
) -> tuple[np.ndarray, dict[str, Any]]:
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
    input_cuda = torch.from_numpy(window).to(device="cuda").contiguous()
    weight_cuda = torch.from_numpy(weight).to(device="cuda").contiguous()
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = module.sm75_wmma_prefix_cuda(
        input_cuda,
        weight_cuda,
        tile_col,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return output.detach().cpu().numpy(), {
        "backend": "cuda",
        "device": device,
        "kernel": KERNEL_IDENTITY,
        "effective_compute_capability": "7.5",
        "elapsed_seconds": elapsed,
        "compiler_evidence": compiler_evidence,
    }


def _validate_effective_route(route: Any) -> dict[str, Any]:
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
    for name, value in {
        "effective_ptx_architecture": "sm_75",
        "effective_cubin_architecture": "sm_75",
    }.items():
        if evidence.get(name) != value:
            raise ValueError(
                f"compiler_evidence.{name} must be {value!r}, "
                f"got {evidence.get(name)!r}"
            )
    for name in ("ptx_wmma_m16n16k16_count", "sass_hmma_1688_count"):
        value = evidence.get(name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"compiler_evidence.{name} must be positive")
    return route


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
    tile_col: int,
    output_json: Path,
    output_npz: Path,
    backend: Callable[..., tuple[np.ndarray, dict[str, Any]]] = run_cuda_backend,
) -> dict[str, Any]:
    output_json = Path(output_json)
    output_npz = Path(output_npz)
    if output_json.resolve() == output_npz.resolve():
        output_json.unlink(missing_ok=True)
        raise ValueError("output_json and output_npz must be distinct paths")
    output_json.unlink(missing_ok=True)
    output_npz.unlink(missing_ok=True)
    phase = "input_validation"
    try:
        tile_col = validate_tile_col(tile_col)
        inputs = load_probe_inputs(
            Path(witness_path),
            Path(cuda_result_path),
            expected_witness_sha256=expected_witness_sha256,
            expected_cuda_result_sha256=expected_cuda_result_sha256,
        )
        phase = "backend_execution"
        prefixes, route = backend(
            inputs["wmma_input_window"],
            inputs["center_weight"],
            tile_col=tile_col,
        )
        phase = "backend_output_validation"
        prefixes = validate_prefixes(prefixes)
        route = _validate_effective_route(route)
        analysis = analyze_prefixes(prefixes, inputs, tile_col=tile_col)
        k_prefixes = np.arange(
            16,
            REDUCTION + 1,
            16,
            dtype=np.int32,
        )
        phase = "primary_publication"
        _write_npz_atomic(
            output_npz,
            k_prefixes=k_prefixes,
            wmma_prefix_fp32=prefixes,
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
    parser.add_argument("--tile-col", type=int, default=DEFAULT_TILE_COL)
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
        tile_col=args.tile_col,
        output_json=args.output_json,
        output_npz=args.output_npz,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
