#!/usr/bin/env python3
"""Recover the hidden K-position composition groups of one SM75 WMMA."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import traceback
from typing import Any, Callable

import numpy as np


SCHEMA = "trellis2mlx.cuda_sm75_wmma_pairwise_group_probe.v1"
EXPECTED_TORCH = "2.10.0+cu128"
EXPECTED_DEVICE = "Tesla T4"
CASE_COUNT = 120
MATRIX_SIZE = 16
KERNEL_IDENTITY = "direct_sm75_wmma_pairwise_group_m16n16k16"

A1_BITS = 0xC1E6
B1_BITS = 0xAD61
A2_BITS = 0xA3CB
B2_BITS = 0x3734
ACCUMULATOR_BITS = 0x3B1507BA
SAME_GROUP_RESULT_BITS = 0x3E792107
SEPARATE_GROUP_RESULT_BITS = 0x3E792108

CPP_SOURCE = """
torch::Tensor sm75_wmma_pairwise_group_cuda(
    torch::Tensor matrix_a,
    torch::Tensor matrix_b,
    double accumulator);
"""

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;

__global__ void sm75_wmma_pairwise_group_kernel(
    const half* matrix_a,
    const half* matrix_b,
    float* output,
    float accumulator,
    int cases) {
  const int case_index = static_cast<int>(blockIdx.x);
  if (case_index >= cases) {
    return;
  }

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
      matrix_a_fragment;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
      matrix_b_fragment;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float>
      accumulator_fragment;

  const int case_offset = case_index * 16 * 16;
  wmma::load_matrix_sync(
      matrix_a_fragment,
      matrix_a + case_offset,
      16);
  wmma::load_matrix_sync(
      matrix_b_fragment,
      matrix_b + case_offset,
      16);
  wmma::fill_fragment(accumulator_fragment, accumulator);
  wmma::mma_sync(
      accumulator_fragment,
      matrix_a_fragment,
      matrix_b_fragment,
      accumulator_fragment);
  wmma::store_matrix_sync(
      output + case_index * 16 * 16,
      accumulator_fragment,
      16,
      wmma::mem_row_major);
}

torch::Tensor sm75_wmma_pairwise_group_cuda(
    torch::Tensor matrix_a,
    torch::Tensor matrix_b,
    double accumulator) {
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
          matrix_a.size(0) == 120 &&
          matrix_a.size(1) == 16 &&
          matrix_a.size(2) == 16,
      "inputs must have shape [120, 16, 16]");

  auto output = torch::empty(
      {120, 16, 16},
      matrix_a.options().dtype(torch::kFloat32));
  sm75_wmma_pairwise_group_kernel<<<
      120,
      32,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const half*>(matrix_a.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(matrix_b.data_ptr<at::Half>()),
          output.data_ptr<float>(),
          static_cast<float>(accumulator),
          120);
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(
      error == cudaSuccess,
      "SM75 WMMA pairwise-group kernel launch failed: ",
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


def _float16_from_bits(bits: int) -> np.float16:
    return np.array(bits, dtype=np.uint16).view(np.float16)[()]


def _float32_from_bits(bits: int) -> np.float32:
    return np.array(bits, dtype=np.uint32).view(np.float32)[()]


def _float32_bits(value: np.float32) -> int:
    return int(np.array(value, dtype=np.float32).view(np.uint32))


def generate_cases() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs = np.asarray(
        list(itertools.combinations(range(MATRIX_SIZE), 2)),
        dtype=np.int32,
    )
    if pairs.shape != (CASE_COUNT, 2):
        raise AssertionError("pair inventory is incomplete")
    matrix_a = np.zeros(
        (CASE_COUNT, MATRIX_SIZE, MATRIX_SIZE),
        dtype=np.float16,
    )
    matrix_b = np.zeros_like(matrix_a)
    a1 = _float16_from_bits(A1_BITS)
    b1 = _float16_from_bits(B1_BITS)
    a2 = _float16_from_bits(A2_BITS)
    b2 = _float16_from_bits(B2_BITS)
    for case_index, (left, right) in enumerate(pairs):
        matrix_a[case_index, :, left] = a1
        matrix_a[case_index, :, right] = a2
        matrix_b[case_index, left, :] = b1
        matrix_b[case_index, right, :] = b2
    return pairs, matrix_a, matrix_b


def _exact_product_bits(left_bits: int, right_bits: int) -> int:
    left = np.float32(_float16_from_bits(left_bits))
    right = np.float32(_float16_from_bits(right_bits))
    return _float32_bits(np.float32(left * right))


def _tensor_core_five_term_bits(terms: list[int]) -> int:
    """Concrete form of the paper's unnormalized five-term Turing adder."""

    if len(terms) != 5:
        raise ValueError("the Turing primitive requires exactly five terms")
    values = [int(value) & 0xFFFFFFFF for value in terms]

    def magnitude_rank(value: int) -> tuple[int, int]:
        return ((value >> 23) & 0xFF, value & 0x7FFFFF)

    maximum_index = max(
        range(len(values)),
        key=lambda index: magnitude_rank(values[index]),
    )
    maximum = values[maximum_index]
    sign = (maximum >> 31) & 1
    exponent = (maximum >> 23) & 0xFF
    mantissa = ((1 if exponent else 0) << 23) | (maximum & 0x7FFFFF)
    accumulator_mask = (1 << 27) - 1

    for index, value in enumerate(values):
        if index == maximum_index:
            value = 0
        term_sign = (value >> 31) & 1
        term_exponent = (value >> 23) & 0xFF
        term_mantissa = (
            ((1 if term_exponent else 0) << 23) | (value & 0x7FFFFF)
        )
        exponent_difference = (exponent - term_exponent) & 0xFF
        shifted = (
            term_mantissa >> exponent_difference
            if exponent_difference < 27
            else 0
        )
        is_addition = sign == term_sign or mantissa == 0
        larger = max(mantissa, shifted)
        smaller = min(mantissa, shifted)
        result = (
            larger + smaller if is_addition else larger - smaller
        ) & accumulator_mask
        if mantissa == 0 and term_mantissa == 0:
            result_sign = sign | term_sign
        elif mantissa == shifted and not is_addition:
            result_sign = 1
        elif mantissa > shifted:
            result_sign = sign
        else:
            result_sign = term_sign
        mantissa = result
        sign = result_sign

    if mantissa == 0:
        return 0
    bit_position = mantissa.bit_length()
    exponent = (exponent + bit_position - 24) & 0xFF
    if bit_position == 24:
        fraction = mantissa & 0x7FFFFF
    elif bit_position > 24:
        fraction = (mantissa >> (bit_position - 24)) & 0x7FFFFF
    else:
        fraction = (mantissa << (24 - bit_position)) & 0x7FFFFF
    return (sign << 31) | (exponent << 23) | fraction


def formal_discriminator_check() -> dict[str, str]:
    product_one = _exact_product_bits(A1_BITS, B1_BITS)
    product_two = _exact_product_bits(A2_BITS, B2_BITS)
    same = _tensor_core_five_term_bits(
        [product_one, product_two, 0, 0, ACCUMULATOR_BITS]
    )
    first_then_second = _tensor_core_five_term_bits(
        [
            product_two,
            0,
            0,
            0,
            _tensor_core_five_term_bits(
                [product_one, 0, 0, 0, ACCUMULATOR_BITS]
            ),
        ]
    )
    second_then_first = _tensor_core_five_term_bits(
        [
            product_one,
            0,
            0,
            0,
            _tensor_core_five_term_bits(
                [product_two, 0, 0, 0, ACCUMULATOR_BITS]
            ),
        ]
    )
    expected = (
        SAME_GROUP_RESULT_BITS,
        SEPARATE_GROUP_RESULT_BITS,
        SEPARATE_GROUP_RESULT_BITS,
    )
    if (same, first_then_second, second_then_first) != expected:
        raise AssertionError("formal discriminator constants are inconsistent")
    return {
        "same_group_result_bits": f"{same:#010x}",
        "separate_group_forward_result_bits": f"{first_then_second:#010x}",
        "separate_group_reverse_result_bits": f"{second_then_first:#010x}",
    }


def validate_outputs(value: np.ndarray) -> np.ndarray:
    outputs = np.asarray(value)
    expected_shape = (CASE_COUNT, MATRIX_SIZE, MATRIX_SIZE)
    if outputs.dtype != np.float32:
        raise ValueError(
            f"WMMA outputs must have dtype float32, got {outputs.dtype}"
        )
    if outputs.shape != expected_shape:
        raise ValueError(
            f"WMMA outputs must have shape {expected_shape}, got {outputs.shape}"
        )
    if not np.all(np.isfinite(outputs)):
        raise ValueError("WMMA outputs contain non-finite values")
    return np.ascontiguousarray(outputs)


def _equivalence_groups(
    same_pairs: list[tuple[int, int]],
) -> list[list[int]] | None:
    adjacency = [{position} for position in range(MATRIX_SIZE)]
    for left, right in same_pairs:
        adjacency[left].add(right)
        adjacency[right].add(left)
    for left in range(MATRIX_SIZE):
        for middle in adjacency[left]:
            if not adjacency[middle].issubset(adjacency[left]):
                return None
    remaining = set(range(MATRIX_SIZE))
    groups: list[list[int]] = []
    while remaining:
        start = min(remaining)
        group = sorted(adjacency[start])
        groups.append(group)
        remaining.difference_update(group)
    return groups


def analyze_outputs(
    outputs: np.ndarray,
    pairs: np.ndarray,
) -> dict[str, Any]:
    outputs = validate_outputs(outputs)
    pair_array = np.asarray(pairs)
    if pair_array.dtype != np.int32 or pair_array.shape != (CASE_COUNT, 2):
        raise ValueError("pairs must have dtype int32 and shape (120, 2)")
    bits = outputs.view(np.uint32)
    same_cells = bits == SAME_GROUP_RESULT_BITS
    separate_cells = bits == SEPARATE_GROUP_RESULT_BITS
    unexpected_cells = int(np.count_nonzero(~(same_cells | separate_cells)))
    same_pairs: list[tuple[int, int]] = []
    separate_pair_count = 0
    mixed_pair_count = 0
    for case_index, pair in enumerate(pair_array):
        if bool(np.all(same_cells[case_index])):
            same_pairs.append((int(pair[0]), int(pair[1])))
        elif bool(np.all(separate_cells[case_index])):
            separate_pair_count += 1
        else:
            mixed_pair_count += 1

    groups: list[list[int]] | None = None
    if unexpected_cells:
        classification = "unexpected_output"
    elif mixed_pair_count:
        classification = "mixed_pair_output"
    else:
        groups = _equivalence_groups(same_pairs)
        if groups is None:
            classification = "non_equivalence_relation"
        elif sorted(map(len, groups)) == [4, 4, 4, 4]:
            classification = "four_groups_of_four"
        else:
            classification = "valid_equivalence_other_shape"
    return {
        "classification": classification,
        "groups": groups,
        "uniform_same_group_pairs": len(same_pairs),
        "uniform_separate_group_pairs": separate_pair_count,
        "unexpected_cells": unexpected_cells,
        "mixed_pairs": mixed_pair_count,
        "same_group_result_bits": f"{SAME_GROUP_RESULT_BITS:#010x}",
        "separate_group_result_bits": f"{SEPARATE_GROUP_RESULT_BITS:#010x}",
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
            and "sm75_wmma_pairwise_group_kernel" in stripped
        ) or (
            kind == "sass"
            and stripped.startswith("Function :")
            and "sm75_wmma_pairwise_group_kernel" in stripped
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
            raise ValueError("target SM75 pairwise-kernel PTX is incomplete")
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
    raise ValueError(
        f"compiler evidence lacks target SM75 pairwise-kernel {kind}"
    )


def classify_compiler_evidence(*, ptx: str, sass: str) -> dict[str, Any]:
    ptx_arch, ptx_symbol, ptx_section = _target_section(ptx, kind="ptx")
    sass_arch, sass_symbol, sass_section = _target_section(sass, kind="sass")
    if ptx_symbol != sass_symbol:
        raise ValueError("PTX and SASS pairwise-kernel symbols differ")
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
        raise ValueError("target pairwise-kernel PTX lacks WMMA m16n16k16")
    if not sass_lines:
        raise ValueError("target pairwise-kernel SASS lacks HMMA.1688")
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
            name="trellis2mlx_sm75_wmma_pairwise_group_v1",
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=["sm75_wmma_pairwise_group_cuda"],
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
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
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
    matrix_a_cuda = torch.from_numpy(matrix_a).to(device="cuda").contiguous()
    matrix_b_cuda = torch.from_numpy(matrix_b).to(device="cuda").contiguous()
    accumulator = float(_float32_from_bits(ACCUMULATOR_BITS))
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = module.sm75_wmma_pairwise_group_cuda(
        matrix_a_cuda,
        matrix_b_cuda,
        accumulator,
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
    ptx_symbol = evidence.get("ptx_target_symbol")
    sass_symbol = evidence.get("sass_target_symbol")
    if (
        not isinstance(ptx_symbol, str)
        or "sm75_wmma_pairwise_group_kernel" not in ptx_symbol
        or ptx_symbol != sass_symbol
    ):
        raise ValueError(
            "compiler_evidence target symbol must identify the same "
            "SM75 pairwise-group kernel in PTX and SASS"
        )
    for name, expected_count in {
        "ptx_wmma_m16n16k16_count": 1,
        "sass_hmma_1688_count": 4,
    }.items():
        if evidence.get(name) != expected_count:
            raise ValueError(
                f"compiler_evidence.{name} must contain exactly one logical "
                f"WMMA lowering ({expected_count}), got {evidence.get(name)!r}"
            )
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
    output_json: Path,
    output_npz: Path,
    backend: Callable[
        [np.ndarray, np.ndarray],
        tuple[np.ndarray, dict[str, Any]],
    ] = run_cuda_backend,
) -> dict[str, Any]:
    output_json = Path(output_json)
    output_npz = Path(output_npz)
    if output_json.resolve() == output_npz.resolve():
        output_json.unlink(missing_ok=True)
        raise ValueError("output_json and output_npz must be distinct paths")
    output_json.unlink(missing_ok=True)
    output_npz.unlink(missing_ok=True)
    phase = "case_generation"
    try:
        formal = formal_discriminator_check()
        pairs, matrix_a, matrix_b = generate_cases()
        phase = "backend_execution"
        outputs, route = backend(matrix_a, matrix_b)
        phase = "backend_output_validation"
        outputs = validate_outputs(outputs)
        route = _validate_effective_route(route)
        analysis = analyze_outputs(outputs, pairs)
        phase = "primary_publication"
        _write_npz_atomic(
            output_npz,
            pair_indices=pairs,
            wmma_pairwise_fp32=outputs,
        )
        report: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "done",
            "failure_phase": None,
            "effective_route": route,
            "discriminator": {
                "a1_bits": f"{A1_BITS:#06x}",
                "b1_bits": f"{B1_BITS:#06x}",
                "a2_bits": f"{A2_BITS:#06x}",
                "b2_bits": f"{B2_BITS:#06x}",
                "accumulator_bits": f"{ACCUMULATOR_BITS:#010x}",
                "formal_check": formal,
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
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = run_probe(
        output_json=args.output_json,
        output_npz=args.output_npz,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
