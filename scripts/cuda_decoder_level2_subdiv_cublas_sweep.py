#!/usr/bin/env python3
"""Identify the CUDA GEMM schedule for the level-two subdivision head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cuda_decoder_block0_gemm_witness import (  # noqa: E402
    CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    CUBLAS_STATUS_NOT_SUPPORTED,
    CUBLAS_STATUS_SUCCESS,
    EXPECTED_DEVICE,
    EXPECTED_TORCH,
    LEGACY_CUBLAS_EXPLICIT_ALGORITHM_IDS,
    _canonical_sha256,
    _invoke_cublas_gemm_ex,
    _load_cublas_gemm_ex,
    _to_cpu_numpy_preserve_dtype,
    _write_json,
    sha256_array,
    sha256_file,
)
from decoder_level2_subdiv_trace_contract import (  # noqa: E402
    load_decoder_level2_subdiv_trace,
)


SCHEMA = "trellis2mlx.cuda_decoder_level2_subdiv_cublas_sweep.v1"
REQUESTED_ROUTE = {
    "operation": "shape_decoder.level2.upsample.to_subdiv",
    "torch": EXPECTED_TORCH,
    "cuda_device": EXPECTED_DEVICE,
    "projection": "torch.nn.functional.linear",
    "explicit_backend": "legacy_cublasGemmEx",
    "bias_variants": ["separate-fp16-add", "fused-beta-one"],
}


def compare_projection_logits(
    source: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    source = np.asarray(source)
    candidate = np.asarray(candidate)
    if (
        source.ndim != 2
        or source.shape[0] == 0
        or source.shape != candidate.shape
    ):
        raise ValueError(
            "projection logits must have the same nonempty two-dimensional shape"
        )
    if source.dtype != np.float16 or candidate.dtype != np.float16:
        raise ValueError("projection logits must both have dtype float16")
    if not np.isfinite(source).all() or not np.isfinite(candidate).all():
        raise ValueError("projection logits contain non-finite values")

    difference = np.abs(
        source.astype(np.float32) - candidate.astype(np.float32)
    )
    decision_flips = (source > 0) != (candidate > 0)
    return {
        "exact": bool(np.array_equal(source, candidate)),
        "nonzero_count": int(np.count_nonzero(source != candidate)),
        "mean_abs": float(difference.mean()),
        "rms": float(np.sqrt(np.mean(np.square(difference)))),
        "max_abs": float(difference.max()),
        "decision_flip_count": int(np.count_nonzero(decision_flips)),
        "rows_with_decision_flip": int(np.count_nonzero(decision_flips.any(axis=1))),
        "source_positive_candidate_nonpositive": int(
            np.count_nonzero((source > 0) & (candidate <= 0))
        ),
        "source_nonpositive_candidate_positive": int(
            np.count_nonzero((source <= 0) & (candidate > 0))
        ),
    }


def _candidate_result(
    *,
    status: int,
    candidate: Any | None,
    source_logits: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": int(status),
        "supported": status == CUBLAS_STATUS_SUCCESS,
    }
    if status == CUBLAS_STATUS_NOT_SUPPORTED:
        if candidate is not None:
            raise ValueError("unsupported cuBLAS route returned output")
        return result
    if status != CUBLAS_STATUS_SUCCESS:
        raise RuntimeError(f"cuBLAS route failed with status {status}")
    if candidate is None:
        raise ValueError("successful cuBLAS route returned no output")
    candidate_np = _to_cpu_numpy_preserve_dtype(candidate)
    result.update(compare_projection_logits(source_logits, candidate_np))
    result["sha256"] = sha256_array(candidate_np)
    return result


def _run_cuda_sweep(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

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
    reduced_precision = bool(
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    )
    if reduced_precision is not True:
        raise ValueError(
            "default CUDA route disabled FP16 reduced-precision reduction"
        )

    x = torch.from_numpy(arrays["level2_block7_output"].copy()).to(
        device="cuda",
        dtype=torch.float16,
    )
    source_weight = torch.from_numpy(
        arrays["level2_upsample_subdiv_weight"].copy()
    ).to(device="cuda", dtype=torch.float16)
    weight = source_weight.T.contiguous()
    bias = torch.from_numpy(
        arrays["level2_upsample_subdiv_bias"].copy()
    ).to(device="cuda", dtype=torch.float16)
    source_logits = arrays["level2_upsample_subdiv_logits"]

    torch.cuda.synchronize()
    source_started = time.perf_counter()
    with torch.no_grad():
        torch_f_linear = functional.linear(x, source_weight, bias)
        torch_product = x @ weight
        torch_separate = torch_product + bias
    torch.cuda.synchronize()
    source_seconds = time.perf_counter() - source_started

    torch_f_linear_np = _to_cpu_numpy_preserve_dtype(torch_f_linear)
    self_authentication = compare_projection_logits(
        source_logits,
        torch_f_linear_np,
    )
    if not self_authentication["exact"]:
        raise ValueError(
            "live CUDA torch.nn.functional.linear does not reproduce source trace"
        )
    separate_analysis = compare_projection_logits(
        source_logits,
        _to_cpu_numpy_preserve_dtype(torch_separate),
    )

    library, gemm_ex = _load_cublas_gemm_ex()
    handle = int(torch.cuda.current_blas_handle())
    rows = int(x.shape[0])

    def run_algorithm(algorithm_id: int, *, fused_bias: bool):
        if fused_bias:
            output = bias.reshape(1, -1).expand(rows, -1).contiguous()
            beta = 1.0
        else:
            output = torch.empty(
                (rows, int(bias.shape[0])),
                device="cuda",
                dtype=torch.float16,
            )
            beta = 0.0
        status = _invoke_cublas_gemm_ex(
            gemm_ex,
            handle=handle,
            x=x,
            weight=weight,
            output=output,
            algorithm_id=algorithm_id,
            beta=beta,
        )
        torch.cuda.synchronize()
        if status != CUBLAS_STATUS_SUCCESS:
            return int(status), None
        if not fused_bias:
            output = output + bias
            torch.cuda.synchronize()
        return int(status), output

    sweep_started = time.perf_counter()
    algorithm_results = []
    exact_separate = []
    exact_fused = []
    for algorithm_id in LEGACY_CUBLAS_EXPLICIT_ALGORITHM_IDS:
        separate_status, separate = run_algorithm(
            algorithm_id,
            fused_bias=False,
        )
        fused_status, fused = run_algorithm(
            algorithm_id,
            fused_bias=True,
        )
        separate_result = _candidate_result(
            status=separate_status,
            candidate=separate,
            source_logits=source_logits,
        )
        fused_result = _candidate_result(
            status=fused_status,
            candidate=fused,
            source_logits=source_logits,
        )
        if separate_result.get("exact"):
            exact_separate.append(int(algorithm_id))
        if fused_result.get("exact"):
            exact_fused.append(int(algorithm_id))
        algorithm_results.append(
            {
                "algorithm_id": int(algorithm_id),
                "separate_bias": separate_result,
                "fused_beta_bias": fused_result,
            }
        )

    default_separate_status, default_separate = run_algorithm(
        CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        fused_bias=False,
    )
    default_fused_status, default_fused = run_algorithm(
        CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        fused_bias=True,
    )
    sweep_seconds = time.perf_counter() - sweep_started
    default_result = {
        "algorithm_id": CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        "separate_bias": _candidate_result(
            status=default_separate_status,
            candidate=default_separate,
            source_logits=source_logits,
        ),
        "fused_beta_bias": _candidate_result(
            status=default_fused_status,
            candidate=default_fused,
            source_logits=source_logits,
        ),
    }
    del library

    return {
        "runtime": {
            "torch": torch.__version__,
            "cuda_device": device,
            "allow_fp16_reduced_precision_reduction": reduced_precision,
            "source_self_authentication_seconds": source_seconds,
            "explicit_sweep_seconds": sweep_seconds,
        },
        "matrix": {
            "rows": rows,
            "reduction": int(x.shape[1]),
            "channels": int(bias.shape[0]),
            "input_dtype": str(arrays["level2_block7_output"].dtype),
            "output_dtype": str(source_logits.dtype),
        },
        "source_self_authentication": {
            "route": "torch.nn.functional.linear",
            "torch_f_linear_exact_source": True,
            "metrics": self_authentication,
            "sha256": sha256_array(torch_f_linear_np),
        },
        "torch_matmul_separate_bias": separate_analysis,
        "algorithm_inventory": {
            "requested_ids": list(LEGACY_CUBLAS_EXPLICIT_ALGORITHM_IDS),
            "complete": list(LEGACY_CUBLAS_EXPLICIT_ALGORITHM_IDS)
            == [*range(0, 24), *range(100, 116)],
        },
        "default_tensor_op_result": default_result,
        "algorithm_results": algorithm_results,
        "exact_matches": {
            "separate_bias_algorithm_ids": exact_separate,
            "fused_beta_bias_algorithm_ids": exact_fused,
        },
    }


def _validate_cuda_result(
    result: dict[str, Any],
    *,
    expected_rows: int,
) -> dict[str, Any]:
    runtime = result.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("CUDA sweep omitted runtime identity")
    for name, expected in {
        "torch": EXPECTED_TORCH,
        "cuda_device": EXPECTED_DEVICE,
    }.items():
        if runtime.get(name) != expected:
            raise ValueError(
                f"CUDA sweep effective {name} must be {expected!r}, "
                f"got {runtime.get(name)!r}"
            )
    authentication = result.get("source_self_authentication")
    if (
        not isinstance(authentication, dict)
        or authentication.get("torch_f_linear_exact_source") is not True
    ):
        raise ValueError("CUDA sweep did not authenticate F.linear to source")

    requested = list(LEGACY_CUBLAS_EXPLICIT_ALGORITHM_IDS)
    if result.get("algorithm_inventory") != {
        "requested_ids": requested,
        "complete": True,
    }:
        raise ValueError("CUDA sweep returned an incomplete algorithm inventory")
    algorithm_results = result.get("algorithm_results")
    if not isinstance(algorithm_results, list):
        raise ValueError("CUDA sweep omitted algorithm results")
    result_ids = [
        entry.get("algorithm_id") if isinstance(entry, dict) else None
        for entry in algorithm_results
    ]
    if result_ids != requested:
        raise ValueError(
            "CUDA sweep algorithm result IDs do not match the requested inventory"
        )
    derived_separate = []
    derived_fused = []
    for entry in algorithm_results:
        separate = entry.get("separate_bias")
        fused = entry.get("fused_beta_bias")
        if not isinstance(separate, dict) or not isinstance(fused, dict):
            raise ValueError("CUDA sweep algorithm result omitted a bias variant")
        if separate.get("exact") is True:
            derived_separate.append(entry["algorithm_id"])
        if fused.get("exact") is True:
            derived_fused.append(entry["algorithm_id"])
    if result.get("exact_matches") != {
        "separate_bias_algorithm_ids": derived_separate,
        "fused_beta_bias_algorithm_ids": derived_fused,
    }:
        raise ValueError(
            "CUDA sweep exact-match summary does not match algorithm results"
        )

    matrix = result.get("matrix")
    if (
        not isinstance(matrix, dict)
        or matrix.get("rows") != expected_rows
        or matrix.get("reduction") != 256
        or matrix.get("channels") != 8
    ):
        raise ValueError("CUDA sweep returned the wrong effective matrix shape")
    return {
        **REQUESTED_ROUTE,
        "torch": runtime["torch"],
        "cuda_device": runtime["cuda_device"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-trace", required=True, type=Path)
    parser.add_argument("--expected-source-trace-sha256", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int, default=178426)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "failure_phase": "request_validation",
        "last_trustworthy_phase": None,
        "source_trace": {
            "path": str(args.source_trace),
            "requested_sha256": args.expected_source_trace_sha256,
        },
        "requested_route": REQUESTED_ROUTE,
    }
    if args.source_trace.resolve() == args.output_json.resolve():
        report.update(
            {
                "error": (
                    "ValueError: source trace and output JSON paths must be "
                    "distinct"
                ),
                "traceback": None,
            }
        )
        print(json.dumps(report, sort_keys=True), file=sys.stderr)
        return 1
    try:
        args.output_json.unlink(missing_ok=True)
        report["last_trustworthy_phase"] = "output_path_validated"
        report["failure_phase"] = "input_validation"
        if not _canonical_sha256(args.expected_source_trace_sha256):
            raise ValueError(
                "expected source trace sha256 must be canonical lowercase hex"
            )
        actual_digest = sha256_file(args.source_trace)
        if actual_digest != args.expected_source_trace_sha256:
            raise ValueError(
                "source trace sha256 mismatch: expected "
                f"{args.expected_source_trace_sha256}, got {actual_digest}"
            )
        arrays = load_decoder_level2_subdiv_trace(args.source_trace)
        rows = int(arrays["level2_block7_output"].shape[0])
        if rows != args.expected_rows:
            raise ValueError(
                f"source trace must contain {args.expected_rows} rows, got {rows}"
            )
        report["source_trace"].update(
            {
                "effective_sha256": actual_digest,
                "size_bytes": args.source_trace.stat().st_size,
                "rows": rows,
                "array_sha256": {
                    name: sha256_array(value)
                    for name, value in arrays.items()
                },
            }
        )
        report["last_trustworthy_phase"] = "input_validated"
        report["failure_phase"] = "cuda_execution"
        result = _run_cuda_sweep(arrays)
        effective_route = _validate_cuda_result(
            result,
            expected_rows=args.expected_rows,
        )
        report.update(result)
        report["effective_route"] = effective_route
        report["status"] = "done"
        report["failure_phase"] = None
        report["last_trustworthy_phase"] = "report_reopened_exact"
        _write_json(args.output_json, report)
        reopened = json.loads(args.output_json.read_text())
        if reopened != report:
            raise ValueError("published sweep report does not reopen exactly")
        return 0
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(args.output_json, report)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
