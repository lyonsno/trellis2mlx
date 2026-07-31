"""Capture an evidence-bound MLX level-two block-zero child witness."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.decoder_level1_trace_contract import (
    decoder_level1_trace_input_sha256,
)
from scripts.decoder_level2_block0_trace_contract import (
    LEVEL2_BLOCK0_NORM_BOUNDARY_ROUTE,
    TRACE_RUN_SCHEMA,
    write_decoder_level2_block0_trace_npz,
)
from scripts.run_mlx_decoder_level1_trace import (
    _failure_sibling,
    _load_parent_state,
    _load_turing_rsqrt_lut,
    _sha256_file,
    _validate_digest,
    _validate_repo_state,
    _write_report,
)


ROUTE = "mlx-shape-decoder-level2-block0-trace"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level0-trace", required=True, type=Path)
    parser.add_argument("--expected-level0-trace-sha256", required=True)
    parser.add_argument("--shape-decoder-checkpoint", required=True, type=Path)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--decoder-silu-lut", required=True, type=Path)
    parser.add_argument("--expected-decoder-silu-lut-sha256", required=True)
    parser.add_argument("--turing-rsqrt-lut", required=True, type=Path)
    parser.add_argument("--expected-turing-rsqrt-lut-sha256", required=True)
    parser.add_argument("--expected-repo-commit", required=True)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def _level2_block0_norm_route_identity(decoder: Any) -> dict[str, Any]:
    from trellmlx.models.shape_slat_decoder import SparseConvNeXtBlock3d

    blocks = [
        block
        for block in decoder.blocks[2]
        if isinstance(block, SparseConvNeXtBlock3d)
    ]
    if len(blocks) != 8:
        raise ValueError(
            "level2 block0 norm route requires eight level-two ConvNeXt "
            f"blocks, got {len(blocks)}"
        )
    norm = blocks[0].norm
    actual = {
        "parameter_dtype": str(norm.weight.dtype).rsplit(".", 1)[-1],
        "hidden_width": int(norm.weight.shape[0]),
        "affine": bool(norm.affine),
        "shape_flow_layernorm": bool(norm.shape_flow_layernorm),
        "decoder_layernorm": bool(norm.decoder_layernorm),
        "authenticated": bool(norm.decoder_layernorm),
    }
    expected = {
        field: LEVEL2_BLOCK0_NORM_BOUNDARY_ROUTE[field]
        for field in actual
    }
    if actual != expected:
        raise ValueError(
            "level2 block0 norm route does not match the evidence contract: "
            f"expected={expected}, actual={actual}"
        )
    return dict(LEVEL2_BLOCK0_NORM_BOUNDARY_ROUTE)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    effective_report_path = args.output_json
    report: dict[str, Any] = {
        "schema": TRACE_RUN_SCHEMA,
        "status": "failed",
        "failure_phase": None,
        "last_trustworthy_phase": "request_received",
        "requested_route": {
            "route": ROUTE,
            "device_type": "metal",
            "decoder_linear_backend": "turing_fda",
            "sparse_conv_matmul_backend": "turing_fda",
            "decoder_layernorm_backend": "cuda-welford-turing-t4",
            "decoder_silu_backend": "cuda-turing-t4-fp16-lut",
            "boundary_routes": {
                "level2_block0_norm": dict(
                    LEVEL2_BLOCK0_NORM_BOUNDARY_ROUTE
                ),
            },
            "parent_state": "externally-captured-level0-trace",
        },
        "effective_route": None,
        "manual_natural_equality": None,
        "input_tensor_sha256": None,
        "requested_report_path": str(args.output_json),
        "effective_report_path": str(effective_report_path),
        "stale_primary_invalidated": False,
        "primary": {
            "path": str(args.output_npz),
            "status": "not_written",
            "sha256": None,
        },
    }
    phase = "request_validation"
    previous_linear = os.environ.get("TRELLIS2MLX_DECODER_LINEAR_BACKEND")
    previous_sparse = os.environ.get("TRELLIS2MLX_SPARSE_CONV_MATMUL_BACKEND")
    try:
        protected_inputs = {
            args.level0_trace.resolve(),
            args.shape_decoder_checkpoint.resolve(),
            args.decoder_silu_lut.resolve(),
            args.turing_rsqrt_lut.resolve(),
        }
        protected_reports = protected_inputs | {args.output_npz.resolve()}
        if args.output_json.resolve() in protected_reports:
            effective_report_path = _failure_sibling(
                args.output_json,
                protected_reports,
            )
            report["effective_report_path"] = str(effective_report_path)
        if args.output_npz.resolve() in protected_inputs:
            raise ValueError("--output-npz collides with an input path")
        if args.output_npz.exists():
            args.output_npz.unlink()
            report["stale_primary_invalidated"] = True
        if args.output_json.resolve() in protected_reports:
            raise ValueError("--output-json collides with an input or primary path")
        for value, label in (
            (
                args.expected_level0_trace_sha256,
                "--expected-level0-trace-sha256",
            ),
            (args.expected_checkpoint_sha256, "--expected-checkpoint-sha256"),
            (
                args.expected_decoder_silu_lut_sha256,
                "--expected-decoder-silu-lut-sha256",
            ),
            (
                args.expected_turing_rsqrt_lut_sha256,
                "--expected-turing-rsqrt-lut-sha256",
            ),
        ):
            _validate_digest(value, label)
        report["last_trustworthy_phase"] = phase

        phase = "parent_trace_validation"
        level0_output, parent_coords = _load_parent_state(args.level0_trace)
        parent_sha = _sha256_file(args.level0_trace)
        if parent_sha != args.expected_level0_trace_sha256:
            raise ValueError(
                "level-zero trace digest mismatch: "
                f"expected={args.expected_level0_trace_sha256}, "
                f"actual={parent_sha}"
            )
        input_identity = decoder_level1_trace_input_sha256(
            level0_output,
            parent_coords,
        )
        report["input_tensor_sha256"] = input_identity
        report["parent_trace"] = {
            "path": str(args.level0_trace.resolve()),
            "sha256": parent_sha,
            "level0_output_shape": list(level0_output.shape),
            "parent_coords_shape": list(parent_coords.shape),
            "input_tensor_sha256": input_identity,
        }
        report["last_trustworthy_phase"] = phase

        phase = "layernorm_lut_validation"
        turing_rsqrt_lut, turing_rsqrt_identity = _load_turing_rsqrt_lut(
            args.turing_rsqrt_lut,
            args.expected_turing_rsqrt_lut_sha256,
        )
        report["turing_rsqrt_lut"] = turing_rsqrt_identity
        report["last_trustworthy_phase"] = phase

        phase = "checkpoint_validation"
        if not args.shape_decoder_checkpoint.is_file():
            raise FileNotFoundError(
                f"shape decoder checkpoint does not exist: "
                f"{args.shape_decoder_checkpoint}"
            )
        checkpoint_sha = _sha256_file(args.shape_decoder_checkpoint)
        if checkpoint_sha != args.expected_checkpoint_sha256:
            raise ValueError(
                "shape decoder checkpoint digest mismatch: "
                f"expected={args.expected_checkpoint_sha256}, "
                f"actual={checkpoint_sha}"
            )
        report["checkpoint"] = {
            "path": str(args.shape_decoder_checkpoint.resolve()),
            "sha256": checkpoint_sha,
            "size_bytes": args.shape_decoder_checkpoint.stat().st_size,
        }
        report["last_trustworthy_phase"] = phase

        phase = "repo_validation"
        report["repo"] = _validate_repo_state(args.expected_repo_commit)
        report["last_trustworthy_phase"] = phase

        phase = "runtime_validation"
        import mlx.core as mx

        from trellmlx.decoder_turing_layernorm import (
            CUDA_WELFORD_TURING_T4_BACKEND as LAYERNORM_BACKEND,
            configure_decoder_layernorm_backend,
            decoder_layernorm_backend_identity,
        )
        from trellmlx.decoder_turing_silu import (
            CUDA_TURING_T4_LUT_BACKEND,
            configure_decoder_silu_backend,
            decoder_silu_backend_identity,
        )

        mx.set_default_device(mx.gpu)
        effective_device = str(mx.default_device())
        if "gpu" not in effective_device.lower():
            raise RuntimeError(
                f"MLX decoder trace requires Metal GPU, got {effective_device}"
            )
        os.environ["TRELLIS2MLX_DECODER_LINEAR_BACKEND"] = "turing_fda"
        os.environ["TRELLIS2MLX_SPARSE_CONV_MATMUL_BACKEND"] = "turing_fda"
        configure_decoder_layernorm_backend(
            LAYERNORM_BACKEND,
            turing_rsqrt_delta_lut=mx.array(turing_rsqrt_lut),
            turing_rsqrt_lut_artifact_sha256_attested=(
                args.expected_turing_rsqrt_lut_sha256
            ),
        )
        configure_decoder_silu_backend(
            CUDA_TURING_T4_LUT_BACKEND,
            output_lut_artifact_path=args.decoder_silu_lut,
            output_lut_artifact_sha256_attested=(
                args.expected_decoder_silu_lut_sha256
            ),
        )
        effective_route = {
            "route": ROUTE,
            "device_type": "metal",
            "device": effective_device,
            "decoder_linear_backend": os.environ[
                "TRELLIS2MLX_DECODER_LINEAR_BACKEND"
            ],
            "sparse_conv_matmul_backend": os.environ[
                "TRELLIS2MLX_SPARSE_CONV_MATMUL_BACKEND"
            ],
            "decoder_layernorm": decoder_layernorm_backend_identity(),
            "decoder_layernorm_lut": turing_rsqrt_identity,
            "decoder_silu": decoder_silu_backend_identity(),
            "parent_state": {
                "path": str(args.level0_trace.resolve()),
                "sha256": parent_sha,
                "input_tensor_sha256": input_identity,
            },
        }
        report["effective_route"] = effective_route
        report["last_trustworthy_phase"] = phase

        phase = "model_load"
        from trellmlx.models.shape_slat_decoder import SLatDecoder
        from trellmlx.weight_loader import load_weights

        decoder = SLatDecoder(out_channels=7, pred_subdiv=True, use_fp16=True)
        unloaded = load_weights(
            decoder,
            str(args.shape_decoder_checkpoint),
            verbose=False,
        )
        if unloaded:
            raise ValueError(
                f"shape decoder checkpoint has {len(unloaded)} unloaded keys"
            )
        effective_route["boundary_routes"] = {
            "level2_block0_norm": _level2_block0_norm_route_identity(decoder),
        }
        report["last_trustworthy_phase"] = phase

        phase = "trace_capture"
        from trellmlx.decoder_level2_block0_trace import (
            capture_mlx_decoder_level2_block0_trace,
        )

        arrays, manual_equality = capture_mlx_decoder_level2_block0_trace(
            decoder,
            mx.array(level0_output),
            mx.array(parent_coords),
        )
        validation = write_decoder_level2_block0_trace_npz(
            args.output_npz,
            arrays,
        )
        report["manual_natural_equality"] = manual_equality
        report["primary"] = {
            "path": str(args.output_npz.resolve()),
            "status": "written",
            "sha256": _sha256_file(args.output_npz),
            "size_bytes": args.output_npz.stat().st_size,
            "validation": validation,
        }
        report.update(
            {
                "status": "done",
                "failure_phase": None,
                "last_trustworthy_phase": "trace_primary_reopened_exact",
            }
        )
        _write_report(effective_report_path, report)
        return 0
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "failure_phase": phase,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        _write_report(effective_report_path, report)
        return 1
    finally:
        if previous_linear is None:
            os.environ.pop("TRELLIS2MLX_DECODER_LINEAR_BACKEND", None)
        else:
            os.environ["TRELLIS2MLX_DECODER_LINEAR_BACKEND"] = previous_linear
        if previous_sparse is None:
            os.environ.pop("TRELLIS2MLX_SPARSE_CONV_MATMUL_BACKEND", None)
        else:
            os.environ["TRELLIS2MLX_SPARSE_CONV_MATMUL_BACKEND"] = previous_sparse


if __name__ == "__main__":
    raise SystemExit(main())
