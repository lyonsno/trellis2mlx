"""Capture a parent-bound MLX level-two subdivision projection witness."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.decoder_level2_block0_trace_contract import (
    load_decoder_level2_block0_trace,
)
from scripts.decoder_level2_subdiv_trace_contract import (
    TRACE_SCHEMA,
    validate_parent_evidence,
    write_decoder_level2_subdiv_trace_npz,
)
from scripts.run_mlx_decoder_level1_trace import (
    _failure_sibling,
    _load_turing_rsqrt_lut,
    _sha256_file,
    _validate_digest,
    _validate_repo_state,
    _write_report,
)
from trellmlx.decoder_level2_subdiv_trace import (
    PROJECTION_BACKENDS,
    capture_mlx_decoder_level2_subdiv_trace,
)


ROUTE = "mlx-shape-decoder-level2-subdiv-trace"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-block0-trace", required=True, type=Path)
    parser.add_argument("--expected-parent-block0-trace-sha256", required=True)
    parser.add_argument("--block0-comparison", required=True, type=Path)
    parser.add_argument("--expected-block0-comparison-sha256", required=True)
    parser.add_argument("--ledger-comparison", required=True, type=Path)
    parser.add_argument("--expected-ledger-comparison-sha256", required=True)
    parser.add_argument("--shape-decoder-checkpoint", required=True, type=Path)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--decoder-silu-lut", required=True, type=Path)
    parser.add_argument("--expected-decoder-silu-lut-sha256", required=True)
    parser.add_argument("--turing-rsqrt-lut", required=True, type=Path)
    parser.add_argument("--expected-turing-rsqrt-lut-sha256", required=True)
    parser.add_argument("--expected-repo-commit", required=True)
    parser.add_argument(
        "--projection-backend",
        required=True,
        choices=PROJECTION_BACKENDS,
    )
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    effective_report_path = args.output_json
    report: dict[str, Any] = {
        "schema": TRACE_SCHEMA,
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
            "projection_backend": args.projection_backend,
            "parent_state": "exact-level2-block0-output",
        },
        "effective_route": None,
        "imports_completed": [],
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
        input_paths = {
            args.parent_block0_trace.resolve(),
            args.block0_comparison.resolve(),
            args.ledger_comparison.resolve(),
            args.shape_decoder_checkpoint.resolve(),
            args.decoder_silu_lut.resolve(),
            args.turing_rsqrt_lut.resolve(),
        }
        if args.output_npz.resolve() in input_paths:
            raise ValueError("--output-npz collides with an input path")
        protected_reports = input_paths | {args.output_npz.resolve()}
        if args.output_json.resolve() in protected_reports:
            effective_report_path = _failure_sibling(
                args.output_json,
                protected_reports,
            )
            report["effective_report_path"] = str(effective_report_path)
            raise ValueError("--output-json collides with an input or primary path")
        if args.output_npz.exists():
            args.output_npz.unlink()
            report["stale_primary_invalidated"] = True
        for value, label in (
            (
                args.expected_parent_block0_trace_sha256,
                "--expected-parent-block0-trace-sha256",
            ),
            (
                args.expected_block0_comparison_sha256,
                "--expected-block0-comparison-sha256",
            ),
            (
                args.expected_ledger_comparison_sha256,
                "--expected-ledger-comparison-sha256",
            ),
            (
                args.expected_checkpoint_sha256,
                "--expected-checkpoint-sha256",
            ),
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

        phase = "parent_authentication"
        parent_files = (
            (
                "parent_block0_trace",
                args.parent_block0_trace,
                args.expected_parent_block0_trace_sha256,
            ),
            (
                "block0_comparison",
                args.block0_comparison,
                args.expected_block0_comparison_sha256,
            ),
            (
                "ledger_comparison",
                args.ledger_comparison,
                args.expected_ledger_comparison_sha256,
            ),
        )
        parent_identity = {}
        for label, path, expected in parent_files:
            if not path.is_file():
                raise FileNotFoundError(f"{label} does not exist: {path}")
            actual = _sha256_file(path)
            if actual != expected:
                raise ValueError(
                    f"{label} digest mismatch: "
                    f"expected={expected}, actual={actual}"
                )
            parent_identity[label] = {
                "path": str(path.resolve()),
                "sha256": actual,
            }
        parent_trace = load_decoder_level2_block0_trace(
            args.parent_block0_trace
        )
        block0_comparison = json.loads(args.block0_comparison.read_text())
        ledger_comparison = json.loads(args.ledger_comparison.read_text())
        parent_evidence = validate_parent_evidence(
            {
                "level2_child_coords": parent_trace["level2_child_coords"],
                "level2_block0_output": parent_trace["level2_block0_output"],
            },
            block0_comparison,
            ledger_comparison,
        )
        report["parent_state"] = {
            **parent_identity,
            "evidence": parent_evidence,
        }
        report["last_trustworthy_phase"] = phase

        phase = "layernorm_lut_validation"
        turing_rsqrt_lut, turing_rsqrt_identity = _load_turing_rsqrt_lut(
            args.turing_rsqrt_lut,
            args.expected_turing_rsqrt_lut_sha256,
        )
        report["turing_rsqrt_lut"] = turing_rsqrt_identity
        report["last_trustworthy_phase"] = phase

        phase = "silu_lut_validation"
        if not args.decoder_silu_lut.is_file():
            raise FileNotFoundError(
                f"decoder SiLU LUT does not exist: {args.decoder_silu_lut}"
            )
        silu_sha = _sha256_file(args.decoder_silu_lut)
        if silu_sha != args.expected_decoder_silu_lut_sha256:
            raise ValueError(
                "decoder SiLU LUT digest mismatch: "
                f"expected={args.expected_decoder_silu_lut_sha256}, "
                f"actual={silu_sha}"
            )
        report["decoder_silu_lut"] = {
            "path": str(args.decoder_silu_lut.resolve()),
            "sha256": silu_sha,
        }
        report["last_trustworthy_phase"] = phase

        phase = "checkpoint_validation"
        if not args.shape_decoder_checkpoint.is_file():
            raise FileNotFoundError(
                "shape decoder checkpoint does not exist: "
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

        report["imports_completed"].append("mlx.core")
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
            "decoder_linear_backend": "turing_fda",
            "sparse_conv_matmul_backend": "turing_fda",
            "decoder_layernorm": decoder_layernorm_backend_identity(),
            "decoder_layernorm_lut": turing_rsqrt_identity,
            "decoder_silu": decoder_silu_backend_identity(),
            "projection_backend": args.projection_backend,
            "parent_state": parent_identity,
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
        report["last_trustworthy_phase"] = phase

        phase = "trace_capture"
        arrays = capture_mlx_decoder_level2_subdiv_trace(
            decoder,
            mx.array(parent_trace["level2_block0_output"]),
            mx.array(parent_trace["level2_child_coords"]),
            projection_backend=args.projection_backend,
        )
        validation = write_decoder_level2_subdiv_trace_npz(
            args.output_npz,
            arrays,
        )
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
