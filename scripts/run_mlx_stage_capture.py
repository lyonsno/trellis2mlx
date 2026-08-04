"""Run trellis2mlx stage-stop captures with durable route identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from trellmlx.source_cuda_gelu import (
    SOURCE_CUDA_BF16_GELU_TANH_BACKEND,
    SOURCE_CUDA_BF16_GELU_TANH_BITS_SHA256,
)


SCHEMA = "trellis2mlx.mlx_stage_capture_route.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
TURING_T4_BACKEND = "cuda-welford-turing-t4"
TURING_T4_ROPE_BACKEND = "cuda-polar-turing-t4"
DEFAULT_ROPE_BACKEND = "mlx-real"
SUPPORTED_ROPE_BACKENDS = (
    DEFAULT_ROPE_BACKEND,
    "source-complex",
    TURING_T4_ROPE_BACKEND,
)
DEFAULT_QK_NORM_BACKEND = "source-cuda-warp32"
SUPPORTED_QK_NORM_BACKENDS = (
    DEFAULT_QK_NORM_BACKEND,
    "mlx-sum",
)
SUPPORTED_ATTENTION_BACKENDS = (
    "fast",
    "mlx-fast",
    "manual",
    "mlx-manual",
    "source-cuda-self",
)
SUPPORTED_ATTENTION_SOFTMAX_BACKENDS = ("mlx-softmax", "source-cuda-turing")
SUPPORTED_ATTENTION_VALUE_BACKENDS = ("mlx-matmul", "source-cuda-sequential")
SHAPE_FLOW_ATTENTION_ROUTE_FIELDS = (
    "shape_flow_attention_backend_requested",
    "shape_flow_attention_backend_effective",
    "shape_flow_attention_softmax_backend_requested",
    "shape_flow_attention_softmax_backend_effective",
    "shape_flow_attention_value_backend_requested",
    "shape_flow_attention_value_backend_effective",
)
SHAPE_FLOW_GELU_ROUTE_FIELDS = (
    "shape_flow_gelu_backend_effective",
    "shape_flow_gelu_table_bits_sha256_effective",
)
STAGE_CAPTURE_SMOKE_PROFILE_TARGET_FACES = {
    "standard": 350_000,
    "source-quality": 500_000,
}
INPUT_PATH_FIELDS = (
    "image",
    "conditioning_sample",
    "shape_slat_sample",
    "shape_slat_support_sample",
    "shared_noise",
    "sparse_flow_trace_sample",
    "sparse_flow_start_sample",
    "sparse_flow_trace_block_input_sample",
    "sparse_flow_block_injection_trace",
    "sparse_flow_block_injection_manifest",
    "sparse_flow_layernorm_correction_report",
    "shape_flow_noise_sample",
    "turing_rsqrt_lut",
    "turing_rope_phase_lut",
    "shape_flow_block_injection_trace",
    "shape_flow_block_injection_manifest",
    "shape_timestep_modulation_lut",
    "shape_timestep_modulation_report",
)


def _parse_sparse_flow_trace_keys(value: str | None) -> list[str]:
    if not value:
        return []
    keys = [key.strip() for key in value.split(",")]
    if any(not key for key in keys):
        raise ValueError("--sparse-flow-trace-keys must be a comma-separated list of non-empty keys")
    return list(dict.fromkeys(keys))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run trellis2mlx stage capture")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--expected-repo-commit",
        help=(
            "Require the effective trellis2mlx checkout to be this commit and "
            "clean before generation."
        ),
    )
    parser.add_argument(
        "--stop-after-stage",
        required=True,
        choices=[
            "conditioning",
            "sparse_coords",
            "sparse_flow_step",
            "sparse_flow_steps",
            "sparse_flow_block_trace",
            "sparse_internals",
            "shape_flow_step",
            "shape_flow_steps",
            "shape_flow_block_trace",
            "shape_slat",
            "decoder_output",
            "mesh_raw",
            "mesh_clean",
            "mesh_uv",
        ],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--no-cascade", action="store_true")
    parser.add_argument("--target-faces", type=int)
    parser.add_argument(
        "--smoke-profile",
        choices=sorted(STAGE_CAPTURE_SMOKE_PROFILE_TARGET_FACES),
        default="standard",
        help=(
            "Named stage-capture budget profile; source-quality expands to the "
            "measured 500k detail route."
        ),
    )
    parser.add_argument("--texture-size", type=int, default=4096)
    parser.add_argument("--no-rembg", action="store_true")
    parser.add_argument("--conditioning-sample")
    parser.add_argument("--shape-slat-sample")
    parser.add_argument("--shape-slat-support-sample")
    parser.add_argument("--shared-noise")
    parser.add_argument("--shared-noise-sparse-only", action="store_true")
    parser.add_argument("--sparse-flow-trace-block-index", type=int, default=0)
    parser.add_argument("--sparse-flow-trace-step-index", type=int, default=0)
    parser.add_argument("--sparse-flow-trace-sample")
    parser.add_argument("--sparse-flow-start-sample")
    parser.add_argument("--sparse-flow-start-step-index", type=int, default=0)
    parser.add_argument("--sparse-flow-trace-block-input-sample")
    parser.add_argument("--sparse-flow-trace-no-kv-cache", action="store_true")
    parser.add_argument("--sparse-flow-trace-keys")
    parser.add_argument("--sparse-flow-block-injection-trace")
    parser.add_argument("--sparse-flow-block-injection-manifest")
    parser.add_argument("--sparse-flow-block-injection-step-index", type=int, default=2)
    parser.add_argument("--sparse-flow-block-injection-block-index", type=int, default=0)
    parser.add_argument(
        "--sparse-flow-block-injection-branch",
        choices=["pos", "neg", "both"],
        default="both",
    )
    parser.add_argument(
        "--sparse-flow-block-injection-stage",
        choices=["norm1", "modulated_self_input", "after_self", "after_cross", "after_mlp"],
        default="modulated_self_input",
    )
    parser.add_argument("--sparse-flow-block-injection-array-key")
    parser.add_argument("--sparse-flow-layernorm-correction-report")
    parser.add_argument("--sparse-flow-layernorm-correction-step-index", type=int, default=2)
    parser.add_argument("--sparse-flow-layernorm-correction-block-index", type=int, default=0)
    parser.add_argument(
        "--sparse-flow-layernorm-correction-branch",
        choices=["pos", "neg", "both"],
        default="pos",
    )
    parser.add_argument(
        "--sparse-flow-layernorm-correction-mode",
        choices=["scale", "bias"],
        default="scale",
    )
    parser.add_argument(
        "--sparse-flow-layernorm-correction-include",
        choices=["improved", "solved", "all"],
        default="improved",
    )
    parser.add_argument("--shape-flow-trace-block-index", type=int, default=0)
    parser.add_argument("--shape-flow-trace-step-index", type=int, default=0)
    parser.add_argument("--shape-flow-trace-keys")
    parser.add_argument(
        "--shape-flow-attention-backend",
        choices=SUPPORTED_ATTENTION_BACKENDS,
    )
    parser.add_argument(
        "--shape-flow-attention-softmax-backend",
        choices=SUPPORTED_ATTENTION_SOFTMAX_BACKENDS,
    )
    parser.add_argument(
        "--shape-flow-attention-value-backend",
        choices=SUPPORTED_ATTENTION_VALUE_BACKENDS,
    )
    parser.add_argument(
        "--shape-flow-layernorm-backend",
        choices=[
            "mlx-two-pass",
            "cuda-welford-metal",
            TURING_T4_BACKEND,
        ],
        default="mlx-two-pass",
    )
    parser.add_argument(
        "--qk-norm-backend",
        choices=SUPPORTED_QK_NORM_BACKENDS,
        default=DEFAULT_QK_NORM_BACKEND,
    )
    parser.add_argument(
        "--rope-backend",
        choices=SUPPORTED_ROPE_BACKENDS,
        default=DEFAULT_ROPE_BACKEND,
    )
    parser.add_argument("--turing-rope-phase-lut")
    parser.add_argument("--expected-turing-rope-phase-lut-sha256")
    parser.add_argument("--turing-rsqrt-lut")
    parser.add_argument("--expected-turing-rsqrt-lut-sha256")
    parser.add_argument("--shape-flow-noise-sample")
    parser.add_argument("--shape-timestep-modulation-lut")
    parser.add_argument("--shape-timestep-modulation-report")
    parser.add_argument("--expected-shape-timestep-modulation-lut-sha256")
    parser.add_argument("--expected-shape-timestep-modulation-report-sha256")
    parser.add_argument(
        "--expected-shape-timestep-modulation-source-checkpoint-sha256"
    )
    parser.add_argument("--shape-flow-block-injection-trace")
    parser.add_argument("--shape-flow-block-injection-manifest")
    parser.add_argument("--shape-flow-block-injection-step-index", type=int, default=0)
    parser.add_argument("--shape-flow-block-injection-block-index", type=int, default=1)
    parser.add_argument(
        "--shape-flow-block-injection-branch",
        choices=["pos", "neg", "both"],
        default="both",
    )
    parser.add_argument(
        "--shape-flow-block-injection-stage",
        choices=[
            "norm1",
            "modulated_self_input",
            "attention_raw",
            "after_self",
            "cross_attention_raw",
            "after_cross",
            "after_mlp",
        ],
        default="attention_raw",
    )
    parser.add_argument("--shape-flow-block-injection-array-key")
    parser.add_argument("--shape-flow-block-injection-scale", type=float, default=1.0)
    return parser


def _git_output(*args: str) -> str:
    process = subprocess.Popen(
        ["git", "-C", str(REPO_ROOT), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode:
        raise subprocess.CalledProcessError(
            process.returncode,
            process.args,
            output=stdout,
            stderr=stderr,
        )
    return stdout


def _read_repo_identity(expected_commit: str | None) -> dict[str, Any]:
    effective_commit = _git_output("rev-parse", "HEAD").strip()
    status_porcelain = _git_output(
        "status",
        "--porcelain",
        "--untracked-files=normal",
    )
    return {
        "commit_requested": expected_commit,
        "commit_effective": effective_commit,
        "dirty": bool(status_porcelain),
        "status_porcelain": status_porcelain,
    }


def build_route_identity(
    args: argparse.Namespace,
    command: list[str],
    *,
    repo_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    image_path = str(Path(args.image))
    output_dir = str(Path(args.output_dir))
    target_faces = _resolve_target_faces(args)
    shape_flow_trace_requested_keys = _parse_sparse_flow_trace_keys(args.shape_flow_trace_keys)
    turing_lut_identity = _validate_turing_rsqrt_route_args(args)
    turing_rope_lut_identity = _validate_turing_rope_route_args(args)
    timestep_modulation_identity = (
        _validate_shape_timestep_modulation_route_args(args)
    )
    if repo_identity is None:
        repo_identity = _read_repo_identity(args.expected_repo_commit)
    attention_backend_requested = os.environ.get(
        "TRELLIS2MLX_ATTENTION_BACKEND",
        "fast",
    ).lower()
    if attention_backend_requested in {"manual", "mlx-manual"}:
        attention_backend = "manual"
    elif attention_backend_requested in {"fast", "mlx-fast"}:
        attention_backend = "fast"
    else:
        attention_backend = f"unsupported:{attention_backend_requested}"
    attention_softmax_requested = os.environ.get(
        "TRELLIS2MLX_ATTENTION_SOFTMAX_BACKEND",
        "mlx-softmax",
    ).lower()
    attention_value_requested = os.environ.get(
        "TRELLIS2MLX_ATTENTION_VALUE_BACKEND",
        "mlx-matmul",
    ).lower()
    if attention_backend == "manual":
        attention_softmax_effective = attention_softmax_requested
        attention_value_effective = attention_value_requested
    else:
        attention_softmax_effective = "fused-fast-attention"
        attention_value_effective = "fused-fast-attention"
    shape_attention_backend_requested = (
        args.shape_flow_attention_backend or attention_backend_requested
    ).lower()
    shape_attention_backend = _normalize_attention_backend(
        shape_attention_backend_requested
    )
    shape_attention_softmax_requested = (
        args.shape_flow_attention_softmax_backend or attention_softmax_requested
    ).lower()
    shape_attention_value_requested = (
        args.shape_flow_attention_value_backend or attention_value_requested
    ).lower()
    if shape_attention_backend == "manual":
        shape_attention_softmax_effective = shape_attention_softmax_requested
        shape_attention_value_effective = shape_attention_value_requested
    elif shape_attention_backend_requested == "source-cuda-self":
        if (
            shape_attention_softmax_requested != "source-cuda-turing"
            or shape_attention_value_requested != "source-cuda-sequential"
        ):
            raise ValueError(
                "source-cuda-self requires source-cuda-turing softmax and "
                "source-cuda-sequential value projection"
            )
        shape_attention_softmax_effective = (
            "source-cuda-turing-widths-1029-7697-fast-otherwise"
        )
        shape_attention_value_effective = (
            "source-cuda-sequential-widths-1029-7697-fast-otherwise"
        )
    else:
        shape_attention_softmax_effective = "fused-fast-attention"
        shape_attention_value_effective = "fused-fast-attention"
    return {
        "schema": SCHEMA,
        "route": {
            "family": "trellis2mlx/mlx",
            "backend": "mlx-metal",
            "attention_backend_requested": attention_backend_requested,
            "attention_backend": attention_backend,
            "attention_softmax_backend_requested": attention_softmax_requested,
            "attention_softmax_backend_effective": attention_softmax_effective,
            "attention_value_backend_requested": attention_value_requested,
            "attention_value_backend_effective": attention_value_effective,
            "repo_root": str(REPO_ROOT),
            "repo_commit_requested": repo_identity["commit_requested"],
            "repo_commit_effective": repo_identity["commit_effective"],
            "repo_dirty": repo_identity["dirty"],
            "repo_status_porcelain": repo_identity["status_porcelain"],
            "seed": args.seed,
            "steps": args.steps,
            "resolution": args.resolution,
            "cascade": not args.no_cascade,
            "target_faces": target_faces,
            "smoke_profile": args.smoke_profile,
            "texture_size": args.texture_size,
            "preprocess_rembg": not args.no_rembg,
            "conditioning_sample_path": (
                str(Path(args.conditioning_sample)) if args.conditioning_sample else None
            ),
            "conditioning_sample_sha256": (
                _sha256_file(args.conditioning_sample) if args.conditioning_sample else None
            ),
            "shape_slat_sample_path": (
                str(Path(args.shape_slat_sample)) if args.shape_slat_sample else None
            ),
            "shape_slat_sample_sha256": (
                _sha256_file(args.shape_slat_sample) if args.shape_slat_sample else None
            ),
            "shape_slat_support_sample_path": (
                str(Path(args.shape_slat_support_sample))
                if args.shape_slat_support_sample else None
            ),
            "shape_slat_support_sample_sha256": (
                _sha256_file(args.shape_slat_support_sample)
                if args.shape_slat_support_sample else None
            ),
            "shared_noise_path": str(Path(args.shared_noise)) if args.shared_noise else None,
            "shared_noise_sha256": _sha256_file(args.shared_noise) if args.shared_noise else None,
            "shared_noise_sparse_only": args.shared_noise_sparse_only,
            "sparse_flow_trace_block_index": args.sparse_flow_trace_block_index,
            "sparse_flow_trace_step_index": args.sparse_flow_trace_step_index,
            "sparse_flow_trace_sample_path": (
                str(Path(args.sparse_flow_trace_sample)) if args.sparse_flow_trace_sample else None
            ),
            "sparse_flow_trace_sample_sha256": (
                _sha256_file(args.sparse_flow_trace_sample) if args.sparse_flow_trace_sample else None
            ),
            "sparse_flow_start_sample_path": (
                str(Path(args.sparse_flow_start_sample)) if args.sparse_flow_start_sample else None
            ),
            "sparse_flow_start_sample_sha256": (
                _sha256_file(args.sparse_flow_start_sample) if args.sparse_flow_start_sample else None
            ),
            "sparse_flow_start_step_index": args.sparse_flow_start_step_index,
            "sparse_flow_trace_block_input_sample_path": (
                str(Path(args.sparse_flow_trace_block_input_sample))
                if args.sparse_flow_trace_block_input_sample else None
            ),
            "sparse_flow_trace_block_input_sample_sha256": (
                _sha256_file(args.sparse_flow_trace_block_input_sample)
                if args.sparse_flow_trace_block_input_sample else None
            ),
            "sparse_flow_trace_uses_kv_cache": not args.sparse_flow_trace_no_kv_cache,
            "sparse_flow_trace_keys": _parse_sparse_flow_trace_keys(args.sparse_flow_trace_keys),
            "sparse_flow_block_injection_trace_path": (
                str(Path(args.sparse_flow_block_injection_trace))
                if args.sparse_flow_block_injection_trace else None
            ),
            "sparse_flow_block_injection_trace_sha256": (
                _sha256_file(args.sparse_flow_block_injection_trace)
                if args.sparse_flow_block_injection_trace else None
            ),
            "sparse_flow_block_injection_manifest_path": (
                str(Path(args.sparse_flow_block_injection_manifest))
                if args.sparse_flow_block_injection_manifest else None
            ),
            "sparse_flow_block_injection_manifest_sha256": (
                _sha256_file(args.sparse_flow_block_injection_manifest)
                if args.sparse_flow_block_injection_manifest else None
            ),
            "sparse_flow_block_injection_step_index": args.sparse_flow_block_injection_step_index,
            "sparse_flow_block_injection_block_index": args.sparse_flow_block_injection_block_index,
            "sparse_flow_block_injection_branch": args.sparse_flow_block_injection_branch,
            "sparse_flow_block_injection_stage": args.sparse_flow_block_injection_stage,
            "sparse_flow_block_injection_array_key": args.sparse_flow_block_injection_array_key,
            "sparse_flow_layernorm_correction_report_path": (
                str(Path(args.sparse_flow_layernorm_correction_report))
                if args.sparse_flow_layernorm_correction_report else None
            ),
            "sparse_flow_layernorm_correction_report_sha256": (
                _sha256_file(args.sparse_flow_layernorm_correction_report)
                if args.sparse_flow_layernorm_correction_report else None
            ),
            "sparse_flow_layernorm_correction_step_index": args.sparse_flow_layernorm_correction_step_index,
            "sparse_flow_layernorm_correction_block_index": args.sparse_flow_layernorm_correction_block_index,
            "sparse_flow_layernorm_correction_branch": args.sparse_flow_layernorm_correction_branch,
            "sparse_flow_layernorm_correction_mode": args.sparse_flow_layernorm_correction_mode,
            "sparse_flow_layernorm_correction_include": args.sparse_flow_layernorm_correction_include,
            "shape_flow_trace_block_index": args.shape_flow_trace_block_index,
            "shape_flow_trace_step_index": args.shape_flow_trace_step_index,
            "shape_flow_trace_key_selection": (
                "explicit" if shape_flow_trace_requested_keys else "full"
            ),
            "shape_flow_trace_requested_keys": shape_flow_trace_requested_keys,
            "shape_flow_trace_keys": (
                shape_flow_trace_requested_keys if shape_flow_trace_requested_keys else None
            ),
            "shape_flow_attention_backend_requested": (
                shape_attention_backend_requested
            ),
            "shape_flow_attention_backend_effective": shape_attention_backend,
            "shape_flow_attention_softmax_backend_requested": (
                shape_attention_softmax_requested
            ),
            "shape_flow_attention_softmax_backend_effective": (
                shape_attention_softmax_effective
            ),
            "shape_flow_attention_value_backend_requested": (
                shape_attention_value_requested
            ),
            "shape_flow_attention_value_backend_effective": (
                shape_attention_value_effective
            ),
            "shape_flow_gelu_backend_effective": (
                SOURCE_CUDA_BF16_GELU_TANH_BACKEND
            ),
            "shape_flow_gelu_table_bits_sha256_effective": (
                SOURCE_CUDA_BF16_GELU_TANH_BITS_SHA256
            ),
            "shape_flow_layernorm_backend_requested": args.shape_flow_layernorm_backend,
            "qk_norm_backend_requested": args.qk_norm_backend,
            "rope_backend_requested": args.rope_backend,
            "turing_rope_phase_lut_path": turing_rope_lut_identity["path"],
            "turing_rope_phase_lut_sha256_requested": (
                turing_rope_lut_identity["sha256_requested"]
            ),
            "turing_rope_phase_lut_sha256_effective": (
                turing_rope_lut_identity["sha256_effective"]
            ),
            "turing_rsqrt_lut_path": turing_lut_identity["path"],
            "turing_rsqrt_lut_sha256_requested": turing_lut_identity[
                "sha256_requested"
            ],
            "turing_rsqrt_lut_sha256_effective": turing_lut_identity[
                "sha256_effective"
            ],
            "turing_rsqrt_lut_content_sha256_effective": (
                turing_lut_identity["content_sha256_effective"]
            ),
            "shape_flow_block_injection_trace_path": (
                str(Path(args.shape_flow_block_injection_trace))
                if args.shape_flow_block_injection_trace else None
            ),
            "shape_flow_block_injection_trace_sha256": (
                _sha256_file(args.shape_flow_block_injection_trace)
                if args.shape_flow_block_injection_trace else None
            ),
            "shape_flow_block_injection_manifest_path": (
                str(Path(args.shape_flow_block_injection_manifest))
                if args.shape_flow_block_injection_manifest else None
            ),
            "shape_flow_block_injection_manifest_sha256": (
                _sha256_file(args.shape_flow_block_injection_manifest)
                if args.shape_flow_block_injection_manifest else None
            ),
            "shape_flow_block_injection_step_index": args.shape_flow_block_injection_step_index,
            "shape_flow_block_injection_block_index": args.shape_flow_block_injection_block_index,
            "shape_flow_block_injection_branch": args.shape_flow_block_injection_branch,
            "shape_flow_block_injection_stage": args.shape_flow_block_injection_stage,
            "shape_flow_block_injection_array_key": args.shape_flow_block_injection_array_key,
            "shape_flow_block_injection_scale": args.shape_flow_block_injection_scale,
            "shape_flow_noise_sample_path": (
                str(Path(args.shape_flow_noise_sample))
                if args.shape_flow_noise_sample else None
            ),
            "shape_flow_noise_sample_sha256": (
                _sha256_file(args.shape_flow_noise_sample)
                if args.shape_flow_noise_sample else None
            ),
            "shape_timestep_modulation_lut_path": (
                timestep_modulation_identity["npz_path"]
            ),
            "shape_timestep_modulation_lut_sha256_requested": (
                timestep_modulation_identity["npz_sha256_requested"]
            ),
            "shape_timestep_modulation_lut_sha256_effective": (
                timestep_modulation_identity["npz_sha256_effective"]
            ),
            "shape_timestep_modulation_report_path": (
                timestep_modulation_identity["report_path"]
            ),
            "shape_timestep_modulation_report_sha256_requested": (
                timestep_modulation_identity["report_sha256_requested"]
            ),
            "shape_timestep_modulation_report_sha256_effective": (
                timestep_modulation_identity["report_sha256_effective"]
            ),
            "shape_timestep_modulation_source_checkpoint_sha256": (
                timestep_modulation_identity["source_checkpoint_sha256"]
            ),
            "shape_timestep_modulation_identity": (
                timestep_modulation_identity["identity"]
            ),
        },
        "source": {
            "image_path": image_path,
            "image_sha256": _sha256_file(image_path),
        },
        "requested_stop": args.stop_after_stage,
        "requested_outputs": {
            stage: args.stop_after_stage == stage
            for stage in (
                "conditioning",
                "sparse_coords",
                "sparse_flow_step",
                "sparse_flow_steps",
                "sparse_flow_block_trace",
                "sparse_internals",
                "shape_flow_step",
                "shape_flow_steps",
                "shape_flow_block_trace",
                "shape_slat",
                "decoder_output",
                "mesh_raw",
                "mesh_clean",
                "mesh_uv",
            )
        },
        "output_dir": output_dir,
        "env": {
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "MLX_METAL_PATH": os.environ.get("MLX_METAL_PATH"),
            "TRELLIS2MLX_ATTENTION_BACKEND": os.environ.get("TRELLIS2MLX_ATTENTION_BACKEND"),
            "TRELLIS2MLX_ATTENTION_SOFTMAX_BACKEND": os.environ.get(
                "TRELLIS2MLX_ATTENTION_SOFTMAX_BACKEND"
            ),
            "TRELLIS2MLX_ATTENTION_VALUE_BACKEND": os.environ.get(
                "TRELLIS2MLX_ATTENTION_VALUE_BACKEND"
            ),
            "TRELLIS2MLX_QK_NORM_BACKEND": os.environ.get(
                "TRELLIS2MLX_QK_NORM_BACKEND"
            ),
        },
        "command": command,
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": _sha256_file(Path(__file__).resolve()),
        "forbidden_inferences": [
            "not Trellis-Mac route evidence",
            "not Microsoft CUDA TRELLIS.2 evidence",
            "not final-GLB parity evidence",
            "not texture/bake parity evidence",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        command = _build_generate_command(args, checkpoint_dir)
    except ValueError as exc:
        checkpoint_npz = checkpoint_dir / f"{args.stop_after_stage}.npz"
        checkpoint_json = checkpoint_dir / f"{args.stop_after_stage}.json"
        preexisting_primary = checkpoint_npz.exists() or checkpoint_json.exists()
        _write_json(
            output_dir / "run_report.json",
            {
                "schema": "trellis2mlx.mlx_stage_capture_run_report.v1",
                "status": "failed",
                "failure_phase": "preflight_shape_flow_attention_route",
                "last_trustworthy_phase": "requested_route_parsed",
                "primary_output_status": (
                    "preexisting_untrusted_preserved"
                    if preexisting_primary
                    else "not_started"
                ),
                "error": str(exc),
                "command": None,
                "exit_code": 2,
            },
        )
        return 2
    try:
        repo_identity = _read_repo_identity(args.expected_repo_commit)
    except (OSError, subprocess.CalledProcessError) as exc:
        _write_json(
            output_dir / "run_report.json",
            {
                "schema": "trellis2mlx.mlx_stage_capture_run_report.v1",
                "status": "failed",
                "failure_phase": "preflight_repo_identity",
                "last_trustworthy_phase": "requested_route_parsed",
                "primary_output_status": "not_started",
                "repo_identity": None,
                "error": str(exc),
                "command": command,
                "exit_code": 2,
            },
        )
        return 2
    if args.expected_repo_commit and (
        repo_identity["commit_effective"] != args.expected_repo_commit
        or repo_identity["dirty"]
    ):
        _write_json(
            output_dir / "run_report.json",
            {
                "schema": "trellis2mlx.mlx_stage_capture_run_report.v1",
                "status": "failed",
                "failure_phase": "preflight_repo_identity",
                "last_trustworthy_phase": "effective_repo_identity_read",
                "primary_output_status": "not_started",
                "repo_identity": repo_identity,
                "command": command,
                "exit_code": 2,
            },
        )
        return 2
    requested_inputs, invalid_inputs = _preflight_input_paths(args)
    turing_lut_identity = _describe_turing_rsqrt_route_args(args)
    turing_rope_lut_identity = _describe_turing_rope_route_args(args)
    try:
        _validate_turing_rsqrt_route_args(args)
    except (OSError, ValueError) as exc:
        _write_json(
            output_dir / "run_report.json",
            {
                "schema": "trellis2mlx.mlx_stage_capture_run_report.v1",
                "status": "failed",
                "failure_phase": "preflight_turing_rsqrt_route",
                "last_trustworthy_phase": "requested_route_parsed",
                "primary_output_status": "not_started",
                "requested_inputs": requested_inputs,
                "invalid_inputs": invalid_inputs,
                "turing_rsqrt_lut_identity": turing_lut_identity,
                "error": str(exc),
                "command": command,
                "exit_code": 2,
            },
        )
        return 2
    try:
        _validate_turing_rope_route_args(args)
    except (OSError, ValueError) as exc:
        _write_json(
            output_dir / "run_report.json",
            {
                "schema": "trellis2mlx.mlx_stage_capture_run_report.v1",
                "status": "failed",
                "failure_phase": "preflight_turing_rope_route",
                "last_trustworthy_phase": "requested_route_parsed",
                "primary_output_status": "not_started",
                "requested_inputs": requested_inputs,
                "invalid_inputs": invalid_inputs,
                "turing_rope_phase_lut_identity": turing_rope_lut_identity,
                "error": str(exc),
                "command": command,
                "exit_code": 2,
            },
        )
        return 2
    try:
        _validate_shape_timestep_modulation_route_args(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _write_json(
            output_dir / "run_report.json",
            {
                "schema": "trellis2mlx.mlx_stage_capture_run_report.v1",
                "status": "failed",
                "failure_phase": "preflight_shape_timestep_modulation_route",
                "last_trustworthy_phase": "requested_route_parsed",
                "primary_output_status": "not_started",
                "requested_inputs": requested_inputs,
                "invalid_inputs": invalid_inputs,
                "error": str(exc),
                "command": command,
                "exit_code": 2,
            },
        )
        return 2
    if invalid_inputs:
        _write_json(
            output_dir / "run_report.json",
            {
                "schema": "trellis2mlx.mlx_stage_capture_run_report.v1",
                "status": "failed",
                "failure_phase": "preflight_inputs",
                "last_trustworthy_phase": "requested_route_parsed",
                "primary_output_status": "not_started",
                "requested_inputs": requested_inputs,
                "invalid_inputs": invalid_inputs,
                "command": command,
                "exit_code": 2,
            },
        )
        return 2
    checkpoint_npz = checkpoint_dir / f"{args.stop_after_stage}.npz"
    checkpoint_json = checkpoint_dir / f"{args.stop_after_stage}.json"
    primary_paths = {checkpoint_npz.resolve(), checkpoint_json.resolve()}
    collisions = [
        {"field": field, "path": path}
        for field, path in requested_inputs.items()
        if path and Path(path).resolve() in primary_paths
    ]
    if collisions:
        _write_json(
            output_dir / "run_report.json",
            {
                "schema": "trellis2mlx.mlx_stage_capture_run_report.v1",
                "status": "failed",
                "failure_phase": "preflight_output_collision",
                "last_trustworthy_phase": "requested_inputs_validated",
                "primary_output_status": "not_started",
                "requested_inputs": requested_inputs,
                "collisions": collisions,
                "command": command,
                "exit_code": 2,
            },
        )
        return 2
    route_identity = build_route_identity(
        args,
        command,
        repo_identity=repo_identity,
    )
    _write_json(output_dir / "route_identity.json", route_identity)
    _write_json(
        output_dir / "run_report.json",
        {
            "schema": "trellis2mlx.mlx_stage_capture_run_report.v1",
            "status": "starting",
            "route_identity": route_identity,
            "last_trustworthy_phase": "route_identity_written",
            "primary_output_status": "not_started",
        },
    )

    for primary_path in (checkpoint_npz, checkpoint_json):
        primary_path.unlink(missing_ok=True)

    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": "."},
        text=True,
        capture_output=True,
    )
    elapsed = time.perf_counter() - started
    (output_dir / "stdout.log").write_text(result.stdout)
    (output_dir / "stderr.log").write_text(result.stderr)

    output_written = checkpoint_npz.exists() or checkpoint_json.exists()
    repo_identity_postflight = None
    repo_identity_postflight_error = None
    postflight_repo_invalid = False
    if result.returncode == 0:
        try:
            repo_identity_postflight = _read_repo_identity(args.expected_repo_commit)
        except (OSError, subprocess.CalledProcessError) as exc:
            repo_identity_postflight_error = str(exc)
            postflight_repo_invalid = True
        else:
            postflight_repo_invalid = bool(
                repo_identity_postflight["commit_effective"]
                != repo_identity["commit_effective"]
                or repo_identity_postflight["dirty"] != repo_identity["dirty"]
                or repo_identity_postflight["status_porcelain"]
                != repo_identity["status_porcelain"]
            )
        route_identity["route"]["repo_identity_postflight"] = repo_identity_postflight
        route_identity["route"]["repo_identity_postflight_error"] = (
            repo_identity_postflight_error
        )

    status = (
        "done"
        if result.returncode == 0 and output_written and not postflight_repo_invalid
        else "failed"
    )
    failure_phase = None
    route_binding_error = None
    primary_output_validation = None
    if result.returncode != 0:
        failure_phase = "generate_subprocess"
    elif not output_written:
        failure_phase = "missing_primary_output"
    elif postflight_repo_invalid:
        failure_phase = "postflight_repo_identity"
    elif args.stop_after_stage in {
        "shape_flow_step",
        "shape_flow_steps",
        "shape_flow_block_trace",
    }:
        try:
            if args.stop_after_stage == "shape_flow_steps":
                primary_output_validation = _validate_shape_flow_steps_checkpoint(
                    checkpoint_npz,
                    expected_steps=args.steps,
                    expected_route=route_identity["route"],
                )
                route_identity["route"]["shape_flow_steps_output"] = primary_output_validation
                effective_backend = primary_output_validation["sampler"][
                    "shape_flow_layernorm_backend"
                ]
                effective_qk_backend = primary_output_validation["sampler"][
                    "qk_norm_backend"
                ]
                effective_rope_backend = primary_output_validation["sampler"][
                    "rope_backend"
                ]
                if effective_rope_backend == TURING_T4_ROPE_BACKEND:
                    route_identity["route"][
                        "shape_flow_turing_rope_phase_lut_sha256_effective"
                    ] = primary_output_validation["sampler"][
                        "shape_flow_turing_rope_phase_lut_sha256"
                    ]
            else:
                effective_backend = _bind_effective_shape_flow_layernorm_backend(
                    route_identity,
                    checkpoint_npz,
                )
                effective_qk_backend = _bind_effective_qk_norm_backend(
                    route_identity,
                    checkpoint_npz,
                )
                effective_rope_backend = _bind_effective_rope_backend(
                    route_identity,
                    checkpoint_npz,
                )
            modulation_binding = (
                _bind_effective_shape_timestep_modulation_identity(
                    route_identity,
                    checkpoint_npz,
                )
            )
            if args.stop_after_stage == "shape_flow_steps":
                if (
                    primary_output_validation["sampler"][
                        "shape_timestep_modulation_route"
                    ]
                    != modulation_binding[
                        "shape_timestep_modulation_route"
                    ]
                ):
                    raise ValueError(
                        "shape_flow_steps modulation validation differs from "
                        "the shared effective binding"
                    )
            else:
                primary_output_validation = modulation_binding
            if args.stop_after_stage == "shape_flow_block_trace":
                _bind_effective_shape_flow_trace_keys(route_identity, checkpoint_npz)
            if args.stop_after_stage in {
                "shape_flow_step",
                "shape_flow_steps",
                "shape_flow_block_trace",
            }:
                _bind_effective_shape_flow_attention_route(
                    route_identity,
                    checkpoint_npz,
                )
                _bind_effective_shape_flow_gelu_route(
                    route_identity,
                    checkpoint_npz,
                )
            route_identity["route"]["shape_flow_layernorm_backend_effective"] = (
                effective_backend
            )
            route_identity["route"]["qk_norm_backend_effective"] = (
                effective_qk_backend
            )
            route_identity["route"]["rope_backend_effective"] = (
                effective_rope_backend
            )
            _write_json(output_dir / "route_identity.json", route_identity)
        except (OSError, ValueError) as exc:
            status = "failed"
            failure_phase = (
                "validate_primary_output"
                if args.stop_after_stage == "shape_flow_steps"
                else "bind_effective_route_identity"
            )
            route_binding_error = str(exc)

    _write_json(output_dir / "route_identity.json", route_identity)
    _write_json(
        output_dir / "run_report.json",
        {
            "schema": "trellis2mlx.mlx_stage_capture_run_report.v1",
            "status": status,
            "route_identity": route_identity,
            "last_trustworthy_phase": (
                f"{args.stop_after_stage}_validated"
                if status == "done"
                and args.stop_after_stage
                in {
                    "shape_flow_step",
                    "shape_flow_steps",
                    "shape_flow_block_trace",
                }
                else f"{args.stop_after_stage}_saved"
                if status == "done" and output_written
                else "route_identity_written"
            ),
            "primary_output_status": (
                "invalid"
                if failure_phase
                in {
                    "bind_effective_route_identity",
                    "validate_primary_output",
                    "postflight_repo_identity",
                }
                else "written"
                if output_written
                else "missing"
            ),
            "primary_output_validation": primary_output_validation,
            "repo_identity_preflight": repo_identity,
            "repo_identity_postflight": repo_identity_postflight,
            "repo_identity_postflight_error": repo_identity_postflight_error,
            "failure_phase": failure_phase,
            "error": route_binding_error,
            "exit_code": result.returncode,
            "elapsed_seconds": {"generate_subprocess": elapsed},
            "artifacts": _artifact_status(checkpoint_dir, args.stop_after_stage),
        },
    )
    if result.returncode != 0:
        return result.returncode
    return 0 if status == "done" else 2


def _preflight_input_paths(args: argparse.Namespace) -> tuple[dict[str, str | None], list[dict[str, str]]]:
    requested = {}
    invalid = []
    for field in INPUT_PATH_FIELDS:
        raw = getattr(args, field)
        requested[field] = str(Path(raw)) if raw else None
        if not raw:
            continue
        path = Path(raw)
        if not path.exists():
            invalid.append({"field": field, "path": str(path), "reason": "missing"})
            continue
        if not path.is_file():
            invalid.append({"field": field, "path": str(path), "reason": "not_file"})
            continue
        if path.stat().st_size == 0:
            invalid.append({"field": field, "path": str(path), "reason": "blank"})
            continue
        if field.endswith("_manifest"):
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                invalid.append({"field": field, "path": str(path), "reason": "invalid_json"})
                continue
            if not isinstance(manifest, dict):
                invalid.append({"field": field, "path": str(path), "reason": "invalid_json"})
    return requested, invalid


def _bind_effective_shape_flow_trace_keys(
    route_identity: dict[str, Any], checkpoint_path: Path
) -> None:
    route = route_identity.get("route")
    if not isinstance(route, dict):
        raise ValueError("route identity has no route object")
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        if "shape_flow_trace_selected_keys" not in checkpoint:
            raise ValueError("shape-flow trace omits effective selected-key metadata")
        selected = np.asarray(checkpoint["shape_flow_trace_selected_keys"])
        if selected.ndim != 1:
            raise ValueError("shape-flow effective selected keys must be a one-dimensional list")
        effective_keys = [str(value) for value in selected.tolist()]
        if not effective_keys or any(not key for key in effective_keys):
            raise ValueError("shape-flow effective selected keys must be non-empty")
        if len(set(effective_keys)) != len(effective_keys):
            raise ValueError("shape-flow effective selected keys contain duplicates")
        missing = [key for key in effective_keys if key not in checkpoint]
        if missing:
            raise ValueError(f"shape-flow trace omits effective selected arrays: {missing}")

    requested = route.get("shape_flow_trace_requested_keys")
    if not isinstance(requested, list):
        raise ValueError("route identity omits requested shape-flow trace keys")
    selection = route.get("shape_flow_trace_key_selection")
    if selection not in {"explicit", "full"}:
        raise ValueError(f"unsupported shape-flow trace key selection {selection!r}")
    if selection == "full" and requested:
        raise ValueError("full shape-flow trace selection cannot carry requested keys")
    if selection == "explicit" and not requested:
        raise ValueError("explicit shape-flow trace selection must carry requested keys")
    if selection == "explicit" and effective_keys != requested:
        raise ValueError("shape-flow effective selected keys differ from the explicit request")
    route["shape_flow_trace_keys"] = effective_keys


def _bind_effective_shape_flow_layernorm_backend(
    route_identity: dict[str, Any],
    checkpoint_path: Path,
) -> str:
    route = route_identity.get("route")
    if not isinstance(route, dict):
        raise ValueError("route identity has no route object")
    requested = route.get("shape_flow_layernorm_backend_requested")
    if requested not in {
        "mlx-two-pass",
        "cuda-welford-metal",
        TURING_T4_BACKEND,
    }:
        raise ValueError(
            f"route identity has unsupported requested shape-flow LayerNorm backend {requested!r}"
        )
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        if "shape_flow_layernorm_backend" not in checkpoint:
            raise ValueError(
                "shape-flow checkpoint omits effective LayerNorm backend metadata"
            )
        backend_array = np.asarray(checkpoint["shape_flow_layernorm_backend"])
        if backend_array.shape != () or backend_array.dtype.kind not in {"U", "S"}:
            raise ValueError(
                "shape-flow effective LayerNorm backend must be a string scalar"
            )
        effective = str(backend_array.item())
        (
            effective_turing_lut_sha256,
            effective_turing_lut_content_sha256,
        ) = _checkpoint_turing_lut_sha256(
            checkpoint,
            backend=effective,
            expected_route=route,
            context="shape-flow checkpoint",
        )
    if effective != requested:
        raise ValueError(
            f"shape-flow effective LayerNorm backend {effective!r} "
            f"does not match requested {requested!r}"
        )
    if requested == TURING_T4_BACKEND:
        route["shape_flow_turing_rsqrt_lut_sha256_effective"] = (
            effective_turing_lut_sha256
        )
        route["shape_flow_turing_rsqrt_lut_content_sha256_effective"] = (
            effective_turing_lut_content_sha256
        )
    return effective


def _bind_effective_shape_timestep_modulation_identity(
    route_identity: dict[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    route = route_identity.get("route")
    if not isinstance(route, dict):
        raise ValueError("route identity has no route object")
    checkpoint_path = Path(checkpoint_path)
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        identity = None
        identity_json = ""
        if "shape_timestep_modulation_lut_json" in checkpoint:
            value = np.asarray(
                checkpoint["shape_timestep_modulation_lut_json"]
            )
            if value.shape != () or value.dtype.kind not in {"U", "S"}:
                raise ValueError(
                    "shape-flow checkpoint timestep modulation identity "
                    "must be a string scalar"
                )
            identity_json = str(value.item())
            if identity_json:
                try:
                    identity = json.loads(identity_json)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "shape-flow checkpoint timestep modulation identity "
                        "is invalid JSON"
                    ) from exc
                if not isinstance(identity, dict):
                    raise ValueError(
                        "shape-flow checkpoint timestep modulation identity "
                        "must decode to an object"
                    )
    effective = _validate_shape_timestep_modulation_identity(
        identity,
        expected_route=route,
    )
    route["shape_timestep_modulation_identity_effective"] = effective
    return {
        "schema": "trellis2mlx.shape_timestep_modulation_binding.v1",
        "path": str(checkpoint_path),
        "sha256": _sha256_file(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
        "shape_timestep_modulation_lut_json": identity_json,
        "shape_timestep_modulation_route": effective,
    }


def _bind_effective_shape_flow_attention_route(
    route_identity: dict[str, Any],
    checkpoint_path: Path,
) -> None:
    route = route_identity.get("route")
    if not isinstance(route, dict):
        raise ValueError("route identity has no route object")
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        missing = [
            field for field in SHAPE_FLOW_ATTENTION_ROUTE_FIELDS
            if field not in checkpoint
        ]
        if missing:
            raise ValueError(
                "shape-flow checkpoint omits shape-flow attention route metadata: "
                f"{missing}"
            )
        effective = {}
        for field in SHAPE_FLOW_ATTENTION_ROUTE_FIELDS:
            value = np.asarray(checkpoint[field])
            if value.shape != () or value.dtype.kind not in {"U", "S"}:
                raise ValueError(
                    f"shape-flow attention route field {field!r} must be a string scalar"
                )
            effective[field] = str(value.item())

    labels = {
        "shape_flow_attention_backend_requested": (
            "requested shape-flow attention backend"
        ),
        "shape_flow_attention_backend_effective": (
            "effective shape-flow attention backend"
        ),
        "shape_flow_attention_softmax_backend_requested": (
            "requested shape-flow attention softmax backend"
        ),
        "shape_flow_attention_softmax_backend_effective": (
            "effective shape-flow attention softmax backend"
        ),
        "shape_flow_attention_value_backend_requested": (
            "requested shape-flow attention value backend"
        ),
        "shape_flow_attention_value_backend_effective": (
            "effective shape-flow attention value backend"
        ),
    }
    for field in SHAPE_FLOW_ATTENTION_ROUTE_FIELDS:
        expected = route.get(field)
        if effective[field] != expected:
            raise ValueError(
                f"{labels[field]} {effective[field]!r} "
                f"does not match requested route {expected!r}"
            )


def _bind_effective_shape_flow_gelu_route(
    route_identity: dict[str, Any],
    checkpoint_path: Path,
) -> None:
    route = route_identity.get("route")
    if not isinstance(route, dict):
        raise ValueError("route identity has no route object")
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        missing = [
            field for field in SHAPE_FLOW_GELU_ROUTE_FIELDS
            if field not in checkpoint
        ]
        if missing:
            raise ValueError(
                "shape-flow checkpoint omits shape-flow GELU route metadata: "
                f"{missing}"
            )
        effective = {}
        for field in SHAPE_FLOW_GELU_ROUTE_FIELDS:
            value = np.asarray(checkpoint[field])
            if value.shape != () or value.dtype.kind not in {"U", "S"}:
                raise ValueError(
                    f"shape-flow GELU route field {field!r} must be a string scalar"
                )
            effective[field] = str(value.item())

    labels = {
        "shape_flow_gelu_backend_effective": "effective shape-flow GELU backend",
        "shape_flow_gelu_table_bits_sha256_effective": (
            "effective shape-flow GELU table bits SHA256"
        ),
    }
    for field in SHAPE_FLOW_GELU_ROUTE_FIELDS:
        expected = route.get(field)
        if effective[field] != expected:
            raise ValueError(
                f"{labels[field]} {effective[field]!r} "
                f"does not match requested route {expected!r}"
            )


def _normalize_attention_backend(requested: str) -> str:
    if requested in {"manual", "mlx-manual"}:
        return "manual"
    if requested in {"fast", "mlx-fast"}:
        return "fast"
    if requested == "source-cuda-self":
        return "source-cuda-self-widths-1029-7697-fast-otherwise"
    return f"unsupported:{requested}"


def _bind_effective_qk_norm_backend(
    route_identity: dict[str, Any],
    checkpoint_path: Path,
) -> str:
    route = route_identity.get("route")
    if not isinstance(route, dict):
        raise ValueError("route identity has no route object")
    requested = route.get("qk_norm_backend_requested")
    if requested not in SUPPORTED_QK_NORM_BACKENDS:
        raise ValueError(
            f"route identity has unsupported requested Q/K norm backend "
            f"{requested!r}"
        )
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        if "qk_norm_backend" not in checkpoint:
            raise ValueError(
                "shape-flow checkpoint omits effective Q/K norm backend metadata"
            )
        backend_array = np.asarray(checkpoint["qk_norm_backend"])
        if backend_array.shape != () or backend_array.dtype.kind not in {"U", "S"}:
            raise ValueError(
                "shape-flow effective Q/K norm backend must be a string scalar"
            )
        effective = str(backend_array.item())
    if effective != requested:
        raise ValueError(
            f"shape-flow effective Q/K norm backend {effective!r} "
            f"does not match requested {requested!r}"
        )
    return effective


def _bind_effective_rope_backend(
    route_identity: dict[str, Any],
    checkpoint_path: Path,
) -> str:
    route = route_identity.get("route")
    if not isinstance(route, dict):
        raise ValueError("route identity has no route object")
    requested = route.get("rope_backend_requested")
    if requested not in SUPPORTED_ROPE_BACKENDS:
        raise ValueError(
            f"route identity has unsupported requested RoPE backend "
            f"{requested!r}"
        )
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        if "rope_backend" not in checkpoint:
            raise ValueError(
                "shape-flow checkpoint omits effective RoPE backend metadata"
            )
        backend_array = np.asarray(checkpoint["rope_backend"])
        if backend_array.shape != () or backend_array.dtype.kind not in {"U", "S"}:
            raise ValueError(
                "shape-flow effective RoPE backend must be a string scalar"
            )
        effective = str(backend_array.item())
        effective_lut_sha256 = _checkpoint_turing_rope_lut_sha256(
            checkpoint,
            backend=effective,
            expected_route=route,
            context="shape-flow checkpoint",
        )
    if effective != requested:
        raise ValueError(
            f"shape-flow effective RoPE backend {effective!r} "
            f"does not match requested {requested!r}"
        )
    if requested == TURING_T4_ROPE_BACKEND:
        route["shape_flow_turing_rope_phase_lut_sha256_effective"] = (
            effective_lut_sha256
        )
    return effective


def _checkpoint_turing_rope_lut_sha256(
    checkpoint: Any,
    *,
    backend: str,
    expected_route: dict[str, Any],
    context: str,
) -> str | None:
    key = "shape_flow_turing_rope_phase_lut_sha256"
    if backend == TURING_T4_ROPE_BACKEND:
        if key not in checkpoint:
            raise ValueError(
                f"{context} omits effective Turing RoPE phase LUT SHA256"
            )
        value = np.asarray(checkpoint[key])
        if value.shape != () or value.dtype.kind not in {"U", "S"}:
            raise ValueError(
                f"{context} effective Turing RoPE phase LUT SHA256 "
                "must be a string scalar"
            )
        effective = str(value.item())
        expected = expected_route.get(
            "turing_rope_phase_lut_sha256_effective"
        )
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or effective != expected
        ):
            raise ValueError(
                f"{context} effective Turing RoPE phase LUT SHA256 "
                f"{effective!r} does not match requested/effective route "
                f"{expected!r}"
            )
        return effective
    if key in checkpoint:
        value = np.asarray(checkpoint[key])
        if (
            value.shape != ()
            or value.dtype.kind not in {"U", "S"}
            or str(value.item())
        ):
            raise ValueError(
                f"{context} carries Turing RoPE phase LUT identity under "
                f"non-Turing backend {backend!r}"
            )
    return None


def _checkpoint_turing_lut_sha256(
    checkpoint: Any,
    *,
    backend: str,
    expected_route: dict[str, Any],
    context: str,
) -> tuple[str | None, str | None]:
    key = "shape_flow_turing_rsqrt_lut_sha256"
    content_key = "shape_flow_turing_rsqrt_lut_content_sha256"
    if backend == TURING_T4_BACKEND:
        if key not in checkpoint:
            raise ValueError(
                f"{context} omits effective Turing rsqrt LUT SHA256"
            )
        if content_key not in checkpoint:
            raise ValueError(
                f"{context} omits effective Turing rsqrt LUT content SHA256"
            )
        value = np.asarray(checkpoint[key])
        if value.shape != () or value.dtype.kind not in {"U", "S"}:
            raise ValueError(
                f"{context} effective Turing rsqrt LUT SHA256 "
                "must be a string scalar"
            )
        effective = str(value.item())
        expected = expected_route.get(
            "turing_rsqrt_lut_sha256_effective"
        )
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or effective != expected
        ):
            raise ValueError(
                f"{context} effective Turing rsqrt LUT SHA256 "
                f"{effective!r} does not match requested/effective route "
                f"{expected!r}"
            )
        content_value = np.asarray(checkpoint[content_key])
        if (
            content_value.shape != ()
            or content_value.dtype.kind not in {"U", "S"}
        ):
            raise ValueError(
                f"{context} effective Turing rsqrt LUT content SHA256 "
                "must be a string scalar"
            )
        content_effective = str(content_value.item())
        content_expected = expected_route.get(
            "turing_rsqrt_lut_content_sha256_effective"
        )
        if (
            not isinstance(content_expected, str)
            or len(content_expected) != 64
            or content_effective != content_expected
        ):
            raise ValueError(
                f"{context} effective Turing rsqrt LUT content SHA256 "
                f"{content_effective!r} does not match requested/effective route "
                f"{content_expected!r}"
            )
        return effective, content_effective
    for identity_key in (key, content_key):
        if identity_key in checkpoint:
            value = np.asarray(checkpoint[identity_key])
            if (
                value.shape != ()
                or value.dtype.kind not in {"U", "S"}
                or str(value.item())
            ):
                raise ValueError(
                    f"{context} carries Turing rsqrt LUT identity under "
                    f"non-Turing backend {backend!r}"
                )
    return None, None


def _validate_shape_flow_steps_checkpoint(
    checkpoint_path: Path,
    *,
    expected_steps: int,
    expected_route: dict[str, Any] | None = None,
) -> dict[str, Any]:
    required = {
        "noise",
        "sample_feats",
        "coords",
        "coords_3d",
        "sample_in",
        "pred_pos",
        "pred_neg",
        "pred_cfg",
        "x0_pos",
        "x0_cfg",
        "std_pos",
        "std_cfg",
        "ratio_raw",
        "std_ratio",
        "ratio_effective",
        "x0_rescaled",
        "x0_after_rescale",
        "pred_final",
        "pred_v_feats",
        "sample_next",
        "t",
        "t_prev",
        "steps",
        "guidance_strength",
        "guidance_rescale",
        "guidance_interval",
        "rescale_t",
        "sigma_min",
        "shape_flow_block_injection_json",
        "shape_flow_layernorm_backend",
        "qk_norm_backend",
        "rope_backend",
    }
    stepped_tensor_names = (
        "sample_in",
        "pred_pos",
        "pred_neg",
        "pred_cfg",
        "x0_pos",
        "x0_cfg",
        "x0_rescaled",
        "x0_after_rescale",
        "pred_final",
        "pred_v_feats",
        "sample_next",
    )
    stepped_scalar_names = (
        "std_pos",
        "std_cfg",
        "ratio_raw",
        "std_ratio",
        "ratio_effective",
    )
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        missing = sorted(required.difference(checkpoint.files))
        if missing:
            raise ValueError(f"shape_flow_steps missing required arrays: {missing}")

        steps_array = np.asarray(checkpoint["steps"])
        if steps_array.shape != () or steps_array.dtype != np.dtype(np.int32):
            raise ValueError("shape_flow_steps steps must be an int32 scalar")
        steps = int(steps_array.item())
        if steps != expected_steps:
            raise ValueError(
                f"shape_flow_steps records {steps} steps, expected {expected_steps}"
            )
        sample_in = np.asarray(checkpoint["sample_in"])
        if sample_in.ndim != 3 or sample_in.shape[0] != expected_steps:
            raise ValueError(
                "shape_flow_steps sample_in must have shape [steps,N,C], "
                f"got {sample_in.shape}"
            )
        step_shape = sample_in.shape
        for name in stepped_tensor_names:
            array = np.asarray(checkpoint[name])
            if array.dtype != np.dtype(np.float32):
                raise ValueError(
                    f"shape_flow_steps {name} dtype must be float32, got {array.dtype}"
                )
            if array.shape != step_shape:
                raise ValueError(
                    f"shape_flow_steps {name} shape {array.shape} does not match {step_shape}"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"shape_flow_steps {name} contains non-finite values")
        for name in stepped_scalar_names:
            array = np.asarray(checkpoint[name])
            if array.dtype != np.dtype(np.float32):
                raise ValueError(
                    f"shape_flow_steps {name} dtype must be float32, got {array.dtype}"
                )
            if array.shape[0:1] != (expected_steps,):
                raise ValueError(
                    f"shape_flow_steps {name} must have leading step axis {expected_steps}, "
                    f"got {array.shape}"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"shape_flow_steps {name} contains non-finite values")

        noise = np.asarray(checkpoint["noise"])
        sample_feats = np.asarray(checkpoint["sample_feats"])
        if noise.dtype != np.dtype(np.float32) or sample_feats.dtype != np.dtype(np.float32):
            raise ValueError(
                "shape_flow_steps noise/sample_feats dtype must both be float32, "
                f"got {noise.dtype} and {sample_feats.dtype}"
            )
        expected_sample_shape = step_shape[1:]
        if noise.shape != expected_sample_shape or sample_feats.shape != expected_sample_shape:
            raise ValueError(
                "shape_flow_steps noise/sample_feats must match [N,C] sample shape "
                f"{expected_sample_shape}, got {noise.shape} and {sample_feats.shape}"
            )
        if not np.array_equal(noise, sample_feats) or not np.array_equal(noise, sample_in[0]):
            raise ValueError("shape_flow_steps initial noise identity is inconsistent")
        if not np.array_equal(sample_in[1:], np.asarray(checkpoint["sample_next"][:-1])):
            raise ValueError("shape_flow_steps recurrence sample_in[s+1] != sample_next[s]")
        if not np.array_equal(
            np.asarray(checkpoint["pred_final"]),
            np.asarray(checkpoint["pred_v_feats"]),
        ):
            raise ValueError("shape_flow_steps pred_v_feats does not match pred_final")

        coords = np.asarray(checkpoint["coords"])
        coords_3d = np.asarray(checkpoint["coords_3d"])
        if coords.dtype != np.dtype(np.int32) or coords_3d.dtype != np.dtype(np.int32):
            raise ValueError(
                "shape_flow_steps coords/coords_3d dtype must both be int32, "
                f"got {coords.dtype} and {coords_3d.dtype}"
            )
        if coords.shape != (step_shape[1], 4) or coords_3d.shape != (step_shape[1], 3):
            raise ValueError(
                "shape_flow_steps coordinates do not match token count: "
                f"coords={coords.shape}, coords_3d={coords_3d.shape}, N={step_shape[1]}"
            )
        if not np.array_equal(coords[:, 1:], coords_3d):
            raise ValueError("shape_flow_steps coords and coords_3d disagree")

        t = np.asarray(checkpoint["t"], dtype=np.float64)
        t_prev = np.asarray(checkpoint["t_prev"], dtype=np.float64)
        if (
            np.asarray(checkpoint["t"]).dtype != np.dtype(np.float32)
            or np.asarray(checkpoint["t_prev"]).dtype != np.dtype(np.float32)
        ):
            raise ValueError("shape_flow_steps t/t_prev dtype must both be float32")
        if t.shape != (expected_steps,) or t_prev.shape != (expected_steps,):
            raise ValueError(
                f"shape_flow_steps t/t_prev must have shape ({expected_steps},)"
            )
        if not np.isfinite(t).all() or not np.isfinite(t_prev).all() or not np.all(t > t_prev):
            raise ValueError("shape_flow_steps timestep schedule is non-finite or non-descending")
        if not np.array_equal(t[1:].astype(np.float32), t_prev[:-1].astype(np.float32)):
            raise ValueError("shape_flow_steps timestep pairs are not contiguous")

        expected_sampler_scalars = {
            "guidance_strength": 7.5,
            "guidance_rescale": 0.5,
            "rescale_t": 3.0,
            "sigma_min": 1e-5,
        }
        sampler_scalars = {}
        for name, expected in expected_sampler_scalars.items():
            array = np.asarray(checkpoint[name])
            if array.shape != () or array.dtype != np.dtype(np.float32):
                raise ValueError(f"shape_flow_steps {name} must be a float32 scalar")
            value = float(array.item())
            if not np.isfinite(value):
                raise ValueError(f"shape_flow_steps {name} must be finite")
            if not np.isclose(value, expected, rtol=0.0, atol=1e-7):
                raise ValueError(
                    f"shape_flow_steps {name}={value} does not match route value {expected}"
                )
            sampler_scalars[name] = value

        guidance_interval = np.asarray(checkpoint["guidance_interval"])
        if (
            guidance_interval.shape != (2,)
            or guidance_interval.dtype != np.dtype(np.float32)
            or not np.isfinite(guidance_interval).all()
            or not np.allclose(
                guidance_interval,
                np.array([0.6, 1.0], dtype=np.float32),
                rtol=0.0,
                atol=1e-7,
            )
        ):
            raise ValueError(
                "shape_flow_steps guidance_interval must be finite route value [0.6, 1.0]"
            )

        injection_array = np.asarray(checkpoint["shape_flow_block_injection_json"])
        if injection_array.shape != () or injection_array.dtype.kind not in {"U", "S"}:
            raise ValueError(
                "shape_flow_steps shape_flow_block_injection_json must be a string scalar"
            )
        injection_json = str(injection_array.item())
        injection_identity = None
        if injection_json:
            try:
                injection_identity = json.loads(injection_json)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "shape_flow_steps shape_flow_block_injection_json is invalid JSON"
                ) from exc
            if not isinstance(injection_identity, dict):
                raise ValueError(
                    "shape_flow_steps shape_flow_block_injection_json must decode to an object"
                )
        injection_route = _validate_shape_flow_injection_identity(
            injection_identity,
            expected_route=expected_route or {},
        )
        modulation_json = ""
        if "shape_timestep_modulation_lut_json" in checkpoint:
            modulation_array = np.asarray(
                checkpoint["shape_timestep_modulation_lut_json"]
            )
            if (
                modulation_array.shape != ()
                or modulation_array.dtype.kind not in {"U", "S"}
            ):
                raise ValueError(
                    "shape_flow_steps shape_timestep_modulation_lut_json "
                    "must be a string scalar"
                )
            modulation_json = str(modulation_array.item())
        modulation_identity = None
        if modulation_json:
            try:
                modulation_identity = json.loads(modulation_json)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "shape_flow_steps shape_timestep_modulation_lut_json "
                    "is invalid JSON"
                ) from exc
            if not isinstance(modulation_identity, dict):
                raise ValueError(
                    "shape_flow_steps shape_timestep_modulation_lut_json "
                    "must decode to an object"
                )
        modulation_route = _validate_shape_timestep_modulation_identity(
            modulation_identity,
            expected_route=expected_route or {},
        )
        backend_array = np.asarray(checkpoint["shape_flow_layernorm_backend"])
        if backend_array.shape != () or backend_array.dtype.kind not in {"U", "S"}:
            raise ValueError(
                "shape_flow_steps shape_flow_layernorm_backend must be a string scalar"
            )
        effective_backend = str(backend_array.item())
        requested_backend = (expected_route or {}).get(
            "shape_flow_layernorm_backend_requested",
            "mlx-two-pass",
        )
        if effective_backend != requested_backend:
            raise ValueError(
                f"shape_flow_steps effective LayerNorm backend {effective_backend!r} "
                f"does not match requested {requested_backend!r}"
            )
        (
            effective_turing_lut_sha256,
            effective_turing_lut_content_sha256,
        ) = _checkpoint_turing_lut_sha256(
            checkpoint,
            backend=effective_backend,
            expected_route=expected_route or {},
            context="shape_flow_steps",
        )
        qk_backend_array = np.asarray(checkpoint["qk_norm_backend"])
        if (
            qk_backend_array.shape != ()
            or qk_backend_array.dtype.kind not in {"U", "S"}
        ):
            raise ValueError(
                "shape_flow_steps qk_norm_backend must be a string scalar"
            )
        effective_qk_backend = str(qk_backend_array.item())
        requested_qk_backend = (expected_route or {}).get(
            "qk_norm_backend_requested",
            DEFAULT_QK_NORM_BACKEND,
        )
        if effective_qk_backend != requested_qk_backend:
            raise ValueError(
                "shape_flow_steps effective Q/K norm backend "
                f"{effective_qk_backend!r} does not match requested "
                f"{requested_qk_backend!r}"
            )
        rope_backend_array = np.asarray(checkpoint["rope_backend"])
        if (
            rope_backend_array.shape != ()
            or rope_backend_array.dtype.kind not in {"U", "S"}
        ):
            raise ValueError(
                "shape_flow_steps rope_backend must be a string scalar"
            )
        effective_rope_backend = str(rope_backend_array.item())
        requested_rope_backend = (expected_route or {}).get(
            "rope_backend_requested",
            DEFAULT_ROPE_BACKEND,
        )
        if effective_rope_backend != requested_rope_backend:
            raise ValueError(
                "shape_flow_steps effective RoPE backend "
                f"{effective_rope_backend!r} does not match requested "
                f"{requested_rope_backend!r}"
            )
        effective_turing_rope_lut_sha256 = (
            _checkpoint_turing_rope_lut_sha256(
                checkpoint,
                backend=effective_rope_backend,
                expected_route=expected_route or {},
                context="shape_flow_steps",
            )
        )

        schedule = np.linspace(1, 0, expected_steps + 1, dtype=np.float64)
        rescale_t = sampler_scalars["rescale_t"]
        schedule = rescale_t * schedule / (1 + (rescale_t - 1) * schedule)
        expected_t = schedule[:-1].astype(np.float32)
        expected_t_prev = schedule[1:].astype(np.float32)
        if not np.array_equal(t.astype(np.float32), expected_t) or not np.array_equal(
            t_prev.astype(np.float32), expected_t_prev
        ):
            raise ValueError("shape_flow_steps t/t_prev do not match the rescaled route schedule")

        pred_final = np.asarray(checkpoint["pred_final"], dtype=np.float32)
        expected_next = sample_in.astype(np.float32) - (
            (expected_t - expected_t_prev)[:, None, None] * pred_final
        )
        sample_next = np.asarray(checkpoint["sample_next"], dtype=np.float32)
        euler_residual = np.abs(sample_next.astype(np.float64) - expected_next.astype(np.float64))
        if not np.allclose(sample_next, expected_next, rtol=2e-5, atol=2e-5):
            raise ValueError(
                "shape_flow_steps Euler transition is inconsistent; "
                f"max_abs_residual={float(np.max(euler_residual))}"
            )

    return {
        "schema": "trellis2mlx.shape_flow_steps_output.v1",
        "path": str(checkpoint_path),
        "sha256": _sha256_file(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
        "step_count": expected_steps,
        "token_count": int(step_shape[1]),
        "channel_count": int(step_shape[2]),
        "recurrence_exact": True,
        "euler_transition_max_abs_residual": float(np.max(euler_residual)),
        "sampler": {
            **sampler_scalars,
            "guidance_interval": [float(value) for value in guidance_interval],
            "shape_flow_block_injection_json": injection_json,
            "shape_flow_block_injection_route": injection_route,
            "shape_timestep_modulation_lut_json": modulation_json,
            "shape_timestep_modulation_route": modulation_route,
            "shape_flow_layernorm_backend": effective_backend,
            "qk_norm_backend": effective_qk_backend,
            "rope_backend": effective_rope_backend,
            "shape_flow_turing_rsqrt_lut_sha256": (
                effective_turing_lut_sha256
            ),
            "shape_flow_turing_rsqrt_lut_content_sha256": (
                effective_turing_lut_content_sha256
            ),
            "shape_flow_turing_rope_phase_lut_sha256": (
                effective_turing_rope_lut_sha256
            ),
        },
        "finite": True,
    }


def _validate_shape_timestep_modulation_identity(
    identity: dict[str, Any] | None,
    *,
    expected_route: dict[str, Any],
) -> dict[str, Any] | None:
    expected_identity = expected_route.get(
        "shape_timestep_modulation_identity"
    )
    if expected_identity is None:
        if identity is not None:
            raise ValueError(
                "shape-flow checkpoint carries timestep modulation identity but "
                "the requested route carries none"
            )
        return None
    if identity is None:
        raise ValueError(
            "shape-flow checkpoint omits requested timestep modulation identity"
        )
    if identity.get("route_identity_evidence") is not True:
        raise ValueError(
            "shape-flow checkpoint timestep modulation identity omits "
            "route_identity_evidence=true"
        )
    if identity != expected_identity:
        raise ValueError(
            "shape-flow checkpoint effective timestep modulation identity does "
            "not match the authenticated requested route"
        )
    return identity


def _validate_shape_flow_injection_identity(
    identity: dict[str, Any] | None,
    *,
    expected_route: dict[str, Any],
) -> dict[str, Any]:
    trace_path = expected_route.get("shape_flow_block_injection_trace_path")
    manifest_path = expected_route.get("shape_flow_block_injection_manifest_path")
    if trace_path and manifest_path:
        raise ValueError("shape_flow_steps route requests both trace and manifest injection")
    if not trace_path and not manifest_path:
        if identity is not None:
            raise ValueError(
                "shape_flow_steps checkpoint carries injection identity but route requested none"
            )
        return {"mode": "none", "route_identity_match": True}
    if identity is None:
        raise ValueError(
            "shape_flow_steps route requested injection but checkpoint identity is empty"
        )
    if identity.get("route_identity_evidence") is not True:
        raise ValueError(
            "shape_flow_steps injection identity omits route_identity_evidence=true"
        )

    if trace_path:
        expected = {
            "trace_sha256": expected_route.get("shape_flow_block_injection_trace_sha256"),
            "branch": expected_route.get("shape_flow_block_injection_branch"),
            "step_index": expected_route.get("shape_flow_block_injection_step_index"),
            "block_index": expected_route.get("shape_flow_block_injection_block_index"),
            "stage": expected_route.get("shape_flow_block_injection_stage"),
        }
        for name, value in expected.items():
            if identity.get(name) != value:
                raise ValueError(
                    f"shape_flow_steps injection {name} {identity.get(name)!r} "
                    f"does not match requested {value!r}"
                )
        if Path(str(identity.get("trace_path"))).resolve() != Path(trace_path).resolve():
            raise ValueError("shape_flow_steps injection trace_path does not match requested path")
        branch = str(expected["branch"])
        block_index = int(expected["block_index"])
        stage = str(expected["stage"])
        requested_array_key = expected_route.get("shape_flow_block_injection_array_key")
        branches = ("pos", "neg") if branch == "both" else (branch,)
        expected_array_key = requested_array_key or ",".join(
            f"{active_branch}_block{block_index}_{stage}" for active_branch in branches
        )
        if identity.get("array_key") != expected_array_key:
            raise ValueError(
                f"shape_flow_steps injection array_key {identity.get('array_key')!r} "
                f"does not match requested/effective {expected_array_key!r}"
            )
        expected_scale = float(expected_route.get("shape_flow_block_injection_scale"))
        try:
            identity_scale = float(identity.get("source_delta_scale"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "shape_flow_steps injection source_delta_scale is not numeric"
            ) from exc
        if not np.isfinite(identity_scale) or identity_scale != expected_scale:
            raise ValueError(
                f"shape_flow_steps injection source_delta_scale {identity_scale!r} "
                f"does not match requested {expected_scale!r}"
            )
        return {
            "mode": "trace",
            "route_identity_match": True,
            "trace_path": str(trace_path),
            "trace_sha256": expected["trace_sha256"],
            "array_key": expected_array_key,
            "branch": branch,
            "step_index": int(expected["step_index"]),
            "block_index": block_index,
            "stage": stage,
            "source_delta_scale": expected_scale,
        }

    if Path(str(identity.get("manifest_path"))).resolve() != Path(manifest_path).resolve():
        raise ValueError("shape_flow_steps injection manifest_path does not match requested path")
    expected_manifest_sha = expected_route.get("shape_flow_block_injection_manifest_sha256")
    if identity.get("manifest_sha256") != expected_manifest_sha:
        raise ValueError(
            f"shape_flow_steps injection manifest_sha256 {identity.get('manifest_sha256')!r} "
            f"does not match requested {expected_manifest_sha!r}"
        )
    if _sha256_file(manifest_path) != expected_manifest_sha:
        raise ValueError(
            "shape_flow_steps requested injection manifest changed after route identity was recorded"
        )
    from trellmlx.shape_block_injection import load_shape_block_injection_manifest

    expected_identity = load_shape_block_injection_manifest(manifest_path).report_identity()
    if identity != expected_identity:
        raise ValueError(
            "shape_flow_steps effective identity does not match requested manifest"
        )
    sites = expected_identity["sites"]
    return {
        "mode": "manifest",
        "route_identity_match": True,
        "manifest_path": str(manifest_path),
        "manifest_sha256": expected_manifest_sha,
        "site_count": len(sites),
    }


def _build_generate_command(args: argparse.Namespace, checkpoint_dir: Path) -> list[str]:
    if (
        args.shape_flow_attention_backend
        or args.shape_flow_attention_softmax_backend
        or args.shape_flow_attention_value_backend
    ) and args.stop_after_stage not in {
        "shape_flow_step",
        "shape_flow_steps",
        "shape_flow_block_trace",
    }:
        raise ValueError(
            "shape-flow attention selectors require "
            "--stop-after-stage shape_flow_step, shape_flow_steps, or "
            "shape_flow_block_trace"
        )
    command = [
        sys.executable,
        "-u",
        "generate.py",
        "--image",
        str(Path(args.image)),
        "--output",
        str(Path(args.output_dir) / "output.glb"),
        "--seed",
        str(args.seed),
        "--resolution",
        str(args.resolution),
        "--steps",
        str(args.steps),
        "--target-faces",
        str(_resolve_target_faces(args)),
        "--texture-size",
        str(args.texture_size),
        "--save-checkpoints",
        str(checkpoint_dir),
        "--stop-after-stage",
        args.stop_after_stage,
    ]
    if args.no_cascade:
        command.append("--no-cascade")
    if args.no_rembg:
        command.append("--no-rembg")
    if args.conditioning_sample:
        command.extend(["--conditioning-sample", str(Path(args.conditioning_sample))])
    if args.shape_slat_sample:
        command.extend(["--shape-slat-sample", str(Path(args.shape_slat_sample))])
    if args.shape_slat_support_sample:
        command.extend([
            "--shape-slat-support-sample",
            str(Path(args.shape_slat_support_sample)),
        ])
    if args.shared_noise:
        command.extend(["--shared-noise", str(Path(args.shared_noise))])
    if args.shared_noise_sparse_only:
        command.append("--shared-noise-sparse-only")
    if args.sparse_flow_start_sample:
        command.extend([
            "--sparse-flow-start-step-index",
            str(args.sparse_flow_start_step_index),
            "--sparse-flow-start-sample",
            str(Path(args.sparse_flow_start_sample)),
        ])
    if args.stop_after_stage in {"sparse_flow_block_trace", "sparse_flow_step"}:
        command.extend(["--sparse-flow-trace-step-index", str(args.sparse_flow_trace_step_index)])
        if args.sparse_flow_trace_sample:
            command.extend(["--sparse-flow-trace-sample", str(Path(args.sparse_flow_trace_sample))])
    if args.stop_after_stage == "sparse_flow_block_trace":
        command.extend(["--sparse-flow-trace-block-index", str(args.sparse_flow_trace_block_index)])
        if args.sparse_flow_trace_block_input_sample:
            command.extend([
                "--sparse-flow-trace-block-input-sample",
                str(Path(args.sparse_flow_trace_block_input_sample)),
            ])
        if args.sparse_flow_trace_no_kv_cache:
            command.append("--sparse-flow-trace-no-kv-cache")
        if args.sparse_flow_trace_keys:
            command.extend(["--sparse-flow-trace-keys", args.sparse_flow_trace_keys])
    if args.sparse_flow_block_injection_trace:
        command.extend([
            "--sparse-flow-block-injection-trace",
            str(Path(args.sparse_flow_block_injection_trace)),
            "--sparse-flow-block-injection-step-index",
            str(args.sparse_flow_block_injection_step_index),
            "--sparse-flow-block-injection-block-index",
            str(args.sparse_flow_block_injection_block_index),
            "--sparse-flow-block-injection-branch",
            args.sparse_flow_block_injection_branch,
            "--sparse-flow-block-injection-stage",
            args.sparse_flow_block_injection_stage,
        ])
        if args.sparse_flow_block_injection_array_key:
            command.extend([
                "--sparse-flow-block-injection-array-key",
                args.sparse_flow_block_injection_array_key,
            ])
    if args.sparse_flow_block_injection_manifest:
        command.extend([
            "--sparse-flow-block-injection-manifest",
            str(Path(args.sparse_flow_block_injection_manifest)),
        ])
    if args.sparse_flow_layernorm_correction_report:
        command.extend([
            "--sparse-flow-layernorm-correction-report",
            str(Path(args.sparse_flow_layernorm_correction_report)),
            "--sparse-flow-layernorm-correction-step-index",
            str(args.sparse_flow_layernorm_correction_step_index),
            "--sparse-flow-layernorm-correction-block-index",
            str(args.sparse_flow_layernorm_correction_block_index),
            "--sparse-flow-layernorm-correction-branch",
            args.sparse_flow_layernorm_correction_branch,
            "--sparse-flow-layernorm-correction-mode",
            args.sparse_flow_layernorm_correction_mode,
            "--sparse-flow-layernorm-correction-include",
            args.sparse_flow_layernorm_correction_include,
        ])
    if args.stop_after_stage == "shape_flow_block_trace":
        command.extend([
            "--shape-flow-trace-block-index",
            str(args.shape_flow_trace_block_index),
            "--shape-flow-trace-step-index",
            str(args.shape_flow_trace_step_index),
        ])
        if args.shape_flow_trace_keys:
            command.extend(["--shape-flow-trace-keys", args.shape_flow_trace_keys])
    if args.stop_after_stage in {
        "shape_flow_step",
        "shape_flow_steps",
        "shape_flow_block_trace",
    }:
        if args.shape_flow_attention_backend:
            command.extend(
                [
                    "--shape-flow-attention-backend",
                    args.shape_flow_attention_backend,
                ]
            )
        if args.shape_flow_attention_softmax_backend:
            command.extend(
                [
                    "--shape-flow-attention-softmax-backend",
                    args.shape_flow_attention_softmax_backend,
                ]
            )
        if args.shape_flow_attention_value_backend:
            command.extend(
                [
                    "--shape-flow-attention-value-backend",
                    args.shape_flow_attention_value_backend,
                ]
            )
    command.extend([
        "--shape-flow-layernorm-backend",
        args.shape_flow_layernorm_backend,
        "--qk-norm-backend",
        args.qk_norm_backend,
        "--rope-backend",
        args.rope_backend,
    ])
    if args.turing_rope_phase_lut:
        command.extend(
            [
                "--turing-rope-phase-lut",
                str(Path(args.turing_rope_phase_lut)),
            ]
        )
    if args.expected_turing_rope_phase_lut_sha256:
        command.extend(
            [
                "--expected-turing-rope-phase-lut-sha256",
                args.expected_turing_rope_phase_lut_sha256,
            ]
        )
    if args.turing_rsqrt_lut:
        command.extend(
            [
                "--turing-rsqrt-lut",
                str(Path(args.turing_rsqrt_lut)),
            ]
        )
    if args.expected_turing_rsqrt_lut_sha256:
        command.extend(
            [
                "--expected-turing-rsqrt-lut-sha256",
                args.expected_turing_rsqrt_lut_sha256,
            ]
        )
    if args.shape_flow_noise_sample:
        command.extend([
            "--shape-flow-noise-sample",
            str(Path(args.shape_flow_noise_sample)),
        ])
    if args.shape_timestep_modulation_lut:
        command.extend(
            [
                "--shape-timestep-modulation-lut",
                str(Path(args.shape_timestep_modulation_lut)),
                "--shape-timestep-modulation-report",
                str(Path(args.shape_timestep_modulation_report)),
                "--expected-shape-timestep-modulation-lut-sha256",
                args.expected_shape_timestep_modulation_lut_sha256,
                "--expected-shape-timestep-modulation-report-sha256",
                args.expected_shape_timestep_modulation_report_sha256,
                "--expected-shape-timestep-modulation-source-checkpoint-sha256",
                (
                    args.expected_shape_timestep_modulation_source_checkpoint_sha256
                ),
            ]
        )
    if args.shape_flow_block_injection_trace:
        command.extend([
            "--shape-flow-block-injection-trace",
            str(Path(args.shape_flow_block_injection_trace)),
            "--shape-flow-block-injection-step-index",
            str(args.shape_flow_block_injection_step_index),
            "--shape-flow-block-injection-block-index",
            str(args.shape_flow_block_injection_block_index),
            "--shape-flow-block-injection-branch",
            args.shape_flow_block_injection_branch,
            "--shape-flow-block-injection-stage",
            args.shape_flow_block_injection_stage,
            "--shape-flow-block-injection-scale",
            str(args.shape_flow_block_injection_scale),
        ])
        if args.shape_flow_block_injection_array_key:
            command.extend([
                "--shape-flow-block-injection-array-key",
                args.shape_flow_block_injection_array_key,
            ])
    if args.shape_flow_block_injection_manifest:
        command.extend([
            "--shape-flow-block-injection-manifest",
            str(Path(args.shape_flow_block_injection_manifest)),
        ])
    return command


def _validate_shape_timestep_modulation_route_args(
    args: argparse.Namespace,
) -> dict[str, object | None]:
    values = (
        args.shape_timestep_modulation_lut,
        args.shape_timestep_modulation_report,
        args.expected_shape_timestep_modulation_lut_sha256,
        args.expected_shape_timestep_modulation_report_sha256,
        args.expected_shape_timestep_modulation_source_checkpoint_sha256,
    )
    if not any(values):
        return {
            "npz_path": None,
            "npz_sha256_requested": None,
            "npz_sha256_effective": None,
            "report_path": None,
            "report_sha256_requested": None,
            "report_sha256_effective": None,
            "source_checkpoint_sha256": None,
            "identity": None,
        }
    if not all(values):
        raise ValueError(
            "source-CUDA shape timestep modulation replay requires the LUT, "
            "witness report, both expected artifact SHA256 values, and the "
            "expected source checkpoint SHA256"
        )
    if not args.no_cascade:
        raise ValueError(
            "source-CUDA shape timestep modulation replay requires --no-cascade"
        )
    if args.stop_after_stage not in {
        "shape_flow_step",
        "shape_flow_steps",
        "shape_flow_block_trace",
    }:
        raise ValueError(
            "source-CUDA shape timestep modulation replay is only valid for "
            "shape-flow diagnostic stops"
        )

    from trellmlx.timestep_modulation_lut import (
        load_source_cuda_timestep_modulation_lut,
    )

    lut = load_source_cuda_timestep_modulation_lut(
        npz_path=args.shape_timestep_modulation_lut,
        report_path=args.shape_timestep_modulation_report,
        expected_npz_sha256=(
            args.expected_shape_timestep_modulation_lut_sha256
        ),
        expected_report_sha256=(
            args.expected_shape_timestep_modulation_report_sha256
        ),
        expected_source_checkpoint_sha256=(
            args.expected_shape_timestep_modulation_source_checkpoint_sha256
        ),
    )
    identity = lut.report_identity()
    return {
        "npz_path": str(Path(args.shape_timestep_modulation_lut)),
        "npz_sha256_requested": (
            args.expected_shape_timestep_modulation_lut_sha256
        ),
        "npz_sha256_effective": identity["npz_sha256_effective"],
        "report_path": str(Path(args.shape_timestep_modulation_report)),
        "report_sha256_requested": (
            args.expected_shape_timestep_modulation_report_sha256
        ),
        "report_sha256_effective": identity["report_sha256_effective"],
        "source_checkpoint_sha256": (
            identity["source_checkpoint_sha256_effective"]
        ),
        "identity": identity,
    }


def _validate_turing_rsqrt_route_args(
    args: argparse.Namespace,
) -> dict[str, str | None]:
    path = getattr(args, "turing_rsqrt_lut", None)
    expected = getattr(
        args, "expected_turing_rsqrt_lut_sha256", None
    )
    if args.shape_flow_layernorm_backend == TURING_T4_BACKEND:
        if not path or not expected:
            raise ValueError(
                f"{TURING_T4_BACKEND} requires --turing-rsqrt-lut and "
                "--expected-turing-rsqrt-lut-sha256"
            )
        effective = _sha256_file(path)
        if effective != expected:
            raise ValueError(
                "Turing rsqrt LUT SHA256 mismatch: "
                f"expected {expected}, got {effective}"
            )
        content_sha256 = _validate_turing_rsqrt_lut_payload(Path(path))
        return {
            "path": str(Path(path)),
            "sha256_requested": expected,
            "sha256_effective": effective,
            "content_sha256_effective": content_sha256,
        }
    if path or expected:
        raise ValueError(
            "Turing rsqrt LUT arguments only apply to "
            f"{TURING_T4_BACKEND}"
        )
    return {
        "path": None,
        "sha256_requested": None,
        "sha256_effective": None,
        "content_sha256_effective": None,
    }


def _describe_turing_rsqrt_route_args(
    args: argparse.Namespace,
) -> dict[str, str | None]:
    raw_path = getattr(args, "turing_rsqrt_lut", None)
    path = Path(raw_path) if raw_path else None
    effective = None
    content_effective = None
    read_error = None
    if path is not None and path.is_file():
        try:
            effective = _sha256_file(path)
        except OSError as exc:
            read_error = f"{type(exc).__name__}: {exc}"
    identity = {
        "path": str(path) if path is not None else None,
        "sha256_requested": getattr(
            args, "expected_turing_rsqrt_lut_sha256", None
        ),
        "sha256_effective": effective,
        "content_sha256_effective": content_effective,
    }
    if read_error is not None:
        identity["read_error"] = read_error
    return identity


def _validate_turing_rsqrt_lut_payload(path: Path) -> str:
    with np.load(path, allow_pickle=False) as loaded:
        if "normalized_delta" not in loaded.files:
            raise ValueError("Turing rsqrt LUT NPZ omits normalized_delta")
        correction = np.asarray(loaded["normalized_delta"])
    if correction.dtype != np.int8 or correction.shape != (1 << 24,):
        raise ValueError(
            "Turing rsqrt LUT normalized_delta must be int8[16777216], "
            f"got {correction.dtype}{correction.shape}"
        )
    return hashlib.sha256(
        np.ascontiguousarray(correction).tobytes()
    ).hexdigest()


def _validate_turing_rope_route_args(
    args: argparse.Namespace,
) -> dict[str, str | None]:
    path = getattr(args, "turing_rope_phase_lut", None)
    expected = getattr(
        args, "expected_turing_rope_phase_lut_sha256", None
    )
    if args.rope_backend == TURING_T4_ROPE_BACKEND:
        if not path or not expected:
            raise ValueError(
                f"{TURING_T4_ROPE_BACKEND} requires "
                "--turing-rope-phase-lut and "
                "--expected-turing-rope-phase-lut-sha256"
            )
        effective = _sha256_file(path)
        if effective != expected:
            raise ValueError(
                "Turing RoPE phase LUT SHA256 mismatch: "
                f"expected {expected}, got {effective}"
            )
        _validate_turing_rope_phase_lut_payload(Path(path))
        return {
            "path": str(Path(path)),
            "sha256_requested": expected,
            "sha256_effective": effective,
        }
    if path or expected:
        raise ValueError(
            "Turing RoPE phase LUT arguments only apply to "
            f"{TURING_T4_ROPE_BACKEND}"
        )
    return {
        "path": None,
        "sha256_requested": None,
        "sha256_effective": None,
    }


def _describe_turing_rope_route_args(
    args: argparse.Namespace,
) -> dict[str, str | None]:
    raw_path = getattr(args, "turing_rope_phase_lut", None)
    path = Path(raw_path) if raw_path else None
    effective = None
    read_error = None
    if path is not None and path.is_file():
        try:
            effective = _sha256_file(path)
        except OSError as exc:
            read_error = f"{type(exc).__name__}: {exc}"
    identity = {
        "path": str(path) if path is not None else None,
        "sha256_requested": getattr(
            args, "expected_turing_rope_phase_lut_sha256", None
        ),
        "sha256_effective": effective,
    }
    if read_error is not None:
        identity["read_error"] = read_error
    return identity


def _validate_turing_rope_phase_lut_payload(path: Path) -> None:
    with np.load(path, allow_pickle=False) as loaded:
        if "phase_pairs" not in loaded.files:
            raise ValueError("Turing RoPE phase LUT NPZ omits phase_pairs")
        phase_pairs = np.asarray(loaded["phase_pairs"])
    if phase_pairs.dtype != np.float32 or phase_pairs.shape != (64, 21, 2):
        raise ValueError(
            "Turing RoPE phase LUT must be float32[64,21,2], "
            f"got {phase_pairs.dtype}{phase_pairs.shape}"
        )
    if not np.isfinite(phase_pairs).all():
        raise ValueError("Turing RoPE phase LUT contains non-finite values")


def _resolve_target_faces(args: argparse.Namespace) -> int:
    if args.target_faces is not None:
        return int(args.target_faces)
    return STAGE_CAPTURE_SMOKE_PROFILE_TARGET_FACES[args.smoke_profile]


def _artifact_status(checkpoint_dir: Path, stage: str) -> dict[str, dict[str, Any]]:
    artifacts = {}
    for suffix in (".npz", ".json"):
        path = checkpoint_dir / f"{stage}{suffix}"
        if path.exists():
            artifacts[f"{stage}{suffix}"] = {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    return artifacts


def _sha256_file(path: str | Path | None) -> str | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


if __name__ == "__main__":
    raise SystemExit(main())
