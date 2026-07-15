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


SCHEMA = "trellis2mlx.mlx_stage_capture_route.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
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
    "shape_flow_block_injection_trace",
    "shape_flow_block_injection_manifest",
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
    parser.add_argument("--shape-flow-noise-sample")
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


def build_route_identity(args: argparse.Namespace, command: list[str]) -> dict[str, Any]:
    image_path = str(Path(args.image))
    output_dir = str(Path(args.output_dir))
    target_faces = _resolve_target_faces(args)
    shape_flow_trace_requested_keys = _parse_sparse_flow_trace_keys(args.shape_flow_trace_keys)
    return {
        "schema": SCHEMA,
        "route": {
            "family": "trellis2mlx/mlx",
            "backend": "mlx-metal",
            "attention_backend": os.environ.get("TRELLIS2MLX_ATTENTION_BACKEND", "fast"),
            "repo_root": str(REPO_ROOT),
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

    command = _build_generate_command(args, checkpoint_dir)
    requested_inputs, invalid_inputs = _preflight_input_paths(args)
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
    route_identity = build_route_identity(args, command)
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
    status = "done" if result.returncode == 0 and output_written else "failed"
    failure_phase = None
    route_binding_error = None
    primary_output_validation = None
    if result.returncode != 0:
        failure_phase = "generate_subprocess"
    elif not output_written:
        failure_phase = "missing_primary_output"
    elif args.stop_after_stage == "shape_flow_block_trace":
        try:
            _bind_effective_shape_flow_trace_keys(route_identity, checkpoint_npz)
            _write_json(output_dir / "route_identity.json", route_identity)
        except (OSError, ValueError) as exc:
            status = "failed"
            failure_phase = "bind_effective_route_identity"
            route_binding_error = str(exc)
    elif args.stop_after_stage == "shape_flow_steps":
        try:
            primary_output_validation = _validate_shape_flow_steps_checkpoint(
                checkpoint_npz,
                expected_steps=args.steps,
            )
            route_identity["route"]["shape_flow_steps_output"] = primary_output_validation
            _write_json(output_dir / "route_identity.json", route_identity)
        except (OSError, ValueError) as exc:
            status = "failed"
            failure_phase = "validate_primary_output"
            route_binding_error = str(exc)

    _write_json(
        output_dir / "run_report.json",
        {
            "schema": "trellis2mlx.mlx_stage_capture_run_report.v1",
            "status": status,
            "route_identity": route_identity,
            "last_trustworthy_phase": (
                f"{args.stop_after_stage}_validated"
                if status == "done" and args.stop_after_stage == "shape_flow_steps"
                else f"{args.stop_after_stage}_saved"
                if status == "done" and output_written
                else "route_identity_written"
            ),
            "primary_output_status": (
                "invalid"
                if failure_phase == "validate_primary_output"
                else "written"
                if output_written
                else "missing"
            ),
            "primary_output_validation": primary_output_validation,
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


def _validate_shape_flow_steps_checkpoint(
    checkpoint_path: Path,
    *,
    expected_steps: int,
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
        },
        "finite": True,
    }


def _build_generate_command(args: argparse.Namespace, checkpoint_dir: Path) -> list[str]:
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
    if args.shape_flow_noise_sample:
        command.extend([
            "--shape-flow-noise-sample",
            str(Path(args.shape_flow_noise_sample)),
        ])
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
