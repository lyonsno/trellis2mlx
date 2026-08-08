#!/usr/bin/env python3
"""Capture an exact eight-step official-source CUDA shape-flow recurrence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np

try:
    from scripts.source_cuda_shape_block_trace import (
        extract_source,
        resolve_model_ref,
    )
    from scripts.source_cuda_shape_flow_suffix_ladder import (
        STEPS,
        _invalidate_primary_output,
        _load_conditioning,
        _run_suffix,
        _sha256,
        _validate_expected_modulation_identity,
        _validate_file,
        load_mlx_trajectory,
    )
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        SOURCE_DIRECT_COMPARISON_CLASS,
        SOURCE_DIRECT_INPUT_DIGESTS,
        SOURCE_DIRECT_ROUTE,
        SOURCE_MODEL_REF,
        _paths_alias,
        _validate_sha256,
        build_source_recurrence_arrays,
        write_source_recurrence_artifact,
    )
except ImportError:
    from source_cuda_shape_block_trace import (  # type: ignore[no-redef]
        extract_source,
        resolve_model_ref,
    )
    from source_cuda_shape_flow_suffix_ladder import (  # type: ignore[no-redef]
        STEPS,
        _invalidate_primary_output,
        _load_conditioning,
        _run_suffix,
        _sha256,
        _validate_expected_modulation_identity,
        _validate_file,
        load_mlx_trajectory,
    )
    from source_cuda_shape_flow_transition0_recoverability import (  # type: ignore[no-redef]
        SOURCE_DIRECT_COMPARISON_CLASS,
        SOURCE_DIRECT_INPUT_DIGESTS,
        SOURCE_DIRECT_ROUTE,
        SOURCE_MODEL_REF,
        _paths_alias,
        _validate_sha256,
        build_source_recurrence_arrays,
        write_source_recurrence_artifact,
    )


SCHEMA = "trellis2mlx.source_cuda_shape_flow_steps.v1"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _failure_report_path(
    requested: Path, output: Path, inputs: dict[str, Path]
) -> tuple[Path, list[str]]:
    protected = {"primary output": output, **inputs}
    collisions = [
        label for label, path in protected.items() if _paths_alias(requested, path)
    ]
    if not collisions:
        return requested, []
    index = 0
    while True:
        suffix = "" if index == 0 else f".{index}"
        candidate = requested.with_name(
            requested.name + f".source-shape-flow-steps.failure.json{suffix}"
        )
        if not any(
            _paths_alias(candidate, path)
            for path in (requested, *protected.values())
        ):
            return candidate, collisions
        index += 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--mlx-shape-flow-steps", required=True, type=Path)
    parser.add_argument("--mlx-shape-flow-steps-sha256", required=True)
    parser.add_argument("--mlx-run-report", required=True, type=Path)
    parser.add_argument("--mlx-run-report-sha256", required=True)
    parser.add_argument(
        "--mlx-timestep-modulation-route",
        choices=("default", "source-cuda-lut"),
    )
    parser.add_argument("--expected-modulation-lut-sha256")
    parser.add_argument("--expected-modulation-report-sha256")
    parser.add_argument("--expected-modulation-source-checkpoint-sha256")
    parser.add_argument("--conditioning", required=True, type=Path)
    parser.add_argument("--conditioning-sha256", required=True)
    parser.add_argument("--source-tar", required=True, type=Path)
    parser.add_argument("--source-tar-sha256", required=True)
    parser.add_argument("--model-repo", default="microsoft/TRELLIS.2-4B")
    parser.add_argument("--pipeline-config", default="pipeline.json")
    parser.add_argument("--sparse-conv-backend", default="none")
    parser.add_argument("--sparse-attn-backend", default="sdpa")
    parser.add_argument("--no-download", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    phase = "arguments_parsed"
    last_trustworthy_phase: str | None = phase
    expected_modulation_identity = (
        {
            "npz_sha256_effective": args.expected_modulation_lut_sha256,
            "report_sha256_effective": args.expected_modulation_report_sha256,
            "source_checkpoint_sha256_effective": (
                args.expected_modulation_source_checkpoint_sha256
            ),
        }
        if args.mlx_timestep_modulation_route == "source-cuda-lut"
        else None
    )
    input_paths = {
        "MLX shape-flow steps": args.mlx_shape_flow_steps,
        "MLX run report": args.mlx_run_report,
        "conditioning": args.conditioning,
        "source tar": args.source_tar,
    }
    failure_report, collisions = _failure_report_path(
        args.output_json, args.output_npz, input_paths
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "requested_route": {
            "route": SOURCE_DIRECT_ROUTE,
            "steps": STEPS,
            "attention_backend": args.sparse_attn_backend,
            "conv_backend": args.sparse_conv_backend,
            "mlx_timestep_modulation_route": (
                args.mlx_timestep_modulation_route
            ),
            "expected_modulation_identity": expected_modulation_identity,
        },
        "effective_route": "not-established",
        "primary_output_status": "missing",
        "failure_phase": None,
        "last_trustworthy_phase": last_trustworthy_phase,
        "requested_output_json": str(args.output_json),
        "effective_failure_report": str(failure_report),
        "requested_output_npz": str(args.output_npz),
        "phase_timings": {},
    }
    try:
        phase = "request_validation"
        phase_started = time.perf_counter()
        if args.output_npz.exists():
            report["primary_output_status"] = "preexisting_untrusted_preserved"
        if collisions:
            raise ValueError(
                "output JSON collides with protected paths: " + ", ".join(collisions)
            )
        if args.mlx_timestep_modulation_route not in {
            "default",
            "source-cuda-lut",
        }:
            raise ValueError(
                "--mlx-timestep-modulation-route must explicitly select "
                "default or source-cuda-lut"
            )
        if (
            args.mlx_timestep_modulation_route == "default"
            and any(
                value is not None
                for value in (
                    args.expected_modulation_lut_sha256,
                    args.expected_modulation_report_sha256,
                    args.expected_modulation_source_checkpoint_sha256,
                )
            )
        ):
            raise ValueError(
                "expected modulation SHA256 values require source-cuda-lut mode"
            )
        _validate_expected_modulation_identity(expected_modulation_identity)
        expected_digests = {
            "MLX shape-flow steps": _validate_sha256(
                args.mlx_shape_flow_steps_sha256,
                label="MLX shape-flow steps SHA256",
            ),
            "MLX run report": _validate_sha256(
                args.mlx_run_report_sha256, label="MLX run report SHA256"
            ),
            "conditioning": _validate_sha256(
                args.conditioning_sha256, label="conditioning SHA256"
            ),
            "source tar": _validate_sha256(
                args.source_tar_sha256, label="source tar SHA256"
            ),
        }
        if set(expected_digests) != set(SOURCE_DIRECT_INPUT_DIGESTS):
            raise AssertionError("direct source input digest contract drifted")
        physical_inputs: dict[str, dict[str, Any]] = {}
        for label, path in input_paths.items():
            _validate_file(path, label=label)
            actual = _sha256(path)
            if actual != expected_digests[label]:
                raise ValueError(
                    f"{label} SHA256 mismatch: {actual} != {expected_digests[label]}"
                )
            physical_inputs[label] = {
                "path": str(path),
                "sha256": actual,
                "size_bytes": path.stat().st_size,
            }
        _invalidate_primary_output(
            args.output_npz,
            protected={"output report": args.output_json, **input_paths},
        )
        report["primary_output_status"] = "missing"
        report["physical_inputs"] = physical_inputs
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        last_trustworthy_phase = phase

        phase = "input_validation"
        phase_started = time.perf_counter()
        trajectory, mlx_identity = load_mlx_trajectory(
            args.mlx_shape_flow_steps,
            args.mlx_run_report,
            args.conditioning,
            expected_modulation_identity=expected_modulation_identity,
        )
        cond_np, neg_cond_np = _load_conditioning(args.conditioning)
        report["inputs"] = {
            "expected_digests": expected_digests,
            "mlx": mlx_identity,
            "coords_shape": [int(value) for value in trajectory["coords"].shape],
            "sample_shape": [int(value) for value in trajectory["noise"].shape],
        }
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        last_trustworthy_phase = phase
        report["last_trustworthy_phase"] = last_trustworthy_phase
        if args.no_download:
            raise RuntimeError("--no-download stops after validated local inputs by request")

        phase = "extract_source"
        phase_started = time.perf_counter()
        source_root = extract_source(args.source_tar, Path.cwd())
        sys.path.insert(0, str(source_root))
        report["source_root"] = str(source_root)
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        last_trustworthy_phase = phase

        phase = "import_runtime"
        phase_started = time.perf_counter()
        os.environ["SPARSE_CONV_BACKEND"] = args.sparse_conv_backend
        os.environ["SPARSE_ATTN_BACKEND"] = args.sparse_attn_backend
        os.environ["ATTN_BACKEND"] = args.sparse_attn_backend
        import torch
        from huggingface_hub import hf_hub_download
        from trellis2 import models as source_models
        from trellis2.modules.sparse import config as sparse_config
        from trellis2.pipelines import samplers

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.set_grad_enabled(False)
        device = torch.device("cuda")
        report.update(
            {
                "torch": torch.__version__,
                "cuda_device": torch.cuda.get_device_name(0),
                "cuda_device_count": torch.cuda.device_count(),
                "sparse_attention_backend": getattr(sparse_config, "ATTN", None),
                "sparse_conv_backend": getattr(sparse_config, "CONV", None),
            }
        )
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        last_trustworthy_phase = phase

        phase = "load_pipeline_config"
        phase_started = time.perf_counter()
        config_path = Path(hf_hub_download(args.model_repo, args.pipeline_config))
        pipeline_args = json.loads(config_path.read_text())["args"]
        sampler_params = {
            **pipeline_args["shape_slat_sampler"]["params"],
            "steps": STEPS,
        }
        sampler = getattr(samplers, pipeline_args["shape_slat_sampler"]["name"])(
            **pipeline_args["shape_slat_sampler"]["args"]
        )
        report["pipeline_config"] = {
            "path": str(config_path),
            "sampler_name": pipeline_args["shape_slat_sampler"]["name"],
            "sampler_args": pipeline_args["shape_slat_sampler"]["args"],
            "sampler_params": sampler_params,
        }
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        last_trustworthy_phase = phase

        phase = "load_model"
        phase_started = time.perf_counter()
        model_ref = resolve_model_ref(
            args.model_repo, pipeline_args["models"]["shape_slat_flow_model_512"]
        )
        if model_ref != SOURCE_MODEL_REF:
            raise ValueError(f"source model ref mismatch: {model_ref} != {SOURCE_MODEL_REF}")
        flow_model = source_models.from_pretrained(model_ref).to(device).eval()
        torch.cuda.synchronize()
        model_load_seconds = time.perf_counter() - phase_started
        if flow_model.training:
            raise RuntimeError("source shape-flow model remained in training mode after eval")
        report["model"] = {
            "model_ref": model_ref,
            "parameter_count": int(
                sum(parameter.numel() for parameter in flow_model.parameters())
            ),
            "load_seconds": model_load_seconds,
            "training": bool(flow_model.training),
        }
        report["phase_timings"][phase] = model_load_seconds
        last_trustworthy_phase = phase

        phase = "source_recurrence"
        phase_started = time.perf_counter()
        coords = torch.from_numpy(trajectory["coords"]).to(
            device=device, dtype=torch.int32
        )
        noise = torch.from_numpy(trajectory["noise"]).to(
            device=device, dtype=torch.float32
        )
        cond = torch.from_numpy(cond_np).to(device=device, dtype=torch.float32)
        neg_cond = torch.from_numpy(neg_cond_np).to(
            device=device, dtype=torch.float32
        )
        captured_steps: list[dict[str, np.ndarray]] = []
        _, step_timings = _run_suffix(
            torch=torch,
            flow_model=flow_model,
            sampler=sampler,
            coords=coords,
            start_feats=noise,
            cond=cond,
            neg_cond=neg_cond,
            params=sampler_params,
            switch_step=0,
            capture_steps=captured_steps,
        )
        if len(captured_steps) != STEPS:
            raise ValueError(
                f"source recurrence captured {len(captured_steps)} of {STEPS} steps"
            )
        first_step = captured_steps[0]
        arrays = build_source_recurrence_arrays(
            coords=np.asarray(trajectory["coords"], dtype=np.int32),
            noise=np.asarray(trajectory["noise"], dtype=np.float32),
            transition0_t=first_step["t"],
            transition0_t_prev=first_step["t_prev"],
            source_transition0_pred_pos=first_step["pred_pos"],
            source_transition0_pred_neg=first_step["pred_neg"],
            source_transition0_guidance=first_step,
            source_transition0_pred_final=first_step["pred_final"],
            source_transition0_sample_next=first_step["sample_next"],
            suffix_steps=captured_steps[1:],
        )
        effective_route = {
            "route": SOURCE_DIRECT_ROUTE,
            "device_type": next(flow_model.parameters()).device.type,
            "cuda_device": torch.cuda.get_device_name(0),
            "attention_backend": getattr(sparse_config, "ATTN", None),
            "conv_backend": getattr(sparse_config, "CONV", None),
            "steps": STEPS,
            "rescale_t": float(sampler_params["rescale_t"]),
            "one_model_load": True,
            "model_ref": model_ref,
            "candidate_names": ["source-native-control"],
            "comparison_class": SOURCE_DIRECT_COMPARISON_CLASS,
            "mlx_timestep_modulation_route": (
                args.mlx_timestep_modulation_route
            ),
            "mlx_timestep_modulation_identity": (
                mlx_identity["shape_timestep_modulation_identity"]
            ),
        }
        report.update(
            {
                "status": "done",
                "effective_route": effective_route,
                "candidates": [
                    {
                        "name": "source-native-control",
                        "source_step_indices": list(range(STEPS)),
                        "source_step_count": STEPS,
                    }
                ],
                "timing": {
                    "model_load_seconds": model_load_seconds,
                    "step_elapsed_seconds": step_timings,
                    "source_steps_completed": len(captured_steps),
                    "source_steps_requested": STEPS,
                    "t4_compute_seconds": time.perf_counter() - started,
                },
                "forbidden_inferences": [
                    "not final mesh, texture, winding, or GLB evidence",
                    "not an MLX implementation route",
                    "not evidence outside the exact bound inputs and source model",
                ],
            }
        )
        receipt = write_source_recurrence_artifact(
            args.output_npz, arrays=arrays, report=report
        )
        report["primary_output"] = receipt
        report["primary_output_status"] = "written"
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        last_trustworthy_phase = phase
        report["last_trustworthy_phase"] = last_trustworthy_phase
        report["elapsed_seconds"] = time.perf_counter() - started
        _write_json(args.output_json, report)
        return 0
    except Exception as exc:
        if report.get("primary_output_status") not in {
            "not_owned_due_to_path_collision",
            "preexisting_untrusted_preserved",
        }:
            report["primary_output_status"] = "missing"
        report.update(
            {
                "status": "failed",
                "failure_phase": phase,
                "last_trustworthy_phase": last_trustworthy_phase,
                "elapsed_seconds": time.perf_counter() - started,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        try:
            _write_json(failure_report, report)
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
