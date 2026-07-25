#!/usr/bin/env python3
"""Continue exact negative block-0 interventions through the source CUDA suffix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Callable

import numpy as np

try:
    from scripts.source_cuda_shape_flow_suffix_ladder import (
        STEPS,
        _compare_arrays,
        _invalidate_primary_output,
        _load_conditioning,
        _load_source_anchor,
        _run_suffix,
        _schedule_pairs,
        _sha256,
        _validate_file,
        classify_anchor,
        load_mlx_trajectory,
    )
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        _array_sha256,
        _load_accepted_suffix,
        _report_path_guard,
        _validate_sha256,
        _write_json,
        load_mlx_transition0_components,
        require_exact_source_control,
    )
except ImportError:
    from source_cuda_shape_flow_suffix_ladder import (  # type: ignore[no-redef]
        STEPS,
        _compare_arrays,
        _invalidate_primary_output,
        _load_conditioning,
        _load_source_anchor,
        _run_suffix,
        _schedule_pairs,
        _sha256,
        _validate_file,
        classify_anchor,
        load_mlx_trajectory,
    )
    from source_cuda_shape_flow_transition0_recoverability import (  # type: ignore[no-redef]
        _array_sha256,
        _load_accepted_suffix,
        _report_path_guard,
        _validate_sha256,
        _write_json,
        load_mlx_transition0_components,
        require_exact_source_control,
    )


SCHEMA = "trellis2mlx.source_cuda_shape_flow_negative_intervention_suffix.v1"
ARTIFACT_SCHEMA = f"{SCHEMA}.artifact"
INTERVENTION_STAGES = ("norm1", "attention_raw", "after_mlp")
EXPECTED_CANDIDATE_NAMES = (
    "source-native-control",
    "source-pos-neg-block0-norm1",
    "source-pos-neg-block0-attention-raw",
    "source-pos-neg-block0-after-mlp",
)


def intervention_candidate_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "source-native-control",
            "positive": "source",
            "negative": "source",
            "post": "source-guidance-rescale-euler",
            "intervention_stage": None,
        },
        *[
            {
                "name": f"source-pos-neg-block0-{stage.replace('_', '-')}",
                "positive": "source",
                "negative": "mlx-after-source-block0-stage",
                "post": "source-guidance-rescale-euler",
                "intervention_stage": stage,
            }
            for stage in INTERVENTION_STAGES
        ],
    ]


def _require_equal(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise ValueError(f"intervention {label} must be {expected!r}, got {actual!r}")


def _require_exact_array(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    label: str,
) -> None:
    if actual.dtype != expected.dtype:
        raise ValueError(
            f"intervention {label} dtype differs: {actual.dtype} != {expected.dtype}"
        )
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise ValueError(f"intervention {label} differs from the admitted witness")


def load_negative_intervention(
    *,
    checkpoint_path: Path,
    report_path: Path,
    expected_checkpoint_sha256: str,
    expected_report_sha256: str,
    expected_stage: str,
    expected_trace_sha256: str,
    expected_coords: np.ndarray,
    expected_noise: np.ndarray,
    expected_mlx_pred_pos: np.ndarray,
    expected_conditioning_sha256: str,
    expected_support_sha256: str,
    expected_noise_sample_sha256: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if expected_stage not in INTERVENTION_STAGES:
        raise ValueError(f"intervention stage is not admitted: {expected_stage!r}")
    expected_checkpoint_sha256 = _validate_sha256(
        expected_checkpoint_sha256, label=f"{expected_stage} checkpoint SHA256"
    )
    expected_report_sha256 = _validate_sha256(
        expected_report_sha256, label=f"{expected_stage} report SHA256"
    )
    expected_trace_sha256 = _validate_sha256(
        expected_trace_sha256, label="source block trace SHA256"
    )
    for label, value in (
        ("conditioning SHA256", expected_conditioning_sha256),
        ("support SHA256", expected_support_sha256),
        ("noise sample SHA256", expected_noise_sample_sha256),
    ):
        _validate_sha256(value, label=label)

    _validate_file(checkpoint_path, label=f"{expected_stage} checkpoint")
    _validate_file(report_path, label=f"{expected_stage} report")
    checkpoint_sha256 = _sha256(checkpoint_path)
    report_sha256 = _sha256(report_path)
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(
            f"{expected_stage} checkpoint SHA256 mismatch: "
            f"{checkpoint_sha256} != {expected_checkpoint_sha256}"
        )
    if report_sha256 != expected_report_sha256:
        raise ValueError(
            f"{expected_stage} report SHA256 mismatch: "
            f"{report_sha256} != {expected_report_sha256}"
        )

    report = json.loads(report_path.read_text())
    _require_equal(
        report.get("schema"),
        "trellis2mlx.mlx_stage_capture_run_report.v1",
        label="report schema",
    )
    _require_equal(report.get("status"), "done", label="report status")
    _require_equal(report.get("exit_code"), 0, label="report exit code")
    _require_equal(report.get("failure_phase"), None, label="report failure phase")
    _require_equal(
        report.get("last_trustworthy_phase"),
        "shape_flow_step_saved",
        label="last trustworthy phase",
    )
    _require_equal(
        report.get("primary_output_status"),
        "written",
        label="primary output status",
    )
    artifact = report.get("artifacts", {}).get("shape_flow_step.npz", {})
    _require_equal(
        artifact.get("sha256"), checkpoint_sha256, label="artifact digest"
    )
    _require_equal(
        artifact.get("size_bytes"),
        checkpoint_path.stat().st_size,
        label="artifact size",
    )

    route_identity = report.get("route_identity", {})
    _require_equal(
        route_identity.get("requested_stop"),
        "shape_flow_step",
        label="requested stop",
    )
    env = route_identity.get("env", {})
    _require_equal(
        env.get("TRELLIS2MLX_ATTENTION_BACKEND"),
        "fast",
        label="attention environment",
    )
    route = route_identity.get("route", {})
    route_requirements = {
        "family": "trellis2mlx/mlx",
        "backend": "mlx-metal",
        "attention_backend": "fast",
        "cascade": False,
        "steps": STEPS,
        "conditioning_sample_sha256": expected_conditioning_sha256,
        "shape_slat_support_sample_sha256": expected_support_sha256,
        "shape_flow_noise_sample_sha256": expected_noise_sample_sha256,
        "shape_flow_block_injection_trace_sha256": expected_trace_sha256,
        "shape_flow_block_injection_step_index": 0,
        "shape_flow_block_injection_block_index": 0,
        "shape_flow_block_injection_branch": "neg",
        "shape_flow_block_injection_stage": expected_stage,
        "shape_flow_block_injection_scale": 1.0,
    }
    for key, expected in route_requirements.items():
        _require_equal(route.get(key), expected, label=f"route {key}")

    required_arrays = {
        "coords",
        "noise",
        "sample_feats",
        "pred_pos",
        "pred_neg",
        "t",
        "t_prev",
        "steps",
        "guidance_strength",
        "guidance_rescale",
        "guidance_interval",
        "rescale_t",
        "shape_flow_block_injection_json",
    }
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        missing = sorted(required_arrays - set(archive.files))
        if missing:
            raise ValueError(
                f"intervention checkpoint is missing arrays: {missing}"
            )
        arrays = {name: np.asarray(archive[name]) for name in required_arrays}

    _require_exact_array(
        arrays["coords"], np.asarray(expected_coords), label="coordinates"
    )
    _require_exact_array(
        arrays["noise"], np.asarray(expected_noise), label="noise"
    )
    _require_exact_array(
        arrays["sample_feats"], np.asarray(expected_noise), label="sample"
    )
    _require_exact_array(
        arrays["pred_pos"],
        np.asarray(expected_mlx_pred_pos),
        label="positive prediction",
    )
    pred_neg = arrays["pred_neg"]
    if (
        pred_neg.dtype != np.float32
        or pred_neg.shape != expected_noise.shape
        or not np.isfinite(pred_neg).all()
    ):
        raise ValueError(
            "intervention negative prediction must be finite float32 with witness shape"
        )

    scalar_requirements = {
        "steps": STEPS,
        "guidance_strength": 7.5,
        "guidance_rescale": 0.5,
        "rescale_t": 3.0,
    }
    for name, expected in scalar_requirements.items():
        value = arrays[name]
        if value.shape != ():
            raise ValueError(f"intervention {name} must be scalar")
        _require_equal(float(value.item()), float(expected), label=name)
    interval = arrays["guidance_interval"]
    if (
        interval.dtype != np.float32
        or interval.shape != (2,)
        or not np.array_equal(interval, np.asarray([0.6, 1.0], dtype=np.float32))
    ):
        raise ValueError("intervention guidance interval is not the admitted route")
    expected_t, expected_t_prev = _schedule_pairs(STEPS, 3.0)[0]
    for name, expected in (("t", expected_t), ("t_prev", expected_t_prev)):
        value = arrays[name]
        if (
            value.dtype != np.float32
            or value.shape != ()
            or not np.isfinite(value)
            or value.item() != np.float32(expected)
        ):
            raise ValueError(f"intervention {name} is invalid")

    raw_injection = arrays["shape_flow_block_injection_json"]
    if raw_injection.shape != () or raw_injection.dtype.kind not in {"U", "S"}:
        raise ValueError("intervention injection identity must be a string scalar")
    injection = json.loads(str(raw_injection.item()))
    injection_requirements = {
        "step_index": 0,
        "block_index": 0,
        "branch": "neg",
        "stage": expected_stage,
    }
    for key, expected in injection_requirements.items():
        _require_equal(injection.get(key), expected, label=f"injection {key}")
    injection_scale = injection.get("source_delta_scale", injection.get("scale"))
    _require_equal(injection_scale, 1.0, label="injection scale")
    _require_equal(
        injection.get("trace_sha256"),
        expected_trace_sha256,
        label="injection trace",
    )
    trace_identity = injection.get("trace_identity")
    if trace_identity is not None:
        if not isinstance(trace_identity, dict):
            raise ValueError("intervention trace identity must be an object")
        for key, expected in {
            "device": "cuda",
            "effective_device_type": "cuda",
            "conditioning_sha256": expected_conditioning_sha256,
            "shape_slat_support_sample_sha256": expected_support_sha256,
            "shape_flow_noise_sample_sha256": expected_noise_sample_sha256,
            "steps": STEPS,
        }.items():
            _require_equal(
                trace_identity.get(key), expected, label=f"trace identity {key}"
            )

    return np.ascontiguousarray(pred_neg), {
        "stage": expected_stage,
        "checkpoint_sha256": checkpoint_sha256,
        "report_sha256": report_sha256,
        "trace_sha256": expected_trace_sha256,
        "route": route,
        "pred_neg_sha256": _array_sha256(pred_neg),
    }


def run_control_gated_interventions(
    *,
    specs: list[dict[str, Any]],
    execute_candidate: Callable[[int, dict[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    if tuple(spec["name"] for spec in specs) != EXPECTED_CANDIDATE_NAMES:
        raise ValueError("intervention candidate specs are incomplete or out of order")
    control = execute_candidate(0, specs[0])
    require_exact_source_control(
        candidate_index=0, metrics=control.get("vs_source_anchor", {})
    )
    results = [control]
    for index, spec in enumerate(specs[1:], start=1):
        results.append(execute_candidate(index, spec))
    return results


def validate_result_manifest(payload: dict[str, Any]) -> None:
    if payload.get("status") != "done":
        raise ValueError("negative intervention suffix result is not done")
    route = payload.get("effective_route", {})
    required_route = {
        "device_type": "cuda",
        "attention_backend": "sdpa",
        "conv_backend": "none",
        "steps": STEPS,
        "one_model_load": True,
        "candidate_names": list(EXPECTED_CANDIDATE_NAMES),
    }
    for key, expected in required_route.items():
        if route.get(key) != expected:
            raise ValueError(
                f"negative intervention suffix route {key} must be {expected!r}"
            )
    candidates = payload.get("candidates", [])
    if [candidate.get("name") for candidate in candidates] != list(
        EXPECTED_CANDIDATE_NAMES
    ):
        raise ValueError("negative intervention candidates are incomplete or out of order")
    if len({candidate.get("output_key") for candidate in candidates}) != len(candidates):
        raise ValueError("negative intervention output keys are not distinct")
    for candidate in candidates:
        if candidate.get("source_step_indices") != list(range(1, STEPS)):
            raise ValueError(f"candidate {candidate.get('name')} has wrong source steps")
        if candidate.get("source_step_count") != STEPS - 1:
            raise ValueError(f"candidate {candidate.get('name')} has wrong step count")
        if len(candidate.get("step_elapsed_seconds", [])) != STEPS - 1:
            raise ValueError(f"candidate {candidate.get('name')} has wrong timing count")
    control = candidates[0].get("vs_source_anchor", {})
    if not control.get("exact") or control.get("nonzero") != 0:
        raise ValueError("source-native control does not exactly reproduce source anchor")
    expected_steps = len(EXPECTED_CANDIDATE_NAMES) * (STEPS - 1)
    timing = payload.get("timing", {})
    for key, expected in {
        "source_steps_completed": expected_steps,
        "source_steps_requested": expected_steps,
        "candidates_completed": len(EXPECTED_CANDIDATE_NAMES),
        "candidates_requested": len(EXPECTED_CANDIDATE_NAMES),
    }.items():
        if timing.get(key) != expected:
            raise ValueError(f"negative intervention timing {key} must be {expected}")


def _artifact_metadata(
    report: dict[str, Any], arrays: dict[str, np.ndarray]
) -> dict[str, Any]:
    return {
        "schema": ARTIFACT_SCHEMA,
        "artifact_status": "computed_pending_serialization",
        "external_report_required": True,
        "effective_route": report["effective_route"],
        "inputs": report["inputs"],
        "candidate_specs": report["candidate_specs"],
        "candidates": report["candidates"],
        "anchors": report["anchors"],
        "arrays": {
            name: {
                "dtype": str(value.dtype),
                "shape": [int(dimension) for dimension in value.shape],
                "sha256": _array_sha256(value),
            }
            for name, value in sorted(arrays.items())
        },
    }


def validate_saved_artifact(
    path: Path, *, candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "coords",
            "source_anchor_shape_slat",
            "mlx_anchor_shape_slat",
            "source_transition0_pred_pos",
            "source_transition0_pred_neg",
            "metadata_json",
            *(
                f"candidate_{index}_transition0_sample_next"
                for index in range(len(EXPECTED_CANDIDATE_NAMES))
            ),
            *(str(candidate["output_key"]) for candidate in candidates),
            *(
                f"intervention_{stage}_pred_neg"
                for stage in INTERVENTION_STAGES
            ),
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(
                f"saved negative intervention artifact is missing arrays: {missing}"
            )
        raw_metadata = np.asarray(archive["metadata_json"])
        if raw_metadata.shape != () or raw_metadata.dtype.kind not in {"U", "S"}:
            raise ValueError("saved intervention metadata_json must be a string scalar")
        metadata = json.loads(str(raw_metadata.item()))
        if metadata.get("schema") != ARTIFACT_SCHEMA:
            raise ValueError("saved intervention artifact metadata schema is invalid")
        if metadata.get("candidates") != candidates:
            raise ValueError("saved intervention candidates differ from report")
        manifest = metadata.get("arrays")
        archive_keys = set(archive.files) - {"metadata_json"}
        if not isinstance(manifest, dict) or set(manifest) != archive_keys:
            raise ValueError("saved intervention artifact array manifest mismatch")
        sample_shape = tuple(np.asarray(archive["source_anchor_shape_slat"]).shape)
        for name, expected in manifest.items():
            array = np.asarray(archive[name])
            if (
                str(array.dtype) != expected.get("dtype")
                or list(array.shape) != expected.get("shape")
                or _array_sha256(array) != expected.get("sha256")
            ):
                raise ValueError(f"{name} differs from intervention manifest")
            expected_dtype = np.int32 if name == "coords" else np.float32
            if array.dtype != expected_dtype:
                raise ValueError(f"{name} must have dtype {expected_dtype}")
            if name == "coords":
                if array.ndim != 2 or array.shape[1] != 4:
                    raise ValueError("coords must have shape [N,4]")
            elif array.shape != sample_shape or not np.isfinite(array).all():
                raise ValueError(f"{name} has invalid shape or values")
        for candidate in candidates:
            output = np.asarray(archive[str(candidate["output_key"])])
            if (
                list(output.shape) != candidate.get("shape")
                or _array_sha256(output) != candidate.get("sha256")
            ):
                raise ValueError(f"{candidate.get('name')} output differs from report")
    return {
        "schema": f"{SCHEMA}.saved_artifact",
        "candidate_count": len(candidates),
        "all_arrays_bound": True,
        "array_count": len(manifest),
    }


def _write_artifact(
    output_path: Path,
    report_path: Path,
    arrays: dict[str, np.ndarray],
    report: dict[str, Any],
) -> int:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **arrays)
        validation = validate_saved_artifact(
            output_path, candidates=report.get("candidates", [])
        )
        report.update(
            {
                "status": "done",
                "primary_output_status": "written",
                "primary_output": {
                    "path": str(output_path),
                    "sha256": _sha256(output_path),
                    "size_bytes": output_path.stat().st_size,
                    "keys": sorted(arrays),
                    "validation": validation,
                },
                "failure_phase": None,
                "last_trustworthy_phase": "all_intervention_candidates_saved",
            }
        )
        _write_json(report_path, report)
        return 0
    except Exception as exc:
        removed = False
        if output_path.is_file():
            output_path.unlink()
            removed = True
        report.update(
            {
                "status": "failed",
                "primary_output_status": "invalid_removed" if removed else "missing",
                "failure_phase": "write_outputs",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(report_path, report)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--mlx-shape-flow-steps", required=True, type=Path)
    parser.add_argument("--mlx-shape-flow-steps-sha256", required=True)
    parser.add_argument("--mlx-run-report", required=True, type=Path)
    parser.add_argument("--mlx-run-report-sha256", required=True)
    parser.add_argument("--conditioning", required=True, type=Path)
    parser.add_argument("--conditioning-sha256", required=True)
    parser.add_argument("--accepted-source-baseline", required=True, type=Path)
    parser.add_argument("--accepted-source-baseline-sha256", required=True)
    parser.add_argument("--accepted-source-report", required=True, type=Path)
    parser.add_argument("--accepted-source-report-sha256", required=True)
    parser.add_argument("--accepted-suffix-result", required=True, type=Path)
    parser.add_argument("--accepted-suffix-result-sha256", required=True)
    parser.add_argument("--accepted-suffix-report", required=True, type=Path)
    parser.add_argument("--accepted-suffix-report-sha256", required=True)
    parser.add_argument("--source-tar", required=True, type=Path)
    parser.add_argument("--source-tar-sha256", required=True)
    parser.add_argument("--source-block-trace-sha256", required=True)
    for stage in INTERVENTION_STAGES:
        option = stage.replace("_", "-")
        parser.add_argument(
            f"--{option}-checkpoint", required=True, type=Path
        )
        parser.add_argument(f"--{option}-checkpoint-sha256", required=True)
        parser.add_argument(f"--{option}-report", required=True, type=Path)
        parser.add_argument(f"--{option}-report-sha256", required=True)
    parser.add_argument("--model-repo", default="microsoft/TRELLIS.2-4B")
    parser.add_argument("--pipeline-config", default="pipeline.json")
    parser.add_argument("--sparse-conv-backend", default="none")
    parser.add_argument("--sparse-attn-backend", default="sdpa")
    parser.add_argument("--no-download", action="store_true")
    return parser


def _stage_arg(args: argparse.Namespace, stage: str, suffix: str) -> Any:
    return getattr(args, f"{stage}_{suffix}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    phase = "arguments_parsed"
    last_trustworthy_phase: str | None = phase
    requested_route = {
        "route": "official-source-cuda-negative-intervention-suffix",
        "steps": STEPS,
        "candidate_specs": intervention_candidate_specs(),
        "attention_backend": args.sparse_attn_backend,
        "conv_backend": args.sparse_conv_backend,
    }
    input_paths = {
        "MLX shape-flow steps": args.mlx_shape_flow_steps,
        "MLX run report": args.mlx_run_report,
        "conditioning": args.conditioning,
        "accepted source baseline": args.accepted_source_baseline,
        "accepted source report": args.accepted_source_report,
        "accepted suffix result": args.accepted_suffix_result,
        "accepted suffix report": args.accepted_suffix_report,
        "source tar": args.source_tar,
    }
    for stage in INTERVENTION_STAGES:
        input_paths[f"{stage} checkpoint"] = _stage_arg(args, stage, "checkpoint")
        input_paths[f"{stage} report"] = _stage_arg(args, stage, "report")
    failure_report_path, report_collisions = _report_path_guard(
        args.output_json, args.output_npz, input_paths
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "requested_route": requested_route,
        "effective_route": "not-established",
        "primary_output_status": "missing",
        "failure_phase": None,
        "last_trustworthy_phase": last_trustworthy_phase,
        "phase_timings": {},
        "requested_output_json": str(args.output_json),
        "effective_failure_report": str(failure_report_path),
    }
    try:
        phase = "request_validation"
        phase_started = time.perf_counter()
        if report_collisions:
            raise ValueError(
                "output JSON collides with protected paths: "
                + ", ".join(report_collisions)
            )
        if args.output_npz.exists():
            report["primary_output_status"] = "preexisting_untrusted_preserved"
        digest_args = {
            "MLX shape-flow steps": args.mlx_shape_flow_steps_sha256,
            "MLX run report": args.mlx_run_report_sha256,
            "conditioning": args.conditioning_sha256,
            "accepted source baseline": args.accepted_source_baseline_sha256,
            "accepted source report": args.accepted_source_report_sha256,
            "accepted suffix result": args.accepted_suffix_result_sha256,
            "accepted suffix report": args.accepted_suffix_report_sha256,
            "source tar": args.source_tar_sha256,
        }
        for stage in INTERVENTION_STAGES:
            digest_args[f"{stage} checkpoint"] = _stage_arg(
                args, stage, "checkpoint_sha256"
            )
            digest_args[f"{stage} report"] = _stage_arg(
                args, stage, "report_sha256"
            )
        expected_digests = {
            label: _validate_sha256(value, label=f"{label} SHA256")
            for label, value in digest_args.items()
        }
        source_trace_sha256 = _validate_sha256(
            args.source_block_trace_sha256, label="source block trace SHA256"
        )
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
            args.mlx_shape_flow_steps, args.mlx_run_report, args.conditioning
        )
        shape = tuple(int(value) for value in trajectory["noise"].shape)
        mlx_components = load_mlx_transition0_components(
            args.mlx_shape_flow_steps,
            expected_sha256=expected_digests["MLX shape-flow steps"],
            expected_shape=shape,
        )
        source_anchor, source_identity = _load_source_anchor(
            args.accepted_source_baseline,
            args.accepted_source_report,
            coords=trajectory["coords"],
            conditioning_sha256=mlx_identity["conditioning_sha256"],
            noise_sha256=mlx_identity["shape_flow_noise_sample_sha256"],
            source_tar_sha256=expected_digests["source tar"],
        )
        suffix_arrays, suffix_identity = _load_accepted_suffix(
            args.accepted_suffix_result,
            args.accepted_suffix_report,
            expected_result_sha256=expected_digests["accepted suffix result"],
            expected_report_sha256=expected_digests["accepted suffix report"],
            source_anchor=source_anchor,
            coords=trajectory["coords"],
            expected_mlx_identity=mlx_identity,
            expected_source_identity=source_identity,
            expected_conditioning_sha256=expected_digests["conditioning"],
            expected_source_tar_sha256=expected_digests["source tar"],
        )
        interventions: dict[str, np.ndarray] = {}
        intervention_identities: dict[str, dict[str, Any]] = {}
        for stage in INTERVENTION_STAGES:
            pred_neg, identity = load_negative_intervention(
                checkpoint_path=_stage_arg(args, stage, "checkpoint"),
                report_path=_stage_arg(args, stage, "report"),
                expected_checkpoint_sha256=expected_digests[
                    f"{stage} checkpoint"
                ],
                expected_report_sha256=expected_digests[f"{stage} report"],
                expected_stage=stage,
                expected_trace_sha256=source_trace_sha256,
                expected_coords=trajectory["coords"],
                expected_noise=mlx_components["noise"],
                expected_mlx_pred_pos=mlx_components["pred_pos"],
                expected_conditioning_sha256=mlx_identity[
                    "conditioning_sha256"
                ],
                expected_support_sha256=mlx_identity[
                    "shape_slat_support_sample_sha256"
                ],
                expected_noise_sample_sha256=mlx_identity[
                    "shape_flow_noise_sample_sha256"
                ],
            )
            interventions[stage] = pred_neg
            intervention_identities[stage] = identity
        cond_np, neg_cond_np = _load_conditioning(args.conditioning)
        report["inputs"] = {
            "expected_digests": expected_digests,
            "source_block_trace_sha256": source_trace_sha256,
            "mlx": mlx_identity,
            "accepted_source": source_identity,
            "accepted_suffix": suffix_identity,
            "interventions": intervention_identities,
            "coords_shape": [int(value) for value in trajectory["coords"].shape],
            "sample_shape": list(shape),
        }
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        last_trustworthy_phase = phase
        if args.no_download:
            raise RuntimeError("--no-download stops after validated local inputs by request")

        phase = "extract_source"
        phase_started = time.perf_counter()
        from source_cuda_shape_block_trace import extract_source

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
        from trellis2.modules.sparse import SparseTensor
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
        sampler_params = {**pipeline_args["shape_slat_sampler"]["params"], "steps": STEPS}
        sampler = getattr(samplers, pipeline_args["shape_slat_sampler"]["name"])(
            **pipeline_args["shape_slat_sampler"]["args"]
        )
        normalization_std = np.asarray(
            pipeline_args["shape_slat_normalization"]["std"], dtype=np.float32
        )[None]
        normalization_mean = np.asarray(
            pipeline_args["shape_slat_normalization"]["mean"], dtype=np.float32
        )[None]
        report["pipeline_config"] = {
            "path": str(config_path),
            "sampler_name": pipeline_args["shape_slat_sampler"]["name"],
            "sampler_params": sampler_params,
            "shape_slat_normalization": pipeline_args["shape_slat_normalization"],
        }
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        last_trustworthy_phase = phase

        phase = "load_model"
        phase_started = time.perf_counter()
        from source_cuda_shape_block_trace import resolve_model_ref

        model_ref = resolve_model_ref(
            args.model_repo, pipeline_args["models"]["shape_slat_flow_model_512"]
        )
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

        phase = "negative_intervention_suffix"
        phase_started = time.perf_counter()
        from source_cuda_shape_block29_basin_map import (
            _flow_forward,
            _guided_prediction,
        )

        coords = torch.from_numpy(trajectory["coords"]).to(
            device=device, dtype=torch.int32
        )
        cond = torch.from_numpy(cond_np).to(device=device, dtype=torch.float32)
        neg_cond = torch.from_numpy(neg_cond_np).to(
            device=device, dtype=torch.float32
        )
        noise = torch.from_numpy(mlx_components["noise"]).to(
            device=device, dtype=torch.float32
        )
        sample = SparseTensor(feats=noise, coords=coords)
        t, t_prev = _schedule_pairs(STEPS, float(sampler_params["rescale_t"]))[0]
        t_model = torch.tensor(
            [1000.0 * t] * sample.shape[0], device=device, dtype=torch.float32
        )
        source_pos = _flow_forward(
            torch, flow_model, sample, t_model, cond, branch="pos", targets=None
        )
        source_neg = _flow_forward(
            torch, flow_model, sample, t_model, neg_cond, branch="neg", targets=None
        )
        source_pos_np = (
            source_pos.feats.detach().float().cpu().numpy().astype(np.float32)
        )
        source_neg_np = (
            source_neg.feats.detach().float().cpu().numpy().astype(np.float32)
        )
        arrays: dict[str, np.ndarray] = {
            "coords": np.asarray(trajectory["coords"], dtype=np.int32),
            "source_anchor_shape_slat": source_anchor,
            "mlx_anchor_shape_slat": suffix_arrays["mlx_anchor_shape_slat"],
            "source_transition0_pred_pos": source_pos_np,
            "source_transition0_pred_neg": source_neg_np,
            **{
                f"intervention_{stage}_pred_neg": value
                for stage, value in interventions.items()
            },
        }
        continuation_seconds = 0.0
        specs = intervention_candidate_specs()

        def execute_candidate(
            index: int, spec: dict[str, Any]
        ) -> dict[str, Any]:
            nonlocal continuation_seconds
            candidate_started = time.perf_counter()
            stage = spec["intervention_stage"]
            if stage is None:
                pred_neg = source_neg
            else:
                pred_neg = sample.replace(
                    torch.from_numpy(interventions[stage]).to(
                        device=sample.device, dtype=sample.dtype
                    )
                )
            pred = _guided_prediction(
                sampler=sampler,
                sample=sample,
                pred_pos=source_pos,
                pred_neg=pred_neg,
                t=t,
                guidance_strength=float(sampler_params["guidance_strength"]),
                guidance_rescale=float(sampler_params["guidance_rescale"]),
                guidance_interval=tuple(
                    float(value) for value in sampler_params["guidance_interval"]
                ),
            )
            start = sample - (t - t_prev) * pred
            start_np = (
                start.feats.detach().float().cpu().numpy().astype(np.float32)
            )
            arrays[f"candidate_{index}_transition0_sample_next"] = start_np
            result_raw, step_timings = _run_suffix(
                torch=torch,
                flow_model=flow_model,
                sampler=sampler,
                coords=coords,
                start_feats=start.feats,
                cond=cond,
                neg_cond=neg_cond,
                params=sampler_params,
                switch_step=1,
            )
            result_raw_np = (
                result_raw.feats.detach().float().cpu().numpy().astype(np.float32)
            )
            result = result_raw_np * normalization_std + normalization_mean
            output_key = f"candidate_{index}_shape_slat"
            arrays[output_key] = result
            vs_source = _compare_arrays(result, source_anchor)
            vs_mlx = _compare_arrays(result, suffix_arrays["mlx_anchor_shape_slat"])
            elapsed = time.perf_counter() - candidate_started
            continuation_seconds += elapsed
            candidate = {
                **spec,
                "output_key": output_key,
                "shape": [int(value) for value in result.shape],
                "sha256": _array_sha256(result),
                "transition0_sample_next_sha256": _array_sha256(start_np),
                "transition0_vs_source_control": None,
                "source_step_indices": list(range(1, STEPS)),
                "source_step_count": STEPS - 1,
                "step_elapsed_seconds": step_timings,
                "elapsed_seconds": elapsed,
                "vs_source_anchor": vs_source,
                "vs_mlx_anchor": vs_mlx,
                "nearest_anchor": classify_anchor(
                    float(vs_source["mean_abs"]), float(vs_mlx["mean_abs"])
                ),
            }
            if index == 0:
                candidate["transition0_vs_source_control"] = {
                    "exact": True,
                    "mean_abs": 0.0,
                    "max_abs": 0.0,
                    "nonzero": 0,
                }
            del result_raw, start, pred_neg
            torch.cuda.empty_cache()
            return candidate

        candidate_results = run_control_gated_interventions(
            specs=specs,
            execute_candidate=execute_candidate,
        )
        control_start = arrays["candidate_0_transition0_sample_next"]
        for index, candidate in enumerate(candidate_results[1:], start=1):
            candidate["transition0_vs_source_control"] = _compare_arrays(
                arrays[f"candidate_{index}_transition0_sample_next"],
                control_start,
            )
        effective_route = {
            "route": requested_route["route"],
            "device_type": next(flow_model.parameters()).device.type,
            "cuda_device": torch.cuda.get_device_name(0),
            "attention_backend": getattr(sparse_config, "ATTN", None),
            "conv_backend": getattr(sparse_config, "CONV", None),
            "steps": STEPS,
            "one_model_load": True,
            "model_ref": model_ref,
            "candidate_names": list(EXPECTED_CANDIDATE_NAMES),
            "comparison_class": (
                "source-positive-negative-block0-intervention-"
                "source-post-and-cuda-suffix"
            ),
        }
        report.update(
            {
                "status": "done",
                "effective_route": effective_route,
                "candidate_specs": specs,
                "candidates": candidate_results,
                "direct_transition0_metrics": {
                    "source_vs_mlx_pred_pos": _compare_arrays(
                        source_pos_np, mlx_components["pred_pos"]
                    ),
                    "source_vs_mlx_pred_neg": _compare_arrays(
                        source_neg_np, mlx_components["pred_neg"]
                    ),
                    "intervention_vs_source_pred_neg": {
                        stage: _compare_arrays(value, source_neg_np)
                        for stage, value in interventions.items()
                    },
                    "intervention_vs_mlx_pred_neg": {
                        stage: _compare_arrays(
                            value, mlx_components["pred_neg"]
                        )
                        for stage, value in interventions.items()
                    },
                },
                "anchors": {
                    "source_sha256": _array_sha256(source_anchor),
                    "mlx_sha256": _array_sha256(
                        suffix_arrays["mlx_anchor_shape_slat"]
                    ),
                },
                "timing": {
                    "model_load_seconds": model_load_seconds,
                    "continuation_seconds": continuation_seconds,
                    "source_steps_completed": len(EXPECTED_CANDIDATE_NAMES)
                    * (STEPS - 1),
                    "source_steps_requested": len(EXPECTED_CANDIDATE_NAMES)
                    * (STEPS - 1),
                    "candidates_completed": len(candidate_results),
                    "candidates_requested": len(EXPECTED_CANDIDATE_NAMES),
                    "t4_compute_seconds_through_matrix": time.perf_counter()
                    - started,
                },
                "forbidden_inferences": [
                    "not final mesh, texture, winding, or GLB evidence",
                    "not proof that an intervention boundary is a production fix",
                    "not evidence outside the exact admitted hostile input and transition",
                ],
            }
        )
        validate_result_manifest(report)
        arrays["metadata_json"] = np.asarray(
            json.dumps(
                _artifact_metadata(report, arrays),
                sort_keys=True,
                allow_nan=False,
            )
        )
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        last_trustworthy_phase = phase

        phase = "write_outputs"
        report["elapsed_seconds"] = time.perf_counter() - started
        return _write_artifact(args.output_npz, args.output_json, arrays, report)
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
            _write_json(failure_report_path, report)
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
