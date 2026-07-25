#!/usr/bin/env python3
"""Localize transition-0 recoverability with source/MLX causal substitutions."""

from __future__ import annotations

import argparse
import hashlib
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


SCHEMA = "trellis2mlx.source_cuda_shape_flow_transition0_recoverability.v1"
ARTIFACT_SCHEMA = f"{SCHEMA}.artifact"
EXPECTED_CANDIDATE_NAMES = (
    "source-native-control",
    "mlx-pos-source-neg",
    "source-pos-mlx-neg",
    "mlx-both-source-post",
    "mlx-final-source-euler",
)
MLX_COMPONENT_NAMES = (
    "noise",
    "sample_in",
    "pred_pos",
    "pred_neg",
    "pred_cfg",
    "x0_pos",
    "x0_cfg",
    "x0_rescaled",
    "x0_after_rescale",
    "pred_final",
    "sample_next",
)


def transition0_candidate_specs() -> list[dict[str, str]]:
    return [
        {
            "name": "source-native-control",
            "positive": "source",
            "negative": "source",
            "post": "source-guidance-rescale-euler",
        },
        {
            "name": "mlx-pos-source-neg",
            "positive": "mlx",
            "negative": "source",
            "post": "source-guidance-rescale-euler",
        },
        {
            "name": "source-pos-mlx-neg",
            "positive": "source",
            "negative": "mlx",
            "post": "source-guidance-rescale-euler",
        },
        {
            "name": "mlx-both-source-post",
            "positive": "mlx",
            "negative": "mlx",
            "post": "source-guidance-rescale-euler",
        },
        {
            "name": "mlx-final-source-euler",
            "positive": "mlx",
            "negative": "mlx",
            "post": "mlx-final-source-euler",
        },
    ]


def compose_candidate_pairs(
    *,
    source_pos: Any,
    source_neg: Any,
    mlx_pos: Any,
    mlx_neg: Any,
) -> dict[str, tuple[Any, Any]]:
    return {
        "source-native-control": (source_pos, source_neg),
        "mlx-pos-source-neg": (mlx_pos, source_neg),
        "source-pos-mlx-neg": (source_pos, mlx_neg),
        "mlx-both-source-post": (mlx_pos, mlx_neg),
    }


def _validate_sha256(value: str, *, label: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{label} must be a canonical lowercase SHA256")
    return normalized


def load_mlx_transition0_components(
    path: Path,
    *,
    expected_sha256: str,
    expected_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    expected_sha256 = _validate_sha256(expected_sha256, label="MLX trajectory SHA256")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"MLX trajectory SHA256 mismatch: {actual_sha256} != {expected_sha256}"
        )
    components: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(MLX_COMPONENT_NAMES) - set(archive.files))
        if missing:
            raise ValueError(f"MLX trajectory is missing transition arrays: {missing}")
        for name in MLX_COMPONENT_NAMES:
            array = np.asarray(archive[name])
            if array.dtype != np.float32:
                raise ValueError(f"MLX transition array {name} must be float32")
            if name == "noise":
                if array.shape != expected_shape:
                    raise ValueError(
                        f"MLX transition array noise has shape {array.shape}, "
                        f"expected {expected_shape}"
                    )
                selected = array
            else:
                if array.ndim != 3 or array.shape[0] != STEPS:
                    raise ValueError(
                        f"MLX transition array {name} must have shape [8,N,C], got {array.shape}"
                    )
                if tuple(array.shape[1:]) != expected_shape:
                    raise ValueError(
                        f"MLX transition array {name} has trailing shape {array.shape[1:]}, "
                        f"expected {expected_shape}"
                    )
                selected = array[0]
            if not np.isfinite(selected).all():
                raise ValueError(f"MLX transition array {name} contains non-finite values")
            components[name] = np.ascontiguousarray(selected)
    if not np.array_equal(components["noise"], components["sample_in"]):
        raise ValueError("MLX transition-0 sample does not equal admitted noise")
    return components


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _paths_alias(left: Path, right: Path) -> bool:
    if left.exists() and right.exists():
        try:
            if os.path.samefile(left, right):
                return True
        except OSError:
            pass
    return left.expanduser().resolve(strict=False) == right.expanduser().resolve(
        strict=False
    )


def _report_path_guard(
    requested_report: Path,
    primary_output: Path,
    input_paths: dict[str, Path],
) -> tuple[Path, list[str]]:
    protected = {"primary output": primary_output, **input_paths}
    collisions = [
        label
        for label, path in protected.items()
        if _paths_alias(requested_report, path)
    ]
    if not collisions:
        return requested_report, []

    suffix = ".transition0-recoverability.failure.json"
    index = 0
    while True:
        discriminator = "" if index == 0 else f".{index}"
        candidate = requested_report.with_name(
            requested_report.name + suffix + discriminator
        )
        if not any(
            _paths_alias(candidate, path)
            for path in [requested_report, primary_output, *input_paths.values()]
        ):
            return candidate, collisions
        index += 1


def _load_accepted_suffix(
    result_path: Path,
    report_path: Path,
    *,
    expected_result_sha256: str,
    expected_report_sha256: str,
    source_anchor: np.ndarray,
    coords: np.ndarray,
    expected_mlx_identity: dict[str, Any],
    expected_source_identity: dict[str, Any],
    expected_conditioning_sha256: str,
    expected_source_tar_sha256: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    expected_result_sha256 = _validate_sha256(
        expected_result_sha256, label="accepted suffix result SHA256"
    )
    expected_report_sha256 = _validate_sha256(
        expected_report_sha256, label="accepted suffix report SHA256"
    )
    if _sha256(result_path) != expected_result_sha256:
        raise ValueError("accepted suffix result SHA256 mismatch")
    if _sha256(report_path) != expected_report_sha256:
        raise ValueError("accepted suffix report SHA256 mismatch")
    report = json.loads(report_path.read_text())
    route = report.get("effective_route", {})
    required_route = {
        "device_type": "cuda",
        "attention_backend": "sdpa",
        "conv_backend": "none",
        "steps": STEPS,
        "one_model_load": True,
    }
    if report.get("status") != "done":
        raise ValueError("accepted suffix report is not done")
    for key, expected in required_route.items():
        if route.get(key) != expected:
            raise ValueError(f"accepted suffix route {key} must be {expected!r}")
    inputs = report.get("inputs", {})
    suffix_mlx = inputs.get("mlx", {})
    mlx_identity_labels = {
        "capture_sha256": "MLX capture",
        "run_report_sha256": "MLX run report",
        "conditioning_sha256": "MLX conditioning",
        "shape_flow_noise_sample_sha256": "MLX shape-flow noise",
        "shape_slat_support_sample_sha256": "MLX support sample",
    }
    for key, label in mlx_identity_labels.items():
        expected = expected_mlx_identity.get(key)
        if not expected or suffix_mlx.get(key) != expected:
            raise ValueError(f"accepted suffix {label} identity mismatch")
    if inputs.get("conditioning_sha256") != expected_conditioning_sha256:
        raise ValueError("accepted suffix conditioning identity mismatch")
    if inputs.get("source_tar_sha256") != expected_source_tar_sha256:
        raise ValueError("accepted suffix source tar identity mismatch")
    suffix_source = inputs.get("accepted_source", {})
    for key, label in (
        ("baseline_sha256", "source baseline"),
        ("report_sha256", "source report"),
    ):
        expected = expected_source_identity.get(key)
        if not expected or suffix_source.get(key) != expected:
            raise ValueError(f"accepted suffix {label} identity mismatch")
    arrays: dict[str, np.ndarray] = {}
    with np.load(result_path, allow_pickle=False) as archive:
        required = {
            "coords",
            "accepted_source_anchor_shape_slat",
            "mlx_anchor_shape_slat",
            "switch_0_shape_slat",
            "switch_1_shape_slat",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"accepted suffix result is missing arrays: {missing}")
        for name in required:
            arrays[name] = np.asarray(archive[name])
    if arrays["coords"].dtype != np.int32 or not np.array_equal(arrays["coords"], coords):
        raise ValueError("accepted suffix coordinates differ from MLX trajectory")
    shape = source_anchor.shape
    for name in required - {"coords"}:
        value = arrays[name]
        if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"accepted suffix array {name} has invalid shape, dtype, or values")
    if not np.array_equal(arrays["accepted_source_anchor_shape_slat"], source_anchor):
        raise ValueError("accepted suffix source anchor differs from accepted source baseline")
    if not np.array_equal(arrays["switch_0_shape_slat"], source_anchor):
        raise ValueError("accepted suffix switch-0 does not equal the source anchor")
    points = report.get("points", [])
    if [point.get("switch_step") for point in points] != list(range(STEPS + 1)):
        raise ValueError("accepted suffix report does not contain the complete switch ladder")
    return arrays, {
        "result_sha256": expected_result_sha256,
        "report_sha256": expected_report_sha256,
        "route": route,
        "switch_1_sha256": _array_sha256(arrays["switch_1_shape_slat"]),
    }


def validate_result_manifest(payload: dict[str, Any]) -> None:
    if payload.get("status") != "done":
        raise ValueError("transition-0 result is not done")
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
            raise ValueError(f"transition-0 route {key} must be {expected!r}")
    candidates = payload.get("candidates", [])
    if [candidate.get("name") for candidate in candidates] != list(
        EXPECTED_CANDIDATE_NAMES
    ):
        raise ValueError("transition-0 candidates are incomplete or out of order")
    if len({candidate.get("output_key") for candidate in candidates}) != len(candidates):
        raise ValueError("transition-0 candidate output keys are not distinct")
    for candidate in candidates:
        if candidate.get("source_step_indices") != list(range(1, STEPS)):
            raise ValueError(f"candidate {candidate.get('name')} has wrong source steps")
        if candidate.get("source_step_count") != STEPS - 1:
            raise ValueError(f"candidate {candidate.get('name')} has wrong step count")
        if len(candidate.get("step_elapsed_seconds", [])) != STEPS - 1:
            raise ValueError(f"candidate {candidate.get('name')} has wrong timing count")
    control = candidates[0]
    source_metrics = control.get("vs_source_anchor", {})
    if not source_metrics.get("exact") or source_metrics.get("nonzero") != 0:
        raise ValueError("source-native control does not exactly reproduce source anchor")
    timing = payload.get("timing", {})
    expected_steps = len(EXPECTED_CANDIDATE_NAMES) * (STEPS - 1)
    expected_timing = {
        "source_steps_completed": expected_steps,
        "source_steps_requested": expected_steps,
        "candidates_completed": len(EXPECTED_CANDIDATE_NAMES),
        "candidates_requested": len(EXPECTED_CANDIDATE_NAMES),
    }
    for key, expected in expected_timing.items():
        if timing.get(key) != expected:
            raise ValueError(f"transition-0 timing {key} must be {expected}")


def _required_matrix_array_keys(candidates: list[dict[str, Any]]) -> set[str]:
    return {
        "coords",
        "source_anchor_shape_slat",
        "mlx_anchor_shape_slat",
        "accepted_switch_1_shape_slat",
        "source_transition0_pred_pos",
        "source_transition0_pred_neg",
        *(f"mlx_transition0_{name}" for name in MLX_COMPONENT_NAMES),
        *(
            f"candidate_{index}_transition0_sample_next"
            for index in range(len(EXPECTED_CANDIDATE_NAMES))
        ),
        *(str(candidate["output_key"]) for candidate in candidates),
    }


def build_artifact_metadata(
    report: dict[str, Any], arrays: dict[str, np.ndarray]
) -> dict[str, Any]:
    array_manifest = {
        name: {
            "dtype": str(np.asarray(value).dtype),
            "shape": [int(dimension) for dimension in np.asarray(value).shape],
            "sha256": _array_sha256(np.asarray(value)),
        }
        for name, value in sorted(arrays.items())
    }
    return {
        "schema": ARTIFACT_SCHEMA,
        "artifact_status": "computed_pending_serialization",
        "external_report_required": True,
        "effective_route": report["effective_route"],
        "inputs": report["inputs"],
        "candidate_specs": report["candidate_specs"],
        "candidates": report["candidates"],
        "anchors": report["anchors"],
        "arrays": array_manifest,
    }


def validate_saved_artifact(
    path: Path, *, candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"coords", "metadata_json"} | {
            str(candidate["output_key"]) for candidate in candidates
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"saved transition-0 artifact is missing arrays: {missing}")
        raw_metadata = np.asarray(archive["metadata_json"])
        if raw_metadata.shape != () or raw_metadata.dtype.kind not in {"U", "S"}:
            raise ValueError(
                "saved transition-0 artifact metadata_json must be a string scalar"
            )
        metadata = json.loads(str(raw_metadata.item()))
        if metadata.get("schema") != ARTIFACT_SCHEMA:
            raise ValueError("saved transition-0 artifact metadata schema is invalid")
        if metadata.get("artifact_status") != "computed_pending_serialization":
            raise ValueError("saved transition-0 artifact metadata status is invalid")
        if metadata.get("external_report_required") is not True:
            raise ValueError(
                "saved transition-0 artifact metadata must require the external report"
            )
        metadata_candidates = metadata.get("candidates", [])
        if [candidate.get("name") for candidate in metadata_candidates] != list(
            EXPECTED_CANDIDATE_NAMES
        ):
            raise ValueError("saved transition-0 metadata omits candidates")
        if metadata_candidates != candidates:
            raise ValueError(
                "saved transition-0 metadata candidates differ from external report"
            )
        manifest = metadata.get("arrays")
        if not isinstance(manifest, dict):
            raise ValueError("saved transition-0 artifact lacks an array manifest")
        archive_array_keys = set(archive.files) - {"metadata_json"}
        if set(manifest) != archive_array_keys:
            missing_from_archive = sorted(set(manifest) - archive_array_keys)
            missing_from_manifest = sorted(archive_array_keys - set(manifest))
            raise ValueError(
                "saved transition-0 artifact array manifest mismatch: "
                f"missing arrays {missing_from_archive}, "
                f"unbound arrays {missing_from_manifest}"
            )
        required_arrays = _required_matrix_array_keys(candidates)
        missing_required = sorted(required_arrays - archive_array_keys)
        if missing_required:
            raise ValueError(
                f"saved transition-0 artifact is missing arrays: {missing_required}"
            )
        sample_shape = tuple(np.asarray(archive["source_anchor_shape_slat"]).shape)
        for name, expected in manifest.items():
            array = np.asarray(archive[name])
            if str(array.dtype) != expected.get("dtype"):
                raise ValueError(f"{name} dtype differs from manifest")
            if list(array.shape) != expected.get("shape"):
                raise ValueError(f"{name} shape differs from manifest")
            if _array_sha256(array) != expected.get("sha256"):
                raise ValueError(f"{name} digest differs from manifest")
            if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
                raise ValueError(f"{name} contains non-finite values")
            expected_dtype = np.int32 if name == "coords" else np.float32
            if array.dtype != expected_dtype:
                raise ValueError(f"{name} must have dtype {expected_dtype}")
            if name == "coords":
                if array.ndim != 2 or array.shape[1] != 4:
                    raise ValueError("coords must have shape [N,4]")
            elif array.shape != sample_shape:
                raise ValueError(f"{name} shape differs from matrix sample shape")
        for candidate in candidates:
            name = str(candidate["name"])
            key = str(candidate["output_key"])
            array = np.asarray(archive[key])
            if list(array.shape) != candidate.get("shape"):
                raise ValueError(f"{name} shape differs from manifest")
            digest = _array_sha256(array)
            if digest != candidate.get("sha256"):
                raise ValueError(f"{name} digest differs from manifest")
            if array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError(f"{name} has invalid dtype or values")
    return {
        "schema": f"{SCHEMA}.saved_artifact",
        "candidate_count": len(candidates),
        "metadata_schema": ARTIFACT_SCHEMA,
        "all_matrix_arrays_bound": True,
        "array_count": len(manifest),
    }


def _write_matrix_artifact(
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
                "last_trustworthy_phase": "all_transition0_candidates_saved",
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


def _candidate_start_states(
    *,
    torch: Any,
    sampler: Any,
    sample: Any,
    source_pos: Any,
    source_neg: Any,
    mlx_components: dict[str, np.ndarray],
    guidance_strength: float,
    guidance_rescale: float,
    guidance_interval: tuple[float, float],
    t: float,
    t_prev: float,
    candidate_names: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    from source_cuda_shape_block29_basin_map import _guided_prediction

    unknown = sorted(set(candidate_names) - set(EXPECTED_CANDIDATE_NAMES))
    if unknown:
        raise ValueError(f"unknown transition-0 candidates: {unknown}")
    if len(set(candidate_names)) != len(candidate_names):
        raise ValueError("transition-0 candidate selection contains duplicates")

    starts: dict[str, Any] = {}
    first_step: dict[str, Any] = {}
    pairs: dict[str, tuple[Any, Any]] = {
        "source-native-control": (source_pos, source_neg)
    }
    branch_interventions = {
        "mlx-pos-source-neg",
        "source-pos-mlx-neg",
        "mlx-both-source-post",
    }
    if branch_interventions.intersection(candidate_names):
        mlx_pos = sample.replace(
            torch.from_numpy(mlx_components["pred_pos"]).to(
                device=sample.device, dtype=sample.dtype
            )
        )
        mlx_neg = sample.replace(
            torch.from_numpy(mlx_components["pred_neg"]).to(
                device=sample.device, dtype=sample.dtype
            )
        )
        pairs.update(
            compose_candidate_pairs(
                source_pos=source_pos,
                source_neg=source_neg,
                mlx_pos=mlx_pos,
                mlx_neg=mlx_neg,
            )
        )
    for name in candidate_names:
        if name not in pairs:
            continue
        pred_pos, pred_neg = pairs[name]
        pred = _guided_prediction(
            sampler=sampler,
            sample=sample,
            pred_pos=pred_pos,
            pred_neg=pred_neg,
            t=t,
            guidance_strength=guidance_strength,
            guidance_rescale=guidance_rescale,
            guidance_interval=guidance_interval,
        )
        start = sample - (t - t_prev) * pred
        starts[name] = start
        first_step[name] = {
            "pred_final": pred.feats.detach().float().cpu().numpy().astype(np.float32),
            "sample_next": start.feats.detach().float().cpu().numpy().astype(np.float32),
        }
    if "mlx-final-source-euler" in candidate_names:
        mlx_final = sample.replace(
            torch.from_numpy(mlx_components["pred_final"]).to(
                device=sample.device, dtype=sample.dtype
            )
        )
        mlx_final_start = sample - (t - t_prev) * mlx_final
        starts["mlx-final-source-euler"] = mlx_final_start
        first_step["mlx-final-source-euler"] = {
            "pred_final": mlx_final.feats.detach().float().cpu().numpy().astype(
                np.float32
            ),
            "sample_next": mlx_final_start.feats.detach().float().cpu().numpy().astype(
                np.float32
            ),
        }
    if set(starts) != set(candidate_names):
        missing = sorted(set(candidate_names) - set(starts))
        raise ValueError(f"failed to construct transition-0 candidates: {missing}")
    return starts, first_step


def require_exact_source_control(
    *, candidate_index: int, metrics: dict[str, Any]
) -> None:
    if candidate_index != 0:
        return
    if not metrics.get("exact") or metrics.get("nonzero") != 0:
        raise ValueError(
            "source-native control does not exactly reproduce source anchor; "
            "aborting before intervention candidates"
        )


def run_control_gated_candidates(
    *,
    specs: list[dict[str, str]],
    build_starts: Callable[
        [tuple[str, ...]], tuple[dict[str, Any], dict[str, Any]]
    ],
    execute_candidate: Callable[
        [int, dict[str, str], Any, Any], dict[str, Any]
    ],
) -> list[dict[str, Any]]:
    names = tuple(spec["name"] for spec in specs)
    if names != EXPECTED_CANDIDATE_NAMES:
        raise ValueError("control-gated candidate specs are incomplete or out of order")

    control_name = names[0]
    control_starts, control_first_steps = build_starts((control_name,))
    control_result = execute_candidate(
        0,
        specs[0],
        control_starts[control_name],
        control_first_steps[control_name],
    )
    require_exact_source_control(
        candidate_index=0,
        metrics=control_result.get("vs_source_anchor", {}),
    )

    results = [control_result]
    intervention_names = names[1:]
    intervention_starts, intervention_first_steps = build_starts(intervention_names)
    for index, spec in enumerate(specs[1:], start=1):
        name = spec["name"]
        results.append(
            execute_candidate(
                index,
                spec,
                intervention_starts[name],
                intervention_first_steps[name],
            )
        )
    return results


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
    requested_route = {
        "route": "official-source-cuda-transition0-recoverability-matrix",
        "steps": STEPS,
        "candidate_specs": transition0_candidate_specs(),
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
        expected_digests = {
            label: _validate_sha256(value, label=f"{label} SHA256")
            for label, value in digest_args.items()
        }
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
            protected={
                "output report": args.output_json,
                "MLX shape-flow steps": args.mlx_shape_flow_steps,
                "MLX run report": args.mlx_run_report,
                "conditioning": args.conditioning,
                "accepted source baseline": args.accepted_source_baseline,
                "accepted source report": args.accepted_source_report,
                "accepted suffix result": args.accepted_suffix_result,
                "accepted suffix report": args.accepted_suffix_report,
                "source tar": args.source_tar,
            },
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
        cond_np, neg_cond_np = _load_conditioning(args.conditioning)
        report["inputs"] = {
            "expected_digests": expected_digests,
            "mlx": mlx_identity,
            "accepted_source": source_identity,
            "accepted_suffix": suffix_identity,
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

        phase = "transition0_matrix"
        phase_started = time.perf_counter()
        from source_cuda_shape_block29_basin_map import _flow_forward

        coords = torch.from_numpy(trajectory["coords"]).to(device=device, dtype=torch.int32)
        cond = torch.from_numpy(cond_np).to(device=device, dtype=torch.float32)
        neg_cond = torch.from_numpy(neg_cond_np).to(device=device, dtype=torch.float32)
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
        source_pos_np = source_pos.feats.detach().float().cpu().numpy().astype(np.float32)
        source_neg_np = source_neg.feats.detach().float().cpu().numpy().astype(np.float32)
        arrays: dict[str, np.ndarray] = {
            "coords": np.asarray(trajectory["coords"], dtype=np.int32),
            "source_anchor_shape_slat": source_anchor,
            "mlx_anchor_shape_slat": suffix_arrays["mlx_anchor_shape_slat"],
            "accepted_switch_1_shape_slat": suffix_arrays["switch_1_shape_slat"],
            "source_transition0_pred_pos": source_pos_np,
            "source_transition0_pred_neg": source_neg_np,
        }
        for name in MLX_COMPONENT_NAMES:
            arrays[f"mlx_transition0_{name}"] = mlx_components[name]
        direct_metrics = {
            "source_vs_mlx_pred_pos": _compare_arrays(
                source_pos_np, mlx_components["pred_pos"]
            ),
            "source_vs_mlx_pred_neg": _compare_arrays(
                source_neg_np, mlx_components["pred_neg"]
            ),
        }
        continuation_seconds = 0.0

        def build_starts(
            candidate_names: tuple[str, ...],
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            return _candidate_start_states(
                torch=torch,
                sampler=sampler,
                sample=sample,
                source_pos=source_pos,
                source_neg=source_neg,
                mlx_components=mlx_components,
                guidance_strength=float(sampler_params["guidance_strength"]),
                guidance_rescale=float(sampler_params["guidance_rescale"]),
                guidance_interval=tuple(
                    float(value) for value in sampler_params["guidance_interval"]
                ),
                t=t,
                t_prev=t_prev,
                candidate_names=candidate_names,
            )

        def execute_candidate(
            index: int,
            spec: dict[str, str],
            start: Any,
            candidate_first_step: dict[str, np.ndarray],
        ) -> dict[str, Any]:
            nonlocal continuation_seconds
            candidate_started = time.perf_counter()
            start_np = candidate_first_step["sample_next"]
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
            candidate_result = {
                **spec,
                "output_key": output_key,
                "shape": [int(value) for value in result.shape],
                "sha256": _array_sha256(result),
                "transition0_sample_next_sha256": _array_sha256(start_np),
                "transition0_vs_mlx_exact": _compare_arrays(
                    start_np, mlx_components["sample_next"]
                ),
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
            del result_raw, start
            torch.cuda.empty_cache()
            return candidate_result

        candidate_results = run_control_gated_candidates(
            specs=transition0_candidate_specs(),
            build_starts=build_starts,
            execute_candidate=execute_candidate,
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
            "comparison_class": "transition0-component-substitution-plus-source-cuda-suffix",
        }
        report.update(
            {
                "status": "done",
                "effective_route": effective_route,
                "candidate_specs": transition0_candidate_specs(),
                "candidates": candidate_results,
                "direct_transition0_metrics": direct_metrics,
                "anchors": {
                    "source_sha256": _array_sha256(source_anchor),
                    "mlx_sha256": _array_sha256(suffix_arrays["mlx_anchor_shape_slat"]),
                    "switch_1_sha256": _array_sha256(
                        suffix_arrays["switch_1_shape_slat"]
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
                    "t4_compute_seconds_through_matrix": time.perf_counter() - started,
                },
                "forbidden_inferences": [
                    "not final mesh, texture, winding, or GLB evidence",
                    "not proof that a branch is locally wrong without source-contract review",
                    "not a production implementation patch",
                    "not evidence outside the exact admitted hostile input and transition",
                ],
            }
        )
        validate_result_manifest(report)
        arrays["metadata_json"] = np.asarray(
            json.dumps(
                build_artifact_metadata(report, arrays),
                sort_keys=True,
                allow_nan=False,
            )
        )
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        last_trustworthy_phase = phase

        phase = "write_outputs"
        report["elapsed_seconds"] = time.perf_counter() - started
        return _write_matrix_artifact(
            args.output_npz, args.output_json, arrays, report
        )
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
