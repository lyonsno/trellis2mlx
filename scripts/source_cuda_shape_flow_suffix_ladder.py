#!/usr/bin/env python3
"""Continue every exact MLX shape-flow prefix through official source CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "trellis2mlx.source_cuda_shape_flow_suffix_ladder.v1"
STEPS = 8
SWITCH_STEPS = tuple(range(STEPS + 1))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _validate_file(path: Path, *, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{label} is blank: {path}")


def _invalidate_primary_output(path: Path, *, protected: dict[str, Path]) -> None:
    resolved = path.resolve()
    for label, protected_path in protected.items():
        if resolved == protected_path.resolve():
            raise ValueError(f"output NPZ collides with {label}: {path}")
    if path.exists():
        if not path.is_file():
            raise ValueError(f"output NPZ exists and is not a file: {path}")
        path.unlink()


def _schedule_pairs(steps: int, rescale_t: float) -> list[tuple[float, float]]:
    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    return [(float(t_seq[index]), float(t_seq[index + 1])) for index in range(steps)]


def suffix_step_indices(switch_step: int, *, steps: int = STEPS) -> list[int]:
    if switch_step < 0 or switch_step > steps:
        raise ValueError(f"switch step must be in [0, {steps}], got {switch_step}")
    return list(range(switch_step, steps))


def classify_anchor(source_distance: float, mlx_distance: float) -> str:
    if not math.isfinite(source_distance) or not math.isfinite(mlx_distance):
        raise ValueError("anchor distances must be finite")
    if source_distance < mlx_distance:
        return "source"
    if mlx_distance < source_distance:
        return "mlx"
    return "equidistant"


def _compare_arrays(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if left.shape != right.shape:
        return {
            "shape_match": False,
            "left_shape": [int(value) for value in left.shape],
            "right_shape": [int(value) for value in right.shape],
        }
    diff = np.abs(left - right)
    return {
        "shape_match": True,
        "mean_abs": float(diff.mean()),
        "max_abs": float(diff.max()),
        "nonzero": int(np.count_nonzero(diff)),
        "exact": bool(np.array_equal(left, right)),
    }


def _required_scalar(archive: Any, name: str, *, dtype: np.dtype) -> Any:
    if name not in archive.files:
        raise ValueError(f"MLX trajectory is missing {name}")
    value = np.asarray(archive[name])
    if value.shape != () or value.dtype != dtype:
        raise ValueError(f"MLX trajectory {name} must be a {dtype} scalar")
    return value.item()


def _validate_expected_modulation_identity(
    identity: dict[str, str] | None,
) -> None:
    if identity is None:
        return
    required = {
        "npz_sha256_effective",
        "report_sha256_effective",
        "source_checkpoint_sha256_effective",
    }
    if set(identity) != required:
        raise ValueError(
            "expected timestep modulation identity must contain exactly "
            f"{sorted(required)}"
        )
    for key in sorted(required):
        value = identity[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(
                f"expected timestep modulation identity {key} must be a "
                "lowercase 64-character SHA256"
            )


def _decode_modulation_checkpoint_identity(value: np.ndarray) -> dict[str, Any] | None:
    if value.shape != () or value.dtype.kind not in {"U", "S"}:
        raise ValueError(
            "MLX trajectory shape_timestep_modulation_lut_json must be a "
            "string scalar"
        )
    text = str(value.item())
    if not text:
        return None
    try:
        identity = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "MLX trajectory checkpoint timestep modulation identity is invalid JSON"
        ) from exc
    if not isinstance(identity, dict):
        raise ValueError(
            "MLX trajectory checkpoint timestep modulation identity must be an object"
        )
    return identity


def load_mlx_trajectory(
    capture_path: Path,
    run_report_path: Path,
    conditioning_path: Path,
    *,
    expected_modulation_identity: dict[str, str] | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    capture_path = Path(capture_path)
    run_report_path = Path(run_report_path)
    conditioning_path = Path(conditioning_path)
    _validate_expected_modulation_identity(expected_modulation_identity)
    report = json.loads(run_report_path.read_text())
    if report.get("status") != "done":
        raise ValueError("MLX run report is not done")
    if report.get("last_trustworthy_phase") != "shape_flow_steps_validated":
        raise ValueError("MLX run report did not validate shape_flow_steps")
    if report.get("primary_output_status") != "written":
        raise ValueError("MLX run report does not admit a written primary output")
    route_identity = report.get("route_identity", {})
    if route_identity.get("requested_stop") != "shape_flow_steps":
        raise ValueError("MLX run report requested_stop is not shape_flow_steps")
    route = route_identity.get("route", {})
    required_route = {
        "family": "trellis2mlx/mlx",
        "backend": "mlx-metal",
        "attention_backend": "fast",
        "steps": STEPS,
        "cascade": False,
    }
    for key, expected in required_route.items():
        if route.get(key) != expected:
            raise ValueError(f"MLX route {key} must be {expected!r}")
    if route.get("shape_flow_block_injection_trace_path") is not None:
        raise ValueError("MLX trajectory unexpectedly requests direct trace injection")
    if route.get("shape_flow_block_injection_manifest_path") is not None:
        raise ValueError("MLX trajectory unexpectedly requests manifest injection")
    route_modulation_identity = route.get("shape_timestep_modulation_identity")
    validation = report.get("primary_output_validation", {})
    validation_sampler = validation.get("sampler", {})
    validated_modulation_identity = validation_sampler.get(
        "shape_timestep_modulation_route"
    )
    if expected_modulation_identity is None:
        if (
            route_modulation_identity is not None
            or validated_modulation_identity is not None
        ):
            raise ValueError(
                "MLX trajectory carries timestep modulation identity under "
                "explicit default mode"
            )
    else:
        if not isinstance(route_modulation_identity, dict):
            raise ValueError(
                "MLX route omits requested timestep modulation identity"
            )
        if route_modulation_identity.get("route_identity_evidence") is not True:
            raise ValueError(
                "MLX route timestep modulation identity omits "
                "route_identity_evidence=true"
            )
        canonical_identity = {
            "schema": "trellis2mlx.source_cuda_timestep_modulation_lut.v1",
            "route": "source-cuda-t4-canonical-shared-adaln-lut",
            "step_indices": list(range(STEPS)),
            "timestep_float32_bits": [
                "0x447a0000",
                "0x446ea2e9",
                "0x44610000",
                "0x44505555",
                "0x443b8000",
                "0x4420b6db",
                "0x43fa0000",
                "0x43960000",
            ],
            "modulation_shape": [STEPS, 9216],
        }
        for key, expected in canonical_identity.items():
            if route_modulation_identity.get(key) != expected:
                raise ValueError(
                    f"MLX route timestep modulation identity {key} is not "
                    "canonical"
                )
        for key, expected in expected_modulation_identity.items():
            if route_modulation_identity.get(key) != expected:
                raise ValueError(
                    f"MLX route timestep modulation identity {key} does not "
                    "match the admitted value"
                )
        route_fields = {
            "npz_sha256_effective": (
                "shape_timestep_modulation_lut_sha256_effective"
            ),
            "report_sha256_effective": (
                "shape_timestep_modulation_report_sha256_effective"
            ),
            "source_checkpoint_sha256_effective": (
                "shape_timestep_modulation_source_checkpoint_sha256"
            ),
        }
        for identity_key, route_key in route_fields.items():
            if route.get(route_key) != expected_modulation_identity[identity_key]:
                raise ValueError(
                    f"MLX route {route_key} does not match the admitted "
                    "timestep modulation identity"
                )
        if validated_modulation_identity != route_modulation_identity:
            raise ValueError(
                "MLX primary validation timestep modulation identity does not "
                "match the requested route"
            )
    conditioning_sha = _sha256(conditioning_path)
    if route.get("conditioning_sample_sha256") != conditioning_sha:
        raise ValueError("conditioning digest does not match admitted MLX route")
    capture_sha = _sha256(capture_path)
    if validation.get("sha256") != capture_sha:
        raise ValueError("MLX trajectory digest does not match run report")
    if validation.get("step_count") != STEPS:
        raise ValueError("MLX run report does not admit eight trajectory steps")

    required_arrays = {
        "noise",
        "sample_feats",
        "coords",
        "coords_3d",
        "sample_in",
        "pred_final",
        "pred_v_feats",
        "sample_next",
        "t",
        "t_prev",
        "guidance_interval",
        "shape_flow_block_injection_json",
    }
    arrays: dict[str, np.ndarray] = {}
    with np.load(capture_path, allow_pickle=False) as archive:
        missing = sorted(required_arrays - set(archive.files))
        if missing:
            raise ValueError(f"MLX trajectory is missing arrays: {missing}")
        for name in required_arrays:
            arrays[name] = np.asarray(archive[name])
        if "shape_timestep_modulation_lut_json" in archive.files:
            arrays["shape_timestep_modulation_lut_json"] = np.asarray(
                archive["shape_timestep_modulation_lut_json"]
            )
        scalar_values = {
            "steps": _required_scalar(archive, "steps", dtype=np.dtype(np.int32)),
            "guidance_strength": _required_scalar(
                archive, "guidance_strength", dtype=np.dtype(np.float32)
            ),
            "guidance_rescale": _required_scalar(
                archive, "guidance_rescale", dtype=np.dtype(np.float32)
            ),
            "rescale_t": _required_scalar(
                archive, "rescale_t", dtype=np.dtype(np.float32)
            ),
            "sigma_min": _required_scalar(
                archive, "sigma_min", dtype=np.dtype(np.float32)
            ),
        }
    if scalar_values != {
        "steps": 8,
        "guidance_strength": np.float32(7.5),
        "guidance_rescale": np.float32(0.5),
        "rescale_t": np.float32(3.0),
        "sigma_min": np.float32(1e-5),
    }:
        raise ValueError(f"unsupported MLX sampler scalars: {scalar_values}")
    guidance_interval = arrays["guidance_interval"]
    if guidance_interval.dtype != np.float32 or not np.array_equal(
        guidance_interval, np.asarray([0.6, 1.0], dtype=np.float32)
    ):
        raise ValueError("unsupported MLX guidance interval")
    injection = arrays["shape_flow_block_injection_json"]
    if injection.shape != () or injection.dtype.kind not in {"U", "S"} or str(injection.item()):
        raise ValueError("MLX trajectory carries unexpected injection identity")
    checkpoint_modulation_identity = (
        _decode_modulation_checkpoint_identity(
            arrays["shape_timestep_modulation_lut_json"]
        )
        if "shape_timestep_modulation_lut_json" in arrays
        else None
    )
    if checkpoint_modulation_identity != route_modulation_identity:
        raise ValueError(
            "MLX trajectory checkpoint timestep modulation identity does not "
            "match the requested route"
        )

    sample_in = arrays["sample_in"]
    pred_final = arrays["pred_final"]
    sample_next = arrays["sample_next"]
    noise = arrays["noise"]
    coords = arrays["coords"]
    if sample_in.dtype != np.float32 or sample_in.ndim != 3 or sample_in.shape[0] != STEPS:
        raise ValueError(f"MLX sample_in must have shape [8,N,C] float32, got {sample_in.shape}")
    for name in ("pred_final", "pred_v_feats", "sample_next"):
        if arrays[name].dtype != np.float32 or arrays[name].shape != sample_in.shape:
            raise ValueError(f"MLX trajectory {name} does not match sample_in")
    if noise.dtype != np.float32 or noise.shape != sample_in.shape[1:]:
        raise ValueError("MLX trajectory noise does not match sample state")
    if coords.dtype != np.int32 or coords.shape != (sample_in.shape[1], 4):
        raise ValueError("MLX trajectory coords do not match sample state")
    for name in ("noise", "sample_in", "pred_final", "sample_next"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"MLX trajectory {name} contains non-finite values")
    if not np.array_equal(sample_in[0], noise):
        raise ValueError("MLX trajectory initial sample does not equal noise")
    if not np.array_equal(sample_in[1:], sample_next[:-1]):
        raise ValueError("MLX trajectory recurrence is not exact")
    if not np.array_equal(arrays["pred_v_feats"], pred_final):
        raise ValueError("MLX trajectory pred_v_feats differs from pred_final")

    expected_pairs = np.asarray(_schedule_pairs(STEPS, 3.0), dtype=np.float32)
    t = arrays["t"]
    t_prev = arrays["t_prev"]
    if t.dtype != np.float32 or t_prev.dtype != np.float32:
        raise ValueError("MLX trajectory schedule must be float32")
    if not np.array_equal(t, expected_pairs[:, 0]) or not np.array_equal(
        t_prev, expected_pairs[:, 1]
    ):
        raise ValueError("MLX trajectory schedule does not match source route")
    expected_next = sample_in - (t - t_prev)[:, None, None] * pred_final
    euler_residual = float(np.max(np.abs(expected_next - sample_next)))
    if euler_residual > 2e-5:
        raise ValueError(f"MLX trajectory Euler recurrence residual is {euler_residual}")

    identity = {
        "backend": route["backend"],
        "attention_backend": route["attention_backend"],
        "capture_sha256": capture_sha,
        "run_report_sha256": _sha256(run_report_path),
        "conditioning_sha256": conditioning_sha,
        "shape_flow_noise_sample_sha256": route.get("shape_flow_noise_sample_sha256"),
        "shape_slat_support_sample_sha256": route.get("shape_slat_support_sample_sha256"),
        "shape_timestep_modulation_identity": route_modulation_identity,
        "euler_transition_max_abs_residual": euler_residual,
    }
    return arrays, identity


def _load_conditioning(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if "cond" not in archive.files or "neg_cond" not in archive.files:
            raise ValueError("conditioning must contain cond and neg_cond")
        cond = np.asarray(archive["cond"], dtype=np.float32)
        neg_cond = np.asarray(archive["neg_cond"], dtype=np.float32)
    if not np.isfinite(cond).all() or not np.isfinite(neg_cond).all():
        raise ValueError("conditioning contains non-finite values")
    return cond, neg_cond


def _load_source_anchor(
    baseline_path: Path,
    report_path: Path,
    *,
    coords: np.ndarray,
    conditioning_sha256: str,
    noise_sha256: str,
    source_tar_sha256: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    report = json.loads(report_path.read_text())
    if report.get("status") != "done" or report.get("primary_output_status") != "written":
        raise ValueError("accepted source baseline report is not done with written output")
    route = report.get("effective_route", {})
    required_route = {
        "device_type": "cuda",
        "attention_backend": "sdpa",
        "conv_backend": "none",
        "steps": STEPS,
        "one_model_load": True,
    }
    for key, expected in required_route.items():
        if route.get(key) != expected:
            raise ValueError(f"accepted source baseline route {key} must be {expected!r}")
    expected_primary_sha = report.get("primary_output", {}).get("sha256")
    baseline_sha = _sha256(baseline_path)
    if expected_primary_sha != baseline_sha:
        raise ValueError("accepted source baseline digest does not match report")
    input_digests = report.get("inputs", {}).get("digests", {})
    expected_inputs = {
        "conditioning": conditioning_sha256,
        "shape-flow noise": noise_sha256,
        "source tar": source_tar_sha256,
    }
    for key, expected in expected_inputs.items():
        if input_digests.get(key) != expected:
            raise ValueError(f"accepted source baseline {key} digest mismatch")
    with np.load(baseline_path, allow_pickle=False) as archive:
        if "coords" not in archive.files or "source_control_shape_slat" not in archive.files:
            raise ValueError("accepted source baseline lacks coords or source control")
        baseline_coords = np.asarray(archive["coords"], dtype=np.int32)
        anchor = np.asarray(archive["source_control_shape_slat"], dtype=np.float32)
    if not np.array_equal(baseline_coords, coords):
        raise ValueError("accepted source baseline coordinates differ from MLX trajectory")
    if anchor.ndim != 2 or anchor.shape[0] != coords.shape[0] or not np.isfinite(anchor).all():
        raise ValueError(f"accepted source anchor has invalid shape or values: {anchor.shape}")
    return anchor, {
        "baseline_sha256": baseline_sha,
        "report_sha256": _sha256(report_path),
        "effective_route": route,
    }


def _run_suffix(
    *,
    torch: Any,
    flow_model: Any,
    sampler: Any,
    coords: Any,
    start_feats: Any,
    cond: Any,
    neg_cond: Any,
    params: dict[str, Any],
    switch_step: int,
    capture_steps: list[dict[str, np.ndarray]] | None = None,
) -> tuple[Any, list[float]]:
    from source_cuda_shape_block29_basin_map import (
        _flow_forward,
        _guidance_capture_to_numpy,
        _guided_prediction,
    )
    from trellis2.modules.sparse import SparseTensor

    sample = SparseTensor(feats=start_feats.clone(), coords=coords)
    step_timings: list[float] = []
    guidance_interval = tuple(float(value) for value in params["guidance_interval"])
    schedule = _schedule_pairs(int(params["steps"]), float(params["rescale_t"]))
    for step_index in suffix_step_indices(switch_step, steps=len(schedule)):
        step_started = time.perf_counter()
        t, t_prev = schedule[step_index]
        sample_in = sample
        t_model = torch.tensor(
            [1000.0 * t] * sample.shape[0], device=sample.device, dtype=torch.float32
        )
        pred_pos = _flow_forward(
            torch, flow_model, sample, t_model, cond, branch="pos", targets=None
        )
        pred_neg = _flow_forward(
            torch, flow_model, sample, t_model, neg_cond, branch="neg", targets=None
        )
        guidance_capture: dict[str, Any] | None = (
            {} if capture_steps is not None else None
        )
        pred = _guided_prediction(
            sampler=sampler,
            sample=sample,
            pred_pos=pred_pos,
            pred_neg=pred_neg,
            t=t,
            guidance_strength=float(params["guidance_strength"]),
            guidance_rescale=float(params["guidance_rescale"]),
            guidance_interval=guidance_interval,
            capture=guidance_capture,
        )
        sample_next = sample - (t - t_prev) * pred
        if capture_steps is not None:
            captured_step = {
                "sample_in": sample_in.feats.detach().float().cpu().numpy().astype(
                    np.float32
                ),
                "pred_pos": pred_pos.feats.detach().float().cpu().numpy().astype(
                    np.float32
                ),
                "pred_neg": pred_neg.feats.detach().float().cpu().numpy().astype(
                    np.float32
                ),
                "pred_final": pred.feats.detach().float().cpu().numpy().astype(
                    np.float32
                ),
                "sample_next": sample_next.feats.detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32),
                "t": np.asarray(t, dtype=np.float32),
                "t_prev": np.asarray(t_prev, dtype=np.float32),
            }
            if guidance_capture is None:
                raise AssertionError("guidance capture was not initialized")
            captured_step.update(_guidance_capture_to_numpy(guidance_capture))
            capture_steps.append(captured_step)
        sample = sample_next
        torch.cuda.synchronize()
        step_timings.append(time.perf_counter() - step_started)
    return sample, step_timings


def validate_result_manifest(payload: dict[str, Any]) -> None:
    if payload.get("status") != "done":
        raise ValueError("suffix result is not done")
    route = payload.get("effective_route", {})
    required_route = {
        "device_type": "cuda",
        "attention_backend": "sdpa",
        "conv_backend": "none",
        "steps": STEPS,
        "one_model_load": True,
        "switch_steps": list(SWITCH_STEPS),
    }
    for key, expected in required_route.items():
        if route.get(key) != expected:
            raise ValueError(f"effective route {key} must be {expected!r}")
    points = payload.get("points", [])
    observed = [point.get("switch_step") for point in points]
    if observed != list(SWITCH_STEPS):
        raise ValueError(f"result switch points differ: {observed}")
    for point in points:
        switch_step = point["switch_step"]
        canonical_key = f"switch_{switch_step}_shape_slat"
        if point.get("output_key") != canonical_key:
            raise ValueError(
                f"switch {switch_step} canonical output key must be {canonical_key!r}"
            )
        expected_indices = suffix_step_indices(switch_step)
        if point.get("source_step_indices") != expected_indices:
            raise ValueError(f"switch {switch_step} source step indices are inconsistent")
        if point.get("source_step_count") != len(expected_indices):
            raise ValueError(f"switch {switch_step} source step count is inconsistent")
        if len(point.get("step_elapsed_seconds", [])) != len(expected_indices):
            raise ValueError(f"switch {switch_step} step timing count is inconsistent")
    timing = payload.get("timing", {})
    expected_source_steps = sum(range(1, STEPS + 1))
    expected_timing = {
        "source_steps_completed": expected_source_steps,
        "source_steps_requested": expected_source_steps,
        "switch_points_completed": len(SWITCH_STEPS),
        "switch_points_requested": len(SWITCH_STEPS),
    }
    for key, expected in expected_timing.items():
        if timing.get(key) != expected:
            raise ValueError(f"timing {key} must be {expected}")
    source_boundary = points[0].get("vs_source_anchor", {})
    if (
        source_boundary.get("exact") is not True
        or source_boundary.get("max_abs") != 0.0
        or source_boundary.get("nonzero") != 0
    ):
        raise ValueError("switch zero is not exact with accepted source anchor")
    mlx_boundary = points[-1].get("vs_mlx_anchor", {})
    if (
        mlx_boundary.get("exact") is not True
        or mlx_boundary.get("max_abs") != 0.0
        or mlx_boundary.get("nonzero") != 0
    ):
        raise ValueError("switch eight is not exact with captured MLX anchor")


def _artifact_metadata(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "trellis2mlx.source_cuda_shape_flow_suffix_ladder.artifact.v1",
        "artifact_status": "computed_pending_serialization",
        "external_report_required": True,
        "effective_route": report["effective_route"],
        "inputs": report.get("inputs", {}),
        "points": report["points"],
        "pairwise": report.get("pairwise", {}),
        "timing": report["timing"],
        "forbidden_inferences": report.get("forbidden_inferences", []),
    }


def validate_saved_artifact(
    path: Path, *, points: list[dict[str, Any]]
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"coords", "metadata_json"} | {
            str(point["output_key"]) for point in points
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"saved suffix artifact is missing arrays: {missing}")
        raw_metadata = np.asarray(archive["metadata_json"])
        if raw_metadata.shape != () or raw_metadata.dtype.kind not in {"U", "S"}:
            raise ValueError("saved suffix artifact metadata_json must be a string scalar")
        metadata = json.loads(str(raw_metadata.item()))
        expected_schema = "trellis2mlx.source_cuda_shape_flow_suffix_ladder.artifact.v1"
        if metadata.get("schema") != expected_schema:
            raise ValueError("saved suffix artifact metadata schema is invalid")
        if metadata.get("artifact_status") != "computed_pending_serialization":
            raise ValueError("saved suffix artifact metadata status is invalid")
        if metadata.get("external_report_required") is not True:
            raise ValueError("saved suffix artifact metadata must require the external report")
        metadata_points = metadata.get("points", [])
        if [point.get("switch_step") for point in metadata_points] != list(SWITCH_STEPS):
            raise ValueError("saved suffix artifact metadata omits switch points")
        for point in points:
            switch_step = int(point["switch_step"])
            key = str(point["output_key"])
            array = np.asarray(archive[key])
            if list(array.shape) != point.get("shape"):
                raise ValueError(f"switch {switch_step} shape differs from manifest")
            digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
            if digest != point.get("sha256"):
                raise ValueError(f"switch {switch_step} digest differs from manifest")
            if array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError(f"switch {switch_step} has invalid dtype or values")
    return {
        "schema": "trellis2mlx.source_cuda_shape_flow_suffix_ladder.saved_artifact.v1",
        "switch_count": len(points),
        "metadata_schema": expected_schema,
        "point_arrays_bound": True,
    }


def _write_primary_artifact(
    output_path: Path,
    report_path: Path,
    arrays: dict[str, np.ndarray],
    report: dict[str, Any],
) -> int:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **arrays)
        validation = validate_saved_artifact(output_path, points=report.get("points", []))
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
                "last_trustworthy_phase": "all_suffixes_and_exact_boundaries_saved",
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
    parser.add_argument("--mlx-run-report", required=True, type=Path)
    parser.add_argument(
        "--mlx-timestep-modulation-route",
        choices=("default", "source-cuda-lut"),
    )
    parser.add_argument("--expected-modulation-lut-sha256")
    parser.add_argument("--expected-modulation-report-sha256")
    parser.add_argument("--expected-modulation-source-checkpoint-sha256")
    parser.add_argument("--conditioning", required=True, type=Path)
    parser.add_argument("--accepted-source-baseline", required=True, type=Path)
    parser.add_argument("--accepted-source-report", required=True, type=Path)
    parser.add_argument("--source-tar", required=True, type=Path)
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
    requested_route = {
        "route": "official-source-cuda-shape-flow-suffix-ladder-from-exact-mlx-prefixes",
        "steps": STEPS,
        "switch_steps": list(SWITCH_STEPS),
        "attention_backend": args.sparse_attn_backend,
        "conv_backend": args.sparse_conv_backend,
        "mlx_timestep_modulation_route": (
            args.mlx_timestep_modulation_route
        ),
        "expected_modulation_identity": expected_modulation_identity,
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "requested_route": requested_route,
        "effective_route": "not-established",
        "primary_output_status": "missing",
        "failure_phase": None,
        "last_trustworthy_phase": last_trustworthy_phase,
        "phase_timings": {},
    }
    primary_written_this_run = False
    try:
        phase = "request_validation"
        phase_started = time.perf_counter()
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
        try:
            _invalidate_primary_output(
                args.output_npz,
                protected={
                    "output report": args.output_json,
                    "MLX shape-flow steps": args.mlx_shape_flow_steps,
                    "MLX run report": args.mlx_run_report,
                    "conditioning": args.conditioning,
                    "accepted source baseline": args.accepted_source_baseline,
                    "accepted source report": args.accepted_source_report,
                    "source tar": args.source_tar,
                },
            )
        except ValueError as exc:
            if "collides with" in str(exc) or "not a file" in str(exc):
                report["primary_output_status"] = "not_owned_due_to_path_collision"
            raise
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        last_trustworthy_phase = phase

        phase = "input_validation"
        phase_started = time.perf_counter()
        for path, label in (
            (args.mlx_shape_flow_steps, "MLX shape-flow steps"),
            (args.mlx_run_report, "MLX run report"),
            (args.conditioning, "conditioning"),
            (args.accepted_source_baseline, "accepted source baseline"),
            (args.accepted_source_report, "accepted source report"),
            (args.source_tar, "source tar"),
        ):
            _validate_file(path, label=label)
        trajectory, mlx_identity = load_mlx_trajectory(
            args.mlx_shape_flow_steps,
            args.mlx_run_report,
            args.conditioning,
            expected_modulation_identity=expected_modulation_identity,
        )
        source_tar_sha = _sha256(args.source_tar)
        source_anchor, source_identity = _load_source_anchor(
            args.accepted_source_baseline,
            args.accepted_source_report,
            coords=trajectory["coords"],
            conditioning_sha256=mlx_identity["conditioning_sha256"],
            noise_sha256=mlx_identity["shape_flow_noise_sample_sha256"],
            source_tar_sha256=source_tar_sha,
        )
        cond_np, neg_cond_np = _load_conditioning(args.conditioning)
        report["inputs"] = {
            "mlx": mlx_identity,
            "accepted_source": source_identity,
            "source_tar_sha256": source_tar_sha,
            "conditioning_sha256": _sha256(args.conditioning),
            "coords_shape": [int(value) for value in trajectory["coords"].shape],
            "sample_shape": [int(value) for value in trajectory["sample_in"].shape],
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
        expected_params = {
            "steps": 8,
            "guidance_strength": 7.5,
            "guidance_rescale": 0.5,
            "guidance_interval": [0.6, 1.0],
            "rescale_t": 3.0,
        }
        for key, expected in expected_params.items():
            if not np.array_equal(
                np.asarray(sampler_params[key], dtype=np.float32),
                np.asarray(expected, dtype=np.float32),
            ):
                raise ValueError(f"pipeline sampler {key} differs from captured route")
        sampler = getattr(samplers, pipeline_args["shape_slat_sampler"]["name"])(
            **pipeline_args["shape_slat_sampler"]["args"]
        )
        report["pipeline_config"] = {
            "path": str(config_path),
            "model_repo": args.model_repo,
            "pipeline_config": args.pipeline_config,
            "sampler_name": pipeline_args["shape_slat_sampler"]["name"],
            "sampler_args": pipeline_args["shape_slat_sampler"]["args"],
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
        report["model"] = {
            "model_ref": model_ref,
            "parameter_count": int(
                sum(parameter.numel() for parameter in flow_model.parameters())
            ),
            "load_seconds": model_load_seconds,
            "training": bool(flow_model.training),
        }
        if flow_model.training:
            raise RuntimeError("source shape-flow model remained in training mode after eval")
        report["phase_timings"][phase] = model_load_seconds
        last_trustworthy_phase = phase

        phase = "suffix_continuation"
        phase_started = time.perf_counter()
        coords = torch.from_numpy(trajectory["coords"]).to(device=device, dtype=torch.int32)
        cond = torch.from_numpy(cond_np).to(device=device, dtype=torch.float32)
        neg_cond = torch.from_numpy(neg_cond_np).to(device=device, dtype=torch.float32)
        normalization_std_np = np.asarray(
            pipeline_args["shape_slat_normalization"]["std"], dtype=np.float32
        )[None]
        normalization_mean_np = np.asarray(
            pipeline_args["shape_slat_normalization"]["mean"], dtype=np.float32
        )[None]
        mlx_anchor_raw = np.asarray(trajectory["sample_next"][-1], dtype=np.float32)
        mlx_anchor = mlx_anchor_raw * normalization_std_np + normalization_mean_np
        arrays: dict[str, np.ndarray] = {
            "coords": trajectory["coords"],
            "accepted_source_anchor_shape_slat": source_anchor,
            "mlx_anchor_shape_slat": mlx_anchor,
            "switch_steps": np.asarray(SWITCH_STEPS, dtype=np.int32),
        }
        points: list[dict[str, Any]] = []
        continuation_seconds = 0.0
        source_steps_completed = 0
        for switch_step in SWITCH_STEPS:
            point_started = time.perf_counter()
            source_indices = suffix_step_indices(switch_step)
            if switch_step == STEPS:
                result_raw_np = mlx_anchor_raw.copy()
                step_timings: list[float] = []
            else:
                start_feats = torch.from_numpy(
                    np.asarray(trajectory["sample_in"][switch_step], dtype=np.float32)
                ).to(device=device, dtype=torch.float32)
                result_raw, step_timings = _run_suffix(
                    torch=torch,
                    flow_model=flow_model,
                    sampler=sampler,
                    coords=coords,
                    start_feats=start_feats,
                    cond=cond,
                    neg_cond=neg_cond,
                    params=sampler_params,
                    switch_step=switch_step,
                )
                result_raw_np = (
                    result_raw.feats.detach().float().cpu().numpy().astype(np.float32)
                )
                del result_raw, start_feats
            result_np = result_raw_np * normalization_std_np + normalization_mean_np
            key = f"switch_{switch_step}_shape_slat"
            arrays[key] = result_np
            vs_source = _compare_arrays(result_np, source_anchor)
            vs_mlx = _compare_arrays(result_np, mlx_anchor)
            elapsed = time.perf_counter() - point_started
            continuation_seconds += elapsed
            source_steps_completed += len(source_indices)
            point = {
                "switch_step": switch_step,
                "source_step_indices": source_indices,
                "source_step_count": len(source_indices),
                "output_key": key,
                "shape": [int(value) for value in result_np.shape],
                "sha256": hashlib.sha256(result_np.tobytes()).hexdigest(),
                "elapsed_seconds": elapsed,
                "step_elapsed_seconds": step_timings,
                "vs_source_anchor": vs_source,
                "vs_mlx_anchor": vs_mlx,
                "nearest_anchor": classify_anchor(
                    float(vs_source["mean_abs"]), float(vs_mlx["mean_abs"])
                ),
            }
            points.append(point)
            report["points"] = points
            report["timing"] = {
                "model_load_seconds": model_load_seconds,
                "suffix_continuation_seconds": continuation_seconds,
                "source_steps_completed": source_steps_completed,
                "source_steps_requested": sum(range(1, STEPS + 1)),
                "switch_points_completed": len(points),
                "switch_points_requested": len(SWITCH_STEPS),
                "t4_compute_seconds_through_continuation": time.perf_counter() - started,
            }
            torch.cuda.empty_cache()

        pairwise: dict[str, Any] = {}
        for left in SWITCH_STEPS:
            for right in SWITCH_STEPS:
                pairwise[f"{left}:{right}"] = _compare_arrays(
                    arrays[f"switch_{left}_shape_slat"],
                    arrays[f"switch_{right}_shape_slat"],
                )
        report["pairwise"] = pairwise
        effective_route = {
            "route": requested_route["route"],
            "device_type": next(flow_model.parameters()).device.type,
            "cuda_device": torch.cuda.get_device_name(0),
            "attention_backend": getattr(sparse_config, "ATTN", None),
            "conv_backend": getattr(sparse_config, "CONV", None),
            "steps": STEPS,
            "switch_steps": list(SWITCH_STEPS),
            "one_model_load": True,
            "model_ref": model_ref,
            "comparison_class": "exact-mlx-prefix-plus-source-cuda-suffix",
        }
        report.update(
            {
                "status": "done",
                "effective_route": effective_route,
                "points": points,
                "timing": {
                    "model_load_seconds": model_load_seconds,
                    "suffix_continuation_seconds": continuation_seconds,
                    "source_steps_completed": source_steps_completed,
                    "source_steps_requested": sum(range(1, STEPS + 1)),
                    "switch_points_completed": len(points),
                    "switch_points_requested": len(SWITCH_STEPS),
                    "t4_compute_seconds_through_continuation": time.perf_counter() - started,
                },
                "forbidden_inferences": [
                    "not final mesh, texture, winding, or GLB evidence",
                    "not a learned-manifold metric",
                    "not proof of a visual basin until quotient-distinct endpoints are decoded",
                    "not a production implementation patch",
                ],
            }
        )
        validate_result_manifest(report)
        arrays["metadata_json"] = np.asarray(
            json.dumps(_artifact_metadata(report), sort_keys=True, allow_nan=False)
        )
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        last_trustworthy_phase = phase

        phase = "write_outputs"
        phase_started = time.perf_counter()
        report["elapsed_seconds"] = time.perf_counter() - started
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        return _write_primary_artifact(
            args.output_npz, args.output_json, arrays, report
        )
    except Exception as exc:
        if report.get("primary_output_status") != "not_owned_due_to_path_collision":
            report["primary_output_status"] = (
                "written"
                if primary_written_this_run and args.output_npz.exists()
                else "missing"
            )
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
            _write_json(args.output_json, report)
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
