#!/usr/bin/env python3
"""Summarize exact block-operation replays into guided endpoint evidence."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


class ReplayContractError(ValueError):
    """Raised when a replay cannot support the requested causal comparison."""


INTERVENTION_DEPTH = {
    "prefix28": 0,
    "attention_raw": 1,
    "after_self": 2,
    "cross_attention_raw": 3,
    "after_cross": 4,
    "after_mlp": 5,
    "source": 6,
}
ROUTE_VECTOR_FIELDS = (
    "effective_device_type",
    "effective_route",
    "conditioning_sha256",
    "shape_flow_noise_sample_sha256",
    "shape_slat_support_sample_sha256",
    "source_tar_sha256",
    "steps",
)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    path: Path
    expected_manifest_class: str | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize shape block operation replays")
    parser.add_argument("--source-trace", required=True, type=Path)
    parser.add_argument("--source-step", required=True, type=Path)
    parser.add_argument("--candidate", required=True, action="append")
    parser.add_argument("--expected-manifest", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expected = dict(_parse_assignment(value, "expected manifest") for value in args.expected_manifest)
    candidates = []
    for value in args.candidate:
        name, path = _parse_assignment(value, "candidate")
        candidates.append(CandidateSpec(name, Path(path), expected.get(name)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = summarize_replays(
            source_trace_path=args.source_trace,
            source_step_path=args.source_step,
            candidates=candidates,
        )
    except Exception as exc:
        failure = {
            "schema": "trellis2mlx.shape_block_operation_replays.failure.v1",
            "status": "failed",
            "failure_phase": "summarize_replays",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "source_trace": str(args.source_trace),
            "source_step": str(args.source_step),
            "candidates": [
                {
                    "name": candidate.name,
                    "path": str(candidate.path),
                    "expected_manifest_class": candidate.expected_manifest_class,
                }
                for candidate in candidates
            ],
        }
        args.output.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        raise
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "status": report["status"],
                "replay_count": len(report["replay_rows"]),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


def summarize_replays(
    *,
    source_trace_path: Path,
    source_step_path: Path,
    candidates: list[CandidateSpec],
) -> dict[str, Any]:
    if not candidates:
        raise ReplayContractError("at least one replay candidate is required")
    candidate_names = [candidate.name for candidate in candidates]
    if "source" in candidate_names or len(candidate_names) != len(set(candidate_names)):
        raise ReplayContractError(
            f"duplicate replay candidate names or reserved source name: {candidate_names}"
        )
    _require_file(source_trace_path, "source trace")
    _require_file(source_step_path, "source step")
    source_trace_identity = _file_identity(source_trace_path)
    source_step_identity = _file_identity(source_step_path)
    source_trace_sha256 = source_trace_identity["sha256"]

    with np.load(source_trace_path, allow_pickle=False) as source_trace, np.load(
        source_step_path, allow_pickle=False
    ) as source_step:
        source_coords = _require_array(source_trace, "coords")
        step_coords = _require_array(source_step, "coords")
        _require_exact(source_coords, step_coords, "source trace coords vs source step coords")
        sample = _squeeze(_require_array(source_step, "sample_feats"), "source sample_feats")
        source_pos = _squeeze(_require_array(source_trace, "pos_final_output"), "source pos")
        source_neg = _squeeze(_require_array(source_trace, "neg_final_output"), "source neg")
        source_inputs = {
            branch: _squeeze(
                _require_array(source_trace, f"{branch}_block29_input"),
                f"source {branch} block29 input",
            )
            for branch in ("pos", "neg")
        }
        source_route_vector = _source_route_vector(source_trace)
        _require_same_shape(sample, source_pos, "source sample vs pos output")
        _require_same_shape(sample, source_neg, "source sample vs neg output")

        t = _scalar(source_step, "t") if "t" in source_step else _scalar(source_trace, "t")
        t_prev = (
            _scalar(source_step, "t_prev")
            if "t_prev" in source_step
            else _scalar(source_trace, "t_prev")
        )
        steps = int(_scalar(source_step, "steps"))
        rescale_t = _scalar(source_step, "rescale_t")
        guidance_interval = _vector(source_step, "guidance_interval", length=2)
        schedule_t, schedule_t_prev = _schedule_coordinates(steps, rescale_t, step_index=0)
        if abs(t - schedule_t) > 1e-7 or abs(t_prev - schedule_t_prev) > 1e-7:
            raise ReplayContractError(
                "source schedule coordinates do not match steps/rescale_t: "
                f"observed ({t}, {t_prev}) vs derived ({schedule_t}, {schedule_t_prev})"
            )
        if int(source_route_vector["steps"]) != steps:
            raise ReplayContractError("source route steps do not match source schedule")
        source_block_index = int(_scalar(source_trace, "trace_block_index"))
        source_step_index = int(_scalar(source_trace, "shape_flow_trace_step_index"))
        if source_block_index != 29 or source_step_index != 0:
            raise ReplayContractError(
                "source trace route is "
                f"block{source_block_index}/step{source_step_index}, expected block29/step0"
            )
        _normalize_model_t(_scalar(source_trace, "t"), t, "source trace")
        source_trace_steps = int(_scalar(source_trace, "steps"))
        source_trace_rescale = _scalar(source_trace, "rescale_t")
        source_trace_t_prev = _scalar(source_trace, "t_prev")
        if (
            source_trace_steps != steps
            or abs(source_trace_rescale - rescale_t) > 1e-7
            or abs(source_trace_t_prev - t_prev) > 1e-7
        ):
            raise ReplayContractError(
                "source trace schedule does not match source step: "
                f"steps={source_trace_steps}, rescale_t={source_trace_rescale}, "
                f"t_prev={source_trace_t_prev}"
            )
        guidance_strength = _scalar(
            source_step,
            "guidance_strength",
            fallback=_scalar(source_trace, "guidance_strength", fallback=7.5),
        )
        guidance_rescale = _scalar(
            source_step,
            "guidance_rescale",
            fallback=_scalar(source_trace, "guidance_rescale", fallback=0.5),
        )
        source_reconstructed = _guided_pred(
            sample,
            source_pos,
            source_neg,
            t=t,
            guidance_strength=guidance_strength,
            guidance_rescale=guidance_rescale,
        )
        source_authoritative = _squeeze(
            _require_array(source_step, "pred_final"), "source pred_final"
        )
        source_reconstruction = _delta_metrics(source_authoritative, source_reconstructed)
        if source_reconstruction["max_abs"] > 1e-5:
            raise ReplayContractError(
                "source guided reconstruction exceeds 1e-5 max_abs: "
                f"{source_reconstruction['max_abs']}"
            )

        source_sample_next = _squeeze(
            _require_array(source_step, "sample_next"), "source sample_next"
        )
        source_euler = sample - (t - t_prev) * source_authoritative
        source_euler_delta = _delta_metrics(source_sample_next, source_euler)
        if source_euler_delta["max_abs"] > 1e-5:
            raise ReplayContractError(
                "source sample_next does not match declared Euler schedule: "
                f"max_abs={source_euler_delta['max_abs']}"
            )
        replay_rows = []
        for candidate in candidates:
            replay_rows.append(
                _summarize_candidate(
                    candidate,
                    source_coords=source_coords,
                    source_pos=source_pos,
                    source_neg=source_neg,
                    source_pred=source_authoritative,
                    source_sample_next=source_sample_next,
                    source_inputs=source_inputs,
                    source_route_vector=source_route_vector,
                    source_trace_sha256=source_trace_sha256,
                    sample=sample,
                    t=t,
                    t_prev=t_prev,
                    steps=steps,
                    rescale_t=rescale_t,
                    guidance_interval=guidance_interval,
                    guidance_strength=guidance_strength,
                    guidance_rescale=guidance_rescale,
                )
            )

    replay_rows.append(
        {
            "name": "source",
            "artifact": str(source_trace_path),
            "artifact_sha256": source_trace_sha256,
            "manifest_identity": {"route_vector": source_route_vector},
            "intervention_stage": "source",
            "intervention_depth": INTERVENTION_DEPTH["source"],
            "intervention_topology": "main_chain",
            "pos_final_output_source_mean_abs": 0.0,
            "neg_final_output_source_mean_abs": 0.0,
            "pred_final_source_mean_abs": 0.0,
            "pred_final_source_max_abs": 0.0,
            "sample_next_source_mean_abs": 0.0,
            "sample_next_source_max_abs": 0.0,
        }
    )
    replay_rows.sort(key=lambda row: row["intervention_depth"])
    depths = [row["intervention_depth"] for row in replay_rows]
    if len(depths) != len(set(depths)):
        raise ReplayContractError(f"replay candidates contain duplicate intervention depths: {depths}")
    main_chain_parent: str | None = None
    prefix_parents = [
        row["name"] for row in replay_rows if row["intervention_stage"] == "prefix28"
    ]
    if any(row["intervention_topology"] == "side_branch" for row in replay_rows):
        if len(prefix_parents) != 1:
            raise ReplayContractError(
                "operation replay side branch requires exactly one validated prefix28 parent; "
                f"found {prefix_parents}"
            )
    prefix_parent = prefix_parents[0] if prefix_parents else None
    for row in replay_rows:
        if row["intervention_topology"] == "main_chain":
            row["causal_parent"] = main_chain_parent
            main_chain_parent = row["name"]
        else:
            row["causal_parent"] = prefix_parent

    return {
        "schema": "trellis2mlx.shape_block_operation_replays.v1",
        "status": "done",
        "comparison_class": "exact_source_prefix28_then_block29_operation_replay",
        "source_trace": source_trace_identity,
        "source_step": source_step_identity,
        "effective_parameters": {
            "block_index": 29,
            "step_index": 0,
            "t": t,
            "t_prev": t_prev,
            "guidance_strength": guidance_strength,
            "guidance_rescale": guidance_rescale,
            "guidance_interval": guidance_interval.tolist(),
            "steps": steps,
            "rescale_t": rescale_t,
            "sigma_min": 1e-5,
        },
        "source_reconstruction": {
            "pred_final_mean_abs": source_reconstruction["mean_abs"],
            "pred_final_max_abs": source_reconstruction["max_abs"],
            "pred_final_nonzero": source_reconstruction["nonzero"],
            "sample_next_mean_abs": source_euler_delta["mean_abs"],
            "sample_next_max_abs": source_euler_delta["max_abs"],
        },
        "replay_rows": replay_rows,
    }


def _summarize_candidate(
    candidate: CandidateSpec,
    *,
    source_coords: np.ndarray,
    source_pos: np.ndarray,
    source_neg: np.ndarray,
    source_pred: np.ndarray,
    source_sample_next: np.ndarray,
    source_inputs: dict[str, np.ndarray],
    source_route_vector: dict[str, Any],
    source_trace_sha256: str,
    sample: np.ndarray,
    t: float,
    t_prev: float,
    steps: int,
    rescale_t: float,
    guidance_interval: np.ndarray,
    guidance_strength: float,
    guidance_rescale: float,
) -> dict[str, Any]:
    _require_file(candidate.path, f"candidate {candidate.name}")
    with np.load(candidate.path, allow_pickle=False) as trace:
        _require_exact(source_coords, _require_array(trace, "coords"), f"{candidate.name} coords")
        block_index = int(_scalar(trace, "trace_block_index"))
        step_index = int(_scalar(trace, "shape_flow_trace_step_index"))
        if block_index != 29 or step_index != 0:
            raise ReplayContractError(
                f"{candidate.name} route is block{block_index}/step{step_index}, expected block29/step0"
            )
        for branch in ("pos", "neg"):
            _require_exact(
                source_inputs[branch],
                _squeeze(
                    _require_array(trace, f"{branch}_block29_input"),
                    f"{candidate.name} {branch} block29 input",
                ),
                f"{candidate.name} block29 input {branch}",
            )

        intervention = _validate_intervention(
            trace,
            candidate_name=candidate.name,
            expected_manifest_class=candidate.expected_manifest_class,
            source_route_vector=source_route_vector,
            source_trace_sha256=source_trace_sha256,
        )
        recorded_model_t = _scalar(trace, "t")
        candidate_t = _normalize_model_t(recorded_model_t, t, candidate.name)
        candidate_steps = int(_scalar(trace, "steps"))
        candidate_rescale = _scalar(trace, "rescale_t")
        candidate_interval = _vector(trace, "guidance_interval", length=2)
        candidate_schedule_t, candidate_t_prev = _schedule_coordinates(
            candidate_steps, candidate_rescale, step_index=step_index
        )
        if (
            candidate_steps != steps
            or abs(candidate_rescale - rescale_t) > 1e-7
            or not np.allclose(candidate_interval, guidance_interval, rtol=0, atol=1e-7)
            or abs(candidate_schedule_t - t) > 1e-7
            or abs(candidate_t_prev - t_prev) > 1e-7
        ):
            raise ReplayContractError(
                f"{candidate.name} schedule does not match source: "
                f"steps={candidate_steps}, rescale_t={candidate_rescale}, "
                f"t/t_prev={candidate_schedule_t}/{candidate_t_prev}"
            )
        if not (candidate_interval[0] <= candidate_t <= candidate_interval[1]):
            raise ReplayContractError(f"{candidate.name} guidance interval excludes sampler t")
        candidate_guidance = _scalar(trace, "guidance_strength", fallback=guidance_strength)
        candidate_rescale = _scalar(trace, "guidance_rescale", fallback=guidance_rescale)
        if candidate_guidance != guidance_strength or candidate_rescale != guidance_rescale:
            raise ReplayContractError(f"{candidate.name} guidance parameters do not match source")

        pos = _squeeze(_require_array(trace, "pos_final_output"), f"{candidate.name} pos")
        neg = _squeeze(_require_array(trace, "neg_final_output"), f"{candidate.name} neg")
        pred = _guided_pred(
            sample,
            pos,
            neg,
            t=t,
            guidance_strength=guidance_strength,
            guidance_rescale=guidance_rescale,
        )
        sample_next = sample - (t - t_prev) * pred
        pos_delta = _delta_metrics(source_pos, pos)
        neg_delta = _delta_metrics(source_neg, neg)
        pred_delta = _delta_metrics(source_pred, pred)
        next_delta = _delta_metrics(source_sample_next, sample_next)
        return {
            "name": candidate.name,
            "artifact": str(candidate.path),
            "artifact_sha256": _sha256(candidate.path),
            "manifest_identity": intervention["identity"],
            "recorded_model_t": recorded_model_t,
            "normalized_sampler_t": candidate_t,
            "intervention_stage": intervention["stage"],
            "intervention_depth": intervention["depth"],
            "intervention_topology": intervention["topology"],
            "pos_final_output_source_mean_abs": pos_delta["mean_abs"],
            "pos_final_output_source_max_abs": pos_delta["max_abs"],
            "neg_final_output_source_mean_abs": neg_delta["mean_abs"],
            "neg_final_output_source_max_abs": neg_delta["max_abs"],
            "pred_final_source_mean_abs": pred_delta["mean_abs"],
            "pred_final_source_max_abs": pred_delta["max_abs"],
            "pred_final_source_rms": pred_delta["rms"],
            "pred_final_source_nonzero": pred_delta["nonzero"],
            "sample_next_source_mean_abs": next_delta["mean_abs"],
            "sample_next_source_max_abs": next_delta["max_abs"],
        }


def _guided_pred(
    sample: np.ndarray,
    pred_pos: np.ndarray,
    pred_neg: np.ndarray,
    *,
    t: float,
    guidance_strength: float,
    guidance_rescale: float,
    sigma_min: float = 1e-5,
) -> np.ndarray:
    sample = np.asarray(sample, dtype=np.float32)
    pred_pos = np.asarray(pred_pos, dtype=np.float32)
    pred_neg = np.asarray(pred_neg, dtype=np.float32)
    pred_cfg = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg
    scale = sigma_min + (1 - sigma_min) * t
    x0_pos = (1 - sigma_min) * sample - scale * pred_pos
    x0_cfg = (1 - sigma_min) * sample - scale * pred_cfg
    std_cfg = float(np.std(x0_cfg, dtype=np.float32))
    ratio = float(np.std(x0_pos, dtype=np.float32)) / std_cfg if std_cfg > 0 else 1.0
    x0_rescaled = x0_cfg * ratio
    x0 = guidance_rescale * x0_rescaled + (1 - guidance_rescale) * x0_cfg
    return np.asarray(((1 - sigma_min) * sample - x0) / scale, dtype=np.float32)


def _read_injection_evidence(trace: Any) -> dict[str, Any] | None:
    if "shape_flow_block_injection_json" not in trace:
        return None
    try:
        evidence = json.loads(str(np.asarray(trace["shape_flow_block_injection_json"]).item()))
    except (ValueError, json.JSONDecodeError) as exc:
        raise ReplayContractError(f"invalid shape_flow_block_injection_json: {exc}") from exc
    if not isinstance(evidence, dict):
        raise ReplayContractError("shape_flow_block_injection_json must decode to an object")
    if not evidence.get("route_identity_evidence"):
        raise ReplayContractError("injection evidence does not assert route_identity_evidence")
    return evidence


def _manifest_identity(trace: Any) -> dict[str, Any] | None:
    evidence = _read_injection_evidence(trace)
    if evidence is None:
        return None
    identity = evidence.get("manifest_identity")
    if isinstance(identity, dict):
        return {
            **identity,
            "manifest_sha256": evidence.get("manifest_sha256"),
            "sites": evidence.get("sites"),
        }
    required = ("comparison_class", "block_index", "step_index", "stage", "branch")
    missing = [name for name in required if name not in evidence]
    if missing:
        raise ReplayContractError(
            "injection evidence has neither manifest_identity nor a complete direct site: "
            + ", ".join(missing)
        )
    return {
        "schema": "trellis2mlx.shape_block_direct_injection.v1",
        **{name: evidence[name] for name in required},
        "trace_sha256": evidence.get("trace_sha256"),
    }


def _validate_intervention(
    trace: Any,
    *,
    candidate_name: str,
    expected_manifest_class: str | None,
    source_route_vector: dict[str, Any],
    source_trace_sha256: str,
) -> dict[str, Any]:
    evidence = _read_injection_evidence(trace)
    if evidence is None:
        raise ReplayContractError(f"{candidate_name} has missing intervention evidence")
    identity = evidence.get("manifest_identity")
    if isinstance(identity, dict):
        sites = evidence.get("sites")
        if not isinstance(sites, list) or len(sites) != 2:
            raise ReplayContractError(
                f"{candidate_name} manifest must contain prefix28 and one block29 intervention site"
            )
        manifest_sha = evidence.get("manifest_sha256")
        _require_sha256(manifest_sha, f"{candidate_name} manifest")
        _validate_site(
            sites[0], candidate_name=candidate_name, block_index=28, stage="after_mlp",
            source_route_vector=source_route_vector,
        )
        stage = str(sites[1].get("stage"))
        if stage not in {
            "attention_raw",
            "after_self",
            "cross_attention_raw",
            "after_cross",
            "after_mlp",
        }:
            raise ReplayContractError(
                f"{candidate_name} block29 intervention site has unsupported stage {stage!r}"
            )
        _validate_site(
            sites[1], candidate_name=candidate_name, block_index=29, stage=stage,
            source_route_vector=source_route_vector,
            expected_trace_sha256=source_trace_sha256,
        )
        observed_class = identity.get("comparison_class")
        expected_from_site = f"exact_source_cuda_prefix28_plus_block29_{stage}"
        if expected_manifest_class is None or observed_class != expected_manifest_class:
            raise ReplayContractError(
                f"{candidate_name} manifest class {observed_class!r} does not match "
                f"required {expected_manifest_class!r}"
            )
        if observed_class != expected_from_site:
            raise ReplayContractError(
                f"{candidate_name} block29 intervention site {stage!r} contradicts "
                f"manifest class {observed_class!r}"
            )
        return {
            "identity": {
                **identity,
                "manifest_sha256": manifest_sha,
                "sites": sites,
            },
            "stage": stage,
            "depth": INTERVENTION_DEPTH[stage],
            "topology": "side_branch" if stage == "cross_attention_raw" else "main_chain",
        }

    if expected_manifest_class is not None:
        raise ReplayContractError(
            f"{candidate_name} expected manifest class but carries only a direct prefix site"
        )
    _validate_site(
        evidence, candidate_name=candidate_name, block_index=28, stage="after_mlp",
        source_route_vector=source_route_vector,
    )
    return {
        "identity": _manifest_identity(trace),
        "stage": "prefix28",
        "depth": INTERVENTION_DEPTH["prefix28"],
        "topology": "main_chain",
    }


def _validate_site(
    site: Any,
    *,
    candidate_name: str,
    block_index: int,
    stage: str,
    source_route_vector: dict[str, Any],
    expected_trace_sha256: str | None = None,
) -> None:
    if not isinstance(site, dict):
        raise ReplayContractError(f"{candidate_name} intervention site must be an object")
    expected = {
        "block_index": block_index,
        "step_index": 0,
        "stage": stage,
        "branch": "both",
    }
    mismatches = {key: site.get(key) for key, value in expected.items() if site.get(key) != value}
    if mismatches:
        raise ReplayContractError(
            f"{candidate_name} block{block_index} intervention site mismatch: {mismatches}"
        )
    expected_array_key = ",".join(
        f"{branch}_block{block_index}_{stage}" for branch in ("pos", "neg")
    )
    observed_array_key = site.get("array_key")
    if observed_array_key != expected_array_key:
        raise ReplayContractError(
            f"{candidate_name} block{block_index} intervention array key "
            f"{observed_array_key!r} does not match declared stage {expected_array_key!r}"
        )
    if float(site.get("source_delta_scale", 1.0)) != 1.0:
        raise ReplayContractError(f"{candidate_name} intervention site is not exact scale 1.0")
    trace_sha256 = site.get("trace_sha256")
    _require_sha256(trace_sha256, f"{candidate_name} block{block_index} trace")
    if expected_trace_sha256 is not None and trace_sha256 != expected_trace_sha256:
        raise ReplayContractError(
            f"{candidate_name} block{block_index} trace digest does not match source trace"
        )
    observed_route = _canonical_route_vector(site.get("trace_identity"), candidate_name)
    if observed_route != source_route_vector:
        raise ReplayContractError(
            f"{candidate_name} block{block_index} route vector does not match source"
        )


def _source_route_vector(trace: Any) -> dict[str, Any]:
    if "route_identity_json" in trace:
        raw = np.asarray(trace["route_identity_json"])
        if raw.size != 1:
            raise ReplayContractError("source route_identity_json must contain one value")
        encoded = raw.reshape(-1)[0]
        if isinstance(encoded, bytes):
            encoded = encoded.decode("utf-8")
        try:
            identity = json.loads(str(encoded))
        except json.JSONDecodeError as exc:
            raise ReplayContractError(f"invalid source route_identity_json: {exc}") from exc
        return _canonical_route_vector(identity, "source")
    evidence = _read_injection_evidence(trace)
    if not isinstance(evidence, dict):
        raise ReplayContractError("source trace has no route identity evidence")
    return _canonical_route_vector(evidence.get("trace_identity"), "source")


def _canonical_route_vector(identity: Any, label: str) -> dict[str, Any]:
    if not isinstance(identity, dict):
        raise ReplayContractError(f"{label} route identity is missing")
    vector = {field: identity.get(field) for field in ROUTE_VECTOR_FIELDS}
    missing = [field for field, value in vector.items() if value in (None, "")]
    if missing:
        raise ReplayContractError(f"{label} route vector is missing: {', '.join(missing)}")
    if vector["effective_device_type"] != "cuda":
        raise ReplayContractError(f"{label} route is not source CUDA")
    for field in (
        "conditioning_sha256", "shape_flow_noise_sample_sha256",
        "shape_slat_support_sample_sha256", "source_tar_sha256",
    ):
        _require_sha256(vector[field], f"{label} {field}")
    vector["steps"] = int(vector["steps"])
    return vector


def _require_sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ReplayContractError(f"{label} has no valid SHA256")


def _normalize_model_t(recorded: float, sampler_t: float, candidate_name: str) -> float:
    normalized = recorded / 1000.0
    if abs(normalized - sampler_t) <= 1e-7:
        return normalized
    raise ReplayContractError(
        f"{candidate_name} model time is not an explicit 1000x sampler-time match: "
        f"recorded {recorded}, normalized {normalized}, "
        f"sampler t {sampler_t}"
    )


def _schedule_coordinates(steps: int, rescale_t: float, *, step_index: int) -> tuple[float, float]:
    if steps <= 0 or step_index < 0 or step_index >= steps:
        raise ReplayContractError(f"invalid schedule steps/index: {steps}/{step_index}")
    sequence = np.linspace(1.0, 0.0, steps + 1, dtype=np.float64)
    sequence = rescale_t * sequence / (1.0 + (rescale_t - 1.0) * sequence)
    return float(sequence[step_index]), float(sequence[step_index + 1])


def _vector(archive: Any, name: str, *, length: int) -> np.ndarray:
    value = _require_array(archive, name).astype(np.float64, copy=False).reshape(-1)
    if value.size != length or not np.all(np.isfinite(value)):
        raise ReplayContractError(f"{name} must contain {length} finite values")
    return value


def _delta_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    _require_same_shape(reference, candidate, "delta inputs")
    delta = np.asarray(candidate, dtype=np.float32) - np.asarray(reference, dtype=np.float32)
    absolute = np.abs(delta)
    return {
        "mean_abs": float(absolute.mean(dtype=np.float64)),
        "max_abs": float(absolute.max(initial=0.0)),
        "rms": float(np.sqrt(np.square(delta, dtype=np.float64).mean(dtype=np.float64))),
        "nonzero": int(np.count_nonzero(absolute)),
    }


def _squeeze(value: np.ndarray, label: str) -> np.ndarray:
    result = np.asarray(value)
    while result.ndim > 2 and result.shape[0] == 1:
        result = result[0]
    if result.ndim != 2:
        raise ReplayContractError(f"{label} must squeeze to rank 2, got {result.shape}")
    return result.astype(np.float32, copy=False)


def _require_array(archive: Any, name: str) -> np.ndarray:
    if name not in archive:
        raise ReplayContractError(f"artifact is missing required array {name}")
    return np.asarray(archive[name])


def _scalar(archive: Any, name: str, fallback: float | None = None) -> float:
    if name not in archive:
        if fallback is None:
            raise ReplayContractError(f"artifact is missing required scalar {name}")
        return float(fallback)
    value = np.asarray(archive[name])
    if value.size != 1:
        raise ReplayContractError(f"{name} must contain one value, got shape {value.shape}")
    return float(value.reshape(-1)[0])


def _require_exact(reference: np.ndarray, candidate: np.ndarray, label: str) -> None:
    if reference.shape != candidate.shape or not np.array_equal(reference, candidate):
        raise ReplayContractError(f"{label} are not exact")


def _require_same_shape(reference: np.ndarray, candidate: np.ndarray, label: str) -> None:
    if reference.shape != candidate.shape:
        raise ReplayContractError(f"{label} shape mismatch: {reference.shape} vs {candidate.shape}")


def _require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ReplayContractError(f"{label} is missing or blank: {path}")


def _file_identity(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_assignment(value: str, label: str) -> tuple[str, str]:
    if "=" not in value:
        raise ReplayContractError(f"{label} must be NAME=VALUE: {value}")
    name, item = value.split("=", 1)
    if not name or not item:
        raise ReplayContractError(f"{label} must be NAME=VALUE: {value}")
    return name, item


if __name__ == "__main__":
    raise SystemExit(main())
