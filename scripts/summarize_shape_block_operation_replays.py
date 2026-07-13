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
    _require_file(source_trace_path, "source trace")
    _require_file(source_step_path, "source step")

    with np.load(source_trace_path, allow_pickle=False) as source_trace, np.load(
        source_step_path, allow_pickle=False
    ) as source_step:
        source_coords = _require_array(source_trace, "coords")
        step_coords = _require_array(source_step, "coords")
        _require_exact(source_coords, step_coords, "source trace coords vs source step coords")
        sample = _squeeze(_require_array(source_step, "sample_feats"), "source sample_feats")
        source_pos = _squeeze(_require_array(source_trace, "pos_final_output"), "source pos")
        source_neg = _squeeze(_require_array(source_trace, "neg_final_output"), "source neg")
        _require_same_shape(sample, source_pos, "source sample vs pos output")
        _require_same_shape(sample, source_neg, "source sample vs neg output")

        t = _scalar(source_step, "t", fallback=_scalar(source_trace, "t"))
        t_prev = _scalar(source_step, "t_prev", fallback=_scalar(source_trace, "t_prev"))
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
                    sample=sample,
                    t=t,
                    t_prev=t_prev,
                    guidance_strength=guidance_strength,
                    guidance_rescale=guidance_rescale,
                )
            )

    replay_rows.append(
        {
            "name": "source",
            "artifact": str(source_trace_path),
            "manifest_identity": None,
            "intervention_depth": len(replay_rows),
            "pos_final_output_source_mean_abs": 0.0,
            "neg_final_output_source_mean_abs": 0.0,
            "pred_final_source_mean_abs": 0.0,
            "pred_final_source_max_abs": 0.0,
            "sample_next_source_mean_abs": 0.0,
            "sample_next_source_max_abs": 0.0,
        }
    )
    for index, row in enumerate(replay_rows):
        row["intervention_depth"] = index

    return {
        "schema": "trellis2mlx.shape_block_operation_replays.v1",
        "status": "done",
        "comparison_class": "exact_source_prefix28_then_block29_operation_replay",
        "source_trace": _file_identity(source_trace_path),
        "source_step": _file_identity(source_step_path),
        "effective_parameters": {
            "block_index": 29,
            "step_index": 0,
            "t": t,
            "t_prev": t_prev,
            "guidance_strength": guidance_strength,
            "guidance_rescale": guidance_rescale,
            "sigma_min": 1e-5,
        },
        "source_reconstruction": {
            "pred_final_mean_abs": source_reconstruction["mean_abs"],
            "pred_final_max_abs": source_reconstruction["max_abs"],
            "pred_final_nonzero": source_reconstruction["nonzero"],
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
    sample: np.ndarray,
    t: float,
    t_prev: float,
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
        recorded_model_t = _scalar(trace, "t")
        candidate_t = _normalize_model_t(recorded_model_t, t, candidate.name)
        candidate_guidance = _scalar(trace, "guidance_strength", fallback=guidance_strength)
        candidate_rescale = _scalar(trace, "guidance_rescale", fallback=guidance_rescale)
        if candidate_guidance != guidance_strength or candidate_rescale != guidance_rescale:
            raise ReplayContractError(f"{candidate.name} guidance parameters do not match source")

        manifest = _manifest_identity(trace)
        if candidate.expected_manifest_class is not None:
            observed = manifest.get("comparison_class") if manifest else None
            if observed != candidate.expected_manifest_class:
                raise ReplayContractError(
                    f"{candidate.name} manifest class {observed!r} does not match "
                    f"{candidate.expected_manifest_class!r}"
                )

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
            "manifest_identity": manifest,
            "recorded_model_t": recorded_model_t,
            "normalized_sampler_t": candidate_t,
            "intervention_depth": 0,
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


def _manifest_identity(trace: Any) -> dict[str, Any] | None:
    if "shape_flow_block_injection_json" not in trace:
        return None
    try:
        evidence = json.loads(str(np.asarray(trace["shape_flow_block_injection_json"]).item()))
    except (ValueError, json.JSONDecodeError) as exc:
        raise ReplayContractError(f"invalid shape_flow_block_injection_json: {exc}") from exc
    if not evidence.get("route_identity_evidence"):
        raise ReplayContractError("injection evidence does not assert route_identity_evidence")
    identity = evidence.get("manifest_identity")
    if isinstance(identity, dict):
        return identity
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


def _normalize_model_t(recorded: float, sampler_t: float, candidate_name: str) -> float:
    if recorded == sampler_t:
        return recorded
    normalized = recorded / 1000.0
    if abs(normalized - sampler_t) <= 1e-7:
        return normalized
    raise ReplayContractError(
        f"{candidate_name} t mismatch: recorded model t {recorded}, normalized {normalized}, "
        f"sampler t {sampler_t}"
    )


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
