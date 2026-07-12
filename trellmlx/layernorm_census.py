"""LayerNorm witness and census diagnostics for TRELLIS flow traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import textwrap
from typing import Any

import numpy as np


REPORT_SCHEMA = "trellis2mlx.layernorm_witness.v1"
BOUNDARY_REPORT_SCHEMA = "trellis2mlx.noaffine_layernorm_boundary_probe.v1"
DEFAULT_REQUESTED_ROUTE = "layernorm-witness-census"
DEFAULT_EFFECTIVE_ROUTE = "local-layernorm-witness-census"
BOUNDARY_EFFECTIVE_ROUTE = "local-noaffine-layernorm-boundary-probe"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a LayerNorm witness/census report from block traces.")
    parser.add_argument("--reference", required=True, type=Path, help="Reference block trace npz")
    parser.add_argument("--candidate", required=True, type=Path, help="Candidate block trace npz")
    parser.add_argument("--output", required=True, type=Path, help="JSON report path")
    parser.add_argument("--trace-prefix", default="neg_block2", help="Trace key prefix, e.g. neg_block2")
    parser.add_argument("--eps", default=1e-6, type=float, help="LayerNorm epsilon")
    parser.add_argument("--requested-route", default=DEFAULT_REQUESTED_ROUTE)
    parser.add_argument("--reference-route-label", default="reference")
    parser.add_argument("--candidate-route-label", default="candidate")
    parser.add_argument("--cuda-capsule", type=Path, help="Optional standalone CUDA partial script path")
    parser.add_argument("--witness-npz", type=Path, help="Optional compact witness npz path")
    parser.add_argument("--boundary-probe", action="store_true", help="Run no-affine LayerNorm boundary probe mode")
    parser.add_argument("--input-key", default="pos_block0_input", help="Boundary probe input array key")
    parser.add_argument("--reference-norm-key", default="pos_block0_norm1", help="Boundary probe reference norm array key")
    parser.add_argument("--candidate-norm-key", default="pos_block0_norm1", help="Boundary probe candidate norm array key")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.boundary_probe:
            report = build_noaffine_layernorm_boundary_report(
                reference_trace_path=args.reference,
                candidate_trace_path=args.candidate,
                input_key=args.input_key,
                reference_norm_key=args.reference_norm_key,
                candidate_norm_key=args.candidate_norm_key,
                eps=args.eps,
                requested_route=args.requested_route,
                reference_route_label=args.reference_route_label,
                candidate_route_label=args.candidate_route_label,
            )
        else:
            report = build_layernorm_witness_report(
                reference_trace_path=args.reference,
                candidate_trace_path=args.candidate,
                trace_prefix=args.trace_prefix,
                eps=args.eps,
                requested_route=args.requested_route,
                reference_route_label=args.reference_route_label,
                candidate_route_label=args.candidate_route_label,
                cuda_capsule_path=args.cuda_capsule,
                witness_npz_path=args.witness_npz,
            )
        _write_json(args.output, report)
        print(json.dumps(_compact_console_summary(report), sort_keys=True))
        return 0
    except Exception as exc:
        report = _failure_report(
            requested_route=args.requested_route,
            reference_trace_path=args.reference,
            candidate_trace_path=args.candidate,
            trace_prefix=args.trace_prefix,
            failure_phase=_classify_failure_phase(exc),
            error=str(exc),
        )
        _write_json(args.output, report)
        print(json.dumps(_compact_console_summary(report), sort_keys=True))
        return 1


def build_noaffine_layernorm_boundary_report(
    *,
    reference_trace_path: Path,
    candidate_trace_path: Path,
    input_key: str,
    reference_norm_key: str,
    candidate_norm_key: str,
    eps: float = 1e-6,
    requested_route: str = "noaffine-layernorm-boundary-probe",
    reference_route_label: str = "reference",
    candidate_route_label: str = "candidate",
) -> dict[str, Any]:
    reference_trace_path = Path(reference_trace_path)
    candidate_trace_path = Path(candidate_trace_path)
    with np.load(reference_trace_path) as reference, np.load(candidate_trace_path) as candidate:
        _require_keys(reference, (reference_norm_key,), "reference")
        _require_keys(candidate, (input_key, candidate_norm_key), "candidate")
        reference_norm = _as_batched_tokens(np.asarray(reference[reference_norm_key], dtype=np.float32), reference_norm_key)
        candidate_norm = _as_batched_tokens(np.asarray(candidate[candidate_norm_key], dtype=np.float32), candidate_norm_key)
        layernorm_input = _as_batched_tokens(np.asarray(candidate[input_key], dtype=np.float32), input_key)
        if reference_norm.shape != candidate_norm.shape:
            raise ValueError(
                f"norm shape mismatch: reference {reference_norm.shape} vs candidate {candidate_norm.shape}"
            )
        if layernorm_input.shape != candidate_norm.shape:
            raise ValueError(
                f"input/norm shape mismatch: input {layernorm_input.shape} vs candidate norm {candidate_norm.shape}"
            )

        normalized = _noaffine_layernorm_two_pass_bf16(layernorm_input, eps=eps)
        return {
            "schema": BOUNDARY_REPORT_SCHEMA,
            "status": "ok",
            "requested_route": requested_route,
            "effective_route": BOUNDARY_EFFECTIVE_ROUTE,
            "eps": float(eps),
            "keys": {
                "input": input_key,
                "reference_norm": reference_norm_key,
                "candidate_norm": candidate_norm_key,
            },
            "known_routes": {
                "reference": reference_route_label,
                "candidate": candidate_route_label,
            },
            "artifacts": {
                "reference_trace": _artifact_identity(reference_trace_path),
                "candidate_trace": _artifact_identity(candidate_trace_path),
            },
            "runtime": _runtime_identity(),
            "input_identity": {
                "shape": list(layernorm_input.shape),
                "dtype": str(layernorm_input.dtype),
            },
            "norm_delta": _array_delta(reference_norm, candidate_norm),
            "candidate_formula_delta": _array_delta(candidate_norm, normalized),
            "reference_formula_delta": _array_delta(reference_norm, normalized),
            "coordinate_summary": _boundary_coordinate_summary(reference_norm, candidate_norm, layernorm_input),
            "rowwise_perturbation_probe": _rowwise_perturbation_probe(
                reference_norm=reference_norm,
                candidate_norm=candidate_norm,
                layernorm_input=layernorm_input,
                eps=eps,
            ),
        }


def _as_batched_tokens(array: np.ndarray, key: str) -> np.ndarray:
    if array.ndim == 2:
        return array[None, ...].astype(np.float32)
    if array.ndim == 3:
        return array.astype(np.float32)
    raise ValueError(f"{key} must have shape [tokens, channels] or [batch, tokens, channels], got {array.shape}")


def build_layernorm_witness_report(
    *,
    reference_trace_path: Path,
    candidate_trace_path: Path,
    trace_prefix: str,
    eps: float = 1e-6,
    requested_route: str = DEFAULT_REQUESTED_ROUTE,
    reference_route_label: str = "reference",
    candidate_route_label: str = "candidate",
    cuda_capsule_path: Path | None = None,
    witness_npz_path: Path | None = None,
) -> dict[str, Any]:
    reference_trace_path = Path(reference_trace_path)
    candidate_trace_path = Path(candidate_trace_path)
    with np.load(reference_trace_path) as reference, np.load(candidate_trace_path) as candidate:
        keys = _trace_keys(trace_prefix)
        _require_keys(reference, tuple(keys.values()), "reference")
        _require_keys(candidate, tuple(keys.values()), "candidate")

        ref_after_cross = np.asarray(reference[keys["after_cross"]], dtype=np.float32)
        cand_after_cross = np.asarray(candidate[keys["after_cross"]], dtype=np.float32)
        ref_shift = np.asarray(reference[keys["shift_mlp"]], dtype=np.float32)
        cand_shift = np.asarray(candidate[keys["shift_mlp"]], dtype=np.float32)
        ref_scale = np.asarray(reference[keys["scale_mlp"]], dtype=np.float32)
        cand_scale = np.asarray(candidate[keys["scale_mlp"]], dtype=np.float32)
        ref_mlp_input = np.asarray(reference[keys["mlp_input"]], dtype=np.float32)
        cand_mlp_input = np.asarray(candidate[keys["mlp_input"]], dtype=np.float32)

        trace_identity = _trace_identity(reference, candidate)
        input_identity = {
            "after_cross": _array_identity(ref_after_cross, cand_after_cross),
            "shift_mlp": _array_identity(ref_shift, cand_shift),
            "scale_mlp": _array_identity(ref_scale, cand_scale),
        }
        input_identity.update(
            {
                "after_cross_exact": input_identity["after_cross"]["exact_match"],
                "shift_mlp_exact": input_identity["shift_mlp"]["exact_match"],
                "scale_mlp_exact": input_identity["scale_mlp"]["exact_match"],
            }
        )

        mlp_delta = _array_delta(ref_mlp_input, cand_mlp_input)
        channel_signature = _channel_signature(ref_mlp_input, cand_mlp_input)
        census = _build_census(
            after_cross=ref_after_cross,
            shift_mlp=ref_shift,
            scale_mlp=ref_scale,
            reference_mlp_input=ref_mlp_input,
            candidate_mlp_input=cand_mlp_input,
            eps=eps,
        )

        if witness_npz_path is not None:
            _write_witness_npz(
                witness_npz_path,
                after_cross=ref_after_cross,
                shift_mlp=ref_shift,
                scale_mlp=ref_scale,
                reference_mlp_input=ref_mlp_input,
                candidate_mlp_input=cand_mlp_input,
            )
        if cuda_capsule_path is not None:
            _write_cuda_capsule(cuda_capsule_path, witness_npz_path=witness_npz_path, eps=eps)

        return {
            "schema": REPORT_SCHEMA,
            "status": "ok",
            "requested_route": requested_route,
            "effective_route": DEFAULT_EFFECTIVE_ROUTE,
            "trace_prefix": trace_prefix,
            "eps": float(eps),
            "known_routes": {
                "reference": reference_route_label,
                "candidate": candidate_route_label,
            },
            "artifacts": {
                "reference_trace": _artifact_identity(reference_trace_path),
                "candidate_trace": _artifact_identity(candidate_trace_path),
                "witness_npz": _artifact_identity(witness_npz_path) if witness_npz_path else None,
                "cuda_capsule": _artifact_identity(cuda_capsule_path) if cuda_capsule_path else None,
            },
            "runtime": _runtime_identity(),
            "trace_identity": trace_identity,
            "input_identity": input_identity,
            "mlp_input_delta": mlp_delta,
            "channel_signature": channel_signature,
            "census": census,
        }


def _trace_keys(trace_prefix: str) -> dict[str, str]:
    return {
        "after_cross": f"{trace_prefix}_after_cross",
        "shift_mlp": f"{trace_prefix}_shift_mlp",
        "scale_mlp": f"{trace_prefix}_scale_mlp",
        "mlp_input": f"{trace_prefix}_mlp_input",
    }


def _require_keys(trace: Any, keys: tuple[str, ...], side: str) -> None:
    missing = [key for key in keys if key not in trace]
    if missing:
        raise KeyError(f"{side} trace missing required arrays: {', '.join(missing)}")


def _trace_identity(reference: Any, candidate: Any) -> dict[str, Any]:
    return {
        "block_index": _same_scalar(reference, candidate, "trace_block_index"),
        "step_index": _same_scalar(reference, candidate, "shape_flow_trace_step_index"),
        "t": _same_scalar(reference, candidate, "t"),
        "steps": _same_scalar(reference, candidate, "steps"),
    }


def _same_scalar(reference: Any, candidate: Any, key: str) -> Any:
    if key not in reference or key not in candidate:
        return None
    ref_value = np.asarray(reference[key]).item()
    cand_value = np.asarray(candidate[key]).item()
    if ref_value != cand_value:
        return {"reference": _json_scalar(ref_value), "candidate": _json_scalar(cand_value)}
    return _json_scalar(ref_value)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _array_identity(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    shape_match = reference.shape == candidate.shape
    dtype_match = str(reference.dtype) == str(candidate.dtype)
    return {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "shape_match": shape_match,
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "dtype_match": dtype_match,
        "exact_match": bool(shape_match and np.array_equal(reference, candidate)),
        **(_array_delta(reference, candidate) if shape_match else {}),
    }


def _array_delta(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    shape_match = reference.shape == candidate.shape
    dtype_match = str(reference.dtype) == str(candidate.dtype)
    summary: dict[str, Any] = {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "shape_match": shape_match,
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "dtype_match": dtype_match,
    }
    if not shape_match:
        return summary
    diff = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    summary.update(_diff_summary(diff))
    summary["exact_match"] = bool(np.array_equal(reference, candidate))
    return summary


def _diff_summary(diff: np.ndarray) -> dict[str, Any]:
    return {
        "max_abs_diff": float(np.max(diff)) if diff.size else 0.0,
        "mean_abs_diff": float(np.mean(diff)) if diff.size else 0.0,
        "rms_diff": float(np.sqrt(np.mean(np.square(diff)))) if diff.size else 0.0,
        "nonzero_count": int(np.count_nonzero(diff)) if diff.size else 0,
    }


def _channel_signature(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape or reference.ndim < 2:
        return {"shape_match": reference.shape == candidate.shape, "differing_channel_count": None, "channels": []}
    diff = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    channel_axis = diff.ndim - 1
    token_axes = tuple(range(channel_axis))
    channel_max = np.max(diff, axis=token_axes)
    differing_channels = np.nonzero(channel_max > 0)[0]
    token_count = int(np.prod(diff.shape[:-1]))
    rows = diff.reshape(token_count, diff.shape[-1])
    channels = []
    for channel in differing_channels[:64]:
        per_token = rows[:, int(channel)]
        nz = per_token > 0
        unique_abs = np.unique(per_token[nz])
        channels.append(
            {
                "channel": int(channel),
                "max_abs_diff": float(np.max(per_token)) if per_token.size else 0.0,
                "mean_abs_diff": float(np.mean(per_token)) if per_token.size else 0.0,
                "token_coverage": int(np.count_nonzero(nz)),
                "token_count": token_count,
                "covers_all_tokens": bool(np.count_nonzero(nz) == token_count),
                "unique_abs_diffs_sample": [float(value) for value in unique_abs[:16]],
                "unique_abs_diff_count": int(len(unique_abs)),
            }
        )
    return {
        "shape_match": True,
        "token_count": token_count,
        "channel_count": int(diff.shape[-1]),
        "differing_channel_count": int(len(differing_channels)),
        "channels_truncated": bool(len(differing_channels) > len(channels)),
        "channels": channels,
    }


def _build_census(
    *,
    after_cross: np.ndarray,
    shift_mlp: np.ndarray,
    scale_mlp: np.ndarray,
    reference_mlp_input: np.ndarray,
    candidate_mlp_input: np.ndarray,
    eps: float,
) -> dict[str, Any]:
    variants = []
    for priority, (name, variant) in enumerate(
        _layernorm_variants(after_cross=after_cross, shift_mlp=shift_mlp, scale_mlp=scale_mlp, eps=eps)
    ):
        vs_reference = _array_delta(reference_mlp_input, variant)
        vs_candidate = _array_delta(candidate_mlp_input, variant)
        variants.append(
            {
                "name": name,
                "priority": priority,
                "max_abs_diff_vs_reference": vs_reference.get("max_abs_diff"),
                "mean_abs_diff_vs_reference": vs_reference.get("mean_abs_diff"),
                "rms_diff_vs_reference": vs_reference.get("rms_diff"),
                "exact_match_reference": vs_reference.get("exact_match"),
                "max_abs_diff_vs_candidate": vs_candidate.get("max_abs_diff"),
                "mean_abs_diff_vs_candidate": vs_candidate.get("mean_abs_diff"),
                "exact_match_candidate": vs_candidate.get("exact_match"),
            }
        )
    variants.sort(
        key=lambda item: (
            item["max_abs_diff_vs_reference"] if item["max_abs_diff_vs_reference"] is not None else float("inf"),
            item["mean_abs_diff_vs_reference"] if item["mean_abs_diff_vs_reference"] is not None else float("inf"),
            item["priority"],
        )
    )
    best = variants[0]["name"] if variants else None
    exact_matches = [variant["name"] for variant in variants if variant["exact_match_reference"]]
    return {
        "variants": variants,
        "best_reference_match": best,
        "exact_reference_matches": exact_matches,
    }


def _layernorm_variants(
    *,
    after_cross: np.ndarray,
    shift_mlp: np.ndarray,
    scale_mlp: np.ndarray,
    eps: float,
) -> list[tuple[str, np.ndarray]]:
    x = after_cross.astype(np.float32)
    shift = shift_mlp.astype(np.float32)
    scale = scale_mlp.astype(np.float32)

    two_pass = _layernorm_two_pass_fp32(x, eps)
    moment = _layernorm_moment_fp32(x, eps)
    two_pass_bf16_norm = _round_float32_to_bf16(two_pass)
    two_pass_bf16_input = _layernorm_two_pass_fp32(_round_float32_to_bf16(x), eps)
    variants = [
        ("numpy_two_pass_fp32", _modulate(two_pass, shift, scale)),
        ("numpy_moment_fp32", _modulate(moment, shift, scale)),
        ("numpy_two_pass_fp32_bf16_norm_then_mod", _modulate(two_pass_bf16_norm, shift, scale)),
        (
            "numpy_two_pass_fp32_bf16_input_bf16_norm_then_mod",
            _modulate(_round_float32_to_bf16(two_pass_bf16_input), shift, scale),
        ),
        (
            "numpy_two_pass_fp32_bf16_norm_bf16_mod",
            _round_float32_to_bf16(_modulate(two_pass_bf16_norm, _round_float32_to_bf16(shift), _round_float32_to_bf16(scale))),
        ),
    ]
    variants.extend(_mlx_layernorm_variants(after_cross=x, shift_mlp=shift, scale_mlp=scale))
    variants.extend(_torch_layernorm_variants(after_cross=x, shift_mlp=shift, scale_mlp=scale, eps=eps))
    return variants


def _mlx_layernorm_variants(
    *,
    after_cross: np.ndarray,
    shift_mlp: np.ndarray,
    scale_mlp: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    try:
        import mlx.core as mx

        from trellmlx.models.sparse_structure_flow import _layernorm_noaffine
    except Exception:
        return []

    variants: list[tuple[str, np.ndarray]] = []
    for name, dtype in (
        ("float32", mx.float32),
        ("bfloat16", mx.bfloat16),
        ("float16", mx.float16),
    ):
        x = mx.array(after_cross).astype(dtype)
        shift = mx.array(shift_mlp).astype(dtype)
        scale = mx.array(scale_mlp).astype(dtype)
        out = (_layernorm_noaffine(x) * (1 + scale) + shift).astype(mx.float32)
        mx.eval(out)
        variants.append((f"mlx_trellmlx_noaffine_{name}", np.array(out, dtype=np.float32)))
    return variants


def _torch_layernorm_variants(
    *,
    after_cross: np.ndarray,
    shift_mlp: np.ndarray,
    scale_mlp: np.ndarray,
    eps: float,
) -> list[tuple[str, np.ndarray]]:
    try:
        import torch
        import torch.nn.functional as F
    except Exception:
        return []

    variants: list[tuple[str, np.ndarray]] = []
    devices = ["cpu"]
    try:
        if torch.backends.mps.is_available():
            devices.append("mps")
    except Exception:
        pass
    for device in devices:
        for name, dtype in (
            ("float32", torch.float32),
            ("bfloat16", torch.bfloat16),
            ("float16", torch.float16),
        ):
            try:
                x = torch.from_numpy(after_cross).to(device=device, dtype=dtype)
                shift = torch.from_numpy(shift_mlp).to(device=device, dtype=dtype)
                scale = torch.from_numpy(scale_mlp).to(device=device, dtype=dtype)
                out = (F.layer_norm(x, (x.shape[-1],), weight=None, bias=None, eps=eps) * (1 + scale) + shift).to(torch.float32)
                variants.append((f"torch_{device}_layer_norm_{name}", out.cpu().numpy().astype(np.float32)))
            except Exception:
                continue
    return variants


def _layernorm_two_pass_fp32(x: np.ndarray, eps: float) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True, dtype=np.float32)
    centered = x - mean
    var = np.mean(centered * centered, axis=-1, keepdims=True, dtype=np.float32)
    return (centered * (1.0 / np.sqrt(var + np.float32(eps), dtype=np.float32))).astype(np.float32)


def _layernorm_moment_fp32(x: np.ndarray, eps: float) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True, dtype=np.float32)
    ex2 = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32)
    var = np.maximum(ex2 - mean * mean, np.float32(0.0))
    return ((x - mean) * (1.0 / np.sqrt(var + np.float32(eps), dtype=np.float32))).astype(np.float32)


def _modulate(normed: np.ndarray, shift: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (normed * (np.float32(1.0) + scale) + shift).astype(np.float32)


def _round_float32_to_bf16(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    bits = x.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = bits + rounding_bias
    rounded = rounded & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def _noaffine_layernorm_two_pass_bf16(x: np.ndarray, *, eps: float) -> np.ndarray:
    xf = np.asarray(x, dtype=np.float32)
    mean = np.mean(xf, axis=-1, keepdims=True, dtype=np.float32)
    centered = xf - mean
    var = np.mean(centered * centered, axis=-1, keepdims=True, dtype=np.float32)
    normed = centered * (np.float32(1.0) / np.sqrt(var + np.float32(eps), dtype=np.float32))
    return _round_float32_to_bf16(normed)


def _boundary_coordinate_summary(
    reference_norm: np.ndarray,
    candidate_norm: np.ndarray,
    layernorm_input: np.ndarray,
) -> dict[str, Any]:
    diff = reference_norm.astype(np.float32) - candidate_norm.astype(np.float32)
    coords = np.argwhere(diff != 0)
    if coords.size == 0:
        return {
            "affected_value_count": 0,
            "affected_token_count": 0,
            "affected_channel_count": 0,
            "unique_signed_diffs": [],
            "tokens": [],
        }
    token_pairs = sorted({(int(batch), int(token)) for batch, token, _ in coords})
    channels = sorted({int(channel) for _, _, channel in coords})
    values, counts = np.unique(diff[diff != 0], return_counts=True)
    means = np.mean(layernorm_input, axis=-1, dtype=np.float32)
    centered = layernorm_input - means[..., None]
    variances = np.mean(centered * centered, axis=-1, dtype=np.float32)
    token_rows = []
    for batch, token in token_pairs:
        token_coords = coords[(coords[:, 0] == batch) & (coords[:, 1] == token)]
        token_diffs = diff[batch, token, token_coords[:, 2]]
        token_values, token_counts = np.unique(token_diffs, return_counts=True)
        token_rows.append(
            {
                "batch": batch,
                "token": token,
                "affected_value_count": int(len(token_coords)),
                "input_mean": float(means[batch, token]),
                "input_variance": float(variances[batch, token]),
                "channels": [int(channel) for channel in token_coords[:, 2]],
                "unique_signed_diffs": [
                    {"value": float(value), "count": int(count)}
                    for value, count in zip(token_values, token_counts)
                ],
            }
        )
    token_rows.sort(key=lambda item: (-item["affected_value_count"], item["batch"], item["token"]))
    return {
        "affected_value_count": int(len(coords)),
        "affected_token_count": int(len(token_pairs)),
        "affected_channel_count": int(len(channels)),
        "affected_channels": channels,
        "unique_signed_diffs": [
            {"value": float(value), "count": int(count)}
            for value, count in zip(values, counts)
        ],
        "tokens": token_rows,
    }


def _rowwise_perturbation_probe(
    *,
    reference_norm: np.ndarray,
    candidate_norm: np.ndarray,
    layernorm_input: np.ndarray,
    eps: float,
) -> dict[str, Any]:
    normalized_float = _noaffine_layernorm_two_pass_float(layernorm_input, eps=eps)
    affected = np.argwhere(reference_norm != candidate_norm)
    token_pairs = sorted({(int(batch), int(token)) for batch, token, _ in affected})
    scale_grid = _perturbation_grid()
    bias_grid = _perturbation_grid()
    return {
        "scale": _score_rowwise_grid(
            reference_norm=reference_norm,
            candidate_norm=candidate_norm,
            normalized_float=normalized_float,
            token_pairs=token_pairs,
            grid=scale_grid,
            mode="scale",
        ),
        "bias": _score_rowwise_grid(
            reference_norm=reference_norm,
            candidate_norm=candidate_norm,
            normalized_float=normalized_float,
            token_pairs=token_pairs,
            grid=bias_grid,
            mode="bias",
        ),
    }


def _noaffine_layernorm_two_pass_float(x: np.ndarray, *, eps: float) -> np.ndarray:
    xf = np.asarray(x, dtype=np.float32)
    mean = np.mean(xf, axis=-1, keepdims=True, dtype=np.float32)
    centered = xf - mean
    var = np.mean(centered * centered, axis=-1, keepdims=True, dtype=np.float32)
    return (centered * (np.float32(1.0) / np.sqrt(var + np.float32(eps), dtype=np.float32))).astype(np.float32)


def _perturbation_grid() -> list[float]:
    return sorted(
        {
            float(value)
            for span, count in ((2e-5, 161), (2e-4, 161), (1e-3, 201))
            for value in np.linspace(-span, span, count)
        }
    )


def _score_rowwise_grid(
    *,
    reference_norm: np.ndarray,
    candidate_norm: np.ndarray,
    normalized_float: np.ndarray,
    token_pairs: list[tuple[int, int]],
    grid: list[float],
    mode: str,
) -> dict[str, Any]:
    rows = []
    solved = 0
    improved = 0
    for batch, token in token_pairs:
        target = reference_norm[batch, token]
        baseline = _row_score(candidate_norm[batch, token], target)
        best = {"value": 0.0, **baseline}
        for value in grid:
            if mode == "scale":
                candidate = _round_float32_to_bf16(normalized_float[batch, token] * np.float32(1.0 + value))
            elif mode == "bias":
                candidate = _round_float32_to_bf16(normalized_float[batch, token] + np.float32(value))
            else:
                raise ValueError(f"unknown perturbation mode: {mode}")
            current = _row_score(candidate, target)
            if _score_tuple(current) < _score_tuple(best):
                best = {"value": float(value), **current}
        is_solved = best["nonzero_count"] == 0
        is_improved = _score_tuple(best) < _score_tuple(baseline)
        solved += int(is_solved)
        improved += int(is_improved)
        rows.append(
            {
                "batch": batch,
                "token": token,
                "baseline": baseline,
                "best": best,
                "solved": bool(is_solved),
                "improved": bool(is_improved),
            }
        )
    rows.sort(key=lambda item: (not item["solved"], not item["improved"], item["batch"], item["token"]))
    return {
        "mode": mode,
        "affected_token_count": int(len(token_pairs)),
        "solved_token_count": int(solved),
        "improved_token_count": int(improved),
        "grid": {
            "min": float(min(grid)) if grid else None,
            "max": float(max(grid)) if grid else None,
            "count": int(len(grid)),
        },
        "tokens": rows,
    }


def _row_score(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    diff = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    return {
        "nonzero_count": int(np.count_nonzero(diff)),
        "mean_abs_diff": float(np.mean(diff)) if diff.size else 0.0,
        "max_abs_diff": float(np.max(diff)) if diff.size else 0.0,
    }


def _score_tuple(score: dict[str, Any]) -> tuple[int, float, float]:
    return (
        int(score["nonzero_count"]),
        float(score["mean_abs_diff"]),
        float(score["max_abs_diff"]),
    )


def _write_witness_npz(
    path: Path,
    *,
    after_cross: np.ndarray,
    shift_mlp: np.ndarray,
    scale_mlp: np.ndarray,
    reference_mlp_input: np.ndarray,
    candidate_mlp_input: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        after_cross=after_cross.astype(np.float32),
        shift_mlp=shift_mlp.astype(np.float32),
        scale_mlp=scale_mlp.astype(np.float32),
        reference_mlp_input=reference_mlp_input.astype(np.float32),
        candidate_mlp_input=candidate_mlp_input.astype(np.float32),
    )


def _write_cuda_capsule(path: Path, *, witness_npz_path: Path | None, eps: float) -> None:
    witness_expr = "Path(sys.argv[1])" if witness_npz_path is None else f"Path({str(witness_npz_path)!r})"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env python3
            \"\"\"Tiny CUDA LayerNorm partial for a frozen TRELLIS block2 witness.\"\"\"

            import json
            from pathlib import Path
            import sys

            import numpy as np
            import torch
            import torch.nn.functional as F


            witness_path = {witness_expr}
            out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else witness_path.with_suffix(".cuda-layernorm.json")
            data = np.load(witness_path)
            device = torch.device("cuda")
            x = torch.from_numpy(data["after_cross"]).to(device=device, dtype=torch.bfloat16)
            shift = torch.from_numpy(data["shift_mlp"]).to(device=device, dtype=torch.bfloat16)
            scale = torch.from_numpy(data["scale_mlp"]).to(device=device, dtype=torch.bfloat16)
            expected = torch.from_numpy(data["reference_mlp_input"]).to(device=device, dtype=torch.float32)

            norm = F.layer_norm(x, (x.shape[-1],), weight=None, bias=None, eps={eps!r})
            out = (norm * (1 + scale) + shift).to(torch.float32)
            diff = (out - expected).abs()
            report = {{
                "schema": "trellis2mlx.cuda_layernorm_partial.v1",
                "witness_path": str(witness_path),
                "torch_version": torch.__version__,
                "cuda_device": torch.cuda.get_device_name(0),
                "output_dtype": str(out.dtype),
                "max_abs_diff_vs_reference": float(diff.max().item()),
                "mean_abs_diff_vs_reference": float(diff.mean().item()),
                "exact_match_reference": bool(torch.equal(out.cpu(), expected.cpu())),
            }}
            out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\\n")
            print(json.dumps(report, sort_keys=True))
            """
        )
    )
    path.chmod(0o755)


def _artifact_identity(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    exists = path.exists()
    ident: dict[str, Any] = {
        "path": str(path),
        "exists": exists,
    }
    if exists and path.is_file():
        ident["size_bytes"] = path.stat().st_size
        ident["sha256"] = _sha256(path)
    return ident


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_identity() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }


def _failure_report(
    *,
    requested_route: str,
    reference_trace_path: Path,
    candidate_trace_path: Path,
    trace_prefix: str,
    failure_phase: str,
    error: str,
) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "status": "failed",
        "requested_route": requested_route,
        "effective_route": DEFAULT_EFFECTIVE_ROUTE,
        "failure_phase": failure_phase,
        "error": error,
        "trace_prefix": trace_prefix,
        "artifacts": {
            "reference_trace": _artifact_identity(Path(reference_trace_path)),
            "candidate_trace": _artifact_identity(Path(candidate_trace_path)),
        },
    }


def _classify_failure_phase(exc: Exception) -> str:
    if isinstance(exc, (FileNotFoundError, KeyError, OSError)):
        return "load"
    return "compare"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _compact_console_summary(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("status") != "ok":
        return {
            "schema": report.get("schema"),
            "status": report.get("status"),
            "failure_phase": report.get("failure_phase"),
            "error": report.get("error"),
        }
    if report.get("schema") == BOUNDARY_REPORT_SCHEMA:
        return {
            "schema": report["schema"],
            "status": report["status"],
            "affected_value_count": report["coordinate_summary"]["affected_value_count"],
            "affected_token_count": report["coordinate_summary"]["affected_token_count"],
            "affected_channel_count": report["coordinate_summary"]["affected_channel_count"],
            "scale_solved_token_count": report["rowwise_perturbation_probe"]["scale"]["solved_token_count"],
            "bias_solved_token_count": report["rowwise_perturbation_probe"]["bias"]["solved_token_count"],
        }
    return {
        "schema": report["schema"],
        "status": report["status"],
        "trace_prefix": report["trace_prefix"],
        "mlp_input_max_abs_diff": report["mlp_input_delta"]["max_abs_diff"],
        "differing_channel_count": report["channel_signature"]["differing_channel_count"],
        "best_reference_match": report["census"]["best_reference_match"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
