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
DEFAULT_REQUESTED_ROUTE = "layernorm-witness-census"
DEFAULT_EFFECTIVE_ROUTE = "local-layernorm-witness-census"


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
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
