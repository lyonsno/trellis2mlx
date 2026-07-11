#!/usr/bin/env python3
"""Replay one official TRELLIS.2 sparse-flow block on captured MLX tensors.

This diagnostic intentionally forces the source route to CPU. Trellis-Mac/MPS
has already been caught as non-authoritative on small numerical witnesses, so a
quiet MPS replay would poison the evidence surface this script is meant to
preserve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


SCHEMA = "trellis2mlx.source_sparse_block_replay.v1"
REQUESTED_ROUTE = "source-sparse-block-replay"
SOURCE_COMPARE_NAMES = [
    "norm1",
    "modulated_self_input",
    "q_pre_norm",
    "k_pre_norm",
    "v",
    "q_post_norm",
    "k_post_norm",
    "q_post_rope",
    "k_post_rope",
    "attention_raw",
    "self_attn",
    "after_self",
    "norm2",
    "cross_q_pre_norm",
    "cross_q_post_norm",
    "cross_attention_raw",
    "cross_attn",
    "after_cross",
    "mlp_input",
    "mlp_fc1",
    "mlp_gelu",
    "mlp_fc2",
    "mlp",
    "mlp_gated",
    "after_mlp",
]
ATTENTION_WITNESS_NAMES = (
    "q_post_rope",
    "k_post_rope",
    "v",
    "attention_raw",
    "self_attn",
)
CROSS_ATTENTION_WITNESS_NAMES = (
    "cross_q_post_norm",
    "cross_attention_raw",
    "cross_attn",
)


def _sha256(path: Path) -> str | None:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable_path(path: Path) -> str:
    return str(Path(path).expanduser())


def build_route_identity(
    *,
    requested_route: str,
    effective_device_type: str,
    source_root: Path,
    checkpoint: Path,
    trace: Path,
    branch: str,
    block_index: int,
    step_index: int | None,
) -> dict[str, Any]:
    if effective_device_type != "cpu":
        raise ValueError(
            "source_sparse_block_replay is CPU-only evidence; "
            f"refusing effective device {effective_device_type!r}"
        )
    return {
        "requested_route": requested_route,
        "effective_route": "official-trellis2-source-cpu-selected-sparse-block",
        "effective_device_type": effective_device_type,
        "source_root": _jsonable_path(source_root),
        "checkpoint": _jsonable_path(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "trace": _jsonable_path(trace),
        "trace_sha256": _sha256(trace),
        "branch": branch,
        "block_index": block_index,
        "step_index": step_index,
        "forced_env": {
            "SPARSE_CONV_BACKEND": "none",
            "SPARSE_ATTN_BACKEND": "sdpa",
            "ATTN_BACKEND": "sdpa",
        },
        "forbidden_inferences": [
            "not a Trellis-Mac/MPS authority route",
            "not an end-to-end CUDA generation route",
            "not final GLB parity evidence",
        ],
    }


def _squeeze_leading_ones(array: np.ndarray) -> np.ndarray:
    out = np.asarray(array)
    while out.ndim > 0 and out.shape[0] == 1:
        out = out[0]
    return out


def _require_key(trace: Any, key: str) -> np.ndarray:
    if key not in trace:
        raise KeyError(f"trace missing required key {key!r}")
    return np.asarray(trace[key])


def _optional_scalar(trace: Any, key: str) -> Any:
    if key not in trace:
        return None
    value = np.asarray(trace[key])
    if value.shape != ():
        raise ValueError(f"trace key {key!r} must be scalar, got {value.shape}")
    return value.item()


def _optional_string(trace: Any, key: str) -> str | None:
    if key not in trace:
        return None
    value = np.asarray(trace[key])
    if value.shape != ():
        raise ValueError(f"trace key {key!r} must be scalar string, got {value.shape}")
    return str(value.item())


def _infer_resolution(token_count: int) -> int:
    resolution = round(token_count ** (1.0 / 3.0))
    if resolution ** 3 != token_count:
        raise ValueError(
            "sparse-flow projected block input token count must be a perfect cube, "
            f"got {token_count}"
        )
    return resolution


def load_trace_payload(trace_path: Path, *, branch: str, block_index: int) -> dict[str, Any]:
    trace_path = Path(trace_path)
    prefix = f"{branch}_block{block_index}"
    with np.load(trace_path) as trace:
        trace_block_index = _optional_scalar(trace, "trace_block_index")
        if trace_block_index is not None and int(trace_block_index) != block_index:
            raise ValueError(
                f"trace declares block {trace_block_index}, requested block {block_index}"
            )
        pieces = [
            _squeeze_leading_ones(_require_key(trace, f"{prefix}_{name}"))
            for name in (
                "shift_msa",
                "scale_msa",
                "gate_msa",
                "shift_mlp",
                "scale_mlp",
                "gate_mlp",
            )
        ]
        x = _squeeze_leading_ones(_require_key(trace, f"{prefix}_input")).astype(np.float32)
        return {
            "x": x,
            "mod": np.stack(pieces, axis=0).astype(np.float32),
            "cross_k": _squeeze_leading_ones(
                _require_key(trace, f"{prefix}_cross_k_cached_post_norm")
            ).astype(np.float32),
            "cross_v": _squeeze_leading_ones(
                _require_key(trace, f"{prefix}_cross_v_cached")
            ).astype(np.float32),
            "resolution": _infer_resolution(int(x.shape[0])),
            "step_index": (
                None
                if _optional_scalar(trace, "sparse_flow_trace_step_index") is None
                else int(_optional_scalar(trace, "sparse_flow_trace_step_index"))
            ),
            "trace_input_mode": _optional_string(trace, "sparse_flow_trace_input_mode"),
            "t": None if _optional_scalar(trace, "t") is None else float(_optional_scalar(trace, "t")),
            "captured": {
                name: _squeeze_leading_ones(_require_key(trace, f"{prefix}_{name}")).astype(
                    np.float32
                )
                for name in SOURCE_COMPARE_NAMES
                if f"{prefix}_{name}" in trace
            },
            "captured_final": {
                name: _squeeze_leading_ones(_require_key(trace, f"{branch}_{name}")).astype(
                    np.float32
                )
                for name in ("final_norm", "final_out_flat", "final_output")
                if f"{branch}_{name}" in trace
            },
        }


def _diff_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    delta = np.asarray(candidate, dtype=np.float32) - np.asarray(reference, dtype=np.float32)
    abs_delta = np.abs(delta)
    return {
        "shape_match": True,
        "shape": list(reference.shape),
        "mean_abs": float(abs_delta.mean(dtype=np.float64)),
        "max_abs": float(abs_delta.max(initial=0.0)),
        "nonzero": int(np.count_nonzero(abs_delta)),
    }


def _require_array(mapping: dict[str, np.ndarray], name: str, *, label: str) -> np.ndarray:
    if name not in mapping:
        raise KeyError(f"{label} missing required attention witness array {name!r}")
    return np.asarray(mapping[name], dtype=np.float32)


def build_attention_witness_arrays(
    *,
    source: dict[str, np.ndarray],
    captured: dict[str, np.ndarray],
    to_out_weight: np.ndarray,
    to_out_bias: np.ndarray | None,
    route_identity: dict[str, Any],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "route_identity_json": np.asarray(json.dumps(route_identity, sort_keys=True)),
        "source_to_out_weight": np.asarray(to_out_weight, dtype=np.float32),
    }
    if to_out_bias is not None:
        arrays["source_to_out_bias"] = np.asarray(to_out_bias, dtype=np.float32)
    for name in ATTENTION_WITNESS_NAMES:
        arrays[f"source_{name}"] = _require_array(source, name, label="source")
        arrays[f"captured_{name}"] = _require_array(captured, name, label="captured")
    return arrays


def build_cross_attention_witness_arrays(
    *,
    source: dict[str, np.ndarray],
    captured: dict[str, np.ndarray],
    cross_k: np.ndarray,
    cross_v: np.ndarray,
    to_out_weight: np.ndarray,
    to_out_bias: np.ndarray | None,
    route_identity: dict[str, Any],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "route_identity_json": np.asarray(json.dumps(route_identity, sort_keys=True)),
        "cross_k": np.asarray(cross_k, dtype=np.float32),
        "cross_v": np.asarray(cross_v, dtype=np.float32),
        "source_to_out_weight": np.asarray(to_out_weight, dtype=np.float32),
    }
    if to_out_bias is not None:
        arrays["source_to_out_bias"] = np.asarray(to_out_bias, dtype=np.float32)
    for name in CROSS_ATTENTION_WITNESS_NAMES:
        arrays[f"source_{name}"] = _require_array(source, name, label="source")
        arrays[f"captured_{name}"] = _require_array(captured, name, label="captured")
    return arrays


def _linear_weight_bias(linear: Any) -> tuple[np.ndarray, np.ndarray | None]:
    weight = linear.weight.detach().float().cpu().numpy()
    bias = None
    if getattr(linear, "bias", None) is not None:
        bias = linear.bias.detach().float().cpu().numpy()
    return weight, bias


def module_parameter_dtype(module: Any, *, fallback: Any) -> Any:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return fallback


def _load_source_model(source_root: Path, checkpoint: Path):
    os.environ["SPARSE_CONV_BACKEND"] = "none"
    os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"
    os.environ["ATTN_BACKEND"] = "sdpa"
    source_root = Path(source_root)
    sys.path.insert(0, str(source_root))

    import torch
    from safetensors.torch import load_file

    try:
        from trellis2.models.sparse_structure_flow import SparseStructureFlowModel
    except ImportError:
        from trellis2.models import SparseStructureFlowModel

    torch.set_grad_enabled(False)
    try:
        model = SparseStructureFlowModel(
            resolution=16,
            in_channels=8,
            out_channels=8,
            model_channels=1536,
            cond_channels=1024,
            num_blocks=30,
            num_heads=12,
            mlp_ratio=5.3334,
            pe_mode="rope",
            share_mod=True,
            initialization="scaled",
            qk_rms_norm=True,
            qk_rms_norm_cross=True,
            dtype="bfloat16",
        )
    except TypeError:
        model = SparseStructureFlowModel(
            in_channels=8,
            out_channels=8,
            model_channels=1536,
            context_channels=1024,
            num_blocks=30,
            num_heads=12,
            mlp_hidden=8192,
            resolution=16,
        )
    state = load_file(str(checkpoint), device="cpu")
    load_result = model.load_state_dict(state, strict=False)
    missing = list(load_result.missing_keys)
    unexpected = list(load_result.unexpected_keys)
    allowed_missing = {"rope_phases"}
    if unexpected or set(missing) - allowed_missing:
        raise RuntimeError(
            "source sparse model state_dict mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    model._trellis2mlx_state_dict_load = {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "allowed_missing_keys": sorted(allowed_missing & set(missing)),
    }
    model.eval()
    return torch, model


def _source_block_replay(torch: Any, model: Any, payload: dict[str, Any], *, block_index: int):
    from trellis2.modules.attention import RotaryPositionEmbedder

    block = model.blocks[block_index]
    compute_dtype = torch.bfloat16
    x = torch.from_numpy(payload["x"]).to(dtype=compute_dtype).unsqueeze(0)
    mod = torch.from_numpy(payload["mod"]).to(dtype=compute_dtype)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
        piece.reshape(1, -1) for piece in mod
    ]

    source: dict[str, np.ndarray] = {}

    h = block.norm1(x)
    source["norm1"] = h.squeeze(0).float().cpu().numpy()
    h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
    source["modulated_self_input"] = h.squeeze(0).float().cpu().numpy()

    attn = block.self_attn
    qkv = attn.to_qkv(h)
    qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, attn.num_heads, -1)
    q, k, v = qkv.unbind(dim=2)
    source["q_pre_norm"] = q.squeeze(0).float().cpu().numpy()
    source["k_pre_norm"] = k.squeeze(0).float().cpu().numpy()
    source["v"] = v.squeeze(0).float().cpu().numpy()
    if getattr(attn, "qk_rms_norm", True):
        q = attn.q_rms_norm(q)
        k = attn.k_rms_norm(k)
    source["q_post_norm"] = q.squeeze(0).float().cpu().numpy()
    source["k_post_norm"] = k.squeeze(0).float().cpu().numpy()
    if getattr(attn, "use_rope", False):
        q = RotaryPositionEmbedder.apply_rotary_embedding(q, model.rope_phases)
        k = RotaryPositionEmbedder.apply_rotary_embedding(k, model.rope_phases)
    source["q_post_rope"] = q.squeeze(0).float().cpu().numpy()
    source["k_post_rope"] = k.squeeze(0).float().cpu().numpy()
    raw = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
    )
    raw = raw.transpose(1, 2).reshape(h.shape[0], h.shape[1], -1)
    source["attention_raw"] = raw.squeeze(0).float().cpu().numpy()
    h = attn.to_out(raw)
    source["self_attn"] = h.squeeze(0).float().cpu().numpy()
    h = h * gate_msa.unsqueeze(1)
    x = x + h
    source["after_self"] = x.squeeze(0).float().cpu().numpy()

    h = block.norm2(x)
    source["norm2"] = h.squeeze(0).float().cpu().numpy()
    attn = block.cross_attn
    q = attn.to_q(h)
    q = q.reshape(q.shape[0], q.shape[1], attn.num_heads, -1)
    source["cross_q_pre_norm"] = q.squeeze(0).float().cpu().numpy()
    if getattr(attn, "qk_rms_norm", True):
        q = attn.q_rms_norm(q)
    source["cross_q_post_norm"] = q.squeeze(0).float().cpu().numpy()
    k = torch.from_numpy(payload["cross_k"]).to(dtype=compute_dtype)
    v = torch.from_numpy(payload["cross_v"]).to(dtype=compute_dtype)
    if k.ndim == 3:
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
    raw = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
    )
    raw = raw.transpose(1, 2).reshape(h.shape[0], h.shape[1], -1)
    source["cross_attention_raw"] = raw.squeeze(0).float().cpu().numpy()
    h = attn.to_out(raw)
    source["cross_attn"] = h.squeeze(0).float().cpu().numpy()
    x = x + h
    source["after_cross"] = x.squeeze(0).float().cpu().numpy()

    h = block.norm3(x)
    h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
    source["mlp_input"] = h.squeeze(0).float().cpu().numpy()
    h_fc1 = block.mlp.mlp[0](h) if hasattr(block.mlp, "mlp") else block.mlp.mlp_0(h)
    source["mlp_fc1"] = h_fc1.squeeze(0).float().cpu().numpy()
    gelu = block.mlp.mlp[1] if hasattr(block.mlp, "mlp") else torch.nn.GELU(approximate="tanh")
    h_gelu = gelu(h_fc1)
    source["mlp_gelu"] = h_gelu.squeeze(0).float().cpu().numpy()
    h_fc2 = block.mlp.mlp[2](h_gelu) if hasattr(block.mlp, "mlp") else block.mlp.mlp_2(h_gelu)
    source["mlp_fc2"] = h_fc2.squeeze(0).float().cpu().numpy()
    source["mlp"] = source["mlp_fc2"]
    h = h_fc2 * gate_mlp.unsqueeze(1)
    source["mlp_gated"] = h.squeeze(0).float().cpu().numpy()
    x = x + h
    source["after_mlp"] = x.squeeze(0).float().cpu().numpy()

    if block_index == len(model.blocks) - 1:
        final = torch.nn.functional.layer_norm(x.float(), x.shape[-1:])
        source["final_norm"] = final.squeeze(0).cpu().numpy()
        out_dtype = module_parameter_dtype(model.out_layer, fallback=compute_dtype)
        final_out = model.out_layer(final.to(dtype=out_dtype))
        source["final_out_flat"] = final_out.squeeze(0).float().cpu().numpy()
        resolution = payload["resolution"]
        source["final_output"] = (
            source["final_out_flat"].reshape(resolution, resolution, resolution, -1).transpose(3, 0, 1, 2)
        )

    return source


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _failure_report(
    *,
    route_identity: dict[str, Any],
    failure_phase: str,
    error: BaseException,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "failed",
        "failure_phase": failure_phase,
        "last_trustworthy_phase": None,
        "primary_output_status": "not_written",
        "route_identity": route_identity,
        "elapsed_seconds": elapsed_seconds,
        "error": f"{type(error).__name__}: {error}",
    }


def _classify_failure_phase(exc: BaseException, *, trace_exists: bool) -> str:
    message = str(exc)
    if not trace_exists:
        return "load_trace"
    if isinstance(exc, KeyError) and "trace missing required key" in message:
        return "load_trace"
    if isinstance(exc, ValueError) and message.startswith(("trace ", "sparse-flow projected")):
        return "load_trace"
    return "source_replay"


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    route_identity = build_route_identity(
        requested_route=REQUESTED_ROUTE,
        effective_device_type="cpu",
        source_root=args.source_root,
        checkpoint=args.checkpoint,
        trace=args.trace,
        branch=args.branch,
        block_index=args.block_index,
        step_index=None,
    )

    phase = "load_trace"
    payload = load_trace_payload(args.trace, branch=args.branch, block_index=args.block_index)
    route_identity["step_index"] = payload["step_index"]

    phase = "load_source_model"
    torch, model = _load_source_model(args.source_root, args.checkpoint)
    effective_device_type = next(model.parameters()).device.type
    route_identity = build_route_identity(
        requested_route=REQUESTED_ROUTE,
        effective_device_type=effective_device_type,
        source_root=args.source_root,
        checkpoint=args.checkpoint,
        trace=args.trace,
        branch=args.branch,
        block_index=args.block_index,
        step_index=payload["step_index"],
    )
    route_identity["trace_input_mode"] = payload["trace_input_mode"]
    route_identity["resolution"] = payload["resolution"]
    route_identity["source_state_dict_load"] = getattr(
        model,
        "_trellis2mlx_state_dict_load",
        None,
    )

    phase = "source_block_replay"
    with torch.inference_mode():
        source = _source_block_replay(torch, model, payload, block_index=args.block_index)

    phase = "compare"
    metrics = {
        name: _diff_metrics(payload["captured"][name], source[name])
        for name in SOURCE_COMPARE_NAMES
        if name in payload["captured"] and name in source
    }
    final_metrics = {
        name: _diff_metrics(payload["captured_final"][name], source[name])
        for name in sorted(payload["captured_final"].keys() & source.keys())
    }
    attention_witness_status = None
    if args.attention_witness_output is not None:
        phase = "write_attention_witness"
        to_out_weight, to_out_bias = _linear_weight_bias(model.blocks[args.block_index].self_attn.to_out)
        arrays = build_attention_witness_arrays(
            source=source,
            captured=payload["captured"],
            to_out_weight=to_out_weight,
            to_out_bias=to_out_bias,
            route_identity=route_identity,
        )
        args.attention_witness_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.attention_witness_output, **arrays)
        attention_witness_status = {
            "path": _jsonable_path(args.attention_witness_output),
            "sha256": _sha256(args.attention_witness_output),
            "arrays": {key: list(value.shape) for key, value in arrays.items()},
        }
    cross_attention_witness_status = None
    if args.cross_attention_witness_output is not None:
        phase = "write_cross_attention_witness"
        to_out_weight, to_out_bias = _linear_weight_bias(model.blocks[args.block_index].cross_attn.to_out)
        arrays = build_cross_attention_witness_arrays(
            source=source,
            captured=payload["captured"],
            cross_k=payload["cross_k"],
            cross_v=payload["cross_v"],
            to_out_weight=to_out_weight,
            to_out_bias=to_out_bias,
            route_identity=route_identity,
        )
        args.cross_attention_witness_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.cross_attention_witness_output, **arrays)
        cross_attention_witness_status = {
            "path": _jsonable_path(args.cross_attention_witness_output),
            "sha256": _sha256(args.cross_attention_witness_output),
            "arrays": {key: list(value.shape) for key, value in arrays.items()},
        }
    return {
        "schema": SCHEMA,
        "status": "ok",
        "failure_phase": None,
        "last_trustworthy_phase": phase,
        "primary_output_status": "metrics_written",
        "route_identity": route_identity,
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": metrics,
        "final_metrics": final_metrics,
        "attention_witness": attention_witness_status,
        "cross_attention_witness": cross_attention_witness_status,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--attention-witness-output", type=Path)
    parser.add_argument("--cross-attention-witness-output", type=Path)
    parser.add_argument("--branch", choices=("pos", "neg"), required=True)
    parser.add_argument("--block-index", required=True, type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    route_identity = {
        "requested_route": REQUESTED_ROUTE,
        "effective_route": "not-established",
        "trace": _jsonable_path(args.trace),
        "checkpoint": _jsonable_path(args.checkpoint),
        "source_root": _jsonable_path(args.source_root),
        "branch": args.branch,
        "block_index": args.block_index,
    }
    phase = "load_trace"
    try:
        report = run(args)
        _write_report(args.output, report)
        return 0
    except Exception as exc:
        phase = _classify_failure_phase(exc, trace_exists=Path(args.trace).exists())
        try:
            route_identity = build_route_identity(
                requested_route=REQUESTED_ROUTE,
                effective_device_type="cpu",
                source_root=args.source_root,
                checkpoint=args.checkpoint,
                trace=args.trace,
                branch=args.branch,
                block_index=args.block_index,
                step_index=None,
            )
        except Exception:
            pass
        _write_report(
            args.output,
            _failure_report(
                route_identity=route_identity,
                failure_phase=phase,
                error=exc,
                elapsed_seconds=time.perf_counter() - started,
            ),
        )
        print(f"source_sparse_block_replay failed in {phase}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
