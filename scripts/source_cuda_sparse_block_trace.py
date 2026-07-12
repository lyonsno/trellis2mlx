#!/usr/bin/env python3
"""Trace official TRELLIS.2 source-CUDA sparse-flow block internals.

This diagnostic differs from ``source_sparse_block_replay.py`` in one load-bearing
way: it does not start from a captured MLX block input. It starts from a saved
source-CUDA sparse-flow step input, runs the official source model forward on
CUDA through the preceding blocks, then records selected block internals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import time
from typing import Any

import numpy as np


SCHEMA = "trellis2mlx.source_cuda_sparse_block_trace.v1"
REQUESTED_ROUTE = "source-cuda-sparse-block-trace"
EFFECTIVE_ROUTE = "official-trellis2-source-cuda-sparse-flow-block-trace"
TRACE_NAMES = (
    "input",
    "shift_msa",
    "scale_msa",
    "gate_msa",
    "shift_mlp",
    "scale_mlp",
    "gate_mlp",
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
    "cross_k_post_norm",
    "cross_v",
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
)
COMPACT_TRACE_NAMES = ("input", "after_self", "after_cross", "after_mlp")


def _jsonable_path(path: Path) -> str:
    return str(Path(path).expanduser())


def _sha256(path: Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_branch_conditioning_key(branch: str) -> str:
    if branch == "pos":
        return "cond"
    if branch == "neg":
        return "neg_cond"
    raise ValueError(f"branch must be 'pos' or 'neg', got {branch!r}")


def build_route_identity(
    *,
    effective_device_type: str,
    source_steps: Path,
    conditioning: Path,
    checkpoint: Path,
    source_root: Path,
    branch: str,
    step_index: int,
    block_indices: tuple[int, ...],
    sparse_conv_backend: str,
    sparse_attn_backend: str,
    trace_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if effective_device_type != "cuda":
        raise ValueError(
            "source_cuda_sparse_block_trace is CUDA-only evidence; "
            f"refusing effective device {effective_device_type!r}"
        )
    return {
        "requested_route": REQUESTED_ROUTE,
        "effective_route": EFFECTIVE_ROUTE,
        "effective_device_type": effective_device_type,
        "source_steps": _jsonable_path(source_steps),
        "source_steps_sha256": _sha256(source_steps),
        "conditioning": _jsonable_path(conditioning),
        "conditioning_sha256": _sha256(conditioning),
        "checkpoint": _jsonable_path(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "source_root": _jsonable_path(source_root),
        "branch": branch,
        "conditioning_key": select_branch_conditioning_key(branch),
        "step_index": int(step_index),
        "block_indices": [int(v) for v in block_indices],
        "trace_names": list(trace_names if trace_names is not None else TRACE_NAMES),
        "forced_env": {
            "SPARSE_CONV_BACKEND": sparse_conv_backend,
            "SPARSE_ATTN_BACKEND": sparse_attn_backend,
            "ATTN_BACKEND": sparse_attn_backend,
        },
        "forbidden_inferences": [
            "not a captured-MLX-block-input replay",
            "not Trellis-Mac/MPS evidence",
            "not final mesh or GLB parity evidence",
            "not a claim about texture/finalization unless downstream artifacts are listed",
        ],
    }


def parse_block_indices(value: str) -> tuple[int, ...]:
    indices = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not indices:
        raise ValueError("--block-indices must contain at least one integer")
    if len(set(indices)) != len(indices):
        raise ValueError(f"--block-indices contains duplicates: {value!r}")
    for index in indices:
        if index < 0:
            raise ValueError(f"block index must be non-negative, got {index}")
    return indices


def parse_trace_names(value: str | None) -> tuple[str, ...]:
    if value is None or value.strip() == "" or value.strip() == "all":
        return TRACE_NAMES
    if value.strip() == "compact":
        return COMPACT_TRACE_NAMES
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    if not names:
        raise ValueError("--trace-names must contain at least one trace name")
    unknown = sorted(set(names) - set(TRACE_NAMES))
    if unknown:
        raise ValueError(f"unknown trace name(s): {', '.join(unknown)}")
    return names


def schedule_pairs(steps: int, rescale_t: float) -> list[tuple[float, float]]:
    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    return [(float(t_seq[i]), float(t_seq[i + 1])) for i in range(steps)]


def apply_sparse_backend_env(conv_backend: str, attn_backend: str) -> dict[str, str]:
    os.environ["SPARSE_CONV_BACKEND"] = conv_backend
    os.environ["SPARSE_ATTN_BACKEND"] = attn_backend
    os.environ["ATTN_BACKEND"] = attn_backend
    return {
        "SPARSE_CONV_BACKEND": conv_backend,
        "SPARSE_ATTN_BACKEND": attn_backend,
        "ATTN_BACKEND": attn_backend,
    }


def extract_source(source_tar: Path, base: Path) -> Path:
    source_tree = base / "trellis2_source"
    if source_tree.is_dir():
        return source_tree
    source_tar = Path(source_tar)
    if not source_tar.is_file():
        alternate = base / "trellis2_source.tar.gz"
        if alternate.is_file():
            source_tar = alternate
        else:
            raise FileNotFoundError(source_tar)
    target = base / "source"
    if not target.is_dir():
        with tarfile.open(source_tar, "r:gz") as tf:
            tf.extractall(target)
    return target


def compare_arrays(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        return {"shape_match": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    diff = np.abs(a - b)
    return {
        "shape_match": True,
        "shape": list(a.shape),
        "mean_abs": float(diff.mean(dtype=np.float64)),
        "max_abs": float(diff.max(initial=0.0)),
        "nonzero": int(np.count_nonzero(diff)),
    }


def compare_saved_source_outputs(
    *,
    branch: str,
    arrays: dict[str, np.ndarray],
    optional_source: dict[str, np.ndarray],
    step_index: int,
    sample_in_np: np.ndarray,
    t: float,
    t_prev: float,
) -> dict[str, Any]:
    source_key = "pred_pos" if branch == "pos" else "pred_neg"
    compare_report: dict[str, Any] = {}
    if source_key in optional_source:
        compare_report[f"final_output_vs_source_steps_{source_key}"] = compare_arrays(
            arrays[f"{branch}_final_output"],
            optional_source[source_key][step_index],
        )
    if "sample_next" in optional_source:
        pred = arrays[f"{branch}_final_output"]
        sample_next = sample_in_np - (t - t_prev) * pred
        arrays[f"{branch}_branch_only_sample_next_from_pred"] = sample_next.astype(np.float32)
        branch_only_report = compare_arrays(
            arrays[f"{branch}_branch_only_sample_next_from_pred"],
            optional_source["sample_next"][step_index],
        )
        branch_only_report.update(
            {
                "comparison_class": "branch_only_euler_vs_saved_cfg_or_scheduler_sample_next",
                "route_identity_evidence": False,
                "reason": (
                    "The saved source sample_next may include CFG/rescale or scheduler state; "
                    "a single branch pred output is not an equivalent route."
                ),
            }
        )
        compare_report["branch_only_sample_next_vs_source_steps_sample_next"] = branch_only_report
    return compare_report


def _to_numpy(tensor: Any, *, squeeze_batch: bool = True) -> np.ndarray:
    value = tensor.detach().float().cpu().numpy()
    if squeeze_batch and value.ndim > 0 and value.shape[0] == 1:
        value = value[0]
    return value.astype(np.float32, copy=False)


def _linear_weight_bias(linear: Any) -> tuple[np.ndarray, np.ndarray | None]:
    weight = linear.weight.detach().float().cpu().numpy()
    bias = None
    if getattr(linear, "bias", None) is not None:
        bias = linear.bias.detach().float().cpu().numpy()
    return weight.astype(np.float32, copy=False), None if bias is None else bias.astype(np.float32, copy=False)


def split_block_modulation(block: Any, mod: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
    if getattr(block, "share_mod", False):
        return (block.modulation + mod).type(mod.dtype).chunk(6, dim=1)
    return block.adaLN_modulation(mod).chunk(6, dim=1)


def _source_mlp_linears(mlp: Any) -> tuple[Any, Any]:
    if hasattr(mlp, "mlp"):
        return mlp.mlp[0], mlp.mlp[2]
    return mlp.mlp_0, mlp.mlp_2


def _trace_block(torch: Any, model: Any, block: Any, x: Any, mod: Any, context: Any) -> tuple[Any, dict[str, np.ndarray]]:
    from trellis2.modules.attention import RotaryPositionEmbedder
    from trellis2.modules.attention.full_attn import scaled_dot_product_attention

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = split_block_modulation(block, mod)
    source: dict[str, np.ndarray] = {
        "input": _to_numpy(x),
        "shift_msa": _to_numpy(shift_msa),
        "scale_msa": _to_numpy(scale_msa),
        "gate_msa": _to_numpy(gate_msa),
        "shift_mlp": _to_numpy(shift_mlp),
        "scale_mlp": _to_numpy(scale_mlp),
        "gate_mlp": _to_numpy(gate_mlp),
    }

    h = block.norm1(x)
    source["norm1"] = _to_numpy(h)
    h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
    source["modulated_self_input"] = _to_numpy(h)

    attn = block.self_attn
    qkv = attn.to_qkv(h)
    qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, attn.num_heads, -1)
    q, k, v = qkv.unbind(dim=2)
    source["q_pre_norm"] = _to_numpy(q)
    source["k_pre_norm"] = _to_numpy(k)
    source["v"] = _to_numpy(v)
    if getattr(attn, "qk_rms_norm", False):
        q = attn.q_rms_norm(q)
        k = attn.k_rms_norm(k)
    source["q_post_norm"] = _to_numpy(q)
    source["k_post_norm"] = _to_numpy(k)
    if getattr(attn, "use_rope", False):
        q = RotaryPositionEmbedder.apply_rotary_embedding(q, model.rope_phases)
        k = RotaryPositionEmbedder.apply_rotary_embedding(k, model.rope_phases)
    source["q_post_rope"] = _to_numpy(q)
    source["k_post_rope"] = _to_numpy(k)
    raw = scaled_dot_product_attention(q, k, v)
    raw_flat = raw.reshape(h.shape[0], h.shape[1], -1)
    source["attention_raw"] = _to_numpy(raw_flat)
    h = attn.to_out(raw_flat)
    source["self_attn"] = _to_numpy(h)
    h = h * gate_msa.unsqueeze(1)
    x = x + h
    source["after_self"] = _to_numpy(x)

    h = block.norm2(x)
    source["norm2"] = _to_numpy(h)
    attn = block.cross_attn
    q = attn.to_q(h)
    kv = attn.to_kv(context)
    q = q.reshape(q.shape[0], q.shape[1], attn.num_heads, -1)
    kv = kv.reshape(kv.shape[0], kv.shape[1], 2, attn.num_heads, -1)
    k, v = kv.unbind(dim=2)
    source["cross_q_pre_norm"] = _to_numpy(q)
    if getattr(attn, "qk_rms_norm", False):
        q = attn.q_rms_norm(q)
        k = attn.k_rms_norm(k)
    source["cross_q_post_norm"] = _to_numpy(q)
    source["cross_k_post_norm"] = _to_numpy(k)
    source["cross_v"] = _to_numpy(v)
    raw = scaled_dot_product_attention(q, k, v)
    raw_flat = raw.reshape(h.shape[0], h.shape[1], -1)
    source["cross_attention_raw"] = _to_numpy(raw_flat)
    h = attn.to_out(raw_flat)
    source["cross_attn"] = _to_numpy(h)
    x = x + h
    source["after_cross"] = _to_numpy(x)

    h = block.norm3(x)
    h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
    source["mlp_input"] = _to_numpy(h)
    fc1, fc2 = _source_mlp_linears(block.mlp)
    h_fc1 = fc1(h)
    source["mlp_fc1"] = _to_numpy(h_fc1)
    gelu = block.mlp.mlp[1] if hasattr(block.mlp, "mlp") else torch.nn.GELU(approximate="tanh")
    h_gelu = gelu(h_fc1)
    source["mlp_gelu"] = _to_numpy(h_gelu)
    h_fc2 = fc2(h_gelu)
    source["mlp_fc2"] = _to_numpy(h_fc2)
    source["mlp"] = source["mlp_fc2"]
    h = h_fc2 * gate_mlp.unsqueeze(1)
    source["mlp_gated"] = _to_numpy(h)
    x = x + h
    source["after_mlp"] = _to_numpy(x)
    return x, source


def _load_source_model(torch: Any, checkpoint: Path, *, device: Any) -> Any:
    from safetensors.torch import load_file

    try:
        from trellis2.models.sparse_structure_flow import SparseStructureFlowModel
    except ImportError:
        from trellis2.models import SparseStructureFlowModel

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
    return model.to(device).eval()


def _load_step_input(source_steps_path: Path, *, step_index: int) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    with np.load(source_steps_path) as source_steps:
        arrays = sorted(source_steps.files)
        if "sample_in" not in source_steps:
            raise KeyError("source steps missing required array 'sample_in'")
        sample_in = np.asarray(source_steps["sample_in"], dtype=np.float32)
        if sample_in.ndim != 6:
            raise ValueError(f"source sample_in must be [S,B,C,Z,Y,X], got {sample_in.shape}")
        if step_index < 0 or step_index >= sample_in.shape[0]:
            raise ValueError(f"step_index={step_index} outside sample_in steps={sample_in.shape[0]}")
        optional: dict[str, np.ndarray] = {}
        for name in ("pred_pos", "pred_neg", "sample_next"):
            if name in source_steps:
                optional[name] = np.asarray(source_steps[name], dtype=np.float32)
        return np.asarray(sample_in[step_index], dtype=np.float32), arrays, optional


def _load_conditioning(conditioning_path: Path, *, branch: str) -> tuple[np.ndarray, dict[str, Any]]:
    key = select_branch_conditioning_key(branch)
    with np.load(conditioning_path) as conditioning:
        arrays = sorted(conditioning.files)
        if key not in conditioning:
            raise KeyError(f"conditioning missing required branch key {key!r}")
        cond = np.asarray(conditioning[key], dtype=np.float32)
    return cond, {"path": str(conditioning_path), "arrays": arrays, "selected_key": key, "shape": list(cond.shape)}


def _write_json(path: Path, report: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--source-steps", default=Path("source_cuda_steps.npz"), type=Path)
    parser.add_argument("--conditioning", default=Path("conditioning.npz"), type=Path)
    parser.add_argument("--checkpoint", default=Path("ss_flow_img_dit_1_3B_64_bf16.safetensors"), type=Path)
    parser.add_argument("--source-tar", default=Path("trellis2_source_tarball.bin"), type=Path)
    parser.add_argument("--branch", choices=("pos", "neg"), default="pos")
    parser.add_argument("--step-index", type=int, default=2)
    parser.add_argument("--block-indices", default="4")
    parser.add_argument(
        "--trace-names",
        default="all",
        help="trace tensor names to write, or aliases: all, compact",
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--rescale-t", type=float, default=5.0)
    parser.add_argument(
        "--sparse-conv-backend",
        default="none",
        choices=("none", "spconv", "torchsparse", "flex_gemm"),
    )
    parser.add_argument(
        "--sparse-attn-backend",
        default="sdpa",
        choices=("xformers", "flash_attn", "flash_attn_3", "sdpa", "naive"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    phase = "setup"
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "failure_phase": None,
        "primary_output_status": "not_written",
        "phases": [],
    }
    route_identity: dict[str, Any] = {
        "requested_route": REQUESTED_ROUTE,
        "effective_route": "not-established",
        "source_steps": _jsonable_path(args.source_steps),
        "conditioning": _jsonable_path(args.conditioning),
        "checkpoint": _jsonable_path(args.checkpoint),
        "branch": args.branch,
        "step_index": int(args.step_index),
    }
    try:
        output_json = Path(args.output_json)
        output_npz = Path(args.output_npz)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_npz.parent.mkdir(parents=True, exist_ok=True)

        phase = "validate_args"
        block_indices = parse_block_indices(args.block_indices)
        trace_names = parse_trace_names(args.trace_names)
        if args.steps <= 0:
            raise ValueError("--steps must be positive")
        if args.step_index < 0 or args.step_index >= args.steps:
            raise ValueError(f"--step-index {args.step_index} outside --steps {args.steps}")
        forced_env = apply_sparse_backend_env(args.sparse_conv_backend, args.sparse_attn_backend)
        t, t_prev = schedule_pairs(args.steps, args.rescale_t)[args.step_index]
        report["request"] = {
            "branch": args.branch,
            "step_index": int(args.step_index),
            "block_indices": [int(v) for v in block_indices],
            "trace_names": list(trace_names),
            "steps": int(args.steps),
            "rescale_t": float(args.rescale_t),
            "t": float(t),
            "t_prev": float(t_prev),
            "t_model": float(1000 * t),
            "requested_sparse_backend": forced_env,
        }
        report["phases"].append(phase)

        phase = "extract_source"
        source_root = extract_source(args.source_tar, Path.cwd())
        sys.path.insert(0, str(source_root))
        report["source_root"] = str(source_root)
        report["phases"].append(phase)

        phase = "import_runtime"
        import torch
        from trellis2.modules.utils import manual_cast
        from trellis2.modules.attention import config as attention_config
        from trellis2.modules.sparse import config as sparse_config

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.set_grad_enabled(False)
        attention_config.BACKEND = args.sparse_attn_backend
        device = torch.device("cuda")
        report.update(
            {
                "torch": torch.__version__,
                "cuda_device": torch.cuda.get_device_name(0),
                "attention_backend": getattr(attention_config, "BACKEND", None),
                "sparse_attention_backend": getattr(sparse_config, "ATTN", None),
                "sparse_conv_backend": getattr(sparse_config, "CONV", None),
            }
        )
        report["phases"].append(phase)

        phase = "load_inputs"
        sample_in_np, source_arrays, optional_source = _load_step_input(
            args.source_steps,
            step_index=args.step_index,
        )
        cond_np, conditioning_identity = _load_conditioning(args.conditioning, branch=args.branch)
        report["inputs"] = {
            "source_steps": str(args.source_steps),
            "source_arrays": source_arrays,
            "source_sample_in_shape": list(sample_in_np.shape),
            "conditioning": conditioning_identity,
        }
        report["phases"].append(phase)

        phase = "load_model"
        model = _load_source_model(torch, args.checkpoint, device=device)
        effective_device_type = next(model.parameters()).device.type
        route_identity = build_route_identity(
            effective_device_type=effective_device_type,
            source_steps=args.source_steps,
            conditioning=args.conditioning,
            checkpoint=args.checkpoint,
            source_root=source_root,
            branch=args.branch,
            step_index=args.step_index,
            block_indices=block_indices,
            trace_names=trace_names,
            sparse_conv_backend=args.sparse_conv_backend,
            sparse_attn_backend=args.sparse_attn_backend,
        )
        route_identity["source_state_dict_load"] = getattr(model, "_trellis2mlx_state_dict_load", None)
        report["route_identity"] = route_identity
        report["parameter_count"] = int(sum(parameter.numel() for parameter in model.parameters()))
        report["phases"].append(phase)

        phase = "trace_forward"
        arrays: dict[str, np.ndarray] = {
            "route_identity_json": np.asarray(json.dumps(route_identity, sort_keys=True)),
            "trace_block_indices": np.asarray(block_indices, dtype=np.int32),
            "trace_step_index": np.asarray(args.step_index, dtype=np.int32),
            "trace_t": np.asarray(t, dtype=np.float32),
            "trace_t_model": np.asarray(1000 * t, dtype=np.float32),
            "sample_in": sample_in_np.astype(np.float32, copy=False),
        }
        sample = torch.from_numpy(sample_in_np).to(device=device, dtype=torch.float32)
        cond = torch.from_numpy(cond_np).to(device=device, dtype=torch.float32)
        t_tensor = torch.tensor([1000 * t] * sample.shape[0], device=device, dtype=torch.float32)
        with torch.inference_mode():
            h = sample.view(*sample.shape[:2], -1).permute(0, 2, 1).contiguous()
            h = model.input_layer(h)
            if model.pe_mode == "ape":
                h = h + model.pos_emb[None]
            t_emb = model.t_embedder(t_tensor)
            if model.share_mod:
                t_emb = model.adaLN_modulation(t_emb)
            t_emb = manual_cast(t_emb, model.dtype)
            h = manual_cast(h, model.dtype)
            cond = manual_cast(cond, model.dtype)
            arrays["post_input_layer"] = _to_numpy(h)
            arrays["t_emb"] = _to_numpy(t_emb)
            for block_index, block in enumerate(model.blocks):
                if block_index in block_indices:
                    h, block_trace = _trace_block(torch, model, block, h, t_emb, cond)
                    for name in trace_names:
                        if name in block_trace:
                            arrays[f"{args.branch}_block{block_index}_{name}"] = block_trace[name]
                else:
                    h = block(h, t_emb, cond, model.rope_phases)
            h = manual_cast(h, sample.dtype)
            final_norm = torch.nn.functional.layer_norm(h, h.shape[-1:])
            final_flat = model.out_layer(final_norm)
            final_output = final_flat.permute(0, 2, 1).view(
                sample.shape[0],
                model.out_channels,
                model.resolution,
                model.resolution,
                model.resolution,
            )
        arrays[f"{args.branch}_final_norm"] = _to_numpy(final_norm)
        arrays[f"{args.branch}_final_out_flat"] = _to_numpy(final_flat)
        arrays[f"{args.branch}_final_output"] = final_output.detach().float().cpu().numpy().astype(np.float32)
        report["phases"].append(phase)

        phase = "compare_saved_source"
        compare_report = compare_saved_source_outputs(
            branch=args.branch,
            arrays=arrays,
            optional_source=optional_source,
            step_index=args.step_index,
            sample_in_np=sample_in_np,
            t=t,
            t_prev=t_prev,
        )
        report["saved_source_comparison"] = compare_report
        report["phases"].append(phase)

        phase = "write_outputs"
        np.savez_compressed(output_npz, **arrays)
        report.update(
            {
                "status": "ok",
                "failure_phase": None,
                "last_trustworthy_phase": phase,
                "primary_output_status": "trace_npz_written",
                "elapsed_seconds": time.perf_counter() - started,
                "output_npz": str(output_npz),
                "output_npz_sha256": _sha256(output_npz),
                "output_arrays": {name: list(value.shape) for name, value in arrays.items()},
            }
        )
        report["phases"].append(phase)
        _write_json(output_json, report)
        return 0
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "failure_phase": phase,
                "last_trustworthy_phase": report["phases"][-1] if report["phases"] else None,
                "primary_output_status": "not_written",
                "route_identity": route_identity,
                "elapsed_seconds": time.perf_counter() - started,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        _write_json(args.output_json, report)
        print(f"source_cuda_sparse_block_trace failed in {phase}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
