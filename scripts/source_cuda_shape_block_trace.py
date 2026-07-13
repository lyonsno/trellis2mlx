#!/usr/bin/env python3
"""Trace official TRELLIS.2 source-CUDA shape-flow block internals.

This witness starts from a source support set plus an exact shape-flow noise
sample, runs the official TRELLIS.2 shape SLat flow model on CUDA, and records
selected block tensors with key names comparable to the MLX
``shape_flow_block_trace`` artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import time
from typing import Any, Iterable

import numpy as np


SCHEMA = "trellis2mlx.source_cuda_shape_block_trace.v1"
REQUESTED_ROUTE = "source-cuda-shape-flow-block-trace"
EFFECTIVE_ROUTE = "official-trellis2-source-cuda-shape-flow-block-trace"
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
    "cross_k_pre_norm",
    "cross_v",
    "cross_q_post_norm",
    "cross_k_post_norm",
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
COMPACT_TRACE_NAMES = ("input", "attention_raw", "after_self", "cross_attention_raw", "after_cross", "after_mlp")


@dataclass(frozen=True)
class SupportAndNoise:
    coords: np.ndarray
    coords_3d: np.ndarray
    noise: np.ndarray
    noise_key: str


@dataclass(frozen=True)
class BlockInputReplay:
    arrays: dict[tuple[str, int], np.ndarray]
    scope: list[str]


def _jsonable_path(path: Path) -> str:
    return str(Path(path).expanduser())


def _sha256_file(path: Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_trace_names(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return list(TRACE_NAMES)
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value == "all":
            return list(TRACE_NAMES)
        if value == "compact":
            return list(COMPACT_TRACE_NAMES)
        names = [part.strip() for part in value.split(",") if part.strip()]
    else:
        names = [str(part).strip() for part in value if str(part).strip()]
    if not names:
        raise ValueError("--trace-names must contain at least one trace name")
    unknown = sorted(set(names) - set(TRACE_NAMES))
    if unknown:
        raise ValueError(f"unknown trace name(s): {', '.join(unknown)}")
    return names


def parse_block_indices(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        indices = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        indices = [int(part) for part in value]
    if not indices:
        raise ValueError("--block-indices must contain at least one integer")
    if len(set(indices)) != len(indices):
        raise ValueError(f"--block-indices contains duplicates: {indices!r}")
    for index in indices:
        if index < 0:
            raise ValueError(f"block index must be non-negative, got {index}")
    return indices


def build_route_identity(
    *,
    device_type: str,
    output_npz: Path,
    conditioning_path: Path,
    support_sample_path: Path,
    noise_sample_path: Path,
    block_indices: Iterable[int],
    trace_names: Iterable[str],
    steps: int,
    seed: int,
    branch: str,
    source_tar: Path | None = None,
    model_repo: str | None = None,
    pipeline_config: str | None = None,
    block_input_replay_sample: Path | None = None,
    block_input_replay_scope: Iterable[str] | None = None,
) -> dict[str, Any]:
    if device_type != "cuda":
        raise RuntimeError(
            "source_cuda_shape_block_trace is CUDA-only evidence; "
            f"refusing effective device {device_type!r}"
        )
    replay_scope = [str(value) for value in (block_input_replay_scope or [])]
    is_replay = block_input_replay_sample is not None
    forbidden_inferences = [
        "not Trellis-Mac/MPS evidence",
        "not final GLB parity evidence",
        "not texture/finalization evidence unless downstream outputs are listed",
    ]
    if is_replay:
        forbidden_inferences.append(
            "not evidence that source-CUDA upstream blocks before the replay point match MLX"
        )
    else:
        forbidden_inferences.append("not a captured-MLX-block-input replay")
    return {
        "schema": SCHEMA,
        "requested_route": REQUESTED_ROUTE,
        "effective_route": (
            f"{EFFECTIVE_ROUTE}-with-captured-block-input-replay" if is_replay else EFFECTIVE_ROUTE
        ),
        "backend": "source-trellis",
        "device": "cuda",
        "effective_device_type": "cuda",
        "primary_output": _jsonable_path(output_npz),
        "conditioning_sample": _jsonable_path(conditioning_path),
        "conditioning_sha256": _sha256_file(conditioning_path),
        "shape_slat_support_sample": _jsonable_path(support_sample_path),
        "shape_slat_support_sample_sha256": _sha256_file(support_sample_path),
        "shape_flow_noise_sample": _jsonable_path(noise_sample_path),
        "shape_flow_noise_sample_sha256": _sha256_file(noise_sample_path),
        "source_tar": None if source_tar is None else _jsonable_path(source_tar),
        "source_tar_sha256": None if source_tar is None else _sha256_file(source_tar),
        "block_input_replay_sample": (
            None if block_input_replay_sample is None else _jsonable_path(block_input_replay_sample)
        ),
        "block_input_replay_sample_sha256": (
            None if block_input_replay_sample is None else _sha256_file(block_input_replay_sample)
        ),
        "block_input_replay_scope": replay_scope,
        "model_repo": model_repo,
        "pipeline_config": pipeline_config,
        "shape_flow_trace_block_indices": [int(v) for v in block_indices],
        "trace_names": [str(v) for v in trace_names],
        "steps": int(steps),
        "seed": int(seed),
        "branch": branch,
        "forbidden_inferences": forbidden_inferences,
    }


def load_support_and_noise(support_sample_path: Path, noise_sample_path: Path) -> SupportAndNoise:
    with np.load(support_sample_path) as support:
        if "coords" not in support:
            raise ValueError(f"support sample {support_sample_path} missing coords")
        coords = np.asarray(support["coords"], dtype=np.int32)
        if "coords_3d" in support:
            coords_3d = np.asarray(support["coords_3d"], dtype=np.int32)
        else:
            coords_3d = coords[:, 1:].astype(np.int32, copy=False)
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"support coords must have shape [N, 4], got {coords.shape}")
    if coords_3d.shape != (coords.shape[0], 3):
        raise ValueError(
            f"support coords_3d must have shape {(coords.shape[0], 3)}, got {coords_3d.shape}"
        )

    with np.load(noise_sample_path) as noise_data:
        if "coords" not in noise_data:
            raise ValueError(f"shape flow noise sample {noise_sample_path} missing coords")
        noise_coords = np.asarray(noise_data["coords"], dtype=np.int32)
        if "noise" in noise_data:
            noise_key = "noise"
        elif "sample_feats" in noise_data:
            noise_key = "sample_feats"
        elif "latents" in noise_data:
            noise_key = "latents"
        else:
            raise ValueError(
                f"shape flow noise sample {noise_sample_path} missing noise/sample_feats/latents"
            )
        noise = np.asarray(noise_data[noise_key], dtype=np.float32)

    if noise_coords.shape != coords.shape or not np.array_equal(noise_coords, coords):
        raise ValueError(
            "shape flow noise sample coordinates do not exactly match support coordinates: "
            f"noise {noise_coords.shape}, support {coords.shape}"
        )
    if noise.ndim != 2:
        raise ValueError(f"shape flow noise must have shape [N, C], got {noise.shape}")
    if noise.shape[0] != coords.shape[0]:
        raise ValueError(
            "shape flow noise/support row mismatch: "
            f"{noise.shape[0]} noise rows vs {coords.shape[0]} coords"
        )
    return SupportAndNoise(
        coords=np.ascontiguousarray(coords),
        coords_3d=np.ascontiguousarray(coords_3d),
        noise=np.ascontiguousarray(noise),
        noise_key=noise_key,
    )


def load_block_input_replay(
    replay_sample_path: Path,
    *,
    branches: Iterable[str],
    block_indices: Iterable[int],
    token_count: int,
) -> BlockInputReplay:
    replay_sample_path = Path(replay_sample_path)
    arrays: dict[tuple[str, int], np.ndarray] = {}
    scope: list[str] = []
    with np.load(replay_sample_path) as data:
        available = set(data.files)
        for branch in branches:
            for block_index in block_indices:
                key = f"{branch}_block{int(block_index)}_input"
                if key not in available:
                    raise ValueError(f"block input replay sample {replay_sample_path} missing {key}")
                value = np.asarray(data[key], dtype=np.float32)
                if value.ndim == 3 and value.shape[0] == 1:
                    value = value[0]
                if value.ndim != 2:
                    raise ValueError(
                        f"{key} must have shape [1, N, C] or [N, C], got {value.shape}"
                    )
                if value.shape[0] != token_count:
                    raise ValueError(
                        f"{key} token count mismatch: expected {token_count}, got {value.shape[0]}"
                    )
                arrays[(str(branch), int(block_index))] = np.ascontiguousarray(value)
                scope.append(key)
    return BlockInputReplay(arrays=arrays, scope=scope)


def schedule_pairs(steps: int, rescale_t: float) -> list[tuple[float, float]]:
    if steps <= 0:
        raise ValueError("--steps must be positive")
    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    return [(float(t_seq[i]), float(t_seq[i + 1])) for i in range(steps)]


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
        with tarfile.open(source_tar, "r:gz") as archive:
            archive.extractall(target)
    return target


def resolve_model_ref(model_repo: str, model_spec: str) -> str:
    if model_spec.startswith("ckpts/"):
        return f"{model_repo}/{model_spec}"
    return model_spec


def apply_sparse_backend_env(conv_backend: str, attn_backend: str) -> dict[str, str]:
    os.environ["SPARSE_CONV_BACKEND"] = conv_backend
    os.environ["SPARSE_ATTN_BACKEND"] = attn_backend
    os.environ["ATTN_BACKEND"] = attn_backend
    return {
        "SPARSE_CONV_BACKEND": conv_backend,
        "SPARSE_ATTN_BACKEND": attn_backend,
        "ATTN_BACKEND": attn_backend,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _tensor_to_numpy(value: Any, *, batched_sparse: bool = False, batched_context: bool = False) -> np.ndarray:
    if hasattr(value, "feats"):
        tensor = value.feats
        arr = tensor.detach().float().cpu().numpy()
        if batched_sparse:
            arr = arr[None, ...]
    else:
        arr = value.detach().float().cpu().numpy()
        if batched_context and arr.ndim == 4:
            arr = arr[:, None, ...]
    return arr.astype(np.float32, copy=False)


def split_block_modulation(block: Any, mod: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
    if getattr(block, "share_mod", False):
        return (block.modulation + mod).type(mod.dtype).chunk(6, dim=1)
    return block.adaLN_modulation(mod).chunk(6, dim=1)


def _source_mlp_linears(mlp: Any) -> tuple[Any, Any]:
    if hasattr(mlp, "mlp"):
        return mlp.mlp[0], mlp.mlp[2]
    return mlp.mlp_0, mlp.mlp_2


def _source_mlp_gelu(torch: Any, mlp: Any) -> Any:
    if hasattr(mlp, "mlp") and len(mlp.mlp) > 1:
        return mlp.mlp[1]
    return torch.nn.GELU(approximate="tanh")


def _trace_shape_block(torch: Any, block: Any, x: Any, mod: Any, context: Any) -> tuple[Any, dict[str, np.ndarray]]:
    from trellis2.modules.sparse.attention.full_attn import sparse_scaled_dot_product_attention

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = split_block_modulation(block, mod)
    source: dict[str, np.ndarray] = {
        "input": _tensor_to_numpy(x, batched_sparse=True),
        "shift_msa": _tensor_to_numpy(shift_msa),
        "scale_msa": _tensor_to_numpy(scale_msa),
        "gate_msa": _tensor_to_numpy(gate_msa),
        "shift_mlp": _tensor_to_numpy(shift_mlp),
        "scale_mlp": _tensor_to_numpy(scale_mlp),
        "gate_mlp": _tensor_to_numpy(gate_mlp),
    }

    h = x.replace(block.norm1(x.feats))
    source["norm1"] = _tensor_to_numpy(h, batched_sparse=True)
    h = h * (1 + scale_msa) + shift_msa
    source["modulated_self_input"] = _tensor_to_numpy(h, batched_sparse=True)

    attn = block.self_attn
    qkv = attn._linear(attn.to_qkv, h)
    qkv = attn._fused_pre(qkv, num_fused=3)
    q, k, v = qkv.unbind(dim=-3)
    source["q_pre_norm"] = _tensor_to_numpy(q, batched_sparse=True)
    source["k_pre_norm"] = _tensor_to_numpy(k, batched_sparse=True)
    source["v"] = _tensor_to_numpy(v, batched_sparse=True)
    if getattr(attn, "qk_rms_norm", False):
        q = attn.q_rms_norm(q)
        k = attn.k_rms_norm(k)
    source["q_post_norm"] = _tensor_to_numpy(q, batched_sparse=True)
    source["k_post_norm"] = _tensor_to_numpy(k, batched_sparse=True)
    if getattr(attn, "use_rope", False):
        q, k = attn.rope(q, k)
    source["q_post_rope"] = _tensor_to_numpy(q, batched_sparse=True)
    source["k_post_rope"] = _tensor_to_numpy(k, batched_sparse=True)
    qkv = qkv.replace(torch.stack([q.feats, k.feats, v.feats], dim=1))
    h = sparse_scaled_dot_product_attention(qkv)
    source["attention_raw"] = _tensor_to_numpy(h, batched_sparse=True)
    h = attn._reshape_chs(h, (-1,))
    h = attn._linear(attn.to_out, h)
    source["self_attn"] = _tensor_to_numpy(h, batched_sparse=True)
    h = h * gate_msa
    x = x + h
    source["after_self"] = _tensor_to_numpy(x, batched_sparse=True)

    h = x.replace(block.norm2(x.feats))
    source["norm2"] = _tensor_to_numpy(h, batched_sparse=True)
    attn = block.cross_attn
    q = attn._linear(attn.to_q, h)
    q = attn._reshape_chs(q, (attn.num_heads, -1))
    kv = attn._linear(attn.to_kv, context)
    kv = attn._fused_pre(kv, num_fused=2)
    k, v = kv.unbind(dim=-3)
    source["cross_q_pre_norm"] = _tensor_to_numpy(q, batched_sparse=True)
    source["cross_k_pre_norm"] = _tensor_to_numpy(k, batched_context=True)
    source["cross_v"] = _tensor_to_numpy(v, batched_context=True)
    if getattr(attn, "qk_rms_norm", False):
        q = attn.q_rms_norm(q)
        k = attn.k_rms_norm(k)
    source["cross_q_post_norm"] = _tensor_to_numpy(q, batched_sparse=True)
    source["cross_k_post_norm"] = _tensor_to_numpy(k, batched_context=True)
    h = sparse_scaled_dot_product_attention(q, k, v)
    source["cross_attention_raw"] = _tensor_to_numpy(h, batched_sparse=True)
    h = attn._reshape_chs(h, (-1,))
    h = attn._linear(attn.to_out, h)
    source["cross_attn"] = _tensor_to_numpy(h, batched_sparse=True)
    x = x + h
    source["after_cross"] = _tensor_to_numpy(x, batched_sparse=True)

    h = x.replace(block.norm3(x.feats))
    h = h * (1 + scale_mlp) + shift_mlp
    source["mlp_input"] = _tensor_to_numpy(h, batched_sparse=True)
    fc1, fc2 = _source_mlp_linears(block.mlp)
    h_fc1 = fc1(h)
    source["mlp_fc1"] = _tensor_to_numpy(h_fc1, batched_sparse=True)
    h_gelu = _source_mlp_gelu(torch, block.mlp)(h_fc1)
    source["mlp_gelu"] = _tensor_to_numpy(h_gelu, batched_sparse=True)
    h_fc2 = fc2(h_gelu)
    source["mlp_fc2"] = _tensor_to_numpy(h_fc2, batched_sparse=True)
    source["mlp"] = source["mlp_fc2"]
    h = h_fc2 * gate_mlp
    source["mlp_gated"] = _tensor_to_numpy(h, batched_sparse=True)
    x = x + h
    source["after_mlp"] = _tensor_to_numpy(x, batched_sparse=True)
    return x, source


def _select_branches(branch: str) -> list[tuple[str, str]]:
    if branch == "both":
        return [("pos", "cond"), ("neg", "neg_cond")]
    if branch == "pos":
        return [("pos", "cond")]
    if branch == "neg":
        return [("neg", "neg_cond")]
    raise ValueError(f"unsupported branch {branch!r}")


def _load_conditioning(path: Path, torch: Any, device: Any) -> dict[str, Any]:
    with np.load(path) as data:
        if "cond" not in data or "neg_cond" not in data:
            raise ValueError(f"conditioning sample {path} must contain cond and neg_cond")
        return {
            "cond": torch.from_numpy(np.asarray(data["cond"], dtype=np.float32)).to(device=device),
            "neg_cond": torch.from_numpy(np.asarray(data["neg_cond"], dtype=np.float32)).to(device=device),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--conditioning", required=True, type=Path)
    parser.add_argument("--shape-slat-support-sample", required=True, type=Path)
    parser.add_argument("--shape-flow-noise-sample", required=True, type=Path)
    parser.add_argument(
        "--block-input-replay-sample",
        type=Path,
        help=(
            "Optional NPZ with {branch}_block{index}_input arrays. When set, the "
            "source-CUDA trace replaces h at each requested block before tracing it."
        ),
    )
    parser.add_argument("--source-tar", default=Path("trellis2_source_tarball.bin"), type=Path)
    parser.add_argument("--model-repo", default="microsoft/TRELLIS.2-4B")
    parser.add_argument("--pipeline-config", default="pipeline.json")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--branch", choices=("pos", "neg", "both"), default="both")
    parser.add_argument("--block-indices", default="0")
    parser.add_argument("--shape-flow-trace-step-index", type=int, default=0)
    parser.add_argument("--trace-names", default="all")
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
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Validate local inputs only; used by failure-smoke tests and offline diagnostics.",
    )
    return parser


def _failure_payload(
    *,
    report: dict[str, Any],
    route_identity: dict[str, Any],
    phase: str,
    last_trustworthy_phase: str | None,
    started: float,
    error: BaseException,
    output_npz: Path,
) -> dict[str, Any]:
    payload = {
        **report,
        "status": "failed",
        "failure_phase": phase,
        "last_trustworthy_phase": last_trustworthy_phase,
        "primary_output_status": "written" if Path(output_npz).exists() else "missing",
        "route_identity": route_identity,
        "elapsed_seconds": time.perf_counter() - started,
        "error": f"{type(error).__name__}: {error}",
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    phase = "arguments_parsed"
    last_trustworthy_phase: str | None = "arguments_parsed"
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "primary_output_status": "not_written",
        "failure_phase": None,
        "last_trustworthy_phase": last_trustworthy_phase,
        "phase_timings": {},
    }
    route_identity: dict[str, Any] = {
        "requested_route": REQUESTED_ROUTE,
        "effective_route": "not-established",
        "primary_output": _jsonable_path(args.output_npz),
        "conditioning_sample": _jsonable_path(args.conditioning),
        "shape_slat_support_sample": _jsonable_path(args.shape_slat_support_sample),
        "shape_flow_noise_sample": _jsonable_path(args.shape_flow_noise_sample),
        "block_input_replay_sample": (
            None
            if args.block_input_replay_sample is None
            else _jsonable_path(args.block_input_replay_sample)
        ),
        "branch": args.branch,
    }

    try:
        output_json = Path(args.output_json)
        output_npz = Path(args.output_npz)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_npz.parent.mkdir(parents=True, exist_ok=True)

        phase = "input_validation"
        phase_started = time.perf_counter()
        block_indices = parse_block_indices(args.block_indices)
        trace_names = parse_trace_names(args.trace_names)
        if args.shape_flow_trace_step_index < 0 or args.shape_flow_trace_step_index >= args.steps:
            raise ValueError(
                f"shape flow trace step {args.shape_flow_trace_step_index} outside steps={args.steps}"
            )
        support_noise = load_support_and_noise(
            args.shape_slat_support_sample,
            args.shape_flow_noise_sample,
        )
        branch_names = [branch_name for branch_name, _ in _select_branches(args.branch)]
        block_input_replay = (
            load_block_input_replay(
                args.block_input_replay_sample,
                branches=branch_names,
                block_indices=block_indices,
                token_count=support_noise.coords.shape[0],
            )
            if args.block_input_replay_sample is not None
            else None
        )
        report["input_identity"] = {
            "coords_shape": [int(v) for v in support_noise.coords.shape],
            "coords_3d_shape": [int(v) for v in support_noise.coords_3d.shape],
            "noise_shape": [int(v) for v in support_noise.noise.shape],
            "noise_key": support_noise.noise_key,
            "block_input_replay_scope": [] if block_input_replay is None else block_input_replay.scope,
        }
        report["requested_sparse_backend"] = apply_sparse_backend_env(
            args.sparse_conv_backend,
            args.sparse_attn_backend,
        )
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        last_trustworthy_phase = phase

        phase = "extract_source"
        phase_started = time.perf_counter()
        source_root = extract_source(args.source_tar, Path.cwd())
        sys.path.insert(0, str(source_root))
        report["source_root"] = str(source_root)
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        last_trustworthy_phase = phase

        phase = "import_runtime"
        phase_started = time.perf_counter()
        import torch
        from huggingface_hub import hf_hub_download
        from trellis2 import models as source_models
        from trellis2.modules.sparse import SparseTensor
        from trellis2.modules.sparse import config as sparse_config
        from trellis2.modules.utils import manual_cast

        if args.no_download:
            raise RuntimeError("--no-download stops before model/config download by request")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.set_grad_enabled(False)
        device = torch.device("cuda")
        report.update(
            {
                "torch": torch.__version__,
                "cuda_device": torch.cuda.get_device_name(0),
                "sparse_attention_backend": getattr(sparse_config, "ATTN", None),
                "sparse_conv_backend": getattr(sparse_config, "CONV", None),
            }
        )
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        last_trustworthy_phase = phase

        phase = "load_pipeline_config"
        phase_started = time.perf_counter()
        pipeline_config_path = Path(hf_hub_download(args.model_repo, args.pipeline_config))
        with pipeline_config_path.open() as handle:
            pipeline_args = json.load(handle)["args"]
        sampler_params = dict(pipeline_args["shape_slat_sampler"]["params"])
        sampler_params["steps"] = int(args.steps)
        rescale_t = float(sampler_params.get("rescale_t", 1.0))
        t, t_prev = schedule_pairs(args.steps, rescale_t)[args.shape_flow_trace_step_index]
        report["pipeline_config"] = {
            "model_repo": args.model_repo,
            "pipeline_config": args.pipeline_config,
            "path": str(pipeline_config_path),
            "shape_slat_sampler_params": sampler_params,
        }
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        last_trustworthy_phase = phase

        phase = "load_model"
        phase_started = time.perf_counter()
        model_ref = resolve_model_ref(
            args.model_repo,
            pipeline_args["models"]["shape_slat_flow_model_512"],
        )
        flow_model = source_models.from_pretrained(model_ref).to(device).eval()
        effective_device_type = next(flow_model.parameters()).device.type
        route_identity = build_route_identity(
            device_type=effective_device_type,
            output_npz=output_npz,
            conditioning_path=args.conditioning,
            support_sample_path=args.shape_slat_support_sample,
            noise_sample_path=args.shape_flow_noise_sample,
            block_indices=block_indices,
            trace_names=trace_names,
            steps=args.steps,
            seed=args.seed,
            branch=args.branch,
            source_tar=args.source_tar,
            model_repo=args.model_repo,
            pipeline_config=args.pipeline_config,
            block_input_replay_sample=args.block_input_replay_sample,
            block_input_replay_scope=[] if block_input_replay is None else block_input_replay.scope,
        )
        report["route_identity"] = route_identity
        report["model"] = {
            "model_ref": model_ref,
            "parameter_count": int(sum(parameter.numel() for parameter in flow_model.parameters())),
        }
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        last_trustworthy_phase = phase

        phase = "trace_forward"
        phase_started = time.perf_counter()
        cond = _load_conditioning(args.conditioning, torch, device)
        coords = torch.from_numpy(support_noise.coords).to(device=device, dtype=torch.int32)
        feats = torch.from_numpy(support_noise.noise).to(device=device, dtype=torch.float32)
        arrays: dict[str, np.ndarray] = {
            "route_identity_json": np.asarray(json.dumps(route_identity, sort_keys=True)),
            "coords": support_noise.coords,
            "coords_3d": support_noise.coords_3d,
            "trace_block_indices": np.asarray(block_indices, dtype=np.int32),
            "trace_block_index": np.asarray(block_indices[0], dtype=np.int32),
            "shape_flow_trace_step_index": np.asarray(args.shape_flow_trace_step_index, dtype=np.int32),
            "t": np.asarray(1000.0 * t, dtype=np.float32),
            "t_normalized": np.asarray(t, dtype=np.float32),
            "t_prev": np.asarray(t_prev, dtype=np.float32),
            "steps": np.asarray(args.steps, dtype=np.int32),
            "rescale_t": np.asarray(rescale_t, dtype=np.float32),
        }
        block_set = set(block_indices)
        with torch.inference_mode():
            for branch_name, cond_key in _select_branches(args.branch):
                x = SparseTensor(feats=feats.clone(), coords=coords)
                t_model = torch.tensor(
                    [1000.0 * t] * x.shape[0],
                    device=device,
                    dtype=torch.float32,
                )
                h = flow_model.input_layer(x)
                h = manual_cast(h, flow_model.dtype)
                t_emb = flow_model.t_embedder(t_model)
                if flow_model.share_mod:
                    t_emb = flow_model.adaLN_modulation(t_emb)
                t_emb = manual_cast(t_emb, flow_model.dtype)
                context = manual_cast(cond[cond_key], flow_model.dtype)
                arrays[f"{branch_name}_input_projected"] = _tensor_to_numpy(h, batched_sparse=True)
                for block_index, block in enumerate(flow_model.blocks):
                    if block_input_replay is not None and (branch_name, block_index) in block_input_replay.arrays:
                        replay_feats = torch.from_numpy(
                            block_input_replay.arrays[(branch_name, block_index)]
                        ).to(device=device, dtype=h.feats.dtype)
                        if tuple(replay_feats.shape) != tuple(h.feats.shape):
                            raise ValueError(
                                f"{branch_name}_block{block_index}_input replay tensor shape "
                                f"{tuple(replay_feats.shape)} does not match live shape {tuple(h.feats.shape)}"
                            )
                        h = h.replace(replay_feats)
                    if block_index in block_set:
                        h, block_trace = _trace_shape_block(torch, block, h, t_emb, context)
                        for name in trace_names:
                            if name in block_trace:
                                arrays[f"{branch_name}_block{block_index}_{name}"] = block_trace[name]
                    else:
                        h = block(h, t_emb, context)
                h = manual_cast(h, x.dtype)
                final_norm = torch.nn.functional.layer_norm(h.feats.float(), h.feats.shape[-1:])
                final_sparse = h.replace(final_norm)
                final_out = flow_model.out_layer(final_sparse)
                arrays[f"{branch_name}_final_norm"] = final_norm.detach().float().cpu().numpy().astype(np.float32)
                arrays[f"{branch_name}_final_out_flat"] = final_out.feats.detach().float().cpu().numpy().astype(np.float32)
                arrays[f"{branch_name}_final_output"] = arrays[f"{branch_name}_final_out_flat"]
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        last_trustworthy_phase = phase

        phase = "write_outputs"
        phase_started = time.perf_counter()
        np.savez(output_npz, **{key: np.ascontiguousarray(value) for key, value in arrays.items()})
        report.update(
            {
                "status": "done",
                "failure_phase": None,
                "last_trustworthy_phase": "shape_flow_block_trace_saved",
                "primary_output_status": "written",
                "primary_output": {
                    "path": str(output_npz),
                    "sha256": _sha256_file(output_npz),
                    "size_bytes": output_npz.stat().st_size,
                    "keys": sorted(arrays.keys()),
                },
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        _write_json(output_json, report)
        return 0
    except Exception as exc:
        payload = _failure_payload(
            report=report,
            route_identity=route_identity,
            phase=phase,
            last_trustworthy_phase=last_trustworthy_phase,
            started=started,
            error=exc,
            output_npz=args.output_npz,
        )
        _write_json(args.output_json, payload)
        print(f"source_cuda_shape_block_trace failed in {phase}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
