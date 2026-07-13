#!/usr/bin/env python3
"""CUDA witness for TRELLIS sparse-flow self-attention raw and output projection."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np


SCHEMA = "trellis2mlx.cuda_sparse_self_attention.v1"
VARIANT_SCHEMA = "trellis2mlx.cuda_attention_variant_matrix.v1"


@dataclass(frozen=True)
class AttentionVariant:
    name: str
    backend: str
    chunk_size: int | None
    compute_dtype: str


def build_variant_specs(
    *,
    source_chunk_size: int = 4096,
    manual_chunk_size: int = 128,
) -> list[AttentionVariant]:
    if source_chunk_size <= 0 or manual_chunk_size <= 0:
        raise ValueError("attention chunk sizes must be positive")
    return [
        AttentionVariant(
            f"source_default_chunk{source_chunk_size}",
            "default",
            source_chunk_size,
            "input",
        ),
        AttentionVariant("default_full", "default", None, "input"),
        AttentionVariant("default_chunk512", "default", 512, "input"),
        AttentionVariant(f"math_chunk{source_chunk_size}", "math", source_chunk_size, "input"),
        AttentionVariant("math_chunk512", "math", 512, "input"),
        AttentionVariant(
            f"manual_fp32_chunk{manual_chunk_size}",
            "manual",
            manual_chunk_size,
            "float32",
        ),
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_qkv(array: np.ndarray, *, key: str) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 3:
        raise ValueError(f"{key} must have shape [N,H,D] or [1,N,H,D], got {value.shape}")
    return np.ascontiguousarray(value)


def _normalize_attention(array: np.ndarray, *, key: str, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    if value.size != int(np.prod(shape)):
        raise ValueError(
            f"{key} element count {value.size} does not match Q/K/V element count {int(np.prod(shape))}"
        )
    return np.ascontiguousarray(value.reshape(shape))


def load_witness(path: Path) -> dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if "route_identity_json" not in data:
            raise KeyError("witness missing required key 'route_identity_json'")
        route_identity = json.loads(str(data["route_identity_json"].item()))
        branches = [branch for branch in ("pos", "neg") if f"{branch}_q" in data]
        if not branches:
            raise ValueError("variant witness must contain pos_q and/or neg_q")
        arrays: dict[str, dict[str, np.ndarray]] = {}
        for branch in branches:
            q = _normalize_qkv(_require(data, f"{branch}_q"), key=f"{branch}_q")
            k = _normalize_qkv(_require(data, f"{branch}_k"), key=f"{branch}_k")
            v = _normalize_qkv(_require(data, f"{branch}_v"), key=f"{branch}_v")
            if q.shape != k.shape or q.shape != v.shape:
                raise ValueError(
                    f"{branch} Q/K/V shape mismatch: q={q.shape}, k={k.shape}, v={v.shape}"
                )
            arrays[branch] = {
                "q": q,
                "k": k,
                "v": v,
                "reference_attention_raw": _normalize_attention(
                    _require(data, f"{branch}_reference_attention_raw"),
                    key=f"{branch}_reference_attention_raw",
                    shape=q.shape,
                ),
                "source_chunked_attention_raw": _normalize_attention(
                    _require(data, f"{branch}_source_chunked_attention_raw"),
                    key=f"{branch}_source_chunked_attention_raw",
                    shape=q.shape,
                ),
            }
    return {
        "path": path,
        "sha256": _sha256_file(path),
        "route_identity": route_identity,
        "branches": arrays,
    }


def metric_np(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        return {
            "shape_match": False,
            "reference_shape": list(a.shape),
            "candidate_shape": list(b.shape),
        }
    diff = np.abs(a - b)
    return {
        "shape_match": True,
        "shape": list(a.shape),
        "mean_abs": float(diff.mean(dtype=np.float64)),
        "max_abs": float(diff.max(initial=0.0)),
        "nonzero": int(np.count_nonzero(diff)),
        "exact": bool(np.array_equal(a, b)),
    }


def _require(data: Any, key: str) -> np.ndarray:
    if key not in data:
        raise KeyError(f"witness missing required key {key!r}")
    return np.asarray(data[key], dtype=np.float32)


def _attention_and_projection(
    *,
    q_np: np.ndarray,
    k_np: np.ndarray,
    v_np: np.ndarray,
    weight_np: np.ndarray,
    bias_np: np.ndarray | None,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    import torch.nn.functional as F

    target = torch.device(device)
    q = torch.from_numpy(np.asarray(q_np, dtype=np.float32)).to(device=target, dtype=torch.bfloat16)
    k = torch.from_numpy(np.asarray(k_np, dtype=np.float32)).to(device=target, dtype=torch.bfloat16)
    v = torch.from_numpy(np.asarray(v_np, dtype=np.float32)).to(device=target, dtype=torch.bfloat16)
    weight = torch.from_numpy(np.asarray(weight_np, dtype=np.float32)).to(
        device=target,
        dtype=torch.bfloat16,
    )
    bias = None
    if bias_np is not None:
        bias = torch.from_numpy(np.asarray(bias_np, dtype=np.float32)).to(
            device=target,
            dtype=torch.bfloat16,
        )
    raw = F.scaled_dot_product_attention(
        q.unsqueeze(0).transpose(1, 2),
        k.unsqueeze(0).transpose(1, 2),
        v.unsqueeze(0).transpose(1, 2),
    )
    raw = raw.transpose(1, 2).reshape(1, q.shape[0], -1)
    projected = F.linear(raw, weight, bias)
    return raw.squeeze(0).float().cpu().numpy(), projected.squeeze(0).float().cpu().numpy()


def _run_sdpa_chunks(q: Any, k: Any, v: Any, *, chunk_size: int | None) -> Any:
    import torch
    import torch.nn.functional as F

    if chunk_size is None or q.shape[2] <= chunk_size:
        return F.scaled_dot_product_attention(q, k, v)
    return torch.cat(
        [
            F.scaled_dot_product_attention(q[:, :, start : start + chunk_size], k, v)
            for start in range(0, q.shape[2], chunk_size)
        ],
        dim=2,
    )


def _run_manual_fp32_chunks(q: Any, k: Any, v: Any, *, chunk_size: int) -> Any:
    import torch

    q32 = q.float()
    k32 = k.float()
    v32 = v.float()
    scale = 1.0 / math.sqrt(q.shape[-1])
    k_transposed = k32.transpose(-2, -1)
    chunks = []
    for start in range(0, q.shape[2], chunk_size):
        q_chunk = q32[:, :, start : start + chunk_size]
        scores = (q_chunk @ k_transposed) * scale
        probs = torch.softmax(scores, dim=-1)
        chunks.append((probs @ v32).to(dtype=q.dtype))
    return torch.cat(chunks, dim=2)


def _run_attention_variant(
    q: Any,
    k: Any,
    v: Any,
    spec: AttentionVariant,
) -> Any:
    if spec.backend == "default":
        return _run_sdpa_chunks(q, k, v, chunk_size=spec.chunk_size)
    if spec.backend == "math":
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.MATH):
            return _run_sdpa_chunks(q, k, v, chunk_size=spec.chunk_size)
    if spec.backend == "manual":
        if spec.chunk_size is None:
            raise ValueError("manual attention requires a finite chunk size")
        return _run_manual_fp32_chunks(q, k, v, chunk_size=spec.chunk_size)
    raise ValueError(f"unsupported attention variant backend {spec.backend!r}")


def _variant_matrix(
    loaded: dict[str, Any],
    *,
    source_chunk_size: int,
    manual_chunk_size: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device("cuda")
    specs = build_variant_specs(
        source_chunk_size=source_chunk_size,
        manual_chunk_size=manual_chunk_size,
    )
    outputs: dict[str, np.ndarray] = {}
    branch_reports: dict[str, Any] = {}
    for branch, arrays in loaded["branches"].items():
        q = torch.from_numpy(arrays["q"]).to(device=device, dtype=torch.bfloat16)
        k = torch.from_numpy(arrays["k"]).to(device=device, dtype=torch.bfloat16)
        v = torch.from_numpy(arrays["v"]).to(device=device, dtype=torch.bfloat16)
        q = q.unsqueeze(0).transpose(1, 2)
        k = k.unsqueeze(0).transpose(1, 2)
        v = v.unsqueeze(0).transpose(1, 2)
        variant_reports: dict[str, Any] = {}
        for spec in specs:
            torch.cuda.synchronize()
            started = time.perf_counter()
            raw = _run_attention_variant(q, k, v, spec)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            raw_np = raw.transpose(1, 2).squeeze(0).float().cpu().numpy()
            outputs[f"{branch}_{spec.name}"] = raw_np
            variant_reports[spec.name] = {
                "backend": spec.backend,
                "chunk_size": spec.chunk_size,
                "compute_dtype": spec.compute_dtype,
                "elapsed_seconds": elapsed,
                "vs_reference": metric_np(arrays["reference_attention_raw"], raw_np),
                "vs_source_chunked": metric_np(arrays["source_chunked_attention_raw"], raw_np),
            }
            del raw
        source_name = f"source_default_chunk{source_chunk_size}"
        source_identity = variant_reports[source_name]["vs_source_chunked"]
        if not source_identity.get("exact", False):
            raise RuntimeError(
                f"{branch} {source_name} did not reproduce captured source-CUDA raw attention: "
                f"{source_identity}"
            )
        best_name, best_report = min(
            variant_reports.items(),
            key=lambda item: item[1]["vs_reference"]["mean_abs"],
        )
        branch_reports[branch] = {
            "shape": list(arrays["q"].shape),
            "source_route_reproduced_exactly": True,
            "best_reference_anchor": best_name,
            "best_reference_metric": best_report["vs_reference"],
            "variants": variant_reports,
        }
        del q, k, v
        torch.cuda.empty_cache()

    report = {
        "schema": VARIANT_SCHEMA,
        "status": "done",
        "failure_phase": None,
        "primary_output_status": "pending",
        "witness_path": str(loaded["path"]),
        "witness_sha256": loaded["sha256"],
        "route_identity": loaded["route_identity"],
        "torch_version": torch.__version__,
        "cuda_available": True,
        "cuda_device": torch.cuda.get_device_name(0),
        "effective_variants": [spec.__dict__ for spec in specs],
        "branches": branch_reports,
    }
    return report, outputs


def _input_metrics(data: Any) -> dict[str, Any]:
    return {
        name: metric_np(_require(data, f"source_{name}"), _require(data, f"captured_{name}"))
        for name in ("q_post_rope", "k_post_rope", "v", "attention_raw", "self_attn")
    }


def _run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import torch

    with np.load(args.witness, allow_pickle=False) as data:
        route_identity = json.loads(str(data["route_identity_json"].item()))
        weight = _require(data, "source_to_out_weight")
        bias = _require(data, "source_to_out_bias") if "source_to_out_bias" in data else None
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        outputs: dict[str, np.ndarray] = {}
        for prefix in ("source", "captured"):
            raw, projected = _attention_and_projection(
                q_np=_require(data, f"{prefix}_q_post_rope"),
                k_np=_require(data, f"{prefix}_k_post_rope"),
                v_np=_require(data, f"{prefix}_v"),
                weight_np=weight,
                bias_np=bias,
                device="cuda",
            )
            outputs[f"cuda_{prefix}_attention_raw"] = raw
            outputs[f"cuda_{prefix}_self_attn"] = projected

        report = {
            "schema": SCHEMA,
            "status": "done",
            "witness_path": str(args.witness),
            "route_identity": route_identity,
            "torch_version": torch.__version__,
            "cuda_available": True,
            "cuda_device": torch.cuda.get_device_name(0),
            "input_shapes": {
                "q": list(_require(data, "source_q_post_rope").shape),
                "k": list(_require(data, "source_k_post_rope").shape),
                "v": list(_require(data, "source_v").shape),
                "to_out_weight": list(weight.shape),
                "to_out_bias": None if bias is None else list(bias.shape),
            },
            "source_vs_captured_inputs": _input_metrics(data),
            "cuda_source_vs_source_raw": metric_np(
                _require(data, "source_attention_raw"),
                outputs["cuda_source_attention_raw"],
            ),
            "cuda_source_vs_source_self_attn": metric_np(
                _require(data, "source_self_attn"),
                outputs["cuda_source_self_attn"],
            ),
            "cuda_source_vs_captured_raw": metric_np(
                _require(data, "captured_attention_raw"),
                outputs["cuda_source_attention_raw"],
            ),
            "cuda_source_vs_captured_self_attn": metric_np(
                _require(data, "captured_self_attn"),
                outputs["cuda_source_self_attn"],
            ),
            "cuda_captured_vs_captured_raw": metric_np(
                _require(data, "captured_attention_raw"),
                outputs["cuda_captured_attention_raw"],
            ),
            "cuda_captured_vs_captured_self_attn": metric_np(
                _require(data, "captured_self_attn"),
                outputs["cuda_captured_self_attn"],
            ),
            "cuda_captured_vs_source_raw": metric_np(
                _require(data, "source_attention_raw"),
                outputs["cuda_captured_attention_raw"],
            ),
            "cuda_captured_vs_source_self_attn": metric_np(
                _require(data, "source_self_attn"),
                outputs["cuda_captured_self_attn"],
            ),
            "cuda_source_vs_cuda_captured_raw": metric_np(
                outputs["cuda_source_attention_raw"],
                outputs["cuda_captured_attention_raw"],
            ),
            "cuda_source_vs_cuda_captured_self_attn": metric_np(
                outputs["cuda_source_self_attn"],
                outputs["cuda_captured_self_attn"],
            ),
        }
    return report, outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witness", type=Path, default=Path("source_sparse_attention_witness.npz"))
    parser.add_argument("--output-json", type=Path, default=Path("cuda_result.json"))
    parser.add_argument("--output-npz", type=Path, default=Path("cuda_result.npz"))
    parser.add_argument(
        "--variant-matrix",
        action="store_true",
        help="Run captured Q/K/V through chunk/backend/precision attention variants.",
    )
    parser.add_argument("--source-chunk-size", type=int, default=4096)
    parser.add_argument("--manual-chunk-size", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    phase = "input_validation"
    try:
        if args.variant_matrix:
            loaded = load_witness(args.witness)
            phase = "attention_variants"
            report, outputs = _variant_matrix(
                loaded,
                source_chunk_size=args.source_chunk_size,
                manual_chunk_size=args.manual_chunk_size,
            )
            phase = "output_persistence"
            np.savez(args.output_npz, **outputs)
            report["primary_output_status"] = "written"
        else:
            report, outputs = _run(args)
            phase = "output_persistence"
            np.savez_compressed(args.output_npz, **outputs)
    except Exception as exc:
        report = {
            "schema": VARIANT_SCHEMA if args.variant_matrix else SCHEMA,
            "status": "failed",
            "failure_phase": phase,
            "primary_output_status": "partial_unverified" if args.output_npz.exists() else "missing",
            "witness_path": str(args.witness),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, sort_keys=True))
        return 1
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
