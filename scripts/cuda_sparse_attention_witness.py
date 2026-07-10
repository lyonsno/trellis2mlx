#!/usr/bin/env python3
"""CUDA witness for TRELLIS sparse-flow self-attention raw and output projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback
from typing import Any

import numpy as np


SCHEMA = "trellis2mlx.cuda_sparse_self_attention.v1"


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report, outputs = _run(args)
        np.savez_compressed(args.output_npz, **outputs)
    except Exception as exc:
        report = {
            "schema": SCHEMA,
            "status": "failed",
            "failure_phase": "cuda_sparse_self_attention",
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
