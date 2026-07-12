#!/usr/bin/env python3
"""CUDA witness for TRELLIS sparse-flow cross-Q formation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import traceback
from typing import Any

import numpy as np


SCHEMA = "trellis2mlx.cuda_sparse_cross_q.v1"


def metric_np(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    normalized_singleton_batch = False
    if a.ndim == b.ndim + 1 and a.shape[0] == 1 and a.shape[1:] == b.shape:
        a = a[0]
        normalized_singleton_batch = True
    elif b.ndim == a.ndim + 1 and b.shape[0] == 1 and b.shape[1:] == a.shape:
        b = b[0]
        normalized_singleton_batch = True
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
        "normalized_singleton_batch": normalized_singleton_batch,
    }


def _require(data: Any, key: str) -> np.ndarray:
    if key not in data:
        raise KeyError(f"witness missing required key {key!r}")
    return np.asarray(data[key], dtype=np.float32)


def _squeeze_batch(array: np.ndarray) -> np.ndarray:
    out = np.asarray(array, dtype=np.float32)
    while out.ndim > 0 and out.shape[0] == 1:
        out = out[0]
    return out


def _multihead_rms_norm_np(x: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    gamma = np.asarray(gamma, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    out = x / np.maximum(norm, np.finfo(np.float32).tiny)
    out = out * gamma * np.float32(math.sqrt(x.shape[-1]))
    return out.astype(np.float32)


def _layer_norm32_torch(x: Any, weight: Any, bias: Any) -> Any:
    import torch.nn.functional as F

    x_dtype = x.dtype
    out = F.layer_norm(x.float(), x.shape[-1:], weight.float(), bias.float(), eps=1e-6)
    return out.to(dtype=x_dtype)


def _q_rms_norm_torch(q: Any, gamma: Any) -> Any:
    import torch
    import torch.nn.functional as F

    q_dtype = q.dtype
    q = q.float()
    out = F.normalize(q, dim=-1) * gamma.float() * math.sqrt(q.shape[-1])
    return out.to(dtype=q_dtype)


def _cross_q_from_after_self(
    *,
    after_self_np: np.ndarray,
    norm2_weight_np: np.ndarray,
    norm2_bias_np: np.ndarray,
    to_q_weight_np: np.ndarray,
    to_q_bias_np: np.ndarray,
    q_rms_norm_gamma_np: np.ndarray,
    device: str,
) -> dict[str, np.ndarray]:
    import torch
    import torch.nn.functional as F

    target = torch.device(device)
    after_self = torch.from_numpy(_squeeze_batch(after_self_np)).to(
        device=target,
        dtype=torch.bfloat16,
    )
    norm2_weight = torch.from_numpy(np.asarray(norm2_weight_np, dtype=np.float32)).to(
        device=target,
        dtype=torch.bfloat16,
    )
    norm2_bias = torch.from_numpy(np.asarray(norm2_bias_np, dtype=np.float32)).to(
        device=target,
        dtype=torch.bfloat16,
    )
    to_q_weight = torch.from_numpy(np.asarray(to_q_weight_np, dtype=np.float32)).to(
        device=target,
        dtype=torch.bfloat16,
    )
    to_q_bias = torch.from_numpy(np.asarray(to_q_bias_np, dtype=np.float32)).to(
        device=target,
        dtype=torch.bfloat16,
    )
    gamma = torch.from_numpy(np.asarray(q_rms_norm_gamma_np, dtype=np.float32)).to(
        device=target,
        dtype=torch.bfloat16,
    )

    norm2 = _layer_norm32_torch(after_self, norm2_weight, norm2_bias)
    pre = F.linear(norm2, to_q_weight, to_q_bias).reshape(norm2.shape[0], 12, -1)
    post = _q_rms_norm_torch(pre, gamma)
    return {
        "norm2": norm2.float().cpu().numpy(),
        "cross_q_pre_norm": pre.float().cpu().numpy(),
        "cross_q_post_norm": post.float().cpu().numpy(),
    }


def _cross_q_from_norm2(
    *,
    norm2_np: np.ndarray,
    to_q_weight_np: np.ndarray,
    to_q_bias_np: np.ndarray,
    q_rms_norm_gamma_np: np.ndarray,
    device: str,
) -> dict[str, np.ndarray]:
    import torch
    import torch.nn.functional as F

    target = torch.device(device)
    norm2 = torch.from_numpy(_squeeze_batch(norm2_np)).to(device=target, dtype=torch.bfloat16)
    to_q_weight = torch.from_numpy(np.asarray(to_q_weight_np, dtype=np.float32)).to(
        device=target,
        dtype=torch.bfloat16,
    )
    to_q_bias = torch.from_numpy(np.asarray(to_q_bias_np, dtype=np.float32)).to(
        device=target,
        dtype=torch.bfloat16,
    )
    gamma = torch.from_numpy(np.asarray(q_rms_norm_gamma_np, dtype=np.float32)).to(
        device=target,
        dtype=torch.bfloat16,
    )

    pre = F.linear(norm2, to_q_weight, to_q_bias).reshape(norm2.shape[0], 12, -1)
    post = _q_rms_norm_torch(pre, gamma)
    return {
        "cross_q_pre_norm": pre.float().cpu().numpy(),
        "cross_q_post_norm": post.float().cpu().numpy(),
    }


def _input_metrics(data: Any) -> dict[str, Any]:
    return {
        name: metric_np(_require(data, f"source_{name}"), _require(data, f"captured_{name}"))
        for name in ("after_self", "norm2", "cross_q_pre_norm", "cross_q_post_norm")
    }


def _run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import torch

    with np.load(args.witness, allow_pickle=False) as data:
        route_identity = json.loads(str(data["route_identity_json"].item()))
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        weights = {
            "norm2_weight_np": _require(data, "source_norm2_weight"),
            "norm2_bias_np": _require(data, "source_norm2_bias"),
            "to_q_weight_np": _require(data, "source_to_q_weight"),
            "to_q_bias_np": _require(data, "source_to_q_bias"),
            "q_rms_norm_gamma_np": _require(data, "source_q_rms_norm_gamma"),
        }
        outputs: dict[str, np.ndarray] = {}
        for prefix in ("source", "captured"):
            after_self_outputs = _cross_q_from_after_self(
                after_self_np=_require(data, f"{prefix}_after_self"),
                device="cuda",
                **weights,
            )
            norm2_outputs = _cross_q_from_norm2(
                norm2_np=_require(data, f"{prefix}_norm2"),
                to_q_weight_np=weights["to_q_weight_np"],
                to_q_bias_np=weights["to_q_bias_np"],
                q_rms_norm_gamma_np=weights["q_rms_norm_gamma_np"],
                device="cuda",
            )
            for name, value in after_self_outputs.items():
                outputs[f"cuda_{prefix}_after_self_{name}"] = value
            for name, value in norm2_outputs.items():
                outputs[f"cuda_{prefix}_norm2_{name}"] = value

        report = {
            "schema": SCHEMA,
            "status": "done",
            "witness_path": str(args.witness),
            "route_identity": route_identity,
            "torch_version": torch.__version__,
            "cuda_available": True,
            "cuda_device": torch.cuda.get_device_name(0),
            "input_shapes": {
                "after_self": list(_squeeze_batch(_require(data, "captured_after_self")).shape),
                "norm2": list(_squeeze_batch(_require(data, "captured_norm2")).shape),
                "cross_q_pre_norm": list(_require(data, "captured_cross_q_pre_norm").shape),
                "cross_q_post_norm": list(_require(data, "captured_cross_q_post_norm").shape),
                "norm2_weight": list(weights["norm2_weight_np"].shape),
                "to_q_weight": list(weights["to_q_weight_np"].shape),
                "q_rms_norm_gamma": list(weights["q_rms_norm_gamma_np"].shape),
            },
            "source_vs_captured_inputs": _input_metrics(data),
            "cuda_captured_after_self_vs_captured_norm2": metric_np(
                _require(data, "captured_norm2"),
                outputs["cuda_captured_after_self_norm2"],
            ),
            "cuda_captured_after_self_vs_source_norm2": metric_np(
                _require(data, "source_norm2"),
                outputs["cuda_captured_after_self_norm2"],
            ),
            "cuda_captured_after_self_vs_captured_cross_q_pre_norm": metric_np(
                _require(data, "captured_cross_q_pre_norm"),
                outputs["cuda_captured_after_self_cross_q_pre_norm"],
            ),
            "cuda_captured_after_self_vs_source_cross_q_pre_norm": metric_np(
                _require(data, "source_cross_q_pre_norm"),
                outputs["cuda_captured_after_self_cross_q_pre_norm"],
            ),
            "cuda_captured_after_self_vs_captured_cross_q_post_norm": metric_np(
                _require(data, "captured_cross_q_post_norm"),
                outputs["cuda_captured_after_self_cross_q_post_norm"],
            ),
            "cuda_captured_after_self_vs_source_cross_q_post_norm": metric_np(
                _require(data, "source_cross_q_post_norm"),
                outputs["cuda_captured_after_self_cross_q_post_norm"],
            ),
            "cuda_captured_norm2_vs_captured_cross_q_pre_norm": metric_np(
                _require(data, "captured_cross_q_pre_norm"),
                outputs["cuda_captured_norm2_cross_q_pre_norm"],
            ),
            "cuda_captured_norm2_vs_captured_cross_q_post_norm": metric_np(
                _require(data, "captured_cross_q_post_norm"),
                outputs["cuda_captured_norm2_cross_q_post_norm"],
            ),
            "cuda_source_after_self_vs_source_norm2": metric_np(
                _require(data, "source_norm2"),
                outputs["cuda_source_after_self_norm2"],
            ),
            "cuda_source_after_self_vs_source_cross_q_pre_norm": metric_np(
                _require(data, "source_cross_q_pre_norm"),
                outputs["cuda_source_after_self_cross_q_pre_norm"],
            ),
            "cuda_source_after_self_vs_source_cross_q_post_norm": metric_np(
                _require(data, "source_cross_q_post_norm"),
                outputs["cuda_source_after_self_cross_q_post_norm"],
            ),
            "cuda_source_norm2_vs_source_cross_q_pre_norm": metric_np(
                _require(data, "source_cross_q_pre_norm"),
                outputs["cuda_source_norm2_cross_q_pre_norm"],
            ),
            "cuda_source_norm2_vs_source_cross_q_post_norm": metric_np(
                _require(data, "source_cross_q_post_norm"),
                outputs["cuda_source_norm2_cross_q_post_norm"],
            ),
            "cuda_source_after_self_vs_cuda_captured_after_self_norm2": metric_np(
                outputs["cuda_source_after_self_norm2"],
                outputs["cuda_captured_after_self_norm2"],
            ),
            "cuda_source_after_self_vs_cuda_captured_after_self_cross_q_post_norm": metric_np(
                outputs["cuda_source_after_self_cross_q_post_norm"],
                outputs["cuda_captured_after_self_cross_q_post_norm"],
            ),
        }
    return report, outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witness", type=Path, default=Path("source_sparse_cross_q_witness.npz"))
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
            "failure_phase": "cuda_sparse_cross_q",
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
