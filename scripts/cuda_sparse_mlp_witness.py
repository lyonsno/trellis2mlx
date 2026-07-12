#!/usr/bin/env python3
"""CUDA witness for exact-input TRELLIS sparse-flow MLP outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback
from typing import Any

import numpy as np


SCHEMA = "trellis2mlx.cuda_sparse_mlp.v1"


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


def _gelu_tanh_torch(x: Any) -> Any:
    import torch

    return torch.nn.functional.gelu(x, approximate="tanh")


def _mlp_forward(
    *,
    mlp_input_np: np.ndarray,
    gate_mlp_np: np.ndarray,
    after_cross_np: np.ndarray,
    fc1_weight_np: np.ndarray,
    fc1_bias_np: np.ndarray,
    fc2_weight_np: np.ndarray,
    fc2_bias_np: np.ndarray,
    device: str,
) -> dict[str, np.ndarray]:
    import torch
    import torch.nn.functional as F

    target = torch.device(device)
    mlp_input = torch.from_numpy(_squeeze_batch(mlp_input_np)).to(device=target, dtype=torch.bfloat16)
    gate_mlp = torch.from_numpy(_squeeze_batch(gate_mlp_np)).to(device=target, dtype=torch.bfloat16)
    after_cross = torch.from_numpy(_squeeze_batch(after_cross_np)).to(device=target, dtype=torch.bfloat16)
    fc1_weight = torch.from_numpy(np.asarray(fc1_weight_np, dtype=np.float32)).to(
        device=target,
        dtype=torch.bfloat16,
    )
    fc1_bias = torch.from_numpy(np.asarray(fc1_bias_np, dtype=np.float32)).to(
        device=target,
        dtype=torch.bfloat16,
    )
    fc2_weight = torch.from_numpy(np.asarray(fc2_weight_np, dtype=np.float32)).to(
        device=target,
        dtype=torch.bfloat16,
    )
    fc2_bias = torch.from_numpy(np.asarray(fc2_bias_np, dtype=np.float32)).to(
        device=target,
        dtype=torch.bfloat16,
    )

    fc1 = F.linear(mlp_input, fc1_weight, fc1_bias)
    gelu = _gelu_tanh_torch(fc1)
    mlp = F.linear(gelu, fc2_weight, fc2_bias)
    mlp_gated = mlp * gate_mlp
    after_mlp = after_cross + mlp_gated
    return {
        "cuda_mlp_fc1": fc1.float().cpu().numpy(),
        "cuda_mlp_gelu": gelu.float().cpu().numpy(),
        "cuda_mlp": mlp.float().cpu().numpy(),
        "cuda_mlp_gated": mlp_gated.float().cpu().numpy(),
        "cuda_after_mlp": after_mlp.float().cpu().numpy(),
    }


def _run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import torch

    with np.load(args.witness, allow_pickle=False) as data:
        route_identity = json.loads(str(data["route_identity_json"].item()))
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        outputs = _mlp_forward(
            mlp_input_np=_require(data, "captured_mlp_input"),
            gate_mlp_np=_require(data, "captured_gate_mlp"),
            after_cross_np=_require(data, "captured_after_cross"),
            fc1_weight_np=_require(data, "source_mlp_fc1_weight"),
            fc1_bias_np=_require(data, "source_mlp_fc1_bias"),
            fc2_weight_np=_require(data, "source_mlp_fc2_weight"),
            fc2_bias_np=_require(data, "source_mlp_fc2_bias"),
            device="cuda",
        )
        report = {
            "schema": SCHEMA,
            "status": "done",
            "witness_path": str(args.witness),
            "route_identity": route_identity,
            "torch_version": torch.__version__,
            "cuda_available": True,
            "cuda_device": torch.cuda.get_device_name(0),
            "input_shapes": {
                "mlp_input": list(_squeeze_batch(_require(data, "captured_mlp_input")).shape),
                "gate_mlp": list(_squeeze_batch(_require(data, "captured_gate_mlp")).shape),
                "after_cross": list(_squeeze_batch(_require(data, "captured_after_cross")).shape),
                "fc1_weight": list(_require(data, "source_mlp_fc1_weight").shape),
                "fc2_weight": list(_require(data, "source_mlp_fc2_weight").shape),
            },
            "cuda_vs_captured_mlp": metric_np(_require(data, "captured_mlp"), outputs["cuda_mlp"]),
            "cuda_vs_captured_mlp_gated": metric_np(
                _require(data, "captured_mlp_gated"),
                outputs["cuda_mlp_gated"],
            ),
            "cuda_vs_captured_after_mlp": metric_np(
                _require(data, "captured_after_mlp"),
                outputs["cuda_after_mlp"],
            ),
        }
        if "captured_mlp_fc1" in data:
            report["cuda_vs_captured_mlp_fc1"] = metric_np(
                _require(data, "captured_mlp_fc1"),
                outputs["cuda_mlp_fc1"],
            )
        if "captured_mlp_gelu" in data:
            report["cuda_vs_captured_mlp_gelu"] = metric_np(
                _require(data, "captured_mlp_gelu"),
                outputs["cuda_mlp_gelu"],
            )
        if "source_mlp" in data:
            report["cuda_vs_source_mlp"] = metric_np(
                _require(data, "source_mlp"),
                outputs["cuda_mlp"],
            )
        if "source_mlp_gated" in data:
            report["cuda_vs_source_mlp_gated"] = metric_np(
                _require(data, "source_mlp_gated"),
                outputs["cuda_mlp_gated"],
            )
        if "source_after_mlp" in data:
            report["cuda_vs_source_after_mlp"] = metric_np(
                _require(data, "source_after_mlp"),
                outputs["cuda_after_mlp"],
            )
    return report, outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witness", type=Path, default=Path("source_sparse_mlp_witness.npz"))
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
            "failure_phase": "cuda_sparse_mlp",
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
