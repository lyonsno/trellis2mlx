#!/usr/bin/env python3
"""Cross-backend witnesses for TRELLIS sparse-flow attention execution."""

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
STAGE_SELECTION_SCHEMA = "trellis2mlx.attention_residual_stage_selection.v1"
STAGE_SCHEMA = "trellis2mlx.attention_residual_stage_capture.v1"
RESIDUAL_ROWS_SCHEMA = "trellis2mlx.block0_split_sqrt_residual_rows.v1"


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
            f"{key} element count {value.size} does not match query output "
            f"element count {int(np.prod(shape))}"
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
            if k.shape != v.shape:
                raise ValueError(
                    f"{branch} K/V shape mismatch: k={k.shape}, v={v.shape}"
                )
            if q.shape[1:] != k.shape[1:]:
                raise ValueError(
                    f"{branch} Q and K/V must share head axes: "
                    f"q={q.shape}, k={k.shape}, v={v.shape}"
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


def build_stage_selection(
    residual_report: dict[str, Any],
    *,
    residual_report_sha256: str,
    token_count: int,
    source_token_count: int | None = None,
    head_count: int,
    chunk_size: int,
    control_count: int,
    head_dim: int | None = None,
    branch: str = "pos",
) -> dict[str, Any]:
    if source_token_count is None:
        source_token_count = token_count
    if (
        token_count <= 0
        or source_token_count <= 0
        or head_count <= 0
        or chunk_size <= 0
    ):
        raise ValueError(
            "token_count, source_token_count, head_count, and chunk_size "
            "must be positive"
        )
    if control_count < 0:
        raise ValueError("control_count must be nonnegative")
    if residual_report.get("schema") != RESIDUAL_ROWS_SCHEMA:
        raise ValueError(
            f"residual report schema must be {RESIDUAL_ROWS_SCHEMA!r}, "
            f"got {residual_report.get('schema')!r}"
        )
    source_rows = residual_report.get("rows")
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError("residual report rows must be a nonempty list")
    witness_sha256 = residual_report.get("witness_sha256")
    if not isinstance(witness_sha256, str) or len(witness_sha256) != 64:
        raise ValueError("residual report witness_sha256 must be a SHA256 digest")

    rows: list[dict[str, Any]] = []
    residual_coordinates: set[tuple[int, int]] = set()
    for source_row in source_rows:
        token = int(source_row["token"])
        head = int(source_row["head"])
        coordinate = (token, head)
        if coordinate in residual_coordinates:
            raise ValueError(f"duplicate residual coordinate {coordinate}")
        residual_coordinates.add(coordinate)
        rows.append(
            {
                "token": token,
                "head": head,
                "kind": "residual",
                "max_abs": float(source_row["max_abs"]),
                "nonzero": int(source_row["nonzero"]),
            }
        )

    total_coordinates = token_count * head_count
    available_controls = total_coordinates - len(residual_coordinates)
    if control_count > available_controls:
        raise ValueError(
            f"requested {control_count} controls but only {available_controls} coordinates remain"
        )
    control_coordinates: set[tuple[int, int]] = set()
    if control_count:
        denominator = max(control_count - 1, 1)
        for control_index in range(control_count):
            flat = round(control_index * (total_coordinates - 1) / denominator)
            for offset in range(total_coordinates):
                candidate = (flat + offset) % total_coordinates
                coordinate = (candidate // head_count, candidate % head_count)
                if coordinate not in residual_coordinates and coordinate not in control_coordinates:
                    control_coordinates.add(coordinate)
                    rows.append(
                        {
                            "token": coordinate[0],
                            "head": coordinate[1],
                            "kind": "zero_residual_control",
                        }
                    )
                    break

    selection: dict[str, Any] = {
        "schema": STAGE_SELECTION_SCHEMA,
        "witness_sha256": witness_sha256,
        "residual_report_sha256": residual_report_sha256,
        "branch": branch,
        "token_count": token_count,
        "head_count": head_count,
        "chunk_size": chunk_size,
        "selection_policy": {
            "residual_rows_requested": "all",
            "residual_rows_selected": len(residual_coordinates),
            "controls_requested": control_count,
            "controls_selected": len(control_coordinates),
        },
        "rows": rows,
    }
    if head_dim is not None:
        selection["head_dim"] = int(head_dim)
    if source_token_count != token_count:
        selection["source_token_count"] = int(source_token_count)
    group_stage_rows(
        rows,
        token_count=token_count,
        head_count=head_count,
        chunk_size=chunk_size,
    )
    return selection


def group_stage_rows(
    rows: list[dict[str, Any]],
    *,
    token_count: int,
    head_count: int,
    chunk_size: int,
) -> list[dict[str, Any]]:
    if token_count <= 0 or head_count <= 0 or chunk_size <= 0:
        raise ValueError("token_count, head_count, and chunk_size must be positive")
    groups: dict[int, dict[str, Any]] = {}
    seen: set[tuple[int, int]] = set()
    for row_index, row in enumerate(rows):
        token = int(row["token"])
        head = int(row["head"])
        if token < 0 or token >= token_count:
            raise ValueError(f"row {row_index} token {token} is outside [0, {token_count})")
        if head < 0 or head >= head_count:
            raise ValueError(f"row {row_index} head {head} is outside [0, {head_count})")
        coordinate = (token, head)
        if coordinate in seen:
            raise ValueError(f"duplicate selected coordinate {coordinate}")
        seen.add(coordinate)
        chunk_start = (token // chunk_size) * chunk_size
        chunk_stop = min(chunk_start + chunk_size, token_count)
        group = groups.setdefault(
            chunk_start,
            {
                "chunk_start": chunk_start,
                "chunk_stop": chunk_stop,
                "row_indices": [],
                "tokens": [],
                "heads": [],
            },
        )
        group["row_indices"].append(row_index)
        group["tokens"].append(token)
        group["heads"].append(head)
    return [groups[start] for start in sorted(groups)]


def load_stage_selection(
    path: Path,
    loaded: dict[str, Any],
    *,
    residual_report_path: Path,
    expected_residual_report_sha256: str,
    expected_branch: str,
    expected_chunk_size: int,
    expected_control_count: int,
) -> dict[str, Any]:
    path = Path(path)
    selection = json.loads(path.read_text())
    if selection.get("schema") != STAGE_SELECTION_SCHEMA:
        raise ValueError(
            f"selection schema must be {STAGE_SELECTION_SCHEMA!r}, got {selection.get('schema')!r}"
        )
    if selection.get("witness_sha256") != loaded["sha256"]:
        raise ValueError(
            "selection witness_sha256 does not match the effective witness: "
            f"requested={selection.get('witness_sha256')!r}, effective={loaded['sha256']!r}"
        )
    residual_report_path = Path(residual_report_path)
    effective_residual_sha256 = _sha256_file(residual_report_path)
    if effective_residual_sha256 != expected_residual_report_sha256:
        raise ValueError(
            "effective residual census SHA256 does not match the caller expectation: "
            f"requested={expected_residual_report_sha256!r}, "
            f"effective={effective_residual_sha256!r}"
        )
    if selection.get("residual_report_sha256") != effective_residual_sha256:
        raise ValueError(
            "selection residual census SHA256 does not match the effective census"
        )
    branch = selection.get("branch")
    if branch != expected_branch:
        raise ValueError(
            f"selection branch {branch!r} does not match caller expectation "
            f"{expected_branch!r}"
        )
    if branch not in loaded["branches"]:
        raise ValueError(f"selection branch {branch!r} is absent from the witness")
    token_count, head_count, head_dim = loaded["branches"][branch]["q"].shape
    source_token_count = loaded["branches"][branch]["k"].shape[0]
    expected_dimensions = {
        "token_count": token_count,
        "source_token_count": source_token_count,
        "head_count": head_count,
        "head_dim": head_dim,
    }
    for key, expected in expected_dimensions.items():
        default = token_count if key == "source_token_count" else -1
        if int(selection.get(key, default)) != expected:
            raise ValueError(
                f"selection {key} {selection.get(key)!r} does not match witness {expected}"
            )
    policy = selection.get("selection_policy")
    if not isinstance(policy, dict) or policy.get("residual_rows_requested") != "all":
        raise ValueError("selection must preserve all residual rows without a cap")
    if int(policy.get("controls_requested", -1)) != expected_control_count:
        raise ValueError(
            "selection controls_requested does not match caller expectation: "
            f"selection={policy.get('controls_requested')!r}, "
            f"expected={expected_control_count}"
        )
    rows = selection.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("selection rows must be a nonempty list")
    chunk_size = int(selection.get("chunk_size", 0))
    if chunk_size != expected_chunk_size:
        raise ValueError(
            f"selection chunk_size {chunk_size} does not match caller expectation "
            f"{expected_chunk_size}"
        )
    groups = group_stage_rows(
        rows,
        token_count=token_count,
        head_count=head_count,
        chunk_size=chunk_size,
    )
    residual_report = json.loads(residual_report_path.read_text())
    canonical_selection = build_stage_selection(
        residual_report,
        residual_report_sha256=effective_residual_sha256,
        token_count=token_count,
        source_token_count=source_token_count,
        head_count=head_count,
        head_dim=head_dim,
        chunk_size=expected_chunk_size,
        control_count=expected_control_count,
        branch=expected_branch,
    )
    if selection != canonical_selection:
        raise ValueError(
            "selection does not match the canonical residual census reconstruction"
        )
    return {
        **selection,
        "path": str(path),
        "sha256": _sha256_file(path),
        "residual_report_path": str(residual_report_path),
        "effective_residual_report_sha256": effective_residual_sha256,
        "groups": groups,
    }


def validate_stage_outputs(
    *,
    selection: dict[str, Any],
    arrays: dict[str, np.ndarray],
    chunk_receipts: list[dict[str, Any]],
    token_count: int,
    source_token_count: int | None = None,
    head_dim: int,
) -> None:
    if source_token_count is None:
        source_token_count = token_count
    for receipt in chunk_receipts:
        expected_query_count = int(receipt["chunk_stop"]) - int(receipt["chunk_start"])
        if int(receipt.get("computed_query_count", -1)) != expected_query_count:
            raise ValueError(
                "chunk receipt computed_query_count does not match the complete chunk: "
                f"{receipt}"
            )
        if receipt.get("selection_applied_after_full_chunk") is not True:
            raise ValueError(f"chunk selection was not applied after full evaluation: {receipt}")
        if receipt.get("stage_evaluation_mode") != "independent_prefix_replays":
            raise ValueError(
                "chunk stages must use independent_prefix_replays to avoid changing "
                f"lazy execution semantics: {receipt}"
            )

    rows = selection["rows"]
    row_count = len(rows)
    expected_shapes = {
        "row_tokens": (row_count,),
        "row_heads": (row_count,),
        "scores_fp32": (row_count, source_token_count),
        "probs_fp32": (row_count, source_token_count),
        "output_fp32": (row_count, head_dim),
        "output_bf16_as_fp32": (row_count, head_dim),
        "source_cuda_bf16_as_fp32": (row_count, head_dim),
    }
    for key, expected_shape in expected_shapes.items():
        if key not in arrays:
            raise ValueError(f"stage output missing required array {key!r}")
        if tuple(arrays[key].shape) != expected_shape:
            raise ValueError(
                f"stage output {key} shape {arrays[key].shape} does not match {expected_shape}"
            )
    expected_tokens = np.asarray([int(row["token"]) for row in rows], dtype=np.int32)
    expected_heads = np.asarray([int(row["head"]) for row in rows], dtype=np.int32)
    if not np.array_equal(arrays["row_tokens"], expected_tokens):
        raise ValueError("stage output row_tokens do not preserve selection order")
    if not np.array_equal(arrays["row_heads"], expected_heads):
        raise ValueError("stage output row_heads do not preserve selection order")


def validate_mlx_final_residual_contract(
    *,
    selection: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    reference = np.asarray(arrays["source_cuda_bf16_as_fp32"], dtype=np.float32)
    candidate = np.asarray(arrays["output_bf16_as_fp32"], dtype=np.float32)
    if reference.shape != candidate.shape or reference.shape[0] != len(selection["rows"]):
        raise ValueError("MLX final residual contract arrays do not match the selection")
    mismatches: list[dict[str, Any]] = []
    observed_nonzero = 0
    for row_index, row in enumerate(selection["rows"]):
        diff = np.abs(reference[row_index] - candidate[row_index])
        actual_nonzero = int(np.count_nonzero(diff))
        actual_max = float(diff.max(initial=0.0))
        observed_nonzero += actual_nonzero
        if row.get("kind") == "residual":
            expected_nonzero = int(row["nonzero"])
            expected_max = float(row["max_abs"])
        else:
            expected_nonzero = 0
            expected_max = 0.0
        if actual_nonzero != expected_nonzero or actual_max != expected_max:
            mismatches.append(
                {
                    "row_index": row_index,
                    "token": int(row["token"]),
                    "head": int(row["head"]),
                    "kind": row.get("kind"),
                    "expected_nonzero": expected_nonzero,
                    "actual_nonzero": actual_nonzero,
                    "expected_max_abs": expected_max,
                    "actual_max_abs": actual_max,
                }
            )
    if mismatches:
        raise ValueError(
            "MLX production-prefix output changed the prior residual contract; "
            f"first mismatches={mismatches[:3]}"
        )
    return {
        "exact": True,
        "rows_checked": len(selection["rows"]),
        "residual_elements_reproduced": observed_nonzero,
        "controls_exact": True,
    }


def validate_persisted_stage_admission(
    *,
    backend: str,
    selection: dict[str, Any],
    arrays: dict[str, np.ndarray],
    chunk_receipts: list[dict[str, Any]],
    persisted_route: dict[str, Any],
    expected_route: dict[str, Any],
) -> dict[str, Any]:
    validate_stage_outputs(
        selection=selection,
        arrays=arrays,
        chunk_receipts=chunk_receipts,
        token_count=int(selection["token_count"]),
        source_token_count=int(
            selection.get("source_token_count", selection["token_count"])
        ),
        head_dim=int(selection["head_dim"]),
    )
    if persisted_route != expected_route:
        raise ValueError("persisted route identity does not match the effective route")
    final_metric = metric_np(
        arrays["source_cuda_bf16_as_fp32"],
        arrays["output_bf16_as_fp32"],
    )
    mlx_contract = None
    if backend == "cuda" and not final_metric.get("exact", False):
        raise ValueError(
            "persisted CUDA split-sqrt output did not reproduce selected source rows exactly"
        )
    if backend == "mlx":
        mlx_contract = validate_mlx_final_residual_contract(
            selection=selection,
            arrays=arrays,
        )
    return {
        "selected_final_vs_source_cuda": final_metric,
        "mlx_prior_residual_contract": mlx_contract,
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


def _empty_stage_arrays(
    *, row_count: int, source_token_count: int, head_dim: int
) -> dict[str, np.ndarray]:
    return {
        "row_tokens": np.empty((row_count,), dtype=np.int32),
        "row_heads": np.empty((row_count,), dtype=np.int32),
        "scores_fp32": np.empty((row_count, source_token_count), dtype=np.float32),
        "probs_fp32": np.empty((row_count, source_token_count), dtype=np.float32),
        "output_fp32": np.empty((row_count, head_dim), dtype=np.float32),
        "output_bf16_as_fp32": np.empty((row_count, head_dim), dtype=np.float32),
        "source_cuda_bf16_as_fp32": np.empty((row_count, head_dim), dtype=np.float32),
    }


def _stage_source_cuda_rows(
    arrays: dict[str, np.ndarray], selection: dict[str, Any]
) -> np.ndarray:
    tokens = np.asarray([int(row["token"]) for row in selection["rows"]], dtype=np.int64)
    heads = np.asarray([int(row["head"]) for row in selection["rows"]], dtype=np.int64)
    return np.asarray(
        arrays["source_chunked_attention_raw"][tokens, heads],
        dtype=np.float32,
    )


def _capture_stage_rows_cuda(
    arrays: dict[str, np.ndarray],
    selection: dict[str, Any],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device("cuda")
    q = torch.from_numpy(arrays["q"]).to(device=device, dtype=torch.bfloat16)
    k = torch.from_numpy(arrays["k"]).to(device=device, dtype=torch.bfloat16)
    v = torch.from_numpy(arrays["v"]).to(device=device, dtype=torch.bfloat16)
    q = q.unsqueeze(0).transpose(1, 2)
    k = k.unsqueeze(0).transpose(1, 2)
    v = v.unsqueeze(0).transpose(1, 2)
    scale = 1.0 / math.sqrt(q.shape[-1])
    scaling_factor = math.sqrt(scale)
    q32 = q.float()
    k_transposed = (k.float() * scaling_factor).transpose(-2, -1)
    v32 = v.float()

    token_count = int(q.shape[2])
    source_token_count = int(k.shape[2])
    head_dim = int(q.shape[3])
    result = _empty_stage_arrays(
        row_count=len(selection["rows"]),
        source_token_count=source_token_count,
        head_dim=head_dim,
    )
    receipts: list[dict[str, Any]] = []
    for group in selection["groups"]:
        start = int(group["chunk_start"])
        stop = int(group["chunk_stop"])
        row_indices = np.asarray(group["row_indices"], dtype=np.int64)
        heads = torch.as_tensor(group["heads"], device=device, dtype=torch.long)
        local_tokens = torch.as_tensor(
            [token - start for token in group["tokens"]],
            device=device,
            dtype=torch.long,
        )
        result["row_tokens"][row_indices] = np.asarray(group["tokens"], dtype=np.int32)
        result["row_heads"][row_indices] = np.asarray(group["heads"], dtype=np.int32)
        stage_elapsed: dict[str, float] = {}

        torch.cuda.synchronize()
        started = time.perf_counter()
        scores = (q32[:, :, start:stop, :] * scaling_factor) @ k_transposed
        torch.cuda.synchronize()
        stage_elapsed["scores_fp32"] = time.perf_counter() - started
        result["scores_fp32"][row_indices] = (
            scores[0, heads, local_tokens].float().cpu().numpy()
        )
        del scores

        torch.cuda.synchronize()
        started = time.perf_counter()
        probs = torch.softmax(
            (q32[:, :, start:stop, :] * scaling_factor) @ k_transposed,
            dim=-1,
        )
        torch.cuda.synchronize()
        stage_elapsed["probs_fp32"] = time.perf_counter() - started
        result["probs_fp32"][row_indices] = (
            probs[0, heads, local_tokens].float().cpu().numpy()
        )
        del probs

        torch.cuda.synchronize()
        started = time.perf_counter()
        output_fp32 = (
            torch.softmax(
                (q32[:, :, start:stop, :] * scaling_factor) @ k_transposed,
                dim=-1,
            )
            @ v32
        )
        torch.cuda.synchronize()
        stage_elapsed["output_fp32"] = time.perf_counter() - started
        result["output_fp32"][row_indices] = (
            output_fp32[0, heads, local_tokens].float().cpu().numpy()
        )
        del output_fp32

        receipts.append(
            {
                "chunk_start": start,
                "chunk_stop": stop,
                "computed_query_count": stop - start,
                "selected_row_count": len(row_indices),
                "selection_applied_after_full_chunk": True,
                "stage_evaluation_mode": "independent_prefix_replays",
                "stage_elapsed_seconds": stage_elapsed,
            }
        )
    torch.cuda.synchronize()
    production_chunks = []
    chunk_size = int(selection["chunk_size"])
    for start in range(0, token_count, chunk_size):
        stop = min(start + chunk_size, token_count)
        production_chunks.append(
            (
                torch.softmax(
                    (q32[:, :, start:stop, :] * scaling_factor) @ k_transposed,
                    dim=-1,
                )
                @ v32
            ).to(dtype=torch.bfloat16)
        )
    production_output = torch.cat(production_chunks, dim=2)
    torch.cuda.synchronize()
    all_tokens = torch.as_tensor(
        [int(row["token"]) for row in selection["rows"]],
        device=device,
        dtype=torch.long,
    )
    all_heads = torch.as_tensor(
        [int(row["head"]) for row in selection["rows"]],
        device=device,
        dtype=torch.long,
    )
    result["output_bf16_as_fp32"] = (
        production_output[0, all_heads, all_tokens].float().cpu().numpy()
    )
    result["source_cuda_bf16_as_fp32"] = _stage_source_cuda_rows(arrays, selection)
    return (
        result,
        receipts,
        {
            "backend": "cuda",
            "runtime": "torch",
            "torch_version": torch.__version__,
            "cuda_available": True,
            "device": torch.cuda.get_device_name(0),
            "production_final_evaluation": {
                "mode": "all_chunks_concatenated_before_selected_row_gather",
                "chunk_size": chunk_size,
                "computed_query_count": token_count,
            },
        },
    )


def _capture_stage_rows_mlx(
    arrays: dict[str, np.ndarray],
    selection: dict[str, Any],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    import mlx.core as mx

    q = mx.array(arrays["q"], dtype=mx.float32).astype(mx.bfloat16)
    k = mx.array(arrays["k"], dtype=mx.float32).astype(mx.bfloat16)
    v = mx.array(arrays["v"], dtype=mx.float32).astype(mx.bfloat16)
    q = q[None, ...].transpose(0, 2, 1, 3)
    k = k[None, ...].transpose(0, 2, 1, 3)
    v = v[None, ...].transpose(0, 2, 1, 3)
    scale = 1.0 / math.sqrt(q.shape[-1])
    scaling_factor = math.sqrt(scale)
    q32 = q.astype(mx.float32)
    k_transposed = (k.astype(mx.float32) * scaling_factor).transpose(0, 1, 3, 2)
    v32 = v.astype(mx.float32)
    mx.eval(q32, k_transposed, v32)

    token_count = int(q.shape[2])
    source_token_count = int(k.shape[2])
    head_dim = int(q.shape[3])
    result = _empty_stage_arrays(
        row_count=len(selection["rows"]),
        source_token_count=source_token_count,
        head_dim=head_dim,
    )
    receipts: list[dict[str, Any]] = []
    for group in selection["groups"]:
        start = int(group["chunk_start"])
        stop = int(group["chunk_stop"])
        row_indices = np.asarray(group["row_indices"], dtype=np.int64)
        heads = np.asarray(group["heads"], dtype=np.int64)
        local_tokens = np.asarray(
            [token - start for token in group["tokens"]],
            dtype=np.int64,
        )
        result["row_tokens"][row_indices] = np.asarray(group["tokens"], dtype=np.int32)
        result["row_heads"][row_indices] = np.asarray(group["heads"], dtype=np.int32)
        stage_elapsed: dict[str, float] = {}

        started = time.perf_counter()
        scores = (q32[:, :, start:stop, :] * scaling_factor) @ k_transposed
        mx.eval(scores)
        stage_elapsed["scores_fp32"] = time.perf_counter() - started
        scores_np = np.asarray(scores, dtype=np.float32)
        result["scores_fp32"][row_indices] = scores_np[0, heads, local_tokens]
        del scores, scores_np
        mx.clear_cache()

        started = time.perf_counter()
        probs = mx.softmax(
            (q32[:, :, start:stop, :] * scaling_factor) @ k_transposed,
            axis=-1,
        )
        mx.eval(probs)
        stage_elapsed["probs_fp32"] = time.perf_counter() - started
        probs_np = np.asarray(probs, dtype=np.float32)
        result["probs_fp32"][row_indices] = probs_np[0, heads, local_tokens]
        del probs, probs_np
        mx.clear_cache()

        started = time.perf_counter()
        output_fp32 = (
            mx.softmax(
                (q32[:, :, start:stop, :] * scaling_factor) @ k_transposed,
                axis=-1,
            )
            @ v32
        )
        mx.eval(output_fp32)
        stage_elapsed["output_fp32"] = time.perf_counter() - started
        output_fp32_np = np.asarray(output_fp32, dtype=np.float32)
        result["output_fp32"][row_indices] = output_fp32_np[0, heads, local_tokens]
        del output_fp32, output_fp32_np
        mx.clear_cache()

        receipts.append(
            {
                "chunk_start": start,
                "chunk_stop": stop,
                "computed_query_count": stop - start,
                "selected_row_count": len(row_indices),
                "selection_applied_after_full_chunk": True,
                "stage_evaluation_mode": "independent_prefix_replays",
                "stage_elapsed_seconds": stage_elapsed,
            }
        )
        mx.clear_cache()
    chunk_size = int(selection["chunk_size"])
    production_chunks = []
    for start in range(0, token_count, chunk_size):
        stop = min(start + chunk_size, token_count)
        production_chunks.append(
            (
                mx.softmax(
                    (q32[:, :, start:stop, :] * scaling_factor) @ k_transposed,
                    axis=-1,
                )
                @ v32
            ).astype(mx.bfloat16)
        )
    production_output = mx.concatenate(production_chunks, axis=2)
    mx.eval(production_output)
    production_output_np = np.asarray(
        production_output.astype(mx.float32),
        dtype=np.float32,
    )
    all_tokens = np.asarray(
        [int(row["token"]) for row in selection["rows"]],
        dtype=np.int64,
    )
    all_heads = np.asarray(
        [int(row["head"]) for row in selection["rows"]],
        dtype=np.int64,
    )
    result["output_bf16_as_fp32"] = production_output_np[
        0, all_heads, all_tokens
    ]
    result["source_cuda_bf16_as_fp32"] = _stage_source_cuda_rows(arrays, selection)
    return (
        result,
        receipts,
        {
            "backend": "mlx",
            "runtime": "mlx",
            "mlx_version": getattr(mx, "__version__", "unknown"),
            "device": str(mx.default_device()),
            "production_final_evaluation": {
                "mode": "all_chunks_concatenated_before_selected_row_gather",
                "chunk_size": chunk_size,
                "computed_query_count": token_count,
            },
        },
    )


def _capture_residual_stages(
    loaded: dict[str, Any],
    selection: dict[str, Any],
    *,
    backend: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    branch = str(selection["branch"])
    arrays = loaded["branches"][branch]
    if backend == "cuda":
        outputs, receipts, backend_identity = _capture_stage_rows_cuda(arrays, selection)
    elif backend == "mlx":
        outputs, receipts, backend_identity = _capture_stage_rows_mlx(arrays, selection)
    else:
        raise ValueError(f"unsupported stage backend {backend!r}")
    token_count, _head_count, head_dim = arrays["q"].shape
    source_token_count = arrays["k"].shape[0]
    validate_stage_outputs(
        selection=selection,
        arrays=outputs,
        chunk_receipts=receipts,
        token_count=token_count,
        source_token_count=source_token_count,
        head_dim=head_dim,
    )
    route_identity = {
        "schema": STAGE_SCHEMA,
        "backend": backend_identity,
        "witness_sha256": loaded["sha256"],
        "selection_sha256": selection["sha256"],
        "residual_report_sha256": selection["residual_report_sha256"],
        "branch": branch,
        "chunk_size": selection["chunk_size"],
        "selection_applied_after_full_chunk": True,
        "source_route_identity": loaded["route_identity"],
    }
    outputs["route_identity_json"] = np.array(
        json.dumps(route_identity, sort_keys=True, separators=(",", ":"))
    )
    final_metric = metric_np(
        outputs["source_cuda_bf16_as_fp32"],
        outputs["output_bf16_as_fp32"],
    )
    mlx_residual_contract = None
    if backend == "mlx":
        mlx_residual_contract = validate_mlx_final_residual_contract(
            selection=selection,
            arrays=outputs,
        )
    report = {
        "schema": STAGE_SCHEMA,
        "status": "done",
        "failure_phase": None,
        "primary_output_status": "pending",
        "route_identity": route_identity,
        "witness_path": str(loaded["path"]),
        "witness_sha256": loaded["sha256"],
        "selection_path": selection["path"],
        "selection_sha256": selection["sha256"],
        "selection_policy": selection["selection_policy"],
        "selected_row_count": len(selection["rows"]),
        "residual_row_count": sum(
            row.get("kind") == "residual" for row in selection["rows"]
        ),
        "control_row_count": sum(
            row.get("kind") == "zero_residual_control" for row in selection["rows"]
        ),
        "chunk_receipts": receipts,
        "stage_shapes": {
            key: list(value.shape)
            for key, value in outputs.items()
            if isinstance(value, np.ndarray)
        },
        "in_memory_selected_final_vs_source_cuda": final_metric,
        "in_memory_mlx_prior_residual_contract": mlx_residual_contract,
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--variant-matrix",
        action="store_true",
        help="Run captured Q/K/V through chunk/backend/precision attention variants.",
    )
    mode.add_argument(
        "--residual-stage-capture",
        action="store_true",
        help="Capture split-sqrt score, softmax, FP32 output, and BF16 output stages.",
    )
    mode.add_argument(
        "--build-stage-selection-from",
        type=Path,
        help="Build an uncapped residual-row selection from a residual report.",
    )
    parser.add_argument("--source-chunk-size", type=int, default=4096)
    parser.add_argument("--manual-chunk-size", type=int, default=128)
    parser.add_argument("--stage-backend", choices=("cuda", "mlx"))
    parser.add_argument("--selection-json", type=Path)
    parser.add_argument("--residual-report-json", type=Path)
    parser.add_argument("--expected-residual-report-sha256")
    parser.add_argument("--selection-output", type=Path)
    parser.add_argument("--stage-chunk-size", type=int, default=128)
    parser.add_argument("--control-count", type=int, default=128)
    parser.add_argument("--stage-branch", default="pos")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    phase = "request_validation"
    report: dict[str, Any] | None = None
    try:
        if args.residual_stage_capture and args.output_npz.exists():
            raise FileExistsError(
                f"refusing to overwrite preexisting primary output {args.output_npz}"
            )
        if args.build_stage_selection_from:
            if args.selection_output is None:
                raise ValueError("--selection-output is required when building a stage selection")
            if args.selection_output.exists():
                raise FileExistsError(
                    f"refusing to overwrite preexisting selection {args.selection_output}"
                )
            phase = "input_validation"
            loaded = load_witness(args.witness)
            residual_report = json.loads(args.build_stage_selection_from.read_text())
            if residual_report.get("witness_sha256") != loaded["sha256"]:
                raise ValueError(
                    "residual report witness_sha256 does not match the effective witness"
                )
            if args.stage_branch not in loaded["branches"]:
                raise ValueError(
                    f"stage branch {args.stage_branch!r} is absent from the witness"
                )
            branch_arrays = loaded["branches"][args.stage_branch]
            token_count, head_count, head_dim = branch_arrays["q"].shape
            source_token_count = branch_arrays["k"].shape[0]
            selection = build_stage_selection(
                residual_report,
                residual_report_sha256=_sha256_file(args.build_stage_selection_from),
                token_count=token_count,
                source_token_count=source_token_count,
                head_count=head_count,
                head_dim=head_dim,
                chunk_size=args.stage_chunk_size,
                control_count=args.control_count,
                branch=args.stage_branch,
            )
            phase = "output_persistence"
            args.selection_output.parent.mkdir(parents=True, exist_ok=True)
            args.selection_output.write_text(
                json.dumps(selection, indent=2, sort_keys=True) + "\n"
            )
            report = {
                "schema": STAGE_SELECTION_SCHEMA,
                "status": "done",
                "failure_phase": None,
                "primary_output_status": "written",
                "witness_path": str(loaded["path"]),
                "witness_sha256": loaded["sha256"],
                "residual_report_path": str(args.build_stage_selection_from),
                "residual_report_sha256": _sha256_file(args.build_stage_selection_from),
                "selection_output": str(args.selection_output),
                "selection_sha256": _sha256_file(args.selection_output),
                "selection_size_bytes": args.selection_output.stat().st_size,
                "selection_policy": selection["selection_policy"],
            }
        elif args.residual_stage_capture:
            if args.stage_backend is None:
                raise ValueError("--stage-backend is required for residual stage capture")
            if args.selection_json is None:
                raise ValueError("--selection-json is required for residual stage capture")
            if args.residual_report_json is None:
                raise ValueError(
                    "--residual-report-json is required for residual stage capture"
                )
            if not args.expected_residual_report_sha256:
                raise ValueError(
                    "--expected-residual-report-sha256 is required for residual stage capture"
                )
            phase = "input_validation"
            loaded = load_witness(args.witness)
            selection = load_stage_selection(
                args.selection_json,
                loaded,
                residual_report_path=args.residual_report_json,
                expected_residual_report_sha256=args.expected_residual_report_sha256,
                expected_branch=args.stage_branch,
                expected_chunk_size=args.stage_chunk_size,
                expected_control_count=args.control_count,
            )
            phase = "attention_stages"
            report, outputs = _capture_residual_stages(
                loaded,
                selection,
                backend=args.stage_backend,
            )
            phase = "output_persistence"
            np.savez_compressed(args.output_npz, **outputs)
            phase = "primary_output_validation"
            with np.load(args.output_npz, allow_pickle=False) as persisted:
                persisted_arrays = {
                    key: np.asarray(persisted[key])
                    for key in (
                        "row_tokens",
                        "row_heads",
                        "scores_fp32",
                        "probs_fp32",
                        "output_fp32",
                        "output_bf16_as_fp32",
                        "source_cuda_bf16_as_fp32",
                    )
                }
                persisted_route = json.loads(str(persisted["route_identity_json"].item()))
            persisted_admission = validate_persisted_stage_admission(
                backend=args.stage_backend,
                selection=selection,
                arrays=persisted_arrays,
                chunk_receipts=report["chunk_receipts"],
                persisted_route=persisted_route,
                expected_route=report["route_identity"],
            )
            report.update(persisted_admission)
            report["primary_output_status"] = "written"
            report["primary_output"] = {
                "path": str(args.output_npz),
                "sha256": _sha256_file(args.output_npz),
                "size_bytes": args.output_npz.stat().st_size,
            }
        elif args.variant_matrix:
            phase = "input_validation"
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
            phase = "input_validation"
            report, outputs = _run(args)
            phase = "output_persistence"
            np.savez_compressed(args.output_npz, **outputs)
    except Exception as exc:
        schema = (
            STAGE_SELECTION_SCHEMA
            if args.build_stage_selection_from
            else STAGE_SCHEMA
            if args.residual_stage_capture
            else VARIANT_SCHEMA
            if args.variant_matrix
            else SCHEMA
        )
        if report is None:
            report = {
                "schema": schema,
                "witness_path": str(args.witness),
            }
        primary_status = "missing"
        if args.residual_stage_capture and args.output_npz.exists():
            primary_status = (
                "preexisting_untrusted_preserved"
                if phase == "request_validation"
                else "written_unadmitted"
                if phase == "source_self_check"
                else "partial_unverified"
            )
        if args.build_stage_selection_from and args.selection_output is not None:
            if args.selection_output.exists():
                primary_status = (
                    "preexisting_untrusted_preserved"
                    if phase == "request_validation"
                    else "partial_unverified"
                )
        report.update(
            {
                "status": "failed",
                "failure_phase": phase,
                "primary_output_status": primary_status,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, sort_keys=True))
        return 1
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
