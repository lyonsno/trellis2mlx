#!/usr/bin/env python3
"""Identify the CUDA SDPA backend used by a TRELLIS cross-attention witness."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
from typing import Any
import warnings

import numpy as np


SCHEMA = "trellis2mlx.cuda_cross_sdpa_backend_witness.v1"
EXPECTED_TORCH = "2.10.0+cu128"
EXPECTED_DEVICE = "Tesla T4"


class BackendIdentificationError(ValueError):
    """The profiler evidence cannot authenticate one CUDA SDPA backend."""


BACKEND_EVENT_TOKENS = {
    "math": (
        "_scaled_dot_product_attention_math",
    ),
    "efficient_attention": (
        "_scaled_dot_product_efficient_attention",
        "_efficient_attention",
    ),
    "flash_attention": (
        "_scaled_dot_product_flash_attention",
        "_flash_attention",
    ),
    "cudnn_attention": (
        "_scaled_dot_product_cudnn_attention",
        "_cudnn_attention",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def backend_specs() -> tuple[tuple[str, str | None], ...]:
    return (
        ("default", None),
        ("math", "MATH"),
        ("efficient_attention", "EFFICIENT_ATTENTION"),
        ("flash_attention", "FLASH_ATTENTION"),
        ("cudnn_attention", "CUDNN_ATTENTION"),
    )


def _require(data: Any, key: str) -> np.ndarray:
    if key not in data:
        raise KeyError(f"witness missing required key {key!r}")
    return np.asarray(data[key], dtype=np.float32)


def _validated_payload(
    *,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    reference: np.ndarray,
    route_identity: dict[str, Any],
    input_artifacts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if q.ndim != 4:
        raise ValueError(f"q must have shape [B,T,H,D], got {q.shape}")
    if k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            f"k and v must have shape [B,S,H,D], got k={k.shape}, v={v.shape}"
        )
    if k.shape != v.shape:
        raise ValueError(f"k and v shapes differ: k={k.shape}, v={v.shape}")
    if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError(
            "q and k/v must share batch, head, and head-dimension axes: "
            f"q={q.shape}, k={k.shape}"
        )
    if reference.shape != q.shape:
        raise ValueError(
            "reference_attention_raw must have the same shape as q: "
            f"reference={reference.shape}, q={q.shape}"
        )
    for name, value in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("reference_attention_raw", reference),
    ):
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")

    return {
        "route_identity": route_identity,
        "input_artifacts": input_artifacts,
        "q": np.ascontiguousarray(q),
        "k": np.ascontiguousarray(k),
        "v": np.ascontiguousarray(v),
        "reference_attention_raw": np.ascontiguousarray(reference),
    }


def load_witness(path: Path) -> dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        q = _require(data, "q")
        k = _require(data, "k")
        v = _require(data, "v")
        reference = _require(data, "reference_attention_raw")
        if "route_identity_json" not in data:
            raise KeyError("witness missing required key 'route_identity_json'")
        route_identity = json.loads(str(data["route_identity_json"].item()))
    return _validated_payload(
        q=q,
        k=k,
        v=v,
        reference=reference,
        route_identity=route_identity,
        input_artifacts={
            "combined_witness": {
                "path": str(path),
                "sha256": sha256_file(path),
            }
        },
    )


def discover_split_traces(root: Path) -> tuple[Path, Path]:
    source_candidates: list[Path] = []
    kv_candidates: list[Path] = []
    for path in sorted(Path(root).rglob("cuda_result.npz")):
        try:
            with np.load(path, allow_pickle=False) as data:
                keys = set(data.files)
        except Exception:
            continue
        if {
            "pos_block0_cross_q_post_norm",
            "pos_block0_cross_attention_raw",
        } <= keys:
            source_candidates.append(path)
        if {
            "pos_block0_cross_k_pre_norm",
            "pos_block0_cross_v",
            "pos_block0_cross_k_post_norm",
        } <= keys:
            kv_candidates.append(path)
    if len(source_candidates) != 1:
        raise ValueError(
            "expected exactly one source trace with cross Q and raw attention, "
            f"found {len(source_candidates)}: {source_candidates}"
        )
    if len(kv_candidates) != 1:
        raise ValueError(
            "expected exactly one K/V trace, "
            f"found {len(kv_candidates)}: {kv_candidates}"
        )
    return source_candidates[0], kv_candidates[0]


def _squeeze_context_batch(value: np.ndarray, *, key: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 5 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 4:
        raise ValueError(
            f"{key} must have shape [B,S,H,D] or [B,1,S,H,D], got {value.shape}"
        )
    return value


def load_split_witness(source_trace: Path, kv_trace: Path) -> dict[str, Any]:
    source_trace = Path(source_trace)
    kv_trace = Path(kv_trace)
    with np.load(source_trace, allow_pickle=False) as data:
        q = _require(data, "pos_block0_cross_q_post_norm")
        reference = _require(data, "pos_block0_cross_attention_raw")
        if "route_identity_json" not in data:
            raise KeyError("source trace missing required key 'route_identity_json'")
        source_identity = json.loads(str(data["route_identity_json"].item()))
    with np.load(kv_trace, allow_pickle=False) as data:
        k = _squeeze_context_batch(
            _require(data, "pos_block0_cross_k_post_norm"),
            key="pos_block0_cross_k_post_norm",
        )
        v = _squeeze_context_batch(
            _require(data, "pos_block0_cross_v"),
            key="pos_block0_cross_v",
        )
        if "route_identity_json" not in data:
            raise KeyError("K/V trace missing required key 'route_identity_json'")
        kv_identity = json.loads(str(data["route_identity_json"].item()))
    return _validated_payload(
        q=q,
        k=k,
        v=v,
        reference=reference,
        route_identity={
            "source_trace": source_identity,
            "kv_trace": kv_identity,
        },
        input_artifacts={
            "source_trace": {
                "path": str(source_trace),
                "sha256": sha256_file(source_trace),
            },
            "kv_trace": {
                "path": str(kv_trace),
                "sha256": sha256_file(kv_trace),
            },
        },
    )


def exact_metric(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    if actual.shape != expected.shape:
        return {
            "shape_match": False,
            "shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "exact": False,
        }
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {
        "shape_match": True,
        "shape": list(actual.shape),
        "exact": bool(np.array_equal(actual, expected)),
        "nonzero": int(np.count_nonzero(actual != expected)),
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()) if difference.size else 0.0,
    }


def identify_default_backend(
    variants: list[dict[str, Any]],
) -> dict[str, Any]:
    defaults = [item for item in variants if item.get("name") == "default"]
    if len(defaults) != 1:
        raise BackendIdentificationError(
            f"expected exactly one default SDPA variant, found {len(defaults)}"
        )
    default = defaults[0]
    profiler_events = [
        str(event) for event in default.get("profiler_events", ())
    ]
    matched_families = {
        family
        for family, tokens in BACKEND_EVENT_TOKENS.items()
        if any(
            token in event.lower()
            for event in profiler_events
            for token in tokens
        )
    }
    if not matched_families:
        raise BackendIdentificationError(
            "default profiler events identify no concrete CUDA SDPA backend"
        )
    if len(matched_families) != 1:
        raise BackendIdentificationError(
            "default profiler events identify multiple CUDA SDPA backends: "
            f"{sorted(matched_families)}"
        )

    effective = next(iter(matched_families))
    matching = [
        item for item in variants if item.get("name") == effective
    ]
    if (
        len(matching) != 1
        or matching[0].get("status") != "done"
        or matching[0].get("vs_source_reference", {}).get("exact") is not True
    ):
        raise BackendIdentificationError(
            "matching forced backend did not complete an exact source replay: "
            f"{effective}"
        )
    return {
        "default_backend_effective": effective,
        "profiler_events": profiler_events,
        "matching_forced_variant": effective,
        "matching_forced_variant_exact": True,
    }


def _run_variant(
    *,
    name: str,
    backend_name: str | None,
    q: Any,
    k: Any,
    v: Any,
    reference: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray | None]:
    import torch
    import torch.nn.functional as functional
    from torch.profiler import ProfilerActivity, profile

    backend = None
    if backend_name is not None:
        from torch.nn.attention import SDPBackend

        backend = getattr(SDPBackend, backend_name, None)
        if backend is None:
            return {
                "name": name,
                "requested_backend": backend_name,
                "status": "unavailable",
                "reason": f"torch.nn.attention.SDPBackend has no {backend_name}",
            }, None

    started = time.perf_counter()
    caught: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as profiler:
                if backend is None:
                    output = functional.scaled_dot_product_attention(q, k, v)
                else:
                    from torch.nn.attention import sdpa_kernel

                    with sdpa_kernel(backends=[backend]):
                        output = functional.scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize()
            caught = [str(item.message) for item in observed]
    except Exception as exc:
        return {
            "name": name,
            "requested_backend": backend_name or "default",
            "status": "unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
            "warnings": caught,
            "elapsed_seconds": time.perf_counter() - started,
        }, None

    output_np = (
        output.transpose(1, 2)
        .to(dtype=torch.float32)
        .contiguous()
        .cpu()
        .numpy()
    )
    event_names = sorted(
        {
            event.key
            for event in profiler.key_averages()
            if any(
                token in event.key.lower()
                for token in (
                    "attention",
                    "cudnn",
                    "efficient",
                    "flash",
                    "scaled_dot_product",
                )
            )
        }
    )
    return {
        "name": name,
        "requested_backend": backend_name or "default",
        "effective_route": "torch-sdpa-default"
        if backend is None
        else "torch-sdpa-forced-single-backend",
        "status": "done",
        "warnings": caught,
        "profiler_events": event_names,
        "elapsed_seconds": time.perf_counter() - started,
        "vs_source_reference": exact_metric(output_np, reference),
    }, output_np


def run_witness(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device_name = torch.cuda.get_device_name(0)
    if torch.__version__ != EXPECTED_TORCH:
        raise RuntimeError(
            f"expected Torch {EXPECTED_TORCH}, got {torch.__version__}"
        )
    if device_name != EXPECTED_DEVICE:
        raise RuntimeError(
            f"expected CUDA device {EXPECTED_DEVICE}, got {device_name}"
        )

    device = torch.device("cuda")
    q = torch.from_numpy(payload["q"]).to(device=device, dtype=torch.bfloat16)
    k = torch.from_numpy(payload["k"]).to(device=device, dtype=torch.bfloat16)
    v = torch.from_numpy(payload["v"]).to(device=device, dtype=torch.bfloat16)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    variants: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for name, backend_name in backend_specs():
        variant, output = _run_variant(
            name=name,
            backend_name=backend_name,
            q=q,
            k=k,
            v=v,
            reference=payload["reference_attention_raw"],
        )
        variants.append(variant)
        if output is not None:
            arrays[f"{name}_attention_raw"] = output

    default = variants[0]
    if (
        default.get("status") != "done"
        or default.get("vs_source_reference", {}).get("exact") is not True
    ):
        raise RuntimeError(
            "default CUDA SDPA did not reproduce the authenticated source output"
        )
    backend_identification = identify_default_backend(variants)

    report = {
        "schema": SCHEMA,
        "status": "done",
        "failure_phase": None,
        "primary_output_status": "written",
        "torch": torch.__version__,
        "cuda_device": device_name,
        "input": {
            "artifacts": payload["input_artifacts"],
            "route_identity": payload["route_identity"],
            "q_shape": list(payload["q"].shape),
            "k_shape": list(payload["k"].shape),
            "v_shape": list(payload["v"].shape),
        },
        "variants": variants,
        "default_backend_effective": backend_identification[
            "default_backend_effective"
        ],
        "default_backend_evidence": {
            key: value
            for key, value in backend_identification.items()
            if key != "default_backend_effective"
        },
    }
    return report, arrays


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--witness", type=Path)
    inputs.add_argument("--discover-input-root", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    args = parser.parse_args(argv)

    phase = "request_received"
    last_trustworthy_phase = phase
    primary_preexisted = args.output_npz.is_file()
    started = time.perf_counter()
    try:
        phase = "input_validation"
        if args.witness is not None:
            payload = load_witness(args.witness)
        else:
            source_trace, kv_trace = discover_split_traces(
                args.discover_input_root
            )
            payload = load_split_witness(source_trace, kv_trace)
        last_trustworthy_phase = "input_validated"
        phase = "cuda_sdpa_backend_sweep"
        report, arrays = run_witness(payload)
        last_trustworthy_phase = "cuda_sdpa_backend_sweep_completed"
        phase = "write_primary"
        np.savez_compressed(args.output_npz, **arrays)
        last_trustworthy_phase = "primary_output_written"
        report["elapsed_seconds"] = time.perf_counter() - started
        report["primary_output"] = {
            "path": str(args.output_npz),
            "sha256": sha256_file(args.output_npz),
            "keys": sorted(arrays),
        }
        args.output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
    except Exception as exc:
        failure_phase = (
            "cuda_sdpa_backend_identification"
            if isinstance(exc, BackendIdentificationError)
            else phase
        )
        report = {
            "schema": SCHEMA,
            "status": "failed",
            "failure_phase": failure_phase,
            "last_trustworthy_phase": last_trustworthy_phase,
            "primary_output_status": (
                "stale-preexisting"
                if primary_preexisted
                else "partial-produced"
                if args.output_npz.is_file()
                else "missing"
            ),
            "witness_path": str(args.witness)
            if args.witness is not None
            else None,
            "discover_input_root": str(args.discover_input_root)
            if args.discover_input_root is not None
            else None,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.perf_counter() - started,
            "traceback": traceback.format_exc(),
        }
        if args.output_npz.is_file():
            report["primary_output"] = {
                "path": str(args.output_npz),
                "sha256": sha256_file(args.output_npz),
            }
        args.output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(report, sort_keys=True))
        return 1

    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
