#!/usr/bin/env python3
"""Probe the source-CUDA terminal shape-flow linear reduction schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import traceback
from typing import Any
import zipfile

import numpy as np


SCHEMA = "trellis2mlx.cuda_terminal_linear_witness.v1"
ROUTE_SCHEMA = "trellis2mlx.shape_flow_terminal_linear_input.v1"
OPERATION = "shape_flow.final_linear"
EXPECTED_TORCH = "2.10.0+cu128"
EXPECTED_DEVICE = "Tesla T4"
CUDA_DEVICE_INDEX = 0
CUDA_DEVICE = "cuda:0"
EXPECTED_ROWS = 6038
INPUT_CHANNELS = 1536
OUTPUT_CHANNELS = 32
STEP_INDEX = 6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: str) -> bool:
    return (
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(path) as archive:
        logical_names = [
            name[:-4] if name.endswith(".npy") else name
            for name in archive.namelist()
        ]
        if len(logical_names) != len(set(logical_names)):
            raise ValueError("input NPZ contains duplicate logical members")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _scalar_json(value: np.ndarray, *, name: str) -> dict[str, Any]:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{name} must be a scalar string")
    try:
        result = json.loads(str(array.item()))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON") from exc
    if not isinstance(result, dict) or not result:
        raise ValueError(f"{name} must decode to a nonempty object")
    return result


def _require_array(
    arrays: dict[str, np.ndarray],
    name: str,
    *,
    shape: tuple[int, ...],
) -> np.ndarray:
    if name not in arrays:
        raise ValueError(f"input witness missing required array {name!r}")
    value = np.asarray(arrays[name])
    if value.dtype != np.float32:
        raise ValueError(f"{name} must have dtype float32, got {value.dtype}")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(value)


def validate_witness_arrays(
    arrays: dict[str, np.ndarray],
    *,
    expected_source_recurrence_sha256: str,
    expected_mlx_exact_state_trace_sha256: str,
    expected_model_checkpoint_sha256: str,
    expected_rows: int = EXPECTED_ROWS,
    input_channels: int = INPUT_CHANNELS,
    output_channels: int = OUTPUT_CHANNELS,
    step_index: int = STEP_INDEX,
) -> dict[str, Any]:
    expected_hashes = {
        "source_recurrence_sha256": expected_source_recurrence_sha256,
        "mlx_exact_state_trace_sha256": expected_mlx_exact_state_trace_sha256,
        "model_checkpoint_sha256": expected_model_checkpoint_sha256,
    }
    for name, value in expected_hashes.items():
        if not _canonical_sha256(value):
            raise ValueError(f"requested {name} must be canonical lowercase sha256")

    route = _scalar_json(
        arrays.get("route_identity_json", np.asarray(None)),
        name="route_identity_json",
    )
    exact_route_values = {
        "schema": ROUTE_SCHEMA,
        "operation": OPERATION,
        "shape_flow_step_index": step_index,
        **expected_hashes,
        "logical_shapes": {
            "input": [expected_rows, input_channels],
            "weight": [output_channels, input_channels],
            "bias": [output_channels],
            "output": [expected_rows, output_channels],
        },
    }
    for name, expected in exact_route_values.items():
        actual = route.get(name)
        if actual != expected:
            raise ValueError(
                f"route_identity_json.{name} must be {expected!r}, got {actual!r}"
            )

    input_shape = (expected_rows, input_channels)
    output_shape = (expected_rows, output_channels)
    normalized = {
        "pos_final_norm": _require_array(
            arrays, "pos_final_norm", shape=input_shape
        ),
        "neg_final_norm": _require_array(
            arrays, "neg_final_norm", shape=input_shape
        ),
        "weight": _require_array(
            arrays, "weight", shape=(output_channels, input_channels)
        ),
        "bias": _require_array(arrays, "bias", shape=(output_channels,)),
        "expected_pos": _require_array(
            arrays, "expected_pos", shape=output_shape
        ),
        "expected_neg": _require_array(
            arrays, "expected_neg", shape=output_shape
        ),
    }
    return {"arrays": normalized, "route_identity": route}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".npz", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _profiler_events(profiler) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for event in profiler.events():
        device_index = getattr(event, "device_index", None)
        events.append(
            {
                "name": str(event.name),
                "device_type": str(event.device_type),
                "device_index": (
                    None if device_index is None else int(device_index)
                ),
                "input_shapes": event.input_shapes,
                "cpu_time_total_us": float(event.cpu_time_total),
                "device_time_total_us": float(
                    getattr(event, "device_time_total", 0.0)
                ),
            }
        )
    return events


def _require_cuda_profiler_device(
    events: list[dict[str, Any]], *, expected_device_index: int
) -> None:
    cuda_events = [
        event
        for event in events
        if "cuda" in str(event.get("device_type", "")).lower()
    ]
    if not cuda_events:
        raise ValueError("Torch profiler captured no CUDA event")
    unexpected = [
        event
        for event in cuda_events
        if event.get("device_index") != expected_device_index
    ]
    if unexpected:
        indices = sorted({event.get("device_index") for event in unexpected})
        raise ValueError(
            "Torch profiler CUDA event device mismatch: expected "
            f"{expected_device_index}, got {indices}"
        )


def _default_probe_rows(rows: int) -> np.ndarray:
    return np.unique(np.linspace(0, rows - 1, num=16, dtype=np.int64))


def _prefix_indices(reduction: int) -> range:
    if reduction <= 0:
        raise ValueError("terminal reduction width must be positive")
    return range(reduction + 1)


def _prefix_ladder(
    torch,
    *,
    pos,
    neg,
    weight,
    bias,
    selected_rows,
) -> tuple[np.ndarray, np.ndarray]:
    probe_weight = torch.zeros_like(weight)
    pos_outputs = []
    neg_outputs = []
    with torch.no_grad():
        for prefix in _prefix_indices(int(weight.shape[1])):
            if prefix:
                probe_weight[:, prefix - 1].copy_(weight[:, prefix - 1])
            pos_outputs.append(
                torch.nn.functional.linear(pos, probe_weight, bias).index_select(
                    0, selected_rows
                )
            )
            neg_outputs.append(
                torch.nn.functional.linear(neg, probe_weight, bias).index_select(
                    0, selected_rows
                )
            )
        pos_stack = torch.stack(pos_outputs).cpu().numpy().astype(
            np.float32, copy=False
        )
        neg_stack = torch.stack(neg_outputs).cpu().numpy().astype(
            np.float32, copy=False
        )
    return pos_stack, neg_stack


def run(args: argparse.Namespace) -> int:
    input_path = Path(args.input).resolve()
    output_npz = Path(args.output_npz).resolve()
    output_json = Path(args.output_json).resolve()
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "failure_phase": None,
        "last_trustworthy_phase": "invocation",
        "primary_output_status": "not_started",
        "requested": {
            "input": str(input_path),
            "input_sha256": args.expected_input_sha256,
            "source_recurrence_sha256": args.expected_source_recurrence_sha256,
            "mlx_exact_state_trace_sha256": (
                args.expected_mlx_exact_state_trace_sha256
            ),
            "model_checkpoint_sha256": args.expected_model_checkpoint_sha256,
            "device": EXPECTED_DEVICE,
            "torch": EXPECTED_TORCH,
            "prefix_count": INPUT_CHANNELS + 1,
        },
        "phase_timings": {},
    }
    phase = "write_running_report"
    output_npz.unlink(missing_ok=True)
    _atomic_json(output_json, report)
    started = time.perf_counter()
    try:
        phase = "preflight_input"
        phase_started = time.perf_counter()
        actual_input_sha256 = sha256_file(input_path)
        if actual_input_sha256 != args.expected_input_sha256:
            raise ValueError(
                "input SHA256 mismatch: requested "
                f"{args.expected_input_sha256}, got {actual_input_sha256}"
            )
        admitted = validate_witness_arrays(
            _load_npz(input_path),
            expected_source_recurrence_sha256=(
                args.expected_source_recurrence_sha256
            ),
            expected_mlx_exact_state_trace_sha256=(
                args.expected_mlx_exact_state_trace_sha256
            ),
            expected_model_checkpoint_sha256=(
                args.expected_model_checkpoint_sha256
            ),
        )
        report["effective_input"] = {
            "path": str(input_path),
            "sha256": actual_input_sha256,
            "route_identity": admitted["route_identity"],
        }
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        report["last_trustworthy_phase"] = phase
        _atomic_json(output_json, report)

        phase = "cuda_identity"
        phase_started = time.perf_counter()
        import torch

        if torch.__version__ != EXPECTED_TORCH:
            raise RuntimeError(
                f"Torch must be {EXPECTED_TORCH}, got {torch.__version__}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.cuda.set_device(CUDA_DEVICE_INDEX)
        device_name = torch.cuda.get_device_name(CUDA_DEVICE_INDEX)
        if device_name != EXPECTED_DEVICE:
            raise RuntimeError(
                f"CUDA device must be {EXPECTED_DEVICE}, got {device_name}"
            )
        device_properties = torch.cuda.get_device_properties(CUDA_DEVICE_INDEX)
        report["effective_cuda"] = {
            "device": device_name,
            "device_index": CUDA_DEVICE_INDEX,
            "compute_capability": [
                int(device_properties.major),
                int(device_properties.minor),
            ],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        }
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        report["last_trustworthy_phase"] = phase
        _atomic_json(output_json, report)

        phase = "self_authenticate_source_linear"
        phase_started = time.perf_counter()
        arrays = admitted["arrays"]
        tensors = {
            name: torch.from_numpy(value).to(device=CUDA_DEVICE)
            for name, value in arrays.items()
        }
        with torch.no_grad():
            actual_pos = torch.nn.functional.linear(
                tensors["pos_final_norm"], tensors["weight"], tensors["bias"]
            )
            actual_neg = torch.nn.functional.linear(
                tensors["neg_final_norm"], tensors["weight"], tensors["bias"]
            )
            pos_exact = torch.equal(actual_pos, tensors["expected_pos"])
            neg_exact = torch.equal(actual_neg, tensors["expected_neg"])
        if not pos_exact or not neg_exact:
            raise ValueError(
                "live CUDA F.linear does not reproduce both admitted source outputs "
                f"exactly: pos={pos_exact}, neg={neg_exact}"
            )
        report["self_authentication"] = {
            "pos_exact_source": True,
            "neg_exact_source": True,
        }
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        report["last_trustworthy_phase"] = phase
        _atomic_json(output_json, report)

        phase = "profile_source_linear"
        phase_started = time.perf_counter()
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        with torch.profiler.profile(
            activities=activities, record_shapes=True
        ) as profiler:
            with torch.no_grad():
                torch.nn.functional.linear(
                    tensors["pos_final_norm"],
                    tensors["weight"],
                    tensors["bias"],
                )
            torch.cuda.synchronize(device=CUDA_DEVICE_INDEX)
        profiler_events = _profiler_events(profiler)
        _require_cuda_profiler_device(
            profiler_events, expected_device_index=CUDA_DEVICE_INDEX
        )
        report["profiler_events"] = profiler_events
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        report["last_trustworthy_phase"] = phase
        _atomic_json(output_json, report)

        phase = "capture_prefix_ladder"
        phase_started = time.perf_counter()
        selected_rows_np = _default_probe_rows(EXPECTED_ROWS)
        selected_rows = torch.from_numpy(selected_rows_np).to(
            device=CUDA_DEVICE
        )
        prefix_pos, prefix_neg = _prefix_ladder(
            torch,
            pos=tensors["pos_final_norm"],
            neg=tensors["neg_final_norm"],
            weight=tensors["weight"],
            bias=tensors["bias"],
            selected_rows=selected_rows,
        )
        expected_ladder_shape = (
            INPUT_CHANNELS + 1,
            selected_rows_np.size,
            OUTPUT_CHANNELS,
        )
        if prefix_pos.shape != expected_ladder_shape or prefix_neg.shape != expected_ladder_shape:
            raise ValueError(
                "prefix ladder shape mismatch: "
                f"pos={prefix_pos.shape}, neg={prefix_neg.shape}, "
                f"expected={expected_ladder_shape}"
            )
        if not np.array_equal(
            prefix_pos[-1], arrays["expected_pos"][selected_rows_np]
        ) or not np.array_equal(
            prefix_neg[-1], arrays["expected_neg"][selected_rows_np]
        ):
            raise ValueError("terminal prefix row does not reproduce source output")
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        report["last_trustworthy_phase"] = phase
        _atomic_json(output_json, report)

        phase = "write_primary"
        phase_started = time.perf_counter()
        witness_identity = {
            "schema": SCHEMA,
            "input_sha256": actual_input_sha256,
            "source_recurrence_sha256": (
                args.expected_source_recurrence_sha256
            ),
            "mlx_exact_state_trace_sha256": (
                args.expected_mlx_exact_state_trace_sha256
            ),
            "model_checkpoint_sha256": (
                args.expected_model_checkpoint_sha256
            ),
            "cuda": report["effective_cuda"],
            "prefix_count": INPUT_CHANNELS + 1,
            "selected_rows": selected_rows_np.tolist(),
            "self_authentication": report["self_authentication"],
        }
        _atomic_npz(
            output_npz,
            prefix_indices=np.arange(INPUT_CHANNELS + 1, dtype=np.int32),
            selected_rows=selected_rows_np,
            prefix_pos=prefix_pos,
            prefix_neg=prefix_neg,
            route_identity_json=np.asarray(
                json.dumps(admitted["route_identity"], sort_keys=True)
            ),
            witness_identity_json=np.asarray(
                json.dumps(witness_identity, sort_keys=True)
            ),
        )
        primary_sha256 = sha256_file(output_npz)
        report.update(
            {
                "status": "done",
                "primary_output_status": "written",
                "primary_output": {
                    "path": str(output_npz),
                    "sha256": primary_sha256,
                    "prefix_count": INPUT_CHANNELS + 1,
                    "selected_rows": selected_rows_np.tolist(),
                },
                "last_trustworthy_phase": "primary_validated",
            }
        )
        report["phase_timings"][phase] = time.perf_counter() - phase_started
        report["elapsed_seconds"] = time.perf_counter() - started
        _atomic_json(output_json, report)
        return 0
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "failure_phase": phase,
                "primary_output_status": "not_written",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        _atomic_json(output_json, report)
        output_npz.unlink(missing_ok=True)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-source-recurrence-sha256", required=True)
    parser.add_argument(
        "--expected-mlx-exact-state-trace-sha256", required=True
    )
    parser.add_argument("--expected-model-checkpoint-sha256", required=True)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
