"""Diagnostic shape-flow block tensor injection helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ShapeBlockInjection:
    trace_path: Path | None
    array_key: str
    branch: str
    step_index: int
    block_index: int
    stage: str
    arrays_by_branch: dict[str, np.ndarray]
    source_shapes_by_branch: dict[str, tuple[int, ...]]
    trace_identity: dict[str, Any]
    source_delta_scale: float = 1.0

    def applies(self, *, step_index: int, branch: str) -> bool:
        return int(step_index) == self.step_index and (
            self.branch == "both" or self.branch == branch
        )

    def array_for_branch(self, branch: str) -> np.ndarray:
        if branch in self.arrays_by_branch:
            return self.arrays_by_branch[branch]
        if self.branch != "both" and self.branch in self.arrays_by_branch:
            return self.arrays_by_branch[self.branch]
        raise KeyError(f"shape block injection has no array for branch {branch!r}")

    def report_identity(self) -> dict[str, Any]:
        return {
            "trace_path": str(self.trace_path) if self.trace_path is not None else None,
            "trace_sha256": _sha256_file(self.trace_path) if self.trace_path else None,
            "array_key": self.array_key,
            "branch": self.branch,
            "step_index": self.step_index,
            "block_index": self.block_index,
            "stage": self.stage,
            "source_delta_scale": self.source_delta_scale,
            "trace_identity": self.trace_identity,
            "comparison_class": "mlx_shape_flow_with_source_cuda_attention_raw_injection",
            "route_identity_evidence": True,
            "source_array_shape_by_branch": {
                name: list(shape) for name, shape in sorted(self.source_shapes_by_branch.items())
            },
            "effective_array_shape_by_branch": {
                name: list(array.shape) for name, array in sorted(self.arrays_by_branch.items())
            },
        }


def load_shape_block_injection(
    trace_path: str | Path,
    *,
    branch: str,
    step_index: int,
    block_index: int,
    stage: str = "attention_raw",
    array_key: str | None = None,
    source_delta_scale: float = 1.0,
) -> ShapeBlockInjection:
    if branch not in {"pos", "neg", "both"}:
        raise ValueError(f"shape block injection branch must be pos, neg, or both; got {branch!r}")
    if stage != "attention_raw":
        raise ValueError(
            "shape block injection currently accepts only attention_raw; "
            f"got {stage!r}"
        )
    if not math.isfinite(source_delta_scale):
        raise ValueError(f"shape block injection source delta scale must be finite, got {source_delta_scale}")

    trace_path = Path(trace_path)
    with np.load(trace_path, allow_pickle=False) as trace:
        branches = ("pos", "neg") if branch == "both" else (branch,)
        arrays_by_branch: dict[str, np.ndarray] = {}
        source_shapes_by_branch: dict[str, tuple[int, ...]] = {}
        keys: list[str] = []
        for active_branch in branches:
            key = array_key or f"{active_branch}_block{block_index}_{stage}"
            if array_key and branch == "both":
                raise ValueError(
                    "an explicit shape block injection array key cannot identify both CFG branches"
                )
            if key not in trace.files:
                raise KeyError(
                    f"shape block injection trace {trace_path} has no array {key!r}; "
                    f"available keys: {sorted(trace.files)}"
                )
            source = np.asarray(trace[key], dtype=np.float32)
            source_shapes_by_branch[active_branch] = source.shape
            arrays_by_branch[active_branch] = _normalize_attention_raw(source, key=key)
            keys.append(key)
        trace_identity = _read_trace_identity(trace)
        effective_route = trace_identity.get("effective_route")
        if not isinstance(effective_route, str) or not effective_route:
            raise ValueError("shape block injection route_identity_json has no effective_route")
        if trace_identity.get("effective_device_type") != "cuda":
            raise ValueError(
                "shape block injection requires effective_device_type='cuda', got "
                f"{trace_identity.get('effective_device_type')!r}"
            )
        source_block_index = _required_scalar(trace, "trace_block_index")
        if int(source_block_index) != int(block_index):
            raise ValueError(
                f"shape block injection trace block index {int(source_block_index)} "
                f"does not match requested {int(block_index)}"
            )
        source_step_index = _required_scalar(trace, "shape_flow_trace_step_index")
        if int(source_step_index) != int(step_index):
            raise ValueError(
                f"shape block injection trace step index {int(source_step_index)} "
                f"does not match requested {int(step_index)}"
            )

    return ShapeBlockInjection(
        trace_path=trace_path,
        array_key=",".join(keys),
        branch=branch,
        step_index=int(step_index),
        block_index=int(block_index),
        stage=stage,
        arrays_by_branch=arrays_by_branch,
        source_shapes_by_branch=source_shapes_by_branch,
        trace_identity=trace_identity,
        source_delta_scale=float(source_delta_scale),
    )


def _normalize_attention_raw(array: np.ndarray, *, key: str) -> np.ndarray:
    if array.ndim == 4 and array.shape[0] == 1:
        return array.reshape(array.shape[0], array.shape[1], array.shape[2] * array.shape[3])
    if array.ndim == 3 and array.shape[0] == 1:
        return array
    raise ValueError(
        f"shape attention_raw array {key!r} must have shape [1,N,H,D] or [1,N,C], "
        f"got {array.shape}"
    )


def _read_trace_identity(trace: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "route_identity_json" not in trace.files:
        raise ValueError("shape block injection trace has no route_identity_json")
    raw = np.asarray(trace["route_identity_json"])
    if raw.size != 1:
        raise ValueError("route_identity_json must contain exactly one JSON value")
    value = raw.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    identity = json.loads(str(value))
    if not isinstance(identity, dict):
        raise ValueError("route_identity_json must decode to an object")
    return identity


def _required_scalar(trace: np.lib.npyio.NpzFile, key: str) -> Any:
    if key not in trace.files:
        raise ValueError(f"shape block injection trace has no {key}")
    value = np.asarray(trace[key])
    if value.size != 1:
        raise ValueError(f"shape block injection trace {key} must contain exactly one value")
    return value.reshape(-1)[0]


def _sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
