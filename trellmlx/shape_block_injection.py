"""Diagnostic shape-flow block tensor injection helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SHAPE_BLOCK_INJECTION_STAGES = {
    "norm1",
    "modulated_self_input",
    "attention_raw",
    "after_self",
    "cross_attention_raw",
    "after_cross",
    "after_mlp",
}


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
        raw_stage_classes = {
            "attention_raw": "mlx_shape_flow_with_source_cuda_attention_raw_injection",
            "cross_attention_raw": "mlx_shape_flow_with_source_cuda_cross_attention_raw_injection",
        }
        comparison_class = raw_stage_classes.get(
            self.stage, "mlx_shape_flow_with_source_cuda_block_stage_injection"
        )
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
            "comparison_class": comparison_class,
            "route_identity_evidence": True,
            "source_array_shape_by_branch": {
                name: list(shape) for name, shape in sorted(self.source_shapes_by_branch.items())
            },
            "effective_array_shape_by_branch": {
                name: list(array.shape) for name, array in sorted(self.arrays_by_branch.items())
            },
        }


@dataclass(frozen=True)
class ShapeBlockInjectionSet:
    manifest_path: Path | None
    sites: tuple[ShapeBlockInjection, ...]
    manifest_identity: dict[str, Any]

    def applies(self, *, step_index: int, branch: str) -> bool:
        return any(site.applies(step_index=step_index, branch=branch) for site in self.sites)

    def active_for_step_branch(
        self,
        *,
        step_index: int,
        branch: str,
    ) -> "ShapeBlockInjectionSet | None":
        active = tuple(
            site for site in self.sites if site.applies(step_index=step_index, branch=branch)
        )
        if not active:
            return None
        return ShapeBlockInjectionSet(
            manifest_path=self.manifest_path,
            sites=active,
            manifest_identity=self.manifest_identity,
        )

    def injection_for_block(self, block_index: int) -> ShapeBlockInjection | None:
        matches = [site for site in self.sites if site.block_index == block_index]
        if not matches:
            return None
        if len(matches) > 1:
            raise ValueError(
                f"shape block injection manifest has multiple active sites for block {block_index}"
            )
        return matches[0]

    def report_identity(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "manifest_sha256": _sha256_file(self.manifest_path) if self.manifest_path else None,
            "comparison_class": "mlx_shape_flow_with_source_cuda_block_stage_injection_set",
            "route_identity_evidence": True,
            "manifest_identity": self.manifest_identity,
            "sites": [site.report_identity() for site in self.sites],
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
    if stage not in SHAPE_BLOCK_INJECTION_STAGES:
        raise ValueError(
            f"unknown shape block injection stage {stage!r}; "
            f"expected one of {sorted(SHAPE_BLOCK_INJECTION_STAGES)}"
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
            arrays_by_branch[active_branch] = _normalize_block_stage(
                source,
                key=key,
                stage=stage,
            )
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
        source_block_indices = _trace_block_indices(trace, trace_identity)
        if int(block_index) not in source_block_indices:
            raise ValueError(
                f"shape block injection trace block indices {source_block_indices} "
                f"do not contain requested {int(block_index)}"
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


def load_shape_block_injection_manifest(
    manifest_path: str | Path,
) -> ShapeBlockInjectionSet:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    sites_raw = manifest.get("sites")
    if not isinstance(sites_raw, list) or not sites_raw:
        raise ValueError("shape block injection manifest must contain a non-empty 'sites' list")

    sites: list[ShapeBlockInjection] = []
    for index, site in enumerate(sites_raw):
        if not isinstance(site, dict):
            raise ValueError(f"shape block injection manifest site {index} must be an object")
        try:
            trace_path = Path(site["trace_path"])
            branch = site["branch"]
            step_index = int(site["step_index"])
            block_index = int(site["block_index"])
            stage = site["stage"]
        except KeyError as exc:
            raise KeyError(
                f"shape block injection manifest site {index} missing {exc.args[0]!r}"
            ) from exc
        if not trace_path.is_absolute():
            trace_path = manifest_path.parent / trace_path
        sites.append(
            load_shape_block_injection(
                trace_path,
                branch=branch,
                step_index=step_index,
                block_index=block_index,
                stage=stage,
                array_key=site.get("array_key"),
                source_delta_scale=float(site.get("source_delta_scale", 1.0)),
            )
        )

    return ShapeBlockInjectionSet(
        manifest_path=manifest_path,
        sites=tuple(sites),
        manifest_identity={key: value for key, value in manifest.items() if key != "sites"},
    )


def _normalize_block_stage(array: np.ndarray, *, key: str, stage: str) -> np.ndarray:
    if stage in {"attention_raw", "cross_attention_raw"} and array.ndim == 4 and array.shape[0] == 1:
        return array.reshape(array.shape[0], array.shape[1], array.shape[2] * array.shape[3])
    if array.ndim == 3 and array.shape[0] == 1:
        return array
    raise ValueError(
        f"shape block stage array {key!r} for {stage!r} must have shape "
        "[1,N,C], or [1,N,H,D] for raw attention stages; "
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


def _trace_block_indices(
    trace: np.lib.npyio.NpzFile,
    trace_identity: dict[str, Any],
) -> list[int]:
    if "trace_block_indices" not in trace.files:
        return [int(_required_scalar(trace, "trace_block_index"))]

    raw = np.asarray(trace["trace_block_indices"])
    if raw.size == 0 or raw.dtype.kind not in {"i", "u"}:
        raise ValueError("shape block injection trace_block_indices must contain integers")
    indices = [int(value) for value in raw.reshape(-1)]
    if len(indices) != len(set(indices)):
        raise ValueError(f"shape block injection trace_block_indices contains duplicates: {indices}")

    scalar = int(_required_scalar(trace, "trace_block_index"))
    if scalar != indices[0]:
        raise ValueError(
            f"shape block injection trace block index {scalar} does not match first "
            f"trace_block_indices value {indices[0]}"
        )

    identity_indices = trace_identity.get("shape_flow_trace_block_indices")
    if not isinstance(identity_indices, list) or identity_indices != indices:
        raise ValueError(
            f"shape block injection trace block indices {indices} do not match route identity "
            f"{identity_indices!r}"
        )
    return indices


def _sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
