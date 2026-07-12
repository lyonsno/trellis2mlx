"""Diagnostic sparse-flow block tensor injection helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SparseBlockInjection:
    trace_path: Path | None
    array_key: str
    branch: str
    step_index: int
    block_index: int
    stage: str
    arrays_by_branch: dict[str, np.ndarray]
    trace_identity: dict[str, Any]

    def applies(self, *, step_index: int, branch: str) -> bool:
        return int(step_index) == self.step_index and (
            self.branch == "both" or self.branch == branch
        )

    def array_for_branch(self, branch: str) -> np.ndarray:
        if branch in self.arrays_by_branch:
            return self.arrays_by_branch[branch]
        if self.branch != "both" and self.branch in self.arrays_by_branch:
            return self.arrays_by_branch[self.branch]
        raise KeyError(f"sparse block injection has no array for branch {branch!r}")

    def report_identity(self) -> dict[str, Any]:
        identity = {
            "trace_path": str(self.trace_path) if self.trace_path is not None else None,
            "trace_sha256": _sha256_file(self.trace_path) if self.trace_path else None,
            "array_key": self.array_key,
            "branch": self.branch,
            "step_index": self.step_index,
            "block_index": self.block_index,
            "stage": self.stage,
            "trace_identity": self.trace_identity,
            "comparison_class": "mlx_sparse_flow_with_named_block_tensor_injection",
            "route_identity_evidence": True,
            "array_shape_by_branch": {
                key: list(value.shape) for key, value in sorted(self.arrays_by_branch.items())
            },
            "array_dtype_by_branch": {
                key: str(value.dtype) for key, value in sorted(self.arrays_by_branch.items())
            },
        }
        return identity


@dataclass(frozen=True)
class SparseBlockInjectionSet:
    trace_path: Path | None
    sites: tuple[SparseBlockInjection, ...]
    manifest_identity: dict[str, Any]

    def applies(self, *, step_index: int, branch: str) -> bool:
        return any(site.applies(step_index=step_index, branch=branch) for site in self.sites)

    def active_for_step_branch(self, *, step_index: int, branch: str) -> "SparseBlockInjectionSet | None":
        active = tuple(
            site for site in self.sites if site.applies(step_index=step_index, branch=branch)
        )
        if not active:
            return None
        return SparseBlockInjectionSet(
            trace_path=self.trace_path,
            sites=active,
            manifest_identity=self.manifest_identity,
        )

    def injection_for_block(self, block_index: int) -> SparseBlockInjection | None:
        matches = [site for site in self.sites if site.block_index == block_index]
        if not matches:
            return None
        if len(matches) > 1:
            raise ValueError(
                f"sparse block injection manifest has multiple active sites for block {block_index}"
            )
        return matches[0]

    def report_identity(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.trace_path) if self.trace_path is not None else None,
            "manifest_sha256": _sha256_file(self.trace_path) if self.trace_path else None,
            "comparison_class": "mlx_sparse_flow_with_named_block_tensor_injection_set",
            "route_identity_evidence": True,
            "manifest_identity": self.manifest_identity,
            "sites": [site.report_identity() for site in self.sites],
        }


def load_sparse_block_injection(
    trace_path: str | Path,
    *,
    branch: str,
    step_index: int,
    block_index: int,
    stage: str,
    array_key: str | None,
) -> SparseBlockInjection:
    if branch not in {"pos", "neg", "both"}:
        raise ValueError("branch must be one of: pos, neg, both")
    if stage not in {"norm1", "modulated_self_input", "after_self"}:
        raise ValueError("stage must be one of: norm1, modulated_self_input, after_self")

    trace_path = Path(trace_path)
    with np.load(trace_path) as trace:
        if branch == "both" and array_key is None:
            branch_keys = {
                name: f"{name}_block{block_index}_{stage}" for name in ("pos", "neg")
            }
            missing = [key for key in branch_keys.values() if key not in trace]
            if missing:
                raise KeyError(
                    "sparse block injection trace missing required array(s): "
                    + ", ".join(repr(key) for key in missing)
                )
            selected_key = ",".join(branch_keys[name] for name in ("pos", "neg"))
            arrays_by_branch = {
                name: np.asarray(trace[key], dtype=np.float32)
                for name, key in branch_keys.items()
            }
        else:
            selected_key = array_key or f"{branch}_block{block_index}_{stage}"
            if selected_key not in trace:
                raise KeyError(
                    f"sparse block injection trace missing required array {selected_key!r}"
                )
            selected = np.asarray(trace[selected_key], dtype=np.float32)
            if branch == "both":
                arrays_by_branch = {"pos": selected, "neg": selected}
            else:
                arrays_by_branch = {branch: selected}

        trace_identity: dict[str, Any] = {}
        if "route_identity_json" in trace:
            raw_identity = np.asarray(trace["route_identity_json"])
            if raw_identity.shape == ():
                trace_identity = json.loads(str(raw_identity.item()))

    return SparseBlockInjection(
        trace_path=trace_path,
        array_key=selected_key,
        branch=branch,
        step_index=int(step_index),
        block_index=int(block_index),
        stage=stage,
        arrays_by_branch=arrays_by_branch,
        trace_identity=trace_identity,
    )


def load_sparse_block_injection_manifest(manifest_path: str | Path) -> SparseBlockInjectionSet:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    sites_raw = manifest.get("sites")
    if not isinstance(sites_raw, list) or not sites_raw:
        raise ValueError("sparse block injection manifest must contain a non-empty 'sites' list")

    sites: list[SparseBlockInjection] = []
    for index, site in enumerate(sites_raw):
        if not isinstance(site, dict):
            raise ValueError(f"manifest site {index} must be an object")
        try:
            trace_path = Path(site["trace_path"])
            branch = site["branch"]
            step_index = site["step_index"]
            block_index = site["block_index"]
            stage = site["stage"]
        except KeyError as exc:
            raise KeyError(f"manifest site {index} missing required key {exc.args[0]!r}") from exc
        if not trace_path.is_absolute():
            trace_path = manifest_path.parent / trace_path
        sites.append(
            load_sparse_block_injection(
                trace_path,
                branch=branch,
                step_index=int(step_index),
                block_index=int(block_index),
                stage=stage,
                array_key=site.get("array_key"),
            )
        )

    manifest_identity = {
        key: value for key, value in manifest.items() if key != "sites"
    }
    return SparseBlockInjectionSet(
        trace_path=manifest_path,
        sites=tuple(sites),
        manifest_identity=manifest_identity,
    )


def _sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
