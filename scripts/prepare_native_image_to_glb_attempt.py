#!/usr/bin/env python3
"""Build and validate one structured native image-to-GLB CUDA packet."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

from scripts.source_cuda_native_image_to_glb_witness import (
    prepare_native_image_to_glb_packet,
)
from trellmlx.native_image_to_glb_attempt import (
    ATTEMPT_MANIFEST,
    AttemptSpecError,
    build_attempt_packet,
    load_attempt_spec_bytes,
    validate_attempt_topology,
)


def prepare_attempt_from_path(
    spec_path: Path,
    report_path: Path,
    *,
    after_read=None,
) -> dict:
    spec_path = Path(spec_path).resolve()
    report_path = Path(report_path).resolve()
    effective_report_path = (
        _failure_report_path(
            spec_path,
            report_path,
            protected=(spec_path,),
            managed=(),
        )
        if _paths_overlap(report_path, spec_path)
        else report_path
    )

    def identity(path: Path) -> dict:
        if not path.is_file():
            return {"path": str(path), "sha256": None, "size_bytes": None}
        data = path.read_bytes()
        return {
            "path": str(path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }

    try:
        spec_bytes = spec_path.read_bytes()
    except OSError as exc:
        report = {
            "schema": "trellis2mlx.native_image_to_glb_attempt_preparation.v1",
            "status": "failed",
            "failure_phase": "spec_read",
            "spec": identity(spec_path),
            "final_spec": identity(spec_path),
            "packet": None,
            "requested_report_path": str(report_path),
            "effective_report_path": str(effective_report_path),
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_report(effective_report_path, report)
        return report

    initial = {
        "path": str(spec_path),
        "sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "size_bytes": len(spec_bytes),
    }
    report = {
        "schema": "trellis2mlx.native_image_to_glb_attempt_preparation.v1",
        "status": "running",
        "failure_phase": None,
        "spec": initial,
        "final_spec": initial,
        "packet": None,
        "requested_report_path": str(report_path),
        "effective_report_path": str(effective_report_path),
    }
    if after_read is not None:
        after_read()
        final = identity(spec_path)
        report["final_spec"] = final
        if final != initial:
            report.update(
                status="failed",
                failure_phase="spec_identity_changed",
                error="attempt spec changed after its admitted byte snapshot was read",
            )
            _write_report(effective_report_path, report)
            return report
    phase = "spec_load"
    try:
        spec = load_attempt_spec_bytes(spec_bytes, source_path=spec_path)
        validate_attempt_topology(spec)
        managed = (spec.capsule_dir.resolve(), spec.output_dir.resolve())
        assets = (
            spec.entrypoint,
            spec.authority_helper,
            spec.image,
            *spec.dinov3_files.values(),
            *spec.rembg_files.values(),
        )
        protected = (spec_path, *(Path(asset.source).resolve() for asset in assets))
        unsafe_report = any(_paths_overlap(report_path, path) for path in protected)
        unsafe_report = unsafe_report or any(
            report_path == root or root in report_path.parents for root in managed
        )
        unsafe_spec = any(
            spec_path == root or root in spec_path.parents for root in managed
        )
        if unsafe_report:
            effective_report_path = _failure_report_path(
                spec_path,
                report_path,
                protected=protected,
                managed=managed,
            )
            report["effective_report_path"] = str(effective_report_path)
            raise AttemptSpecError(
                "attempt report path aliases a protected spec/asset or managed topology"
            )
        if unsafe_spec:
            raise AttemptSpecError(
                "attempt spec path is inside a managed capsule/output topology"
            )
        final = identity(spec_path)
        report["final_spec"] = final
        if final != initial:
            raise _SpecIdentityChanged(
                "attempt spec changed after its admitted byte snapshot was read"
            )
        _write_report(effective_report_path, report)
        with tempfile.TemporaryDirectory(
            prefix="native-image-to-glb-attempt-",
            dir=spec_path.parent,
        ) as temporary:
            staging_root = Path(temporary)
            staged_spec = replace(
                spec,
                capsule_dir=staging_root / "capsule",
                output_dir=staging_root / "packet",
            )
            phase = "capsule_build"
            packet = build_attempt_packet(staged_spec)
            phase = "packet_prepare"
            packet = prepare_native_image_to_glb_packet(packet)
            final = identity(spec_path)
            report["final_spec"] = final
            if final != initial:
                raise _SpecIdentityChanged(
                    "attempt spec changed during packet construction"
                )
            phase = "packet_publish"
            publication = _publish_pair(
                packet.capsule_dir,
                packet.output_dir,
                spec.capsule_dir,
                spec.output_dir,
                identity_guard=lambda: identity(spec_path) == initial,
            )
            try:
                packet = replace(
                    packet,
                    capsule_dir=spec.capsule_dir,
                    output_dir=spec.output_dir,
                )
                attempt_path = packet.capsule_dir / ATTEMPT_MANIFEST
                witness_manifest = packet.dataset_dir / "witness-manifest.json"
                terminal_spec = identity(spec_path)
                report["final_spec"] = terminal_spec
                if terminal_spec != initial:
                    raise _SpecIdentityChanged(
                        "attempt spec changed after packet installation"
                    )
                packet_report = {
                    "run_id": packet.run_id,
                    "dataset_id": packet.dataset_id,
                    "kernel_id": packet.kernel_id,
                    "capsule_dir": str(packet.capsule_dir),
                    "output_dir": str(packet.output_dir),
                    "attempt_manifest": identity(attempt_path),
                    "input_manifest": identity(witness_manifest),
                }
                publication.commit()
            except BaseException:
                publication.rollback()
                raise
        report.update(
            status="passed",
            failure_phase=None,
            final_spec=terminal_spec,
            packet=packet_report,
        )
    except BaseException as exc:
        report.update(
            status="failed",
            failure_phase=(
                "spec_identity_changed"
                if isinstance(exc, _SpecIdentityChanged)
                else phase
            ),
            final_spec=identity(spec_path),
            error=f"{type(exc).__name__}: {exc}",
        )
    _write_report(effective_report_path, report)
    return report


class _SpecIdentityChanged(RuntimeError):
    pass


@dataclass
class _PairPublication:
    backups: dict[Path, Path]
    installed: tuple[Path, ...]
    terminal: bool = False

    def commit(self) -> None:
        if self.terminal:
            raise RuntimeError("pair publication is already terminal")
        for backup in self.backups.values():
            if backup.is_dir():
                shutil.rmtree(backup)
            else:
                backup.unlink(missing_ok=True)
        self.terminal = True

    def rollback(self) -> None:
        if self.terminal:
            return
        for final in reversed(self.installed):
            if final.is_dir():
                shutil.rmtree(final)
            else:
                final.unlink(missing_ok=True)
        for final, backup in self.backups.items():
            os.replace(backup, final)
        self.terminal = True


def _paths_overlap(left: Path, right: Path) -> bool:
    left = Path(left).resolve()
    right = Path(right).resolve()
    return left == right or left in right.parents or right in left.parents


def _failure_report_path(
    spec_path: Path,
    report_path: Path,
    *,
    protected: tuple[Path, ...],
    managed: tuple[Path, ...],
) -> Path:
    token = hashlib.sha256(f"{spec_path}\0{report_path}".encode()).hexdigest()[:16]
    candidates = (
        spec_path.parent / f".{report_path.name}.preparation-failure-{token}.json",
        Path(tempfile.gettempdir())
        / "trellis2mlx-attempt-failures"
        / f"preparation-failure-{token}.json",
    )
    for candidate in candidates:
        candidate = candidate.resolve()
        if any(_paths_overlap(candidate, path) for path in protected):
            continue
        if any(candidate == root or root in candidate.parents for root in managed):
            continue
        return candidate
    raise AttemptSpecError("no non-overlapping failure-report coordinate is available")


def _publish_pair(
    staged_capsule: Path,
    staged_output: Path,
    final_capsule: Path,
    final_output: Path,
    *,
    identity_guard,
) -> _PairPublication:
    staged = (Path(staged_capsule), Path(staged_output))
    finals = (Path(final_capsule).resolve(), Path(final_output).resolve())
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    for path in finals:
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for final in finals:
            if final.exists():
                backup = final.with_name(f".{final.name}.backup-{uuid.uuid4().hex}")
                os.replace(final, backup)
                backups[final] = backup
        for source, final in zip(staged, finals, strict=True):
            os.replace(source, final)
            installed.append(final)
        if not identity_guard():
            raise _SpecIdentityChanged("attempt spec changed during packet publication")
    except BaseException:
        for final in reversed(installed):
            if final.is_dir():
                shutil.rmtree(final)
            else:
                final.unlink(missing_ok=True)
        for final, backup in backups.items():
            os.replace(backup, final)
        raise
    return _PairPublication(backups=backups, installed=tuple(installed))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(path: Path, report: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = prepare_attempt_from_path(args.spec, args.report)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
