#!/usr/bin/env python3
"""Measure PyMeshLab continuation endpoints under exact source custody."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
from typing import Any

import numpy as np

from trellmlx.glb_aabb_crop import open_triangle_glb, sha256_file, write_geometry_glb


ROUTE = "pymeshlab-topology-preserving-continuation-v1"
HARNESS_PATH = Path(__file__).resolve()
MESHLAB_COMMIT = "dc48b91ae562756a6988048c5d5c7f1d2b687256"
VCGLIB_COMMIT = "c94ef4e12e9ea3ae986d9af91005be8328d13719"
FILTER_CONFIG = {
    "targetperc": 0.0,
    "qualitythr": 0.3,
    "preserveboundary": False,
    "boundaryweight": 1.0,
    "preservenormal": False,
    "preservetopology": True,
    "optimalplacement": True,
    "planarquadric": False,
    "planarweight": 0.001,
    "qualityweight": False,
    "autoclean": True,
    "selected": False,
}


class AssayError(RuntimeError):
    def __init__(self, phase: str, message: str):
        super().__init__(message)
        self.phase = phase


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--target-faces", required=True, nargs="+", type=int)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--expected-pymeshlab-version", required=True)
    parser.add_argument(
        "--lineage-ref",
        action="append",
        required=True,
        nargs=3,
        metavar=("LABEL", "PATH", "EXPECTED_SHA256"),
    )
    return parser.parse_args(argv)


def _import_pymeshlab():
    return importlib.import_module("pymeshlab")


def _same_or_within(path: Path, directory: Path) -> bool:
    resolved = path.resolve()
    root = directory.resolve()
    return resolved == root or resolved.is_relative_to(root)


def _protected_paths(args: argparse.Namespace) -> list[Path]:
    return [args.input, *(Path(ref[1]) for ref in args.lineage_ref)]


def _effective_report_path(args: argparse.Namespace) -> tuple[Path, bool]:
    protected = [path.resolve() for path in _protected_paths(args)]
    output_dir = args.output_dir.resolve()
    requested = args.report.resolve()
    if requested not in protected and not _same_or_within(requested, output_dir):
        return args.report, False

    candidate = args.input.with_name(args.input.name + ".assay-error.json")
    while candidate.resolve() in protected or _same_or_within(candidate, output_dir):
        candidate = candidate.with_name(candidate.name + ".assay-error.json")
    return candidate, True


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _request(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "input": str(args.input),
        "expected_input_sha256": args.expected_input_sha256,
        "output_dir": str(args.output_dir),
        "report": str(args.report),
        "target_faces": list(args.target_faces),
        "repeats": args.repeats,
        "expected_pymeshlab_version": args.expected_pymeshlab_version,
        "lineage_refs": [
            {"label": label, "path": path, "expected_sha256": digest}
            for label, path, digest in args.lineage_ref
        ],
    }


def _validate_paths(args: argparse.Namespace, report_path: Path) -> None:
    protected = [path.resolve() for path in _protected_paths(args)]
    output_dir = args.output_dir.resolve()
    if len(set(protected)) != len(protected):
        raise AssayError("validate_paths", "input and lineage paths must be distinct")
    if args.report.resolve() in protected:
        raise AssayError("validate_paths", "requested report must not replace protected evidence")
    if report_path.resolve() in protected:
        raise AssayError("validate_paths", "effective report must not replace protected evidence")
    if _same_or_within(report_path, output_dir):
        raise AssayError("validate_paths", "effective report must not be inside output directory")
    for path in protected:
        if path == output_dir or path.is_relative_to(output_dir):
            raise AssayError("validate_paths", "output directory must not contain protected evidence")


def _clean_output_dir(path: Path) -> None:
    if path.exists():
        if not path.is_dir() or path.is_symlink():
            raise AssayError("prepare_outputs", f"output surface is not a directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _source_counts(path: Path) -> tuple[int, int]:
    try:
        with open_triangle_glb(path) as view:
            return int(len(view.vertices)), int(len(view.faces))
    except Exception as exc:
        raise AssayError("load_source", str(exc)) from exc


def _module_identity(package: Any) -> dict[str, Any]:
    module_path = Path(package.__file__) if getattr(package, "__file__", None) else None
    module_sha256 = None
    if module_path is not None and module_path.is_file():
        module_sha256 = sha256_file(module_path)
    package_version = getattr(package, "__version__", None)
    version_source = "module-attribute" if package_version is not None else None
    if package_version is None:
        try:
            package_version = importlib.metadata.version("pymeshlab")
            version_source = "distribution-metadata"
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "package": "pymeshlab",
        "distribution": "pymeshlab",
        "package_version": package_version,
        "package_version_source": version_source,
        "module_path": str(module_path) if module_path is not None else None,
        "module_sha256": module_sha256,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def _save_mesh(path: Path, mesh: Any) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertex_matrix(), dtype=np.float64)
    faces = np.asarray(mesh.face_matrix(), dtype=np.uint32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise AssayError("simplify", f"unexpected vertex shape: {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise AssayError("simplify", f"unexpected face shape: {faces.shape}")
    if not np.isfinite(vertices).all():
        raise AssayError("simplify", "simplified vertices contain non-finite values")
    write_geometry_glb(path, vertices.astype(np.float32), faces)
    return {
        "achieved_vertices": int(len(vertices)),
        "achieved_faces": int(len(faces)),
        "vertex_array_sha256": _array_sha256(vertices),
        "face_array_sha256": _array_sha256(faces),
        "mesh_sha256": sha256_file(path),
        "mesh_path": str(path),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    effective_report, report_rerouted = _effective_report_path(args)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "status": "failed",
        "failure_phase": "validate_paths",
        "primary_output_status": "not_started",
        "route": {
            "id": ROUTE,
            "harness_path": str(HARNESS_PATH),
            "harness_sha256": sha256_file(HARNESS_PATH),
        },
        "request": _request(args),
        "report": {
            "requested_path": str(args.report),
            "effective_path": str(effective_report),
            "rerouted": report_rerouted,
        },
        "source": None,
        "continuation_provenance": {
            "classification": "continuation-not-common-source",
            "refs": None,
        },
        "effective_filter_config": {**FILTER_CONFIG, "targetfacenum": "per-target"},
        "trajectory_semantics": {
            "target_affects": "termination-only-in-inspected-source",
            "meshlab_commit": MESHLAB_COMMIT,
            "vcglib_commit": VCGLIB_COMMIT,
            "runtime_determinism": "must-be-observed-per-specimen",
        },
        "targets": None,
        "target_contract": None,
        "repeat_stability": None,
        "trajectory_admission": None,
        "last_trustworthy_evidence": None,
        "partial_output_cleanup": None,
        "elapsed_seconds": None,
    }

    try:
        _validate_paths(args, effective_report)
        _write_json(effective_report, report)

        report["failure_phase"] = "source_identity"
        observed_source_sha256 = sha256_file(args.input)
        if observed_source_sha256 != args.expected_input_sha256:
            raise AssayError(
                "source_identity",
                f"input SHA256 mismatch: expected {args.expected_input_sha256}, observed {observed_source_sha256}",
            )
        source_vertices, source_faces = _source_counts(args.input)
        report["source"] = {
            "path": str(args.input),
            "sha256": observed_source_sha256,
            "vertices": source_vertices,
            "faces": source_faces,
        }

        report["failure_phase"] = "lineage_identity"
        lineage_records: list[dict[str, Any]] = []
        for label, raw_path, expected_sha256 in args.lineage_ref:
            path = Path(raw_path)
            observed_sha256 = sha256_file(path)
            if observed_sha256 != expected_sha256:
                raise AssayError(
                    "lineage_identity",
                    f"lineage {label!r} SHA256 mismatch: expected {expected_sha256}, observed {observed_sha256}",
                )
            lineage_records.append(
                {"label": label, "path": str(path), "sha256": observed_sha256}
            )
        report["continuation_provenance"]["refs"] = lineage_records
        _write_json(effective_report, report)

        report["failure_phase"] = "validate_targets"
        if args.repeats < 2:
            raise AssayError("validate_targets", "repeats must be at least two")
        if len(set(args.target_faces)) != len(args.target_faces):
            raise AssayError("validate_targets", "target face counts must be unique")
        if any(target <= 0 or target >= source_faces for target in args.target_faces):
            raise AssayError(
                "validate_targets",
                f"every target must be positive and below source face count {source_faces}",
            )
        targets = sorted(args.target_faces, reverse=True)

        report["failure_phase"] = "runtime_identity"
        package = _import_pymeshlab()
        report["route"].update(_module_identity(package))
        if report["route"]["package_version"] != args.expected_pymeshlab_version:
            raise AssayError(
                "runtime_identity",
                "PyMeshLab version mismatch: "
                f"expected {args.expected_pymeshlab_version}, observed {report['route']['package_version']}",
            )
        _write_json(effective_report, report)

        report["failure_phase"] = "prepare_outputs"
        _clean_output_dir(args.output_dir)
        report["primary_output_status"] = "partial"

        report["failure_phase"] = "simplify"
        target_records: list[dict[str, Any]] = []
        all_repeat_exact = True
        for target in targets:
            runs: list[dict[str, Any]] = []
            for repeat in range(1, args.repeats + 1):
                if sha256_file(args.input) != observed_source_sha256:
                    raise AssayError("source_identity", "input GLB changed during assay")
                mesh_set = package.MeshSet()
                mesh_set.load_new_mesh(str(args.input))
                run_started = time.perf_counter()
                mesh_set.meshing_decimation_quadric_edge_collapse(
                    targetfacenum=target,
                    **FILTER_CONFIG,
                )
                run_elapsed = time.perf_counter() - run_started
                mesh_path = args.output_dir / f"target-{target}-repeat-{repeat}.glb"
                run = {
                    "repeat": repeat,
                    "requested_faces": target,
                    **_save_mesh(mesh_path, mesh_set.current_mesh()),
                    "elapsed_seconds": run_elapsed,
                }
                run["target_satisfied"] = run["achieved_faces"] <= target
                runs.append(run)
                report["last_trustworthy_evidence"] = {
                    key: run[key]
                    for key in (
                        "repeat",
                        "requested_faces",
                        "achieved_vertices",
                        "achieved_faces",
                        "target_satisfied",
                        "vertex_array_sha256",
                        "face_array_sha256",
                        "mesh_sha256",
                    )
                }

            identity_fields = (
                "achieved_vertices",
                "achieved_faces",
                "vertex_array_sha256",
                "face_array_sha256",
                "mesh_sha256",
            )
            repeat_exact = all(
                all(run[field] == runs[0][field] for field in identity_fields)
                for run in runs[1:]
            )
            all_repeat_exact = all_repeat_exact and repeat_exact
            target_records.append(
                {
                    "requested_faces": target,
                    "target_satisfied": all(run["target_satisfied"] for run in runs),
                    "repeat_exact": repeat_exact,
                    "runs": runs,
                }
            )

        report["failure_phase"] = "publish"
        if sha256_file(args.input) != observed_source_sha256:
            raise AssayError("source_identity", "input GLB changed before report publication")
        for record in lineage_records:
            if sha256_file(Path(record["path"])) != record["sha256"]:
                raise AssayError(
                    "lineage_identity",
                    f"lineage {record['label']!r} changed before report publication",
                )
        unsatisfied = [
            record["requested_faces"] for record in target_records if not record["target_satisfied"]
        ]
        single_trajectory = all_repeat_exact and not unsatisfied
        report.update(
            {
                "status": "completed",
                "failure_phase": None,
                "primary_output_status": "validated",
                "targets": target_records,
                "target_contract": {
                    "all_targets_satisfied": not unsatisfied,
                    "unsatisfied_targets": unsatisfied,
                },
                "repeat_stability": {"all_exact": all_repeat_exact},
                "trajectory_admission": {
                    "repeat_exact_all_targets": all_repeat_exact,
                    "effective_single_trajectory_admitted": single_trajectory,
                    "required_interpretation": (
                        "ordered-endpoint-snapshots-under-source-proven-termination-only-target"
                        if single_trajectory
                        else "per-target-replicate-distributions"
                    ),
                },
            }
        )
    except BaseException as exc:
        report["status"] = "failed"
        report["failure_phase"] = getattr(exc, "phase", report["failure_phase"])
        report["error"] = f"{type(exc).__name__}: {exc}"
        output_dir_existed = args.output_dir.is_dir() and not args.output_dir.is_symlink()
        artifact_count = (
            sum(1 for path in args.output_dir.rglob("*") if path.is_file())
            if output_dir_existed
            else 0
        )
        removed = False
        cleanup_error = None
        if output_dir_existed:
            try:
                shutil.rmtree(args.output_dir)
                removed = True
            except Exception as cleanup_exc:  # pragma: no cover - filesystem-specific
                cleanup_error = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
        report["partial_output_cleanup"] = {
            "output_dir_existed": output_dir_existed,
            "artifact_count": artifact_count,
            "removed": removed,
            "error": cleanup_error,
        }
        if artifact_count:
            report["primary_output_status"] = (
                "partial_removed" if removed else "partial_cleanup_failed"
            )
        else:
            report["primary_output_status"] = "not_started"
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        _write_json(effective_report, report)

    if report["status"] != "completed":
        print(f"{report['failure_phase']}: {report['error']}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
