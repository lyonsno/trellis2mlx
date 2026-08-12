#!/usr/bin/env python3
"""Exercise the production-shaped Trellis2MLX cleanup and local-QEM route."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from trellmlx.glb_aabb_crop import open_triangle_glb, sha256_file, write_geometry_glb


ROUTE = "trellis2mlx-cleanup-local-qem-cleanup-v1"
QEM_ROUTE = "trellis2mlx-sequential-qem-v1"
HARNESS_PATH = Path(__file__).resolve()
CLEANUP_CONFIG = {
    "max_hole_perimeter": 3e-2,
    "keep_largest": False,
    "min_component_area": 1e-5,
    "verbose": False,
}
QEM_CONFIG = {
    "lambda_edge_length": 1e-2,
    "lambda_skinny": 1e-3,
    "initial_thresh": 1e-8,
    "return_receipt": True,
    "verbose": True,
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
    parser.add_argument("--target-faces", required=True, type=int)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--continuation-classification", required=True)
    parser.add_argument("--expected-mlx-version", required=True)
    parser.add_argument("--expected-qem-module-sha256", required=True)
    parser.add_argument("--expected-cleanup-module-sha256", required=True)
    parser.add_argument(
        "--lineage-ref",
        action="append",
        required=True,
        nargs=3,
        metavar=("LABEL", "PATH", "EXPECTED_SHA256"),
    )
    return parser.parse_args(argv)


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _same_or_within(path: Path, directory: Path) -> bool:
    resolved = path.resolve()
    root = directory.resolve()
    return resolved == root or resolved.is_relative_to(root)


def _protected_paths(args: argparse.Namespace) -> list[Path]:
    return [args.input, *(Path(ref[1]) for ref in args.lineage_ref)]


def _effective_report_path(args: argparse.Namespace) -> tuple[Path, bool]:
    protected = {path.resolve() for path in _protected_paths(args)}
    output_dir = args.output_dir.resolve()
    requested = args.report.resolve()
    if requested not in protected and not _same_or_within(requested, output_dir):
        return args.report, False
    candidate_parent = output_dir.parent
    if candidate_parent == output_dir:
        raise AssayError(
            "validate_paths",
            "cannot route a failure report outside a filesystem-root output directory",
        )
    stem = args.input.name + ".assay-error"
    for suffix in range(100):
        numbered = "" if suffix == 0 else f".{suffix}"
        candidate = candidate_parent / f"{stem}{numbered}.json"
        if candidate.resolve() not in protected and not _same_or_within(
            candidate, output_dir
        ):
            return candidate, True
    raise AssayError("validate_paths", "could not select an outside-custody failure report")


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


def _load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        with open_triangle_glb(path) as view:
            vertices = np.asarray(view.vertices, dtype=np.float32).copy()
            faces = np.asarray(view.faces, dtype=np.int32).copy()
    except Exception as exc:
        raise AssayError("load_source", str(exc)) from exc
    _validate_mesh(vertices, faces, "source")
    return vertices, faces


def _validate_mesh(vertices: np.ndarray, faces: np.ndarray, stage: str) -> None:
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise AssayError(stage, f"{stage} vertices must be nonempty VEC3")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise AssayError(stage, f"{stage} faces must be nonempty triangles")
    if not np.isfinite(vertices).all():
        raise AssayError(stage, f"{stage} vertices contain non-finite values")
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise AssayError(stage, f"{stage} faces contain invalid vertex indices")


def _load_runtime() -> SimpleNamespace:
    import mlx
    from trellmlx import mesh_cleanup, simplify_qem_metal

    qem_path = Path(simplify_qem_metal.__file__).resolve()
    cleanup_path = Path(mesh_cleanup.__file__).resolve()
    return SimpleNamespace(
        cleanup=mesh_cleanup.cleanup_mesh,
        simplify=simplify_qem_metal.simplify_qem,
        identity={
            "mlx_available": bool(simplify_qem_metal.HAS_MLX),
            "mlx_version": _package_version("mlx", mlx),
            "qem_module_path": str(qem_path),
            "qem_module_sha256": sha256_file(qem_path),
            "cleanup_module_path": str(cleanup_path),
            "cleanup_module_sha256": sha256_file(cleanup_path),
        },
    )


def _package_version(distribution: str, package: Any) -> str | None:
    version = getattr(package, "__version__", None)
    if version is not None:
        return str(version)
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def _validate_runtime(args: argparse.Namespace, runtime: SimpleNamespace) -> None:
    expected = {
        "mlx_available": True,
        "mlx_version": args.expected_mlx_version,
        "qem_module_sha256": args.expected_qem_module_sha256,
        "cleanup_module_sha256": args.expected_cleanup_module_sha256,
    }
    mismatches = {
        key: {"expected": value, "observed": runtime.identity.get(key)}
        for key, value in expected.items()
        if runtime.identity.get(key) != value
    }
    if mismatches:
        raise AssayError("runtime_identity", f"runtime identity mismatch: {mismatches}")


def _validate_runtime_custody(args: argparse.Namespace, runtime: SimpleNamespace) -> None:
    output_dir = args.output_dir.resolve()
    for key in ("qem_module_path", "cleanup_module_path"):
        module_path = Path(runtime.identity[key]).resolve()
        if module_path == output_dir or module_path.is_relative_to(output_dir):
            raise AssayError(
                "runtime_custody",
                f"output directory must not contain authenticated runtime module {module_path}",
            )


def _request(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "input": str(args.input),
        "expected_input_sha256": args.expected_input_sha256,
        "output_dir": str(args.output_dir),
        "report": str(args.report),
        "target_faces": args.target_faces,
        "max_iterations": args.max_iterations,
        "continuation_classification": args.continuation_classification,
        "expected_mlx_version": args.expected_mlx_version,
        "expected_qem_module_sha256": args.expected_qem_module_sha256,
        "expected_cleanup_module_sha256": args.expected_cleanup_module_sha256,
        "lineage_refs": [
            {"label": label, "path": path, "expected_sha256": digest}
            for label, path, digest in args.lineage_ref
        ],
    }


def _stage_record(
    stage: str,
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    elapsed: float,
) -> dict[str, Any]:
    _validate_mesh(vertices, faces, stage)
    write_geometry_glb(path, vertices, faces)
    return {
        "stage": stage,
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "vertex_array_sha256": _array_sha256(vertices),
        "face_array_sha256": _array_sha256(faces),
        "mesh_path": str(path),
        "mesh_sha256": sha256_file(path),
        "elapsed_seconds": elapsed,
    }


def _revalidate_custody(
    args: argparse.Namespace,
    source_sha256: str,
    lineage_records: list[dict[str, Any]],
) -> None:
    if sha256_file(args.input) != source_sha256:
        raise AssayError("source_identity", "input GLB changed during assay")
    for record in lineage_records:
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise AssayError(
                "lineage_identity", f"lineage {record['label']!r} changed during assay"
            )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    effective_report, report_rerouted = _effective_report_path(args)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "status": "failed",
        "failure_phase": "validate_paths",
        "active_stage": None,
        "primary_output_status": "not_started",
        "route": {
            "id": ROUTE,
            "harness_path": str(HARNESS_PATH),
            "harness_sha256": sha256_file(HARNESS_PATH),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "runtime": None,
        },
        "request": _request(args),
        "report": {
            "requested_path": str(args.report),
            "effective_path": str(effective_report),
            "rerouted": report_rerouted,
        },
        "source": None,
        "continuation_provenance": {
            "classification": args.continuation_classification,
            "refs": None,
        },
        "effective_config": {
            "intermediate_cleanup": {**CLEANUP_CONFIG, "do_fix_normals": False},
            "local_qem": {
                **QEM_CONFIG,
                "target_faces": args.target_faces,
                "max_iterations": args.max_iterations,
            },
            "final_cleanup": {**CLEANUP_CONFIG, "do_fix_normals": True},
        },
        "stages": {},
        "target_contract": None,
        "claim_ceiling": None,
        "last_trustworthy_evidence": None,
        "elapsed_seconds": None,
    }
    lineage_records: list[dict[str, Any]] = []

    try:
        _validate_paths(args, effective_report)
        _write_json(effective_report, report)

        report["failure_phase"] = "source_identity"
        observed_source_sha256 = sha256_file(args.input)
        if observed_source_sha256 != args.expected_input_sha256:
            raise AssayError(
                "source_identity",
                f"input SHA256 mismatch: expected {args.expected_input_sha256}, "
                f"observed {observed_source_sha256}",
            )
        vertices, faces = _load_mesh(args.input)
        source_faces = int(len(faces))
        report["source"] = {
            "path": str(args.input),
            "sha256": observed_source_sha256,
            "vertices": int(len(vertices)),
            "faces": source_faces,
        }

        report["failure_phase"] = "lineage_identity"
        for label, raw_path, expected_sha256 in args.lineage_ref:
            path = Path(raw_path)
            observed_sha256 = sha256_file(path)
            if observed_sha256 != expected_sha256:
                raise AssayError(
                    "lineage_identity",
                    f"lineage {label!r} SHA256 mismatch: expected {expected_sha256}, "
                    f"observed {observed_sha256}",
                )
            lineage_records.append(
                {"label": label, "path": str(path), "sha256": observed_sha256}
            )
        report["continuation_provenance"]["refs"] = lineage_records

        report["failure_phase"] = "validate_target"
        if args.target_faces <= 0 or args.target_faces >= source_faces:
            raise AssayError(
                "validate_target",
                f"target faces must be positive and below source count {source_faces}",
            )
        if args.max_iterations <= 0:
            raise AssayError("validate_target", "max iterations must be positive")
        _write_json(effective_report, report)

        report["failure_phase"] = "runtime_identity"
        runtime = _load_runtime()
        report["route"]["runtime"] = runtime.identity
        _validate_runtime(args, runtime)
        _validate_runtime_custody(args, runtime)
        _write_json(effective_report, report)

        report["failure_phase"] = "prepare_outputs"
        _clean_output_dir(args.output_dir)
        report["primary_output_status"] = "partial"

        report["failure_phase"] = "intermediate_cleanup"
        report["active_stage"] = "intermediate_cleanup"
        _write_json(effective_report, report)
        _revalidate_custody(args, observed_source_sha256, lineage_records)
        stage_started = time.perf_counter()
        vertices, faces = runtime.cleanup(
            vertices, faces, **CLEANUP_CONFIG, do_fix_normals=False
        )
        stage = _stage_record(
            "intermediate_cleanup",
            args.output_dir / "01-intermediate-cleanup.glb",
            vertices,
            faces,
            time.perf_counter() - stage_started,
        )
        report["stages"]["intermediate_cleanup"] = stage
        report["last_trustworthy_evidence"] = stage
        _write_json(effective_report, report)

        report["failure_phase"] = "local_qem"
        report["active_stage"] = "local_qem"
        _write_json(effective_report, report)
        _revalidate_custody(args, observed_source_sha256, lineage_records)
        stage_started = time.perf_counter()
        vertices, faces, receipt = runtime.simplify(
            vertices,
            faces,
            args.target_faces,
            **QEM_CONFIG,
            max_iterations=args.max_iterations,
        )
        report["failure_phase"] = "local_qem_receipt"
        if receipt.get("route") != QEM_ROUTE:
            raise AssayError(
                "local_qem_receipt", f"unexpected QEM receipt route: {receipt.get('route')!r}"
            )
        if receipt.get("requested_target_faces") != args.target_faces:
            raise AssayError("local_qem_receipt", "QEM receipt target does not match request")
        if receipt.get("achieved_faces") != len(faces):
            raise AssayError("local_qem_receipt", "QEM receipt face count does not match output")
        if receipt.get("max_iterations") != args.max_iterations:
            raise AssayError("local_qem_receipt", "QEM receipt iteration cap does not match request")
        qem_stage = _stage_record(
            "local_qem",
            args.output_dir / "02-local-qem.glb",
            vertices,
            faces,
            time.perf_counter() - stage_started,
        )
        qem_stage["receipt"] = receipt
        report["stages"]["local_qem"] = qem_stage
        report["last_trustworthy_evidence"] = qem_stage
        _write_json(effective_report, report)

        report["failure_phase"] = "final_cleanup"
        report["active_stage"] = "final_cleanup"
        _write_json(effective_report, report)
        _revalidate_custody(args, observed_source_sha256, lineage_records)
        stage_started = time.perf_counter()
        vertices, faces = runtime.cleanup(
            vertices, faces, **CLEANUP_CONFIG, do_fix_normals=True
        )
        final_stage = _stage_record(
            "final_cleanup",
            args.output_dir / "03-final-cleanup.glb",
            vertices,
            faces,
            time.perf_counter() - stage_started,
        )
        report["stages"]["final_cleanup"] = final_stage
        report["last_trustworthy_evidence"] = final_stage
        _write_json(effective_report, report)

        report["failure_phase"] = "publish"
        report["active_stage"] = "publish"
        _revalidate_custody(args, observed_source_sha256, lineage_records)
        qem_satisfied = bool(receipt["target_satisfied"])
        final_satisfied = len(faces) <= args.target_faces
        report.update(
            {
                "status": "completed",
                "failure_phase": None,
                "active_stage": None,
                "primary_output_status": "validated",
                "target_contract": {
                    "requested_faces": args.target_faces,
                    "achieved_faces_after_qem": int(receipt["achieved_faces"]),
                    "achieved_faces_after_final_cleanup": int(len(faces)),
                    "qem_target_satisfied": qem_satisfied,
                    "final_face_budget_satisfied": final_satisfied,
                    "termination_reason": receipt["termination_reason"],
                },
                "claim_ceiling": (
                    "pipeline-executed-target-satisfied"
                    if qem_satisfied and final_satisfied
                    else "pipeline-executed-target-unsatisfied"
                ),
            }
        )
    except BaseException as exc:
        report["status"] = "failed"
        report["failure_phase"] = getattr(exc, "phase", report["failure_phase"])
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["primary_output_status"] = (
            "partial_preserved" if report["last_trustworthy_evidence"] else "not_started"
        )
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
