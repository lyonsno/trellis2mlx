#!/usr/bin/env python3
"""Probe repeated fresh-mesh trajectories on an authenticated mtlmesh Metal route."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

from trellmlx.glb_aabb_crop import open_triangle_glb, sha256_file, write_geometry_glb


ROUTE = "source-native-mtlmesh-metal-step-v1"


class AssayError(RuntimeError):
    def __init__(self, phase: str, message: str):
        super().__init__(message)
        self.phase = phase


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--expected-input-sha256")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-backend-sha256", required=True)
    parser.add_argument("--expected-extension-sha256", required=True)
    parser.add_argument("--expected-metallib-sha256", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=1e-8)
    parser.add_argument("--lambda-edge-length", type=float, default=1e-2)
    parser.add_argument("--lambda-skinny", type=float, default=1e-3)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


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


def _same_or_within(path: Path, directory: Path) -> bool:
    resolved = path.resolve()
    root = directory.resolve()
    return resolved == root or resolved.is_relative_to(root)


def _legacy_report_temporary(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def _effective_report_path(args: argparse.Namespace) -> tuple[Path, bool]:
    source = args.input.resolve()
    output = args.output_dir.resolve()
    source_root = args.source_root.resolve()
    requested = args.report.resolve()
    requested_temporary = _legacy_report_temporary(args.report).resolve()
    unsafe = (
        requested == source
        or requested_temporary == source
        or _same_or_within(requested, output)
        or _same_or_within(requested, source_root)
        or source_root.is_relative_to(requested)
    )
    if not unsafe:
        return args.report, False
    if requested_temporary == source:
        candidate = args.report.with_name(args.report.name + ".assay-error.json")
    else:
        candidate = args.input.with_name(args.input.name + ".assay-error.json")
    while (
        candidate.resolve() == source
        or _legacy_report_temporary(candidate).resolve() == source
        or _same_or_within(candidate, output)
        or _same_or_within(candidate, source_root)
        or source_root.is_relative_to(candidate.resolve())
    ):
        candidate = candidate.with_name(candidate.name + ".assay-error.json")
    return candidate, True


def _git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no git output"
        raise AssayError("authenticate_route", detail)
    return completed.stdout.strip()


def _probe_source_route(args: argparse.Namespace) -> dict[str, Any]:
    root = args.source_root.resolve()
    if not root.is_dir():
        raise AssayError("authenticate_route", f"source root does not exist: {root}")
    commit = _git_output(root, "rev-parse", "HEAD")
    if commit != args.expected_source_commit:
        raise AssayError(
            "authenticate_route",
            f"source commit mismatch: expected {args.expected_source_commit}, found {commit}",
        )
    if _git_output(root, "status", "--porcelain"):
        raise AssayError("authenticate_route", "source worktree has tracked or untracked changes")

    backend = root / "cumesh" / "metal_backend.py"
    metallib = root / "cumesh" / "cumesh.metallib"
    extensions = sorted((root / "cumesh").glob("_C*.so"))
    if not backend.is_file() or not metallib.is_file() or len(extensions) != 1:
        raise AssayError(
            "authenticate_route",
            "source route must contain metal_backend.py, one _C extension, and cumesh.metallib",
        )
    extension = extensions[0]
    identities = {
        "backend": (backend, args.expected_backend_sha256),
        "extension": (extension, args.expected_extension_sha256),
        "metallib": (metallib, args.expected_metallib_sha256),
    }
    actual_hashes: dict[str, str] = {}
    for name, (path, expected) in identities.items():
        actual = _sha256(path)
        actual_hashes[name] = actual
        if actual != expected:
            raise AssayError(
                "authenticate_route",
                f"{name} SHA256 mismatch: expected {expected}, found {actual}",
            )

    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = None
    return {
        "id": ROUTE,
        "source_root": str(root),
        "source_commit": commit,
        "source_remote": _git_output(root, "remote", "get-url", "origin"),
        "backend_path": str(backend),
        "backend_sha256": actual_hashes["backend"],
        "extension_path": str(extension),
        "extension_sha256": actual_hashes["extension"],
        "metallib_path": str(metallib),
        "metallib_sha256": actual_hashes["metallib"],
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch_version,
        "mps_available": None,
    }


def _load_mesh_class(route: dict[str, Any]):
    root = Path(route["source_root"]).resolve()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module = importlib.import_module("cumesh.metal_backend")
    module_path = Path(module.__file__).resolve()
    if not module_path.is_relative_to(root):
        raise AssayError(
            "authenticate_route", f"cumesh resolved outside source root: {module_path}"
        )
    extension = importlib.import_module("cumesh._C")
    extension_path = Path(extension.__file__).resolve()
    if _sha256(extension_path) != route["extension_sha256"]:
        raise AssayError("authenticate_route", "imported extension identity changed")
    torch = importlib.import_module("torch")
    route["mps_available"] = bool(torch.backends.mps.is_available())
    if not route["mps_available"]:
        raise AssayError("authenticate_route", "Torch MPS is unavailable")
    return module.MtlMesh


def _load_source(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        with open_triangle_glb(path) as view:
            return (
                np.asarray(view.vertices, dtype=np.float32).copy(),
                np.asarray(view.faces, dtype=np.int32).copy(),
            )
    except Exception as exc:
        raise AssayError("load_source", str(exc)) from exc


def _initialize_mesh(mesh: Any, vertices: np.ndarray, faces: np.ndarray) -> None:
    if mesh.__class__.__module__.startswith("cumesh."):
        torch = importlib.import_module("torch")
        mesh.init(torch.from_numpy(vertices), torch.from_numpy(faces))
    else:
        mesh.init(vertices, faces)


def _to_numpy(value: Any, dtype: np.dtype) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _clean_output_dir(path: Path) -> None:
    if path.exists():
        if not path.is_dir() or path.is_symlink():
            raise AssayError("prepare_outputs", f"output surface is not a directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _validate(args: argparse.Namespace, report_path: Path) -> None:
    source = args.input.resolve()
    output = args.output_dir.resolve()
    report = report_path.resolve()
    source_root = args.source_root.resolve()
    if source == output or source.is_relative_to(output):
        raise AssayError("validate_paths", "output directory must not contain the input")
    if report == source or report == output or report.is_relative_to(output):
        raise AssayError("validate_paths", "report must not replace input or live in output")
    if (
        output == source_root
        or output.is_relative_to(source_root)
        or source_root.is_relative_to(output)
    ):
        raise AssayError("validate_paths", "output directory must not overlap source root")
    if (
        report == source_root
        or report.is_relative_to(source_root)
        or source_root.is_relative_to(report)
    ):
        raise AssayError("validate_paths", "effective report must not overlap source root")
    if args.report.resolve() == source or _legacy_report_temporary(args.report).resolve() == source:
        raise AssayError("validate_paths", "requested report path collides with input custody")
    if _same_or_within(args.report, source_root) or source_root.is_relative_to(
        args.report.resolve()
    ):
        raise AssayError("validate_paths", "requested report must not overlap source root")
    if args.repeats < 2:
        raise AssayError("validate_config", "repeats must be at least two")
    if args.max_steps < 1:
        raise AssayError("validate_config", "max-steps must be positive")
    values = (args.threshold, args.lambda_edge_length, args.lambda_skinny)
    if not all(np.isfinite(value) and value >= 0 for value in values):
        raise AssayError("validate_config", "simplifier scalars must be finite and nonnegative")


def _request(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "input": str(args.input),
        "expected_input_sha256": args.expected_input_sha256,
        "output_dir": str(args.output_dir),
        "report": str(args.report),
        "source_root": str(args.source_root),
        "expected_source_commit": args.expected_source_commit,
        "expected_backend_sha256": args.expected_backend_sha256,
        "expected_extension_sha256": args.expected_extension_sha256,
        "expected_metallib_sha256": args.expected_metallib_sha256,
        "repeats": args.repeats,
        "max_steps": args.max_steps,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    effective_report, report_rerouted = _effective_report_path(args)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "status": "failed",
        "failure_phase": "validate_paths",
        "primary_output_status": "not_started",
        "request": _request(args),
        "report": {
            "requested_path": str(args.report),
            "effective_path": str(effective_report),
            "rerouted": report_rerouted,
        },
        "route": None,
        "effective_config": {
            "fresh_mesh_per_repeat": True,
            "target_face_count": None,
            "max_steps": args.max_steps,
            "initial_threshold": args.threshold,
            "threshold_growth": 10.0,
            "threshold_growth_predicate": "removed_faces / input_faces < 0.01",
            "lambda_edge_length": args.lambda_edge_length,
            "lambda_skinny": args.lambda_skinny,
        },
        "source": None,
        "runs": None,
        "step_stability": None,
        "repeat_stability": None,
        "last_trustworthy_evidence": None,
        "partial_output_cleanup": None,
        "elapsed_seconds": None,
    }
    try:
        _validate(args, effective_report)
        _write_json(effective_report, report)

        report["failure_phase"] = "source_identity"
        source_sha256 = sha256_file(args.input)
        report["source"] = {
            "path": str(args.input),
            "sha256": source_sha256,
            "expected_sha256": args.expected_input_sha256,
            "identity_match": (
                args.expected_input_sha256 is None
                or source_sha256 == args.expected_input_sha256
            ),
        }
        if args.expected_input_sha256 is not None and source_sha256 != args.expected_input_sha256:
            raise AssayError(
                "source_identity",
                f"input SHA256 mismatch: expected {args.expected_input_sha256}, found {source_sha256}",
            )

        report["failure_phase"] = "authenticate_route"
        route = _probe_source_route(args)
        mesh_class = _load_mesh_class(route)
        report["route"] = route
        _write_json(effective_report, report)

        report["failure_phase"] = "load_source"
        vertices, faces = _load_source(args.input)
        report["source"].update({
            "vertices": int(len(vertices)),
            "faces": int(len(faces)),
            "vertex_sha256": _array_sha256(vertices),
            "face_sha256": _array_sha256(faces),
        })

        report["failure_phase"] = "prepare_outputs"
        _clean_output_dir(args.output_dir)
        report["primary_output_status"] = "partial"

        report["failure_phase"] = "run_steps"
        runs: list[dict[str, Any]] = []
        for repeat in range(1, args.repeats + 1):
            if sha256_file(args.input) != source_sha256:
                raise AssayError("source_identity", "input GLB changed during assay")
            mesh = mesh_class()
            _initialize_mesh(mesh, vertices.copy(), faces.copy())
            threshold = float(args.threshold)
            steps: list[dict[str, Any]] = []
            run_started = time.perf_counter()
            for step in range(1, args.max_steps + 1):
                input_faces = int(mesh.num_faces)
                step_started = time.perf_counter()
                output_vertices, output_faces = mesh.simplify_step(
                    float(args.lambda_edge_length),
                    float(args.lambda_skinny),
                    threshold,
                    False,
                )
                output_vertices = int(output_vertices)
                output_faces = int(output_faces)
                mesh_vertices_raw, mesh_faces_raw = mesh.read()
                mesh_vertices = _to_numpy(mesh_vertices_raw, np.float32)
                mesh_faces = _to_numpy(mesh_faces_raw, np.uint32)
                if len(mesh_vertices) != output_vertices or len(mesh_faces) != output_faces:
                    raise AssayError("run_steps", "reported and read mesh counts disagree")
                stem = f"repeat-{repeat}-step-{step}-faces-{output_faces}"
                mesh_path = args.output_dir / f"{stem}.glb"
                write_geometry_glb(mesh_path, mesh_vertices, mesh_faces)
                removed = input_faces - output_faces
                record = {
                    "step": step,
                    "threshold": threshold,
                    "input_faces": input_faces,
                    "output_vertices": output_vertices,
                    "output_faces": output_faces,
                    "removed_faces": removed,
                    "removed_fraction": removed / max(input_faces, 1),
                    "vertex_sha256": _array_sha256(mesh_vertices),
                    "face_sha256": _array_sha256(mesh_faces),
                    "mesh_path": str(mesh_path),
                    "mesh_sha256": sha256_file(mesh_path),
                    "elapsed_seconds": time.perf_counter() - step_started,
                }
                steps.append(record)
                report["last_trustworthy_evidence"] = {
                    "repeat": repeat,
                    **record,
                }
                if record["removed_fraction"] < 1e-2:
                    threshold *= 10.0
            runs.append(
                {
                    "repeat": repeat,
                    "steps": steps,
                    "elapsed_seconds": time.perf_counter() - run_started,
                }
            )

        report["failure_phase"] = "analyze_stability"
        step_stability: list[dict[str, Any]] = []
        all_exact = True
        identity_fields = (
            "output_vertices",
            "output_faces",
            "vertex_sha256",
            "face_sha256",
            "mesh_sha256",
        )
        for index in range(args.max_steps):
            records = [run["steps"][index] for run in runs]
            exact = all(
                all(record[field] == records[0][field] for field in identity_fields)
                for record in records[1:]
            )
            all_exact = all_exact and exact
            step_stability.append(
                {
                    "step": index + 1,
                    "exact": exact,
                    "distinct_output_face_counts": sorted(
                        {record["output_faces"] for record in records}
                    ),
                    "distinct_face_sha256": sorted(
                        {record["face_sha256"] for record in records}
                    ),
                    "distinct_vertex_sha256": sorted(
                        {record["vertex_sha256"] for record in records}
                    ),
                }
            )
        if sha256_file(args.input) != source_sha256:
            raise AssayError("source_identity", "input GLB changed before report publication")
        final_route = _probe_source_route(args)
        for field in (
            "source_root",
            "source_commit",
            "backend_sha256",
            "extension_sha256",
            "metallib_sha256",
        ):
            if final_route[field] != route[field]:
                raise AssayError(
                    "source_identity", f"source route identity changed at field {field}"
                )
        report.update(
            {
                "status": "completed",
                "failure_phase": None,
                "primary_output_status": "validated",
                "runs": runs,
                "step_stability": step_stability,
                "repeat_stability": {"all_steps_exact": all_exact},
            }
        )
    except BaseException as exc:
        report["status"] = "failed"
        report["failure_phase"] = getattr(exc, "phase", report["failure_phase"])
        report["error"] = f"{type(exc).__name__}: {exc}"
        existed = args.output_dir.is_dir() and not args.output_dir.is_symlink()
        artifact_count = (
            sum(1 for path in args.output_dir.rglob("*") if path.is_file()) if existed else 0
        )
        removed = False
        cleanup_error = None
        if existed:
            try:
                shutil.rmtree(args.output_dir)
                removed = True
            except Exception as cleanup_exc:  # pragma: no cover - filesystem-specific
                cleanup_error = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
        report["partial_output_cleanup"] = {
            "output_dir_existed": existed,
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
