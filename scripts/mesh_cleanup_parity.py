#!/usr/bin/env python
"""Write a reference-cleanup parity report for a fixture or raw mesh NPZ."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fixture", choices=["two-triangle-sheet"])
    source.add_argument("--raw-mesh", type=Path, help="NPZ containing vertices and faces arrays")
    parser.add_argument("--target-faces", type=int, default=200_000)
    parser.add_argument(
        "--local-simplifier",
        choices=["fast-simplification", "qem-probe"],
        default="fast-simplification",
        help="Local simplifier used inside the reference-cleanup harness route",
    )
    parser.add_argument("--reference-python", type=Path, help="Python executable with cumesh available")
    parser.add_argument("--require-reference", action="store_true", help="Fail if the reference backend cannot run")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--overwrite-report", action="store_true")
    return parser.parse_args(argv)


class ReferenceBackendError(RuntimeError):
    def __init__(self, backend: dict[str, Any]):
        super().__init__(backend.get("error", "reference backend failed"))
        self.backend = backend


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n")


def _ensure_report_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"report already exists: {path}; pass --overwrite-report to replace it")


def _fixture_two_triangle_sheet() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    return vertices, faces


def _load_mesh(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, Any], str]:
    if args.fixture:
        vertices, faces = _fixture_two_triangle_sheet()
        return vertices, faces, {"route": "fixture", "name": args.fixture}, f"fixture:{args.fixture}"

    if not args.raw_mesh.exists():
        raise FileNotFoundError(f"raw mesh does not exist: {args.raw_mesh}")
    data = np.load(args.raw_mesh)
    if "vertices" not in data or "faces" not in data:
        raise ValueError(f"raw mesh NPZ must contain vertices and faces arrays: {args.raw_mesh}")
    return (
        np.asarray(data["vertices"]),
        np.asarray(data["faces"]),
        {"route": "raw-mesh", "path": str(args.raw_mesh)},
        "raw-mesh",
    )


def _probe_reference_backend() -> dict[str, Any]:
    try:
        import cumesh  # type: ignore  # noqa: F401
    except Exception as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    return {"status": "available", "module": "cumesh"}


def _external_reference_code() -> str:
    return r'''
import json
import sys
import time

import numpy as np
import torch
import cumesh


def record(trace, operation, input_faces, output_faces, requested_target_faces=None, do_fix_normals=None):
    entry = {
        "operation": operation,
        "input_faces": int(input_faces),
        "output_faces": int(output_faces),
    }
    if requested_target_faces is not None:
        entry["requested_target_faces"] = int(requested_target_faces)
    if do_fix_normals is not None:
        entry["do_fix_normals"] = bool(do_fix_normals)
    trace.append(entry)


mesh_path, target_faces_text, output_npz, trace_json = sys.argv[1:5]
target_faces = int(target_faces_text)
data = np.load(mesh_path)
vertices = np.asarray(data["vertices"], dtype=np.float32)
faces = np.asarray(data["faces"], dtype=np.int32)

t0 = time.perf_counter()
mesh = cumesh.CuMesh()
mesh.init(torch.from_numpy(vertices), torch.from_numpy(faces))
trace = []

coarse_target = target_faces * 3
if mesh.num_faces > coarse_target:
    input_faces = mesh.num_faces
    mesh.simplify(coarse_target, verbose=False)
    record(trace, "simplify_coarse", input_faces, mesh.num_faces, requested_target_faces=coarse_target)

input_faces = mesh.num_faces
mesh.remove_duplicate_faces()
mesh.repair_non_manifold_edges()
mesh.remove_small_connected_components(1e-5)
mesh.fill_holes(max_hole_perimeter=3e-2)
record(trace, "cleanup_initial", input_faces, mesh.num_faces, do_fix_normals=False)

if mesh.num_faces > target_faces:
    input_faces = mesh.num_faces
    mesh.simplify(target_faces, verbose=False)
    record(trace, "simplify_final", input_faces, mesh.num_faces, requested_target_faces=target_faces)

input_faces = mesh.num_faces
mesh.remove_duplicate_faces()
mesh.repair_non_manifold_edges()
mesh.remove_small_connected_components(1e-5)
mesh.fill_holes(max_hole_perimeter=3e-2)
record(trace, "cleanup_final", input_faces, mesh.num_faces, do_fix_normals=False)

input_faces = mesh.num_faces
mesh.unify_face_orientations()
record(trace, "unify_face_orientations", input_faces, mesh.num_faces)

out_vertices, out_faces = mesh.read()
np.savez_compressed(
    output_npz,
    vertices=out_vertices.cpu().numpy().astype(np.float32, copy=False),
    faces=out_faces.cpu().numpy().astype(np.int64, copy=False),
)
Path = __import__("pathlib").Path
Path(trace_json).write_text(json.dumps({
    "operation_trace": trace,
    "elapsed_seconds": time.perf_counter() - t0,
}, indent=2, sort_keys=True) + "\n")
'''


def _run_external_reference_cleanup(
    *,
    reference_python: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
) -> tuple[np.ndarray | None, np.ndarray | None, list[dict[str, Any]], dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="trellis2mlx-cumesh-reference-") as tmp:
        tmp_path = Path(tmp)
        input_npz = tmp_path / "input_mesh.npz"
        output_npz = tmp_path / "output_mesh.npz"
        trace_json = tmp_path / "trace.json"
        np.savez_compressed(
            input_npz,
            vertices=np.asarray(vertices, dtype=np.float32),
            faces=np.asarray(faces, dtype=np.int32),
        )
        cmd = [
            str(reference_python),
            "-c",
            _external_reference_code(),
            str(input_npz),
            str(target_faces),
            str(output_npz),
            str(trace_json),
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            return None, None, [], {
                "status": "failed",
                "python": str(reference_python),
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "error": completed.stderr or completed.stdout or "reference backend failed",
            }
        output = np.load(output_npz)
        trace = json.loads(trace_json.read_text())
        return (
            np.asarray(output["vertices"]),
            np.asarray(output["faces"]),
            trace.get("operation_trace", []),
            {
                "status": "available",
                "python": str(reference_python),
                "route": "external-cumesh",
                "elapsed_seconds": trace.get("elapsed_seconds"),
            },
        )


def _fixture_simplify(vertices, faces, target_reduction=None, target_count=None):
    if target_count is None:
        return vertices, faces
    return vertices, faces[:target_count]


def _fixture_qem_probe_simplify(vertices, faces, target_reduction=None, target_count=None):
    return _fixture_simplify(vertices, faces, target_reduction=target_reduction, target_count=target_count)


def _qem_probe_simplify(vertices, faces, target_reduction=None, target_count=None):
    if target_count is None:
        raise ValueError("qem-probe simplifier requires target_count")
    from trellmlx.simplify_qem_metal import simplify_qem

    return simplify_qem(vertices, faces, int(target_count), verbose=False)


def _fixture_cleanup(vertices, faces, **kwargs):
    return vertices, faces


def _run_local_reference_cleanup(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    target_faces: int,
    fixture_mode: bool,
    local_simplifier: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    from generate import _cleanup_and_simplify_mesh

    operation_trace: list[dict[str, Any]] = []
    kwargs: dict[str, Any] = {}
    if fixture_mode:
        from trellmlx.mesh_cleanup import orient_faces_by_adjacency
        simplify = _fixture_qem_probe_simplify if local_simplifier == "qem-probe" else _fixture_simplify

        kwargs.update(
            cleanup_mesh=_fixture_cleanup,
            simplify=simplify,
            orient_faces_by_adjacency=orient_faces_by_adjacency,
        )
    elif local_simplifier == "qem-probe":
        kwargs["simplify"] = _qem_probe_simplify

    output_vertices, output_faces = _cleanup_and_simplify_mesh(
        vertices,
        faces,
        target_faces=target_faces,
        no_cleanup=False,
        reference_cleanup=True,
        operation_trace=operation_trace,
        log=lambda *args, **kwargs: None,
        **kwargs,
    )
    return output_vertices, output_faces, operation_trace


def _failure_report(
    *,
    requested_route: str,
    report_path: Path,
    phase: str,
    error: str,
) -> dict[str, Any]:
    return {
        "schema": "trellis2mlx.mesh_cleanup_parity_report.v1",
        "status": "failed",
        "requested_route": requested_route,
        "effective_route": "none",
        "failure_phase": phase,
        "error": error,
        "report_path": str(report_path),
        "last_trustworthy_evidence": {"report_written": True},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    from trellmlx.mesh_cleanup_parity import build_mesh_cleanup_parity_report

    vertices, faces, asset, requested_route = _load_mesh(args)
    t0 = time.perf_counter()
    output_vertices, output_faces, operation_trace = _run_local_reference_cleanup(
        vertices,
        faces,
        target_faces=args.target_faces,
        fixture_mode=asset["route"] == "fixture",
        local_simplifier=args.local_simplifier,
    )
    reference_vertices = None
    reference_faces = None
    reference_operation_trace: list[dict[str, Any]] | None = None
    reference_backend = _probe_reference_backend()
    if args.reference_python:
        (
            reference_vertices,
            reference_faces,
            reference_operation_trace,
            reference_backend,
        ) = _run_external_reference_cleanup(
            reference_python=args.reference_python,
            vertices=vertices,
            faces=faces,
            target_faces=args.target_faces,
        )
        if args.require_reference and reference_backend.get("status") != "available":
            raise ReferenceBackendError(reference_backend)

    report = build_mesh_cleanup_parity_report(
        requested_route=requested_route,
        effective_route=f"local-reference-cleanup:{args.local_simplifier}",
        input_vertices=vertices,
        input_faces=faces,
        output_vertices=output_vertices,
        output_faces=output_faces,
        operation_trace=operation_trace,
        reference_backend=reference_backend,
        reference_vertices=reference_vertices,
        reference_faces=reference_faces,
        reference_operation_trace=reference_operation_trace,
    )
    report.update({
        "status": "ok",
        "asset": asset,
        "settings": {
            "target_faces": args.target_faces,
            "local_simplifier": args.local_simplifier,
        },
        "elapsed_seconds": time.perf_counter() - t0,
    })
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _ensure_report_writable(args.report, args.overwrite_report)
    requested_route = "raw-mesh" if args.raw_mesh else f"fixture:{args.fixture}"
    try:
        report = run(args)
    except ReferenceBackendError as exc:
        report = _failure_report(
            requested_route=requested_route,
            report_path=args.report,
            phase="reference_backend",
            error=exc.backend.get("error", str(exc)),
        )
        report["effective_route"] = "local-reference-cleanup"
        report["reference_backend"] = exc.backend
        _write_json(args.report, report)
        return 1
    except Exception as exc:
        report = _failure_report(
            requested_route=requested_route,
            report_path=args.report,
            phase="load_mesh" if isinstance(exc, (FileNotFoundError, ValueError)) else "run_cleanup",
            error=f"{type(exc).__name__}: {exc}",
        )
        _write_json(args.report, report)
        return 1
    _write_json(args.report, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
