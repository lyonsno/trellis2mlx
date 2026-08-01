"""Capture every release geometry stage from the local Metal mtlmesh route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Callable

import numpy as np

from scripts.source_cuda_cumesh_postprocess_witness import (
    FORBIDDEN_INFERENCES,
    HEX_SHA256,
    STAGE_SPECS,
    WitnessError,
    _effective_report_path,
    _same_path,
    elapsed,
    mesh_summary,
    read_binary_ply,
    sha256_file,
    write_binary_ply,
)
from trellmlx.source_mtlmesh import postprocess_source_native
from trellmlx.source_route_identity import (
    SourceRouteIdentityError,
    probe_cumesh_route_identity,
    validate_source_route_identity,
)


EXPECTED_SOURCE_COMMIT = "212079e55772cff3d648a21372392c37e0643f3b"
HEX_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--target-faces", type=int, required=True)
    parser.add_argument("--expected-source-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    return parser


def run_witness(
    *,
    input_ply: Path,
    output_dir: Path,
    report_json: Path,
    expected_input_sha256: str,
    target_faces: int,
    expected_source_root: Path,
    expected_source_commit: str = EXPECTED_SOURCE_COMMIT,
    identity_probe: Callable[[], dict[str, Any]] | None = None,
    postprocessor: Callable[..., tuple[np.ndarray, np.ndarray, list[dict]]] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    input_ply = Path(input_ply)
    output_dir = Path(output_dir)
    requested_report_json = Path(report_json)
    expected_source_root = Path(expected_source_root).resolve(strict=False)
    stage_paths = {
        operation: output_dir / filename for operation, filename in STAGE_SPECS
    }
    effective_report_json, report_rerouted = _effective_report_path(
        requested_report_json,
        protected_paths=[input_ply, *stage_paths.values()],
    )
    report: dict[str, Any] = {
        "schema": "trellis2mlx.source_metal_mtlmesh_postprocess_witness.v1",
        "status": "failed",
        "failure_phase": None,
        "last_trustworthy_phase": "request_received",
        "primary_output_status": "not_started",
        "requested_report_json": str(requested_report_json),
        "effective_report_json": str(effective_report_json),
        "report_rerouted": report_rerouted,
        "requested_route": {
            "input_ply": str(input_ply),
            "expected_input_sha256": expected_input_sha256,
            "output_dir": str(output_dir),
            "target_faces": int(target_faces),
            "expected_source_root": str(expected_source_root),
            "expected_source_commit": expected_source_commit,
            "geometry_route": "metal-mtlmesh-standard-non-remesh",
        },
        "effective_route": None,
        "input_mesh": None,
        "stage_artifacts": [],
        "aggregate_trace": None,
        "forbidden_inferences": [
            *FORBIDDEN_INFERENCES,
            "not CUDA CuMesh evidence",
            "not proof of parity with release CuMesh",
        ],
    }
    phase = "request_validation"

    try:
        if _same_path(requested_report_json, input_ply):
            raise WitnessError("report path aliases protected input")
        for operation, stage_path in stage_paths.items():
            if _same_path(input_ply, stage_path):
                raise WitnessError(f"input aliases stage output for {operation}")
        if not HEX_SHA256.fullmatch(expected_input_sha256):
            raise WitnessError("--expected-input-sha256 must be 64 lowercase hex characters")
        if target_faces <= 0:
            raise WitnessError("--target-faces must be positive")
        if not HEX_GIT_COMMIT.fullmatch(expected_source_commit):
            raise WitnessError(
                "--expected-source-commit must be 40 lowercase hex characters"
            )

        phase = "input_validation"
        if not input_ply.is_file():
            raise WitnessError(f"input PLY does not exist: {input_ply}")
        actual_input_sha256 = sha256_file(input_ply)
        if actual_input_sha256 != expected_input_sha256:
            raise WitnessError(
                "input SHA256 mismatch: "
                f"expected {expected_input_sha256}, got {actual_input_sha256}"
            )
        input_vertices, input_faces = read_binary_ply(input_ply)
        report["input_mesh"] = {
            "sha256": actual_input_sha256,
            "size_bytes": input_ply.stat().st_size,
            **mesh_summary(input_vertices, input_faces),
        }
        report["last_trustworthy_phase"] = "input_validated"
        _write_report(effective_report_json, report)

        phase = "stale_output_cleanup"
        output_dir.mkdir(parents=True, exist_ok=True)
        for stage_path in stage_paths.values():
            if stage_path.exists():
                if not stage_path.is_file():
                    raise WitnessError(f"stale stage output is not a file: {stage_path}")
                stage_path.unlink()
        report["last_trustworthy_phase"] = "stale_outputs_removed"
        _write_report(effective_report_json, report)

        phase = "runtime_validation"
        probe = identity_probe or probe_cumesh_route_identity
        identity = _effective_identity(probe())
        try:
            identity = validate_source_route_identity(
                identity,
                expected_root=expected_source_root,
            )
        except SourceRouteIdentityError as exc:
            raise WitnessError(str(exc)) from exc
        if identity.get("git_commit") != expected_source_commit:
            raise WitnessError(
                "effective mtlmesh source commit mismatch: "
                f"expected {expected_source_commit}, got {identity.get('git_commit')}"
            )
        if identity.get("git_status_porcelain"):
            raise WitnessError(
                "effective mtlmesh source checkout is dirty: "
                f"{identity['git_status_porcelain']}"
            )
        if identity.get("has_MtlMesh") is not True:
            raise WitnessError("effective route does not expose cumesh.metal_backend.MtlMesh")
        report["effective_route"] = {
            **identity,
            "geometry_route": "metal-mtlmesh-standard-non-remesh",
            "target_faces": int(target_faces),
        }
        report["status"] = "running"
        report["last_trustworthy_phase"] = "runtime_validated"
        _write_report(effective_report_json, report)

        phase = "geometry_sequence"
        last_stage: tuple[np.ndarray, np.ndarray] | None = None

        def stage_callback(
            operation: str,
            operation_input_faces: int,
            operation_output_faces: int,
            details: dict[str, Any],
            vertices: np.ndarray,
            faces: np.ndarray,
        ) -> None:
            nonlocal last_stage
            index = len(report["stage_artifacts"])
            if index >= len(STAGE_SPECS):
                raise WitnessError(f"unexpected extra geometry operation: {operation}")
            expected_operation, filename = STAGE_SPECS[index]
            if operation != expected_operation:
                raise WitnessError(
                    f"unexpected geometry operation at stage {index + 1}: "
                    f"expected {expected_operation}, got {operation}"
                )
            vertices = np.ascontiguousarray(vertices, dtype=np.float32)
            faces = np.ascontiguousarray(faces, dtype=np.int32)
            if len(faces) != int(operation_output_faces):
                raise WitnessError(
                    f"{operation} callback output count {operation_output_faces} "
                    f"does not match readback {len(faces)}"
                )
            stage_path = output_dir / filename
            write_binary_ply(stage_path, vertices, faces)
            reopened_vertices, reopened_faces = read_binary_ply(stage_path)
            if not np.array_equal(reopened_vertices, vertices):
                raise WitnessError(f"{operation} reopened vertices differ from Metal readback")
            if not np.array_equal(reopened_faces, faces):
                raise WitnessError(f"{operation} reopened faces differ from Metal readback")
            report["stage_artifacts"].append(
                {
                    "index": index + 1,
                    "operation": operation,
                    "input_faces": int(operation_input_faces),
                    "output_faces": int(operation_output_faces),
                    "output_vertices": int(len(vertices)),
                    "details": details,
                    "path": str(stage_path),
                    "sha256": sha256_file(stage_path),
                    "size_bytes": stage_path.stat().st_size,
                    "status": "validated",
                }
            )
            last_stage = (vertices, faces)
            report["primary_output_status"] = "partial"
            report["last_trustworthy_phase"] = f"stage_validated:{operation}"
            _write_report(effective_report_json, report)

        processor = postprocessor or postprocess_source_native
        final_vertices, final_faces, aggregate_trace = processor(
            input_vertices,
            input_faces,
            int(target_faces),
            verbose=False,
            expected_source_root=expected_source_root,
            stage_callback=stage_callback,
        )
        report["aggregate_trace"] = aggregate_trace

        phase = "stage_validation"
        actual_operations = [item["operation"] for item in report["stage_artifacts"]]
        expected_operations = [operation for operation, _ in STAGE_SPECS]
        if actual_operations != expected_operations:
            raise WitnessError(
                "validated stage set does not match release sequence: "
                f"expected {expected_operations}, got {actual_operations}"
            )
        if last_stage is None:
            raise WitnessError("release sequence produced no stage artifacts")
        final_vertices = np.asarray(final_vertices, dtype=np.float32)
        final_faces = np.asarray(final_faces, dtype=np.int32)
        if not np.array_equal(final_vertices, last_stage[0]):
            raise WitnessError("final vertices differ from the orientation stage")
        if not np.array_equal(final_faces, last_stage[1]):
            raise WitnessError("final faces differ from the orientation stage")
        for item in report["stage_artifacts"]:
            path = Path(item["path"])
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                raise WitnessError(
                    f"validated stage artifact is missing or changed: {item['operation']}"
                )

        report.update(
            {
                "status": "done",
                "failure_phase": None,
                "last_trustworthy_phase": "all_stages_validated",
                "primary_output_status": "validated",
                "elapsed_seconds": elapsed(started),
            }
        )
        _write_report(effective_report_json, report)
        return report
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "failure_phase": phase,
                "primary_output_status": (
                    "partial" if report["stage_artifacts"] else "not_started"
                ),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": elapsed(started),
            }
        )
        _write_report(effective_report_json, report)
        raise


def _effective_identity(identity: dict[str, Any]) -> dict[str, Any]:
    effective = dict(identity)
    if "git_status_porcelain" not in effective:
        git_root = effective.get("git_root")
        if git_root:
            completed = subprocess.run(
                [
                    "git",
                    "-C",
                    str(git_root),
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise WitnessError(
                    "could not inspect effective mtlmesh source status: "
                    f"{completed.stderr.strip()}"
                )
            effective["git_status_porcelain"] = completed.stdout.strip()
    return effective


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = run_witness(
            input_ply=args.input_ply,
            output_dir=args.output_dir,
            report_json=args.report_json,
            expected_input_sha256=args.expected_input_sha256,
            target_faces=args.target_faces,
            expected_source_root=args.expected_source_root,
            expected_source_commit=args.expected_source_commit,
        )
    except Exception:
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
