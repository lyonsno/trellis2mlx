"""Run fresh progressive trajectories through unmodified source CUDA CuMesh."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Callable
import zipfile

import numpy as np

try:
    from scripts.source_cuda_cumesh_postprocess_witness import (
        CUMESH_COMMIT,
        HEX_SHA256,
        TRELLIS_COMMIT,
        WitnessError,
        _effective_report_path,
        _same_path,
        _validate_effective_route,
        _write_report,
        prepare_release_runtime,
        read_binary_ply,
        sha256_file,
        write_binary_ply,
    )
except ModuleNotFoundError:
    from source_cuda_cumesh_postprocess_witness import (
        CUMESH_COMMIT,
        HEX_SHA256,
        TRELLIS_COMMIT,
        WitnessError,
        _effective_report_path,
        _same_path,
        _validate_effective_route,
        _write_report,
        prepare_release_runtime,
        read_binary_ply,
        sha256_file,
        write_binary_ply,
    )


GEOMETRY_ROUTE = "release-cumesh-native-atomic-progressive-replicates"
ADJACENCY_ORDER = "native-atomic-fill"
LAMBDA_EDGE_LENGTH = 1e-2
LAMBDA_SKINNY = 1e-3
INITIAL_THRESHOLD = 1e-8
THRESHOLD_GROWTH = 10.0
THRESHOLD_GROWTH_CUTOFF = 1e-2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--output-archive", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=8)
    return parser


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _to_numpy(value: Any, dtype: np.dtype[Any]) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(value, dtype=dtype)


def _validate_mesh(vertices: np.ndarray, faces: np.ndarray) -> None:
    if vertices.ndim != 2 or vertices.shape[1:] != (3,):
        raise WitnessError(f"vertices must have shape [V, 3], got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1:] != (3,):
        raise WitnessError(f"faces must have shape [F, 3], got {faces.shape}")
    if not np.all(np.isfinite(vertices)):
        raise WitnessError("vertices contain nonfinite values")
    if len(faces) and (np.any(faces < 0) or np.any(faces >= len(vertices))):
        raise WitnessError("faces contain out-of-range vertex indices")


def _validate_native_route(route: dict[str, Any]) -> None:
    _validate_effective_route(route)
    required_authority = {
        "trellis_source_clean": True,
        "cumesh_source_clean_before_build": True,
        "cumesh_instrumentation": None,
    }
    for field, expected in required_authority.items():
        if field not in route or route[field] is not expected:
            if field == "cumesh_instrumentation" and field in route:
                raise WitnessError(
                    "assay requires unmodified official CuMesh without instrumentation"
                )
            raise WitnessError(
                f"effective route {field} must be explicitly {expected!r}"
            )
    if route.get("cuda_available") is not True:
        raise WitnessError("effective route does not report CUDA available")


def _step_mesh(
    runtime: Any,
    mesh: Any,
    threshold: float,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    output_vertices, output_faces = mesh.cu_mesh.simplify_step(
        LAMBDA_EDGE_LENGTH,
        LAMBDA_SKINNY,
        threshold,
        False,
    )
    raw_vertices, raw_faces = runtime.read_mesh(mesh)
    vertices = _to_numpy(raw_vertices, np.float32)
    faces = _to_numpy(raw_faces, np.int32)
    _validate_mesh(vertices, faces)
    if int(output_vertices) != len(vertices) or int(output_faces) != len(faces):
        raise WitnessError("reported and read mesh counts disagree")
    return int(output_vertices), int(output_faces), vertices, faces


def _paths_overlap(left: Path, right: Path) -> bool:
    left = Path(left).resolve(strict=False)
    right = Path(right).resolve(strict=False)
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _atomic_temporary(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def _effective_progressive_report_path(
    requested: Path,
    *,
    protected_paths: list[Path],
    destructive_roots: list[Path],
) -> tuple[Path, bool]:
    def safe(candidate: Path) -> bool:
        temporary = _atomic_temporary(candidate)
        return not any(
            _same_path(path, protected)
            for path in (candidate, temporary)
            for protected in protected_paths
        ) and not any(
            _paths_overlap(path, root)
            for path in (candidate, temporary)
            for root in destructive_roots
        )

    if safe(requested):
        return requested, False
    fallback_parent = Path(destructive_roots[0]).parent.parent
    base = fallback_parent / (requested.name + ".failure.json")
    candidate = base
    index = 1
    while candidate.exists() or not safe(candidate):
        candidate = base.with_name(f"{base.stem}.{index}{base.suffix}")
        index += 1
    return candidate, True


def _archive_members(
    archive: Path,
    member_dir: Path,
    member_records: list[dict[str, Any]],
) -> dict[str, Any]:
    temporary = archive.with_name(archive.name + ".tmp")
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as bundle:
        for record in member_records:
            bundle.write(member_dir / record["member"], arcname=record["member"])
    temporary.replace(archive)

    expected = {record["member"]: record for record in member_records}
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)):
            raise WitnessError("reopened archive contains duplicate member names")
        if set(names) != set(expected):
            raise WitnessError("reopened archive member set mismatch")
        for name, record in expected.items():
            payload = bundle.read(name)
            if len(payload) != record["size_bytes"]:
                raise WitnessError(f"reopened archive member size mismatch: {name}")
            if hashlib.sha256(payload).hexdigest() != record["sha256"]:
                raise WitnessError(f"reopened archive member SHA256 mismatch: {name}")
    return {
        "path": str(archive),
        "sha256": sha256_file(archive),
        "size_bytes": archive.stat().st_size,
        "members": [record["member"] for record in member_records],
        "member_count": len(member_records),
    }


def validate_output_pair(
    report_path: Path,
    archive_path: Path,
    *,
    expected_input_sha256: str,
) -> dict[str, Any]:
    """Admit a downloaded producer report/archive pair as one evidence object."""
    report_path = Path(report_path)
    archive_path = Path(archive_path)
    try:
        report = json.loads(report_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WitnessError(f"invalid producer report: {exc}") from exc
    if report.get("schema") != "trellis2mlx.source_cuda_cumesh_progressive_replicates.v1":
        raise WitnessError("producer report schema is invalid")
    if report.get("status") != "done" or report.get("failure_phase") is not None:
        raise WitnessError("producer report does not record successful terminality")
    if report.get("primary_output_status") != "validated":
        raise WitnessError("producer report primary output is not validated")
    if not HEX_SHA256.fullmatch(expected_input_sha256):
        raise WitnessError("expected input SHA256 must be 64 lowercase hex characters")

    requested_route = report.get("requested_route")
    required_request = {
        "expected_input_sha256": expected_input_sha256,
        "trellis_commit": TRELLIS_COMMIT,
        "cumesh_commit": CUMESH_COMMIT,
        "geometry_route": GEOMETRY_ROUTE,
        "repeats": 5,
        "max_steps": 8,
        "lambda_edge_length": LAMBDA_EDGE_LENGTH,
        "lambda_skinny": LAMBDA_SKINNY,
        "initial_threshold": INITIAL_THRESHOLD,
    }
    if not isinstance(requested_route, dict):
        raise WitnessError("producer report lacks requested route identity")
    for field, expected in required_request.items():
        if requested_route.get(field) != expected:
            raise WitnessError(
                f"requested route {field} mismatch: expected {expected!r}, "
                f"got {requested_route.get(field)!r}"
            )
    if "target_faces" in requested_route:
        raise WitnessError("requested route falsely declares a target face count")

    effective_route = report.get("effective_route")
    if not isinstance(effective_route, dict):
        raise WitnessError("producer report lacks effective route identity")
    _validate_native_route(effective_route)
    required_effective_route = {
        "geometry_route": GEOMETRY_ROUTE,
        "input_sha256": expected_input_sha256,
        "adjacency_order": ADJACENCY_ORDER,
    }
    for field, expected in required_effective_route.items():
        if effective_route.get(field) != expected:
            raise WitnessError(
                f"effective route {field} mismatch: expected {expected!r}, "
                f"got {effective_route.get(field)!r}"
            )
    if "target_faces" in effective_route:
        raise WitnessError("effective route falsely declares a target face count")

    expected_config = {
        "fresh_mesh_per_repeat": True,
        "adjacency_order": ADJACENCY_ORDER,
        "canonical_adjacency": False,
        "instrumentation": None,
        "target_face_count": None,
        "threshold_growth": THRESHOLD_GROWTH,
        "threshold_growth_predicate": "removed_faces / input_faces < 0.01",
    }
    if report.get("effective_config") != expected_config:
        raise WitnessError("producer report effective configuration is not exact")
    input_mesh = report.get("input_mesh")
    if not isinstance(input_mesh, dict) or input_mesh.get("sha256") != expected_input_sha256:
        raise WitnessError("producer report input mesh identity is invalid")
    for field in ("vertices", "faces"):
        if not isinstance(input_mesh.get(field), int) or input_mesh[field] < 0:
            raise WitnessError(f"producer report input mesh {field} is invalid")
    for field in ("vertex_sha256", "face_sha256"):
        if not isinstance(input_mesh.get(field), str) or not HEX_SHA256.fullmatch(input_mesh[field]):
            raise WitnessError(f"producer report input mesh {field} is invalid")
    archive_record = report.get("output_archive")
    if not isinstance(archive_record, dict):
        raise WitnessError("producer report lacks output archive identity")
    if not archive_path.is_file():
        raise WitnessError("downloaded output archive is missing")
    actual_sha256 = sha256_file(archive_path)
    if actual_sha256 != archive_record.get("sha256"):
        raise WitnessError("archive SHA256 differs from producer report")
    if archive_path.stat().st_size != archive_record.get("size_bytes"):
        raise WitnessError("archive size differs from producer report")

    runs = report.get("runs")
    if not isinstance(runs, list) or len(runs) != 5:
        raise WitnessError("producer report must contain exactly five replicate trajectories")
    expected_members: dict[str, dict[str, Any]] = {}
    for expected_repeat, run in enumerate(runs, start=1):
        if (
            not isinstance(run, dict)
            or run.get("repeat") != expected_repeat
            or not isinstance(run.get("steps"), list)
            or len(run["steps"]) != 8
        ):
            raise WitnessError("producer report contains malformed run records")
        previous_output_faces = input_mesh["faces"]
        expected_threshold = INITIAL_THRESHOLD
        for expected_step, step in enumerate(run["steps"], start=1):
            if not isinstance(step, dict) or step.get("step") != expected_step:
                raise WitnessError("producer report step ordering is invalid")
            if step.get("threshold") != expected_threshold:
                raise WitnessError("producer report threshold schedule is invalid")
            if step.get("input_faces") != previous_output_faces:
                raise WitnessError("producer report face-count trajectory is discontinuous")
            output_faces = step.get("output_faces")
            output_vertices = step.get("output_vertices")
            if (
                not isinstance(output_faces, int)
                or not isinstance(output_vertices, int)
                or output_faces < 0
                or output_faces > previous_output_faces
                or output_vertices < 0
            ):
                raise WitnessError("producer report output mesh counts are invalid")
            removed_faces = previous_output_faces - output_faces
            removed_fraction = removed_faces / max(previous_output_faces, 1)
            if step.get("removed_faces") != removed_faces or not np.isclose(
                step.get("removed_fraction"), removed_fraction, rtol=0, atol=1e-15
            ):
                raise WitnessError("producer report removal arithmetic is invalid")
            for field in ("vertex_sha256", "face_sha256"):
                if not isinstance(step.get(field), str) or not HEX_SHA256.fullmatch(step[field]):
                    raise WitnessError(f"producer report step {field} is invalid")
            member = step.get("member") if isinstance(step, dict) else None
            if not isinstance(member, dict) or not isinstance(member.get("member"), str):
                raise WitnessError("producer report contains malformed member records")
            name = member["member"]
            expected_name = (
                f"repeat-{expected_repeat}/step-{expected_step}-faces-{output_faces}.ply"
            )
            if name != expected_name:
                raise WitnessError("producer report archive member naming is invalid")
            if name in expected_members:
                raise WitnessError(f"producer report repeats archive member {name}")
            expected_members[name] = {"member": member, "step": step}
            previous_output_faces = output_faces
            if removed_fraction < THRESHOLD_GROWTH_CUTOFF:
                expected_threshold *= THRESHOLD_GROWTH
    if list(expected_members) != archive_record.get("members"):
        raise WitnessError("producer report member list differs from trajectory records")
    if len(expected_members) != archive_record.get("member_count"):
        raise WitnessError("producer report member count is inconsistent")

    identity_fields = (
        "output_vertices",
        "output_faces",
        "vertex_sha256",
        "face_sha256",
    )
    expected_step_stability: list[dict[str, Any]] = []
    all_exact = True
    for index in range(8):
        records = [run["steps"][index] for run in runs]
        exact = all(
            all(record[field] == records[0][field] for field in identity_fields)
            for record in records[1:]
        )
        all_exact = all_exact and exact
        expected_step_stability.append(
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
    if report.get("step_stability") != expected_step_stability:
        raise WitnessError("producer report step stability summary is invalid")
    if report.get("repeat_stability") != {"all_steps_exact": all_exact}:
        raise WitnessError("producer report repeat stability summary is invalid")

    try:
        with zipfile.ZipFile(archive_path) as bundle:
            names = bundle.namelist()
            if len(names) != len(set(names)):
                raise WitnessError("archive contains duplicate member names")
            if set(names) != set(expected_members):
                raise WitnessError("archive member set differs from producer report")
            with tempfile.TemporaryDirectory(prefix="cumesh-pair-admission-") as directory:
                scratch = Path(directory) / "member.ply"
                for name, records in expected_members.items():
                    expected = records["member"]
                    step = records["step"]
                    payload = bundle.read(name)
                    if len(payload) != expected.get("size_bytes"):
                        raise WitnessError(f"archive member size differs for {name}")
                    if hashlib.sha256(payload).hexdigest() != expected.get("sha256"):
                        raise WitnessError(f"archive member SHA256 differs for {name}")
                    scratch.write_bytes(payload)
                    vertices, faces = read_binary_ply(scratch)
                    _validate_mesh(vertices, faces)
                    if len(vertices) != step["output_vertices"] or len(faces) != step["output_faces"]:
                        raise WitnessError(f"PLY count differs from step record for {name}")
                    if _array_sha256(vertices) != step["vertex_sha256"]:
                        raise WitnessError(f"PLY vertex digest differs from step record for {name}")
                    if _array_sha256(faces) != step["face_sha256"]:
                        raise WitnessError(f"PLY face digest differs from step record for {name}")
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise WitnessError(f"invalid downloaded output archive: {exc}") from exc
    return {
        "schema": "trellis2mlx.source_cuda_cumesh_progressive_pair_admission.v1",
        "report_sha256": sha256_file(report_path),
        "archive_sha256": actual_sha256,
        "member_count": len(expected_members),
        "status": "admitted",
    }


def run_witness(
    *,
    input_ply: Path,
    output_archive: Path,
    output_json: Path,
    expected_input_sha256: str,
    work_dir: Path,
    repeats: int = 5,
    max_steps: int = 8,
    runtime_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    input_ply = Path(input_ply)
    output_archive = Path(output_archive)
    requested_output_json = Path(output_json)
    work_dir = Path(work_dir)
    destructive_roots = [work_dir / "TRELLIS.2", work_dir / "CuMesh"]
    archive_temporary = _atomic_temporary(output_archive)
    effective_output_json, report_rerouted = _effective_progressive_report_path(
        requested_output_json,
        protected_paths=[input_ply, output_archive, archive_temporary],
        destructive_roots=destructive_roots,
    )
    report: dict[str, Any] = {
        "schema": "trellis2mlx.source_cuda_cumesh_progressive_replicates.v1",
        "status": "failed",
        "failure_phase": None,
        "last_trustworthy_phase": "request_received",
        "last_trustworthy_evidence": None,
        "primary_output_status": "not_started",
        "requested_output_json": str(requested_output_json),
        "effective_output_json": str(effective_output_json),
        "report_rerouted": report_rerouted,
        "requested_route": {
            "input_ply": str(input_ply),
            "expected_input_sha256": expected_input_sha256,
            "output_archive": str(output_archive),
            "work_dir": str(work_dir),
            "trellis_commit": TRELLIS_COMMIT,
            "cumesh_commit": CUMESH_COMMIT,
            "geometry_route": GEOMETRY_ROUTE,
            "repeats": int(repeats),
            "max_steps": int(max_steps),
            "lambda_edge_length": LAMBDA_EDGE_LENGTH,
            "lambda_skinny": LAMBDA_SKINNY,
            "initial_threshold": INITIAL_THRESHOLD,
        },
        "effective_route": None,
        "effective_config": {
            "fresh_mesh_per_repeat": True,
            "adjacency_order": ADJACENCY_ORDER,
            "canonical_adjacency": False,
            "instrumentation": None,
            "target_face_count": None,
            "threshold_growth": THRESHOLD_GROWTH,
            "threshold_growth_predicate": "removed_faces / input_faces < 0.01",
        },
        "input_mesh": None,
        "runs": None,
        "step_stability": None,
        "repeat_stability": None,
        "output_archive": None,
        "partial_output_cleanup": None,
        "setup_commands": [],
        "elapsed_seconds": None,
    }
    phase = "request_validation"
    member_dir: Path | None = None
    member_records: list[dict[str, Any]] = []
    archive_owned = False

    try:
        if any(
            _paths_overlap(path, root)
            for path in (
                input_ply,
                output_archive,
                archive_temporary,
                requested_output_json,
                _atomic_temporary(requested_output_json),
                effective_output_json,
                _atomic_temporary(effective_output_json),
            )
            for root in destructive_roots
        ):
            raise WitnessError("input or output path overlaps a destructive runtime root")
        if _same_path(input_ply, archive_temporary):
            raise WitnessError("archive temporary path collides with protected input")
        if any(
            (
                _same_path(input_ply, output_archive),
                _same_path(input_ply, requested_output_json),
                _same_path(output_archive, requested_output_json),
            )
        ):
            raise WitnessError("input and output paths must be distinct")
        if output_archive.suffix != ".zip":
            raise WitnessError("--output-archive must end in .zip")
        if not HEX_SHA256.fullmatch(expected_input_sha256):
            raise WitnessError("--expected-input-sha256 must be 64 lowercase hex characters")
        if repeats < 2 or max_steps < 1:
            raise WitnessError("repeats must be at least two and max-steps must be positive")

        phase = "input_validation"
        if not input_ply.is_file():
            raise WitnessError(f"input PLY does not exist: {input_ply}")
        actual_input_sha256 = sha256_file(input_ply)
        if actual_input_sha256 != expected_input_sha256:
            raise WitnessError("input PLY SHA256 mismatch")
        source_vertices, source_faces = read_binary_ply(input_ply)
        _validate_mesh(source_vertices, source_faces)
        report["input_mesh"] = {
            "sha256": actual_input_sha256,
            "size_bytes": input_ply.stat().st_size,
            "vertices": int(len(source_vertices)),
            "faces": int(len(source_faces)),
            "vertex_sha256": _array_sha256(source_vertices),
            "face_sha256": _array_sha256(source_faces),
        }
        report["last_trustworthy_phase"] = "input_validated"
        _write_report(effective_output_json, report)

        phase = "stale_output_cleanup"
        output_archive.parent.mkdir(parents=True, exist_ok=True)
        if output_archive.exists():
            if not output_archive.is_file():
                raise WitnessError(f"stale output archive is not a file: {output_archive}")
            output_archive.unlink()
        report["last_trustworthy_phase"] = "stale_output_removed"
        _write_report(effective_output_json, report)

        phase = "runtime_setup"
        runtime = (runtime_factory or prepare_release_runtime)(
            work_dir=work_dir,
            report=report,
        )
        phase = "runtime_validation"
        _validate_native_route(runtime.effective_route)
        report["effective_route"] = {
            **runtime.effective_route,
            "geometry_route": GEOMETRY_ROUTE,
            "input_sha256": actual_input_sha256,
            "adjacency_order": ADJACENCY_ORDER,
        }
        report["status"] = "running"
        report["last_trustworthy_phase"] = "runtime_validated"
        _write_report(effective_output_json, report)

        phase = "trajectory_collection"
        member_dir = Path(tempfile.mkdtemp(prefix="cumesh-progressive-members-"))
        runs: list[dict[str, Any]] = []
        for repeat in range(1, repeats + 1):
            if sha256_file(input_ply) != actual_input_sha256:
                raise WitnessError("input PLY changed during trajectory collection")
            mesh = runtime.create_mesh(source_vertices.copy(), source_faces.copy())
            threshold = INITIAL_THRESHOLD
            steps: list[dict[str, Any]] = []
            run_started = time.perf_counter()
            for step in range(1, max_steps + 1):
                input_faces = int(mesh.num_faces)
                step_started = time.perf_counter()
                output_vertices, output_faces, vertices, faces = _step_mesh(
                    runtime,
                    mesh,
                    threshold,
                )
                if output_faces > input_faces:
                    raise WitnessError("simplify step increased the face count")
                member = f"repeat-{repeat}/step-{step}-faces-{output_faces}.ply"
                member_path = member_dir / member
                write_binary_ply(member_path, vertices, faces)
                member_record = {
                    "member": member,
                    "sha256": sha256_file(member_path),
                    "size_bytes": member_path.stat().st_size,
                }
                member_records.append(member_record)
                removed_faces = input_faces - output_faces
                step_record = {
                    "step": step,
                    "threshold": threshold,
                    "input_faces": input_faces,
                    "output_vertices": output_vertices,
                    "output_faces": output_faces,
                    "removed_faces": removed_faces,
                    "removed_fraction": removed_faces / max(input_faces, 1),
                    "vertex_sha256": _array_sha256(vertices),
                    "face_sha256": _array_sha256(faces),
                    "member": member_record,
                    "elapsed_seconds": time.perf_counter() - step_started,
                }
                steps.append(step_record)
                report["last_trustworthy_evidence"] = {
                    "repeat": repeat,
                    **step_record,
                }
                if step_record["removed_fraction"] < THRESHOLD_GROWTH_CUTOFF:
                    threshold *= THRESHOLD_GROWTH
            runs.append(
                {
                    "repeat": repeat,
                    "steps": steps,
                    "elapsed_seconds": time.perf_counter() - run_started,
                }
            )
        report["runs"] = runs
        report["primary_output_status"] = "partial"
        report["last_trustworthy_phase"] = "trajectories_collected"
        _write_report(effective_output_json, report)

        phase = "stability_analysis"
        identity_fields = (
            "output_vertices",
            "output_faces",
            "vertex_sha256",
            "face_sha256",
        )
        step_stability: list[dict[str, Any]] = []
        all_exact = True
        for index in range(max_steps):
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
        report["step_stability"] = step_stability
        report["repeat_stability"] = {"all_steps_exact": all_exact}

        phase = "archive_write"
        report["output_archive"] = _archive_members(
            output_archive,
            member_dir,
            member_records,
        )
        archive_owned = True

        phase = "postflight_validation"
        if sha256_file(input_ply) != actual_input_sha256:
            raise WitnessError("input PLY changed before report publication")
        _validate_native_route(runtime.effective_route)
        report.update(
            {
                "status": "done",
                "failure_phase": None,
                "last_trustworthy_phase": "archive_validated",
                "primary_output_status": "validated",
            }
        )
    except Exception as exc:
        partial_count = len(member_records)
        if archive_owned and output_archive.exists():
            output_archive.unlink()
        report.update(
            {
                "status": "failed",
                "failure_phase": phase,
                "error": f"{type(exc).__name__}: {exc}",
                "primary_output_status": (
                    "partial_removed" if partial_count or archive_owned else "not_started"
                ),
                "partial_output_cleanup": {
                    "member_count": partial_count,
                    "archive_removed": not output_archive.exists(),
                },
            }
        )
        raise WitnessError(report["error"]) from exc
    finally:
        if member_dir is not None:
            shutil.rmtree(member_dir, ignore_errors=True)
        report["elapsed_seconds"] = time.perf_counter() - started
        _write_report(effective_output_json, report)

    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_witness(
            input_ply=args.input_ply,
            output_archive=args.output_archive,
            output_json=args.output_json,
            expected_input_sha256=args.expected_input_sha256,
            work_dir=args.work_dir,
            repeats=args.repeats,
            max_steps=args.max_steps,
        )
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
