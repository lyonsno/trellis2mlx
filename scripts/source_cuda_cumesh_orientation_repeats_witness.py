"""Repeat release CuMesh face-orientation unification on one exact CUDA mesh."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

try:
    from scripts.source_cuda_cumesh_postprocess_witness import (
        CUMESH_COMMIT,
        EXPECTED_CUDA_CAPABILITY,
        EXPECTED_CUDA_DEVICE_NAME,
        HEX_SHA256,
        TRELLIS_COMMIT,
        TRELLIS_POSTPROCESS_SHA256,
        WitnessError,
        _effective_report_path,
        _same_path,
        _tensor_to_numpy,
        _validate_effective_route,
        _write_report,
        elapsed,
        mesh_summary,
        prepare_release_runtime,
        read_binary_ply,
        sha256_file,
    )
except ModuleNotFoundError:
    from source_cuda_cumesh_postprocess_witness import (
        CUMESH_COMMIT,
        EXPECTED_CUDA_CAPABILITY,
        EXPECTED_CUDA_DEVICE_NAME,
        HEX_SHA256,
        TRELLIS_COMMIT,
        TRELLIS_POSTPROCESS_SHA256,
        WitnessError,
        _effective_report_path,
        _same_path,
        _tensor_to_numpy,
        _validate_effective_route,
        _write_report,
        elapsed,
        mesh_summary,
        prepare_release_runtime,
        read_binary_ply,
        sha256_file,
    )


SCHEMA = "trellis2mlx.source_cuda_cumesh_orientation_repeats.v1"
GEOMETRY_ROUTE = "release-cumesh-orientation-only-multirepeat"
FORBIDDEN_INFERENCES = [
    "not full TRELLIS.2 geometry-postprocess evidence",
    "not UV unwrap, texture bake, or final-material evidence",
    "not evidence that the input mesh came from official CUDA inference",
    "local edge-conflict count is not a visual-quality score",
    "a repeated CUDA assignment is not by itself a corrected Metal objective",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--repeats", required=True, type=int)
    parser.add_argument("--work-dir", required=True, type=Path)
    return parser


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _effective_report_custody(
    requested: Path,
    *,
    protected_paths: list[tuple[str, Path]],
) -> tuple[Path, bool, str | None]:
    paths = [path for _, path in protected_paths]
    effective, rerouted = _effective_report_path(
        requested,
        protected_paths=paths,
    )
    temporary = effective.with_name(effective.name + ".tmp")
    collision = next(
        (
            label
            for label, protected in protected_paths
            if _same_path(temporary, protected)
        ),
        None,
    )
    while any(_same_path(temporary, protected) for protected in paths):
        effective, _ = _effective_report_path(
            effective.with_name(effective.name + ".failure.json"),
            protected_paths=paths,
        )
        temporary = effective.with_name(effective.name + ".tmp")
        rerouted = True
    return effective, rerouted, collision


def _orientation_delta(left: np.ndarray, right: np.ndarray) -> dict[str, int]:
    if left.shape != right.shape:
        raise WitnessError(
            f"orientation comparison shape mismatch: {left.shape} != {right.shape}"
        )
    same = np.zeros(len(left), dtype=bool)
    reversed_rows = np.zeros(len(left), dtype=bool)
    reversed_right = right[:, [0, 2, 1]]
    for shift in range(3):
        same |= np.all(left == np.roll(right, shift, axis=1), axis=1)
        reversed_rows |= np.all(
            left == np.roll(reversed_right, shift, axis=1),
            axis=1,
        )
    return {
        "same": int(same.sum()),
        "reversed": int(reversed_rows.sum()),
        "neither": int((~(same | reversed_rows)).sum()),
    }


def _same_direction_conflicts(faces: np.ndarray) -> dict[str, int]:
    directed = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]),
        axis=0,
    )
    undirected = np.sort(directed, axis=1)
    order = np.lexsort((undirected[:, 1], undirected[:, 0]))
    sorted_edges = undirected[order]
    sorted_directed = directed[order]
    starts_mask = np.empty(len(sorted_edges), dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = np.any(sorted_edges[1:] != sorted_edges[:-1], axis=1)
    starts = np.flatnonzero(starts_mask)
    ends = np.concatenate((starts[1:], np.array([len(sorted_edges)])))
    counts = ends - starts
    manifold_starts = starts[counts == 2]
    first = sorted_directed[manifold_starts]
    second = sorted_directed[manifold_starts + 1]
    conflicts = np.all(first == second, axis=1)
    return {
        "undirected_edges": int(len(starts)),
        "boundary_edges": int((counts == 1).sum()),
        "manifold_edges": int(len(manifold_starts)),
        "nonmanifold_edges": int((counts > 2).sum()),
        "same_direction_conflicts": int(conflicts.sum()),
        "opposite_direction_manifold_edges": int((~conflicts).sum()),
    }


def _default_collector(
    runtime: Any,
    vertices: np.ndarray,
    faces: np.ndarray,
    repeats: int,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for _ in range(repeats):
        mesh = runtime.create_mesh(vertices, faces)
        runtime.torch.cuda.synchronize()
        started = time.perf_counter()
        mesh.unify_face_orientations()
        runtime.torch.cuda.synchronize()
        raw_vertices, raw_faces = runtime.read_mesh(mesh)
        samples.append(
            {
                "vertices": _tensor_to_numpy(raw_vertices, np.float32),
                "faces": _tensor_to_numpy(raw_faces, np.int32),
                "elapsed_seconds": elapsed(started),
            }
        )
    return samples


def run_witness(
    *,
    input_ply: Path,
    output_npz: Path,
    output_json: Path,
    expected_input_sha256: str,
    repeats: int,
    work_dir: Path,
    runtime_factory: Callable[..., Any] | None = None,
    collector: Callable[..., list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    input_ply = Path(input_ply)
    output_npz = Path(output_npz)
    requested_output_json = Path(output_json)
    work_dir = Path(work_dir)
    temporary_output_npz = output_npz.with_name(output_npz.name + ".tmp")
    effective_output_json, report_rerouted, report_temp_collision = (
        _effective_report_custody(
            requested_output_json,
            protected_paths=[
                ("protected input", input_ply),
                ("output NPZ", output_npz),
                ("temporary output NPZ", temporary_output_npz),
            ],
        )
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "failure_phase": None,
        "last_trustworthy_phase": "request_received",
        "primary_output_status": "not_started",
        "requested_output_json": str(requested_output_json),
        "effective_output_json": str(effective_output_json),
        "report_rerouted": report_rerouted,
        "requested_route": {
            "input_ply": str(input_ply),
            "expected_input_sha256": expected_input_sha256,
            "output_npz": str(output_npz),
            "repeats": int(repeats),
            "geometry_route": GEOMETRY_ROUTE,
            "trellis_commit": TRELLIS_COMMIT,
            "trellis_postprocess_sha256": TRELLIS_POSTPROCESS_SHA256,
            "cumesh_commit": CUMESH_COMMIT,
            "cuda_device_name": EXPECTED_CUDA_DEVICE_NAME,
            "cuda_capability": list(EXPECTED_CUDA_CAPABILITY),
            "target_faces": None,
        },
        "effective_route": None,
        "input_mesh": None,
        "repeat_count": 0,
        "repeats": [],
        "pairwise": {},
        "forbidden_inferences": FORBIDDEN_INFERENCES,
        "setup_commands": [],
    }
    phase = "request_validation"
    output_cleanup_owned = False
    try:
        if report_temp_collision is not None:
            raise WitnessError(
                "report temporary output aliases " + report_temp_collision
            )
        if _same_path(input_ply, output_npz):
            raise WitnessError("output NPZ aliases protected input")
        if _same_path(requested_output_json, output_npz):
            raise WitnessError("output JSON aliases output NPZ")
        if _same_path(temporary_output_npz, requested_output_json):
            raise WitnessError("temporary output aliases requested report")
        if _same_path(temporary_output_npz, input_ply):
            raise WitnessError("temporary output aliases protected input")
        if _same_path(temporary_output_npz, effective_output_json):
            raise WitnessError("temporary output aliases effective report")
        if not HEX_SHA256.fullmatch(expected_input_sha256):
            raise WitnessError(
                "--expected-input-sha256 must be 64 lowercase hex characters"
            )
        if repeats <= 0:
            raise WitnessError("--repeats must be positive")

        phase = "input_validation"
        if not input_ply.is_file():
            raise WitnessError(f"input PLY does not exist: {input_ply}")
        actual_input_sha256 = sha256_file(input_ply)
        if actual_input_sha256 != expected_input_sha256:
            raise WitnessError(
                "input SHA256 mismatch: "
                f"expected {expected_input_sha256}, got {actual_input_sha256}"
            )
        vertices, faces = read_binary_ply(input_ply)
        if not len(vertices) or not len(faces):
            raise WitnessError("input PLY must contain vertices and faces")
        report["input_mesh"] = {
            "sha256": actual_input_sha256,
            "size_bytes": input_ply.stat().st_size,
            **mesh_summary(vertices, faces),
        }
        report["requested_route"]["target_faces"] = int(len(faces))
        report["last_trustworthy_phase"] = "input_validated"
        _write_report(effective_output_json, report)

        phase = "stale_output_cleanup"
        for path, label in (
            (output_npz, "stale output NPZ"),
            (temporary_output_npz, "stale temporary output NPZ"),
        ):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.exists():
                raise WitnessError(f"{label} is not a file: {path}")
        output_cleanup_owned = True
        report["last_trustworthy_phase"] = "stale_output_removed"
        _write_report(effective_output_json, report)

        phase = "runtime_setup"
        runtime_builder = runtime_factory or prepare_release_runtime
        runtime = runtime_builder(work_dir=work_dir, report=report)
        report["effective_route"] = {
            **dict(runtime.effective_route),
            "geometry_route": GEOMETRY_ROUTE,
            "input_ply_sha256": actual_input_sha256,
            "repeats": int(repeats),
            "target_faces": int(len(faces)),
        }

        phase = "runtime_validation"
        _validate_effective_route(report["effective_route"])
        report["last_trustworthy_phase"] = "runtime_validated"
        report["status"] = "running"
        _write_report(effective_output_json, report)

        phase = "collection"
        collect = collector or _default_collector
        samples = collect(runtime, vertices, faces, int(repeats))

        phase = "collection_validation"
        if len(samples) != repeats:
            raise WitnessError(
                f"collector returned {len(samples)} repeats, expected {repeats}"
            )
        arrays: dict[str, np.ndarray] = {
            "vertices": vertices,
            "input_faces": faces,
        }
        sample_faces: list[np.ndarray] = []
        for index, sample in enumerate(samples):
            actual_vertices = np.ascontiguousarray(
                sample["vertices"],
                dtype=np.float32,
            )
            actual_faces = np.ascontiguousarray(sample["faces"], dtype=np.int32)
            if not np.array_equal(actual_vertices, vertices):
                raise WitnessError(f"repeat {index} changed vertices")
            if actual_faces.shape != faces.shape:
                raise WitnessError(
                    f"repeat {index} changed face shape: "
                    f"{actual_faces.shape} != {faces.shape}"
                )
            if not np.array_equal(
                np.sort(actual_faces, axis=1),
                np.sort(faces, axis=1),
            ):
                raise WitnessError(f"repeat {index} changed triangle membership")
            delta = _orientation_delta(actual_faces, faces)
            if delta["neither"]:
                raise WitnessError(
                    f"repeat {index} contains {delta['neither']} non-orientation deltas"
                )
            name = f"repeat_{index:02d}_faces"
            arrays[name] = actual_faces
            sample_faces.append(actual_faces)
            report["repeats"].append(
                {
                    "index": index,
                    "faces_sha256": _array_sha256(actual_faces),
                    "elapsed_seconds": float(sample["elapsed_seconds"]),
                    "versus_input": delta,
                    "topology": _same_direction_conflicts(actual_faces),
                }
            )

        for left in range(repeats):
            for right in range(left + 1, repeats):
                label = f"repeat_{left:02d}_vs_repeat_{right:02d}"
                report["pairwise"][label] = _orientation_delta(
                    sample_faces[left],
                    sample_faces[right],
                )
        report["repeat_count"] = int(repeats)
        report["last_trustworthy_phase"] = "all_repeats_validated_in_memory"
        _write_report(effective_output_json, report)

        phase = "output_write"
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        with temporary_output_npz.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary_output_npz.replace(output_npz)

        phase = "output_validation"
        with np.load(output_npz, allow_pickle=False) as reopened:
            if set(reopened.files) != set(arrays):
                raise WitnessError("reopened output NPZ keys changed")
            for name, expected in arrays.items():
                if not np.array_equal(reopened[name], expected):
                    raise WitnessError(f"reopened output array differs: {name}")

        report.update(
            {
                "status": "done",
                "failure_phase": None,
                "last_trustworthy_phase": "output_reopened_and_validated",
                "primary_output_status": "validated",
                "output_npz": {
                    "path": str(output_npz),
                    "sha256": sha256_file(output_npz),
                    "size_bytes": output_npz.stat().st_size,
                },
                "elapsed_seconds": elapsed(started),
            }
        )
        _write_report(effective_output_json, report)
        return report
    except Exception as exc:
        if output_cleanup_owned:
            protected_cleanup_paths = (
                input_ply,
                requested_output_json,
                effective_output_json,
            )
            for path in (output_npz, temporary_output_npz):
                aliases_protected = any(
                    _same_path(path, protected)
                    for protected in protected_cleanup_paths
                )
                if not aliases_protected and (
                    path.is_file() or path.is_symlink()
                ):
                    path.unlink()
        report.update(
            {
                "status": "failed",
                "failure_phase": phase,
                "primary_output_status": "not_started",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": elapsed(started),
            }
        )
        _write_report(effective_output_json, report)
        raise


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_witness(
            input_ply=args.input_ply,
            output_npz=args.output_npz,
            output_json=args.output_json,
            expected_input_sha256=args.expected_input_sha256,
            repeats=args.repeats,
            work_dir=args.work_dir,
        )
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
