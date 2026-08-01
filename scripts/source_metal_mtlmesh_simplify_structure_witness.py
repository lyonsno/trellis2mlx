"""Capture the Metal connectivity state consumed by QEM simplify iteration one."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Callable

import numpy as np

from scripts.source_cuda_cumesh_postprocess_witness import (
    HEX_SHA256,
    WitnessError,
    _effective_report_path,
    _same_path,
    read_binary_ply,
    sha256_file,
)
from trellmlx.source_route_identity import (
    SourceRouteIdentityError,
    probe_cumesh_route_identity,
    validate_source_route_identity,
)


ARRAY_DTYPES = {
    "vert2face": np.dtype(np.int32),
    "vert2face_cnt": np.dtype(np.int32),
    "vert2face_offset": np.dtype(np.int32),
    "edges": np.dtype(np.int32),
    "edge2face_cnt": np.dtype(np.int32),
    "boundaries": np.dtype(np.int32),
    "vert_is_boundary": np.dtype(np.uint8),
}
HEX_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-source-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    return parser


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def _default_collector(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> dict[str, np.ndarray]:
    import torch
    from cumesh import CuMesh

    mesh = CuMesh()
    mesh.init(torch.from_numpy(vertices), torch.from_numpy(faces))
    mesh.get_vertex_face_adjacency()
    mesh.get_edges()
    mesh.get_boundary_info()
    cache = mesh.read_all_cache()
    return {
        name: np.ascontiguousarray(cache[name].detach().cpu().numpy())
        for name in ARRAY_DTYPES
    }


def _validate_arrays(
    arrays: dict[str, np.ndarray],
    *,
    num_vertices: int,
    num_faces: int,
) -> dict[str, np.ndarray]:
    missing = sorted(set(ARRAY_DTYPES) - set(arrays))
    extra = sorted(set(arrays) - set(ARRAY_DTYPES))
    if missing or extra:
        raise WitnessError(
            f"collector array set mismatch: missing={missing}, extra={extra}"
        )

    normalized: dict[str, np.ndarray] = {}
    for name, dtype in ARRAY_DTYPES.items():
        array = np.asarray(arrays[name])
        if array.dtype != dtype:
            raise WitnessError(
                f"{name} dtype mismatch: expected {dtype}, got {array.dtype}"
            )
        normalized[name] = np.ascontiguousarray(array)

    if normalized["vert2face"].shape != (num_faces * 3,):
        raise WitnessError("vert2face shape does not equal three entries per face")
    for name in ("vert2face_cnt", "vert2face_offset"):
        if normalized[name].shape != (num_vertices + 1,):
            raise WitnessError(f"{name} shape does not equal V+1")
    if normalized["edges"].ndim != 2 or normalized["edges"].shape[1:] != (2,):
        raise WitnessError("edges must have shape [E, 2]")
    num_edges = len(normalized["edges"])
    if normalized["edge2face_cnt"].shape != (num_edges,):
        raise WitnessError("edge2face_cnt shape does not equal E")
    if normalized["boundaries"].ndim != 1:
        raise WitnessError("boundaries must be one-dimensional")
    if normalized["vert_is_boundary"].shape != (num_vertices,):
        raise WitnessError("vert_is_boundary shape does not equal V")
    if normalized["vert2face_offset"][0] != 0:
        raise WitnessError("vert2face_offset must start at zero")
    if normalized["vert2face_offset"][-1] != num_faces * 3:
        raise WitnessError("vert2face_offset terminal value does not equal 3F")
    if np.any(np.diff(normalized["vert2face_offset"]) < 0):
        raise WitnessError("vert2face_offset is not monotonic")
    if np.any(normalized["vert2face"] < 0) or np.any(
        normalized["vert2face"] >= num_faces
    ):
        raise WitnessError("vert2face contains an out-of-range face index")
    if np.any(normalized["boundaries"] < 0) or np.any(
        normalized["boundaries"] >= num_edges
    ):
        raise WitnessError("boundaries contains an out-of-range edge index")
    return normalized


def run_witness(
    *,
    input_ply: Path,
    output_npz: Path,
    report_json: Path,
    expected_input_sha256: str,
    expected_source_root: Path,
    expected_source_commit: str,
    identity_probe: Callable[[], dict[str, Any]] | None = None,
    collector: Callable[[np.ndarray, np.ndarray], dict[str, np.ndarray]] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    input_ply = Path(input_ply)
    output_npz = Path(output_npz)
    requested_report_json = Path(report_json)
    expected_source_root = Path(expected_source_root).resolve(strict=False)
    effective_report_json, report_rerouted = _effective_report_path(
        requested_report_json,
        protected_paths=[input_ply, output_npz],
    )
    report: dict[str, Any] = {
        "schema": "trellis2mlx.source_metal_mtlmesh_simplify_structure_witness.v1",
        "status": "failed",
        "failure_phase": None,
        "last_trustworthy_phase": "request_received",
        "primary_output_status": "not_started",
        "requested_report_json": str(requested_report_json),
        "effective_report_json": str(effective_report_json),
        "report_rerouted": report_rerouted,
        "requested_route": {
            "input_ply": str(input_ply),
            "output_npz": str(output_npz),
            "expected_input_sha256": expected_input_sha256,
            "expected_source_root": str(expected_source_root),
            "expected_source_commit": expected_source_commit,
            "geometry_route": "metal-mtlmesh-simplify-structure",
        },
        "effective_route": None,
        "input_mesh": None,
        "arrays": None,
        "output_npz": None,
        "elapsed_seconds": None,
    }
    phase = "request_validation"

    try:
        if _same_path(input_ply, output_npz):
            raise WitnessError("output NPZ aliases protected input PLY")
        if output_npz.suffix != ".npz":
            raise WitnessError("--output-npz must end in .npz")
        if not HEX_SHA256.fullmatch(expected_input_sha256):
            raise WitnessError(
                "--expected-input-sha256 must be 64 lowercase hex characters"
            )
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
        vertices, faces = read_binary_ply(input_ply)
        report["input_mesh"] = {
            "sha256": actual_input_sha256,
            "vertices": int(len(vertices)),
            "faces": int(len(faces)),
        }
        report["last_trustworthy_phase"] = "input_validated"
        _write_report(effective_report_json, report)

        phase = "stale_output_cleanup"
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        if output_npz.exists():
            if not output_npz.is_file():
                raise WitnessError(f"stale output is not a file: {output_npz}")
            output_npz.unlink()
        report["last_trustworthy_phase"] = "stale_output_removed"
        _write_report(effective_report_json, report)

        phase = "runtime_validation"
        probe = identity_probe or probe_cumesh_route_identity
        try:
            identity = validate_source_route_identity(
                probe(),
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
            raise WitnessError("effective route does not expose MtlMesh")
        report["effective_route"] = {
            **identity,
            "input_ply": str(input_ply),
            "input_sha256": actual_input_sha256,
            "geometry_route": "metal-mtlmesh-simplify-structure",
        }
        report["status"] = "running"
        report["last_trustworthy_phase"] = "runtime_validated"
        _write_report(effective_report_json, report)

        phase = "structure_collection"
        collect = collector or _default_collector
        arrays = _validate_arrays(
            collect(vertices, faces),
            num_vertices=len(vertices),
            num_faces=len(faces),
        )
        report["arrays"] = {
            name: {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": _array_sha256(array),
                "size_bytes": int(array.nbytes),
            }
            for name, array in arrays.items()
        }
        report["last_trustworthy_phase"] = "structure_collected"
        report["primary_output_status"] = "partial"
        _write_report(effective_report_json, report)

        phase = "output_write"
        np.savez(output_npz, **arrays)

        phase = "output_validation"
        with np.load(output_npz, allow_pickle=False) as reopened:
            if set(reopened.files) != set(arrays):
                raise WitnessError("reopened NPZ array set differs from collected state")
            for name, expected in arrays.items():
                if not np.array_equal(reopened[name], expected):
                    raise WitnessError(f"reopened {name} differs from collected state")
        report["output_npz"] = {
            "path": str(output_npz),
            "sha256": sha256_file(output_npz),
            "size_bytes": output_npz.stat().st_size,
        }
        report["status"] = "done"
        report["failure_phase"] = None
        report["last_trustworthy_phase"] = "output_validated"
        report["primary_output_status"] = "validated"
    except Exception as exc:
        report["status"] = "failed"
        report["failure_phase"] = phase
        report["error"] = f"{type(exc).__name__}: {exc}"
        if output_npz.exists():
            output_npz.unlink()
        report["primary_output_status"] = "not_started"
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        _write_report(effective_report_json, report)

    if report["status"] != "done":
        raise WitnessError(report["error"])
    return report


def main() -> int:
    args = build_parser().parse_args()
    report = run_witness(
        input_ply=args.input_ply,
        output_npz=args.output_npz,
        report_json=args.report_json,
        expected_input_sha256=args.expected_input_sha256,
        expected_source_root=args.expected_source_root,
        expected_source_commit=args.expected_source_commit,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
