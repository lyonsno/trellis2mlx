"""Capture source-CUDA connectivity consumed by QEM simplify iteration one."""

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
        HEX_SHA256,
        WitnessError,
        _same_path,
        _validate_effective_route,
        prepare_release_runtime,
        read_binary_ply,
        sha256_file,
    )
except ModuleNotFoundError:
    from source_cuda_cumesh_postprocess_witness import (
        CUMESH_COMMIT,
        HEX_SHA256,
        WitnessError,
        _same_path,
        _validate_effective_route,
        prepare_release_runtime,
        read_binary_ply,
        sha256_file,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    return parser


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


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


def _default_collector(
    runtime: Any,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> dict[str, np.ndarray]:
    torch = runtime.torch
    mesh = runtime.cumesh.CuMesh()
    mesh.init(
        torch.from_numpy(vertices).cuda(),
        torch.from_numpy(faces).cuda(),
    )
    mesh.get_vertex_face_adjacency()
    mesh.get_edges()
    mesh.get_boundary_info()
    cache = mesh.read_all_cache()
    arrays = {
        name: np.ascontiguousarray(cache[name].detach().cpu().numpy())
        for name in ARRAY_DTYPES
    }
    # CUDA's generic uint64 buffer reader exposes little-endian words as
    # [low, high]; the edge's semantic encoding is [high=min, low=max].
    arrays["edges"] = np.ascontiguousarray(arrays["edges"][:, ::-1])
    return arrays


def run_witness(
    *,
    input_ply: Path,
    output_npz: Path,
    output_json: Path,
    expected_input_sha256: str,
    work_dir: Path,
    runtime_factory: Callable[..., Any] | None = None,
    collector: Callable[[Any, np.ndarray, np.ndarray], dict[str, np.ndarray]]
    | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    input_ply = Path(input_ply)
    output_npz = Path(output_npz)
    output_json = Path(output_json)
    work_dir = Path(work_dir)
    report: dict[str, Any] = {
        "schema": "trellis2mlx.source_cuda_cumesh_simplify_structure_witness.v1",
        "status": "failed",
        "failure_phase": None,
        "last_trustworthy_phase": "request_received",
        "primary_output_status": "not_started",
        "requested_route": {
            "input_ply": str(input_ply),
            "output_npz": str(output_npz),
            "output_json": str(output_json),
            "expected_input_sha256": expected_input_sha256,
            "work_dir": str(work_dir),
            "cumesh_commit": CUMESH_COMMIT,
            "geometry_route": "release-cumesh-simplify-structure",
            "target_faces": 1,
        },
        "effective_route": None,
        "input_mesh": None,
        "arrays": None,
        "output_npz": None,
        "elapsed_seconds": None,
        "setup_commands": [],
    }
    phase = "request_validation"

    try:
        if _same_path(input_ply, output_npz):
            raise WitnessError("output NPZ aliases protected input PLY")
        if _same_path(input_ply, output_json):
            raise WitnessError("output JSON aliases protected input PLY")
        if _same_path(output_npz, output_json):
            raise WitnessError("output NPZ aliases output JSON")
        if output_npz.suffix != ".npz":
            raise WitnessError("--output-npz must end in .npz")
        if not HEX_SHA256.fullmatch(expected_input_sha256):
            raise WitnessError(
                "--expected-input-sha256 must be 64 lowercase hex characters"
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
        _write_report(output_json, report)

        phase = "stale_output_cleanup"
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        if output_npz.exists():
            if not output_npz.is_file():
                raise WitnessError(f"stale output is not a file: {output_npz}")
            output_npz.unlink()
        report["last_trustworthy_phase"] = "stale_output_removed"
        _write_report(output_json, report)

        phase = "runtime_validation"
        make_runtime = runtime_factory or prepare_release_runtime
        runtime = make_runtime(work_dir=work_dir, report=report)
        _validate_effective_route(runtime.effective_route)
        report["effective_route"] = {
            **runtime.effective_route,
            "input_ply": str(input_ply),
            "input_sha256": actual_input_sha256,
            "geometry_route": "release-cumesh-simplify-structure",
            "edge_readback": "uint64-little-endian-words-canonicalized-to-min-max",
        }
        report["status"] = "running"
        report["last_trustworthy_phase"] = "runtime_validated"
        _write_report(output_json, report)

        phase = "structure_collection"
        collect = collector or _default_collector
        arrays = _validate_arrays(
            collect(runtime, vertices, faces),
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
        _write_report(output_json, report)

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
        _write_report(output_json, report)

    if report["status"] != "done":
        raise WitnessError(report["error"])
    return report


def main() -> int:
    args = build_parser().parse_args()
    report = run_witness(
        input_ply=args.input_ply,
        output_npz=args.output_npz,
        output_json=args.output_json,
        expected_input_sha256=args.expected_input_sha256,
        work_dir=args.work_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
