"""Capture one source-CUDA QEM simplify step and its consumed adjacency."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
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
    "post_vertices": np.dtype(np.float32),
    "post_faces": np.dtype(np.int32),
}
LAMBDA_EDGE_LENGTH = 1e-2
LAMBDA_SKINNY = 1e-3
THRESHOLD = 1e-8
INSTRUMENTATION_SCHEMA = (
    "trellis2mlx.cumesh_reuse_adjacency_instrumentation.v1"
)
INSTRUMENTED_FILES = (
    "src/cumesh.h",
    "src/ext.cpp",
    "src/simplify.cu",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--instrumentation-patch", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-patch-sha256", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    return parser


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _run(command: list[str], report: dict[str, Any]) -> subprocess.CompletedProcess:
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    report["setup_commands"].append(
        {
            "command": command,
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    if completed.returncode != 0:
        raise WitnessError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr or completed.stdout}"
        )
    return completed


def _instrumentation_callback(
    patch: Path,
    expected_patch_sha256: str,
) -> Callable[[Path, dict[str, Any]], dict[str, Any]]:
    patch = Path(patch).resolve(strict=False)

    def apply(cumesh_root: Path, report: dict[str, Any]) -> dict[str, Any]:
        _run(
            ["git", "-C", str(cumesh_root), "apply", "--check", str(patch)],
            report,
        )
        _run(["git", "-C", str(cumesh_root), "apply", str(patch)], report)
        _run(["git", "-C", str(cumesh_root), "diff", "--check"], report)
        status = _run(
            [
                "git",
                "-C",
                str(cumesh_root),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            report,
        ).stdout
        changed_files = sorted(
            line[3:] for line in status.splitlines() if len(line) >= 4
        )
        if changed_files != sorted(INSTRUMENTED_FILES):
            raise WitnessError(
                "instrumentation changed unexpected CuMesh files: "
                f"expected {sorted(INSTRUMENTED_FILES)}, got {changed_files}"
            )
        return {
            "schema": INSTRUMENTATION_SCHEMA,
            "patch_path": str(patch),
            "patch_sha256": expected_patch_sha256,
            "changed_files": changed_files,
            "base_commit": CUMESH_COMMIT,
            "diagnostic_only": True,
            "read_only_trace": False,
            "semantic_change": "reuse_precomputed_vertex_face_adjacency",
        }

    return apply


def _validate_arrays(
    arrays: dict[str, np.ndarray],
    *,
    num_faces: int,
) -> dict[str, np.ndarray]:
    if set(arrays) != set(ARRAY_DTYPES):
        raise WitnessError("collector array set mismatch")
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
    if (
        normalized["post_vertices"].ndim != 2
        or normalized["post_vertices"].shape[1:] != (3,)
    ):
        raise WitnessError("post_vertices must have shape [V, 3]")
    if (
        normalized["post_faces"].ndim != 2
        or normalized["post_faces"].shape[1:] != (3,)
    ):
        raise WitnessError("post_faces must have shape [F, 3]")
    if len(normalized["post_faces"]) and (
        np.any(normalized["post_faces"] < 0)
        or np.any(
            normalized["post_faces"] >= len(normalized["post_vertices"])
        )
    ):
        raise WitnessError("post_faces contains an out-of-range vertex index")
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
    adjacency = np.ascontiguousarray(
        mesh.read_all_cache()["vert2face"].detach().cpu().numpy(),
        dtype=np.int32,
    )
    mesh.cu_mesh.simplify_step(
        LAMBDA_EDGE_LENGTH,
        LAMBDA_SKINNY,
        THRESHOLD,
        False,
        True,
    )
    post_vertices, post_faces = mesh.read()
    return {
        "vert2face": adjacency,
        "post_vertices": np.ascontiguousarray(
            post_vertices.detach().cpu().numpy(),
            dtype=np.float32,
        ),
        "post_faces": np.ascontiguousarray(
            post_faces.detach().cpu().numpy(),
            dtype=np.int32,
        ),
    }


def run_witness(
    *,
    input_ply: Path,
    instrumentation_patch: Path,
    output_npz: Path,
    output_json: Path,
    expected_input_sha256: str,
    expected_patch_sha256: str,
    work_dir: Path,
    runtime_factory: Callable[..., Any] | None = None,
    collector: Callable[[Any, np.ndarray, np.ndarray], dict[str, np.ndarray]]
    | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    input_ply = Path(input_ply)
    instrumentation_patch = Path(instrumentation_patch)
    output_npz = Path(output_npz)
    output_json = Path(output_json)
    work_dir = Path(work_dir)
    report: dict[str, Any] = {
        "schema": "trellis2mlx.source_cuda_cumesh_first_simplify_step.v1",
        "status": "failed",
        "failure_phase": None,
        "last_trustworthy_phase": "request_received",
        "primary_output_status": "not_started",
        "requested_route": {
            "input_ply": str(input_ply),
            "instrumentation_patch": str(instrumentation_patch),
            "output_npz": str(output_npz),
            "output_json": str(output_json),
            "expected_input_sha256": expected_input_sha256,
            "expected_patch_sha256": expected_patch_sha256,
            "work_dir": str(work_dir),
            "cumesh_commit": CUMESH_COMMIT,
            "geometry_route": "release-cumesh-first-simplify-step",
            "target_faces": 1,
            "lambda_edge_length": LAMBDA_EDGE_LENGTH,
            "lambda_skinny": LAMBDA_SKINNY,
            "threshold": THRESHOLD,
            "consumed_adjacency_source": "captured-source-cuda-vert2face",
        },
        "effective_route": None,
        "instrumentation_patch": None,
        "input_mesh": None,
        "output_mesh": None,
        "arrays": None,
        "output_npz": None,
        "elapsed_seconds": None,
        "setup_commands": [],
    }
    phase = "request_validation"

    try:
        if any(
            (
                _same_path(input_ply, output_npz),
                _same_path(input_ply, output_json),
                _same_path(output_npz, output_json),
                _same_path(instrumentation_patch, output_npz),
                _same_path(instrumentation_patch, output_json),
            )
        ):
            raise WitnessError("input and output paths must be distinct")
        if output_npz.suffix != ".npz":
            raise WitnessError("--output-npz must end in .npz")
        if not HEX_SHA256.fullmatch(expected_input_sha256):
            raise WitnessError(
                "--expected-input-sha256 must be 64 lowercase hex characters"
            )
        if not HEX_SHA256.fullmatch(expected_patch_sha256):
            raise WitnessError(
                "--expected-patch-sha256 must be 64 lowercase hex characters"
            )

        phase = "input_validation"
        if sha256_file(input_ply) != expected_input_sha256:
            raise WitnessError("input PLY SHA256 mismatch")
        vertices, faces = read_binary_ply(input_ply)
        report["input_mesh"] = {
            "sha256": expected_input_sha256,
            "vertices": int(len(vertices)),
            "faces": int(len(faces)),
        }
        report["last_trustworthy_phase"] = "input_validated"
        _write_report(output_json, report)

        phase = "patch_validation"
        if not instrumentation_patch.is_file():
            raise WitnessError(
                f"instrumentation patch does not exist: {instrumentation_patch}"
            )
        if sha256_file(instrumentation_patch) != expected_patch_sha256:
            raise WitnessError("instrumentation patch SHA256 mismatch")
        report["instrumentation_patch"] = {
            "path": str(instrumentation_patch),
            "sha256": expected_patch_sha256,
            "size_bytes": instrumentation_patch.stat().st_size,
        }
        report["last_trustworthy_phase"] = "patch_validated"
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
        runtime = make_runtime(
            work_dir=work_dir,
            report=report,
            cumesh_instrumentation=_instrumentation_callback(
                instrumentation_patch,
                expected_patch_sha256,
            ),
        )
        _validate_effective_route(runtime.effective_route)
        instrumentation = runtime.effective_route.get("cumesh_instrumentation")
        if (
            not isinstance(instrumentation, dict)
            or instrumentation.get("schema") != INSTRUMENTATION_SCHEMA
            or instrumentation.get("patch_sha256") != expected_patch_sha256
            or instrumentation.get("changed_files")
            != list(INSTRUMENTED_FILES)
        ):
            raise WitnessError(
                "effective route lacks authenticated adjacency-reuse instrumentation"
            )
        report["effective_route"] = {
            **runtime.effective_route,
            "input_sha256": expected_input_sha256,
            "geometry_route": "release-cumesh-first-simplify-step",
            "lambda_edge_length": LAMBDA_EDGE_LENGTH,
            "lambda_skinny": LAMBDA_SKINNY,
            "threshold": THRESHOLD,
            "consumed_adjacency_source": "captured-source-cuda-vert2face",
        }
        report["status"] = "running"
        report["last_trustworthy_phase"] = "runtime_validated"
        _write_report(output_json, report)

        phase = "first_step_collection"
        arrays = _validate_arrays(
            (collector or _default_collector)(runtime, vertices, faces),
            num_faces=len(faces),
        )
        report["output_mesh"] = {
            "vertices": int(len(arrays["post_vertices"])),
            "faces": int(len(arrays["post_faces"])),
        }
        report["arrays"] = {
            name: {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": _array_sha256(array),
                "size_bytes": int(array.nbytes),
            }
            for name, array in arrays.items()
        }
        report["last_trustworthy_phase"] = "first_step_collected"
        report["primary_output_status"] = "partial"
        _write_report(output_json, report)

        phase = "output_write"
        np.savez(output_npz, **arrays)

        phase = "output_validation"
        with np.load(output_npz, allow_pickle=False) as reopened:
            if set(reopened.files) != set(arrays):
                raise WitnessError("reopened NPZ array set mismatch")
            for name, expected in arrays.items():
                if not np.array_equal(reopened[name], expected):
                    raise WitnessError(f"reopened {name} differs from collection")
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
        if work_dir.is_absolute() and work_dir.exists():
            try:
                shutil.rmtree(work_dir)
                report["runtime_cleanup"] = {
                    "path": str(work_dir),
                    "removed": True,
                }
            except OSError as exc:
                report["runtime_cleanup"] = {
                    "path": str(work_dir),
                    "removed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        report["elapsed_seconds"] = time.perf_counter() - started
        _write_report(output_json, report)

    if report["status"] != "done":
        raise WitnessError(report["error"])
    return report


def main() -> int:
    args = build_parser().parse_args()
    report = run_witness(
        input_ply=args.input_ply,
        instrumentation_patch=args.instrumentation_patch,
        output_npz=args.output_npz,
        output_json=args.output_json,
        expected_input_sha256=args.expected_input_sha256,
        expected_patch_sha256=args.expected_patch_sha256,
        work_dir=args.work_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
