"""Capture source-CUDA first-step QEMs and edge costs with read-only instrumentation."""

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
    "qems": np.dtype(np.float32),
    "edge_collapse_costs": np.dtype(np.float32),
}
COMPONENT_ARRAY_DTYPES = {
    **ARRAY_DTYPES,
    "component_edge_collapse_costs": np.dtype(np.float32),
    "qem_costs": np.dtype(np.float32),
    "edge_length2": np.dtype(np.float32),
    "skinny_avgs": np.dtype(np.float32),
    "skinny_terms": np.dtype(np.float32),
}
BASE_INSTRUMENTATION_SCHEMA = "trellis2mlx.cumesh_qem_cost_instrumentation.v1"
COMPONENT_INSTRUMENTATION_SCHEMA = (
    "trellis2mlx.cumesh_qem_cost_component_instrumentation.v2"
)
CANONICAL_COMPONENT_INSTRUMENTATION_SCHEMA = (
    "trellis2mlx.cumesh_canonical_qem_cost_component_instrumentation.v1"
)
BASE_GEOMETRY_ROUTE = "release-cumesh-qem-cost-trace-instrumented"
COMPONENT_GEOMETRY_ROUTE = (
    "release-cumesh-qem-cost-component-trace-instrumented"
)
CANONICAL_COMPONENT_GEOMETRY_ROUTE = (
    "release-cumesh-canonical-adjacency-qem-cost-component-"
    "trace-instrumented"
)
INSTRUMENTED_FILES = (
    "cumesh/cumesh.py",
    "src/cumesh.h",
    "src/ext.cpp",
    "src/simplify.cu",
)
CANONICAL_INSTRUMENTED_FILES = (
    "cumesh/cumesh.py",
    "src/connectivity.cu",
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
    parser.add_argument("--component-trace", action="store_true")
    parser.add_argument("--allow-masked-attribution", action="store_true")
    parser.add_argument("--canonical-adjacency-patch", type=Path)
    parser.add_argument("--expected-canonical-adjacency-patch-sha256")
    parser.add_argument("--trace-adjacency-patch", type=Path)
    parser.add_argument("--expected-trace-adjacency-patch-sha256")
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
    *,
    component_trace: bool = False,
    canonical_adjacency_patch: Path | None = None,
    expected_canonical_adjacency_patch_sha256: str | None = None,
    trace_adjacency_patch: Path | None = None,
    expected_trace_adjacency_patch_sha256: str | None = None,
) -> Callable[[Path, dict[str, Any]], dict[str, Any]]:
    patch = Path(patch).resolve(strict=False)
    canonical_mode = canonical_adjacency_patch is not None
    patch_records = [
        {
            "role": "qem_component_trace",
            "path": str(patch),
            "sha256": expected_patch_sha256,
        }
    ]
    if canonical_mode:
        canonical_adjacency_patch = Path(
            canonical_adjacency_patch
        ).resolve(strict=False)
        trace_adjacency_patch = Path(
            trace_adjacency_patch
        ).resolve(strict=False)
        patch_records = [
            {
                "role": "canonical_adjacency",
                "path": str(canonical_adjacency_patch),
                "sha256": expected_canonical_adjacency_patch_sha256,
            },
            *patch_records,
            {
                "role": "trace_local_adjacency_sort",
                "path": str(trace_adjacency_patch),
                "sha256": expected_trace_adjacency_patch_sha256,
            },
        ]

    def apply(cumesh_root: Path, report: dict[str, Any]) -> dict[str, Any]:
        for record in patch_records:
            _run(
                [
                    "git",
                    "-C",
                    str(cumesh_root),
                    "apply",
                    "--check",
                    record["path"],
                ],
                report,
            )
            _run(
                [
                    "git",
                    "-C",
                    str(cumesh_root),
                    "apply",
                    record["path"],
                ],
                report,
            )
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
        expected_files = (
            CANONICAL_INSTRUMENTED_FILES
            if canonical_mode
            else INSTRUMENTED_FILES
        )
        if changed_files != sorted(expected_files):
            raise WitnessError(
                "instrumentation changed unexpected CuMesh files: "
                f"expected {sorted(expected_files)}, got {changed_files}"
            )
        return {
            "schema": (
                CANONICAL_COMPONENT_INSTRUMENTATION_SCHEMA
                if canonical_mode
                else COMPONENT_INSTRUMENTATION_SCHEMA
                if component_trace
                else BASE_INSTRUMENTATION_SCHEMA
            ),
            "patch_path": str(patch),
            "patch_sha256": expected_patch_sha256,
            "patches": patch_records if canonical_mode else None,
            "changed_files": changed_files,
            "base_commit": CUMESH_COMMIT,
            "read_only_trace": True,
            "adjacency_order": (
                "ascending-face-id-per-vertex"
                if canonical_mode
                else "native"
            ),
        }

    return apply


def _default_collector(
    runtime: Any,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    component_trace: bool = False,
) -> dict[str, np.ndarray]:
    torch = runtime.torch
    mesh = runtime.cumesh.CuMesh()
    mesh.init(
        torch.from_numpy(vertices).cuda(),
        torch.from_numpy(faces).cuda(),
    )
    trace = (
        mesh.read_qem_cost_component_trace()
        if component_trace
        else mesh.read_qem_cost_trace()
    )
    cache = mesh.read_all_cache()
    arrays = {
        "vert2face": np.ascontiguousarray(
            cache["vert2face"].detach().cpu().numpy()
        ),
    }
    trace_names = (
        COMPONENT_ARRAY_DTYPES if component_trace else ARRAY_DTYPES
    )
    arrays.update(
        {
            name: np.ascontiguousarray(trace[name].detach().cpu().numpy())
            for name in trace_names
            if name != "vert2face"
        }
    )
    return arrays


def _validate_arrays(
    arrays: dict[str, np.ndarray],
    *,
    num_vertices: int,
    num_faces: int,
    component_trace: bool = False,
) -> dict[str, np.ndarray]:
    array_dtypes = (
        COMPONENT_ARRAY_DTYPES if component_trace else ARRAY_DTYPES
    )
    if set(arrays) != set(array_dtypes):
        raise WitnessError(
            "collector array set mismatch: "
            f"expected {sorted(array_dtypes)}, got {sorted(arrays)}"
        )
    normalized: dict[str, np.ndarray] = {}
    for name, dtype in array_dtypes.items():
        array = np.asarray(arrays[name])
        if array.dtype != dtype:
            raise WitnessError(
                f"{name} dtype mismatch: expected {dtype}, got {array.dtype}"
            )
        normalized[name] = np.ascontiguousarray(array)
    if normalized["vert2face"].shape != (num_faces * 3,):
        raise WitnessError("vert2face shape does not equal three entries per face")
    if normalized["qems"].shape != (num_vertices, 10):
        raise WitnessError("qems shape must equal [V, 10]")
    if normalized["edge_collapse_costs"].ndim != 1:
        raise WitnessError("edge_collapse_costs must be one-dimensional")
    if len(normalized["edge_collapse_costs"]) == 0:
        raise WitnessError("edge_collapse_costs is empty")
    edge_shape = normalized["edge_collapse_costs"].shape
    for name in set(array_dtypes) - {"vert2face", "qems"}:
        if normalized[name].shape != edge_shape:
            raise WitnessError(
                f"{name} shape must equal edge_collapse_costs shape"
            )
    return normalized


def _component_self_consistency(
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    ordinary = arrays["edge_collapse_costs"].view(np.uint32)
    component = arrays["component_edge_collapse_costs"].view(np.uint32)
    mismatch = ordinary != component
    mismatch_count = int(np.count_nonzero(mismatch))
    first = int(np.flatnonzero(mismatch)[0]) if mismatch_count else None
    return {
        "bit_exact": mismatch_count == 0,
        "bit_mismatch_count": mismatch_count,
        "first_bit_mismatch_index": first,
        "ordinary_sha256": _array_sha256(arrays["edge_collapse_costs"]),
        "component_sha256": _array_sha256(
            arrays["component_edge_collapse_costs"]
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
    component_trace: bool = False,
    allow_masked_attribution: bool = False,
    canonical_adjacency_patch: Path | None = None,
    expected_canonical_adjacency_patch_sha256: str | None = None,
    trace_adjacency_patch: Path | None = None,
    expected_trace_adjacency_patch_sha256: str | None = None,
    runtime_factory: Callable[..., Any] | None = None,
    collector: Callable[[Any, np.ndarray, np.ndarray], dict[str, np.ndarray]]
    | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    input_ply = Path(input_ply)
    instrumentation_patch = Path(instrumentation_patch)
    canonical_values = (
        canonical_adjacency_patch,
        expected_canonical_adjacency_patch_sha256,
        trace_adjacency_patch,
        expected_trace_adjacency_patch_sha256,
    )
    canonical_mode = any(value is not None for value in canonical_values)
    if canonical_adjacency_patch is not None:
        canonical_adjacency_patch = Path(canonical_adjacency_patch)
    if trace_adjacency_patch is not None:
        trace_adjacency_patch = Path(trace_adjacency_patch)
    output_npz = Path(output_npz)
    output_json = Path(output_json)
    work_dir = Path(work_dir)
    geometry_route = (
        CANONICAL_COMPONENT_GEOMETRY_ROUTE
        if canonical_mode
        else COMPONENT_GEOMETRY_ROUTE
        if component_trace
        else BASE_GEOMETRY_ROUTE
    )
    report: dict[str, Any] = {
        "schema": (
            "trellis2mlx.source_cuda_cumesh_qem_cost_witness.v2"
            if component_trace
            else "trellis2mlx.source_cuda_cumesh_qem_cost_witness.v1"
        ),
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
            "geometry_route": geometry_route,
            "target_faces": 1,
            "allow_masked_attribution": allow_masked_attribution,
            "canonical_adjacency_patch": (
                str(canonical_adjacency_patch)
                if canonical_adjacency_patch is not None
                else None
            ),
            "expected_canonical_adjacency_patch_sha256": (
                expected_canonical_adjacency_patch_sha256
            ),
            "trace_adjacency_patch": (
                str(trace_adjacency_patch)
                if trace_adjacency_patch is not None
                else None
            ),
            "expected_trace_adjacency_patch_sha256": (
                expected_trace_adjacency_patch_sha256
            ),
        },
        "effective_route": None,
        "input_mesh": None,
        "instrumentation_patch": None,
        "arrays": None,
        "backend_self_consistency": None,
        "component_attribution": None,
        "output_npz": None,
        "runtime_cleanup": None,
        "elapsed_seconds": None,
        "setup_commands": [],
    }
    phase = "request_validation"

    try:
        if canonical_mode and (
            not all(value is not None for value in canonical_values)
            or not component_trace
        ):
            raise WitnessError(
                "canonical QEM route requires both additional patches, both "
                "expected SHA256 values, and --component-trace"
            )
        for left, right, message in (
            (input_ply, output_npz, "output NPZ aliases protected input PLY"),
            (input_ply, output_json, "output JSON aliases protected input PLY"),
            (instrumentation_patch, output_npz, "output NPZ aliases patch"),
            (instrumentation_patch, output_json, "output JSON aliases patch"),
            (output_npz, output_json, "output NPZ aliases output JSON"),
        ):
            if _same_path(left, right):
                raise WitnessError(message)
        for protected_patch in (
            canonical_adjacency_patch,
            trace_adjacency_patch,
        ):
            if protected_patch is not None and (
                _same_path(protected_patch, output_npz)
                or _same_path(protected_patch, output_json)
            ):
                raise WitnessError("output aliases canonical route patch")
        if output_npz.suffix != ".npz":
            raise WitnessError("--output-npz must end in .npz")
        if allow_masked_attribution and not component_trace:
            raise WitnessError(
                "--allow-masked-attribution requires --component-trace"
            )
        if not HEX_SHA256.fullmatch(expected_input_sha256):
            raise WitnessError(
                "--expected-input-sha256 must be 64 lowercase hex characters"
            )
        if not HEX_SHA256.fullmatch(expected_patch_sha256):
            raise WitnessError(
                "--expected-patch-sha256 must be 64 lowercase hex characters"
            )
        if canonical_mode:
            for name, value in (
                (
                    "--expected-canonical-adjacency-patch-sha256",
                    expected_canonical_adjacency_patch_sha256,
                ),
                (
                    "--expected-trace-adjacency-patch-sha256",
                    expected_trace_adjacency_patch_sha256,
                ),
            ):
                if not HEX_SHA256.fullmatch(value):
                    raise WitnessError(
                        f"{name} must be 64 lowercase hex characters"
                    )

        phase = "input_validation"
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

        phase = "patch_validation"
        if not instrumentation_patch.is_file():
            raise WitnessError(
                f"instrumentation patch does not exist: {instrumentation_patch}"
            )
        actual_patch_sha256 = sha256_file(instrumentation_patch)
        if actual_patch_sha256 != expected_patch_sha256:
            raise WitnessError(
                "instrumentation patch SHA256 mismatch: "
                f"expected {expected_patch_sha256}, got {actual_patch_sha256}"
            )
        report["instrumentation_patch"] = {
            "path": str(instrumentation_patch),
            "sha256": actual_patch_sha256,
            "size_bytes": instrumentation_patch.stat().st_size,
        }
        if canonical_mode:
            canonical_records = []
            for role, path, expected_sha256 in (
                (
                    "canonical_adjacency",
                    canonical_adjacency_patch,
                    expected_canonical_adjacency_patch_sha256,
                ),
                (
                    "trace_local_adjacency_sort",
                    trace_adjacency_patch,
                    expected_trace_adjacency_patch_sha256,
                ),
            ):
                if not path.is_file():
                    raise WitnessError(
                        f"{role} patch does not exist: {path}"
                    )
                actual_sha256 = sha256_file(path)
                if actual_sha256 != expected_sha256:
                    raise WitnessError(f"{role} patch SHA256 mismatch")
                canonical_records.append(
                    {
                        "role": role,
                        "path": str(path),
                        "sha256": actual_sha256,
                        "size_bytes": path.stat().st_size,
                    }
                )
            report["canonical_route_patches"] = canonical_records
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
                component_trace=component_trace,
                canonical_adjacency_patch=canonical_adjacency_patch,
                expected_canonical_adjacency_patch_sha256=(
                    expected_canonical_adjacency_patch_sha256
                ),
                trace_adjacency_patch=trace_adjacency_patch,
                expected_trace_adjacency_patch_sha256=(
                    expected_trace_adjacency_patch_sha256
                ),
            ),
        )
        _validate_effective_route(runtime.effective_route)
        instrumentation = runtime.effective_route.get("cumesh_instrumentation")
        if not isinstance(instrumentation, dict):
            raise WitnessError("effective route is missing CuMesh instrumentation")
        if instrumentation.get("patch_sha256") != expected_patch_sha256:
            raise WitnessError("effective instrumentation patch SHA256 mismatch")
        expected_instrumentation_schema = (
            CANONICAL_COMPONENT_INSTRUMENTATION_SCHEMA
            if canonical_mode
            else COMPONENT_INSTRUMENTATION_SCHEMA
            if component_trace
            else BASE_INSTRUMENTATION_SCHEMA
        )
        if instrumentation.get("schema") != expected_instrumentation_schema:
            raise WitnessError("effective instrumentation schema mismatch")
        expected_files = (
            CANONICAL_INSTRUMENTED_FILES
            if canonical_mode
            else INSTRUMENTED_FILES
        )
        if instrumentation.get("changed_files") != list(expected_files):
            raise WitnessError("effective instrumentation changed-file set mismatch")
        if canonical_mode:
            expected_patches = [
                {
                    "role": "canonical_adjacency",
                    "path": str(
                        canonical_adjacency_patch.resolve(strict=False)
                    ),
                    "sha256": expected_canonical_adjacency_patch_sha256,
                },
                {
                    "role": "qem_component_trace",
                    "path": str(
                        instrumentation_patch.resolve(strict=False)
                    ),
                    "sha256": expected_patch_sha256,
                },
                {
                    "role": "trace_local_adjacency_sort",
                    "path": str(
                        trace_adjacency_patch.resolve(strict=False)
                    ),
                    "sha256": expected_trace_adjacency_patch_sha256,
                },
            ]
            if instrumentation.get("patches") != expected_patches:
                raise WitnessError(
                    "effective canonical instrumentation patch stack mismatch"
                )
        report["effective_route"] = {
            **runtime.effective_route,
            "input_ply": str(input_ply),
            "input_sha256": actual_input_sha256,
            "geometry_route": geometry_route,
            "adjacency_order": (
                "ascending-face-id-per-vertex"
                if canonical_mode
                else "native"
            ),
        }
        report["status"] = "running"
        report["last_trustworthy_phase"] = "runtime_validated"
        _write_report(output_json, report)

        phase = "trace_collection"
        collected = (
            collector(runtime, vertices, faces)
            if collector is not None
            else _default_collector(
                runtime,
                vertices,
                faces,
                component_trace=component_trace,
            )
        )
        arrays = _validate_arrays(
            collected,
            num_vertices=len(vertices),
            num_faces=len(faces),
            component_trace=component_trace,
        )
        if component_trace:
            phase = "backend_self_consistency"
            self_consistency = _component_self_consistency(arrays)
            report["backend_self_consistency"] = self_consistency
            report["component_attribution"] = {
                "global_admitted": self_consistency["bit_exact"],
                "masked_admitted": (
                    allow_masked_attribution
                    and not self_consistency["bit_exact"]
                ),
                "rejected_edge_count": self_consistency[
                    "bit_mismatch_count"
                ],
                "mask_predicate": (
                    "component_edge_collapse_costs bits equal "
                    "edge_collapse_costs bits"
                ),
            }
            if (
                not self_consistency["bit_exact"]
                and not allow_masked_attribution
            ):
                raise WitnessError(
                    "CUDA component total differs from the ordinary edge-cost "
                    f"kernel at {self_consistency['bit_mismatch_count']} edges"
                )
        report["arrays"] = {
            name: {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": _array_sha256(array),
                "size_bytes": int(array.nbytes),
                "nonfinite": (
                    int(np.count_nonzero(~np.isfinite(array)))
                    if np.issubdtype(array.dtype, np.floating)
                    else 0
                ),
            }
            for name, array in arrays.items()
        }
        report["last_trustworthy_phase"] = "trace_collected"
        report["primary_output_status"] = "partial"
        _write_report(output_json, report)

        phase = "output_write"
        np.savez_compressed(output_npz, **arrays)

        phase = "output_validation"
        with np.load(output_npz, allow_pickle=False) as reopened:
            if set(reopened.files) != set(arrays):
                raise WitnessError("reopened NPZ array set differs from trace")
            for name, expected in arrays.items():
                if not np.array_equal(reopened[name], expected, equal_nan=True):
                    raise WitnessError(f"reopened {name} differs from trace")
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
        component_trace=args.component_trace,
        allow_masked_attribution=args.allow_masked_attribution,
        canonical_adjacency_patch=args.canonical_adjacency_patch,
        expected_canonical_adjacency_patch_sha256=(
            args.expected_canonical_adjacency_patch_sha256
        ),
        trace_adjacency_patch=args.trace_adjacency_patch,
        expected_trace_adjacency_patch_sha256=(
            args.expected_trace_adjacency_patch_sha256
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
