"""Run Metal QEM simplify with CUDA's captured first-step adjacency order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
from typing import Any, Callable

import numpy as np

from scripts.compare_simplify_structure_witnesses import (
    _adjacency_order,
    _load_witness,
)
from scripts.source_cuda_cumesh_postprocess_witness import (
    HEX_SHA256,
    WitnessError,
    _effective_report_path,
    _same_path,
    read_binary_ply,
    sha256_file,
    write_binary_ply,
)
from scripts.source_metal_mtlmesh_simplify_structure_witness import (
    _validate_arrays,
)
from trellmlx.source_route_identity import (
    SourceRouteIdentityError,
    probe_cumesh_route_identity,
    validate_source_route_identity,
)


HEX_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
Runner = Callable[
    [np.ndarray, np.ndarray, dict[str, np.ndarray], int],
    tuple[np.ndarray, np.ndarray, dict[str, Any]],
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--cuda-report-json", type=Path, required=True)
    parser.add_argument("--cuda-npz", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-cuda-report-sha256", required=True)
    parser.add_argument("--expected-cuda-npz-sha256", required=True)
    parser.add_argument("--expected-source-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--target-faces", type=int, required=True)
    return parser


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def _as_int(value: Any) -> int:
    return int(value.item()) if hasattr(value, "item") else int(value)


def _default_runner(
    vertices: np.ndarray,
    faces: np.ndarray,
    cuda_arrays: dict[str, np.ndarray],
    target_faces: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import torch
    from cumesh import CuMesh

    mesh = CuMesh()
    mesh.init(torch.from_numpy(vertices), torch.from_numpy(faces))
    mesh.get_vertex_face_adjacency()
    mesh.get_edges()
    mesh.get_boundary_info()
    live_arrays = _validate_arrays(
        {
            name: np.ascontiguousarray(value.detach().cpu().numpy())
            for name, value in mesh.read_all_cache().items()
            if name in cuda_arrays
        },
        num_vertices=len(vertices),
        num_faces=len(faces),
    )

    non_ordering_exact = {
        name: bool(np.array_equal(cuda_arrays[name], live_arrays[name]))
        for name in cuda_arrays
        if name != "vert2face"
    }
    if not all(non_ordering_exact.values()):
        mismatches = sorted(
            name for name, exact in non_ordering_exact.items() if not exact
        )
        raise WitnessError(
            f"live Metal structure differs outside adjacency order: {mismatches}"
        )
    order = _adjacency_order(
        cuda_arrays["vert2face"],
        live_arrays["vert2face"],
        cuda_arrays["vert2face_offset"],
    )
    if not order["segment_multisets_exact"]:
        raise WitnessError(
            "live Metal adjacency membership differs from CUDA at vertex "
            f"{order['first_multiset_mismatch_vertex']}"
        )

    injected = torch.from_numpy(cuda_arrays["vert2face"])
    mesh.replace_vertex_face_adjacency(injected)
    injected_readback = np.ascontiguousarray(
        mesh.read_all_cache()["vert2face"].detach().cpu().numpy()
    )
    if not np.array_equal(injected_readback, cuda_arrays["vert2face"]):
        raise WitnessError("Metal adjacency injection readback differs from CUDA")

    threshold = 1e-8
    lambda_edge_length = 1e-2
    lambda_skinny = 1e-3
    step_trace: list[dict[str, Any]] = []
    first_step = True
    while mesh.num_faces > target_faces:
        before = int(mesh.num_faces)
        new_vertices, new_faces = mesh.simplify_step(
            lambda_edge_length,
            lambda_skinny,
            threshold,
            False,
            reuse_vertex_face_adjacency=first_step,
        )
        new_vertices = _as_int(new_vertices)
        new_faces = _as_int(new_faces)
        removed = before - new_faces
        step_trace.append(
            {
                "iteration": len(step_trace) + 1,
                "input_faces": before,
                "output_faces": new_faces,
                "output_vertices": new_vertices,
                "removed_faces": removed,
                "threshold": threshold,
                "reused_cuda_adjacency": first_step,
            }
        )
        first_step = False
        if new_faces <= target_faces:
            break
        if removed / max(before, 1) < 1e-2:
            threshold *= 10

    output_vertices, output_faces = mesh.read()
    return (
        np.ascontiguousarray(output_vertices.detach().cpu().numpy(), dtype=np.float32),
        np.ascontiguousarray(output_faces.detach().cpu().numpy(), dtype=np.int32),
        {
            "first_step_reused_cuda_adjacency": True,
            "live_structure_non_ordering_exact": non_ordering_exact,
            "pre_injection_order_delta": order,
            "injected_readback_exact": True,
            "step_trace": step_trace,
        },
    )


def run_witness(
    *,
    input_ply: Path,
    cuda_report_json: Path,
    cuda_npz: Path,
    output_ply: Path,
    report_json: Path,
    expected_input_sha256: str,
    expected_cuda_report_sha256: str,
    expected_cuda_npz_sha256: str,
    expected_source_root: Path,
    expected_source_commit: str,
    target_faces: int,
    identity_probe: Callable[[], dict[str, Any]] | None = None,
    runner: Runner | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    input_ply = Path(input_ply)
    cuda_report_json = Path(cuda_report_json)
    cuda_npz = Path(cuda_npz)
    output_ply = Path(output_ply)
    requested_report_json = Path(report_json)
    expected_source_root = Path(expected_source_root).resolve(strict=False)
    effective_report_json, report_rerouted = _effective_report_path(
        requested_report_json,
        protected_paths=[input_ply, cuda_report_json, cuda_npz, output_ply],
    )
    report: dict[str, Any] = {
        "schema": "trellis2mlx.source_metal_cuda_adjacency_simplify_witness.v1",
        "status": "failed",
        "failure_phase": None,
        "last_trustworthy_phase": "request_received",
        "primary_output_status": "not_started",
        "requested_report_json": str(requested_report_json),
        "effective_report_json": str(effective_report_json),
        "report_rerouted": report_rerouted,
        "requested_route": {
            "input_ply": str(input_ply),
            "cuda_report_json": str(cuda_report_json),
            "cuda_npz": str(cuda_npz),
            "output_ply": str(output_ply),
            "expected_input_sha256": expected_input_sha256,
            "expected_cuda_report_sha256": expected_cuda_report_sha256,
            "expected_cuda_npz_sha256": expected_cuda_npz_sha256,
            "expected_source_root": str(expected_source_root),
            "expected_source_commit": expected_source_commit,
            "target_faces": int(target_faces),
            "geometry_route": "metal-mtlmesh-cuda-adjacency-first-step",
        },
        "effective_route": None,
        "input_mesh": None,
        "cuda_structure": None,
        "injection": None,
        "output_mesh": None,
        "elapsed_seconds": None,
    }
    phase = "request_validation"

    try:
        protected = [input_ply, cuda_report_json, cuda_npz]
        if any(_same_path(output_ply, path) for path in protected):
            raise WitnessError("output PLY aliases a protected input")
        if output_ply.suffix != ".ply":
            raise WitnessError("--output-ply must end in .ply")
        for name, value in (
            ("--expected-input-sha256", expected_input_sha256),
            ("--expected-cuda-report-sha256", expected_cuda_report_sha256),
            ("--expected-cuda-npz-sha256", expected_cuda_npz_sha256),
        ):
            if not HEX_SHA256.fullmatch(value):
                raise WitnessError(f"{name} must be 64 lowercase hex characters")
        if not HEX_GIT_COMMIT.fullmatch(expected_source_commit):
            raise WitnessError(
                "--expected-source-commit must be 40 lowercase hex characters"
            )
        if target_faces <= 0:
            raise WitnessError("--target-faces must be positive")

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
        _write_report(effective_report_json, report)

        phase = "stale_output_cleanup"
        output_ply.parent.mkdir(parents=True, exist_ok=True)
        if output_ply.exists():
            if not output_ply.is_file():
                raise WitnessError(f"stale output is not a file: {output_ply}")
            output_ply.unlink()
        report["last_trustworthy_phase"] = "stale_output_removed"
        _write_report(effective_report_json, report)

        phase = "cuda_structure_validation"
        cuda_report, cuda_arrays = _load_witness(
            "cuda",
            cuda_report_json,
            cuda_npz,
            expected_cuda_report_sha256,
            expected_cuda_npz_sha256,
        )
        if cuda_report["input_mesh"]["sha256"] != actual_input_sha256:
            raise WitnessError(
                "CUDA structure input SHA256 does not match Metal input SHA256"
            )
        cuda_arrays = _validate_arrays(
            cuda_arrays,
            num_vertices=len(vertices),
            num_faces=len(faces),
        )
        report["cuda_structure"] = {
            "report_path": str(cuda_report_json),
            "report_sha256": expected_cuda_report_sha256,
            "npz_path": str(cuda_npz),
            "npz_sha256": expected_cuda_npz_sha256,
            "cuda_device_name": cuda_report["effective_route"].get(
                "cuda_device_name"
            ),
            "geometry_route": cuda_report["effective_route"]["geometry_route"],
            "comparison_edge_normalization": cuda_report.get(
                "_comparison_edge_normalization",
                "none",
            ),
        }
        report["last_trustworthy_phase"] = "cuda_structure_validated"
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
            "geometry_route": "metal-mtlmesh-cuda-adjacency-first-step",
            "cuda_geometry_route": cuda_report["effective_route"]["geometry_route"],
            "cuda_device_name": cuda_report["effective_route"].get(
                "cuda_device_name"
            ),
            "input_sha256": actual_input_sha256,
            "target_faces": int(target_faces),
        }
        report["status"] = "running"
        report["last_trustworthy_phase"] = "runtime_validated"
        _write_report(effective_report_json, report)

        phase = "injected_simplify"
        execute = runner or _default_runner
        output_vertices, output_faces, injection = execute(
            vertices,
            faces,
            cuda_arrays,
            int(target_faces),
        )
        output_vertices = np.ascontiguousarray(output_vertices, dtype=np.float32)
        output_faces = np.ascontiguousarray(output_faces, dtype=np.int32)
        if output_vertices.ndim != 2 or output_vertices.shape[1:] != (3,):
            raise WitnessError("runner output vertices must have shape [V, 3]")
        if output_faces.ndim != 2 or output_faces.shape[1:] != (3,):
            raise WitnessError("runner output faces must have shape [F, 3]")
        if len(output_faces) and (
            np.any(output_faces < 0) or np.any(output_faces >= len(output_vertices))
        ):
            raise WitnessError("runner output contains out-of-range face indices")
        if injection.get("first_step_reused_cuda_adjacency") is not True:
            raise WitnessError("runner did not attest first-step CUDA adjacency reuse")
        report["injection"] = injection
        report["last_trustworthy_phase"] = "injected_simplify_completed"
        report["primary_output_status"] = "partial"
        _write_report(effective_report_json, report)

        phase = "output_validation"
        write_binary_ply(output_ply, output_vertices, output_faces)
        reopened_vertices, reopened_faces = read_binary_ply(output_ply)
        if not np.array_equal(reopened_vertices, output_vertices):
            raise WitnessError("reopened output vertices differ from Metal readback")
        if not np.array_equal(reopened_faces, output_faces):
            raise WitnessError("reopened output faces differ from Metal readback")
        report["output_mesh"] = {
            "path": str(output_ply),
            "sha256": sha256_file(output_ply),
            "size_bytes": output_ply.stat().st_size,
            "vertices": int(len(output_vertices)),
            "faces": int(len(output_faces)),
        }
        report["status"] = "done"
        report["failure_phase"] = None
        report["last_trustworthy_phase"] = "output_validated"
        report["primary_output_status"] = "validated"
    except Exception as exc:
        report["status"] = "failed"
        report["failure_phase"] = phase
        report["error"] = f"{type(exc).__name__}: {exc}"
        if output_ply.exists():
            output_ply.unlink()
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
        cuda_report_json=args.cuda_report_json,
        cuda_npz=args.cuda_npz,
        output_ply=args.output_ply,
        report_json=args.report_json,
        expected_input_sha256=args.expected_input_sha256,
        expected_cuda_report_sha256=args.expected_cuda_report_sha256,
        expected_cuda_npz_sha256=args.expected_cuda_npz_sha256,
        expected_source_root=args.expected_source_root,
        expected_source_commit=args.expected_source_commit,
        target_faces=args.target_faces,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
