"""Compare one Metal CuMesh simplify step against a source-CUDA oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Callable

import numpy as np

from scripts.compare_simplify_structure_witnesses import _adjacency_order
from scripts.source_cuda_cumesh_first_simplify_step_witness import (
    ARRAY_DTYPES,
    LAMBDA_EDGE_LENGTH,
    LAMBDA_SKINNY,
    THRESHOLD,
    _validate_arrays,
)
from scripts.source_cuda_cumesh_postprocess_witness import (
    HEX_SHA256,
    WitnessError,
    _effective_report_path,
    _same_path,
    read_binary_ply,
    sha256_file,
)
from scripts.source_metal_cuda_qem_cost_witness import (
    _load_turing_rsqrt_lut,
)
from trellmlx.source_route_identity import (
    SourceRouteIdentityError,
    probe_cumesh_route_identity,
    validate_source_route_identity,
)


HEX_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
Runner = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, dict[str, Any]],
]
LutLoader = Callable[[Path, str], tuple[np.ndarray, dict[str, Any]]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--cuda-report-json", type=Path, required=True)
    parser.add_argument("--cuda-npz", type=Path, required=True)
    parser.add_argument("--turing-rsqrt-npz", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-cuda-report-sha256", required=True)
    parser.add_argument("--expected-cuda-npz-sha256", required=True)
    parser.add_argument("--expected-turing-rsqrt-sha256", required=True)
    parser.add_argument("--expected-source-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-extension-sha256", required=True)
    parser.add_argument("--expected-metallib-sha256", required=True)
    return parser


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _load_cuda_oracle(
    *,
    report_path: Path,
    npz_path: Path,
    expected_report_sha256: str,
    expected_npz_sha256: str,
    expected_input_sha256: str,
    num_faces: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if sha256_file(report_path) != expected_report_sha256:
        raise WitnessError("CUDA first-step report SHA256 mismatch")
    if sha256_file(npz_path) != expected_npz_sha256:
        raise WitnessError("CUDA first-step NPZ SHA256 mismatch")
    report = json.loads(report_path.read_text())
    if (
        report.get("schema")
        != "trellis2mlx.source_cuda_cumesh_first_simplify_step.v1"
        or report.get("status") != "done"
        or report.get("primary_output_status") != "validated"
    ):
        raise WitnessError("CUDA first-step report is not a validated oracle")
    if report.get("input_mesh", {}).get("sha256") != expected_input_sha256:
        raise WitnessError("CUDA first-step oracle input SHA256 mismatch")
    route = report.get("effective_route") or {}
    if (
        route.get("cuda_device_name") != "Tesla T4"
        or route.get("geometry_route")
        != "release-cumesh-first-simplify-step"
    ):
        raise WitnessError("CUDA first-step oracle route is not authenticated T4")
    with np.load(npz_path, allow_pickle=False) as archive:
        if set(archive.files) != set(ARRAY_DTYPES):
            raise WitnessError("CUDA first-step NPZ array set mismatch")
        arrays = {
            name: np.ascontiguousarray(archive[name])
            for name in archive.files
        }
    arrays = _validate_arrays(arrays, num_faces=num_faces)
    records = report.get("arrays") or {}
    for name, array in arrays.items():
        record = records.get(name) or {}
        if (
            record.get("shape") != list(array.shape)
            or record.get("dtype") != str(array.dtype)
            or record.get("sha256") != _array_sha256(array)
        ):
            raise WitnessError(f"CUDA first-step {name} metadata mismatch")
    if report.get("output_npz", {}).get("sha256") != expected_npz_sha256:
        raise WitnessError("CUDA first-step report does not bind the NPZ digest")
    return report, arrays


def _enrich_route_identity(identity: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(identity)
    if not enriched.get("extension_path"):
        import cumesh._C as extension

        enriched["extension_path"] = str(Path(extension.__file__).resolve())
    if not enriched.get("metallib_path") and enriched.get("git_root"):
        enriched["metallib_path"] = str(
            Path(enriched["git_root"]) / "cumesh" / "cumesh.metallib"
        )
    for key in ("extension", "metallib"):
        path = Path(enriched[f"{key}_path"])
        if not path.is_file():
            raise WitnessError(f"effective {key} artifact is missing: {path}")
        enriched[f"{key}_sha256"] = sha256_file(path)
    return enriched


def _validate_route(
    identity: dict[str, Any],
    *,
    expected_source_root: Path,
    expected_source_commit: str,
    expected_extension_sha256: str,
    expected_metallib_sha256: str,
) -> dict[str, Any]:
    try:
        identity = validate_source_route_identity(
            identity,
            expected_root=expected_source_root,
        )
    except SourceRouteIdentityError as exc:
        raise WitnessError(str(exc)) from exc
    if Path(identity.get("git_root", "")).resolve(strict=False) != (
        expected_source_root.resolve(strict=False)
    ):
        raise WitnessError(
            "effective mtlmesh git root does not match expected source root"
        )
    if identity.get("git_commit") != expected_source_commit:
        raise WitnessError("effective mtlmesh source commit mismatch")
    if identity.get("has_MtlMesh") is not True:
        raise WitnessError("effective route does not expose MtlMesh")
    identity = _enrich_route_identity(identity)
    for key, expected in (
        ("extension_sha256", expected_extension_sha256),
        ("metallib_sha256", expected_metallib_sha256),
    ):
        if identity[key] != expected:
            raise WitnessError(f"effective {key} mismatch")
    return identity


def _default_runner(
    vertices: np.ndarray,
    faces: np.ndarray,
    cuda_adjacency: np.ndarray,
    rsqrt_delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import torch
    from cumesh import CuMesh

    mesh = CuMesh()
    mesh.init(torch.from_numpy(vertices), torch.from_numpy(faces))
    mesh.get_vertex_face_adjacency()
    cache = mesh.read_all_cache()
    metal_adjacency = np.ascontiguousarray(
        cache["vert2face"].detach().cpu().numpy(),
        dtype=np.int32,
    )
    offsets = np.ascontiguousarray(
        cache["vert2face_offset"].detach().cpu().numpy(),
        dtype=np.int32,
    )
    order = _adjacency_order(cuda_adjacency, metal_adjacency, offsets)
    if not order["segment_multisets_exact"]:
        raise WitnessError(
            "Metal adjacency membership differs from CUDA at vertex "
            f"{order['first_multiset_mismatch_vertex']}"
        )
    mesh.replace_vertex_face_adjacency(torch.from_numpy(cuda_adjacency))
    readback = np.ascontiguousarray(
        mesh.read_all_cache()["vert2face"].detach().cpu().numpy(),
        dtype=np.int32,
    )
    if not np.array_equal(readback, cuda_adjacency):
        raise WitnessError("Metal adjacency injection readback differs from CUDA")
    new_vertices, new_faces = mesh.simplify_step_turing(
        torch.from_numpy(rsqrt_delta),
        LAMBDA_EDGE_LENGTH,
        LAMBDA_SKINNY,
        THRESHOLD,
        False,
        reuse_vertex_face_adjacency=True,
    )
    output_vertices, output_faces = mesh.read()
    return (
        np.ascontiguousarray(
            output_vertices.detach().cpu().numpy(),
            dtype=np.float32,
        ),
        np.ascontiguousarray(
            output_faces.detach().cpu().numpy(),
            dtype=np.int32,
        ),
        {
            "cuda_adjacency_segment_multisets_exact": True,
            "cuda_adjacency_readback_exact": True,
            "pre_injection_order_delta": order,
            "simplify_route": "turing-rsqrt-lut",
            "returned_vertices": int(new_vertices),
            "returned_faces": int(new_faces),
        },
    )


def _array_comparison(
    expected: np.ndarray,
    actual: np.ndarray,
) -> dict[str, Any]:
    same_shape = expected.shape == actual.shape
    exact = bool(same_shape and np.array_equal(expected, actual))
    result: dict[str, Any] = {
        "expected_shape": list(expected.shape),
        "actual_shape": list(actual.shape),
        "same_shape": same_shape,
        "bit_exact": exact,
        "expected_sha256": _array_sha256(expected),
        "actual_sha256": _array_sha256(actual),
        "mismatch_count": None,
        "first_mismatch_index": None,
        "max_abs": None,
    }
    if same_shape:
        mismatch = expected != actual
        result["mismatch_count"] = int(np.count_nonzero(mismatch))
        if np.any(mismatch):
            result["first_mismatch_index"] = [
                int(index) for index in np.argwhere(mismatch)[0]
            ]
        if np.issubdtype(expected.dtype, np.floating):
            result["max_abs"] = float(
                np.max(np.abs(expected.astype(np.float64) - actual))
            )
    return result


def run_comparison(
    *,
    input_ply: Path,
    cuda_report_json: Path,
    cuda_npz: Path,
    turing_rsqrt_npz: Path,
    output_npz: Path,
    report_json: Path,
    expected_input_sha256: str,
    expected_cuda_report_sha256: str,
    expected_cuda_npz_sha256: str,
    expected_turing_rsqrt_sha256: str,
    expected_source_root: Path,
    expected_source_commit: str,
    expected_extension_sha256: str,
    expected_metallib_sha256: str,
    identity_probe: Callable[[], dict[str, Any]] | None = None,
    lut_loader: LutLoader | None = None,
    runner: Runner | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    input_ply = Path(input_ply)
    cuda_report_json = Path(cuda_report_json)
    cuda_npz = Path(cuda_npz)
    turing_rsqrt_npz = Path(turing_rsqrt_npz)
    output_npz = Path(output_npz)
    requested_report_json = Path(report_json)
    expected_source_root = Path(expected_source_root)
    protected = [
        input_ply,
        cuda_report_json,
        cuda_npz,
        turing_rsqrt_npz,
        output_npz,
    ]
    effective_report_json, rerouted = _effective_report_path(
        requested_report_json,
        protected_paths=protected,
    )
    report: dict[str, Any] = {
        "schema": "trellis2mlx.cuda_metal_first_simplify_step_comparison.v1",
        "status": "failed",
        "failure_phase": None,
        "last_trustworthy_phase": "request_received",
        "primary_output_status": "not_started",
        "requested_report_json": str(requested_report_json),
        "effective_report_json": str(effective_report_json),
        "report_rerouted": rerouted,
        "requested_route": {
            "input_ply": str(input_ply),
            "cuda_report_json": str(cuda_report_json),
            "cuda_npz": str(cuda_npz),
            "turing_rsqrt_npz": str(turing_rsqrt_npz),
            "output_npz": str(output_npz),
            "expected_input_sha256": expected_input_sha256,
            "expected_cuda_report_sha256": expected_cuda_report_sha256,
            "expected_cuda_npz_sha256": expected_cuda_npz_sha256,
            "expected_turing_rsqrt_sha256": expected_turing_rsqrt_sha256,
            "expected_source_root": str(expected_source_root),
            "expected_source_commit": expected_source_commit,
            "expected_extension_sha256": expected_extension_sha256,
            "expected_metallib_sha256": expected_metallib_sha256,
            "simplify_route": "turing-rsqrt-lut",
        },
        "effective_route": None,
        "input_mesh": None,
        "cuda_oracle": None,
        "turing_rsqrt": None,
        "injection": None,
        "comparison": None,
        "output_npz": None,
        "elapsed_seconds": None,
    }
    phase = "request_validation"
    try:
        paths = [input_ply, cuda_report_json, cuda_npz, turing_rsqrt_npz]
        if any(_same_path(output_npz, path) for path in paths):
            raise WitnessError("output NPZ aliases a protected input")
        if output_npz.suffix != ".npz":
            raise WitnessError("--output-npz must end in .npz")
        for name, value in (
            ("input", expected_input_sha256),
            ("CUDA report", expected_cuda_report_sha256),
            ("CUDA NPZ", expected_cuda_npz_sha256),
            ("Turing rsqrt", expected_turing_rsqrt_sha256),
            ("extension", expected_extension_sha256),
            ("metallib", expected_metallib_sha256),
        ):
            if not HEX_SHA256.fullmatch(value):
                raise WitnessError(f"expected {name} SHA256 is malformed")
        if not HEX_GIT_COMMIT.fullmatch(expected_source_commit):
            raise WitnessError("expected source commit is malformed")

        phase = "input_validation"
        if sha256_file(input_ply) != expected_input_sha256:
            raise WitnessError("input PLY SHA256 mismatch")
        vertices, faces = read_binary_ply(input_ply)
        report["input_mesh"] = {
            "sha256": expected_input_sha256,
            "vertices": int(len(vertices)),
            "faces": int(len(faces)),
        }
        cuda_report, cuda_arrays = _load_cuda_oracle(
            report_path=cuda_report_json,
            npz_path=cuda_npz,
            expected_report_sha256=expected_cuda_report_sha256,
            expected_npz_sha256=expected_cuda_npz_sha256,
            expected_input_sha256=expected_input_sha256,
            num_faces=len(faces),
        )
        report["cuda_oracle"] = {
            "report_sha256": expected_cuda_report_sha256,
            "npz_sha256": expected_cuda_npz_sha256,
            "cuda_device_name": cuda_report["effective_route"][
                "cuda_device_name"
            ],
            "geometry_route": cuda_report["effective_route"][
                "geometry_route"
            ],
        }
        report["last_trustworthy_phase"] = "inputs_validated"
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
        identity = _validate_route(
            (identity_probe or probe_cumesh_route_identity)(),
            expected_source_root=expected_source_root,
            expected_source_commit=expected_source_commit,
            expected_extension_sha256=expected_extension_sha256,
            expected_metallib_sha256=expected_metallib_sha256,
        )
        load_lut = lut_loader or _load_turing_rsqrt_lut
        rsqrt_delta, lut_record = load_lut(
            turing_rsqrt_npz,
            expected_sha256=expected_turing_rsqrt_sha256,
        )
        report["turing_rsqrt"] = lut_record
        report["effective_route"] = {
            **identity,
            "source_commit": expected_source_commit,
            "simplify_route": "turing-rsqrt-lut",
            "input_sha256": expected_input_sha256,
        }
        report["status"] = "running"
        report["last_trustworthy_phase"] = "runtime_validated"
        _write_report(effective_report_json, report)

        phase = "metal_first_step"
        metal_vertices, metal_faces, injection = (runner or _default_runner)(
            vertices,
            faces,
            cuda_arrays["vert2face"],
            rsqrt_delta,
        )
        metal_vertices = np.ascontiguousarray(metal_vertices, dtype=np.float32)
        metal_faces = np.ascontiguousarray(metal_faces, dtype=np.int32)
        if (
            metal_vertices.ndim != 2
            or metal_vertices.shape[1:] != (3,)
            or metal_faces.ndim != 2
            or metal_faces.shape[1:] != (3,)
        ):
            raise WitnessError("Metal first-step arrays have invalid shapes")
        if (
            injection.get("cuda_adjacency_segment_multisets_exact") is not True
            or injection.get("cuda_adjacency_readback_exact") is not True
            or injection.get("simplify_route") != "turing-rsqrt-lut"
        ):
            raise WitnessError("Metal runner did not attest the required route")
        report["injection"] = injection
        comparisons = {
            "post_vertices": _array_comparison(
                cuda_arrays["post_vertices"], metal_vertices
            ),
            "post_faces": _array_comparison(
                cuda_arrays["post_faces"], metal_faces
            ),
        }
        comparisons["bit_exact"] = all(
            comparisons[name]["bit_exact"]
            for name in ("post_vertices", "post_faces")
        )
        report["comparison"] = comparisons
        report["last_trustworthy_phase"] = "metal_first_step_compared"
        report["primary_output_status"] = "partial"
        _write_report(effective_report_json, report)

        phase = "output_write"
        np.savez(
            output_npz,
            metal_post_vertices=metal_vertices,
            metal_post_faces=metal_faces,
        )
        phase = "output_validation"
        with np.load(output_npz, allow_pickle=False) as archive:
            if set(archive.files) != {
                "metal_post_vertices",
                "metal_post_faces",
            }:
                raise WitnessError("reopened Metal output array set mismatch")
            if not np.array_equal(
                archive["metal_post_vertices"], metal_vertices
            ) or not np.array_equal(archive["metal_post_faces"], metal_faces):
                raise WitnessError("reopened Metal output differs from readback")
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
    report = run_comparison(
        input_ply=args.input_ply,
        cuda_report_json=args.cuda_report_json,
        cuda_npz=args.cuda_npz,
        turing_rsqrt_npz=args.turing_rsqrt_npz,
        output_npz=args.output_npz,
        report_json=args.report_json,
        expected_input_sha256=args.expected_input_sha256,
        expected_cuda_report_sha256=args.expected_cuda_report_sha256,
        expected_cuda_npz_sha256=args.expected_cuda_npz_sha256,
        expected_turing_rsqrt_sha256=args.expected_turing_rsqrt_sha256,
        expected_source_root=args.expected_source_root,
        expected_source_commit=args.expected_source_commit,
        expected_extension_sha256=args.expected_extension_sha256,
        expected_metallib_sha256=args.expected_metallib_sha256,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
