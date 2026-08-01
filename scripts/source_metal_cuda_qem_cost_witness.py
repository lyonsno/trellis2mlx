"""Compare native Metal QEM costs against an authenticated source-CUDA trace."""

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


HEX_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
CUDA_GEOMETRY_ROUTE = "release-cumesh-qem-cost-trace-instrumented"
CUDA_INSTRUMENTATION_SCHEMA = (
    "trellis2mlx.cumesh_qem_cost_instrumentation.v1"
)
CUDA_ARRAY_DTYPES = {
    "vert2face": np.dtype(np.int32),
    "qems": np.dtype(np.float32),
    "edge_collapse_costs": np.dtype(np.float32),
}
METAL_ARRAY_DTYPES = {
    "qems": np.dtype(np.float32),
    "edge_collapse_costs": np.dtype(np.float32),
}
Runner = Callable[
    [np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray | None],
    tuple[dict[str, np.ndarray], dict[str, Any]],
]
NATIVE_METAL_ROUTE = "metal-mtlmesh-cuda-adjacency-qem-cost-trace"
TURING_RSQRT_ROUTE = (
    "metal-mtlmesh-cuda-adjacency-turing-rsqrt-qem-cost-trace"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--cuda-report-json", type=Path, required=True)
    parser.add_argument("--cuda-npz", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-cuda-report-sha256", required=True)
    parser.add_argument("--expected-cuda-npz-sha256", required=True)
    parser.add_argument("--expected-source-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--rsqrt-lut-npz", type=Path)
    parser.add_argument("--expected-rsqrt-lut-sha256")
    parser.add_argument("--expected-metallib-sha256")
    parser.add_argument("--metal-math-profile")
    return parser


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _index(shape: tuple[int, ...], flat_index: int | None) -> list[int] | None:
    if flat_index is None:
        return None
    return [int(value) for value in np.unravel_index(flat_index, shape)]


def _ordered_float32_bits(bits: np.ndarray) -> np.ndarray:
    bits64 = bits.astype(np.uint64)
    return np.where(
        bits64 & np.uint64(0x80000000),
        np.uint64(0xFFFFFFFF) - bits64,
        bits64 + np.uint64(0x80000000),
    )


def compare_float32_arrays(
    cuda: np.ndarray,
    metal: np.ndarray,
    *,
    chunk_size: int = 1_000_000,
) -> dict[str, Any]:
    cuda = np.asarray(cuda)
    metal = np.asarray(metal)
    if cuda.shape != metal.shape:
        raise WitnessError(
            f"comparison shape mismatch: CUDA {cuda.shape}, Metal {metal.shape}"
        )
    if cuda.dtype != np.float32 or metal.dtype != np.float32:
        raise WitnessError("comparison arrays must both be float32")
    if chunk_size <= 0:
        raise WitnessError("comparison chunk size must be positive")

    cuda_flat = cuda.reshape(-1)
    metal_flat = metal.reshape(-1)
    bit_mismatch_count = 0
    nonfinite_class_mismatch_count = 0
    finite_pair_count = 0
    finite_sign_mismatch_count = 0
    absolute_sum = 0.0
    max_abs = 0.0
    max_abs_index: int | None = None
    max_ulp = 0
    first_bit_mismatch_index: int | None = None
    ulp_counts = {
        "0": 0,
        "1": 0,
        "2": 0,
        "3-4": 0,
        "5-8": 0,
        "9-16": 0,
        "17-256": 0,
        ">256": 0,
    }

    for start in range(0, cuda_flat.size, chunk_size):
        end = min(start + chunk_size, cuda_flat.size)
        cuda_chunk = cuda_flat[start:end]
        metal_chunk = metal_flat[start:end]
        cuda_bits = cuda_chunk.view(np.uint32)
        metal_bits = metal_chunk.view(np.uint32)
        bit_mismatch = cuda_bits != metal_bits
        mismatch_count = int(np.count_nonzero(bit_mismatch))
        bit_mismatch_count += mismatch_count
        if first_bit_mismatch_index is None and mismatch_count:
            first_bit_mismatch_index = start + int(np.flatnonzero(bit_mismatch)[0])

        cuda_class = np.select(
            [np.isnan(cuda_chunk), np.isposinf(cuda_chunk), np.isneginf(cuda_chunk)],
            [3, 1, 2],
            default=0,
        )
        metal_class = np.select(
            [
                np.isnan(metal_chunk),
                np.isposinf(metal_chunk),
                np.isneginf(metal_chunk),
            ],
            [3, 1, 2],
            default=0,
        )
        nonfinite_class_mismatch_count += int(
            np.count_nonzero(cuda_class != metal_class)
        )
        finite = (cuda_class == 0) & (metal_class == 0)
        count = int(np.count_nonzero(finite))
        finite_pair_count += count
        if not count:
            continue

        cuda_finite = cuda_chunk[finite]
        metal_finite = metal_chunk[finite]
        finite_sign_mismatch_count += int(
            np.count_nonzero(np.signbit(cuda_finite) != np.signbit(metal_finite))
        )
        absolute = np.abs(cuda_finite - metal_finite)
        absolute_sum += float(np.sum(absolute, dtype=np.float64))
        local_max_position = int(np.argmax(absolute))
        local_max = float(absolute[local_max_position])
        if max_abs_index is None or local_max > max_abs:
            finite_positions = np.flatnonzero(finite)
            max_abs = local_max
            max_abs_index = start + int(finite_positions[local_max_position])

        cuda_ordered = _ordered_float32_bits(cuda_bits[finite])
        metal_ordered = _ordered_float32_bits(metal_bits[finite])
        ulps = np.maximum(cuda_ordered, metal_ordered) - np.minimum(
            cuda_ordered,
            metal_ordered,
        )
        max_ulp = max(max_ulp, int(ulps.max(initial=0)))
        ulp_counts["0"] += int(np.count_nonzero(ulps == 0))
        ulp_counts["1"] += int(np.count_nonzero(ulps == 1))
        ulp_counts["2"] += int(np.count_nonzero(ulps == 2))
        ulp_counts["3-4"] += int(np.count_nonzero((ulps >= 3) & (ulps <= 4)))
        ulp_counts["5-8"] += int(np.count_nonzero((ulps >= 5) & (ulps <= 8)))
        ulp_counts["9-16"] += int(
            np.count_nonzero((ulps >= 9) & (ulps <= 16))
        )
        ulp_counts["17-256"] += int(
            np.count_nonzero((ulps >= 17) & (ulps <= 256))
        )
        ulp_counts[">256"] += int(np.count_nonzero(ulps > 256))

    return {
        "shape": list(cuda.shape),
        "elements": int(cuda.size),
        "bit_exact": bit_mismatch_count == 0,
        "bit_mismatch_count": bit_mismatch_count,
        "bit_mismatch_fraction": (
            bit_mismatch_count / cuda.size if cuda.size else 0.0
        ),
        "first_bit_mismatch_index": _index(
            cuda.shape,
            first_bit_mismatch_index,
        ),
        "nonfinite_class_mismatch_count": nonfinite_class_mismatch_count,
        "finite_pair_count": finite_pair_count,
        "finite_sign_mismatch_count": finite_sign_mismatch_count,
        "mean_abs": (
            absolute_sum / finite_pair_count if finite_pair_count else None
        ),
        "max_abs": max_abs if finite_pair_count else None,
        "max_abs_index": _index(cuda.shape, max_abs_index),
        "max_ulp": max_ulp if finite_pair_count else None,
        "ulp_counts": ulp_counts,
    }


def _load_cuda_trace(
    *,
    report_json: Path,
    npz_path: Path,
    expected_report_sha256: str,
    expected_npz_sha256: str,
    expected_input_sha256: str,
    num_vertices: int,
    num_faces: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if sha256_file(report_json) != expected_report_sha256:
        raise WitnessError("CUDA QEM report SHA256 mismatch")
    if sha256_file(npz_path) != expected_npz_sha256:
        raise WitnessError("CUDA QEM NPZ SHA256 mismatch")
    report = json.loads(report_json.read_text())
    if report.get("status") != "done":
        raise WitnessError("CUDA QEM report status is not done")
    if report.get("primary_output_status") != "validated":
        raise WitnessError("CUDA QEM primary output is not validated")
    if report.get("input_mesh", {}).get("sha256") != expected_input_sha256:
        raise WitnessError("CUDA QEM input SHA256 mismatch")
    route = report.get("effective_route") or {}
    if route.get("geometry_route") != CUDA_GEOMETRY_ROUTE:
        raise WitnessError("CUDA QEM geometry route is not authenticated")
    if route.get("cuda_available") is not True:
        raise WitnessError("CUDA QEM route did not use CUDA")
    if route.get("cuda_device_name") != "Tesla T4":
        raise WitnessError("CUDA QEM route did not use the requested Tesla T4")
    instrumentation = route.get("cumesh_instrumentation") or {}
    if instrumentation.get("schema") != CUDA_INSTRUMENTATION_SCHEMA:
        raise WitnessError("CUDA QEM instrumentation schema mismatch")
    patch_sha256 = instrumentation.get("patch_sha256")
    if not isinstance(patch_sha256, str) or not HEX_SHA256.fullmatch(
        patch_sha256
    ):
        raise WitnessError("CUDA QEM instrumentation patch identity is missing")

    output_record = report.get("output_npz") or {}
    recorded_path = Path(output_record.get("path", ""))
    path_matches = (
        recorded_path.resolve(strict=False) == npz_path.resolve(strict=False)
        if recorded_path.is_absolute()
        else recorded_path.name == npz_path.name
    )
    if not path_matches or output_record.get("sha256") != expected_npz_sha256:
        raise WitnessError("CUDA QEM report does not authenticate the supplied NPZ")

    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {
            name: np.ascontiguousarray(archive[name])
            for name in archive.files
        }
    if set(arrays) != set(CUDA_ARRAY_DTYPES):
        raise WitnessError("CUDA QEM NPZ array set mismatch")
    if set(arrays) != set(report.get("arrays") or {}):
        raise WitnessError("CUDA QEM report/NPZ array set mismatch")
    for name, dtype in CUDA_ARRAY_DTYPES.items():
        array = arrays[name]
        record = report["arrays"][name]
        if array.dtype != dtype:
            raise WitnessError(f"CUDA QEM {name} dtype mismatch")
        if record.get("shape") != list(array.shape):
            raise WitnessError(f"CUDA QEM {name} shape mismatch")
        if record.get("dtype") != str(array.dtype):
            raise WitnessError(f"CUDA QEM {name} recorded dtype mismatch")
        if record.get("sha256") != _array_sha256(array):
            raise WitnessError(f"CUDA QEM {name} content SHA256 mismatch")
    if arrays["vert2face"].shape != (num_faces * 3,):
        raise WitnessError("CUDA QEM adjacency shape mismatch")
    if arrays["qems"].shape != (num_vertices, 10):
        raise WitnessError("CUDA QEM coefficient shape mismatch")
    if arrays["edge_collapse_costs"].ndim != 1:
        raise WitnessError("CUDA edge-collapse costs must be one-dimensional")
    return report, arrays


def _load_turing_rsqrt_lut(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise WitnessError("Turing rsqrt LUT NPZ SHA256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if "normalized_delta" not in archive.files:
            raise WitnessError("Turing rsqrt LUT NPZ lacks normalized_delta")
        delta = np.ascontiguousarray(archive["normalized_delta"])
    if delta.dtype != np.int8 or delta.shape != (1 << 24,):
        raise WitnessError(
            "Turing rsqrt normalized_delta must be int8 with 2^24 entries"
        )
    return delta, {
        "npz_path": str(path),
        "npz_sha256": actual_sha256,
        "normalized_delta": {
            "shape": list(delta.shape),
            "dtype": str(delta.dtype),
            "sha256": _array_sha256(delta),
        },
    }


def _default_runner(
    vertices: np.ndarray,
    faces: np.ndarray,
    cuda_arrays: dict[str, np.ndarray],
    rsqrt_delta: np.ndarray | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch
    import cumesh._C as cumesh_extension
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
    order = _adjacency_order(
        cuda_arrays["vert2face"],
        metal_adjacency,
        offsets,
    )
    if not order["segment_multisets_exact"]:
        raise WitnessError(
            "Metal adjacency membership differs from CUDA at vertex "
            f"{order['first_multiset_mismatch_vertex']}"
        )

    mesh.replace_vertex_face_adjacency(
        torch.from_numpy(cuda_arrays["vert2face"])
    )
    readback = np.ascontiguousarray(
        mesh.read_all_cache()["vert2face"].detach().cpu().numpy(),
        dtype=np.int32,
    )
    if not np.array_equal(readback, cuda_arrays["vert2face"]):
        raise WitnessError("Metal adjacency injection readback differs from CUDA")

    if rsqrt_delta is None:
        trace = mesh.read_qem_cost_trace(reuse_vertex_face_adjacency=True)
        qem_kernel = "metal-native-rsqrt"
    else:
        trace = mesh.read_qem_cost_trace_turing(
            torch.from_numpy(rsqrt_delta),
            reuse_vertex_face_adjacency=True,
        )
        qem_kernel = "turing-rsqrt-lut"
    arrays = {
        name: np.ascontiguousarray(trace[name].detach().cpu().numpy())
        for name in METAL_ARRAY_DTYPES
    }
    metallib_path = Path(cumesh_extension.__file__).with_name("cumesh.metallib")
    return arrays, {
        "injected_readback_exact": True,
        "segment_multisets_exact": True,
        "pre_injection_order_delta": order,
        "qem_kernel": qem_kernel,
        "metallib_path": str(metallib_path),
        "metallib_sha256": sha256_file(metallib_path),
    }


def run_witness(
    *,
    input_ply: Path,
    cuda_report_json: Path,
    cuda_npz: Path,
    output_npz: Path,
    report_json: Path,
    expected_input_sha256: str,
    expected_cuda_report_sha256: str,
    expected_cuda_npz_sha256: str,
    expected_source_root: Path,
    expected_source_commit: str,
    rsqrt_lut_npz: Path | None = None,
    expected_rsqrt_lut_sha256: str | None = None,
    expected_metallib_sha256: str | None = None,
    metal_math_profile: str | None = None,
    identity_probe: Callable[[], dict[str, Any]] | None = None,
    runner: Runner | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    input_ply = Path(input_ply)
    cuda_report_json = Path(cuda_report_json)
    cuda_npz = Path(cuda_npz)
    output_npz = Path(output_npz)
    requested_report_json = Path(report_json)
    expected_source_root = Path(expected_source_root).resolve(strict=False)
    rsqrt_lut_npz = Path(rsqrt_lut_npz) if rsqrt_lut_npz is not None else None
    turing_route_requested = (
        rsqrt_lut_npz is not None or expected_rsqrt_lut_sha256 is not None
    )
    geometry_route = (
        TURING_RSQRT_ROUTE if turing_route_requested else NATIVE_METAL_ROUTE
    )
    effective_report_json, report_rerouted = _effective_report_path(
        requested_report_json,
        protected_paths=[
            input_ply,
            cuda_report_json,
            cuda_npz,
            output_npz,
            *([rsqrt_lut_npz] if rsqrt_lut_npz is not None else []),
        ],
    )
    report: dict[str, Any] = {
        "schema": "trellis2mlx.source_metal_cuda_qem_cost_witness.v1",
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
            "output_npz": str(output_npz),
            "expected_input_sha256": expected_input_sha256,
            "expected_cuda_report_sha256": expected_cuda_report_sha256,
            "expected_cuda_npz_sha256": expected_cuda_npz_sha256,
            "expected_source_root": str(expected_source_root),
            "expected_source_commit": expected_source_commit,
            "geometry_route": geometry_route,
            "rsqrt_lut_npz": (
                str(rsqrt_lut_npz) if rsqrt_lut_npz is not None else None
            ),
            "expected_rsqrt_lut_sha256": expected_rsqrt_lut_sha256,
            "expected_metallib_sha256": expected_metallib_sha256,
            "metal_math_profile": metal_math_profile,
        },
        "effective_route": None,
        "input_mesh": None,
        "cuda_trace": None,
        "rsqrt_lut": None,
        "injection": None,
        "metal_arrays": None,
        "comparison": None,
        "output_npz": None,
        "elapsed_seconds": None,
    }
    phase = "request_validation"

    try:
        if (rsqrt_lut_npz is None) != (expected_rsqrt_lut_sha256 is None):
            raise WitnessError(
                "--rsqrt-lut-npz and --expected-rsqrt-lut-sha256 must both be supplied"
            )
        if (expected_metallib_sha256 is None) != (metal_math_profile is None):
            raise WitnessError(
                "--expected-metallib-sha256 and --metal-math-profile must both be supplied"
            )
        protected = [input_ply, cuda_report_json, cuda_npz]
        if rsqrt_lut_npz is not None:
            protected.append(rsqrt_lut_npz)
        if any(_same_path(output_npz, path) for path in protected):
            raise WitnessError("output NPZ aliases a protected input")
        if output_npz.suffix != ".npz":
            raise WitnessError("--output-npz must end in .npz")
        for name, value in (
            ("--expected-input-sha256", expected_input_sha256),
            ("--expected-cuda-report-sha256", expected_cuda_report_sha256),
            ("--expected-cuda-npz-sha256", expected_cuda_npz_sha256),
        ):
            if not HEX_SHA256.fullmatch(value):
                raise WitnessError(
                    f"{name} must be 64 lowercase hex characters"
                )
        if (
            expected_rsqrt_lut_sha256 is not None
            and not HEX_SHA256.fullmatch(expected_rsqrt_lut_sha256)
        ):
            raise WitnessError(
                "--expected-rsqrt-lut-sha256 must be 64 lowercase hex characters"
            )
        if (
            expected_metallib_sha256 is not None
            and not HEX_SHA256.fullmatch(expected_metallib_sha256)
        ):
            raise WitnessError(
                "--expected-metallib-sha256 must be 64 lowercase hex characters"
            )
        if metal_math_profile is not None and not metal_math_profile.strip():
            raise WitnessError("--metal-math-profile must not be empty")
        if not HEX_GIT_COMMIT.fullmatch(expected_source_commit):
            raise WitnessError(
                "--expected-source-commit must be 40 lowercase hex characters"
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
        _write_report(effective_report_json, report)

        phase = "stale_output_cleanup"
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        if output_npz.exists():
            if not output_npz.is_file():
                raise WitnessError(f"stale output is not a file: {output_npz}")
            output_npz.unlink()
        report["last_trustworthy_phase"] = "stale_output_removed"
        _write_report(effective_report_json, report)

        phase = "cuda_trace_validation"
        cuda_report, cuda_arrays = _load_cuda_trace(
            report_json=cuda_report_json,
            npz_path=cuda_npz,
            expected_report_sha256=expected_cuda_report_sha256,
            expected_npz_sha256=expected_cuda_npz_sha256,
            expected_input_sha256=actual_input_sha256,
            num_vertices=len(vertices),
            num_faces=len(faces),
        )
        cuda_route = cuda_report["effective_route"]
        report["cuda_trace"] = {
            "report_path": str(cuda_report_json),
            "report_sha256": expected_cuda_report_sha256,
            "npz_path": str(cuda_npz),
            "npz_sha256": expected_cuda_npz_sha256,
            "cuda_device_name": cuda_route["cuda_device_name"],
            "geometry_route": cuda_route["geometry_route"],
            "instrumentation": cuda_route["cumesh_instrumentation"],
        }
        report["last_trustworthy_phase"] = "cuda_trace_validated"
        _write_report(effective_report_json, report)

        rsqrt_delta = None
        if rsqrt_lut_npz is not None:
            phase = "rsqrt_lut_validation"
            rsqrt_delta, rsqrt_record = _load_turing_rsqrt_lut(
                rsqrt_lut_npz,
                expected_sha256=expected_rsqrt_lut_sha256,
            )
            report["rsqrt_lut"] = rsqrt_record
            report["last_trustworthy_phase"] = "rsqrt_lut_validated"
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
            "geometry_route": geometry_route,
            "qem_kernel": (
                "turing-rsqrt-lut"
                if rsqrt_delta is not None
                else "metal-native-rsqrt"
            ),
            "cuda_geometry_route": cuda_route["geometry_route"],
            "cuda_device_name": cuda_route["cuda_device_name"],
            "input_sha256": actual_input_sha256,
        }
        report["status"] = "running"
        report["last_trustworthy_phase"] = "runtime_validated"
        _write_report(effective_report_json, report)

        phase = "metal_trace_collection"
        execute = runner or _default_runner
        metal_arrays, injection = execute(
            vertices,
            faces,
            cuda_arrays,
            rsqrt_delta,
        )
        if injection.get("injected_readback_exact") is not True:
            raise WitnessError("runner did not attest exact adjacency readback")
        if injection.get("segment_multisets_exact") is not True:
            raise WitnessError("runner did not attest adjacency membership parity")
        expected_qem_kernel = report["effective_route"]["qem_kernel"]
        if injection.get("qem_kernel") != expected_qem_kernel:
            raise WitnessError(
                "runner QEM kernel route mismatch: "
                f"expected {expected_qem_kernel}, got {injection.get('qem_kernel')}"
            )
        if expected_metallib_sha256 is not None:
            if injection.get("metallib_sha256") != expected_metallib_sha256:
                raise WitnessError(
                    "runner metallib route mismatch: "
                    f"expected {expected_metallib_sha256}, "
                    f"got {injection.get('metallib_sha256')}"
                )
            metallib_path = injection.get("metallib_path")
            if not isinstance(metallib_path, str) or not metallib_path:
                raise WitnessError("runner did not attest the metallib path")
            report["effective_route"]["metallib_path"] = metallib_path
            report["effective_route"][
                "metallib_sha256"
            ] = expected_metallib_sha256
            report["effective_route"]["metal_math_profile"] = metal_math_profile
        normalized: dict[str, np.ndarray] = {}
        for name, dtype in METAL_ARRAY_DTYPES.items():
            array = np.asarray(metal_arrays.get(name))
            if array.dtype != dtype:
                raise WitnessError(f"Metal {name} dtype mismatch")
            normalized[name] = np.ascontiguousarray(array)
        if normalized["qems"].shape != cuda_arrays["qems"].shape:
            raise WitnessError("Metal QEM coefficient shape mismatch")
        if (
            normalized["edge_collapse_costs"].shape
            != cuda_arrays["edge_collapse_costs"].shape
        ):
            raise WitnessError("Metal edge-collapse cost shape mismatch")
        metal_arrays = normalized
        report["injection"] = injection
        report["metal_arrays"] = {
            name: {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": _array_sha256(array),
                "size_bytes": int(array.nbytes),
                "nonfinite": int(np.count_nonzero(~np.isfinite(array))),
            }
            for name, array in metal_arrays.items()
        }
        report["comparison"] = {
            name: compare_float32_arrays(cuda_arrays[name], metal_arrays[name])
            for name in METAL_ARRAY_DTYPES
        }
        report["last_trustworthy_phase"] = "metal_trace_compared"
        report["primary_output_status"] = "partial"
        _write_report(effective_report_json, report)

        phase = "output_write"
        output_arrays = {
            **metal_arrays,
            "injected_vert2face": cuda_arrays["vert2face"],
        }
        np.savez_compressed(output_npz, **output_arrays)

        phase = "output_validation"
        with np.load(output_npz, allow_pickle=False) as reopened:
            if set(reopened.files) != set(output_arrays):
                raise WitnessError("reopened Metal NPZ array set mismatch")
            for name, expected in output_arrays.items():
                actual = np.asarray(reopened[name])
                if actual.dtype != expected.dtype or actual.shape != expected.shape:
                    raise WitnessError(f"reopened Metal {name} metadata mismatch")
                if _array_sha256(actual) != _array_sha256(expected):
                    raise WitnessError(f"reopened Metal {name} differs from trace")
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
        cuda_report_json=args.cuda_report_json,
        cuda_npz=args.cuda_npz,
        output_npz=args.output_npz,
        report_json=args.report_json,
        expected_input_sha256=args.expected_input_sha256,
        expected_cuda_report_sha256=args.expected_cuda_report_sha256,
        expected_cuda_npz_sha256=args.expected_cuda_npz_sha256,
        expected_source_root=args.expected_source_root,
        expected_source_commit=args.expected_source_commit,
        rsqrt_lut_npz=args.rsqrt_lut_npz,
        expected_rsqrt_lut_sha256=args.expected_rsqrt_lut_sha256,
        expected_metallib_sha256=args.expected_metallib_sha256,
        metal_math_profile=args.metal_math_profile,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
