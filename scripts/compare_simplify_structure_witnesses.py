"""Compare authenticated CUDA and Metal QEM simplify structure witnesses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.source_cuda_cumesh_postprocess_witness import HEX_SHA256, sha256_file


EXPECTED_ROUTES = {
    "cuda": "release-cumesh-simplify-structure",
    "metal": "metal-mtlmesh-simplify-structure",
}
CANONICAL_CUDA_EDGE_READBACK = (
    "uint64-little-endian-words-canonicalized-to-min-max"
)


def _load_witness(
    role: str,
    report_json: Path,
    npz_path: Path,
    expected_report_sha256: str,
    expected_npz_sha256: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not HEX_SHA256.fullmatch(expected_report_sha256):
        raise ValueError(f"{role} expected report SHA256 is malformed")
    if not HEX_SHA256.fullmatch(expected_npz_sha256):
        raise ValueError(f"{role} expected NPZ SHA256 is malformed")
    report_json = Path(report_json)
    npz_path = Path(npz_path)
    actual_report_sha256 = sha256_file(report_json)
    actual_npz_sha256 = sha256_file(npz_path)
    if actual_report_sha256 != expected_report_sha256:
        raise ValueError(f"{role} report SHA256 mismatch")
    if actual_npz_sha256 != expected_npz_sha256:
        raise ValueError(f"{role} NPZ SHA256 mismatch")
    report = json.loads(report_json.read_text())
    if report.get("status") != "done":
        raise ValueError(f"{role} report status is not done")
    if report.get("primary_output_status") != "validated":
        raise ValueError(f"{role} primary output is not validated")
    if report.get("effective_route", {}).get("geometry_route") != EXPECTED_ROUTES[role]:
        raise ValueError(f"{role} effective geometry route mismatch")
    output_record = report.get("output_npz", {})
    recorded_output_path = Path(output_record.get("path", ""))
    output_path_matches = (
        recorded_output_path.resolve(strict=False) == npz_path.resolve(strict=False)
        if recorded_output_path.is_absolute()
        else recorded_output_path.name == npz_path.name
    )
    if not output_path_matches:
        raise ValueError(f"{role} report does not identify the supplied NPZ")
    if output_record.get("sha256") != actual_npz_sha256:
        raise ValueError(f"{role} report NPZ SHA256 mismatch")

    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {
            name: np.ascontiguousarray(archive[name])
            for name in archive.files
        }
    if set(arrays) != set(report.get("arrays", {})):
        raise ValueError(f"{role} report/NPZ array set mismatch")
    for name, array in arrays.items():
        record = report["arrays"][name]
        if record.get("shape") != list(array.shape):
            raise ValueError(f"{role} {name} shape mismatch")
        if record.get("dtype") != str(array.dtype):
            raise ValueError(f"{role} {name} dtype mismatch")
        if record.get("sha256") != hashlib.sha256(array.tobytes()).hexdigest():
            raise ValueError(f"{role} {name} content SHA256 mismatch")
    if (
        role == "cuda"
        and report["effective_route"].get("edge_readback")
        != CANONICAL_CUDA_EDGE_READBACK
    ):
        arrays["edges"] = np.ascontiguousarray(arrays["edges"][:, ::-1])
        report["_comparison_edge_normalization"] = (
            "raw-little-endian-words-to-min-max"
        )
    return report, arrays


def _adjacency_order(
    cuda: np.ndarray,
    metal: np.ndarray,
    offsets: np.ndarray,
) -> dict[str, Any]:
    differing_positions = np.flatnonzero(cuda != metal)
    if not len(differing_positions):
        return {
            "differing_vertices": 0,
            "segment_multisets_exact": True,
            "first_multiset_mismatch_vertex": None,
        }
    differing_vertices = np.unique(
        np.searchsorted(offsets[1:], differing_positions, side="right")
    )
    first_mismatch = None
    for vertex in differing_vertices:
        start = int(offsets[vertex])
        end = int(offsets[vertex + 1])
        if not np.array_equal(
            np.sort(cuda[start:end]),
            np.sort(metal[start:end]),
        ):
            first_mismatch = int(vertex)
            break
    return {
        "differing_vertices": int(len(differing_vertices)),
        "segment_multisets_exact": first_mismatch is None,
        "first_multiset_mismatch_vertex": first_mismatch,
    }


def compare_witnesses(
    *,
    cuda_report_json: Path,
    cuda_npz: Path,
    metal_report_json: Path,
    metal_npz: Path,
    expected_cuda_report_sha256: str,
    expected_cuda_npz_sha256: str,
    expected_metal_report_sha256: str,
    expected_metal_npz_sha256: str,
) -> dict[str, Any]:
    cuda_report, cuda_arrays = _load_witness(
        "cuda",
        cuda_report_json,
        cuda_npz,
        expected_cuda_report_sha256,
        expected_cuda_npz_sha256,
    )
    metal_report, metal_arrays = _load_witness(
        "metal",
        metal_report_json,
        metal_npz,
        expected_metal_report_sha256,
        expected_metal_npz_sha256,
    )
    if set(cuda_arrays) != set(metal_arrays):
        raise ValueError("CUDA and Metal array sets differ")

    array_results: dict[str, Any] = {}
    for name in cuda_arrays:
        cuda = cuda_arrays[name]
        metal = metal_arrays[name]
        same_shape = cuda.shape == metal.shape
        same_dtype = cuda.dtype == metal.dtype
        exact = bool(same_shape and same_dtype and np.array_equal(cuda, metal))
        record = {
            "cuda_shape": list(cuda.shape),
            "metal_shape": list(metal.shape),
            "cuda_dtype": str(cuda.dtype),
            "metal_dtype": str(metal.dtype),
            "exact": exact,
            "differing_entries": None,
            "first_differing_flat_index": None,
        }
        if same_shape and same_dtype:
            differing = np.flatnonzero(cuda.reshape(-1) != metal.reshape(-1))
            record["differing_entries"] = int(len(differing))
            record["first_differing_flat_index"] = (
                int(differing[0]) if len(differing) else None
            )
        array_results[name] = record

    offsets_exact = array_results["vert2face_offset"]["exact"]
    vert2face_same_shape = (
        cuda_arrays["vert2face"].shape == metal_arrays["vert2face"].shape
    )
    if offsets_exact and vert2face_same_shape:
        order_result = _adjacency_order(
            cuda_arrays["vert2face"],
            metal_arrays["vert2face"],
            cuda_arrays["vert2face_offset"],
        )
    else:
        order_result = {
            "differing_vertices": None,
            "segment_multisets_exact": False,
            "first_multiset_mismatch_vertex": None,
        }

    non_adjacency = set(cuda_arrays) - {"vert2face"}
    return {
        "schema": "trellis2mlx.simplify_structure_comparison.v1",
        "status": "done",
        "input_sha256_exact": (
            cuda_report["input_mesh"]["sha256"]
            == metal_report["input_mesh"]["sha256"]
        ),
        "all_non_adjacency_arrays_exact": all(
            array_results[name]["exact"] for name in non_adjacency
        ),
        "arrays": array_results,
        "vert2face_order": order_result,
        "cuda": {
            "report_path": str(cuda_report_json),
            "report_sha256": expected_cuda_report_sha256,
            "npz_path": str(cuda_npz),
            "npz_sha256": expected_cuda_npz_sha256,
            "comparison_edge_normalization": cuda_report.get(
                "_comparison_edge_normalization",
                "none",
            ),
        },
        "metal": {
            "report_path": str(metal_report_json),
            "report_sha256": expected_metal_report_sha256,
            "npz_path": str(metal_npz),
            "npz_sha256": expected_metal_npz_sha256,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-report-json", type=Path, required=True)
    parser.add_argument("--cuda-npz", type=Path, required=True)
    parser.add_argument("--metal-report-json", type=Path, required=True)
    parser.add_argument("--metal-npz", type=Path, required=True)
    parser.add_argument("--expected-cuda-report-sha256", required=True)
    parser.add_argument("--expected-cuda-npz-sha256", required=True)
    parser.add_argument("--expected-metal-report-sha256", required=True)
    parser.add_argument("--expected-metal-npz-sha256", required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    result = compare_witnesses(
        cuda_report_json=args.cuda_report_json,
        cuda_npz=args.cuda_npz,
        metal_report_json=args.metal_report_json,
        metal_npz=args.metal_npz,
        expected_cuda_report_sha256=args.expected_cuda_report_sha256,
        expected_cuda_npz_sha256=args.expected_cuda_npz_sha256,
        expected_metal_report_sha256=args.expected_metal_report_sha256,
        expected_metal_npz_sha256=args.expected_metal_npz_sha256,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
