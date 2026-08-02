"""Run full Metal geometry with canonical adjacency and Turing rsqrt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from scripts.source_metal_cuda_qem_cost_witness import (
    _load_turing_rsqrt_lut,
)
from scripts.source_metal_mtlmesh_postprocess_witness import (
    EXPECTED_SOURCE_COMMIT,
    run_witness as run_base_witness,
)
from trellmlx.canonical_cumesh import (
    mesh_state_digest_observer,
    simplify_with_canonical_adjacency_step_loop,
)
from trellmlx.source_mtlmesh import postprocess_source_native


ADJACENCY_ORDER = "ascending-face-id-per-vertex"
GEOMETRY_ROUTE = (
    "metal-mtlmesh-canonical-adjacency-turing-rsqrt-non-remesh"
)
MATH_PROFILE = (
    "metal3.1-safe-precise-fp32-contract-on-cuda-nan-canonical"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--target-faces", type=int, required=True)
    parser.add_argument("--expected-source-root", type=Path, required=True)
    parser.add_argument(
        "--expected-source-commit",
        default=EXPECTED_SOURCE_COMMIT,
    )
    parser.add_argument("--rsqrt-lut-npz", type=Path, required=True)
    parser.add_argument("--expected-rsqrt-lut-sha256", required=True)
    parser.add_argument("--expected-extension-sha256", required=True)
    parser.add_argument("--expected-metallib-sha256", required=True)
    parser.add_argument(
        "--record-simplify-step-digests",
        action="store_true",
    )
    return parser


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes()
    ).hexdigest()


def _build_postprocessor(
    *,
    rsqrt_lut_npz: Path,
    expected_rsqrt_lut_sha256: str,
    lut_loader: Callable[..., tuple[np.ndarray, dict[str, Any]]] = (
        _load_turing_rsqrt_lut
    ),
    source_postprocessor: Callable[..., tuple[
        np.ndarray,
        np.ndarray,
        list[dict],
    ]] = postprocess_source_native,
    tensor_factory: Callable[[np.ndarray], Any] | None = None,
    record_step_digests: bool = False,
) -> Callable[..., tuple[np.ndarray, np.ndarray, list[dict]]]:
    def processor(
        vertices: np.ndarray,
        faces: np.ndarray,
        target_faces: int,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        rsqrt_delta, lut_identity = lut_loader(
            Path(rsqrt_lut_npz),
            expected_sha256=expected_rsqrt_lut_sha256,
        )
        if tensor_factory is None:
            import torch

            rsqrt_lut = torch.from_numpy(rsqrt_delta)
        else:
            rsqrt_lut = tensor_factory(rsqrt_delta)

        def canonical_runner(mesh: Any, requested_target: int, **options):
            trace = simplify_with_canonical_adjacency_step_loop(
                mesh,
                int(requested_target),
                lambda_edge_length=float(options["lambda_edge_length"]),
                lambda_skinny=float(options["lambda_skinny"]),
                thresh=float(options["thresh"]),
                rsqrt_lut=rsqrt_lut,
                step_observer=(
                    mesh_state_digest_observer
                    if record_step_digests
                    else None
                ),
            )
            return {
                "simplifier_route": (
                    "canonical-adjacency-turing-rsqrt-step-loop"
                ),
                "adjacency_order": ADJACENCY_ORDER,
                "reuse_vertex_face_adjacency": True,
                "record_step_digests": record_step_digests,
                "rsqrt_lut_sha256": expected_rsqrt_lut_sha256,
                "simplifier_step_trace": trace,
            }

        final_vertices, final_faces, trace = source_postprocessor(
            vertices,
            faces,
            int(target_faces),
            **kwargs,
            simplification_runner=canonical_runner,
        )
        return final_vertices, final_faces, [
            {
                "operation": "canonical_route_identity",
                "geometry_route": GEOMETRY_ROUTE,
                "adjacency_order": ADJACENCY_ORDER,
                "math_profile": MATH_PROFILE,
                "rsqrt_lut": {
                    **lut_identity,
                    "normalized_delta_sha256": _array_sha256(rsqrt_delta),
                },
            },
            *trace,
        ]

    return processor


def run_witness(
    *,
    input_ply: Path,
    output_dir: Path,
    report_json: Path,
    expected_input_sha256: str,
    target_faces: int,
    expected_source_root: Path,
    expected_source_commit: str,
    rsqrt_lut_npz: Path,
    expected_rsqrt_lut_sha256: str,
    expected_extension_sha256: str,
    expected_metallib_sha256: str,
    identity_probe: Callable[[], dict[str, Any]] | None = None,
    lut_loader: Callable[..., tuple[np.ndarray, dict[str, Any]]] = (
        _load_turing_rsqrt_lut
    ),
    source_postprocessor: Callable[..., tuple[
        np.ndarray,
        np.ndarray,
        list[dict],
    ]] = postprocess_source_native,
    record_simplify_step_digests: bool = False,
) -> dict[str, Any]:
    processor = _build_postprocessor(
        rsqrt_lut_npz=rsqrt_lut_npz,
        expected_rsqrt_lut_sha256=expected_rsqrt_lut_sha256,
        lut_loader=lut_loader,
        source_postprocessor=source_postprocessor,
        record_step_digests=record_simplify_step_digests,
    )
    return run_base_witness(
        input_ply=input_ply,
        output_dir=output_dir,
        report_json=report_json,
        expected_input_sha256=expected_input_sha256,
        target_faces=int(target_faces),
        expected_source_root=expected_source_root,
        expected_source_commit=expected_source_commit,
        expected_extension_sha256=expected_extension_sha256,
        expected_metallib_sha256=expected_metallib_sha256,
        identity_probe=identity_probe,
        postprocessor=processor,
        requested_route_overrides={
            "geometry_route": GEOMETRY_ROUTE,
            "adjacency_order": ADJACENCY_ORDER,
            "reuse_vertex_face_adjacency": True,
            "rsqrt_lut_npz": str(rsqrt_lut_npz),
            "expected_rsqrt_lut_sha256": expected_rsqrt_lut_sha256,
            "metal_math_profile": MATH_PROFILE,
            "record_simplify_step_digests": (
                record_simplify_step_digests
            ),
        },
        effective_route_overrides={
            "geometry_route": GEOMETRY_ROUTE,
            "adjacency_order": ADJACENCY_ORDER,
            "reuse_vertex_face_adjacency": True,
            "rsqrt_lut_npz": str(rsqrt_lut_npz),
            "rsqrt_lut_sha256": expected_rsqrt_lut_sha256,
            "metal_math_profile": MATH_PROFILE,
            "record_simplify_step_digests": (
                record_simplify_step_digests
            ),
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_witness(
            input_ply=args.input_ply,
            output_dir=args.output_dir,
            report_json=args.report_json,
            expected_input_sha256=args.expected_input_sha256,
            target_faces=args.target_faces,
            expected_source_root=args.expected_source_root,
            expected_source_commit=args.expected_source_commit,
            rsqrt_lut_npz=args.rsqrt_lut_npz,
            expected_rsqrt_lut_sha256=args.expected_rsqrt_lut_sha256,
            expected_extension_sha256=args.expected_extension_sha256,
            expected_metallib_sha256=args.expected_metallib_sha256,
            record_simplify_step_digests=(
                args.record_simplify_step_digests
            ),
        )
    except Exception:
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
