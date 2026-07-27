"""Generate a 3D mesh from a single image using trellis2mlx.

Two-pass pipeline matching the TRELLIS.2 reference:
  1. Image → DINOv3 features
  2. Sparse structure flow → occupancy grid → LR coordinates
  3. LR SLat flow → denormalize → decoder upsample → HR coordinates
  4. HR SLat flow → denormalize → full decode → mesh extraction

Usage:
    PYTHONPATH=. python generate.py --image photo.png --output mesh.glb
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import mlx.core as mx
import numpy as np

from trellmlx.checkpoint_yield import maybe_checkpoint_yield
from trellmlx.modules.attention import (
    DEFAULT_QK_NORM_BACKEND,
    SUPPORTED_QK_NORM_BACKENDS,
    get_qk_norm_backend,
)
from trellmlx.modules.rope import (
    CUDA_POLAR_TURING_T4_BACKEND,
    MLX_REAL_BACKEND,
    SUPPORTED_ROPE_BACKENDS,
    configure_rope_backend,
    get_rope_backend,
    get_turing_phase_lut_sha256,
)
from trellmlx.shape_flow_layernorm import (
    CUDA_WELFORD_TURING_T4_BACKEND,
    DEFAULT_BACKEND as DEFAULT_SHAPE_FLOW_LAYERNORM_BACKEND,
    SUPPORTED_BACKENDS as SHAPE_FLOW_LAYERNORM_BACKENDS,
    configure_shape_flow_layernorm_backend,
    get_shape_flow_layernorm_backend,
    get_shape_flow_turing_rsqrt_lut_sha256,
)
from trellmlx.source_cuda_gelu import (
    SOURCE_CUDA_BF16_GELU_TANH_BACKEND,
    SOURCE_CUDA_BF16_GELU_TANH_BITS_SHA256,
)

# SLat normalization from pipeline.json
SHAPE_SLAT_MEAN = np.array([
    0.781296, 0.018091, -0.495192, -0.558457, 1.06053, 0.093252,
    1.518149, -0.933218, -0.732996, 2.604095, -0.118341, -2.143904,
    0.495076, -2.179512, -2.130751, -0.996944, 0.261421, -2.217463,
    1.260067, -0.150213, 3.790713, 1.481266, -1.046058, -1.523667,
    -0.059621, 2.22078, 1.621212, 0.87723, 0.567247, -3.175944,
    -3.186688, 1.578665,
], dtype=np.float32)

SHAPE_SLAT_STD = np.array([
    5.972266, 4.706852, 5.44501, 5.209927, 5.32022, 4.547237,
    5.020802, 5.444004, 5.226681, 5.683095, 4.831436, 5.286469,
    5.652043, 5.367606, 5.525084, 4.730578, 4.805265, 5.124013,
    5.530808, 5.619001, 5.10393, 5.41767, 5.269677, 5.547194,
    5.634698, 5.235274, 6.110351, 5.511298, 6.237273, 4.879207,
    5.347008, 5.405691,
], dtype=np.float32)

TEX_SLAT_MEAN = np.array([
    3.501659, 2.212398, 2.226094, 0.251093, -0.026248, -0.687364,
    0.439898, -0.928075, 0.029398, -0.339596, -0.869527, 1.038479,
    -0.972385, 0.126042, -1.129303, 0.455149, -1.209521, 2.069067,
    0.544735, 2.569128, -0.323407, 2.293, -1.925608, -1.217717,
    1.213905, 0.971588, -0.023631, 0.10675, 2.021786, 0.250524,
    -0.662387, -0.768862,
], dtype=np.float32)

TEX_SLAT_STD = np.array([
    2.665652, 2.743913, 2.765121, 2.595319, 3.037293, 2.291316,
    2.144656, 2.911822, 2.969419, 2.501689, 2.154811, 3.163343,
    2.621215, 2.381943, 3.186697, 3.021588, 2.295916, 3.234985,
    3.233086, 2.26014, 2.874801, 2.810596, 3.29272, 2.674999,
    2.680878, 2.372054, 2.451546, 2.353556, 2.995195, 2.379849,
    2.786195, 2.77519,
], dtype=np.float32)


def _denormalize_slat(slat: mx.array, mean=SHAPE_SLAT_MEAN, std=SHAPE_SLAT_STD) -> mx.array:
    return slat * mx.array(std) + mx.array(mean)


def _normalize_slat(slat: mx.array, mean=SHAPE_SLAT_MEAN, std=SHAPE_SLAT_STD) -> mx.array:
    return (slat - mx.array(mean)) / mx.array(std)


def _parse_sparse_flow_trace_keys(value: str | None) -> list[str]:
    if not value:
        return []
    keys = [key.strip() for key in value.split(",")]
    if any(not key for key in keys):
        raise ValueError("--sparse-flow-trace-keys must be a comma-separated list of non-empty keys")
    return list(dict.fromkeys(keys))


def _filter_sparse_flow_trace_payload(payload: dict[str, np.ndarray], selected_keys: list[str]) -> dict[str, np.ndarray]:
    if not selected_keys:
        return payload
    missing = [key for key in selected_keys if key not in payload]
    if missing:
        available = ", ".join(sorted(payload))
        raise ValueError(
            "--sparse-flow-trace-keys selected missing key(s): "
            f"{missing}; available keys: {available}"
        )
    return {key: payload[key] for key in selected_keys}


def _parse_shape_flow_trace_keys(value: str | None) -> list[str]:
    if not value:
        return []
    keys = [key.strip() for key in value.split(",")]
    if any(not key for key in keys):
        raise ValueError("--shape-flow-trace-keys must be a comma-separated list of non-empty keys")
    return list(dict.fromkeys(keys))


def _shape_flow_attention_route_from_env() -> dict[str, str]:
    backend_requested = os.environ.get(
        "TRELLIS2MLX_ATTENTION_BACKEND",
        "fast",
    ).lower()
    if backend_requested in {"manual", "mlx-manual"}:
        backend_effective = "manual"
    elif backend_requested in {"fast", "mlx-fast"}:
        backend_effective = "fast"
    elif backend_requested == "source-cuda-self":
        backend_effective = "source-cuda-self-widths-1029-7697-fast-otherwise"
    else:
        raise ValueError(
            "TRELLIS2MLX_ATTENTION_BACKEND must be one of "
            "'fast', 'mlx-fast', 'manual', 'mlx-manual', or "
            "'source-cuda-self', "
            f"got {backend_requested!r}"
        )
    softmax_requested = os.environ.get(
        "TRELLIS2MLX_ATTENTION_SOFTMAX_BACKEND",
        "mlx-softmax",
    ).lower()
    if softmax_requested not in {"mlx-softmax", "source-cuda-turing"}:
        raise ValueError(
            "TRELLIS2MLX_ATTENTION_SOFTMAX_BACKEND must be one of "
            "'mlx-softmax' or 'source-cuda-turing', "
            f"got {softmax_requested!r}"
        )
    value_requested = os.environ.get(
        "TRELLIS2MLX_ATTENTION_VALUE_BACKEND",
        "mlx-matmul",
    ).lower()
    if value_requested not in {"mlx-matmul", "source-cuda-sequential"}:
        raise ValueError(
            "TRELLIS2MLX_ATTENTION_VALUE_BACKEND must be one of "
            "'mlx-matmul' or 'source-cuda-sequential', "
            f"got {value_requested!r}"
        )
    if backend_effective == "manual":
        softmax_effective = softmax_requested
        value_effective = value_requested
    elif backend_requested == "source-cuda-self":
        if (
            softmax_requested != "source-cuda-turing"
            or value_requested != "source-cuda-sequential"
        ):
            raise ValueError(
                "source-cuda-self requires source-cuda-turing softmax and "
                "source-cuda-sequential value projection"
            )
        softmax_effective = (
            "source-cuda-turing-widths-1029-7697-fast-otherwise"
        )
        value_effective = (
            "source-cuda-sequential-widths-1029-7697-fast-otherwise"
        )
    else:
        softmax_effective = "fused-fast-attention"
        value_effective = "fused-fast-attention"
    return {
        "shape_flow_attention_backend_requested": backend_requested,
        "shape_flow_attention_backend_effective": backend_effective,
        "shape_flow_attention_softmax_backend_requested": softmax_requested,
        "shape_flow_attention_softmax_backend_effective": softmax_effective,
        "shape_flow_attention_value_backend_requested": value_requested,
        "shape_flow_attention_value_backend_effective": value_effective,
        "shape_flow_gelu_backend_effective": (
            SOURCE_CUDA_BF16_GELU_TANH_BACKEND
        ),
        "shape_flow_gelu_table_bits_sha256_effective": (
            SOURCE_CUDA_BF16_GELU_TANH_BITS_SHA256
        ),
    }


def _configure_shape_flow_attention_route(
    args: argparse.Namespace,
) -> dict[str, str] | None:
    if args.stop_after_stage != "shape_flow_block_trace":
        return None
    if args.shape_flow_attention_backend:
        os.environ["TRELLIS2MLX_ATTENTION_BACKEND"] = (
            args.shape_flow_attention_backend
        )
    if args.shape_flow_attention_softmax_backend:
        os.environ["TRELLIS2MLX_ATTENTION_SOFTMAX_BACKEND"] = (
            args.shape_flow_attention_softmax_backend
        )
    if args.shape_flow_attention_value_backend:
        os.environ["TRELLIS2MLX_ATTENTION_VALUE_BACKEND"] = (
            args.shape_flow_attention_value_backend
        )
    return _shape_flow_attention_route_from_env()


def _filter_shape_flow_trace_payload(
    payload: dict[str, np.ndarray], selected_keys: list[str]
) -> dict[str, np.ndarray]:
    if not selected_keys:
        return payload
    missing = [key for key in selected_keys if key not in payload]
    if missing:
        available = ", ".join(sorted(payload))
        raise ValueError(
            "--shape-flow-trace-keys selected missing key(s): "
            f"{missing}; available keys: {available}"
        )
    return {key: payload[key] for key in selected_keys}


def _select_shape_flow_trace_payload(
    payload: dict[str, np.ndarray], requested_keys: list[str]
) -> tuple[dict[str, np.ndarray], list[str]]:
    effective_keys = requested_keys or list(payload)
    return _filter_shape_flow_trace_payload(payload, requested_keys), effective_keys


def _requantize_coords(hr_coords_np, lr_resolution, hr_resolution):
    """Requantize decoder output coords to the target resolution.

    Maps from the decoder's upsampled space back to hr_resolution,
    deduplicates, and returns unique coords.

    Args:
        hr_coords_np: [N, 4] int array (batch, z, y, x) at decoder output res
        lr_resolution: input coordinate resolution (e.g. 32)
        hr_resolution: target mesh resolution (e.g. 256)

    Returns:
        unique_coords: [M, 4] int array at hr_resolution
    """
    spatial = hr_coords_np[:, 1:4].astype(np.float64)
    grid_res = hr_resolution // 16
    spatial = np.clip(
        np.round((spatial + 0.5) / lr_resolution * (grid_res - 1)),
        0, grid_res - 1,
    ).astype(np.int32)
    result = hr_coords_np.copy()
    result[:, 1:4] = spatial
    unique_coords = np.unique(result, axis=0)
    return unique_coords


def _select_uv_method(method, vertices, faces):
    """Select UV unwrap function based on method and mesh geometry.

    Returns (unwrap_fn, method_name).
    """
    from trellmlx.texture_bake import uv_unwrap, uv_unwrap_cube, uv_unwrap_lscm

    if method == "cube":
        return uv_unwrap_cube, "cube"
    if method == "xatlas":
        return uv_unwrap, "xatlas"
    if method == "lscm":
        return uv_unwrap_lscm, "lscm"

    # Auto: xatlas with max_iterations=0 (fast, no pathological behavior).
    # LSCM available as explicit option for experimentation.
    return uv_unwrap, "xatlas"


def _cleanup_and_simplify_mesh(
    vertices,
    faces,
    *,
    target_faces,
    no_cleanup,
    keep_largest=False,
    simplify_first=False,
    reference_cleanup=False,
    qem_simplify=False,
    qem_backend="mlx",
    source_native_source_root=None,
    source_native_python=None,
    cleanup_mesh=None,
    fill_holes=None,
    simplify=None,
    source_native_simplify=None,
    source_native_cleanup=None,
    source_native_orient=None,
    source_native_postprocess=None,
    orient_faces_by_adjacency=None,
    operation_trace=None,
    log=print,
):
    """Run mesh cleanup and multi-pass simplification.

    With simplify_first=True, simplifies the raw mesh before cleanup.
    Much faster on large meshes (cleanup on 200K faces vs 6M faces).
    """
    def run_qem_simplify(vertices, faces, target_faces, *, verbose=True):
        source_native_kwargs = {"verbose": verbose}
        if source_native_source_root is not None:
            source_native_kwargs["expected_source_root"] = source_native_source_root
        if source_native_python is not None:
            source_native_kwargs["reference_python"] = source_native_python
        if qem_backend == "mlx":
            from trellmlx.simplify_qem_metal import simplify_qem
            return simplify_qem(vertices, faces, target_faces, verbose=verbose)
        if qem_backend == "source-native":
            if source_native_simplify is None:
                from trellmlx.source_mtlmesh import simplify_source_native
                return simplify_source_native(
                    vertices,
                    faces,
                    target_faces,
                    **source_native_kwargs,
                )
            return source_native_simplify(
                vertices,
                faces,
                target_faces,
                **source_native_kwargs,
            )
        raise ValueError(f"unknown qem_backend: {qem_backend}")

    def run_source_native_orient(vertices, faces, *, verbose=False):
        source_native_kwargs = {"verbose": verbose}
        if source_native_source_root is not None:
            source_native_kwargs["expected_source_root"] = source_native_source_root
        if source_native_python is not None:
            source_native_kwargs["reference_python"] = source_native_python
        if source_native_orient is None:
            from trellmlx.source_mtlmesh import orient_source_native
            return orient_source_native(vertices, faces, **source_native_kwargs)
        return source_native_orient(vertices, faces, **source_native_kwargs)

    def run_source_native_cleanup(vertices, faces, *, verbose=True):
        source_native_kwargs = {"verbose": verbose}
        if source_native_source_root is not None:
            source_native_kwargs["expected_source_root"] = source_native_source_root
        if source_native_python is not None:
            source_native_kwargs["reference_python"] = source_native_python
        if source_native_cleanup is None:
            from trellmlx.source_mtlmesh import cleanup_source_native
            return cleanup_source_native(vertices, faces, **source_native_kwargs)
        return source_native_cleanup(vertices, faces, **source_native_kwargs)

    def run_source_native_postprocess(vertices, faces, target_faces, *, verbose=True):
        source_native_kwargs = {"verbose": verbose}
        if source_native_source_root is not None:
            source_native_kwargs["expected_source_root"] = source_native_source_root
        if source_native_python is not None:
            source_native_kwargs["reference_python"] = source_native_python
        if source_native_postprocess is None:
            from trellmlx.source_mtlmesh import postprocess_source_native
            return postprocess_source_native(vertices, faces, target_faces, **source_native_kwargs)
        return source_native_postprocess(vertices, faces, target_faces, **source_native_kwargs)

    if qem_simplify and qem_backend not in {"mlx", "source-native"}:
        raise ValueError(f"unknown qem_backend: {qem_backend}")

    local_cleanup_override = cleanup_mesh is not None

    def final_cleanup(vertices, faces):
        if no_cleanup:
            return vertices, faces
        t0 = time.perf_counter()
        vertices, faces = cleanup_mesh(vertices, faces, keep_largest=keep_largest, verbose=False)
        log(f"  Cleanup final: {time.perf_counter()-t0:.1f}s", flush=True)
        return vertices, faces

    def reference_final_cleanup(vertices, faces):
        if no_cleanup:
            return vertices, faces
        use_source_native_cleanup = (
            qem_simplify
            and qem_backend == "source-native"
            and (source_native_cleanup is not None or not local_cleanup_override)
        )
        use_source_native_orientation = qem_simplify and qem_backend == "source-native"
        if use_source_native_orientation and (
            source_native_orient is not None or orient_faces_by_adjacency is None
        ):
            orient_reference_faces = run_source_native_orient
            orient_operation = "orient_faces_source_native"
        elif orient_faces_by_adjacency is None:
            from trellmlx.mesh_cleanup import orient_faces_by_adjacency as orient_reference_faces
            orient_operation = "orient_faces_by_adjacency"
        else:
            orient_reference_faces = orient_faces_by_adjacency
            orient_operation = "orient_faces_by_adjacency"

        t0 = time.perf_counter()
        cleanup_input_faces = len(faces)
        if use_source_native_cleanup:
            vertices, faces = run_source_native_cleanup(vertices, faces, verbose=False)
            cleanup_operation = "cleanup_final_source_native"
        else:
            vertices, faces = cleanup_mesh(
                vertices,
                faces,
                keep_largest=keep_largest,
                do_fix_normals=False,
                verbose=False,
            )
            cleanup_operation = "cleanup_final"
        if operation_trace is not None:
            operation_trace.append({
                "operation": cleanup_operation,
                "input_faces": cleanup_input_faces,
                "output_faces": len(faces),
                "do_fix_normals": False,
            })
        orient_input_faces = len(faces)
        vertices, faces = orient_reference_faces(vertices, faces, verbose=False)
        if operation_trace is not None:
            operation_trace.append({
                "operation": orient_operation,
                "input_faces": orient_input_faces,
                "output_faces": len(faces),
            })
        log(f"  Reference cleanup final cleanup/orient: {time.perf_counter()-t0:.1f}s", flush=True)
        return vertices, faces

    if not no_cleanup:
        if cleanup_mesh is None:
            from trellmlx.mesh_cleanup import cleanup_mesh

    if reference_cleanup and not no_cleanup and target_faces and len(faces) > target_faces:
        if qem_simplify and qem_backend != "source-native":
            raise ValueError("reference_cleanup cannot be combined with qem_simplify until the QEM parity gate passes")
        use_combined_source_native = (
            qem_simplify
            and qem_backend == "source-native"
            and (
                source_native_postprocess is not None
                or not any([
                    source_native_simplify is not None,
                    source_native_cleanup is not None,
                    source_native_orient is not None,
                    orient_faces_by_adjacency is not None,
                    local_cleanup_override,
                ])
            )
        )
        if use_combined_source_native:
            vertices, faces, source_trace = run_source_native_postprocess(
                vertices,
                faces,
                target_faces,
                verbose=True,
            )
            if operation_trace is not None:
                operation_trace.extend(source_trace)
            if keep_largest:
                from trellmlx.mesh_cleanup import keep_largest_component
                keep_largest_input_faces = len(faces)
                vertices, faces = keep_largest_component(vertices, faces, verbose=False)
                if operation_trace is not None:
                    operation_trace.append({
                        "operation": "keep_largest_component",
                        "input_faces": keep_largest_input_faces,
                        "output_faces": len(faces),
                    })
            log(f"  Source-native reference postprocess: {len(vertices):,}V {len(faces):,}F", flush=True)
            return vertices, faces

        if simplify is None:
            import fast_simplification
            simplify = fast_simplification.simplify
        use_source_native_cleanup = (
            qem_simplify
            and qem_backend == "source-native"
            and (source_native_cleanup is not None or not local_cleanup_override)
        )

        coarse_target = target_faces * 3
        if len(faces) > coarse_target:
            t0 = time.perf_counter()
            simplify_input_faces = len(faces)
            if qem_simplify:
                vertices, faces = run_qem_simplify(vertices, faces, coarse_target, verbose=True)
                simplify_operation = "simplify_coarse_source_native_qem"
            else:
                vertices, faces = simplify(vertices, faces, target_count=coarse_target)
                simplify_operation = "simplify_coarse"
            if operation_trace is not None:
                operation_trace.append({
                    "operation": simplify_operation,
                    "input_faces": simplify_input_faces,
                    "requested_target_faces": coarse_target,
                    "output_faces": len(faces),
                })
            log(
                f"  Reference cleanup coarse simplify: {len(vertices):,}V {len(faces):,}F "
                f"({time.perf_counter()-t0:.1f}s)",
                flush=True,
            )

        t0 = time.perf_counter()
        cleanup_input_faces = len(faces)
        if use_source_native_cleanup:
            vertices, faces = run_source_native_cleanup(vertices, faces, verbose=True)
            cleanup_operation = "cleanup_initial_source_native"
        else:
            vertices, faces = cleanup_mesh(
                vertices,
                faces,
                keep_largest=keep_largest,
                do_fix_normals=False,
                verbose=True,
            )
            cleanup_operation = "cleanup_initial"
        if operation_trace is not None:
            operation_trace.append({
                "operation": cleanup_operation,
                "input_faces": cleanup_input_faces,
                "output_faces": len(faces),
                "do_fix_normals": False,
            })
        log(
            f"  Reference cleanup pass 1: {len(vertices):,}V {len(faces):,}F "
            f"({time.perf_counter()-t0:.1f}s)",
            flush=True,
        )

        if len(faces) > target_faces:
            t0 = time.perf_counter()
            simplify_input_faces = len(faces)
            if qem_simplify:
                vertices, faces = run_qem_simplify(vertices, faces, target_faces, verbose=True)
                simplify_operation = "simplify_final_source_native_qem"
            else:
                vertices, faces = simplify(vertices, faces, target_count=target_faces)
                simplify_operation = "simplify_final"
            if operation_trace is not None:
                operation_trace.append({
                    "operation": simplify_operation,
                    "input_faces": simplify_input_faces,
                    "requested_target_faces": target_faces,
                    "output_faces": len(faces),
                })
            log(
                f"  Reference cleanup final simplify: {len(vertices):,}V {len(faces):,}F "
                f"({time.perf_counter()-t0:.1f}s)",
                flush=True,
            )

        return reference_final_cleanup(vertices, faces)

    # Simplify-first mode: reduce face count before expensive cleanup
    if simplify_first and target_faces and len(faces) > target_faces:
        if qem_simplify:
            # QEM simplify-first: use fast-simplification for coarse pass,
            # then QEM for final pass with topology guards
            if simplify is None:
                import fast_simplification
                simplify = fast_simplification.simplify
            coarse_target = target_faces * 3
            if len(faces) > coarse_target:
                t0 = time.perf_counter()
                for _ in range(3):
                    ratio = coarse_target / len(faces)
                    if ratio >= 1: break
                    vertices, faces = simplify(vertices, faces, target_reduction=1.0 - ratio)
                    if len(faces) <= coarse_target * 1.1: break
                log(f"  Pre-simplify (coarse): {len(vertices):,}V {len(faces):,}F "
                    f"({time.perf_counter()-t0:.1f}s)", flush=True)
            if not no_cleanup:
                t0 = time.perf_counter()
                vertices, faces = cleanup_mesh(vertices, faces, keep_largest=keep_largest, do_fix_normals=False)
                log(f"  Cleanup: {time.perf_counter()-t0:.1f}s", flush=True)
            t0 = time.perf_counter()
            vertices, faces = run_qem_simplify(vertices, faces, target_faces, verbose=True)
            log(f"  QEM final: {len(vertices):,}V {len(faces):,}F "
                f"({time.perf_counter()-t0:.1f}s)", flush=True)
            vertices, faces = final_cleanup(vertices, faces)
            return vertices, faces
        else:
            if simplify is None:
                import fast_simplification
                simplify = fast_simplification.simplify
            t0 = time.perf_counter()
            for _ in range(3):
                ratio = target_faces / len(faces)
                if ratio >= 1: break
                vertices, faces = simplify(vertices, faces, target_reduction=1.0 - ratio)
                if len(faces) <= target_faces * 1.1: break
            log(f"  Pre-simplify: {len(vertices):,}V {len(faces):,}F ({time.perf_counter()-t0:.1f}s)", flush=True)
            # Single cleanup pass on the small mesh
            if not no_cleanup:
                t0 = time.perf_counter()
                vertices, faces = cleanup_mesh(vertices, faces, keep_largest=keep_largest)
                log(f"  Cleanup: {time.perf_counter()-t0:.1f}s", flush=True)
            return vertices, faces

    if not no_cleanup:
        t0 = time.perf_counter()
        vertices, faces = cleanup_mesh(vertices, faces, keep_largest=keep_largest, do_fix_normals=False)
        log(f"  Cleanup pass 1: {time.perf_counter()-t0:.1f}s", flush=True)

    if not target_faces or len(faces) <= target_faces:
        vertices, faces = final_cleanup(vertices, faces)
        return vertices, faces

    if simplify is None:
        import fast_simplification
        simplify = fast_simplification.simplify

    # Pass 1: Aggressive simplify to 3x target, then cleanup
    t0 = time.perf_counter()
    coarse_target = target_faces * 3
    if len(faces) > coarse_target:
        ratio = coarse_target / len(faces)
        vertices, faces = simplify(
            vertices, faces, target_reduction=1.0 - ratio,
        )
        log(f"  Coarse simplify: {len(faces):,}F ({time.perf_counter()-t0:.1f}s)", flush=True)
        if not no_cleanup:
            vertices, faces = cleanup_mesh(vertices, faces, keep_largest=keep_largest, do_fix_normals=False, verbose=False)

    if len(faces) <= target_faces:
        vertices, faces = final_cleanup(vertices, faces)
        return vertices, faces

    # Pass 2: Final simplify to target, then cleanup
    t0 = time.perf_counter()
    if qem_simplify and len(faces) > target_faces:
        vertices, faces = run_qem_simplify(
            vertices, faces, target_faces, verbose=True)
        log(f"  QEM final: {len(vertices):,}V {len(faces):,}F "
            f"({time.perf_counter()-t0:.1f}s)", flush=True)
        vertices, faces = final_cleanup(vertices, faces)
    else:
        did_final_simplify = False
        for attempt in range(3):
            if len(faces) <= target_faces:
                break
            ratio = target_faces / len(faces)
            target_reduction = 1.0 - ratio
            if target_reduction <= 0:
                break
            vertices, faces = simplify(
                vertices, faces, target_reduction=target_reduction,
            )
            did_final_simplify = True
            if len(faces) <= target_faces * 1.1:
                break
        log(f"  Final simplify: {len(vertices):,}V {len(faces):,}F "
            f"({time.perf_counter()-t0:.1f}s)", flush=True)
        if did_final_simplify:
            vertices, faces = final_cleanup(vertices, faces)

    return vertices, faces


def _load_turing_rsqrt_lut(
    path: str | Path,
    expected_sha256: str,
) -> tuple[mx.array, str]:
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            "Turing rsqrt LUT SHA256 mismatch: "
            f"expected {expected_sha256}, got {digest}"
        )
    with np.load(path, allow_pickle=False) as loaded:
        if "normalized_delta" not in loaded.files:
            raise ValueError(
                "Turing rsqrt LUT NPZ omits normalized_delta"
            )
        correction = np.asarray(loaded["normalized_delta"])
    if correction.dtype != np.int8 or correction.shape != (1 << 24,):
        raise ValueError(
            "Turing rsqrt LUT normalized_delta must be int8[16777216], "
            f"got {correction.dtype}{correction.shape}"
        )
    return mx.array(correction), digest


def _load_turing_rope_phase_lut(
    path: str | Path,
    expected_sha256: str,
) -> tuple[mx.array, str]:
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            "Turing RoPE phase LUT SHA256 mismatch: "
            f"expected {expected_sha256}, got {digest}"
        )
    with np.load(path, allow_pickle=False) as loaded:
        if "phase_pairs" not in loaded.files:
            raise ValueError("Turing RoPE phase LUT NPZ omits phase_pairs")
        phase_pairs = np.asarray(loaded["phase_pairs"])
    if phase_pairs.dtype != np.float32 or phase_pairs.shape != (64, 21, 2):
        raise ValueError(
            "Turing RoPE phase LUT must be float32[64,21,2], "
            f"got {phase_pairs.dtype}{phase_pairs.shape}"
        )
    if not np.isfinite(phase_pairs).all():
        raise ValueError("Turing RoPE phase LUT contains non-finite values")
    return mx.array(phase_pairs), digest


def main():
    parser = argparse.ArgumentParser(description="Generate 3D mesh from image via MLX")
    parser.add_argument("--image", nargs="+", help="Input image(s) — multiple images enable multi-view conditioning")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="/tmp/trellis-mlx-mesh.glb")
    parser.add_argument("--resolution", type=int, default=1024,
                        help="Mesh resolution (default: 1024, matching reference cascade)")
    parser.add_argument("--max-tokens", type=int, default=49152,
                        help="Max tokens for HR SLat pass (reduces resolution if exceeded)")
    parser.add_argument("--target-faces", type=int, default=200_000,
                        help="Simplify mesh to this face count (0 to disable, default: 200K)")
    parser.add_argument("--steps", type=int, default=12,
                        help="Number of ODE sampler steps (default: 12, try 6 for 2x speed)")
    parser.add_argument("--no-cascade", action="store_true",
                        help="Skip two-pass cascade (LR→upsample→HR). Uses single 512 SLat pass. "
                             "Much faster (~4x) but lower mesh quality and more holes.")
    parser.add_argument("--compile", action="store_true",
                        help="Use mx.compile for flow model forward passes (experimental, not faster)")
    parser.add_argument("--quantize", type=int, default=0, choices=[0, 4, 8],
                        help="Quantize flow models to INT4 or INT8 (0=disabled, default: 0)")
    parser.add_argument("--no-rembg", action="store_true",
                        help="Skip background removal (rembg) preprocessing")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Skip mesh cleanup entirely (no dedup, no repair, no hole fill)")
    parser.add_argument("--keep-largest", action="store_true",
                        help="Keep only the largest connected component (removes floors, floaters, extra objects)")
    parser.add_argument("--simplify-first", action="store_true",
                        help="Simplify before cleanup (much faster on large meshes, skips multi-pass)")
    parser.add_argument("--reference-cleanup", action="store_true",
                        help="Use reference cleanup order: coarse simplify, cleanup, final simplify, "
                             "final cleanup, adjacency orientation")
    parser.add_argument("--texture-size", type=int, default=1024,
                        help="Texture map resolution (default: 1024, try 2048 or 4096 for higher quality)")
    parser.add_argument("--texture-backend", choices=["cpu", "gpu"], default="gpu",
                        help="Texture bake backend: gpu (MLX Metal, default) or cpu (numpy)")
    parser.add_argument("--uv-method", choices=["auto", "lscm", "xatlas", "cube"], default="auto",
                        help="UV unwrap method: auto (xatlas, default), lscm, xatlas, or cube")
    parser.add_argument("--qem-simplify", action="store_true",
                        help="Use QEM simplification with topology guards (Metal-accelerated, "
                             "prevents holes from simplification). Slower but preserves mesh quality.")
    parser.add_argument("--qem-backend", choices=["mlx", "source-native"], default="mlx",
                        help="QEM simplifier backend for --qem-simplify. 'mlx' is the local probe; "
                             "'source-native' calls the reference mtlmesh/cumesh backend when installed.")
    parser.add_argument("--source-native-source-root",
                        help="Expected mtlmesh/cumesh source root for --qem-backend source-native "
                             "(for example /Users/noahlyons/dev/trellis-mac/deps/mtlmesh).")
    parser.add_argument("--source-native-python",
                        help="Python executable that can import the expected mtlmesh/cumesh and torch "
                             "for --qem-backend source-native.")
    parser.add_argument("--save-checkpoints", metavar="DIR",
                        help="Save intermediate representations to DIR for replay")
    parser.add_argument(
        "--stop-after-stage",
        choices=[
            "conditioning",
            "sparse_coords",
            "sparse_flow_step",
            "sparse_flow_steps",
            "sparse_flow_block_trace",
            "sparse_internals",
            "shape_flow_step",
            "shape_flow_steps",
            "shape_flow_block_trace",
            "shape_slat",
            "decoder_output",
            "mesh_raw",
            "mesh_clean",
            "mesh_uv",
        ],
        default=None,
        help="Stop after writing the named checkpoint stage. Requires --save-checkpoints.",
    )
    parser.add_argument("--shared-noise", metavar="NPZ",
                        help="Diagnostic: load sparse-structure noise from an NPZ containing ss_noise.")
    parser.add_argument("--shared-noise-sparse-only", action="store_true",
                        help="Diagnostic: with --shared-noise, use ss_noise for sparse structure but ignore "
                             "slat_noise_pool so shape SLat keeps the route's normal random noise.")
    parser.add_argument("--conditioning-sample", metavar="NPZ",
                        help="Diagnostic: load image conditioning from an NPZ containing cond and neg_cond.")
    parser.add_argument("--shape-slat-sample", metavar="NPZ",
                        help="Diagnostic: load shape_slat NPZ containing feats and coords, then run "
                             "only the shape decoder to decoder_output.")
    parser.add_argument("--shape-slat-support-sample", metavar="NPZ",
                        help="Diagnostic: load shape_slat NPZ coords as no-cascade shape-flow support "
                             "while still sampling MLX shape features.")
    parser.add_argument("--sparse-flow-trace-block-index", type=int, default=0,
                        help="Diagnostic: sparse flow block index to trace with --stop-after-stage "
                             "sparse_flow_block_trace (default: 0).")
    parser.add_argument("--sparse-flow-trace-step-index", type=int, default=0,
                        help="Diagnostic: sparse flow sampler step index to trace with --stop-after-stage "
                             "sparse_flow_block_trace (default: 0).")
    parser.add_argument("--sparse-flow-trace-sample", metavar="NPZ",
                        help="Diagnostic: load sparse-flow trace sample from an NPZ containing sample_in. "
                             "If sample_in is stacked by step, --sparse-flow-trace-step-index selects the row.")
    parser.add_argument("--sparse-flow-start-sample", metavar="NPZ",
                        help="Diagnostic: continue sparse-flow sampling from an NPZ containing sample_in. "
                             "If sample_in is stacked by step, --sparse-flow-start-step-index selects the row.")
    parser.add_argument("--sparse-flow-start-step-index", type=int, default=0,
                        help="Diagnostic: sampler step index corresponding to --sparse-flow-start-sample "
                             "(default: 0).")
    parser.add_argument("--sparse-flow-trace-block-input-sample", metavar="NPZ",
                        help="Diagnostic: replay projected block inputs from a block trace NPZ containing "
                             "pos_blockN_input and neg_blockN_input arrays.")
    parser.add_argument("--sparse-flow-trace-no-kv-cache", action="store_true",
                        help="Diagnostic: disable sparse-flow block-trace cross-attention KV cache "
                             "to match direct reference trace hooks.")
    parser.add_argument("--sparse-flow-trace-keys",
                        help="Diagnostic: comma-separated final sparse-flow block-trace payload keys to save. "
                             "Omit to save the full block trace.")
    parser.add_argument("--sparse-flow-block-injection-trace", metavar="NPZ",
                        help="Diagnostic: inject named sparse-flow block tensors from an NPZ trace.")
    parser.add_argument("--sparse-flow-block-injection-manifest", metavar="JSON",
                        help="Diagnostic: inject multiple named sparse-flow block tensors from a JSON manifest.")
    parser.add_argument("--sparse-flow-block-injection-step-index", type=int, default=2,
                        help="Diagnostic: sampler step index where --sparse-flow-block-injection-trace applies.")
    parser.add_argument("--sparse-flow-block-injection-block-index", type=int, default=0,
                        help="Diagnostic: sparse-flow block index where block injection applies.")
    parser.add_argument("--sparse-flow-block-injection-branch",
                        choices=["pos", "neg", "both"], default="both",
                        help="Diagnostic: CFG branch where block injection applies.")
    parser.add_argument("--sparse-flow-block-injection-stage",
                        choices=["norm1", "modulated_self_input", "after_self", "after_cross", "after_mlp"],
                        default="modulated_self_input",
                        help="Diagnostic: block tensor stage to replace.")
    parser.add_argument("--sparse-flow-block-injection-array-key",
                        help="Diagnostic: explicit trace array key for block injection.")
    parser.add_argument("--sparse-flow-layernorm-correction-report", metavar="JSON",
                        help="Diagnostic: apply row-wise no-affine LayerNorm correction from a "
                             "boundary-probe report.")
    parser.add_argument("--sparse-flow-layernorm-correction-step-index", type=int, default=2,
                        help="Diagnostic: sampler step index where row-wise LayerNorm correction applies.")
    parser.add_argument("--sparse-flow-layernorm-correction-block-index", type=int, default=0,
                        help="Diagnostic: sparse-flow block index where row-wise LayerNorm correction applies.")
    parser.add_argument("--sparse-flow-layernorm-correction-branch",
                        choices=["pos", "neg", "both"], default="pos",
                        help="Diagnostic: CFG branch where row-wise LayerNorm correction applies.")
    parser.add_argument("--sparse-flow-layernorm-correction-mode",
                        choices=["scale", "bias"], default="scale",
                        help="Diagnostic: row-wise LayerNorm perturbation mode from the boundary report.")
    parser.add_argument("--sparse-flow-layernorm-correction-include",
                        choices=["improved", "solved", "all"], default="improved",
                        help="Diagnostic: boundary-report rows to apply (default: improved).")
    parser.add_argument("--shape-flow-trace-block-index", type=int, default=0,
                        help="Diagnostic: shape SLat flow block index to trace with "
                             "--stop-after-stage shape_flow_block_trace (default: 0).")
    parser.add_argument("--shape-flow-trace-step-index", type=int, default=0,
                        help="Diagnostic: shape SLat flow sampler step index to trace with "
                             "--stop-after-stage shape_flow_block_trace (default: 0).")
    parser.add_argument("--shape-flow-trace-keys",
                        help="Diagnostic: comma-separated final shape-flow block-trace payload keys "
                             "to save. Omit to save the full block trace.")
    parser.add_argument(
        "--shape-flow-attention-backend",
        choices=[
            "fast",
            "mlx-fast",
            "manual",
            "mlx-manual",
            "source-cuda-self",
        ],
        help=(
            "Diagnostic: shape-flow-only attention backend for "
            "--stop-after-stage shape_flow_block_trace."
        ),
    )
    parser.add_argument(
        "--shape-flow-attention-softmax-backend",
        choices=["mlx-softmax", "source-cuda-turing"],
        help=(
            "Diagnostic: shape-flow-only manual softmax backend for "
            "--stop-after-stage shape_flow_block_trace."
        ),
    )
    parser.add_argument(
        "--shape-flow-attention-value-backend",
        choices=["mlx-matmul", "source-cuda-sequential"],
        help=(
            "Diagnostic: shape-flow-only manual value projection backend for "
            "--stop-after-stage shape_flow_block_trace."
        ),
    )
    parser.add_argument(
        "--shape-flow-layernorm-backend",
        choices=SHAPE_FLOW_LAYERNORM_BACKENDS,
        default=DEFAULT_SHAPE_FLOW_LAYERNORM_BACKEND,
        help="Shape SLat no-affine LayerNorm backend.",
    )
    parser.add_argument(
        "--qk-norm-backend",
        choices=SUPPORTED_QK_NORM_BACKENDS,
        default=DEFAULT_QK_NORM_BACKEND,
        help="Q/K per-head L2-normalization reduction backend.",
    )
    parser.add_argument(
        "--rope-backend",
        choices=SUPPORTED_ROPE_BACKENDS,
        default=MLX_REAL_BACKEND,
        help="RoPE phase-generation and rotation backend.",
    )
    parser.add_argument(
        "--turing-rope-phase-lut",
        metavar="NPZ",
        help=(
            "Required with cuda-polar-turing-t4: NPZ containing the "
            "authenticated Tesla T4 torch.polar phase_pairs table."
        ),
    )
    parser.add_argument(
        "--expected-turing-rope-phase-lut-sha256",
        help="Require the Turing RoPE phase LUT NPZ to match this exact SHA256.",
    )
    parser.add_argument(
        "--turing-rsqrt-lut",
        metavar="NPZ",
        help=(
            "Required with cuda-welford-turing-t4: NPZ containing the "
            "normalized T4 MUFU.RSQ signed-ULP LUT."
        ),
    )
    parser.add_argument(
        "--expected-turing-rsqrt-lut-sha256",
        help=(
            "Require the Turing rsqrt LUT NPZ to match this exact SHA256."
        ),
    )
    parser.add_argument("--shape-flow-noise-sample", metavar="NPZ",
                        help="Diagnostic: replay exact shape SLat first-step noise from an NPZ "
                             "containing coords plus noise or sample_feats.")
    parser.add_argument("--shape-flow-block-injection-trace", metavar="NPZ",
                        help="Diagnostic: inject a named shape-flow block tensor from an NPZ trace.")
    parser.add_argument("--shape-flow-block-injection-manifest", metavar="JSON",
                        help="Diagnostic: inject multiple named shape-flow block tensors from a JSON manifest.")
    parser.add_argument("--shape-flow-block-injection-step-index", type=int, default=0,
                        help="Diagnostic: shape-flow sampler step where block injection applies.")
    parser.add_argument("--shape-flow-block-injection-block-index", type=int, default=1,
                        help="Diagnostic: shape-flow block where block injection applies.")
    parser.add_argument("--shape-flow-block-injection-branch",
                        choices=["pos", "neg", "both"], default="both",
                        help="Diagnostic: shape-flow CFG branch where block injection applies.")
    parser.add_argument("--shape-flow-block-injection-stage",
                        choices=["norm1", "modulated_self_input", "attention_raw",
                                 "after_self", "cross_attention_raw", "after_cross", "after_mlp"],
                        default="attention_raw",
                        help="Diagnostic: shape-flow block tensor stage to replace.")
    parser.add_argument("--shape-flow-block-injection-array-key",
                        help="Diagnostic: explicit trace array key for single-branch shape injection.")
    parser.add_argument("--shape-flow-block-injection-scale", type=float, default=1.0,
                        help="Diagnostic: scale source-minus-live block tensor before injection.")
    parser.add_argument("--checkpoint-stop-file", metavar="PATH",
                        help="Cooperatively exit with a checkpoint-yield receipt if PATH exists "
                        "after a durable checkpoint boundary. Requires --save-checkpoints.")
    parser.add_argument("--resume", metavar="DIR",
                        help="Resume from checkpoints in DIR (skips completed inference stages)")
    parser.add_argument("--edit-target", metavar="IMAGE",
                        help="VS3D editing: target reference image showing desired appearance. "
                             "Requires --image (source). Stage 1 uses VS3D RASI+PMG guidance "
                             "anchored to the source latent, steered toward this target image.")
    parser.add_argument("--vs3d-cfg-src", type=float, default=1.5,
                        help="VS3D CFG weight for source branch in RASI (default: 1.5)")
    parser.add_argument("--vs3d-cfg-tgt", type=float, default=9.0,
                        help="VS3D CFG weight for target branch in PMG (default: 9.0)")
    parser.add_argument("--vs3d-steps-src", type=int, default=12,
                        help="Steps for source Stage 1 run to get x_src anchor (default: 12)")
    parser.add_argument("--vs3d-guidance-low", type=float, default=0.6,
                        help="VS3D guidance interval lower bound (default: 0.6).")
    parser.add_argument("--vs3d-guidance-high", type=float, default=1.0,
                        help="VS3D guidance interval upper bound (default: 1.0).")
    parser.add_argument("--vs3d-rasi-k", type=int, default=0,
                        help="RASI inner optimization steps (default: 0 = RASI disabled). "
                             "Set >0 to enable RASI source anchoring.")
    args = parser.parse_args()
    os.environ["TRELLIS2MLX_QK_NORM_BACKEND"] = args.qk_norm_backend

    if args.checkpoint_stop_file and not args.save_checkpoints:
        parser.error("--save-checkpoints is required when --checkpoint-stop-file is set")
    if args.stop_after_stage and not args.save_checkpoints:
        parser.error("--save-checkpoints is required when --stop-after-stage is set")
    if args.shape_slat_sample and args.stop_after_stage != "decoder_output":
        parser.error("--shape-slat-sample requires --stop-after-stage decoder_output")
    if args.shape_slat_support_sample and not args.no_cascade:
        parser.error("--shape-slat-support-sample requires --no-cascade")
    if args.shape_slat_sample and args.shape_slat_support_sample:
        parser.error("--shape-slat-sample and --shape-slat-support-sample are mutually exclusive")
    if args.shared_noise_sparse_only and not args.shared_noise:
        parser.error("--shared-noise-sparse-only requires --shared-noise")
    if args.sparse_flow_start_sample and args.sparse_flow_trace_sample:
        parser.error("--sparse-flow-start-sample and --sparse-flow-trace-sample are mutually exclusive")
    if args.sparse_flow_start_sample and args.stop_after_stage == "sparse_flow_block_trace":
        parser.error(
            "--sparse-flow-start-sample does not apply to sparse_flow_block_trace; "
            "use --sparse-flow-trace-sample"
        )
    if args.sparse_flow_block_injection_trace and args.sparse_flow_block_injection_manifest:
        parser.error(
            "--sparse-flow-block-injection-trace and "
            "--sparse-flow-block-injection-manifest are mutually exclusive"
        )
    if args.sparse_flow_layernorm_correction_report and (
        args.sparse_flow_block_injection_trace or args.sparse_flow_block_injection_manifest
    ):
        parser.error(
            "--sparse-flow-layernorm-correction-report is mutually exclusive with "
            "sparse-flow block injection"
        )
    if (args.sparse_flow_block_injection_trace or args.sparse_flow_block_injection_manifest) and args.compile:
        parser.error("--compile is not supported with sparse-flow block injection")
    if args.sparse_flow_layernorm_correction_report and args.compile:
        parser.error("--compile is not supported with sparse-flow LayerNorm correction")
    if args.shape_flow_block_injection_trace and args.shape_flow_block_injection_manifest:
        parser.error(
            "--shape-flow-block-injection-trace and "
            "--shape-flow-block-injection-manifest are mutually exclusive"
        )
    if (args.shape_flow_block_injection_trace or args.shape_flow_block_injection_manifest) and args.compile:
        parser.error("--compile is not supported with shape-flow block injection")
    if args.stop_after_stage == "shape_flow_block_trace" and not args.no_cascade:
        parser.error("--stop-after-stage shape_flow_block_trace requires --no-cascade")
    if (
        args.shape_flow_attention_backend
        or args.shape_flow_attention_softmax_backend
        or args.shape_flow_attention_value_backend
    ) and args.stop_after_stage != "shape_flow_block_trace":
        parser.error(
            "shape-flow attention selectors require "
            "--stop-after-stage shape_flow_block_trace"
        )
    if args.shape_flow_layernorm_backend == CUDA_WELFORD_TURING_T4_BACKEND:
        if (
            not args.turing_rsqrt_lut
            or not args.expected_turing_rsqrt_lut_sha256
        ):
            parser.error(
                f"--shape-flow-layernorm-backend "
                f"{CUDA_WELFORD_TURING_T4_BACKEND} requires "
                "--turing-rsqrt-lut and "
                "--expected-turing-rsqrt-lut-sha256"
            )
        try:
            turing_lut, turing_lut_sha256 = _load_turing_rsqrt_lut(
                args.turing_rsqrt_lut,
                args.expected_turing_rsqrt_lut_sha256,
            )
            configure_shape_flow_layernorm_backend(
                args.shape_flow_layernorm_backend,
                turing_rsqrt_delta_lut=turing_lut,
                turing_rsqrt_lut_sha256=turing_lut_sha256,
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    else:
        if (
            args.turing_rsqrt_lut
            or args.expected_turing_rsqrt_lut_sha256
        ):
            parser.error(
                "--turing-rsqrt-lut and its expected SHA256 only apply to "
                f"{CUDA_WELFORD_TURING_T4_BACKEND}"
            )
        configure_shape_flow_layernorm_backend(
            args.shape_flow_layernorm_backend
        )
    if args.rope_backend == CUDA_POLAR_TURING_T4_BACKEND:
        if (
            not args.turing_rope_phase_lut
            or not args.expected_turing_rope_phase_lut_sha256
        ):
            parser.error(
                f"--rope-backend {CUDA_POLAR_TURING_T4_BACKEND} requires "
                "--turing-rope-phase-lut and "
                "--expected-turing-rope-phase-lut-sha256"
            )
        try:
            rope_phase_lut, rope_phase_lut_sha256 = (
                _load_turing_rope_phase_lut(
                    args.turing_rope_phase_lut,
                    args.expected_turing_rope_phase_lut_sha256,
                )
            )
            configure_rope_backend(
                args.rope_backend,
                turing_phase_lut=rope_phase_lut,
                turing_phase_lut_sha256=rope_phase_lut_sha256,
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    else:
        if (
            args.turing_rope_phase_lut
            or args.expected_turing_rope_phase_lut_sha256
        ):
            parser.error(
                "--turing-rope-phase-lut and its expected SHA256 only apply "
                f"to {CUDA_POLAR_TURING_T4_BACKEND}"
            )
        configure_rope_backend(args.rope_backend)
    shared_noise = np.load(args.shared_noise) if args.shared_noise else None

    # === Resume from checkpoints ===
    if args.resume:
        from trellmlx.checkpoint import load_checkpoint, has_checkpoint, list_checkpoints
        available = list_checkpoints(args.resume)
        print(f"Resuming from {args.resume} (stages: {', '.join(available)})", flush=True)

        if has_checkpoint(args.resume, "texture") and has_checkpoint(args.resume, "mesh_raw"):
            mesh_data = load_checkpoint(args.resume, "mesh_raw")
            tex_data = load_checkpoint(args.resume, "texture")

            vertices = mesh_data["vertices"]
            faces = mesh_data["faces"]
            mesh_grid_size = int(mesh_data["mesh_grid_size"])
            tex_np = tex_data["tex_np"]
            tex_coords_spatial = tex_data["tex_coords_spatial"]

            print(f"  Loaded mesh: {len(vertices):,}V {len(faces):,}F", flush=True)
            print(f"  Loaded texture: {tex_np.shape[0]:,} voxels, {tex_np.shape[1]} channels", flush=True)

            t_total = time.perf_counter()

            # Re-run cleanup + simplification with current settings
            vertices, faces = _cleanup_and_simplify_mesh(
                vertices, faces,
                target_faces=args.target_faces,
                no_cleanup=args.no_cleanup,
                keep_largest=args.keep_largest,
                simplify_first=args.simplify_first,
                reference_cleanup=args.reference_cleanup,
                qem_simplify=args.qem_simplify,
                qem_backend=args.qem_backend,
                source_native_source_root=args.source_native_source_root,
                source_native_python=args.source_native_python,
            )
            if args.save_checkpoints:
                from trellmlx.checkpoint import save_checkpoint
                save_checkpoint(args.save_checkpoints, "mesh_clean",
                                vertices=vertices, faces=faces,
                                mesh_grid_size=mesh_grid_size)
                maybe_checkpoint_yield(
                    stop_file=args.checkpoint_stop_file,
                    checkpoint_dir=args.save_checkpoints,
                    completed_stage="mesh_clean",
                    next_stage="texture_bake",
                    output_path=args.output,
                )

            # Jump straight to texture baking
            from trellmlx.texture_bake import bake_texture
            unwrap_fn, method_name = _select_uv_method(args.uv_method, vertices, faces)
            t0 = time.perf_counter()
            uv_verts, uv_faces, uvs, vmapping = unwrap_fn(vertices, faces)
            print(f"  UV unwrap ({method_name}): {len(uv_verts):,}V {len(uv_faces):,}F "
                  f"({time.perf_counter()-t0:.1f}s)", flush=True)
            if args.save_checkpoints:
                from trellmlx.checkpoint import save_checkpoint
                save_checkpoint(args.save_checkpoints, "mesh_uv",
                                vertices=uv_verts, faces=uv_faces,
                                uvs=uvs, vmapping=vmapping,
                                mesh_grid_size=mesh_grid_size,
                                uv_method=method_name)
                maybe_checkpoint_yield(
                    stop_file=args.checkpoint_stop_file,
                    checkpoint_dir=args.save_checkpoints,
                    completed_stage="mesh_uv",
                    next_stage="texture_bake",
                    output_path=args.output,
                )

            base_color, metallic_roughness, alpha_mode = bake_texture(
                uv_verts, uv_faces, uvs, vmapping,
                tex_coords_spatial, tex_np, mesh_grid_size,
                texture_size=args.texture_size,
                backend=args.texture_backend,
            )

            # Export
            import trimesh
            from trimesh.visual.material import PBRMaterial
            from PIL import Image

            if len(uv_verts) > 0 and len(uv_faces) > 0:
                export_verts = uv_verts.copy()
                export_verts[:, 1], export_verts[:, 2] = uv_verts[:, 2].copy(), -uv_verts[:, 1].copy()
                export_uvs = uvs.copy()
                export_uvs[:, 1] = 1 - export_uvs[:, 1]
                mesh = trimesh.Trimesh(vertices=export_verts, faces=uv_faces, process=False)
                normals = mesh.vertex_normals

                material = PBRMaterial(
                    baseColorTexture=Image.fromarray(base_color),
                    baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
                    metallicRoughnessTexture=Image.fromarray(metallic_roughness),
                    metallicFactor=1.0, roughnessFactor=1.0,
                    alphaMode=alpha_mode, doubleSided=True,
                )
                textured_mesh = trimesh.Trimesh(
                    vertices=export_verts, faces=uv_faces,
                    vertex_normals=normals, process=False,
                    visual=trimesh.visual.TextureVisuals(uv=export_uvs, material=material),
                )
                textured_mesh.export(args.output)
                print(f"\n  Saved: {args.output} ({os.path.getsize(args.output)/1e6:.1f}MB)", flush=True)
            else:
                print("  WARNING: Empty mesh!", flush=True)
            print(f"\nResume total: {time.perf_counter()-t_total:.1f}s", flush=True)
            return

        elif has_checkpoint(args.resume, "mesh_raw"):
            print("  Only mesh checkpoint found — will re-run texture stages", flush=True)
            # Could add partial resume here later
        else:
            print(f"  No usable checkpoints in {args.resume}, running full pipeline", flush=True)

    mx.random.seed(args.seed)
    n_steps = args.steps
    t_total = time.perf_counter()

    HF_4B = os.path.expanduser(
        "~/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/"
        "snapshots/af44b45f2e35a493886929c6d786e563ec68364d/ckpts/"
    )
    HF_LARGE = os.path.expanduser(
        "~/.cache/huggingface/hub/models--microsoft--TRELLIS-image-large/"
        "snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/"
    )

    from trellmlx.weight_loader import load_weights
    from trellmlx.samplers import flow_euler_sample
    from trellmlx.cleanup import cleanup_model, cleanup

    if args.shape_slat_sample:
        print("=== Shape SLat replay: Decode Shape ===", flush=True)
        shape_slat_sample_npz = np.load(args.shape_slat_sample)
        missing = {"feats", "coords"} - set(shape_slat_sample_npz.files)
        if missing:
            raise ValueError(
                "--shape-slat-sample NPZ must contain feats and coords arrays; "
                f"missing {sorted(missing)}"
            )
        shape_feats_np = np.asarray(shape_slat_sample_npz["feats"], dtype=np.float32)
        quant_coords = np.asarray(shape_slat_sample_npz["coords"], dtype=np.int32)
        if shape_feats_np.ndim != 2 or shape_feats_np.shape[1] != 32:
            raise ValueError(
                "--shape-slat-sample feats must have shape [N, 32], "
                f"got {shape_feats_np.shape}"
            )
        if quant_coords.ndim != 2 or quant_coords.shape[1] != 4:
            raise ValueError(
                "--shape-slat-sample coords must have shape [N, 4], "
                f"got {quant_coords.shape}"
            )
        if quant_coords.shape[0] != shape_feats_np.shape[0]:
            raise ValueError(
                "--shape-slat-sample feats and coords row counts must match, "
                f"got {shape_feats_np.shape[0]} and {quant_coords.shape[0]}"
            )

        mesh_grid_size = args.resolution
        sample_meta_path = Path(args.shape_slat_sample).with_suffix(".json")
        if sample_meta_path.exists():
            with sample_meta_path.open() as f:
                sample_meta = json.load(f)
            mesh_grid_size = int(sample_meta.get("mesh_grid_size", mesh_grid_size))

        from trellmlx.models.shape_slat_decoder import SLatDecoder

        shape_decoder = SLatDecoder(out_channels=7, pred_subdiv=True)
        load_weights(shape_decoder, HF_4B + "shape_dec_next_dc_f16c32_fp16.safetensors", verbose=False)

        t0 = time.perf_counter()
        dec_out, dec_coords, shape_subs = shape_decoder(
            mx.array(shape_feats_np),
            mx.array(quant_coords),
            return_subs=True,
        )
        mx.eval(dec_out)
        print(
            f"  Shape SLat replay decoded: {time.perf_counter()-t0:.1f}s "
            f"({dec_out.shape[0]:,} voxels)",
            flush=True,
        )

        cleanup_model(shape_decoder)
        del shape_decoder
        gc.collect()

        from trellmlx.checkpoint import save_checkpoint

        save_checkpoint(
            args.save_checkpoints,
            "decoder_output",
            feats=np.array(dec_out).astype(np.float32, copy=False),
            coords=np.array(dec_coords).astype(np.int32, copy=False),
            shape_subs=[np.array(mask) for mask in shape_subs],
            mesh_grid_size=mesh_grid_size,
        )
        print("  Stop after stage: decoder_output", flush=True)
        return

    vs3d_mode = bool(args.edit_target)
    if vs3d_mode and not os.path.exists(args.edit_target):
        raise FileNotFoundError(f"--edit-target path does not exist: {args.edit_target!r}. "
                                f"Did you forget to pass edit_target as a greenroom param?")

    if args.quantize:
        from trellmlx.quantize import quantize_model

    # === Image conditioning ===
    if args.conditioning_sample:
        conditioning_sample_npz = np.load(args.conditioning_sample)
        missing = {"cond", "neg_cond"} - set(conditioning_sample_npz.files)
        if missing:
            raise ValueError(
                "--conditioning-sample NPZ must contain cond and neg_cond arrays; "
                f"missing {sorted(missing)}"
            )
        cond = mx.array(conditioning_sample_npz["cond"])
        neg_cond = mx.array(conditioning_sample_npz["neg_cond"])
        print(
            f"  Conditioning sample: {args.conditioning_sample} "
            f"cond={cond.shape} neg_cond={neg_cond.shape}",
            flush=True,
        )
    elif args.image:
        # Preprocess: background removal + center crop
        image_paths = args.image
        if not args.no_rembg:
            from trellmlx.preprocess import preprocess_image
            import tempfile
            processed_paths = []
            for img_path in image_paths:
                print(f"  Preprocessing {img_path}...", flush=True)
                processed = preprocess_image(img_path)
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                processed.save(tmp.name)
                processed_paths.append(tmp.name)
            image_paths = processed_paths

        if len(image_paths) == 1:
            cond = _extract_image_features(image_paths[0])
        else:
            # Multi-view: extract features per view, concatenate along sequence dim
            view_features = []
            for img_path in image_paths:
                feat = _extract_image_features(img_path)
                view_features.append(feat)
            cond = mx.concatenate(view_features, axis=1)
            print(f"  Multi-view: {len(image_paths)} views → {cond.shape[1]} context tokens", flush=True)
        neg_cond = mx.zeros_like(cond)
    else:
        print("No image — random conditioning", flush=True)
        cond = mx.random.normal((1, 10, 1024))
        neg_cond = mx.zeros_like(cond)
    if args.save_checkpoints:
        from trellmlx.checkpoint import save_checkpoint
        save_checkpoint(
            args.save_checkpoints,
            "conditioning",
            cond=np.array(cond),
            neg_cond=np.array(neg_cond),
        )
        if args.stop_after_stage == "conditioning":
            print("  Stop after stage: conditioning", flush=True)
            return

    # VS3D: extract target conditioning and relabel source cond
    if vs3d_mode:
        if not args.image:
            raise ValueError("--edit-target requires --image (source image)")
        print(f"  VS3D mode: extracting target features from {args.edit_target}...", flush=True)
        if not args.no_rembg:
            from trellmlx.preprocess import preprocess_image
            import tempfile
            tgt_processed = preprocess_image(args.edit_target)
            tgt_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tgt_processed.save(tgt_tmp.name)
            tgt_image_path = tgt_tmp.name
        else:
            tgt_image_path = args.edit_target
        cond_src = cond          # source image conditioning
        cond_tgt = _extract_image_features(tgt_image_path)
        print(f"  VS3D: cond_src {cond_src.shape}, cond_tgt {cond_tgt.shape}", flush=True)

    # === Stage 1: Sparse Structure ===
    print("=== Stage 1: Sparse Structure ===", flush=True)
    from trellmlx.models.sparse_structure_flow import SparseStructureFlowModel
    from trellmlx.models.sparse_structure_decoder import SparseStructureDecoder

    ss_flow = SparseStructureFlowModel()
    load_weights(ss_flow, HF_4B + "ss_flow_img_dit_1_3B_64_bf16.safetensors", verbose=False)
    if args.quantize:
        quantize_model(ss_flow, bits=args.quantize)
    if args.compile:
        ss_flow.compile()
    ss_dec = SparseStructureDecoder()
    load_weights(ss_dec, HF_LARGE + "ss_dec_conv3d_16l8_fp16.safetensors", verbose=False)

    if shared_noise is not None:
        noise = mx.array(shared_noise["ss_noise"]).astype(mx.float32)
        print(f"  Shared sparse noise: {args.shared_noise} {noise.shape}", flush=True)
    else:
        noise = mx.random.normal((1, 8, 16, 16, 16)).astype(mx.float32)
    t0 = time.perf_counter()

    if vs3d_mode:
        from trellmlx.vs3d import vs3d_flow_sample
        # Step 1: run source generation to get x_src anchor
        print(f"  VS3D: source pass ({args.vs3d_steps_src} steps) to get x_src...", flush=True)
        mx.random.seed(args.seed)
        src_noise = mx.random.normal((1, 8, 16, 16, 16))
        x_src = flow_euler_sample(
            ss_flow, src_noise, cond_src, neg_cond,
            steps=args.vs3d_steps_src, verbose=False,
        )
        mx.eval(x_src)
        print(f"  VS3D: source pass done ({time.perf_counter()-t0:.1f}s)", flush=True)
        # Step 2: VS3D editing pass
        t1 = time.perf_counter()
        mx.random.seed(args.seed)
        noise = mx.random.normal((1, 8, 16, 16, 16))
        z_s = vs3d_flow_sample(
            model=ss_flow,
            noise=noise,
            cond_src=cond_src,
            cond_tgt=cond_tgt,
            neg_cond=neg_cond,
            x_src=x_src,
            stage="dense",
            steps=n_steps,
            cfg_w_src=args.vs3d_cfg_src,
            cfg_w_tgt=args.vs3d_cfg_tgt,
            guidance_interval=(args.vs3d_guidance_low, args.vs3d_guidance_high),
            rasi_K=args.vs3d_rasi_k,
            verbose=True,
        )
        mx.eval(z_s)
        print(f"  VS3D editing pass: {time.perf_counter()-t1:.1f}s", flush=True)
    else:
        sparse_block_injection = None
        sparse_block_injection_json = ""
        sparse_flow_layernorm_correction_json = ""
        if args.sparse_flow_block_injection_trace and args.sparse_flow_block_injection_manifest:
            raise ValueError(
                "--sparse-flow-block-injection-trace and "
                "--sparse-flow-block-injection-manifest are mutually exclusive"
            )
        if args.sparse_flow_block_injection_manifest:
            from trellmlx.sparse_block_injection import load_sparse_block_injection_manifest

            sparse_block_injection = load_sparse_block_injection_manifest(
                args.sparse_flow_block_injection_manifest,
            )
            sparse_block_injection_json = json.dumps(
                sparse_block_injection.report_identity(),
                sort_keys=True,
            )
        elif args.sparse_flow_layernorm_correction_report:
            from trellmlx.sparse_block_injection import load_sparse_layernorm_correction

            sparse_block_injection = load_sparse_layernorm_correction(
                args.sparse_flow_layernorm_correction_report,
                branch=args.sparse_flow_layernorm_correction_branch,
                step_index=args.sparse_flow_layernorm_correction_step_index,
                block_index=args.sparse_flow_layernorm_correction_block_index,
                mode=args.sparse_flow_layernorm_correction_mode,
                include=args.sparse_flow_layernorm_correction_include,
            )
            sparse_flow_layernorm_correction_json = json.dumps(
                sparse_block_injection.report_identity(),
                sort_keys=True,
            )
        elif args.sparse_flow_block_injection_trace:
            from trellmlx.sparse_block_injection import load_sparse_block_injection

            sparse_block_injection = load_sparse_block_injection(
                args.sparse_flow_block_injection_trace,
                branch=args.sparse_flow_block_injection_branch,
                step_index=args.sparse_flow_block_injection_step_index,
                block_index=args.sparse_flow_block_injection_block_index,
                stage=args.sparse_flow_block_injection_stage,
                array_key=args.sparse_flow_block_injection_array_key,
            )
            sparse_block_injection_json = json.dumps(
                sparse_block_injection.report_identity(),
                sort_keys=True,
            )
        # Cast conditioning to fp32 to match the fp32 sparse structure model
        if args.stop_after_stage == "sparse_flow_block_trace":
            pos_cond = cond.astype(mx.float32)
            neg_cond_fp32 = neg_cond.astype(mx.float32)
            use_kv_cache = not args.sparse_flow_trace_no_kv_cache
            pos_kv_cache = ss_flow.build_cross_kv_cache(pos_cond) if use_kv_cache else None
            neg_kv_cache = ss_flow.build_cross_kv_cache(neg_cond_fp32) if use_kv_cache else None
            trace_step_index = args.sparse_flow_trace_step_index
            if trace_step_index < 0 or trace_step_index >= n_steps:
                raise ValueError(
                    f"--sparse-flow-trace-step-index must be in [0, {n_steps - 1}], "
                    f"got {trace_step_index}"
                )
            trace_block_index = args.sparse_flow_trace_block_index
            trace_input_mode = "sampler_sample"
            pos_block_input_key = ""
            neg_block_input_key = ""

            def schedule_trace_t():
                t_seq = np.linspace(1, 0, n_steps + 1)
                t_seq = 5.0 * t_seq / (1 + (5.0 - 1) * t_seq)
                return float(t_seq[trace_step_index])

            def trace_t_from_npz(trace_npz):
                if "t" in trace_npz and np.ndim(trace_npz["t"]) > 0:
                    t_value = float(trace_npz["t"][trace_step_index])
                elif "t" in trace_npz:
                    t_value = float(trace_npz["t"])
                else:
                    return schedule_trace_t()
                return t_value / 1000.0 if t_value > 1.0 else t_value

            def load_projected_block_input(block_input_npz, label):
                if label == "pos":
                    candidate_keys = [
                        f"pos_block{trace_block_index}_input",
                        "pos_block_input",
                        f"block{trace_block_index}_input",
                        "block_input",
                    ]
                else:
                    candidate_keys = [
                        f"neg_block{trace_block_index}_input",
                        "neg_block_input",
                        f"block{trace_block_index}_input",
                        "block_input",
                    ]
                for key in candidate_keys:
                    if key in block_input_npz:
                        return key, mx.array(block_input_npz[key]).astype(mx.float32)
                raise ValueError(
                    "--sparse-flow-trace-block-input-sample NPZ must contain "
                    f"one of {candidate_keys}; available keys: {block_input_npz.files}"
                )

            def active_trace_injection(branch):
                if sparse_block_injection is None:
                    return None
                if hasattr(sparse_block_injection, "active_for_step_branch"):
                    return sparse_block_injection.active_for_step_branch(
                        step_index=trace_step_index,
                        branch=branch,
                    )
                if sparse_block_injection.applies(step_index=trace_step_index, branch=branch):
                    return sparse_block_injection
                return None

            if args.sparse_flow_trace_block_input_sample:
                if args.sparse_flow_trace_sample:
                    raise ValueError(
                        "--sparse-flow-trace-block-input-sample and --sparse-flow-trace-sample "
                        "are mutually exclusive"
                    )
                block_input_npz = np.load(args.sparse_flow_trace_block_input_sample)
                pos_block_input_key, pos_block_input = load_projected_block_input(block_input_npz, "pos")
                neg_block_input_key, neg_block_input = load_projected_block_input(block_input_npz, "neg")
                trace_t = trace_t_from_npz(block_input_npz)
                t_tensor = mx.array([1000.0 * trace_t], dtype=mx.float32)
                pos_trace = ss_flow.trace_projected_block_input(
                    pos_block_input,
                    t_tensor,
                    pos_cond,
                    block_index=trace_block_index,
                    cross_kv_cache=pos_kv_cache,
                    sparse_block_injection=active_trace_injection("pos"),
                    sparse_block_injection_branch="pos",
                )
                neg_trace = ss_flow.trace_projected_block_input(
                    neg_block_input,
                    t_tensor,
                    neg_cond_fp32,
                    block_index=trace_block_index,
                    cross_kv_cache=neg_kv_cache,
                    sparse_block_injection=active_trace_injection("neg"),
                    sparse_block_injection_branch="neg",
                )
                trace_input_mode = "projected_block_input"
            elif args.sparse_flow_trace_sample:
                trace_sample_npz = np.load(args.sparse_flow_trace_sample)
                if "sample_in" not in trace_sample_npz:
                    raise ValueError(
                        "--sparse-flow-trace-sample NPZ must contain a sample_in array"
                    )
                trace_sample_np = trace_sample_npz["sample_in"]
                if trace_sample_np.ndim == 6:
                    if trace_step_index >= trace_sample_np.shape[0]:
                        raise ValueError(
                            f"trace sample contains {trace_sample_np.shape[0]} steps, "
                            f"cannot select step {trace_step_index}"
                        )
                    trace_sample_np = trace_sample_np[trace_step_index]
                elif trace_sample_np.ndim != 5:
                    raise ValueError(
                        "sample_in must have shape [B,C,R,R,R] or [steps,B,C,R,R,R], "
                        f"got {trace_sample_np.shape}"
                    )
                trace_sample = mx.array(trace_sample_np).astype(mx.float32)
                trace_t = trace_t_from_npz(trace_sample_npz)
                t_tensor = mx.array([1000.0 * trace_t], dtype=mx.float32)
                pos_trace = ss_flow.trace_block(
                    trace_sample,
                    t_tensor,
                    pos_cond,
                    block_index=trace_block_index,
                    cross_kv_cache=pos_kv_cache,
                    sparse_block_injection=active_trace_injection("pos"),
                    sparse_block_injection_branch="pos",
                )
                neg_trace = ss_flow.trace_block(
                    trace_sample,
                    t_tensor,
                    neg_cond_fp32,
                    block_index=trace_block_index,
                    cross_kv_cache=neg_kv_cache,
                    sparse_block_injection=active_trace_injection("neg"),
                    sparse_block_injection_branch="neg",
                )
            elif trace_step_index == 0:
                trace_sample = noise
                trace_t = 1.0
                t_tensor = mx.array([1000.0 * trace_t], dtype=mx.float32)
                pos_trace = ss_flow.trace_block(
                    trace_sample,
                    t_tensor,
                    pos_cond,
                    block_index=trace_block_index,
                    cross_kv_cache=pos_kv_cache,
                    sparse_block_injection=active_trace_injection("pos"),
                    sparse_block_injection_branch="pos",
                )
                neg_trace = ss_flow.trace_block(
                    trace_sample,
                    t_tensor,
                    neg_cond_fp32,
                    block_index=trace_block_index,
                    cross_kv_cache=neg_kv_cache,
                    sparse_block_injection=active_trace_injection("neg"),
                    sparse_block_injection_branch="neg",
                )
            else:
                trace_steps = []
                _ = flow_euler_sample(
                    ss_flow,
                    noise,
                    pos_cond,
                    neg_cond_fp32,
                    steps=n_steps,
                    verbose=False,
                    capture_steps=trace_steps,
                )
                if trace_step_index >= len(trace_steps):
                    raise ValueError(
                        f"sparse flow captured {len(trace_steps)} steps, cannot trace step "
                        f"{trace_step_index}"
                    )
                trace_sample = trace_steps[trace_step_index]["sample_in"]
                trace_t = float(np.array(trace_steps[trace_step_index]["t"]))
                t_tensor = mx.array([1000.0 * trace_t], dtype=mx.float32)
                pos_trace = ss_flow.trace_block(
                    trace_sample,
                    t_tensor,
                    pos_cond,
                    block_index=trace_block_index,
                    cross_kv_cache=pos_kv_cache,
                    sparse_block_injection=active_trace_injection("pos"),
                    sparse_block_injection_branch="pos",
                )
                neg_trace = ss_flow.trace_block(
                    trace_sample,
                    t_tensor,
                    neg_cond_fp32,
                    block_index=trace_block_index,
                    cross_kv_cache=neg_kv_cache,
                    sparse_block_injection=active_trace_injection("neg"),
                    sparse_block_injection_branch="neg",
                )
            def trace_np(value):
                return np.array(value.astype(mx.float32))[None].astype(np.float32, copy=False)

            from trellmlx.checkpoint import save_checkpoint
            selected_trace_keys = _parse_sparse_flow_trace_keys(args.sparse_flow_trace_keys)
            trace_payload = {
                f"pos_{name}": trace_np(value)
                for name, value in pos_trace.items()
            }
            trace_payload.update(
                {
                    f"neg_{name}": trace_np(value)
                    for name, value in neg_trace.items()
                }
            )
            trace_payload = _filter_sparse_flow_trace_payload(trace_payload, selected_trace_keys)
            save_checkpoint(
                args.save_checkpoints,
                "sparse_flow_block_trace",
                **trace_payload,
                trace_block_index=np.array(trace_block_index, dtype=np.int32),
                sparse_flow_trace_step_index=np.array(trace_step_index, dtype=np.int32),
                sparse_flow_trace_sample_path=np.array(args.sparse_flow_trace_sample or ""),
                sparse_flow_trace_block_input_sample_path=np.array(
                    args.sparse_flow_trace_block_input_sample or ""
                ),
                sparse_flow_trace_input_mode=np.array(trace_input_mode),
                sparse_flow_trace_pos_block_input_key=np.array(pos_block_input_key),
                sparse_flow_trace_neg_block_input_key=np.array(neg_block_input_key),
                sparse_flow_trace_uses_kv_cache=np.array(use_kv_cache, dtype=np.bool_),
                sparse_flow_trace_selected_keys=np.array(selected_trace_keys, dtype=str),
                t=np.array(1000.0 * trace_t, dtype=np.float32),
                steps=np.array(n_steps, dtype=np.int32),
                rescale_t=np.array(5.0, dtype=np.float32),
            )
            print(
                "  Stop after stage: "
                f"sparse_flow_block_trace step={trace_step_index} block={trace_block_index}",
                flush=True,
            )
            return

        step_capture = {} if args.stop_after_stage == "sparse_flow_step" else None
        step_captures = [] if args.stop_after_stage == "sparse_flow_steps" else None
        sparse_flow_start_sample = noise
        sparse_flow_start_step_index = 0
        sparse_flow_start_sample_path = ""
        if args.sparse_flow_start_sample:
            sparse_flow_start_step_index = args.sparse_flow_start_step_index
            if sparse_flow_start_step_index < 0 or sparse_flow_start_step_index >= n_steps:
                raise ValueError(
                    f"--sparse-flow-start-step-index must be in [0, {n_steps - 1}], "
                    f"got {sparse_flow_start_step_index}"
                )
            start_sample_npz = np.load(args.sparse_flow_start_sample)
            if "sample_in" not in start_sample_npz:
                raise ValueError(
                    "--sparse-flow-start-sample NPZ must contain a sample_in array"
                )
            start_sample_np = start_sample_npz["sample_in"]
            if start_sample_np.ndim == 6:
                if sparse_flow_start_step_index >= start_sample_np.shape[0]:
                    raise ValueError(
                        f"start sample contains {start_sample_np.shape[0]} steps, "
                        f"cannot select step {sparse_flow_start_step_index}"
                    )
                start_sample_np = start_sample_np[sparse_flow_start_step_index]
            elif start_sample_np.ndim != 5:
                raise ValueError(
                    "sample_in must have shape [B,C,R,R,R] or [steps,B,C,R,R,R], "
                    f"got {start_sample_np.shape}"
                )
            sparse_flow_start_sample = mx.array(start_sample_np).astype(mx.float32)
            sparse_flow_start_sample_path = args.sparse_flow_start_sample
            print(
                "  Continuing sparse flow from "
                f"{args.sparse_flow_start_sample} step {sparse_flow_start_step_index}",
                flush=True,
            )
        elif args.stop_after_stage == "sparse_flow_step" and args.sparse_flow_trace_sample:
            trace_step_index = args.sparse_flow_trace_step_index
            if trace_step_index < 0 or trace_step_index >= n_steps:
                raise ValueError(
                    f"--sparse-flow-trace-step-index must be in [0, {n_steps - 1}], "
                    f"got {trace_step_index}"
                )
            step_sample_npz = np.load(args.sparse_flow_trace_sample)
            if "sample_in" not in step_sample_npz:
                raise ValueError(
                    "--sparse-flow-trace-sample NPZ must contain a sample_in array"
                )
            step_sample_np = step_sample_npz["sample_in"]
            if step_sample_np.ndim == 6:
                if trace_step_index >= step_sample_np.shape[0]:
                    raise ValueError(
                        f"trace sample contains {step_sample_np.shape[0]} steps, "
                        f"cannot select step {trace_step_index}"
                    )
                step_sample_np = step_sample_np[trace_step_index]
            elif step_sample_np.ndim != 5:
                raise ValueError(
                    "sample_in must have shape [B,C,R,R,R] or [steps,B,C,R,R,R], "
                    f"got {step_sample_np.shape}"
                )
            sparse_flow_start_sample = mx.array(step_sample_np).astype(mx.float32)
            sparse_flow_start_step_index = trace_step_index

        z_s = flow_euler_sample(ss_flow, sparse_flow_start_sample,
                                cond.astype(mx.float32), neg_cond.astype(mx.float32),
                                steps=n_steps, verbose=False,
                                capture_first_step=step_capture,
                                capture_steps=step_captures,
                                stop_after_first_step=args.stop_after_stage == "sparse_flow_step",
                                start_step_index=sparse_flow_start_step_index,
                                sparse_block_injection=sparse_block_injection)
        mx.eval(z_s)

    print(f"  Sampled: {time.perf_counter()-t0:.1f}s", flush=True)

    if args.save_checkpoints and args.stop_after_stage == "sparse_flow_step":
        from trellmlx.checkpoint import save_checkpoint
        save_checkpoint(
            args.save_checkpoints,
            "sparse_flow_step",
            noise=np.array(noise).astype(np.float32, copy=False),
            sample_in=np.array(step_capture["sample_in"]).astype(np.float32, copy=False),
            pred_pos=np.array(step_capture["pred_pos"]).astype(np.float32, copy=False),
            pred_neg=np.array(step_capture["pred_neg"]).astype(np.float32, copy=False),
            pred_cfg=np.array(step_capture["pred_cfg"]).astype(np.float32, copy=False),
            x0_pos=np.array(step_capture["x0_pos"]).astype(np.float32, copy=False),
            x0_cfg=np.array(step_capture["x0_cfg"]).astype(np.float32, copy=False),
            std_pos=np.array(step_capture["std_pos"]).astype(np.float32, copy=False),
            std_cfg=np.array(step_capture["std_cfg"]).astype(np.float32, copy=False),
            ratio_raw=np.array(step_capture["ratio_raw"]).astype(np.float32, copy=False),
            std_ratio=np.array(step_capture["std_ratio"]).astype(np.float32, copy=False),
            ratio_effective=np.array(step_capture["ratio_effective"]).astype(np.float32, copy=False),
            x0_rescaled=np.array(step_capture["x0_rescaled"]).astype(np.float32, copy=False),
            x0_after_rescale=np.array(step_capture["x0_after_rescale"]).astype(np.float32, copy=False),
            pred_final=np.array(step_capture["pred_final"]).astype(np.float32, copy=False),
            sample_next=np.array(step_capture["sample_next"]).astype(np.float32, copy=False),
            t=np.array(step_capture["t"]).astype(np.float32, copy=False),
            t_prev=np.array(step_capture["t_prev"]).astype(np.float32, copy=False),
            steps=np.array(n_steps, dtype=np.int32),
            guidance_strength=np.array(7.5, dtype=np.float32),
            guidance_rescale=np.array(0.7, dtype=np.float32),
            guidance_interval=np.array([0.6, 1.0], dtype=np.float32),
            rescale_t=np.array(5.0, dtype=np.float32),
            sigma_min=np.array(1e-5, dtype=np.float32),
            sparse_flow_start_sample_path=np.array(sparse_flow_start_sample_path),
            sparse_flow_start_step_index=np.array(sparse_flow_start_step_index, dtype=np.int32),
            sparse_flow_block_injection_json=np.array(sparse_block_injection_json),
            sparse_flow_layernorm_correction_json=np.array(sparse_flow_layernorm_correction_json),
        )
        print("  Stop after stage: sparse_flow_step", flush=True)
        return

    if args.save_checkpoints and args.stop_after_stage == "sparse_flow_steps":
        from trellmlx.checkpoint import save_checkpoint

        def stack_step(name: str) -> np.ndarray:
            return np.stack(
                [np.array(step[name]).astype(np.float32, copy=False) for step in step_captures],
                axis=0,
            )

        save_checkpoint(
            args.save_checkpoints,
            "sparse_flow_steps",
            noise=np.array(noise).astype(np.float32, copy=False),
            sample_in=stack_step("sample_in"),
            pred_pos=stack_step("pred_pos"),
            pred_neg=stack_step("pred_neg"),
            pred_cfg=stack_step("pred_cfg"),
            x0_pos=stack_step("x0_pos"),
            x0_cfg=stack_step("x0_cfg"),
            std_pos=stack_step("std_pos"),
            std_cfg=stack_step("std_cfg"),
            ratio_raw=stack_step("ratio_raw"),
            std_ratio=stack_step("std_ratio"),
            ratio_effective=stack_step("ratio_effective"),
            x0_rescaled=stack_step("x0_rescaled"),
            x0_after_rescale=stack_step("x0_after_rescale"),
            pred_final=stack_step("pred_final"),
            sample_next=stack_step("sample_next"),
            t=np.array([np.array(step["t"]).item() for step in step_captures], dtype=np.float32),
            t_prev=np.array([np.array(step["t_prev"]).item() for step in step_captures], dtype=np.float32),
            steps=np.array(n_steps, dtype=np.int32),
            guidance_strength=np.array(7.5, dtype=np.float32),
            guidance_rescale=np.array(0.7, dtype=np.float32),
            guidance_interval=np.array([0.6, 1.0], dtype=np.float32),
            rescale_t=np.array(5.0, dtype=np.float32),
            sigma_min=np.array(1e-5, dtype=np.float32),
            sparse_flow_start_sample_path=np.array(sparse_flow_start_sample_path),
            sparse_flow_start_step_index=np.array(sparse_flow_start_step_index, dtype=np.int32),
            sparse_flow_block_injection_json=np.array(sparse_block_injection_json),
            sparse_flow_layernorm_correction_json=np.array(sparse_flow_layernorm_correction_json),
        )
        print("  Stop after stage: sparse_flow_steps", flush=True)
        return

    logits = ss_dec(z_s.astype(mx.float32))
    mx.eval(logits)
    decoded = np.array(logits[0, 0] > 0)

    lr_resolution = 32
    ratio = decoded.shape[0] // lr_resolution
    decoded_ds = decoded.reshape(
        lr_resolution, ratio, lr_resolution, ratio, lr_resolution, ratio
    ).any(axis=(1, 3, 5))
    lr_coords = np.argwhere(decoded_ds)
    print(f"  {len(lr_coords)} sparse voxels at {lr_resolution}³", flush=True)

    lr_coords_4d = np.column_stack([np.zeros(len(lr_coords), dtype=np.int32), lr_coords])
    if args.save_checkpoints and args.stop_after_stage == "sparse_internals":
        from trellmlx.checkpoint import save_checkpoint
        save_checkpoint(
            args.save_checkpoints,
            "sparse_internals",
            z_s=np.array(z_s).astype(np.float32, copy=False),
            logits=np.array(logits).astype(np.float32, copy=False),
            decoded=decoded.astype(np.bool_),
            decoded_ds=decoded_ds.astype(np.bool_),
            coords=lr_coords_4d.astype(np.int32, copy=False),
            lr_resolution=lr_resolution,
            sparse_decoder_resolution=int(decoded.shape[0]),
        )
        print("  Stop after stage: sparse_internals", flush=True)
        return

    cleanup_model(ss_flow, ss_dec)

    if args.save_checkpoints:
        from trellmlx.checkpoint import save_checkpoint
        save_checkpoint(
            args.save_checkpoints,
            "sparse_coords",
            coords=lr_coords_4d.astype(np.int32, copy=False),
            coords_3d=lr_coords.astype(np.int32, copy=False),
            lr_resolution=lr_resolution,
        )
        if args.stop_after_stage == "sparse_coords":
            print("  Stop after stage: sparse_coords", flush=True)
            return

    shape_slat_support_mesh_grid_size = None
    if args.shape_slat_support_sample:
        shape_slat_support_sample_npz = np.load(args.shape_slat_support_sample)
        if "coords" not in shape_slat_support_sample_npz.files:
            raise ValueError("--shape-slat-support-sample NPZ must contain coords array")
        support_coords = np.asarray(shape_slat_support_sample_npz["coords"], dtype=np.int32)
        if support_coords.ndim != 2 or support_coords.shape[1] != 4:
            raise ValueError(
                "--shape-slat-support-sample coords must have shape [N, 4], "
                f"got {support_coords.shape}"
            )
        if support_coords.shape[0] == 0:
            raise ValueError("--shape-slat-support-sample coords must not be empty")
        support_meta_path = Path(args.shape_slat_support_sample).with_suffix(".json")
        if support_meta_path.exists():
            with support_meta_path.open() as f:
                support_meta = json.load(f)
            shape_slat_support_mesh_grid_size = int(support_meta.get("mesh_grid_size", 0)) or None
        lr_coords_4d = support_coords.astype(np.int32, copy=False)
        lr_coords = lr_coords_4d[:, 1:4].astype(np.int32, copy=False)
        print(
            "  Shape SLat support replay: "
            f"{args.shape_slat_support_sample} ({len(lr_coords):,} coords)",
            flush=True,
        )

    # === Stage 2a: LR Shape Latent ===
    print("\n=== Stage 2a: LR Shape Latent ===", flush=True)
    shape_flow_attention_route = _configure_shape_flow_attention_route(args)
    from trellmlx.models.slat_flow import SLatFlowModel

    # Sampler params from pipeline.json
    SHAPE_SAMPLER = dict(steps=n_steps, guidance_strength=7.5, guidance_rescale=0.5,
                         guidance_interval=(0.6, 1.0), rescale_t=3.0)
    TEX_SAMPLER = dict(steps=n_steps, guidance_strength=1.0, guidance_rescale=0.0,
                       guidance_interval=(0.6, 0.9), rescale_t=3.0)

    lr_slat_flow = SLatFlowModel.for_shape()
    load_weights(lr_slat_flow, HF_4B + "slat_flow_img2shape_dit_1_3B_512_bf16.safetensors", verbose=False)
    if args.quantize:
        quantize_model(lr_slat_flow, bits=args.quantize)
    if args.compile:
        lr_slat_flow.compile()

    shape_block_injection = None
    shape_block_injection_json = ""
    if args.shape_flow_block_injection_manifest:
        from trellmlx.shape_block_injection import load_shape_block_injection_manifest

        shape_block_injection = load_shape_block_injection_manifest(
            args.shape_flow_block_injection_manifest,
        )
        shape_block_injection_json = json.dumps(
            shape_block_injection.report_identity(),
            sort_keys=True,
        )
    elif args.shape_flow_block_injection_trace:
        from trellmlx.shape_block_injection import load_shape_block_injection

        shape_block_injection = load_shape_block_injection(
            args.shape_flow_block_injection_trace,
            branch=args.shape_flow_block_injection_branch,
            step_index=args.shape_flow_block_injection_step_index,
            block_index=args.shape_flow_block_injection_block_index,
            stage=args.shape_flow_block_injection_stage,
            array_key=args.shape_flow_block_injection_array_key,
            source_delta_scale=args.shape_flow_block_injection_scale,
        )
        shape_block_injection_json = json.dumps(
            shape_block_injection.report_identity(),
            sort_keys=True,
        )

    N_lr = len(lr_coords)
    if args.shape_flow_noise_sample:
        shape_flow_noise_sample_npz = np.load(args.shape_flow_noise_sample)
        if "coords" not in shape_flow_noise_sample_npz.files:
            raise ValueError("--shape-flow-noise-sample NPZ must contain coords array")
        noise_key = "noise" if "noise" in shape_flow_noise_sample_npz.files else "sample_feats"
        if noise_key not in shape_flow_noise_sample_npz.files:
            raise ValueError("--shape-flow-noise-sample NPZ must contain noise or sample_feats array")
        noise_coords = np.asarray(shape_flow_noise_sample_npz["coords"], dtype=np.int32)
        if noise_coords.shape != lr_coords_4d.shape or not np.array_equal(noise_coords, lr_coords_4d):
            raise ValueError(
                "shape flow noise sample coords do not exactly match shape support coords: "
                f"{noise_coords.shape} vs {lr_coords_4d.shape}"
            )
        noise_np = np.asarray(shape_flow_noise_sample_npz[noise_key], dtype=np.float32)
        if noise_np.ndim != 2 or noise_np.shape[0] != N_lr:
            raise ValueError(
                "--shape-flow-noise-sample noise/sample_feats must have shape [N, C], "
                f"got {noise_np.shape} for N={N_lr}"
            )
        lr_noise = mx.array(noise_np).astype(mx.float32)
        print(f"  Shape flow noise replay: {args.shape_flow_noise_sample} {lr_noise.shape}", flush=True)
    elif shared_noise is not None and args.shared_noise_sparse_only:
        print("  Ignoring shared slat_noise_pool; using route-local random shape SLat noise", flush=True)
        lr_noise = mx.random.normal((N_lr, 32))
    elif shared_noise is not None and "slat_noise_pool" in shared_noise:
        shared_shape_noise = shared_noise["slat_noise_pool"]
        if shared_shape_noise.shape[0] == 1 and N_lr != 1:
            shared_shape_noise = np.broadcast_to(shared_shape_noise, (N_lr, shared_shape_noise.shape[1]))
        elif shared_shape_noise.shape[0] < N_lr:
            raise ValueError(
                "shared slat_noise_pool has fewer rows than LR shape coords: "
                f"{shared_shape_noise.shape[0]} < {N_lr}"
            )
        lr_noise = mx.array(shared_shape_noise[:N_lr]).astype(mx.float32)
        print(f"  Shared shape SLat noise: {args.shared_noise} {lr_noise.shape}", flush=True)
    else:
        lr_noise = mx.random.normal((N_lr, 32))

    t0 = time.perf_counter()
    shape_step_capture = {} if args.stop_after_stage == "shape_flow_step" else None
    shape_step_captures = [] if args.stop_after_stage == "shape_flow_steps" else None
    if args.stop_after_stage == "shape_flow_block_trace":
        if shape_flow_attention_route is None:
            raise RuntimeError("shape-flow attention route was not configured")
        shape_trace_step_index = args.shape_flow_trace_step_index
        if shape_trace_step_index < 0 or shape_trace_step_index >= n_steps:
            raise ValueError(
                f"--shape-flow-trace-step-index must be in [0, {n_steps - 1}], "
                f"got {shape_trace_step_index}"
            )
        shape_trace_block_index = args.shape_flow_trace_block_index
        shape_cond = cond_tgt if vs3d_mode else cond
        shape_neg_cond = neg_cond
        shape_pos_kv_cache = lr_slat_flow.build_cross_kv_cache(shape_cond)
        shape_neg_kv_cache = lr_slat_flow.build_cross_kv_cache(shape_neg_cond)

        def shape_schedule_trace_t():
            t_seq = np.linspace(1, 0, n_steps + 1)
            t_seq = (
                SHAPE_SAMPLER["rescale_t"]
                * t_seq
                / (1 + (SHAPE_SAMPLER["rescale_t"] - 1) * t_seq)
            )
            return float(t_seq[shape_trace_step_index])

        if shape_trace_step_index == 0:
            shape_trace_sample = lr_noise
            shape_trace_t = shape_schedule_trace_t()
        else:
            shape_trace_steps = []
            _ = flow_euler_sample(
                lr_slat_flow,
                lr_noise,
                shape_cond,
                shape_neg_cond,
                verbose=False,
                coords=mx.array(lr_coords),
                capture_steps=shape_trace_steps,
                **SHAPE_SAMPLER,
            )
            if shape_trace_step_index >= len(shape_trace_steps):
                raise ValueError(
                    f"shape flow captured {len(shape_trace_steps)} steps, cannot trace step "
                    f"{shape_trace_step_index}"
                )
            shape_trace_sample = shape_trace_steps[shape_trace_step_index]["sample_in"]
            shape_trace_t = float(np.array(shape_trace_steps[shape_trace_step_index]["t"]))
            shape_trace_t = shape_trace_t / 1000.0 if shape_trace_t > 1.0 else shape_trace_t

        shape_t_tensor = mx.array([1000.0 * shape_trace_t], dtype=mx.float32)
        pos_trace = lr_slat_flow.trace_block(
            shape_trace_sample,
            shape_t_tensor,
            shape_cond,
            coords=mx.array(lr_coords),
            block_index=shape_trace_block_index,
            cross_kv_cache=shape_pos_kv_cache,
            shape_block_injection=(
                shape_block_injection
                if shape_block_injection is not None
                and shape_block_injection.applies(step_index=shape_trace_step_index, branch="pos")
                else None
            ),
            shape_block_injection_branch="pos",
        )
        neg_trace = lr_slat_flow.trace_block(
            shape_trace_sample,
            shape_t_tensor,
            shape_neg_cond,
            coords=mx.array(lr_coords),
            block_index=shape_trace_block_index,
            cross_kv_cache=shape_neg_kv_cache,
            shape_block_injection=(
                shape_block_injection
                if shape_block_injection is not None
                and shape_block_injection.applies(step_index=shape_trace_step_index, branch="neg")
                else None
            ),
            shape_block_injection_branch="neg",
        )

        def trace_np(value):
            return np.array(value.astype(mx.float32))[None].astype(np.float32, copy=False)

        from trellmlx.checkpoint import save_checkpoint

        trace_payload = {
            f"pos_{name}": trace_np(value)
            for name, value in pos_trace.items()
        }
        trace_payload.update(
            {
                f"neg_{name}": trace_np(value)
                for name, value in neg_trace.items()
            }
        )
        requested_trace_keys = _parse_shape_flow_trace_keys(args.shape_flow_trace_keys)
        trace_payload, effective_trace_keys = _select_shape_flow_trace_payload(
            trace_payload, requested_trace_keys
        )
        save_checkpoint(
            args.save_checkpoints,
            "shape_flow_block_trace",
            **trace_payload,
            coords=lr_coords_4d.astype(np.int32, copy=False),
            coords_3d=lr_coords.astype(np.int32, copy=False),
            trace_block_index=np.array(shape_trace_block_index, dtype=np.int32),
            shape_flow_trace_step_index=np.array(shape_trace_step_index, dtype=np.int32),
            shape_flow_trace_requested_keys=np.array(requested_trace_keys, dtype=str),
            shape_flow_trace_selected_keys=np.array(effective_trace_keys, dtype=str),
            **{
                field: np.array(value)
                for field, value in shape_flow_attention_route.items()
            },
            shape_slat_support_sample_path=np.array(args.shape_slat_support_sample or ""),
            t=np.array(1000.0 * shape_trace_t, dtype=np.float32),
            steps=np.array(n_steps, dtype=np.int32),
            guidance_strength=np.array(SHAPE_SAMPLER["guidance_strength"], dtype=np.float32),
            guidance_rescale=np.array(SHAPE_SAMPLER["guidance_rescale"], dtype=np.float32),
            guidance_interval=np.array(SHAPE_SAMPLER["guidance_interval"], dtype=np.float32),
            rescale_t=np.array(SHAPE_SAMPLER["rescale_t"], dtype=np.float32),
            shape_flow_block_injection_json=np.array(shape_block_injection_json),
            shape_flow_layernorm_backend=np.array(get_shape_flow_layernorm_backend()),
            qk_norm_backend=np.array(get_qk_norm_backend()),
            rope_backend=np.array(get_rope_backend()),
            shape_flow_turing_rsqrt_lut_sha256=np.array(
                get_shape_flow_turing_rsqrt_lut_sha256() or ""
            ),
            shape_flow_turing_rope_phase_lut_sha256=np.array(
                get_turing_phase_lut_sha256() or ""
            ),
        )
        print(
            f"  Stop after stage: shape_flow_block_trace step={shape_trace_step_index} "
            f"block={shape_trace_block_index}",
            flush=True,
        )
        return

    lr_slat = flow_euler_sample(
        lr_slat_flow, lr_noise, cond_tgt if vs3d_mode else cond, neg_cond,
        verbose=False,
        coords=mx.array(lr_coords),
        capture_first_step=shape_step_capture,
        capture_steps=shape_step_captures,
        stop_after_first_step=args.stop_after_stage == "shape_flow_step",
        shape_block_injection=shape_block_injection,
        **SHAPE_SAMPLER,
    )
    mx.eval(lr_slat)
    print(f"  Sampled: {time.perf_counter()-t0:.1f}s ({N_lr} tokens)", flush=True)

    if args.save_checkpoints and args.stop_after_stage == "shape_flow_step":
        from trellmlx.checkpoint import save_checkpoint

        save_checkpoint(
            args.save_checkpoints,
            "shape_flow_step",
            noise=np.array(lr_noise).astype(np.float32, copy=False),
            sample_feats=np.array(lr_noise).astype(np.float32, copy=False),
            coords=lr_coords_4d.astype(np.int32, copy=False),
            coords_3d=lr_coords.astype(np.int32, copy=False),
            pred_pos=np.array(shape_step_capture["pred_pos"]).astype(np.float32, copy=False),
            pred_neg=np.array(shape_step_capture["pred_neg"]).astype(np.float32, copy=False),
            pred_cfg=np.array(shape_step_capture["pred_cfg"]).astype(np.float32, copy=False),
            x0_pos=np.array(shape_step_capture["x0_pos"]).astype(np.float32, copy=False),
            x0_cfg=np.array(shape_step_capture["x0_cfg"]).astype(np.float32, copy=False),
            std_pos=np.array(shape_step_capture["std_pos"]).astype(np.float32, copy=False),
            std_cfg=np.array(shape_step_capture["std_cfg"]).astype(np.float32, copy=False),
            ratio_raw=np.array(shape_step_capture["ratio_raw"]).astype(np.float32, copy=False),
            std_ratio=np.array(shape_step_capture["std_ratio"]).astype(np.float32, copy=False),
            ratio_effective=np.array(shape_step_capture["ratio_effective"]).astype(np.float32, copy=False),
            x0_rescaled=np.array(shape_step_capture["x0_rescaled"]).astype(np.float32, copy=False),
            x0_after_rescale=np.array(shape_step_capture["x0_after_rescale"]).astype(np.float32, copy=False),
            pred_final=np.array(shape_step_capture["pred_final"]).astype(np.float32, copy=False),
            pred_v_feats=np.array(shape_step_capture["pred_final"]).astype(np.float32, copy=False),
            sample_next=np.array(shape_step_capture["sample_next"]).astype(np.float32, copy=False),
            t=np.array(shape_step_capture["t"]).astype(np.float32, copy=False),
            t_prev=np.array(shape_step_capture["t_prev"]).astype(np.float32, copy=False),
            steps=np.array(n_steps, dtype=np.int32),
            guidance_strength=np.array(SHAPE_SAMPLER["guidance_strength"], dtype=np.float32),
            guidance_rescale=np.array(SHAPE_SAMPLER["guidance_rescale"], dtype=np.float32),
            guidance_interval=np.array(SHAPE_SAMPLER["guidance_interval"], dtype=np.float32),
            rescale_t=np.array(SHAPE_SAMPLER["rescale_t"], dtype=np.float32),
            shape_flow_block_injection_json=np.array(shape_block_injection_json),
            shape_flow_layernorm_backend=np.array(get_shape_flow_layernorm_backend()),
            qk_norm_backend=np.array(get_qk_norm_backend()),
            rope_backend=np.array(get_rope_backend()),
            shape_flow_turing_rsqrt_lut_sha256=np.array(
                get_shape_flow_turing_rsqrt_lut_sha256() or ""
            ),
            shape_flow_turing_rope_phase_lut_sha256=np.array(
                get_turing_phase_lut_sha256() or ""
            ),
        )
        print("  Stop after stage: shape_flow_step", flush=True)
        return

    if args.save_checkpoints and args.stop_after_stage == "shape_flow_steps":
        from trellmlx.checkpoint import save_checkpoint

        def shape_stack_step(name: str) -> np.ndarray:
            return np.stack(
                [np.array(step[name]).astype(np.float32, copy=False) for step in shape_step_captures],
                axis=0,
            )

        save_checkpoint(
            args.save_checkpoints,
            "shape_flow_steps",
            noise=np.array(lr_noise).astype(np.float32, copy=False),
            sample_feats=np.array(lr_noise).astype(np.float32, copy=False),
            coords=lr_coords_4d.astype(np.int32, copy=False),
            coords_3d=lr_coords.astype(np.int32, copy=False),
            sample_in=shape_stack_step("sample_in"),
            pred_pos=shape_stack_step("pred_pos"),
            pred_neg=shape_stack_step("pred_neg"),
            pred_cfg=shape_stack_step("pred_cfg"),
            x0_pos=shape_stack_step("x0_pos"),
            x0_cfg=shape_stack_step("x0_cfg"),
            std_pos=shape_stack_step("std_pos"),
            std_cfg=shape_stack_step("std_cfg"),
            ratio_raw=shape_stack_step("ratio_raw"),
            std_ratio=shape_stack_step("std_ratio"),
            ratio_effective=shape_stack_step("ratio_effective"),
            x0_rescaled=shape_stack_step("x0_rescaled"),
            x0_after_rescale=shape_stack_step("x0_after_rescale"),
            pred_final=shape_stack_step("pred_final"),
            pred_v_feats=shape_stack_step("pred_final"),
            sample_next=shape_stack_step("sample_next"),
            t=np.array(
                [np.array(step["t"]).item() for step in shape_step_captures],
                dtype=np.float32,
            ),
            t_prev=np.array(
                [np.array(step["t_prev"]).item() for step in shape_step_captures],
                dtype=np.float32,
            ),
            steps=np.array(n_steps, dtype=np.int32),
            guidance_strength=np.array(SHAPE_SAMPLER["guidance_strength"], dtype=np.float32),
            guidance_rescale=np.array(SHAPE_SAMPLER["guidance_rescale"], dtype=np.float32),
            guidance_interval=np.array(SHAPE_SAMPLER["guidance_interval"], dtype=np.float32),
            rescale_t=np.array(SHAPE_SAMPLER["rescale_t"], dtype=np.float32),
            sigma_min=np.array(1e-5, dtype=np.float32),
            shape_flow_block_injection_json=np.array(shape_block_injection_json),
            shape_flow_layernorm_backend=np.array(get_shape_flow_layernorm_backend()),
            qk_norm_backend=np.array(get_qk_norm_backend()),
            rope_backend=np.array(get_rope_backend()),
            shape_flow_turing_rsqrt_lut_sha256=np.array(
                get_shape_flow_turing_rsqrt_lut_sha256() or ""
            ),
            shape_flow_turing_rope_phase_lut_sha256=np.array(
                get_turing_phase_lut_sha256() or ""
            ),
        )
        print("  Stop after stage: shape_flow_steps", flush=True)
        return

    lr_slat = _denormalize_slat(lr_slat)
    mx.eval(lr_slat)

    cleanup_model(lr_slat_flow)
    del lr_slat_flow
    gc.collect()

    from trellmlx.models.shape_slat_decoder import SLatDecoder

    if args.no_cascade:
        # No-cascade mode: use LR SLat directly for decoding.
        # Skips upsample + HR SLat pass. Much faster but lower quality.
        hr_slat = lr_slat
        quant_coords = lr_coords_4d
        num_tokens = N_lr
        hr_resolution = shape_slat_support_mesh_grid_size or lr_resolution * 16
        hr_coords_3d = quant_coords[:, 1:4]
        print(f"  No-cascade: using LR SLat directly ({num_tokens:,} tokens, "
              f"grid_size={hr_resolution})", flush=True)
    else:
        # === Stage 2b: Upsample to get HR coordinates ===
        print("\n=== Stage 2b: Upsample → HR coordinates ===", flush=True)

        decoder = SLatDecoder(out_channels=7, pred_subdiv=True)
        load_weights(decoder, HF_4B + "shape_dec_next_dc_f16c32_fp16.safetensors", verbose=False)

        t0 = time.perf_counter()
        hr_coords_raw = decoder.upsample(lr_slat, mx.array(lr_coords_4d), upsample_times=4)
        mx.eval(hr_coords_raw)
        print(f"  Upsampled: {time.perf_counter()-t0:.1f}s ({hr_coords_raw.shape[0]:,} voxels)", flush=True)

        decoder_output_res = lr_resolution * 16  # 32 * 16 = 512
        hr_resolution = args.resolution
        hr_coords_np = np.array(hr_coords_raw)
        while True:
            quant_coords = _requantize_coords(hr_coords_np, decoder_output_res, hr_resolution)
            num_tokens = len(quant_coords)
            if num_tokens < args.max_tokens or hr_resolution == 1024:
                if hr_resolution != args.resolution:
                    print(f"  Resolution reduced to {hr_resolution} ({num_tokens:,} tokens)", flush=True)
                break
            hr_resolution -= 128

        hr_coords_3d = quant_coords[:, 1:4]
        print(f"  HR coords: {num_tokens:,} tokens at res {hr_resolution}, "
              f"coord range [{hr_coords_3d.min()}, {hr_coords_3d.max()}]", flush=True)

        cleanup_model(decoder)
        del decoder
        gc.collect()

        # === Stage 2c: HR Shape Latent (second SLat pass) ===
        print("\n=== Stage 2c: HR Shape Latent ===", flush=True)

        hr_slat_flow = SLatFlowModel.for_shape()
        load_weights(hr_slat_flow, HF_4B + "slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors", verbose=False)
        if args.quantize:
            quantize_model(hr_slat_flow, bits=args.quantize)
        if args.compile:
            hr_slat_flow.compile()

        hr_noise = mx.random.normal((num_tokens, 32))

        t0 = time.perf_counter()
        hr_slat = flow_euler_sample(
            hr_slat_flow, hr_noise, cond_tgt if vs3d_mode else cond, neg_cond,
            verbose=False,
            coords=mx.array(hr_coords_3d),
            **SHAPE_SAMPLER,
        )
        mx.eval(hr_slat)
        print(f"  Sampled: {time.perf_counter()-t0:.1f}s ({num_tokens:,} tokens)", flush=True)

        hr_slat = _denormalize_slat(hr_slat)
        mx.eval(hr_slat)

        cleanup_model(hr_slat_flow)
        del hr_slat_flow
        gc.collect()

    # Keep hr_slat — needed for texture conditioning
    if args.save_checkpoints:
        from trellmlx.checkpoint import save_checkpoint
        save_checkpoint(
            args.save_checkpoints,
            "shape_slat",
            feats=np.array(hr_slat).astype(np.float32, copy=False),
            coords=quant_coords.astype(np.int32, copy=False),
            coords_3d=hr_coords_3d.astype(np.int32, copy=False),
            mesh_grid_size=hr_resolution,
            cascade=not args.no_cascade,
        )
        if args.stop_after_stage == "shape_slat":
            print("  Stop after stage: shape_slat", flush=True)
            return

    # === Stage 3: Shape Decode ===
    print("\n=== Stage 3: Decode Shape ===", flush=True)

    shape_decoder = SLatDecoder(out_channels=7, pred_subdiv=True)
    load_weights(shape_decoder, HF_4B + "shape_dec_next_dc_f16c32_fp16.safetensors", verbose=False)

    t0 = time.perf_counter()
    dec_out, dec_coords, shape_subs = shape_decoder(
        hr_slat, mx.array(quant_coords), return_subs=True,
    )
    mx.eval(dec_out)
    print(f"  Decoded: {time.perf_counter()-t0:.1f}s ({dec_out.shape[0]:,} voxels)", flush=True)
    print(f"  Subdivision masks: {len(shape_subs)} levels", flush=True)

    cleanup_model(shape_decoder)
    del shape_decoder
    gc.collect()

    # === Mesh Extraction ===
    print("\n=== Mesh Extraction ===", flush=True)
    from trellmlx.mesh_extract import decoder_output_to_mesh

    dec_coords_np = np.array(dec_coords)
    dec_feats_np = np.array(dec_out)

    # The decoder output coords span [0, hr_resolution). The decoder input
    # coords are at hr_resolution//16 (from requantization), and 4 upsamples
    # (2^4=16x) bring them back to hr_resolution scale.
    # grid_size = hr_resolution gives correct [-0.5, 0.5] world-space scaling.
    mesh_grid_size = hr_resolution
    if args.save_checkpoints:
        from trellmlx.checkpoint import save_checkpoint
        save_checkpoint(
            args.save_checkpoints,
            "decoder_output",
            feats=dec_feats_np.astype(np.float32, copy=False),
            coords=dec_coords_np.astype(np.int32, copy=False),
            shape_subs=[np.array(mask) for mask in shape_subs],
            mesh_grid_size=mesh_grid_size,
        )
        if args.stop_after_stage == "decoder_output":
            print("  Stop after stage: decoder_output", flush=True)
            return
    print(f"  {dec_coords_np.shape[0]:,} voxels, coord range "
          f"[{dec_coords_np[:,1:].min()}, {dec_coords_np[:,1:].max()}], "
          f"grid_size={mesh_grid_size}", flush=True)

    t0 = time.perf_counter()
    vertices, faces = decoder_output_to_mesh(
        dec_feats_np,
        dec_coords_np,
        resolution=mesh_grid_size,
    )
    print(f"  Extracted: {time.perf_counter()-t0:.1f}s", flush=True)
    print(f"  {len(vertices):,} vertices, {len(faces):,} faces", flush=True)

    # Save raw mesh checkpoint (before cleanup/simplification)
    if args.save_checkpoints:
        from trellmlx.checkpoint import save_checkpoint
        save_checkpoint(args.save_checkpoints, "mesh_raw",
                        vertices=vertices, faces=faces,
                        mesh_grid_size=mesh_grid_size)
        maybe_checkpoint_yield(
            stop_file=args.checkpoint_stop_file,
            checkpoint_dir=args.save_checkpoints,
            completed_stage="mesh_raw",
            next_stage="texture",
            output_path=args.output,
            resume_supported=False,
            resume_blocker="mesh_raw checkpoint exists, but mesh-only resume is not implemented",
        )
        if args.stop_after_stage == "mesh_raw":
            print("  Stop after stage: mesh_raw", flush=True)
            return

    vertices, faces = _cleanup_and_simplify_mesh(
        vertices,
        faces,
        target_faces=args.target_faces,
        no_cleanup=args.no_cleanup,
        keep_largest=args.keep_largest,
        simplify_first=args.simplify_first,
        reference_cleanup=args.reference_cleanup,
        qem_simplify=args.qem_simplify,
        qem_backend=args.qem_backend,
        source_native_source_root=args.source_native_source_root,
        source_native_python=args.source_native_python,
    )
    if args.save_checkpoints:
        from trellmlx.checkpoint import save_checkpoint
        save_checkpoint(args.save_checkpoints, "mesh_clean",
                        vertices=vertices, faces=faces,
                        mesh_grid_size=mesh_grid_size)
        maybe_checkpoint_yield(
            stop_file=args.checkpoint_stop_file,
            checkpoint_dir=args.save_checkpoints,
            completed_stage="mesh_clean",
            next_stage="texture",
            output_path=args.output,
            resume_supported=False,
            resume_blocker="mesh_clean checkpoint exists, but texture checkpoint is still required for resume",
        )
        if args.stop_after_stage == "mesh_clean":
            print("  Stop after stage: mesh_clean", flush=True)
            return

    # === Stage 4: Texture SLat ===
    print("\n=== Stage 4: Texture SLat ===", flush=True)

    # Load texture flow model (same architecture, in_channels=64)
    tex_flow = SLatFlowModel.for_texture()
    load_weights(tex_flow, HF_4B + "slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors", verbose=False)
    if args.quantize:
        quantize_model(tex_flow, bits=args.quantize)
    if args.compile:
        tex_flow.compile()

    # Re-normalize shape SLat for texture conditioning
    shape_cond = _normalize_slat(hr_slat)
    mx.eval(shape_cond)

    # Sample texture latent: noise (32ch) conditioned on shape SLat (32ch)
    # Texture uses guidance_strength=1.0 (no CFG — single forward pass per step)
    tex_noise = mx.random.normal((num_tokens, 32))
    t0 = time.perf_counter()
    tex_slat = flow_euler_sample(
        tex_flow, tex_noise, cond_tgt if vs3d_mode else cond, neg_cond,
        verbose=False,
        coords=mx.array(hr_coords_3d),
        concat_cond=shape_cond,
        **TEX_SAMPLER,
    )
    mx.eval(tex_slat)
    print(f"  Sampled: {time.perf_counter()-t0:.1f}s ({num_tokens:,} tokens)", flush=True)

    tex_slat = _denormalize_slat(tex_slat, mean=TEX_SLAT_MEAN, std=TEX_SLAT_STD)
    mx.eval(tex_slat)

    cleanup_model(tex_flow)
    del tex_flow
    gc.collect()

    # === Stage 5: Texture Decode ===
    print("\n=== Stage 5: Texture Decode ===", flush=True)

    tex_decoder = SLatDecoder(out_channels=6, pred_subdiv=False)
    load_weights(tex_decoder, HF_4B + "tex_dec_next_dc_f16c32_fp16.safetensors", verbose=False)

    t0 = time.perf_counter()
    tex_out, tex_coords = tex_decoder(
        tex_slat, mx.array(quant_coords), guide_subs=shape_subs,
    )
    mx.eval(tex_out)
    # Output transform: * 0.5 + 0.5 to map to [0, 1]
    tex_out = tex_out * 0.5 + 0.5
    mx.eval(tex_out)
    print(f"  Decoded: {time.perf_counter()-t0:.1f}s ({tex_out.shape[0]:,} voxels, {tex_out.shape[1]} channels)", flush=True)

    tex_np = np.array(tex_out)
    print(f"  PBR attrs: RGB [{tex_np[:,:3].min():.2f}, {tex_np[:,:3].max():.2f}] "
          f"metallic [{tex_np[:,3].min():.2f}, {tex_np[:,3].max():.2f}] "
          f"roughness [{tex_np[:,4].min():.2f}, {tex_np[:,4].max():.2f}]", flush=True)

    cleanup_model(tex_decoder)
    del tex_decoder
    gc.collect()

    # Save texture checkpoint (before baking)
    if args.save_checkpoints:
        from trellmlx.checkpoint import save_checkpoint
        save_checkpoint(args.save_checkpoints, "texture",
                        tex_np=tex_np,
                        tex_coords_spatial=np.array(tex_coords)[:, 1:4],
                        mesh_grid_size=mesh_grid_size)
        maybe_checkpoint_yield(
            stop_file=args.checkpoint_stop_file,
            checkpoint_dir=args.save_checkpoints,
            completed_stage="texture",
            next_stage="texture_bake",
            output_path=args.output,
        )
        if args.stop_after_stage == "mesh_uv":
            print("  Stop after stage: mesh_uv", flush=True)
            return

    # === Stage 6: Texture Baking ===
    print("\n=== Stage 6: Texture Baking ===", flush=True)
    from trellmlx.texture_bake import uv_unwrap, uv_unwrap_cube, bake_texture

    tex_coords_spatial = np.array(tex_coords)[:, 1:4]  # drop batch dim

    # UV unwrap
    unwrap_fn, method_name = _select_uv_method(args.uv_method, vertices, faces)
    t0 = time.perf_counter()
    uv_verts, uv_faces, uvs, vmapping = unwrap_fn(vertices, faces)
    print(f"  UV unwrap ({method_name}): {len(uv_verts):,}V {len(uv_faces):,}F "
          f"({time.perf_counter()-t0:.1f}s)", flush=True)
    if args.save_checkpoints:
        from trellmlx.checkpoint import save_checkpoint
        save_checkpoint(args.save_checkpoints, "mesh_uv",
                        vertices=uv_verts, faces=uv_faces,
                        uvs=uvs, vmapping=vmapping,
                        mesh_grid_size=mesh_grid_size,
                        uv_method=method_name)
        maybe_checkpoint_yield(
            stop_file=args.checkpoint_stop_file,
            checkpoint_dir=args.save_checkpoints,
            completed_stage="mesh_uv",
            next_stage="texture_bake",
            output_path=args.output,
        )

    # Bake PBR textures
    base_color, metallic_roughness, alpha_mode = bake_texture(
        uv_verts, uv_faces, uvs, vmapping,
        tex_coords_spatial, tex_np, mesh_grid_size,
        texture_size=args.texture_size,
        backend=args.texture_backend,
    )

    # === Export ===
    import trimesh
    from trimesh.visual.material import PBRMaterial
    from PIL import Image

    if len(uv_verts) > 0 and len(uv_faces) > 0:
        # Swap Y and Z axes, invert Y for GLB compatibility (matches reference)
        export_verts = uv_verts.copy()
        export_verts[:, 1], export_verts[:, 2] = uv_verts[:, 2].copy(), -uv_verts[:, 1].copy()

        # Flip UV V-coordinate for GLB
        export_uvs = uvs.copy()
        export_uvs[:, 1] = 1 - export_uvs[:, 1]

        # Build vertex normals
        mesh = trimesh.Trimesh(vertices=export_verts, faces=uv_faces, process=False)
        normals = mesh.vertex_normals

        material = PBRMaterial(
            baseColorTexture=Image.fromarray(base_color),
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
            metallicRoughnessTexture=Image.fromarray(metallic_roughness),
            metallicFactor=1.0,
            roughnessFactor=1.0,
            alphaMode=alpha_mode,
            doubleSided=True,
        )

        textured_mesh = trimesh.Trimesh(
            vertices=export_verts,
            faces=uv_faces,
            vertex_normals=normals,
            process=False,
            visual=trimesh.visual.TextureVisuals(uv=export_uvs, material=material),
        )
        textured_mesh.export(args.output)
        print(f"\n  Saved: {args.output} ({os.path.getsize(args.output)/1e6:.1f}MB)", flush=True)
    else:
        print("  WARNING: Empty mesh!", flush=True)

    total = time.perf_counter() - t_total
    print(f"\nTotal: {total:.1f}s", flush=True)
    from trellmlx.modules.sparse_conv import clear_neighbor_map_cache
    clear_neighbor_map_cache()
    cleanup()


def _extract_image_features(image_path, resolution=512):
    """Extract DINOv3 image features, preferring native MLX path."""
    try:
        from trellmlx.models.dinov3 import extract_features
        features = extract_features(image_path, image_size=resolution)
        print(f"  Features: {features.shape} (MLX)", flush=True)
        return features
    except Exception as e:
        print(f"  MLX DINOv3 failed ({e}), trying PyTorch...", flush=True)

    try:
        import torch, sys
        sys.path.insert(0, os.path.expanduser("~/dev/trellis-mac/TRELLIS.2"))
        from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor
        from PIL import Image
        extractor = DinoV3FeatureExtractor("facebook/dinov3-vitl16-pretrain-lvd1689m", image_size=resolution)
        extractor.to("cpu")
        img = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            features = extractor([img])
        print(f"  Features: {features.shape} (PyTorch)", flush=True)
        return mx.array(features.numpy())
    except Exception as e:
        raise RuntimeError(
            f"Image feature extraction failed for {image_path!r}. "
            "Download the native DINOv3 weights with "
            "`hf download facebook/dinov3-vitl16-pretrain-lvd1689m`, "
            "or omit --image for random conditioning."
        ) from e


if __name__ == "__main__":
    main()
