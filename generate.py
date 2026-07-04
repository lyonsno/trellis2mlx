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
import os
import time

import mlx.core as mx
import numpy as np

from trellmlx.checkpoint_yield import maybe_checkpoint_yield

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


def _mesh_coord_space():
    from trellmlx.texture_bake import TRELLIS_WORLD_COORD_SPACE

    return TRELLIS_WORLD_COORD_SPACE


def _glb_export_vertices(vertices):
    export_verts = vertices.copy()
    export_verts[:, 1], export_verts[:, 2] = vertices[:, 2].copy(), -vertices[:, 1].copy()
    return export_verts


def _orient_uv_faces_for_export(uv_verts, uv_faces, args):
    if args.no_uv_visible_orient:
        print("  UV visible island orientation: skipped (--no-uv-visible-orient)", flush=True)
        return uv_faces

    from trellmlx.mesh_cleanup import orient_uv_islands_by_visible_exterior

    export_verts = _glb_export_vertices(uv_verts)
    t0 = time.perf_counter()
    _, oriented_faces = orient_uv_islands_by_visible_exterior(
        export_verts,
        uv_faces,
        image_size=args.uv_visible_orient_size,
        verbose=True,
    )
    changed = int((oriented_faces != uv_faces).any(axis=1).sum())
    print(
        f"  UV visible island orientation: checked at {args.uv_visible_orient_size}px, "
        f"changed {changed:,} faces ({time.perf_counter()-t0:.1f}s)",
        flush=True,
    )
    return oriented_faces


def _validate_mesh_checkpoint_vertices(vertices, *, mesh_grid_size, coord_space=None, stage="mesh"):
    expected = _mesh_coord_space()
    if coord_space is not None and coord_space != expected:
        raise ValueError(
            f"{stage} checkpoint has unsupported mesh_coord_space={coord_space!r}; "
            f"expected {expected!r}"
        )

    from trellmlx.texture_bake import validate_trellis_world_positions

    return validate_trellis_world_positions(
        vertices,
        mesh_grid_size,
        context=f"{stage} checkpoint vertices",
    )


def _apply_voxel_remesh_if_requested(
    vertices,
    faces,
    *,
    pitch,
    no_cleanup,
    keep_largest=False,
    cleanup_mesh=None,
    voxel_remesh=None,
    log=print,
):
    if not pitch:
        return vertices, faces

    if voxel_remesh is None:
        from trellmlx.mesh_remesh import voxel_remesh

    t0 = time.perf_counter()
    remesh_vertices, remesh_faces = voxel_remesh(vertices, faces, pitch=pitch)
    log(
        f"  Voxel remesh (pitch={pitch:g}): "
        f"{len(vertices):,}V {len(faces):,}F → "
        f"{len(remesh_vertices):,}V {len(remesh_faces):,}F "
        f"({time.perf_counter()-t0:.1f}s)",
        flush=True,
    )
    if not no_cleanup:
        if cleanup_mesh is None:
            from trellmlx.mesh_cleanup import cleanup_mesh
        t0 = time.perf_counter()
        remesh_vertices, remesh_faces = cleanup_mesh(
            remesh_vertices,
            remesh_faces,
            keep_largest=keep_largest,
            do_fix_normals=True,
            verbose=False,
        )
        log(f"  Voxel remesh cleanup final: {time.perf_counter()-t0:.1f}s", flush=True)
    return remesh_vertices, remesh_faces


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
    cleanup_mesh=None,
    fill_holes=None,
    simplify=None,
    log=print,
):
    """Run mesh cleanup and multi-pass simplification.

    With simplify_first=True, simplifies the raw mesh before cleanup.
    Much faster on large meshes (cleanup on 200K faces vs 6M faces).
    """
    def final_cleanup(vertices, faces):
        if no_cleanup:
            return vertices, faces
        t0 = time.perf_counter()
        vertices, faces = cleanup_mesh(vertices, faces, keep_largest=keep_largest, verbose=False)
        log(f"  Cleanup final: {time.perf_counter()-t0:.1f}s", flush=True)
        return vertices, faces

    if not no_cleanup:
        if cleanup_mesh is None:
            from trellmlx.mesh_cleanup import cleanup_mesh

    if reference_cleanup and not no_cleanup and target_faces and len(faces) > target_faces:
        if fill_holes is None:
            from trellmlx.mesh_cleanup import fill_small_holes as fill_holes
        if simplify is None:
            import fast_simplification
            simplify = fast_simplification.simplify

        t0 = time.perf_counter()
        vertices, faces = fill_holes(
            vertices,
            faces,
            max_hole_perimeter=3e-2,
            verbose=True,
        )
        log(f"  Reference cleanup initial hole fill: {len(vertices):,}V {len(faces):,}F "
            f"({time.perf_counter()-t0:.1f}s)", flush=True)

        coarse_target = target_faces * 3
        if len(faces) > coarse_target:
            t0 = time.perf_counter()
            ratio = coarse_target / len(faces)
            vertices, faces = simplify(vertices, faces, target_reduction=1.0 - ratio)
            log(f"  Reference cleanup coarse simplify: {len(vertices):,}V {len(faces):,}F "
                f"({time.perf_counter()-t0:.1f}s)", flush=True)

        t0 = time.perf_counter()
        vertices, faces = cleanup_mesh(
            vertices,
            faces,
            keep_largest=keep_largest,
            do_fix_normals=False,
        )
        log(f"  Reference cleanup pass 1: {time.perf_counter()-t0:.1f}s", flush=True)

        if len(faces) > target_faces:
            t0 = time.perf_counter()
            if qem_simplify:
                from trellmlx.simplify_qem_metal import simplify_qem
                vertices, faces = simplify_qem(vertices, faces, target_faces, verbose=True)
            else:
                for _ in range(3):
                    if len(faces) <= target_faces:
                        break
                    ratio = target_faces / len(faces)
                    target_reduction = 1.0 - ratio
                    if target_reduction <= 0:
                        break
                    vertices, faces = simplify(
                        vertices,
                        faces,
                        target_reduction=target_reduction,
                    )
                    if len(faces) <= target_faces * 1.1:
                        break
            log(f"  Reference cleanup final simplify: {len(vertices):,}V {len(faces):,}F "
                f"({time.perf_counter()-t0:.1f}s)", flush=True)

        vertices, faces = final_cleanup(vertices, faces)
        return vertices, faces

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
            from trellmlx.simplify_qem_metal import simplify_qem
            t0 = time.perf_counter()
            vertices, faces = simplify_qem(vertices, faces, target_faces, verbose=True)
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
        from trellmlx.simplify_qem_metal import simplify_qem
        vertices, faces = simplify_qem(
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
                        help="Use reference-like mesh postprocess order: initial hole fill, coarse simplify, "
                             "cleanup, final simplify, final cleanup.")
    parser.add_argument("--texture-size", type=int, default=1024,
                        help="Texture map resolution (default: 1024, try 2048 or 4096 for higher quality)")
    parser.add_argument("--texture-backend", choices=["cpu", "gpu"], default="gpu",
                        help="Texture bake backend: gpu (MLX Metal, default) or cpu (numpy)")
    parser.add_argument("--uv-method", choices=["auto", "lscm", "xatlas", "cube"], default="auto",
                        help="UV unwrap method: auto (xatlas, default), lscm, xatlas, or cube")
    parser.add_argument("--no-uv-visible-orient", action="store_true",
                        help="Skip post-UV visible-exterior island orientation before texture bake/export")
    parser.add_argument("--uv-visible-orient-size", type=int, default=192,
                        help="Image size for post-UV visible-exterior island orientation (default: 192)")
    parser.add_argument("--qem-simplify", action="store_true",
                        help="Use QEM simplification with topology guards (Metal-accelerated, "
                             "prevents holes from simplification). Slower but preserves mesh quality.")
    parser.add_argument("--voxel-remesh-pitch", type=float, default=0.0,
                        help="Opt-in topology rebuild via filled voxel grid before UV/texture bake. "
                             "0 disables (default). A pitch near 1/128 is useful for 512 no-cascade witnesses.")
    parser.add_argument("--save-checkpoints", metavar="DIR",
                        help="Save intermediate representations to DIR for replay")
    parser.add_argument("--stop-after-stage",
                        choices=["conditioning", "sparse_coords", "sparse_flow_step", "sparse_flow_block_trace", "sparse_internals", "shape_slat", "decoder_output", "mesh_raw"],
                        default=None,
                        help="Stop after writing the named checkpoint stage. Requires --save-checkpoints.")
    parser.add_argument("--shared-noise", metavar="NPZ",
                        help="Diagnostic: load sparse-structure noise from an NPZ containing ss_noise.")
    parser.add_argument("--no-cfg-rescale-clamp", action="store_true",
                        help="Diagnostic: disable MLX CFG-rescale ratio clamp for reference-parity witnesses.")
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

    if args.checkpoint_stop_file and not args.save_checkpoints:
        parser.error("--save-checkpoints is required when --checkpoint-stop-file is set")
    if args.stop_after_stage and not args.save_checkpoints:
        parser.error("--save-checkpoints is required when --stop-after-stage is set")
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
            _validate_mesh_checkpoint_vertices(
                vertices,
                mesh_grid_size=mesh_grid_size,
                coord_space=mesh_data.get("mesh_coord_space"),
                stage="mesh_raw",
            )
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
            )
            vertices, faces = _apply_voxel_remesh_if_requested(
                vertices,
                faces,
                pitch=args.voxel_remesh_pitch,
                no_cleanup=args.no_cleanup,
                keep_largest=args.keep_largest,
            )
            if args.save_checkpoints:
                from trellmlx.checkpoint import save_checkpoint
                save_checkpoint(args.save_checkpoints, "mesh_clean",
                                vertices=vertices, faces=faces,
                                mesh_grid_size=mesh_grid_size,
                                mesh_coord_space=_mesh_coord_space())

            # Jump straight to texture baking
            from trellmlx.texture_bake import bake_texture
            unwrap_fn, method_name = _select_uv_method(args.uv_method, vertices, faces)
            t0 = time.perf_counter()
            uv_verts, uv_faces, uvs, vmapping = unwrap_fn(vertices, faces)
            print(f"  UV unwrap ({method_name}): {len(uv_verts):,}V {len(uv_faces):,}F "
                  f"({time.perf_counter()-t0:.1f}s)", flush=True)
            uv_faces = _orient_uv_faces_for_export(uv_verts, uv_faces, args)
            if args.save_checkpoints:
                from trellmlx.checkpoint import save_checkpoint
                save_checkpoint(args.save_checkpoints, "mesh_uv",
                                vertices=uv_verts, faces=uv_faces,
                                uvs=uvs, vmapping=vmapping,
                                mesh_grid_size=mesh_grid_size,
                                mesh_coord_space=_mesh_coord_space())

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
                export_verts = _glb_export_vertices(uv_verts)
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

    vs3d_mode = bool(args.edit_target)
    if vs3d_mode and not os.path.exists(args.edit_target):
        raise FileNotFoundError(f"--edit-target path does not exist: {args.edit_target!r}. "
                                f"Did you forget to pass edit_target as a greenroom param?")

    if args.quantize:
        from trellmlx.quantize import quantize_model

    # === Image conditioning ===
    if args.image:
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
    # Upcast sparse structure flow to fp32 for CFG rescale stability.
    # bf16 precision noise gets amplified ~5x per Euler step through the
    # std ratio in CFG rescale, producing catastrophic divergence after
    # 4 steps. fp32 reduces this. Cost: ~2.4GB extra for ~20s.
    ss_flow.apply(lambda x: x.astype(mx.float32))
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
        # Cast conditioning to fp32 to match the fp32 sparse structure model
        if args.stop_after_stage == "sparse_flow_block_trace":
            t_tensor = mx.array([1000.0], dtype=mx.float32)
            pos_cond = cond.astype(mx.float32)
            neg_cond_fp32 = neg_cond.astype(mx.float32)
            pos_kv_cache = ss_flow.build_cross_kv_cache(pos_cond)
            neg_kv_cache = ss_flow.build_cross_kv_cache(neg_cond_fp32)
            pos_trace = ss_flow.trace_first_block(
                noise, t_tensor, pos_cond, cross_kv_cache=pos_kv_cache
            )
            neg_trace = ss_flow.trace_first_block(
                noise, t_tensor, neg_cond_fp32, cross_kv_cache=neg_kv_cache
            )
            mx.eval(*pos_trace.values(), *neg_trace.values())
            if args.save_checkpoints:
                from trellmlx.checkpoint import save_checkpoint
                save_checkpoint(
                    args.save_checkpoints,
                    "sparse_flow_block_trace",
                    pos_input_projected=np.array(pos_trace["input_projected"])[None].astype(np.float32, copy=False),
                    pos_block0_norm1=np.array(pos_trace["block0_norm1"])[None].astype(np.float32, copy=False),
                    pos_block0_modulated_self_input=np.array(pos_trace["block0_modulated_self_input"])[None].astype(np.float32, copy=False),
                    pos_block0_q_pre_norm=np.array(pos_trace["block0_q_pre_norm"])[None].astype(np.float32, copy=False),
                    pos_block0_k_pre_norm=np.array(pos_trace["block0_k_pre_norm"])[None].astype(np.float32, copy=False),
                    pos_block0_v=np.array(pos_trace["block0_v"])[None].astype(np.float32, copy=False),
                    pos_block0_q_post_norm=np.array(pos_trace["block0_q_post_norm"])[None].astype(np.float32, copy=False),
                    pos_block0_k_post_norm=np.array(pos_trace["block0_k_post_norm"])[None].astype(np.float32, copy=False),
                    pos_block0_q_post_rope=np.array(pos_trace["block0_q_post_rope"])[None].astype(np.float32, copy=False),
                    pos_block0_k_post_rope=np.array(pos_trace["block0_k_post_rope"])[None].astype(np.float32, copy=False),
                    pos_block0_attention_raw=np.array(pos_trace["block0_attention_raw"])[None].astype(np.float32, copy=False),
                    pos_block0_self_attn=np.array(pos_trace["block0_self_attn"])[None].astype(np.float32, copy=False),
                    pos_block0_after_self=np.array(pos_trace["block0_after_self"])[None].astype(np.float32, copy=False),
                    pos_block0_cross_attn=np.array(pos_trace["block0_cross_attn"])[None].astype(np.float32, copy=False),
                    pos_block0_after_cross=np.array(pos_trace["block0_after_cross"])[None].astype(np.float32, copy=False),
                    pos_block0_mlp=np.array(pos_trace["block0_mlp"])[None].astype(np.float32, copy=False),
                    pos_block0_after_mlp=np.array(pos_trace["block0_after_mlp"])[None].astype(np.float32, copy=False),
                    neg_input_projected=np.array(neg_trace["input_projected"])[None].astype(np.float32, copy=False),
                    neg_block0_norm1=np.array(neg_trace["block0_norm1"])[None].astype(np.float32, copy=False),
                    neg_block0_modulated_self_input=np.array(neg_trace["block0_modulated_self_input"])[None].astype(np.float32, copy=False),
                    neg_block0_q_pre_norm=np.array(neg_trace["block0_q_pre_norm"])[None].astype(np.float32, copy=False),
                    neg_block0_k_pre_norm=np.array(neg_trace["block0_k_pre_norm"])[None].astype(np.float32, copy=False),
                    neg_block0_v=np.array(neg_trace["block0_v"])[None].astype(np.float32, copy=False),
                    neg_block0_q_post_norm=np.array(neg_trace["block0_q_post_norm"])[None].astype(np.float32, copy=False),
                    neg_block0_k_post_norm=np.array(neg_trace["block0_k_post_norm"])[None].astype(np.float32, copy=False),
                    neg_block0_q_post_rope=np.array(neg_trace["block0_q_post_rope"])[None].astype(np.float32, copy=False),
                    neg_block0_k_post_rope=np.array(neg_trace["block0_k_post_rope"])[None].astype(np.float32, copy=False),
                    neg_block0_attention_raw=np.array(neg_trace["block0_attention_raw"])[None].astype(np.float32, copy=False),
                    neg_block0_self_attn=np.array(neg_trace["block0_self_attn"])[None].astype(np.float32, copy=False),
                    neg_block0_after_self=np.array(neg_trace["block0_after_self"])[None].astype(np.float32, copy=False),
                    neg_block0_cross_attn=np.array(neg_trace["block0_cross_attn"])[None].astype(np.float32, copy=False),
                    neg_block0_after_cross=np.array(neg_trace["block0_after_cross"])[None].astype(np.float32, copy=False),
                    neg_block0_mlp=np.array(neg_trace["block0_mlp"])[None].astype(np.float32, copy=False),
                    neg_block0_after_mlp=np.array(neg_trace["block0_after_mlp"])[None].astype(np.float32, copy=False),
                    t=np.array(1000.0, dtype=np.float32),
                    steps=np.array(n_steps, dtype=np.int32),
                    cfg_rescale_clamp=np.array(not args.no_cfg_rescale_clamp, dtype=np.bool_),
                )
            print("  Stop after stage: sparse_flow_block_trace", flush=True)
            return

        step_capture = {} if args.stop_after_stage == "sparse_flow_step" else None
        z_s = flow_euler_sample(ss_flow, noise,
                                cond.astype(mx.float32), neg_cond.astype(mx.float32),
                                steps=n_steps, verbose=False,
                                cfg_rescale_clamp=not args.no_cfg_rescale_clamp,
                                capture_first_step=step_capture,
                                stop_after_first_step=args.stop_after_stage == "sparse_flow_step")
        mx.eval(z_s)

    print(f"  Sampled: {time.perf_counter()-t0:.1f}s", flush=True)

    if args.save_checkpoints and args.stop_after_stage == "sparse_flow_step":
        from trellmlx.checkpoint import save_checkpoint
        save_checkpoint(
            args.save_checkpoints,
            "sparse_flow_step",
            noise=np.array(noise).astype(np.float32, copy=False),
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
            cfg_rescale_clamp=np.array(not args.no_cfg_rescale_clamp, dtype=np.bool_),
        )
        print("  Stop after stage: sparse_flow_step", flush=True)
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

    if args.save_checkpoints and args.stop_after_stage == "sparse_internals":
        from trellmlx.checkpoint import save_checkpoint
        lr_coords_4d = np.column_stack([np.zeros(len(lr_coords), dtype=np.int32), lr_coords])
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

    # === Stage 2a: LR Shape Latent ===
    print("\n=== Stage 2a: LR Shape Latent ===", flush=True)
    from trellmlx.models.slat_flow import SLatFlowModel

    # Sampler params from pipeline.json
    SHAPE_SAMPLER = dict(steps=n_steps, guidance_strength=7.5, guidance_rescale=0.5,
                         guidance_interval=(0.6, 1.0), rescale_t=3.0)
    TEX_SAMPLER = dict(steps=n_steps, guidance_strength=1.0, guidance_rescale=0.0,
                       guidance_interval=(0.6, 0.9), rescale_t=3.0)

    lr_slat_flow = SLatFlowModel()
    load_weights(lr_slat_flow, HF_4B + "slat_flow_img2shape_dit_1_3B_512_bf16.safetensors", verbose=False)
    if args.quantize:
        quantize_model(lr_slat_flow, bits=args.quantize)
    if args.compile:
        lr_slat_flow.compile()

    N_lr = len(lr_coords)
    lr_noise = mx.random.normal((N_lr, 32))
    lr_coords_4d = np.column_stack([np.zeros(N_lr, dtype=np.int32), lr_coords])
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

    t0 = time.perf_counter()
    lr_slat = flow_euler_sample(
        lr_slat_flow, lr_noise, cond_tgt if vs3d_mode else cond, neg_cond,
        verbose=False,
        coords=mx.array(lr_coords),
        **SHAPE_SAMPLER,
    )
    mx.eval(lr_slat)
    print(f"  Sampled: {time.perf_counter()-t0:.1f}s ({N_lr} tokens)", flush=True)

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
        hr_resolution = lr_resolution * 16  # 512 mesh grid for 32 LR coords
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

        hr_slat_flow = SLatFlowModel()
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
                        mesh_grid_size=mesh_grid_size,
                        mesh_coord_space=_mesh_coord_space())
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
    )
    vertices, faces = _apply_voxel_remesh_if_requested(
        vertices,
        faces,
        pitch=args.voxel_remesh_pitch,
        no_cleanup=args.no_cleanup,
        keep_largest=args.keep_largest,
    )
    if args.save_checkpoints:
        from trellmlx.checkpoint import save_checkpoint
        save_checkpoint(args.save_checkpoints, "mesh_clean",
                        vertices=vertices, faces=faces,
                        mesh_grid_size=mesh_grid_size,
                        mesh_coord_space=_mesh_coord_space())

    # === Stage 4: Texture SLat ===
    print("\n=== Stage 4: Texture SLat ===", flush=True)

    # Load texture flow model (same architecture, in_channels=64)
    tex_flow = SLatFlowModel(in_channels=64, out_channels=32)
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
    uv_faces = _orient_uv_faces_for_export(uv_verts, uv_faces, args)
    if args.save_checkpoints:
        from trellmlx.checkpoint import save_checkpoint
        save_checkpoint(args.save_checkpoints, "mesh_uv",
                        vertices=uv_verts, faces=uv_faces,
                        uvs=uvs, vmapping=vmapping,
                        mesh_grid_size=mesh_grid_size,
                        mesh_coord_space=_mesh_coord_space())

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
        export_verts = _glb_export_vertices(uv_verts)

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
