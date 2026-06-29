"""SDF divergence diagnostic: compare pre-simplified vs full-cleaned mesh SDF.

Identifies spatial regions where pre-simplification causes the winding number
to flip sign, producing the asymmetric blob artifacts in remesh output.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/sdf_divergence_diagnostic.py \
        --checkpoint /path/to/checkpoints \
        --resolution 128 \
        --output-dir /tmp/sdf-diagnostic
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trellmlx.topology_metrics import load_mesh_from_checkpoint


def compute_sdf_igl(vertices, faces, query_points, label="", verbose=True):
    """Compute SDF via igl signed_distance."""
    import igl
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int64)
    q = np.ascontiguousarray(query_points, dtype=np.float64)

    chunk_size = 500_000
    sdf = np.empty(len(query_points), dtype=np.float64)
    t0 = time.perf_counter()
    for i in range(0, len(query_points), chunk_size):
        S, I, C, N = igl.signed_distance(
            q[i:i + chunk_size], v, f,
            sign_type=igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER,
        )
        sdf[i:i + chunk_size] = S
    if verbose:
        n_inside = (sdf < 0).sum()
        print(f"  {label}: {n_inside:,} inside, range [{sdf.min():.6f}, {sdf.max():.6f}] "
              f"({time.perf_counter() - t0:.1f}s)", flush=True)
    return sdf.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="SDF divergence diagnostic")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--resolution", type=int, default=128,
                        help="SDF grid resolution for comparison (128 is fast enough)")
    parser.add_argument("--output-dir", default="/tmp/sdf-diagnostic")
    parser.add_argument("--simplify-target", type=int, default=500_000,
                        help="Pre-simplification target (default: 500K, matching remesh)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load raw mesh
    vertices, faces, metadata = load_mesh_from_checkpoint(args.checkpoint)
    grid_size = metadata.get("mesh_grid_size", 512)
    print(f"Raw mesh: {len(vertices):,}V {len(faces):,}F (grid_size={grid_size})", flush=True)

    # Build query grid
    import trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    center = mesh.bounds.mean(axis=0)
    extent = mesh.bounds[1] - mesh.bounds[0]
    scale = extent.max()
    expanded_scale = (args.resolution + 3) / args.resolution * scale
    half = expanded_scale / 2.0
    voxel_size = expanded_scale / args.resolution

    grid_x = np.linspace(center[0] - half, center[0] + half, args.resolution)
    grid_y = np.linspace(center[1] - half, center[1] + half, args.resolution)
    grid_z = np.linspace(center[2] - half, center[2] + half, args.resolution)
    xx, yy, zz = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
    query_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    print(f"Query grid: {args.resolution}^3 = {len(query_points):,} points", flush=True)

    # Variant A: pre-cleaned (dedup + non-manifold repair + hole fill)
    print("\n=== Variant A: Pre-cleaned (no simplification) ===", flush=True)
    from trellmlx.mesh_cleanup import fill_small_holes, repair_non_manifold_edges, remove_duplicate_faces
    clean_v, clean_f = remove_duplicate_faces(vertices, faces, verbose=True)
    clean_v, clean_f = repair_non_manifold_edges(clean_v, clean_f, verbose=True)
    clean_v, clean_f = fill_small_holes(clean_v, clean_f, max_hole_perimeter=3e-2, verbose=True)
    print(f"  Cleaned: {len(clean_v):,}V {len(clean_f):,}F", flush=True)

    sdf_full = compute_sdf_igl(clean_v, clean_f, query_points, label="full-cleaned")

    # Variant B: pre-cleaned + pre-simplified (what remesh actually uses)
    print("\n=== Variant B: Pre-cleaned + Pre-simplified ===", flush=True)
    import fast_simplification
    simp_v, simp_f = fast_simplification.simplify(
        clean_v, clean_f, target_reduction=1.0 - args.simplify_target / len(clean_f),
    )
    print(f"  Simplified: {len(simp_v):,}V {len(simp_f):,}F", flush=True)

    sdf_simp = compute_sdf_igl(simp_v, simp_f, query_points, label="pre-simplified")

    # Variant C: raw mesh (no cleaning at all)
    print("\n=== Variant C: Raw mesh (no cleaning) ===", flush=True)
    sdf_raw = compute_sdf_igl(vertices, faces, query_points, label="raw")

    # Analysis: find sign disagreements
    print("\n=== Sign Divergence Analysis ===", flush=True)

    sign_full = np.sign(sdf_full)
    sign_simp = np.sign(sdf_simp)
    sign_raw = np.sign(sdf_raw)

    # Where do signs differ?
    disagree_simp_vs_full = sign_full != sign_simp
    disagree_raw_vs_full = sign_full != sign_raw

    n_disagree_simp = disagree_simp_vs_full.sum()
    n_disagree_raw = disagree_raw_vs_full.sum()
    n_total = len(query_points)

    print(f"  Simplified vs Full: {n_disagree_simp:,} sign disagreements "
          f"({n_disagree_simp/n_total*100:.2f}%)", flush=True)
    print(f"  Raw vs Full: {n_disagree_raw:,} sign disagreements "
          f"({n_disagree_raw/n_total*100:.2f}%)", flush=True)

    # Where are the disagreements spatially?
    if n_disagree_simp > 0:
        disagree_points = query_points[disagree_simp_vs_full]
        centroid = disagree_points.mean(axis=0)
        bbox_min = disagree_points.min(axis=0)
        bbox_max = disagree_points.max(axis=0)
        print(f"\n  Simplified-vs-Full disagreement region:", flush=True)
        print(f"    Centroid: [{centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}]",
              flush=True)
        print(f"    Bbox: [{bbox_min[0]:.4f}, {bbox_min[1]:.4f}, {bbox_min[2]:.4f}] → "
              f"[{bbox_max[0]:.4f}, {bbox_max[1]:.4f}, {bbox_max[2]:.4f}]", flush=True)
        print(f"    Extent: [{bbox_max[0]-bbox_min[0]:.4f}, {bbox_max[1]-bbox_min[1]:.4f}, "
              f"{bbox_max[2]-bbox_min[2]:.4f}]", flush=True)

        # Classify: which direction is the error?
        # "false interior" = simplified says inside, full says outside
        false_interior = (sign_simp < 0) & (sign_full >= 0)
        false_exterior = (sign_simp >= 0) & (sign_full < 0)
        print(f"    False interior (simp=in, full=out): {false_interior.sum():,}", flush=True)
        print(f"    False exterior (simp=out, full=in): {false_exterior.sum():,}", flush=True)

        if false_interior.sum() > 0:
            fi_points = query_points[false_interior]
            fi_centroid = fi_points.mean(axis=0)
            print(f"    False interior centroid: [{fi_centroid[0]:.4f}, {fi_centroid[1]:.4f}, "
                  f"{fi_centroid[2]:.4f}]", flush=True)

    if n_disagree_raw > 0:
        disagree_points = query_points[disagree_raw_vs_full]
        centroid = disagree_points.mean(axis=0)
        print(f"\n  Raw-vs-Full disagreement region:", flush=True)
        print(f"    Centroid: [{centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}]",
              flush=True)

    # Near-surface analysis: focus on points within 2 voxels of the surface
    near_surface = np.abs(sdf_full) < 2 * voxel_size
    if near_surface.sum() > 0:
        ns_disagree_simp = (disagree_simp_vs_full & near_surface).sum()
        ns_disagree_raw = (disagree_raw_vs_full & near_surface).sum()
        ns_total = near_surface.sum()
        print(f"\n  Near-surface only ({ns_total:,} points within 2 voxels):", flush=True)
        print(f"    Simplified vs Full: {ns_disagree_simp:,} disagreements "
              f"({ns_disagree_simp/ns_total*100:.2f}%)", flush=True)
        print(f"    Raw vs Full: {ns_disagree_raw:,} disagreements "
              f"({ns_disagree_raw/ns_total*100:.2f}%)", flush=True)

    # SDF magnitude divergence (even where signs agree)
    abs_diff = np.abs(sdf_full - sdf_simp)
    print(f"\n  SDF magnitude divergence (full vs simplified):", flush=True)
    print(f"    Mean: {abs_diff.mean():.6f}", flush=True)
    print(f"    Max: {abs_diff.max():.6f}", flush=True)
    print(f"    P95: {np.percentile(abs_diff, 95):.6f}", flush=True)
    print(f"    P99: {np.percentile(abs_diff, 99):.6f}", flush=True)

    # Save diagnostic data
    report = {
        "checkpoint": args.checkpoint,
        "resolution": args.resolution,
        "voxel_size": float(voxel_size),
        "full_cleaned_faces": len(clean_f),
        "simplified_faces": len(simp_f),
        "total_query_points": n_total,
        "sign_disagree_simp_vs_full": int(n_disagree_simp),
        "sign_disagree_raw_vs_full": int(n_disagree_raw),
        "sign_disagree_pct_simp": float(n_disagree_simp / n_total * 100),
        "sign_disagree_pct_raw": float(n_disagree_raw / n_total * 100),
        "sdf_diff_mean": float(abs_diff.mean()),
        "sdf_diff_max": float(abs_diff.max()),
        "sdf_diff_p95": float(np.percentile(abs_diff, 95)),
    }
    if n_disagree_simp > 0:
        report["false_interior_count"] = int(false_interior.sum())
        report["false_exterior_count"] = int(false_exterior.sum())
        report["disagree_centroid"] = disagree_points.mean(axis=0).tolist()
        report["disagree_bbox_min"] = disagree_points.min(axis=0).tolist()
        report["disagree_bbox_max"] = disagree_points.max(axis=0).tolist()

    report_path = os.path.join(args.output_dir, "sdf-divergence-report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {report_path}", flush=True)

    # Save SDF grids for visualization
    np.savez(
        os.path.join(args.output_dir, "sdf_grids.npz"),
        sdf_full=sdf_full.reshape(args.resolution, args.resolution, args.resolution),
        sdf_simp=sdf_simp.reshape(args.resolution, args.resolution, args.resolution),
        sdf_raw=sdf_raw.reshape(args.resolution, args.resolution, args.resolution),
        grid_origin=np.array([center[0] - half, center[1] - half, center[2] - half]),
        voxel_size=np.array([voxel_size]),
    )
    print(f"SDF grids saved: {os.path.join(args.output_dir, 'sdf_grids.npz')}", flush=True)


if __name__ == "__main__":
    main()
