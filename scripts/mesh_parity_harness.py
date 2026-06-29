"""Mesh postprocess parity harness.

Load raw mesh from checkpoints, run topology metrics on:
1. Raw extracted mesh
2. After cleanup (current baseline)
3. After remesh + cleanup (candidate)

Produces a comparative report.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/mesh_parity_harness.py \
        --checkpoint /path/to/checkpoints \
        --label "official-control-512" \
        --target-faces 350000 \
        [--remesh-resolution 256] \
        [--output-dir /tmp/parity-results]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trellmlx.topology_metrics import (
    compute_topology_metrics,
    load_mesh_from_checkpoint,
    TopologyReport,
)


def run_cleanup(vertices, faces, target_faces, keep_largest=False):
    """Run the current cleanup-first pipeline."""
    from trellmlx.mesh_cleanup import cleanup_mesh
    import fast_simplification

    # Match the default simplify-first path from generate.py
    if target_faces and len(faces) > target_faces:
        for _ in range(3):
            ratio = target_faces / len(faces)
            if ratio >= 1:
                break
            vertices, faces = fast_simplification.simplify(
                vertices, faces, target_reduction=1.0 - ratio
            )
            if len(faces) <= target_faces * 1.1:
                break

    vertices, faces = cleanup_mesh(
        vertices, faces, keep_largest=keep_largest, verbose=True,
    )
    return vertices, faces


def run_remesh_then_cleanup(vertices, faces, target_faces, resolution=256,
                            band=3.0, project_back=0.9, keep_largest=False):
    """Run remesh reconstruction then cleanup."""
    from trellmlx.remesh_reconstruct import remesh_narrow_band
    from trellmlx.mesh_cleanup import cleanup_mesh
    import fast_simplification

    # Step 1: Remesh
    vertices, faces = remesh_narrow_band(
        vertices, faces,
        resolution=resolution,
        band=band,
        project_back=project_back,
        verbose=True,
    )

    # Step 2: Simplify if needed
    if target_faces and len(faces) > target_faces:
        for _ in range(3):
            ratio = target_faces / len(faces)
            if ratio >= 1:
                break
            vertices, faces = fast_simplification.simplify(
                vertices, faces, target_reduction=1.0 - ratio
            )
            if len(faces) <= target_faces * 1.1:
                break

    # Step 3: Cleanup
    vertices, faces = cleanup_mesh(
        vertices, faces, keep_largest=keep_largest, verbose=True,
    )
    return vertices, faces


def main():
    parser = argparse.ArgumentParser(description="Mesh postprocess parity harness")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint directory containing mesh_raw.npz")
    parser.add_argument("--label", default="",
                        help="Label for this run")
    parser.add_argument("--target-faces", type=int, default=350_000)
    parser.add_argument("--remesh-resolution", type=int, default=0,
                        help="Resolution for SDF remesh grid (0=match mesh_grid_size)")
    parser.add_argument("--remesh-band", type=float, default=1.0)
    parser.add_argument("--remesh-project-back", type=float, default=0.0)
    parser.add_argument("--output-dir", default="/tmp/parity-results")
    parser.add_argument("--skip-remesh", action="store_true",
                        help="Skip remesh candidate (faster, metrics-only)")
    parser.add_argument("--export-glb", action="store_true",
                        help="Export GLB files for visual comparison")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== Mesh Parity Harness ===", flush=True)
    print(f"Checkpoint: {args.checkpoint}", flush=True)
    print(f"Label: {args.label}", flush=True)
    print(f"Target faces: {args.target_faces:,}", flush=True)

    # Load raw mesh
    vertices, faces, metadata = load_mesh_from_checkpoint(args.checkpoint)
    grid_size = metadata.get("mesh_grid_size", 512)
    print(f"Raw mesh: {len(vertices):,}V {len(faces):,}F (grid_size={grid_size})",
          flush=True)

    # Resolve remesh resolution: 0 means match mesh_grid_size
    if args.remesh_resolution <= 0:
        args.remesh_resolution = int(grid_size)

    config_base = {
        "checkpoint": args.checkpoint,
        "label": args.label,
        "mesh_grid_size": grid_size,
        "target_faces": args.target_faces,
    }

    reports = []

    # 1. Raw mesh metrics
    print("\n--- Raw Mesh Metrics ---", flush=True)
    raw_report = compute_topology_metrics(
        vertices, faces,
        source_path=args.checkpoint,
        route="raw",
        config=config_base,
    )
    reports.append(raw_report)
    print(raw_report.summary(), flush=True)

    # 2. Cleanup-first baseline
    print("\n--- Cleanup-First Baseline ---", flush=True)
    t0 = time.perf_counter()
    clean_verts, clean_faces = run_cleanup(
        vertices.copy(), faces.copy(), args.target_faces,
    )
    cleanup_time = time.perf_counter() - t0
    print(f"Cleanup time: {cleanup_time:.1f}s", flush=True)

    cleanup_report = compute_topology_metrics(
        clean_verts, clean_faces,
        source_path=args.checkpoint,
        route="cleanup-first",
        config={**config_base, "pipeline": "simplify-first → cleanup"},
    )
    reports.append(cleanup_report)
    print(cleanup_report.summary(), flush=True)

    if args.export_glb:
        _export_glb(clean_verts, clean_faces, args.output_dir, f"{args.label}-cleanup-first")

    # 3. Remesh candidate
    if not args.skip_remesh:
        print("\n--- Remesh Candidate ---", flush=True)
        t0 = time.perf_counter()
        remesh_verts, remesh_faces = run_remesh_then_cleanup(
            vertices.copy(), faces.copy(),
            args.target_faces,
            resolution=args.remesh_resolution,
            band=args.remesh_band,
            project_back=args.remesh_project_back,
        )
        remesh_time = time.perf_counter() - t0
        print(f"Remesh time: {remesh_time:.1f}s", flush=True)

        remesh_report = compute_topology_metrics(
            remesh_verts, remesh_faces,
            source_path=args.checkpoint,
            route="remesh-candidate",
            config={
                **config_base,
                "pipeline": "remesh-narrow-band → simplify → cleanup",
                "remesh_resolution": args.remesh_resolution,
                "remesh_band": args.remesh_band,
                "remesh_project_back": args.remesh_project_back,
            },
        )
        reports.append(remesh_report)
        print(remesh_report.summary(), flush=True)

        if args.export_glb:
            _export_glb(remesh_verts, remesh_faces, args.output_dir,
                        f"{args.label}-remesh-candidate")

    # Write comparison report
    report_path = os.path.join(args.output_dir, f"{args.label or 'parity'}-report.json")
    report_data = {
        "label": args.label,
        "checkpoint": args.checkpoint,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "variants": [],
    }
    for r in reports:
        report_data["variants"].append({
            "route": r.route,
            "config": r.config,
            "n_vertices": r.n_vertices,
            "n_faces": r.n_faces,
            "n_boundary_edges": r.n_boundary_edges,
            "n_boundary_loops": r.n_boundary_loops,
            "boundary_loop_sizes": r.boundary_loop_sizes[:20],
            "n_non_manifold_edges": r.n_non_manifold_edges,
            "n_connected_components": r.n_connected_components,
            "component_face_counts": r.component_face_counts[:10],
            "n_degenerate_faces": r.n_degenerate_faces,
            "compute_time_s": r.compute_time_s,
        })

    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nReport saved: {report_path}", flush=True)

    # Print comparison table
    print("\n=== Comparison ===", flush=True)
    print(f"{'Metric':<25} ", end="")
    for r in reports:
        print(f"{r.route:<20} ", end="")
    print()
    print("-" * (25 + 21 * len(reports)))
    for metric in ["n_vertices", "n_faces", "n_boundary_edges", "n_boundary_loops",
                    "n_non_manifold_edges", "n_connected_components", "n_degenerate_faces"]:
        print(f"{metric:<25} ", end="")
        for r in reports:
            print(f"{getattr(r, metric):<20,} ", end="")
        print()


def _export_glb(vertices, faces, output_dir, name):
    """Export mesh as GLB for visual comparison."""
    import trimesh

    export_verts = vertices.copy()
    export_verts[:, 1], export_verts[:, 2] = vertices[:, 2].copy(), -vertices[:, 1].copy()

    mesh = trimesh.Trimesh(vertices=export_verts, faces=faces, process=False)
    path = os.path.join(output_dir, f"{name}.glb")
    mesh.export(path)
    print(f"  Exported: {path} ({os.path.getsize(path)/1e6:.1f}MB)", flush=True)


if __name__ == "__main__":
    main()
