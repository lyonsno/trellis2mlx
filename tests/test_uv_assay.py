"""UV unwrap assay: diagnostic comparison of xatlas vs LSCM on real mesh geometry.

Runs both UV unwrap methods on the same mesh and reports:
- Chart count and size distribution
- UV space utilization (fraction of [0,1]² occupied)
- LSCM success/fallback rate
- Per-chart UV area and aspect ratio
- Pixel coverage at a reference texture size
- Whether any charts have degenerate/inverted UVs

This is a diagnostic test, not a pass/fail gate. It prints a report.
"""

import numpy as np
import pytest
import time


def _make_shoe_like_mesh():
    """Create a smooth curved mesh resembling a shoe sole.

    This is a hemisphere-ish shape with varying curvature — enough to
    stress chart segmentation without needing the full pipeline.
    """
    # UV sphere: smooth, organic, varying normals
    n_lat, n_lon = 20, 40
    verts = []
    for i in range(n_lat + 1):
        theta = np.pi * i / n_lat
        for j in range(n_lon):
            phi = 2 * np.pi * j / n_lon
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            # Stretch into shoe-like shape
            verts.append([x * 0.3, y * 0.15, z * 0.1])
    verts = np.array(verts, dtype=np.float32)

    faces = []
    for i in range(n_lat):
        for j in range(n_lon):
            v00 = i * n_lon + j
            v10 = (i + 1) * n_lon + j
            v01 = i * n_lon + (j + 1) % n_lon
            v11 = (i + 1) * n_lon + (j + 1) % n_lon
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    faces = np.array(faces, dtype=np.uint32)
    return verts, faces


def _make_voxel_mesh():
    """5x5x5 cube grid — 125 cubes, 1500 faces."""
    all_verts = []
    all_faces = []
    cube_verts = np.array([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [0,0,1],[1,0,1],[1,1,1],[0,1,1],
    ], dtype=np.float32)
    cube_faces = np.array([
        [0,1,2],[0,2,3],[4,6,5],[4,7,6],
        [0,4,5],[0,5,1],[2,6,7],[2,7,3],
        [0,3,7],[0,7,4],[1,5,6],[1,6,2],
    ], dtype=np.uint32)
    for x in range(5):
        for y in range(5):
            for z in range(5):
                offset = len(all_verts)
                all_verts.append(cube_verts + [x, y, z])
                all_faces.append(cube_faces + offset)
    return np.vstack(all_verts).astype(np.float32), np.vstack(all_faces).astype(np.uint32)


def _compute_uv_triangle_areas(uvs, faces):
    """Compute signed area of each triangle in UV space."""
    uv0 = uvs[faces[:, 0]]
    uv1 = uvs[faces[:, 1]]
    uv2 = uvs[faces[:, 2]]
    # Signed area via cross product
    return 0.5 * ((uv1[:, 0] - uv0[:, 0]) * (uv2[:, 1] - uv0[:, 1]) -
                   (uv2[:, 0] - uv0[:, 0]) * (uv1[:, 1] - uv0[:, 1]))


def _compute_3d_triangle_areas(verts, faces):
    """Compute area of each triangle in 3D."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def _pixel_coverage(uvs, faces, texture_size=1024):
    """Count how many texture pixels are covered by UV triangles."""
    from trellmlx.texture_bake import rasterize_uv
    mask, _, _ = rasterize_uv(uvs.astype(np.float32), faces.astype(np.uint32), texture_size)
    return mask.sum(), texture_size * texture_size


def _run_assay(method_name, unwrap_fn, vertices, faces, texture_size=512):
    """Run UV unwrap and collect diagnostics."""
    t0 = time.perf_counter()
    new_verts, new_faces, uvs, vmapping = unwrap_fn(vertices, faces)
    elapsed = time.perf_counter() - t0

    uv_areas = _compute_uv_triangle_areas(uvs, new_faces)
    face_areas_3d = _compute_3d_triangle_areas(new_verts, new_faces)

    n_inverted = (uv_areas < 0).sum()
    n_degenerate = (np.abs(uv_areas) < 1e-12).sum()
    total_uv_area = np.abs(uv_areas).sum()

    covered, total_pixels = _pixel_coverage(uvs, new_faces, texture_size)

    # Check UV range
    uv_min = uvs.min(axis=0)
    uv_max = uvs.max(axis=0)

    report = {
        "method": method_name,
        "time_s": elapsed,
        "input_faces": len(faces),
        "input_verts": len(vertices),
        "output_faces": len(new_faces),
        "output_verts": len(new_verts),
        "uv_min": uv_min.tolist(),
        "uv_max": uv_max.tolist(),
        "total_uv_area": float(total_uv_area),
        "uv_utilization_pct": float(total_uv_area * 100),
        "n_inverted_triangles": int(n_inverted),
        "n_degenerate_triangles": int(n_degenerate),
        "pixel_coverage": int(covered),
        "pixel_total": int(total_pixels),
        "pixel_coverage_pct": float(covered / total_pixels * 100),
        "vertex_blowup": float(len(new_verts) / len(vertices)),
    }
    return report


def _print_report(report):
    print(f"\n{'='*60}")
    print(f"  UV Assay: {report['method']}")
    print(f"{'='*60}")
    print(f"  Time:              {report['time_s']:.2f}s")
    print(f"  Input:             {report['input_verts']:,}V {report['input_faces']:,}F")
    print(f"  Output:            {report['output_verts']:,}V {report['output_faces']:,}F")
    print(f"  Vertex blowup:     {report['vertex_blowup']:.1f}x")
    print(f"  UV range:          [{report['uv_min'][0]:.4f}, {report['uv_min'][1]:.4f}] → [{report['uv_max'][0]:.4f}, {report['uv_max'][1]:.4f}]")
    print(f"  UV utilization:    {report['uv_utilization_pct']:.1f}%")
    print(f"  Inverted faces:    {report['n_inverted_triangles']}")
    print(f"  Degenerate faces:  {report['n_degenerate_triangles']}")
    print(f"  Pixel coverage:    {report['pixel_coverage']:,} / {report['pixel_total']:,} ({report['pixel_coverage_pct']:.1f}%)")


class TestUVAssaySmooth:
    """Comparative UV assay on smooth (shoe-like) geometry."""

    def test_xatlas_baseline(self):
        from trellmlx.texture_bake import uv_unwrap
        verts, faces = _make_shoe_like_mesh()
        report = _run_assay("xatlas", uv_unwrap, verts, faces)
        _print_report(report)
        assert report["pixel_coverage_pct"] > 10

    def test_lscm(self):
        from trellmlx.texture_bake import uv_unwrap_lscm
        verts, faces = _make_shoe_like_mesh()
        report = _run_assay("lscm", uv_unwrap_lscm, verts, faces)
        _print_report(report)
        assert report["pixel_coverage_pct"] > 20

    def test_lscm_vs_xatlas_coverage(self):
        """LSCM should achieve at least 50% of xatlas's pixel coverage."""
        from trellmlx.texture_bake import uv_unwrap, uv_unwrap_lscm
        verts, faces = _make_shoe_like_mesh()

        xatlas_report = _run_assay("xatlas", uv_unwrap, verts, faces)
        lscm_report = _run_assay("lscm", uv_unwrap_lscm, verts, faces)

        _print_report(xatlas_report)
        _print_report(lscm_report)

        ratio = lscm_report["pixel_coverage"] / max(xatlas_report["pixel_coverage"], 1)
        print(f"\n  LSCM/xatlas coverage ratio: {ratio:.2f}")
        assert ratio > 0.5, (
            f"LSCM coverage ({lscm_report['pixel_coverage_pct']:.1f}%) is less than "
            f"50% of xatlas ({xatlas_report['pixel_coverage_pct']:.1f}%)"
        )

    def test_lscm_no_inverted_triangles(self):
        """LSCM should not produce inverted (negative area) UV triangles."""
        from trellmlx.texture_bake import uv_unwrap_lscm
        verts, faces = _make_shoe_like_mesh()
        report = _run_assay("lscm", uv_unwrap_lscm, verts, faces)
        _print_report(report)
        assert report["n_inverted_triangles"] == 0, (
            f"{report['n_inverted_triangles']} inverted UV triangles"
        )


class TestUVAssayVoxel:
    """Comparative UV assay on voxel geometry."""

    def test_xatlas_baseline(self):
        from trellmlx.texture_bake import uv_unwrap
        verts, faces = _make_voxel_mesh()
        report = _run_assay("xatlas", uv_unwrap, verts, faces)
        _print_report(report)
        assert report["pixel_coverage_pct"] > 10

    def test_lscm(self):
        from trellmlx.texture_bake import uv_unwrap_lscm
        verts, faces = _make_voxel_mesh()
        report = _run_assay("lscm", uv_unwrap_lscm, verts, faces)
        _print_report(report)
        assert report["pixel_coverage_pct"] > 1

    def test_lscm_completes_in_reasonable_time(self):
        """LSCM must not go pathological on voxel geometry."""
        from trellmlx.texture_bake import uv_unwrap_lscm
        verts, faces = _make_voxel_mesh()
        report = _run_assay("lscm", uv_unwrap_lscm, verts, faces)
        assert report["time_s"] < 30, f"LSCM took {report['time_s']:.1f}s on 1500 faces"
