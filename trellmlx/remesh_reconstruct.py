"""CPU-based mesh remeshing via narrow-band SDF + marching cubes.

Approximates the official TRELLIS.2 `cumesh.remeshing.remesh_narrow_band_dc`
pipeline:
1. Pre-clean the input mesh (fill holes, remove non-manifold edges)
2. Compute a signed distance field from the cleaned mesh
3. Re-extract the surface via marching cubes
4. Optionally project vertices back to the original surface

The official path uses GPU dual contouring (which preserves sharp edges);
this CPU version uses marching cubes (which rounds features to grid
resolution). At matched resolution the topology improvement is comparable
but geometric fidelity is lower on sharp features.
"""

import numpy as np
import time
from typing import Optional


def remesh_narrow_band(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    resolution: int = 512,
    band: float = 1.0,
    project_back: float = 0.0,
    center: Optional[np.ndarray] = None,
    scale: Optional[float] = None,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Remesh via narrow-band SDF + marching cubes.

    Mirrors the official `cumesh.remeshing.remesh_narrow_band_dc` interface.
    Defaults match the official callers (example.py, app.py):
    - `resolution`: should match mesh grid_size (default 512)
    - `band`: 1 (official default)
    - `project_back`: 0 (official callers all use 0)

    For large meshes (>500K faces), pre-simplifies for SDF computation.

    Args:
        vertices: [V, 3] float32
        faces: [F, 3] int
        resolution: voxel grid resolution (should match mesh_grid_size)
        band: narrow band width in voxels
        project_back: projection factor (0=no projection, 1=snap to surface).
            Official callers use 0. Non-zero projects onto pre-simplified
            mesh which can introduce asymmetric deformation.
        center: center of the remesh volume (auto-computed if None)
        scale: scale of the remesh volume (auto-computed if None)
        verbose: print progress

    Returns:
        new_vertices: [V', 3] float32
        new_faces: [F', 3] int64
    """
    import trimesh
    from skimage import measure

    t0 = time.perf_counter()

    if len(vertices) == 0 or len(faces) == 0:
        return vertices.copy(), faces.copy()

    # Pre-clean the input mesh before SDF computation (F9: official does this)
    # Fill holes and remove non-manifold edges to improve winding number accuracy
    if verbose:
        print(f"  Pre-cleaning for SDF ({len(faces):,}F)...", flush=True)
    from trellmlx.mesh_cleanup import fill_small_holes, repair_non_manifold_edges, remove_duplicate_faces
    clean_verts, clean_faces = remove_duplicate_faces(vertices, faces, verbose=False)
    clean_verts, clean_faces = repair_non_manifold_edges(clean_verts, clean_faces, verbose=False)
    clean_verts, clean_faces = fill_small_holes(
        clean_verts, clean_faces, max_hole_perimeter=3e-2, verbose=False,
    )
    if verbose:
        print(f"  Pre-cleaned: {len(clean_verts):,}V {len(clean_faces):,}F", flush=True)

    # Use the full pre-cleaned mesh for SDF computation.
    # Pre-simplification causes 16% sign errors near the surface (diagnostic
    # showed 72K false-exterior + 7.8K false-interior points at 128^3),
    # producing asymmetric blob artifacts. The full cleaned mesh takes ~23s
    # for igl signed_distance vs ~3s simplified, but eliminates the divergence.
    sdf_verts, sdf_faces = clean_verts, clean_faces

    orig_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # Compute bounding box from original mesh
    if center is None:
        center = orig_mesh.bounds.mean(axis=0)
    if scale is None:
        extent = orig_mesh.bounds[1] - orig_mesh.bounds[0]
        scale = extent.max()

    # F1: Use official band-expansion formula for grid scale
    # Official: scale = (resolution + 3 * band) / resolution * scale
    expanded_scale = (resolution + 3 * band) / resolution * scale
    half = expanded_scale / 2.0
    voxel_size = expanded_scale / resolution
    band_dist = band * voxel_size

    if verbose:
        print(f"  Remesh: resolution={resolution}, band={band}, "
              f"project_back={project_back}", flush=True)
        print(f"  Scale: {scale:.6f} → {expanded_scale:.6f} "
              f"(expansion factor {(resolution + 3*band)/resolution:.4f})", flush=True)
        print(f"  Voxel size: {voxel_size:.6f}, band_dist: {band_dist:.6f}", flush=True)

    # Grid coordinates
    grid_x = np.linspace(center[0] - half, center[0] + half, resolution)
    grid_y = np.linspace(center[1] - half, center[1] + half, resolution)
    grid_z = np.linspace(center[2] - half, center[2] + half, resolution)

    xx, yy, zz = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
    query_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    if verbose:
        print(f"  Computing SDF for {len(query_points):,} points...", flush=True)
        t1 = time.perf_counter()

    # Try libigl's fast winding number for signed distance (much faster)
    sdf = _compute_sdf_igl(sdf_verts, sdf_faces, query_points, verbose)
    if sdf is None:
        sdf_mesh = trimesh.Trimesh(vertices=sdf_verts, faces=sdf_faces, process=False)
        sdf = _compute_sdf_trimesh(sdf_mesh, query_points, band_dist, verbose)

    sdf = sdf.reshape(resolution, resolution, resolution)

    if verbose:
        print(f"  SDF computed in {time.perf_counter() - t1:.1f}s", flush=True)
        print(f"  SDF range: [{sdf.min():.6f}, {sdf.max():.6f}]", flush=True)
        n_inside = (sdf < 0).sum()
        n_near = (np.abs(sdf) < band_dist).sum()
        print(f"  Inside: {n_inside:,}, near surface: {n_near:,}", flush=True)

    # Extract surface from SDF — dual contouring (preserves sharp features)
    # with marching cubes as fallback
    grid_origin = np.array([center[0] - half, center[1] - half, center[2] - half])
    try:
        from trellmlx.dual_contouring import dual_contour_grid
        if verbose:
            print(f"  Dual contouring...", flush=True)
            t_dc = time.perf_counter()
        new_verts, new_faces = dual_contour_grid(
            sdf, voxel_size, grid_origin,
        )
        if verbose:
            print(f"  Dual contouring: {len(new_verts):,}V {len(new_faces):,}F "
                  f"({time.perf_counter() - t_dc:.1f}s)", flush=True)
    except Exception as e:
        if verbose:
            print(f"  Dual contouring failed ({e}), falling back to marching cubes",
                  flush=True)
        try:
            new_verts, new_faces, _, _ = measure.marching_cubes(
                sdf, level=0.0, spacing=(voxel_size, voxel_size, voxel_size),
            )
            new_verts = new_verts + grid_origin
        except ValueError as e2:
            if verbose:
                print(f"  Marching cubes also failed: {e2}", flush=True)
            return vertices.copy(), faces.copy()

    if verbose:
        print(f"  Marching cubes: {len(new_verts):,}V {len(new_faces):,}F", flush=True)

    if len(new_verts) == 0:
        return vertices.copy(), faces.copy()

    # Project back to original surface
    # NOTE: Official callers use project_back=0. Non-zero projection onto
    # a pre-simplified mesh causes asymmetric deformation (review F4/F5).
    if project_back > 0:
        if verbose:
            print(f"  Projecting {len(new_verts):,} vertices back "
                  f"(factor={project_back})...", flush=True)
        proj_mesh = trimesh.Trimesh(vertices=sdf_verts, faces=sdf_faces, process=False)
        chunk_size = 100_000
        all_closest = np.empty_like(new_verts)
        for i in range(0, len(new_verts), chunk_size):
            chunk = new_verts[i:i + chunk_size]
            closest, dist, face_id = trimesh.proximity.closest_point(proj_mesh, chunk)
            all_closest[i:i + chunk_size] = closest
        new_verts = new_verts * (1 - project_back) + all_closest * project_back
        if verbose:
            print(f"  Projected back", flush=True)

    new_faces = new_faces.astype(np.int64)

    if verbose:
        print(f"  Remesh total: {time.perf_counter() - t0:.1f}s", flush=True)

    return new_verts.astype(np.float32), new_faces


def _compute_sdf_igl(vertices, faces, query_points, verbose=False):
    """Compute SDF using libigl's signed_distance with fast winding number."""
    try:
        import igl
    except ImportError:
        if verbose:
            print("  libigl not available, falling back to trimesh", flush=True)
        return None

    try:
        v = np.ascontiguousarray(vertices, dtype=np.float64)
        f = np.ascontiguousarray(faces, dtype=np.int64)
        q = np.ascontiguousarray(query_points, dtype=np.float64)

        # Process in chunks to limit memory
        chunk_size = 500_000
        sdf = np.empty(len(query_points), dtype=np.float64)
        for i in range(0, len(query_points), chunk_size):
            chunk = q[i:i + chunk_size]
            S, I, C, N = igl.signed_distance(
                chunk, v, f,
                sign_type=igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER,
            )
            sdf[i:i + chunk_size] = S

        if verbose:
            n_inside = (sdf < 0).sum()
            print(f"  igl SDF: {n_inside:,} inside, "
                  f"dist range [{sdf.min():.6f}, {sdf.max():.6f}]", flush=True)

        return sdf.astype(np.float32)
    except Exception as e:
        if verbose:
            print(f"  igl SDF failed ({e}), falling back to trimesh", flush=True)
        return None


def _compute_sdf_trimesh(mesh, query_points, band_dist, verbose=False):
    """Compute SDF using trimesh (slower fallback)."""
    import trimesh

    chunk_size = 100_000
    sdf = np.full(len(query_points), band_dist * 2, dtype=np.float32)

    for i in range(0, len(query_points), chunk_size):
        chunk = query_points[i:i + chunk_size]
        closest, dist, face_id = trimesh.proximity.closest_point(mesh, chunk)
        signs = 1.0 - 2.0 * mesh.contains(chunk).astype(np.float32)
        sdf[i:i + chunk_size] = dist * signs

    return sdf
