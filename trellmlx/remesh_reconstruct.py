"""CPU-based mesh remeshing via narrow-band SDF + marching cubes.

Approximates the official TRELLIS.2 `cumesh.remeshing.remesh_narrow_band_dc`
pipeline:
1. Compute a narrow-band signed distance field from the input mesh
2. Re-extract the surface via marching cubes
3. Project vertices back to the original surface

This is a CPU reference implementation for testing whether topology
reconstruction reduces holes. Not final Metal/MLX parity.
"""

import numpy as np
import time
from typing import Optional


def remesh_narrow_band(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    resolution: int = 256,
    band: float = 3.0,
    project_back: float = 0.9,
    center: Optional[np.ndarray] = None,
    scale: Optional[float] = None,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Remesh via narrow-band SDF + marching cubes.

    Mirrors the official `cumesh.remeshing.remesh_narrow_band_dc` interface:
    - `resolution`: grid resolution for SDF computation
    - `band`: narrow band width in voxels
    - `project_back`: fraction to project new vertices back toward original surface
      (0 = no projection, 1 = snap to original surface)

    For large meshes (>500K faces), pre-simplifies to ~200K faces for SDF
    computation to keep memory and time manageable. The original full-resolution
    mesh is used for final vertex projection.

    Args:
        vertices: [V, 3] float32
        faces: [F, 3] int
        resolution: voxel grid resolution
        band: narrow band width in voxels
        project_back: projection factor (0-1)
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

    # For SDF computation, use a simplified mesh if the input is very large.
    # This keeps memory usage bounded while preserving the surface shape.
    SDF_FACE_LIMIT = 200_000
    if len(faces) > SDF_FACE_LIMIT:
        if verbose:
            print(f"  Pre-simplifying for SDF: {len(faces):,}F → ~{SDF_FACE_LIMIT:,}F...",
                  flush=True)
        import fast_simplification
        t_simp = time.perf_counter()
        sdf_verts, sdf_faces = fast_simplification.simplify(
            vertices, faces, target_reduction=1.0 - SDF_FACE_LIMIT / len(faces),
        )
        if verbose:
            print(f"  Pre-simplified: {len(sdf_verts):,}V {len(sdf_faces):,}F "
                  f"({time.perf_counter() - t_simp:.1f}s)", flush=True)
    else:
        sdf_verts, sdf_faces = vertices, faces

    sdf_mesh = trimesh.Trimesh(vertices=sdf_verts, faces=sdf_faces, process=False)
    orig_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # Compute bounding box from original
    if center is None:
        center = orig_mesh.bounds.mean(axis=0)
    if scale is None:
        extent = orig_mesh.bounds[1] - orig_mesh.bounds[0]
        scale = extent.max() * 1.1  # 10% padding

    half = scale / 2.0
    voxel_size = scale / resolution
    band_dist = band * voxel_size

    if verbose:
        print(f"  Remesh: resolution={resolution}, band={band}, "
              f"project_back={project_back}", flush=True)
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
        sdf = _compute_sdf_trimesh(sdf_mesh, query_points, band_dist, verbose)

    sdf = sdf.reshape(resolution, resolution, resolution)

    if verbose:
        print(f"  SDF computed in {time.perf_counter() - t1:.1f}s", flush=True)
        print(f"  SDF range: [{sdf.min():.6f}, {sdf.max():.6f}]", flush=True)
        n_inside = (sdf < 0).sum()
        n_near = (np.abs(sdf) < band_dist).sum()
        print(f"  Inside: {n_inside:,}, near surface: {n_near:,}", flush=True)

    # Marching cubes to extract the zero-level set
    try:
        new_verts, new_faces, normals, values = measure.marching_cubes(
            sdf, level=0.0, spacing=(voxel_size, voxel_size, voxel_size),
        )
    except ValueError as e:
        if verbose:
            print(f"  Marching cubes failed: {e}", flush=True)
        return vertices.copy(), faces.copy()

    # Shift vertices to world coordinates
    new_verts = new_verts + np.array([center[0] - half, center[1] - half, center[2] - half])

    if verbose:
        print(f"  Marching cubes: {len(new_verts):,}V {len(new_faces):,}F", flush=True)

    if len(new_verts) == 0:
        return vertices.copy(), faces.copy()

    # Project back to original surface (use original full-res mesh for accuracy)
    if project_back > 0:
        if verbose:
            print(f"  Projecting {len(new_verts):,} vertices back...", flush=True)
        # Use simplified mesh for projection if original is huge
        proj_mesh = sdf_mesh if len(faces) > SDF_FACE_LIMIT else orig_mesh
        chunk_size = 100_000
        all_closest = np.empty_like(new_verts)
        for i in range(0, len(new_verts), chunk_size):
            chunk = new_verts[i:i + chunk_size]
            closest, dist, face_id = trimesh.proximity.closest_point(proj_mesh, chunk)
            all_closest[i:i + chunk_size] = closest
        new_verts = new_verts * (1 - project_back) + all_closest * project_back
        if verbose:
            print(f"  Projected back (factor={project_back})", flush=True)

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
