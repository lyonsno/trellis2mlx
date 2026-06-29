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

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # Compute bounding box
    if center is None:
        center = mesh.bounds.mean(axis=0)
    if scale is None:
        extent = mesh.bounds[1] - mesh.bounds[0]
        scale = extent.max() * 1.1  # 10% padding

    # Build the voxel grid
    half = scale / 2.0
    voxel_size = scale / resolution

    # Grid coordinates
    lin = np.linspace(-half, half, resolution) + center[0]  # shift doesn't matter, we use per-axis
    grid_x = np.linspace(center[0] - half, center[0] + half, resolution)
    grid_y = np.linspace(center[1] - half, center[1] + half, resolution)
    grid_z = np.linspace(center[2] - half, center[2] + half, resolution)

    if verbose:
        print(f"  Remesh: resolution={resolution}, band={band}, "
              f"project_back={project_back}", flush=True)
        print(f"  Grid: [{center[0]-half:.3f}, {center[0]+half:.3f}] x "
              f"[{center[1]-half:.3f}, {center[1]+half:.3f}] x "
              f"[{center[2]-half:.3f}, {center[2]+half:.3f}]", flush=True)
        print(f"  Voxel size: {voxel_size:.6f}", flush=True)

    # Compute signed distance in narrow band using trimesh's proximity
    # Build a regular grid of query points within the narrow band
    # For efficiency, only query points near the surface
    band_dist = band * voxel_size

    # Sample query points on a regular grid
    xx, yy, zz = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
    query_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    if verbose:
        print(f"  Computing SDF for {len(query_points):,} points...", flush=True)
        t1 = time.perf_counter()

    # Use trimesh's proximity for unsigned distance + sign from winding number
    # For large grids, compute in chunks
    chunk_size = 500_000
    sdf = np.full(len(query_points), band_dist * 2, dtype=np.float32)

    for i in range(0, len(query_points), chunk_size):
        chunk = query_points[i:i + chunk_size]
        # Unsigned distance
        closest, dist, face_id = trimesh.proximity.closest_point(mesh, chunk)
        # Sign via winding number (inside = negative)
        signs = 1.0 - 2.0 * mesh.contains(chunk).astype(np.float32)
        sdf[i:i + chunk_size] = dist * signs

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

    # Project back to original surface
    if project_back > 0:
        closest, dist, face_id = trimesh.proximity.closest_point(mesh, new_verts)
        new_verts = new_verts * (1 - project_back) + closest * project_back
        if verbose:
            print(f"  Projected back (factor={project_back}): "
                  f"mean dist {dist.mean():.6f}", flush=True)

    new_faces = new_faces.astype(np.int64)

    if verbose:
        print(f"  Remesh total: {time.perf_counter() - t0:.1f}s", flush=True)

    return new_verts.astype(np.float32), new_faces
