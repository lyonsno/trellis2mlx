"""CPU dual contouring on a regular SDF grid.

Places vertices inside cells at the QEF-optimal position given Hermite data
(edge zero-crossings + surface normals). Preserves sharp edges and corners
unlike marching cubes which rounds features to grid resolution.

Algorithm:
1. Find edges where SDF changes sign → compute zero-crossing + normal
2. For each cell with sign-changing edges, solve QEF for vertex position
3. For each sign-changing edge, emit a quad from the 4 adjacent cells

Reference: Ju et al., "Dual Contouring of Hermite Data" (2002)
"""

import numpy as np
from typing import Optional


def dual_contour_grid(
    sdf: np.ndarray,
    voxel_size: float,
    origin: np.ndarray,
    *,
    clamp_svd: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a triangle mesh from a regular SDF grid via dual contouring.

    Args:
        sdf: [Nx, Ny, Nz] float32 signed distance field
        voxel_size: uniform voxel edge length
        origin: [3] world-space position of grid corner (0,0,0)
        clamp_svd: SVD singular value clamp for QEF solve (prevents
            vertices flying to infinity on under-constrained cells)

    Returns:
        vertices: [V, 3] float32
        faces: [F, 3] int64
    """
    Nx, Ny, Nz = sdf.shape

    # Step 1: Find sign-changing edges and compute Hermite data
    # Edges along each axis: x-edges [Nx-1, Ny, Nz], etc.
    hermite_x, hermite_y, hermite_z = _find_edge_crossings(sdf, voxel_size, origin)

    # Step 2: Identify active cells (cells with at least one sign-changing edge)
    # A cell (i,j,k) has 12 edges. We check which cells are active.
    active_cells, cell_to_vertex_idx = _find_active_cells(
        sdf, hermite_x, hermite_y, hermite_z,
    )

    if len(active_cells) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64)

    # Step 3: Solve QEF for each active cell → vertex positions
    vertices = _solve_qef_batch(
        active_cells, hermite_x, hermite_y, hermite_z,
        voxel_size, origin, clamp_svd,
    )

    # Step 4: Generate quads/triangles from sign-changing edges
    faces = _generate_faces(
        sdf, hermite_x, hermite_y, hermite_z,
        cell_to_vertex_idx, Nx, Ny, Nz,
    )

    if len(faces) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64)

    return vertices.astype(np.float32), faces.astype(np.int64)


def _find_edge_crossings(sdf, voxel_size, origin):
    """Find zero-crossings on grid edges and compute intersection + normal.

    Returns three dicts mapping (i,j,k) edge index → (point, normal).
    For x-edges, (i,j,k) is the edge from (i,j,k) to (i+1,j,k).
    """
    Nx, Ny, Nz = sdf.shape
    grad = np.gradient(sdf, voxel_size)  # [3] arrays of [Nx, Ny, Nz]

    hermite_x = {}  # (i,j,k) → (crossing_point, normal)
    hermite_y = {}
    hermite_z = {}

    # X-edges: between (i,j,k) and (i+1,j,k)
    for i in range(Nx - 1):
        s0 = sdf[i, :, :]
        s1 = sdf[i + 1, :, :]
        sign_change = (s0 * s1) < 0
        jj, kk = np.where(sign_change)
        for j, k in zip(jj, kk):
            t = s0[j, k] / (s0[j, k] - s1[j, k])
            t = np.clip(t, 0.01, 0.99)
            pt = origin + np.array([
                (i + t) * voxel_size,
                j * voxel_size,
                k * voxel_size,
            ])
            # Interpolate gradient for normal
            n = np.array([
                (1 - t) * grad[0][i, j, k] + t * grad[0][min(i + 1, Nx - 1), j, k],
                (1 - t) * grad[1][i, j, k] + t * grad[1][min(i + 1, Nx - 1), j, k],
                (1 - t) * grad[2][i, j, k] + t * grad[2][min(i + 1, Nx - 1), j, k],
            ])
            norm = np.linalg.norm(n)
            if norm > 1e-10:
                n /= norm
            hermite_x[(i, j, k)] = (pt, n)

    # Y-edges: between (i,j,k) and (i,j+1,k)
    for j in range(Ny - 1):
        s0 = sdf[:, j, :]
        s1 = sdf[:, j + 1, :]
        sign_change = (s0 * s1) < 0
        ii, kk = np.where(sign_change)
        for i, k in zip(ii, kk):
            t = s0[i, k] / (s0[i, k] - s1[i, k])
            t = np.clip(t, 0.01, 0.99)
            pt = origin + np.array([
                i * voxel_size,
                (j + t) * voxel_size,
                k * voxel_size,
            ])
            n = np.array([
                (1 - t) * grad[0][i, j, k] + t * grad[0][i, min(j + 1, Ny - 1), k],
                (1 - t) * grad[1][i, j, k] + t * grad[1][i, min(j + 1, Ny - 1), k],
                (1 - t) * grad[2][i, j, k] + t * grad[2][i, min(j + 1, Ny - 1), k],
            ])
            norm = np.linalg.norm(n)
            if norm > 1e-10:
                n /= norm
            hermite_y[(i, j, k)] = (pt, n)

    # Z-edges: between (i,j,k) and (i,j,k+1)
    for k in range(Nz - 1):
        s0 = sdf[:, :, k]
        s1 = sdf[:, :, k + 1]
        sign_change = (s0 * s1) < 0
        ii, jj = np.where(sign_change)
        for i, j in zip(ii, jj):
            t = s0[i, j] / (s0[i, j] - s1[i, j])
            t = np.clip(t, 0.01, 0.99)
            pt = origin + np.array([
                i * voxel_size,
                j * voxel_size,
                (k + t) * voxel_size,
            ])
            n = np.array([
                (1 - t) * grad[0][i, j, k] + t * grad[0][i, j, min(k + 1, Nz - 1)],
                (1 - t) * grad[1][i, j, k] + t * grad[1][i, j, min(k + 1, Nz - 1)],
                (1 - t) * grad[2][i, j, k] + t * grad[2][i, j, min(k + 1, Nz - 1)],
            ])
            norm = np.linalg.norm(n)
            if norm > 1e-10:
                n /= norm
            hermite_z[(i, j, k)] = (pt, n)

    return hermite_x, hermite_y, hermite_z


def _find_active_cells(sdf, hermite_x, hermite_y, hermite_z):
    """Find cells that contain at least one sign-changing edge.

    A cell (i,j,k) spans from grid node (i,j,k) to (i+1,j+1,k+1).
    Its 12 edges are:
      4 x-edges: (i,j,k), (i,j+1,k), (i,j,k+1), (i,j+1,k+1)
      4 y-edges: (i,j,k), (i+1,j,k), (i,j,k+1), (i+1,j,k+1)
      4 z-edges: (i,j,k), (i+1,j,k), (i,j+1,k), (i+1,j+1,k)
    """
    Nx, Ny, Nz = sdf.shape
    active_set = set()

    # Each x-edge (i,j,k) belongs to cells:
    # (i, j-1, k-1), (i, j, k-1), (i, j-1, k), (i, j, k)
    for (i, j, k) in hermite_x:
        for dj in (-1, 0):
            for dk in (-1, 0):
                cj, ck = j + dj, k + dk
                if 0 <= i < Nx - 1 and 0 <= cj < Ny - 1 and 0 <= ck < Nz - 1:
                    active_set.add((i, cj, ck))

    # Each y-edge (i,j,k) belongs to cells:
    # (i-1, j, k-1), (i, j, k-1), (i-1, j, k), (i, j, k)
    for (i, j, k) in hermite_y:
        for di in (-1, 0):
            for dk in (-1, 0):
                ci, ck = i + di, k + dk
                if 0 <= ci < Nx - 1 and 0 <= j < Ny - 1 and 0 <= ck < Nz - 1:
                    active_set.add((ci, j, ck))

    # Each z-edge (i,j,k) belongs to cells:
    # (i-1, j-1, k), (i, j-1, k), (i-1, j, k), (i, j, k)
    for (i, j, k) in hermite_z:
        for di in (-1, 0):
            for dj in (-1, 0):
                ci, cj = i + di, j + dj
                if 0 <= ci < Nx - 1 and 0 <= cj < Ny - 1 and 0 <= k < Nz - 1:
                    active_set.add((ci, cj, k))

    active_cells = sorted(active_set)
    cell_to_vertex_idx = {cell: idx for idx, cell in enumerate(active_cells)}
    return active_cells, cell_to_vertex_idx


def _solve_qef_batch(active_cells, hermite_x, hermite_y, hermite_z,
                     voxel_size, origin, clamp_svd):
    """Solve QEF for each active cell to find optimal vertex position.

    The QEF minimizes sum_i (n_i . (v - p_i))^2 where (p_i, n_i) are
    the Hermite data from edge crossings in/around the cell.
    Falls back to cell center if QEF is under-constrained.
    """
    n_cells = len(active_cells)
    vertices = np.empty((n_cells, 3), dtype=np.float64)

    for idx, (ci, cj, ck) in enumerate(active_cells):
        # Gather Hermite data for this cell's 12 edges
        points = []
        normals = []

        # 4 x-edges of cell (ci,cj,ck)
        for dj, dk in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            key = (ci, cj + dj, ck + dk)
            if key in hermite_x:
                pt, n = hermite_x[key]
                points.append(pt)
                normals.append(n)

        # 4 y-edges
        for di, dk in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            key = (ci + di, cj, ck + dk)
            if key in hermite_y:
                pt, n = hermite_y[key]
                points.append(pt)
                normals.append(n)

        # 4 z-edges
        for di, dj in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            key = (ci + di, cj + dj, ck)
            if key in hermite_z:
                pt, n = hermite_z[key]
                points.append(pt)
                normals.append(n)

        cell_center = origin + np.array([
            (ci + 0.5) * voxel_size,
            (cj + 0.5) * voxel_size,
            (ck + 0.5) * voxel_size,
        ])

        if len(points) == 0:
            vertices[idx] = cell_center
            continue

        points = np.array(points)
        normals = np.array(normals)

        # QEF: minimize ||A @ v - b||^2
        # where A[i] = n_i, b[i] = n_i . p_i
        A = normals  # [M, 3]
        b = np.sum(normals * points, axis=1)  # [M]

        # SVD solve with singular value clamping
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        s_clamped = np.where(s > clamp_svd, s, 0.0)
        s_inv = np.where(s_clamped > 0, 1.0 / s_clamped, 0.0)
        v = Vt.T @ (s_inv * (U.T @ b))

        # Clamp to cell bounds (prevent vertices flying outside)
        cell_min = origin + np.array([ci, cj, ck], dtype=np.float64) * voxel_size
        cell_max = cell_min + voxel_size
        v = np.clip(v, cell_min, cell_max)

        vertices[idx] = v

    return vertices


def _generate_faces(sdf, hermite_x, hermite_y, hermite_z,
                    cell_to_vertex_idx, Nx, Ny, Nz):
    """Generate quads from sign-changing edges, then triangulate.

    Each sign-changing edge is shared by 4 cells. The quad connects
    the vertices of those 4 cells. We triangulate each quad into 2 triangles.
    """
    quads = []

    # X-edges: edge (i,j,k) is shared by cells
    # (i, j-1, k-1), (i, j, k-1), (i, j, k), (i, j-1, k)
    for (i, j, k) in hermite_x:
        cells = [
            (i, j - 1, k - 1),
            (i, j, k - 1),
            (i, j, k),
            (i, j - 1, k),
        ]
        vidxs = [cell_to_vertex_idx.get(c) for c in cells]
        if all(v is not None for v in vidxs):
            # Orient based on sign change direction
            if sdf[i, j, k] > 0:
                quads.append(vidxs)
            else:
                quads.append([vidxs[0], vidxs[3], vidxs[2], vidxs[1]])

    # Y-edges: edge (i,j,k) shared by cells
    # (i-1, j, k-1), (i, j, k-1), (i, j, k), (i-1, j, k)
    for (i, j, k) in hermite_y:
        cells = [
            (i - 1, j, k - 1),
            (i, j, k - 1),
            (i, j, k),
            (i - 1, j, k),
        ]
        vidxs = [cell_to_vertex_idx.get(c) for c in cells]
        if all(v is not None for v in vidxs):
            if sdf[i, j, k] > 0:
                quads.append(vidxs)
            else:
                quads.append([vidxs[0], vidxs[3], vidxs[2], vidxs[1]])

    # Z-edges: edge (i,j,k) shared by cells
    # (i-1, j-1, k), (i, j-1, k), (i, j, k), (i-1, j, k)
    for (i, j, k) in hermite_z:
        cells = [
            (i - 1, j - 1, k),
            (i, j - 1, k),
            (i, j, k),
            (i - 1, j, k),
        ]
        vidxs = [cell_to_vertex_idx.get(c) for c in cells]
        if all(v is not None for v in vidxs):
            if sdf[i, j, k] > 0:
                quads.append(vidxs)
            else:
                quads.append([vidxs[0], vidxs[3], vidxs[2], vidxs[1]])

    if not quads:
        return np.zeros((0, 3), dtype=np.int64)

    # Triangulate quads: each quad → 2 triangles
    quads = np.array(quads, dtype=np.int64)
    tri1 = quads[:, [0, 1, 2]]
    tri2 = quads[:, [0, 2, 3]]
    faces = np.concatenate([tri1, tri2], axis=0)

    return faces
