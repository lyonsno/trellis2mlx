"""Texture baking: UV unwrap + rasterize + sample PBR attributes.

Takes a mesh (vertices, faces) and per-voxel PBR attributes from the
texture decoder, produces a textured mesh with PBR material maps.

Pipeline:
1. UV unwrap via xatlas
2. Rasterize mesh in UV space (barycentric interpolation → 3D positions)
3. Trilinear sample PBR attrs from voxel grid at rasterized positions
4. Inpaint UV seam gaps
5. Assemble PBR material
"""

import numpy as np


def uv_unwrap(vertices, faces):
    """UV unwrap a mesh using xatlas.

    Args:
        vertices: [V, 3] float32
        faces: [F, 3] int

    Returns:
        new_vertices: [V', 3] float32 (may have more vertices due to seam splits)
        new_faces: [F, 3] uint32
        uvs: [V', 2] float32 in [0, 1]
        vmapping: [V'] int — maps new vertex indices to original vertex indices
    """
    import xatlas

    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices.astype(np.float32), faces.astype(np.uint32))
    atlas.generate()
    vmapping, new_faces, uvs = atlas[0]

    new_vertices = vertices[vmapping]
    return new_vertices, new_faces, uvs, vmapping


def rasterize_uv(uvs, faces, texture_size=1024):
    """Rasterize triangles in UV space to get per-pixel barycentric coords.

    For each pixel in the texture, determines which triangle covers it and
    computes barycentric weights for interpolation.

    Args:
        uvs: [V, 2] float32 UV coordinates in [0, 1]
        faces: [F, 3] uint32
        texture_size: output texture resolution

    Returns:
        pixel_mask: [H, W] bool — which pixels are covered
        pixel_face_idx: [H, W] int — which face covers each pixel (-1 if none)
        pixel_bary: [H, W, 3] float32 — barycentric weights
    """
    H = W = texture_size
    F = len(faces)

    # Triangle vertices in pixel space
    uv_px = uvs * texture_size  # [V, 2]
    tri_uvs = uv_px[faces]      # [F, 3, 2]

    # Compute bounding boxes per triangle
    bb_min = np.floor(tri_uvs.min(axis=1)).astype(np.int32)  # [F, 2]
    bb_max = np.ceil(tri_uvs.max(axis=1)).astype(np.int32)   # [F, 2]
    bb_min = np.clip(bb_min, 0, texture_size - 1)
    bb_max = np.clip(bb_max, 0, texture_size - 1)

    # Output buffers
    pixel_face_idx = np.full((H, W), -1, dtype=np.int32)
    pixel_bary = np.zeros((H, W, 3), dtype=np.float32)

    # Precompute edge vectors for barycentric computation
    v0 = tri_uvs[:, 0]  # [F, 2]
    v1 = tri_uvs[:, 1]  # [F, 2]
    v2 = tri_uvs[:, 2]  # [F, 2]
    d00 = np.sum((v1 - v0) * (v1 - v0), axis=1)  # [F]
    d01 = np.sum((v1 - v0) * (v2 - v0), axis=1)  # [F]
    d11 = np.sum((v2 - v0) * (v2 - v0), axis=1)  # [F]
    inv_denom = 1.0 / (d00 * d11 - d01 * d01 + 1e-10)  # [F]

    # Process triangles in batches to keep memory bounded
    BATCH = 10000
    for start in range(0, F, BATCH):
        end = min(start + BATCH, F)
        batch_indices = np.arange(start, end)

        for fi in batch_indices:
            x_min, y_min = bb_min[fi]
            x_max, y_max = bb_max[fi]

            if x_min == x_max or y_min == y_max:
                continue

            # Generate pixel centers in this bbox
            xs = np.arange(x_min, x_max + 1) + 0.5
            ys = np.arange(y_min, y_max + 1) + 0.5
            px, py = np.meshgrid(xs, ys)
            points = np.stack([px.ravel(), py.ravel()], axis=1)  # [N, 2]

            # Barycentric coords
            dp = points - v0[fi]  # [N, 2]
            dp_d0 = dp[:, 0] * (v1[fi, 0] - v0[fi, 0]) + dp[:, 1] * (v1[fi, 1] - v0[fi, 1])
            dp_d1 = dp[:, 0] * (v2[fi, 0] - v0[fi, 0]) + dp[:, 1] * (v2[fi, 1] - v0[fi, 1])

            u = (d11[fi] * dp_d0 - d01[fi] * dp_d1) * inv_denom[fi]
            v = (d00[fi] * dp_d1 - d01[fi] * dp_d0) * inv_denom[fi]

            inside = (u >= 0) & (v >= 0) & (u + v <= 1)

            if not inside.any():
                continue

            # Write to output
            pxi = (points[inside, 0] - 0.5).astype(np.int32)
            pyi = (points[inside, 1] - 0.5).astype(np.int32)
            valid = (pxi >= 0) & (pxi < W) & (pyi >= 0) & (pyi < H)
            pxi = pxi[valid]
            pyi = pyi[valid]
            u_in = u[inside][valid]
            v_in = v[inside][valid]

            pixel_face_idx[pyi, pxi] = fi
            pixel_bary[pyi, pxi, 0] = 1.0 - u_in - v_in
            pixel_bary[pyi, pxi, 1] = u_in
            pixel_bary[pyi, pxi, 2] = v_in

    pixel_mask = pixel_face_idx >= 0
    return pixel_mask, pixel_face_idx, pixel_bary


def sample_voxel_attrs(positions, voxel_coords, voxel_attrs, grid_size):
    """Trilinear sample PBR attributes from a sparse voxel grid.

    For each sample position, finds the 8 enclosing voxel corners in the
    sparse grid, computes trilinear weights, and interpolates. Falls back
    to nearest-neighbor for positions where not all 8 corners exist.

    Args:
        positions: [N, 3] float32 world-space positions to sample
        voxel_coords: [M, 3] int voxel coordinates (spatial, no batch dim)
        voxel_attrs: [M, C] float32 per-voxel attributes
        grid_size: int — coordinate space extent for normalization

    Returns:
        sampled: [N, C] float32 interpolated attributes
    """
    N = len(positions)
    C = voxel_attrs.shape[1]

    # Convert positions from world space to voxel coordinate space
    # world = (coord + 0.5) / grid_size - 0.5  →  coord = (world + 0.5) * grid_size - 0.5
    voxel_pos = (positions + 0.5) * grid_size - 0.5  # [N, 3] continuous voxel coords

    # Floor to get the base corner of the enclosing cube
    base = np.floor(voxel_pos).astype(np.int32)  # [N, 3]
    frac = voxel_pos - base.astype(np.float32)    # [N, 3] fractional part in [0, 1)

    # Build sparse coord → index lookup
    packed = (voxel_coords[:, 0].astype(np.int64) << 42 |
              voxel_coords[:, 1].astype(np.int64) << 21 |
              voxel_coords[:, 2].astype(np.int64))
    coord_to_idx = {}
    for i in range(len(voxel_coords)):
        coord_to_idx[packed[i]] = i

    # 8 corner offsets for trilinear interpolation
    offsets = np.array([[dz, dy, dx]
                        for dz in range(2) for dy in range(2) for dx in range(2)],
                       dtype=np.int32)  # [8, 3]

    # Look up all 8 corners for each sample point
    corner_coords = base[:, None, :] + offsets[None, :, :]  # [N, 8, 3]
    corner_packed = (corner_coords[:, :, 0].astype(np.int64) << 42 |
                     corner_coords[:, :, 1].astype(np.int64) << 21 |
                     corner_coords[:, :, 2].astype(np.int64))  # [N, 8]

    # Vectorized lookup: flatten, batch lookup, reshape
    MISSING = -1
    flat_packed = corner_packed.ravel()  # [N*8]
    flat_indices = np.array([coord_to_idx.get(k, MISSING) for k in flat_packed],
                            dtype=np.int64)
    corner_indices = flat_indices.reshape(N, 8)

    # Compute trilinear weights: w = product of (1-frac) or frac per axis
    # For corner (dz, dy, dx): weight = wz * wy * wx
    # where wz = frac[:,0] if dz else (1-frac[:,0]), etc.
    weights = np.ones((N, 8), dtype=np.float32)
    for c, (dz, dy, dx) in enumerate(offsets):
        weights[:, c] *= frac[:, 0] if dz else (1.0 - frac[:, 0])
        weights[:, c] *= frac[:, 1] if dy else (1.0 - frac[:, 1])
        weights[:, c] *= frac[:, 2] if dx else (1.0 - frac[:, 2])

    # Interpolate: for each sample, sum weight * attr for existing corners,
    # renormalize by total weight of existing corners
    result = np.zeros((N, C), dtype=np.float32)
    weight_sum = np.zeros(N, dtype=np.float32)

    for c in range(8):
        valid = corner_indices[:, c] != MISSING
        if valid.any():
            idx = corner_indices[valid, c]
            result[valid] += weights[valid, c:c+1] * voxel_attrs[idx]
            weight_sum[valid] += weights[valid, c]

    # Normalize by total weight (handles sparse corners gracefully)
    nonzero = weight_sum > 0
    result[nonzero] /= weight_sum[nonzero, None]

    # Fallback: positions with no corners at all get nearest-neighbor
    if not nonzero.all():
        from scipy.spatial import cKDTree
        missing = ~nonzero
        voxel_world = (voxel_coords.astype(np.float32) + 0.5) / grid_size - 0.5
        tree = cKDTree(voxel_world)
        _, nn_idx = tree.query(positions[missing], k=1)
        result[missing] = voxel_attrs[nn_idx]

    return result


def inpaint_texture(image, mask, radius=3):
    """Inpaint missing pixels in a texture map.

    Args:
        image: [H, W, C] uint8 texture
        mask: [H, W] bool — True where pixels are valid

    Returns:
        inpainted: [H, W, C] uint8
    """
    try:
        import cv2
        inpaint_mask = (~mask).astype(np.uint8)
        if image.ndim == 2 or image.shape[2] == 1:
            # Single channel: cv2.inpaint needs 1 or 3 channel input
            img_2d = image.squeeze() if image.ndim == 3 else image
            result = cv2.inpaint(img_2d, inpaint_mask, radius, cv2.INPAINT_TELEA)
            return result[:, :, None] if image.ndim == 3 else result
        elif image.shape[2] == 3:
            return cv2.inpaint(image, inpaint_mask, radius, cv2.INPAINT_TELEA)
        else:
            # Inpaint RGB together, then each extra channel separately
            rgb = cv2.inpaint(image[:, :, :3], inpaint_mask, radius, cv2.INPAINT_TELEA)
            rest = []
            for c in range(3, image.shape[2]):
                ch = cv2.inpaint(image[:, :, c], inpaint_mask, radius, cv2.INPAINT_TELEA)
                rest.append(ch[:, :, None])
            return np.concatenate([rgb] + rest, axis=2)
    except ImportError:
        # QUALITY GAP: scipy nearest-neighbor fill produces blocky seam
        # artifacts. The cv2.inpaint Telea algorithm smoothly extends color
        # across seam boundaries. Install opencv-python to fix:
        #   uv pip install opencv-python
        from scipy.ndimage import distance_transform_edt
        result = image.copy()
        for c in range(image.shape[2]):
            channel = image[:, :, c].astype(np.float32)
            _, indices = distance_transform_edt(~mask, return_indices=True)
            result[:, :, c] = channel[indices[0], indices[1]]
        return result


def bake_texture(vertices, faces, uvs, vmapping,
                 voxel_coords, voxel_attrs, grid_size,
                 texture_size=1024):
    """Full texture baking pipeline.

    Args:
        vertices: [V', 3] UV-unwrapped vertices
        faces: [F, 3] UV-unwrapped faces
        uvs: [V', 2] UV coordinates
        vmapping: [V'] original vertex index mapping
        voxel_coords: [M, 3] int texture decoder output coords (spatial)
        voxel_attrs: [M, 6] float32 PBR attrs (RGB, metallic, roughness, alpha)
        grid_size: int — decoder output coord space extent
        texture_size: output texture resolution

    Returns:
        base_color: [H, W, 4] uint8 RGBA
        metallic_roughness: [H, W, 3] uint8 (zero, roughness, metallic)
    """
    import time

    H = W = texture_size

    # Step 1: Rasterize in UV space
    print(f"    Rasterizing UV space ({len(faces):,} tris, {H}x{W})...", flush=True)
    t0 = time.perf_counter()
    mask, face_idx, bary = rasterize_uv(uvs, faces, texture_size)
    print(f"    {mask.sum():,} pixels covered ({time.perf_counter()-t0:.1f}s)", flush=True)

    # Step 2: Interpolate 3D positions from barycentric coords
    t0 = time.perf_counter()
    tri_verts = vertices[faces]  # [F, 3, 3]
    covered = np.where(mask)
    fi = face_idx[covered]
    b = bary[covered]  # [N, 3]

    positions = (b[:, 0:1] * tri_verts[fi, 0] +
                 b[:, 1:2] * tri_verts[fi, 1] +
                 b[:, 2:3] * tri_verts[fi, 2])  # [N, 3]
    print(f"    Interpolated {len(positions):,} positions ({time.perf_counter()-t0:.1f}s)", flush=True)

    # Step 3: Sample PBR attrs from voxel grid
    t0 = time.perf_counter()
    sampled = sample_voxel_attrs(positions, voxel_coords, voxel_attrs, grid_size)
    print(f"    Sampled voxel attrs ({time.perf_counter()-t0:.1f}s)", flush=True)

    # Step 4: Build texture images
    # PBR layout: [0:3] RGB, [3] metallic, [4] roughness, [5] alpha
    base_color_f = np.zeros((H, W, 4), dtype=np.float32)
    base_color_f[covered[0], covered[1], :3] = sampled[:, :3]
    base_color_f[covered[0], covered[1], 3] = sampled[:, 5] if sampled.shape[1] > 5 else 1.0

    mr_f = np.zeros((H, W, 3), dtype=np.float32)
    mr_f[covered[0], covered[1], 1] = sampled[:, 4] if sampled.shape[1] > 4 else 0.5  # roughness
    mr_f[covered[0], covered[1], 2] = sampled[:, 3] if sampled.shape[1] > 3 else 0.0  # metallic

    # Clamp and convert to uint8
    base_color = np.clip(base_color_f * 255, 0, 255).astype(np.uint8)
    mr = np.clip(mr_f * 255, 0, 255).astype(np.uint8)

    # Step 5: Inpaint seams
    t0 = time.perf_counter()
    base_color[:, :, :3] = inpaint_texture(base_color[:, :, :3], mask)
    base_color[:, :, 3:] = inpaint_texture(base_color[:, :, 3:], mask)
    mr = inpaint_texture(mr, mask)
    print(f"    Inpainted seams ({time.perf_counter()-t0:.1f}s)", flush=True)

    return base_color, mr
