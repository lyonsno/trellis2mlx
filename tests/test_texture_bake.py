"""Tests for texture baking — UV rasterization and voxel attribute sampling.

Compares MLX GPU implementations against the numpy reference on identical inputs.
Uses synthetic meshes with known UV layouts for deterministic verification.
"""

import numpy as np
import pytest

from trellmlx.texture_bake import rasterize_uv, sample_voxel_attrs


def _make_uv_quad():
    """A single quad (2 triangles) covering the full UV [0,1]x[0,1] space.

    Returns vertices, faces, uvs where the quad maps directly to the
    texture so every pixel should be covered.
    """
    vertices = np.array([
        [-0.5, -0.5, 0.0],
        [ 0.5, -0.5, 0.0],
        [ 0.5,  0.5, 0.0],
        [-0.5,  0.5, 0.0],
    ], dtype=np.float32)
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.uint32)
    uvs = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=np.float32)
    return vertices, faces, uvs


def _make_partial_uv_triangle():
    """A single triangle covering roughly half the UV space (lower-left).

    Vertices at (0,0), (1,0), (0,1) in UV — covers the lower-left triangle.
    """
    vertices = np.array([
        [-0.5, -0.5, 0.0],
        [ 0.5, -0.5, 0.0],
        [-0.5,  0.5, 0.0],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    uvs = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=np.float32)
    return vertices, faces, uvs


class TestRasterizeUV:
    def test_full_quad_covers_all_pixels(self):
        """A quad spanning full UV should cover nearly all pixels."""
        _, faces, uvs = _make_uv_quad()
        texture_size = 16
        mask, face_idx, bary = rasterize_uv(uvs, faces, texture_size)

        # Nearly all pixels should be covered (some edge pixels may miss)
        coverage = mask.sum() / (texture_size * texture_size)
        assert coverage > 0.9, f"Expected >90% coverage, got {coverage:.1%}"

    def test_half_triangle_covers_roughly_half(self):
        """A triangle covering the lower-left half should cover ~50% of pixels."""
        _, faces, uvs = _make_partial_uv_triangle()
        texture_size = 32
        mask, face_idx, bary = rasterize_uv(uvs, faces, texture_size)

        coverage = mask.sum() / (texture_size * texture_size)
        assert 0.3 < coverage < 0.7, f"Expected ~50% coverage, got {coverage:.1%}"

    def test_barycentric_weights_sum_to_one(self):
        """Barycentric weights at covered pixels should sum to 1."""
        _, faces, uvs = _make_uv_quad()
        mask, face_idx, bary = rasterize_uv(uvs, faces, texture_size=16)

        bary_sum = bary[mask].sum(axis=1)
        np.testing.assert_allclose(bary_sum, 1.0, atol=1e-5)

    def test_barycentric_weights_non_negative(self):
        """Barycentric weights should be non-negative at covered pixels."""
        _, faces, uvs = _make_uv_quad()
        mask, face_idx, bary = rasterize_uv(uvs, faces, texture_size=16)

        assert (bary[mask] >= -1e-6).all()

    def test_output_shapes(self):
        _, faces, uvs = _make_uv_quad()
        sz = 8
        mask, face_idx, bary = rasterize_uv(uvs, faces, texture_size=sz)

        assert mask.shape == (sz, sz)
        assert mask.dtype == bool
        assert face_idx.shape == (sz, sz)
        assert bary.shape == (sz, sz, 3)

    def test_mlx_matches_numpy(self):
        """MLX GPU rasterizer should match numpy reference within tolerance.

        Allows small edge-pixel disagreement (boundary rounding differences)
        but requires >93% overlap and matching barycentric weights where both
        agree on coverage.
        """
        from trellmlx.texture_bake import rasterize_uv_mlx

        _, faces, uvs = _make_uv_quad()
        texture_size = 32

        mask_np, fidx_np, bary_np = rasterize_uv(uvs, faces, texture_size)
        mask_mlx, fidx_mlx, bary_mlx = rasterize_uv_mlx(uvs, faces, texture_size)

        # Coverage should be very similar (allow edge pixel differences)
        overlap = (mask_np & mask_mlx).sum()
        union = (mask_np | mask_mlx).sum()
        iou = overlap / max(union, 1)
        assert iou > 0.93, f"IoU {iou:.3f} too low"

        # Where both agree on coverage, face assignment and bary should match
        both = mask_np & mask_mlx
        np.testing.assert_array_equal(fidx_np[both], fidx_mlx[both])
        np.testing.assert_allclose(bary_np[both], bary_mlx[both], atol=1e-4)


class TestSampleVoxelAttrs:
    def test_exact_voxel_center_returns_voxel_value(self):
        """Sampling at a voxel center should return that voxel's attrs exactly."""
        grid_size = 4
        # Single voxel at coord (1, 1, 1)
        coords = np.array([[1, 1, 1]], dtype=np.int32)
        attrs = np.array([[1.0, 0.5, 0.25, 0.0, 0.8, 1.0]], dtype=np.float32)

        # World position of voxel center: (coord + 0.5) / grid_size - 0.5
        pos = np.array([[(1 + 0.5) / 4 - 0.5,
                         (1 + 0.5) / 4 - 0.5,
                         (1 + 0.5) / 4 - 0.5]], dtype=np.float32)

        result = sample_voxel_attrs(pos, coords, attrs, grid_size)
        np.testing.assert_allclose(result, attrs, atol=1e-5)

    def test_interpolation_between_two_voxels(self):
        """Sampling halfway between two voxels should average their attrs."""
        grid_size = 4
        coords = np.array([[1, 1, 1], [2, 1, 1]], dtype=np.int32)
        attrs = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ], dtype=np.float32)

        # Midpoint between voxel centers along axis 0
        mid_coord = 1.5  # halfway between coord 1 and 2
        pos = np.array([[(mid_coord + 0.5) / 4 - 0.5,
                         (1 + 0.5) / 4 - 0.5,
                         (1 + 0.5) / 4 - 0.5]], dtype=np.float32)

        result = sample_voxel_attrs(pos, coords, attrs, grid_size)
        np.testing.assert_allclose(result, 0.5, atol=0.1)

    def test_output_shape(self):
        grid_size = 4
        coords = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int32)
        attrs = np.random.rand(2, 6).astype(np.float32)
        positions = np.random.rand(10, 3).astype(np.float32) - 0.5

        result = sample_voxel_attrs(positions, coords, attrs, grid_size)
        assert result.shape == (10, 6)

    def test_mlx_matches_numpy(self):
        """Vectorized sampler should match numpy reference within tolerance."""
        from trellmlx.texture_bake import sample_voxel_attrs_fast

        grid_size = 8
        np.random.seed(42)
        n_voxels = 100
        coords = np.random.randint(0, grid_size, (n_voxels, 3)).astype(np.int32)
        # Deduplicate coords
        coords = np.unique(coords, axis=0)
        attrs = np.random.rand(len(coords), 6).astype(np.float32)
        positions = np.random.rand(50, 3).astype(np.float32) - 0.5

        result_np = sample_voxel_attrs(positions, coords, attrs, grid_size)
        result_mlx = sample_voxel_attrs_fast(positions, coords, attrs, grid_size)

        np.testing.assert_allclose(result_np, result_mlx, atol=1e-4)
