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

    def test_mlx_orthogonal_aspect_faces(self):
        """Faces with orthogonal aspect ratios should not OOM or produce wrong results.

        Regression test for budget estimate bug: tall-narrow + wide-short faces
        would underestimate the GPU tensor size when using per-face area instead
        of running max_w * max_h.
        """
        from trellmlx.texture_bake import rasterize_uv_mlx

        # Create faces with deliberately orthogonal UV bboxes
        # Face 0: tall narrow strip (UV x in [0, 0.02], y in [0, 0.8])
        # Face 1: wide short strip (UV x in [0.1, 0.9], y in [0.1, 0.12])
        uvs = np.array([
            [0.0, 0.0], [0.02, 0.0], [0.01, 0.8],   # face 0: tall
            [0.1, 0.1], [0.9, 0.1], [0.5, 0.12],     # face 1: wide
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)

        # Should not OOM or crash
        mask, face_idx, bary = rasterize_uv_mlx(uvs, faces, texture_size=64)
        assert mask.shape == (64, 64)
        # Both faces should produce some covered pixels
        assert (face_idx == 0).any(), "Tall face produced no coverage"
        assert (face_idx == 1).any(), "Wide face produced no coverage"

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

    def test_fast_sampler_handles_empty_inputs_without_min_assert(self):
        """Empty voxel arrays should return an empty-shaped result, not call min()."""
        from trellmlx.texture_bake import sample_voxel_attrs_fast

        positions = np.zeros((0, 3), dtype=np.float32)
        coords = np.zeros((0, 3), dtype=np.int32)
        attrs = np.zeros((0, 6), dtype=np.float32)

        result = sample_voxel_attrs_fast(positions, coords, attrs, grid_size=4)

        assert result.shape == (0, 6)
        assert result.dtype == np.float32

    def test_fast_sampler_matches_numpy_with_negative_query_corners(self):
        """Out-of-grid query corners should behave like sparse misses, not assert."""
        from trellmlx.texture_bake import sample_voxel_attrs_fast

        grid_size = 4
        coords = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int32)
        attrs = np.array([
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        ], dtype=np.float32)
        positions = np.array([[-0.5, -0.5, -0.5]], dtype=np.float32)

        result_np = sample_voxel_attrs(positions, coords, attrs, grid_size)
        result_fast = sample_voxel_attrs_fast(positions, coords, attrs, grid_size)

        np.testing.assert_allclose(result_fast, result_np, atol=1e-5)


class TestBakeTexture:
    def test_bake_texture_uses_fast_raster_and_sampler(self, monkeypatch):
        """The main bake route should exercise the optimized helpers."""
        import trellmlx.texture_bake as texture_bake

        vertices, faces, uvs = _make_uv_quad()
        calls = []

        def fake_rasterize_uv_mlx(got_uvs, got_faces, texture_size):
            calls.append(("raster", texture_size))
            mask = np.zeros((texture_size, texture_size), dtype=bool)
            mask[0, 0] = True
            face_idx = np.zeros((texture_size, texture_size), dtype=np.int64)
            bary = np.zeros((texture_size, texture_size, 3), dtype=np.float32)
            bary[0, 0] = [1.0, 0.0, 0.0]
            return mask, face_idx, bary

        def fake_sample_voxel_attrs_fast(positions, voxel_coords, voxel_attrs, grid_size):
            calls.append(("sample", len(positions)))
            return np.array([[0.2, 0.4, 0.6, 0.8, 0.5, 1.0]], dtype=np.float32)

        def fake_inpaint_texture(image, mask, radius=3):
            return image

        monkeypatch.setattr(texture_bake, "rasterize_uv_mlx", fake_rasterize_uv_mlx)
        monkeypatch.setattr(texture_bake, "sample_voxel_attrs_fast", fake_sample_voxel_attrs_fast)
        monkeypatch.setattr(texture_bake, "inpaint_texture", fake_inpaint_texture)

        base_color, metallic_roughness, _ = texture_bake.bake_texture(
            vertices,
            faces,
            uvs,
            np.arange(len(vertices)),
            np.array([[0, 0, 0]], dtype=np.int32),
            np.array([[0.2, 0.4, 0.6, 0.8, 0.5, 1.0]], dtype=np.float32),
            grid_size=4,
            texture_size=4,
            backend="gpu",
        )

        assert calls == [("raster", 4), ("sample", 1)]
        assert base_color.shape == (4, 4, 4)
        assert metallic_roughness.shape == (4, 4, 3)


class TestBakeTextureBackend:
    """Tests for bake_texture backend selection (gpu vs cpu)."""

    @staticmethod
    def _make_textured_mesh():
        """Create a minimal mesh + voxel grid for bake_texture testing.

        Returns vertices, faces, uvs, vmapping, voxel_coords, voxel_attrs,
        grid_size suitable for bake_texture.
        """
        # Simple quad in world space
        vertices = np.array([
            [-0.25, -0.25, 0.0],
            [ 0.25, -0.25, 0.0],
            [ 0.25,  0.25, 0.0],
            [-0.25,  0.25, 0.0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        uvs = np.array([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
        ], dtype=np.float32)
        vmapping = np.arange(4, dtype=np.int32)

        # Fill a small voxel grid around the mesh
        grid_size = 4
        coords = []
        attrs = []
        for z in range(grid_size):
            for y in range(grid_size):
                for x in range(grid_size):
                    coords.append([z, y, x])
                    # Predictable PBR: RGB = normalized coords, metallic=0.1,
                    # roughness=0.5, alpha=1.0
                    attrs.append([x / grid_size, y / grid_size, z / grid_size,
                                  0.1, 0.5, 1.0])
        voxel_coords = np.array(coords, dtype=np.int32)
        voxel_attrs = np.array(attrs, dtype=np.float32)
        return vertices, faces, uvs, vmapping, voxel_coords, voxel_attrs, grid_size

    def test_bake_texture_accepts_backend_parameter(self):
        """bake_texture should accept a backend='gpu' parameter."""
        from trellmlx.texture_bake import bake_texture
        mesh = self._make_textured_mesh()
        vertices, faces, uvs, vmapping, vc, va, gs = mesh

        # Should not raise — backend parameter must be accepted
        bc, mr, _ = bake_texture(
            vertices, faces, uvs, vmapping, vc, va, gs,
            texture_size=8, backend="gpu",
        )
        assert bc.shape == (8, 8, 4)
        assert mr.shape == (8, 8, 3)

    def test_bake_texture_gpu_produces_nonzero_output(self):
        """GPU backend should produce actual texture content, not zeros."""
        from trellmlx.texture_bake import bake_texture
        mesh = self._make_textured_mesh()
        vertices, faces, uvs, vmapping, vc, va, gs = mesh

        bc, mr, _ = bake_texture(
            vertices, faces, uvs, vmapping, vc, va, gs,
            texture_size=16, backend="gpu",
        )
        # Base color should have nonzero content in covered region
        assert bc[:, :, :3].sum() > 0, "GPU backend produced empty base color"

    def test_bake_texture_gpu_matches_cpu(self):
        """GPU and CPU backends should produce similar textures.

        Allows small per-pixel differences from edge rasterization,
        but the overall texture must be very close.
        """
        from trellmlx.texture_bake import bake_texture
        mesh = self._make_textured_mesh()
        vertices, faces, uvs, vmapping, vc, va, gs = mesh
        sz = 16

        bc_cpu, mr_cpu, _ = bake_texture(
            vertices, faces, uvs, vmapping, vc, va, gs,
            texture_size=sz, backend="cpu",
        )
        bc_gpu, mr_gpu, _ = bake_texture(
            vertices, faces, uvs, vmapping, vc, va, gs,
            texture_size=sz, backend="gpu",
        )

        # Overall texture similarity: mean absolute difference < 5 uint8 levels
        # (allows for edge pixel rasterization differences)
        bc_diff = np.abs(bc_cpu.astype(np.float32) - bc_gpu.astype(np.float32))
        assert bc_diff.mean() < 5.0, (
            f"GPU/CPU base color mean diff {bc_diff.mean():.1f} too large"
        )

    def test_bake_texture_default_backend_produces_texture(self):
        """Default backend (no parameter) should keep API callers working."""
        from trellmlx.texture_bake import bake_texture
        mesh = self._make_textured_mesh()
        vertices, faces, uvs, vmapping, vc, va, gs = mesh

        # No backend param should still produce a texture.
        bc, mr, _ = bake_texture(
            vertices, faces, uvs, vmapping, vc, va, gs,
            texture_size=8,
        )
        assert bc.shape == (8, 8, 4)

    def test_bake_texture_default_backend_uses_gpu_path(self, monkeypatch):
        """The API default should preserve the previous GPU-accelerated route."""
        import trellmlx.texture_bake as texture_bake

        mesh = self._make_textured_mesh()
        vertices, faces, uvs, vmapping, vc, va, gs = mesh
        calls = []

        def fake_rasterize_uv_mlx(got_uvs, got_faces, texture_size):
            calls.append(("gpu_raster", texture_size))
            mask = np.zeros((texture_size, texture_size), dtype=bool)
            mask[0, 0] = True
            face_idx = np.zeros((texture_size, texture_size), dtype=np.int64)
            bary = np.zeros((texture_size, texture_size, 3), dtype=np.float32)
            bary[0, 0] = [1.0, 0.0, 0.0]
            return mask, face_idx, bary

        def fake_sample_voxel_attrs_fast(positions, voxel_coords, voxel_attrs, grid_size):
            calls.append(("gpu_sample", len(positions)))
            return np.array([[0.2, 0.4, 0.6, 0.8, 0.5, 1.0]], dtype=np.float32)

        def forbidden_cpu_raster(*args, **kwargs):
            raise AssertionError("default bake_texture() unexpectedly used CPU rasterizer")

        def forbidden_cpu_sample(*args, **kwargs):
            raise AssertionError("default bake_texture() unexpectedly used CPU sampler")

        monkeypatch.setattr(texture_bake, "rasterize_uv_mlx", fake_rasterize_uv_mlx)
        monkeypatch.setattr(texture_bake, "sample_voxel_attrs_fast", fake_sample_voxel_attrs_fast)
        monkeypatch.setattr(texture_bake, "rasterize_uv", forbidden_cpu_raster)
        monkeypatch.setattr(texture_bake, "sample_voxel_attrs", forbidden_cpu_sample)
        monkeypatch.setattr(texture_bake, "inpaint_texture", lambda image, mask, radius=3: image)

        bc, mr, _ = texture_bake.bake_texture(
            vertices, faces, uvs, vmapping, vc, va, gs,
            texture_size=4,
        )

        assert calls == [("gpu_raster", 4), ("gpu_sample", 1)]
        assert bc.shape == (4, 4, 4)
        assert mr.shape == (4, 4, 3)

    def test_bake_texture_rejects_invalid_backend(self):
        """Invalid backend values should raise ValueError, not silently fallback."""
        from trellmlx.texture_bake import bake_texture
        mesh = self._make_textured_mesh()
        vertices, faces, uvs, vmapping, vc, va, gs = mesh

        with pytest.raises(ValueError, match="backend must be"):
            bake_texture(
                vertices, faces, uvs, vmapping, vc, va, gs,
                texture_size=8, backend="metal",
            )

    def test_sample_voxel_attrs_fast_used_in_gpu_backend(self):
        """GPU backend should use sample_voxel_attrs_fast, not the dict version."""
        from unittest.mock import patch
        from trellmlx.texture_bake import bake_texture
        mesh = self._make_textured_mesh()
        vertices, faces, uvs, vmapping, vc, va, gs = mesh

        with patch("trellmlx.texture_bake.sample_voxel_attrs_fast",
                    wraps=__import__("trellmlx.texture_bake",
                                     fromlist=["sample_voxel_attrs_fast"]).sample_voxel_attrs_fast
                    ) as mock_fast:
            bake_texture(
                vertices, faces, uvs, vmapping, vc, va, gs,
                texture_size=8, backend="gpu",
            )
            assert mock_fast.called, "GPU backend did not call sample_voxel_attrs_fast"


class TestBakeTextureInpaintRadius:
    """Scalar PBR channels should use inpainting radius 1, matching reference."""

    def test_metallic_roughness_inpaint_radius_matches_reference(self):
        """Inpaint radius for scalar channels should be 1, not 3."""
        from unittest.mock import patch, call
        from trellmlx.texture_bake import bake_texture
        mesh = TestBakeTextureBackend._make_textured_mesh()
        vertices, faces, uvs, vmapping, vc, va, gs = mesh

        inpaint_calls = []
        original_inpaint = __import__("trellmlx.texture_bake",
                                       fromlist=["inpaint_texture"]).inpaint_texture

        def tracking_inpaint(image, mask, radius=3):
            inpaint_calls.append(radius)
            return original_inpaint(image, mask, radius=radius)

        with patch("trellmlx.texture_bake.inpaint_texture", side_effect=tracking_inpaint):
            bake_texture(
                vertices, faces, uvs, vmapping, vc, va, gs,
                texture_size=8,
            )
        # base_color RGB: radius 3, alpha: radius 1, metallic_roughness: radius 1
        assert inpaint_calls == [3, 1, 1], (
            f"Expected inpaint radii [3, 1, 1], got {inpaint_calls}"
        )


class TestBakeTextureAlphaDetection:
    """bake_texture should return alpha mode based on baked alpha values."""

    def test_returns_alpha_mode(self):
        """bake_texture should return a third value: alpha_mode string."""
        from trellmlx.texture_bake import bake_texture
        mesh = TestBakeTextureBackend._make_textured_mesh()
        vertices, faces, uvs, vmapping, vc, va, gs = mesh

        result = bake_texture(
            vertices, faces, uvs, vmapping, vc, va, gs,
            texture_size=8,
        )
        assert len(result) == 3, f"Expected 3 return values, got {len(result)}"
        bc, mr, alpha_mode = result
        assert alpha_mode in ("OPAQUE", "BLEND")
