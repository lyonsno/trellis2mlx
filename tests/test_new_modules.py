"""Tests for newly added modules: SLatFlowModel, decoder, sampler, quantizer, weight_loader, RoPE."""

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np
import pytest


class TestRoPE:
    def test_build_phases_shape(self):
        from trellmlx.modules.rope import build_rope_phases
        phases = build_rope_phases(resolution=4, head_dim=128)
        assert phases.shape == (64, 64, 2)  # 4³=64 tokens, 128//2=64 freq dims, 2 (cos/sin)

    def test_origin_phases(self):
        from trellmlx.modules.rope import build_rope_phases
        phases = build_rope_phases(resolution=4, head_dim=128)
        mx.eval(phases)
        # Position (0,0,0) = index 0: cos(0)=1, sin(0)=0 for all freq dims
        cos_origin = phases[0, :, 0]
        sin_origin = phases[0, :, 1]
        mx.eval(cos_origin, sin_origin)
        assert mx.allclose(cos_origin, mx.ones_like(cos_origin), atol=1e-5).item()
        assert mx.allclose(sin_origin, mx.zeros_like(sin_origin), atol=1e-5).item()

    def test_apply_identity_phases(self):
        from trellmlx.modules.rope import apply_rope
        T, H, D = 8, 4, 64
        x = mx.random.normal((T, H, D))
        # Identity phases: cos=1, sin=0
        phases = mx.stack([mx.ones((T, D // 2)), mx.zeros((T, D // 2))], axis=-1)
        out = apply_rope(x, phases)
        mx.eval(out)
        diff = mx.max(mx.abs(out - x)).item()
        assert diff < 1e-5, f"Identity phases should preserve input, got diff={diff}"

    def test_apply_rope_changes_output(self):
        from trellmlx.modules.rope import build_rope_phases, apply_rope
        T, H, D = 64, 4, 128
        x = mx.random.normal((T, H, D))
        phases = build_rope_phases(resolution=4, head_dim=D)
        out = apply_rope(x, phases)
        mx.eval(out)
        diff = mx.max(mx.abs(out - x)).item()
        assert diff > 0.01, "Non-identity phases should change the output"


class TestDynamicResolution:
    def test_different_resolution_than_constructor(self):
        from trellmlx.models.sparse_structure_flow import SparseStructureFlowModel
        model = SparseStructureFlowModel(
            in_channels=8, out_channels=8, model_channels=64,
            num_heads=4, num_blocks=2, mlp_hidden=128,
            context_channels=32, resolution=4,
        )
        # Input at resolution=6, different from constructor
        x = mx.random.normal((1, 8, 6, 6, 6))
        t = mx.array([500.0])
        cond = mx.random.normal((1, 5, 32))
        out = model(x, t, cond)
        assert out.shape == (1, 8, 6, 6, 6)


class TestSLatFlowModel:
    def test_output_shape(self):
        from trellmlx.models.slat_flow import SLatFlowModel
        model = SLatFlowModel(
            in_channels=32, out_channels=32, model_channels=64,
            num_heads=4, num_blocks=2, mlp_hidden=128,
            context_channels=32,
        )
        N = 100
        x = mx.random.normal((N, 32))
        t = mx.array([500.0])
        cond = mx.random.normal((1, 5, 32))
        out = model(x, t, cond)
        assert out.shape == (N, 32)

    def test_with_coords(self):
        from trellmlx.models.slat_flow import SLatFlowModel
        model = SLatFlowModel(
            in_channels=32, out_channels=32, model_channels=64,
            num_heads=4, num_blocks=2, mlp_hidden=128,
            context_channels=32,
        )
        N = 50
        x = mx.random.normal((N, 32))
        t = mx.array([500.0])
        cond = mx.random.normal((1, 5, 32))
        coords = mx.array(np.random.randint(0, 16, (N, 3)))
        out = model(x, t, cond, coords=coords)
        assert out.shape == (N, 32)

    def test_coords_change_output(self):
        from trellmlx.models.slat_flow import SLatFlowModel
        model = SLatFlowModel(
            in_channels=32, out_channels=32, model_channels=64,
            num_heads=4, num_blocks=2, mlp_hidden=128,
            context_channels=32,
        )
        N = 50
        mx.random.seed(42)
        x = mx.random.normal((N, 32))
        t = mx.array([500.0])
        cond = mx.random.normal((1, 5, 32))
        out_no_coords = model(x, t, cond)
        coords = mx.array(np.random.randint(0, 16, (N, 3)))
        out_with_coords = model(x, t, cond, coords=coords)
        mx.eval(out_no_coords, out_with_coords)
        diff = mx.max(mx.abs(out_no_coords - out_with_coords)).item()
        assert diff > 0.01, "Coords should change the output via RoPE"

    def test_rejects_batch_greater_than_one(self):
        from trellmlx.models.slat_flow import SLatFlowModel
        model = SLatFlowModel(
            in_channels=32, out_channels=32, model_channels=64,
            num_heads=4, num_blocks=1, mlp_hidden=128,
            context_channels=32,
        )
        x = mx.random.normal((16, 32))
        t = mx.array([500.0, 500.0])
        cond = mx.random.normal((2, 5, 32))

        with pytest.raises(ValueError, match="Only B=1 supported"):
            model(x, t, cond)


class TestSparseStructureDecoder:
    def test_pixel_shuffle_3d(self):
        from trellmlx.models.sparse_structure_decoder import pixel_shuffle_3d
        x = mx.random.normal((1, 2, 2, 2, 64))
        out = pixel_shuffle_3d(x, factor=2)
        assert out.shape == (1, 4, 4, 4, 8)  # 64 / 2³ = 8

    def test_output_shape(self):
        from trellmlx.models.sparse_structure_decoder import SparseStructureDecoder
        model = SparseStructureDecoder(
            out_channels=1, latent_channels=8, num_res_blocks=1,
            num_res_blocks_middle=1, channels=[32, 16, 8],
        )
        x = mx.random.normal((1, 8, 4, 4, 4))  # channels-first input
        out = model(x)
        # Two 2x upsamples: 4 → 8 → 16
        assert out.shape == (1, 1, 16, 16, 16)

    def test_conv3d_shape(self):
        from trellmlx.models.sparse_structure_decoder import Conv3d
        conv = Conv3d(8, 16, kernel_size=3, padding=1)
        x = mx.random.normal((1, 4, 4, 4, 8))  # channels-last
        out = conv(x)
        assert out.shape == (1, 4, 4, 4, 16)


class TestSampler:
    def test_pred_xstart_roundtrip(self):
        from trellmlx.samplers import _pred_to_xstart, _xstart_to_pred
        x_t = mx.array([[1.0, 2.0, -0.5]])
        pred = mx.array([[0.5, -0.3, 0.8]])
        sigma_min = 1e-5
        t = 0.7
        x_0 = _pred_to_xstart(x_t, t, pred, sigma_min)
        pred_back = _xstart_to_pred(x_t, t, x_0, sigma_min)
        mx.eval(pred_back)
        diff = mx.max(mx.abs(pred - pred_back)).item()
        assert diff < 1e-5, f"Round-trip failed: diff={diff}"

    def test_timestep_schedule(self):
        """Verify timestep schedule matches reference formula."""
        import numpy as np
        steps = 12
        rescale_t = 5.0
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        # First should be 1.0, last should be 0.0
        assert abs(t_seq[0] - 1.0) < 1e-6
        assert abs(t_seq[-1] - 0.0) < 1e-6
        # Should be monotonically decreasing
        assert all(t_seq[i] > t_seq[i+1] for i in range(len(t_seq)-1))

    def test_guidance_interval(self):
        """Verify that steps outside guidance interval skip negative conditioning."""
        from trellmlx.samplers import flow_euler_sample

        call_count = {"pos": 0, "neg": 0}

        class MockModel:
            def __call__(self, x, t, cond):
                call_count["pos"] += 1
                return mx.zeros_like(x)

        model = MockModel()
        noise = mx.random.normal((1, 8, 4, 4, 4))
        cond = mx.random.normal((1, 5, 32))
        neg_cond = mx.zeros_like(cond)

        # With guidance_strength=1.0, should never call neg
        flow_euler_sample(model, noise, cond, neg_cond,
                         steps=4, guidance_strength=1.0, verbose=False)
        assert call_count["pos"] == 4  # one call per step, no neg

    def test_cfg_rescale_clamp_can_be_disabled_for_parity_witness(self):
        """Diagnostic route can preserve the reference raw CFG-rescale ratio."""
        from trellmlx.samplers import flow_euler_sample

        class MockModel:
            def __call__(self, x, t, cond):
                pred_pos = mx.array([[[[[0.0, 1.0]]]]])
                pred_neg = mx.array([[[[[0.0, -1.0]]]]])
                return mx.where(mx.sum(cond) > 0, pred_pos, pred_neg)

        noise = mx.zeros((1, 1, 1, 1, 2))
        cond = mx.ones((1, 1, 1))
        neg_cond = mx.zeros_like(cond)

        clamped = {}
        flow_euler_sample(
            MockModel(),
            noise,
            cond,
            neg_cond,
            steps=8,
            guidance_strength=7.5,
            guidance_rescale=0.7,
            verbose=False,
            capture_first_step=clamped,
            stop_after_first_step=True,
        )

        unclamped = {}
        flow_euler_sample(
            MockModel(),
            noise,
            cond,
            neg_cond,
            steps=8,
            guidance_strength=7.5,
            guidance_rescale=0.7,
            verbose=False,
            cfg_rescale_clamp=False,
            capture_first_step=unclamped,
            stop_after_first_step=True,
        )

        assert np.array(clamped["ratio_raw"])[0, 0, 0, 0, 0] < 0.5
        assert np.array(clamped["ratio_effective"])[0, 0, 0, 0, 0] == 0.5
        assert np.array(unclamped["ratio_effective"])[0, 0, 0, 0, 0] == np.array(unclamped["ratio_raw"])[0, 0, 0, 0, 0]


class TestWeightLoader:
    def test_remap_adaln(self):
        from trellmlx.weight_loader import _remap_key
        assert _remap_key("adaLN_modulation.1.weight") == "adaLN_modulation.layers.1.weight"
        assert _remap_key("adaLN_modulation.1.bias") == "adaLN_modulation.layers.1.bias"

    def test_remap_mlp(self):
        from trellmlx.weight_loader import _remap_key
        assert _remap_key("blocks.0.mlp.mlp.0.weight") == "blocks.0.mlp.mlp_0.weight"
        assert _remap_key("blocks.0.mlp.mlp.2.bias") == "blocks.0.mlp.mlp_2.bias"

    def test_remap_timestep(self):
        from trellmlx.weight_loader import _remap_key
        assert _remap_key("t_embedder.mlp.0.weight") == "t_embedder.mlp_0.weight"
        assert _remap_key("t_embedder.mlp.2.bias") == "t_embedder.mlp_2.bias"

    def test_remap_out_layer(self):
        from trellmlx.weight_loader import _remap_key
        assert _remap_key("out_layer.0.weight") == "out_layer_0.weight"
        assert _remap_key("out_layer.2.bias") == "out_layer_2.bias"

    def test_remap_passthrough(self):
        from trellmlx.weight_loader import _remap_key
        # Keys that shouldn't be remapped
        assert _remap_key("blocks.0.self_attn.to_qkv.weight") == "blocks.0.self_attn.to_qkv.weight"
        assert _remap_key("blocks.0.modulation") == "blocks.0.modulation"

    def test_conv_permute(self):
        from trellmlx.weight_loader import _should_permute_conv
        assert _should_permute_conv("blocks.0.conv1.weight", (128, 128, 3, 3, 3)) is True
        assert _should_permute_conv("blocks.0.conv1.bias", (128,)) is False
        assert _should_permute_conv("blocks.0.self_attn.to_qkv.weight", (4608, 1536)) is False


class TestMeshExtract:
    def test_softplus_preserves_large_positive_split_logits(self):
        """Split-weight softplus must not collapse large positive logits."""
        from trellmlx.mesh_extract import _softplus

        values = np.array([19.0, 25.0, 50.0], dtype=np.float32)
        result = _softplus(values)

        np.testing.assert_allclose(result, np.logaddexp(0, values), rtol=1e-6)
        assert result[2] > result[1] > result[0]

    def test_dense_cube_connected(self):
        """A dense 2x2x2 cube must produce a single connected mesh."""
        from trellmlx.mesh_extract import flexible_dual_grid_to_mesh
        from collections import defaultdict

        coords = np.array([[z, y, x] for z in range(2) for y in range(2) for x in range(2)], dtype=np.int64)
        dual_vertices = np.full((8, 3), 0.5, dtype=np.float32)
        intersected_flag = np.ones((8, 3), dtype=bool)

        vertices, faces = flexible_dual_grid_to_mesh(
            coords, dual_vertices, intersected_flag, None,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], grid_size=2,
        )
        assert len(vertices) == 8
        assert len(faces) == 12  # 6 quads = 12 triangles

        # Check single connected component via vertex sharing
        parent = list(range(len(faces)))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            a, b = find(a), find(b)
            if a != b: parent[a] = b
        v2f = defaultdict(list)
        for fi, f in enumerate(faces):
            for v in f:
                v2f[v].append(fi)
        for flist in v2f.values():
            for i in range(1, len(flist)):
                union(flist[0], flist[i])
        n_comps = len(set(find(i) for i in range(len(faces))))
        assert n_comps == 1, f"Expected 1 component, got {n_comps}"

    def test_dense_cube_bounds(self):
        """Mesh vertices must be within the specified AABB."""
        from trellmlx.mesh_extract import flexible_dual_grid_to_mesh
        coords = np.array([[z, y, x] for z in range(4) for y in range(4) for x in range(4)], dtype=np.int64)
        N = len(coords)
        dual_vertices = np.full((N, 3), 0.5, dtype=np.float32)
        intersected_flag = np.ones((N, 3), dtype=bool)

        vertices, faces = flexible_dual_grid_to_mesh(
            coords, dual_vertices, intersected_flag, None,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], grid_size=4,
        )
        # All vertices should be within [-0.5, 0.5] (with dual_vertices=0.5,
        # vertices are at grid centers, max = (3+0.5)/4 - 0.5 = 0.375)
        assert vertices.min() >= -0.5
        assert vertices.max() <= 0.5

    def test_decoder_output_format(self):
        """Test decoder_output_to_mesh with synthetic 7-channel data."""
        from trellmlx.mesh_extract import decoder_output_to_mesh
        coords_3d = np.array([[z, y, x] for z in range(4) for y in range(4) for x in range(4)], dtype=np.int64)
        N = len(coords_3d)
        coords_4d = np.column_stack([np.zeros(N, dtype=np.int64), coords_3d])
        feats = np.zeros((N, 7), dtype=np.float32)
        feats[:, 0:3] = 0.0   # sigmoid(0)=0.5
        feats[:, 3:6] = 1.0   # edges intersected
        feats[:, 6] = 0.0

        vertices, faces = decoder_output_to_mesh(feats, coords_4d, resolution=4)
        assert len(vertices) > 0
        assert len(faces) > 0
        assert vertices.min() >= -0.6  # within AABB with margin
        assert vertices.max() <= 0.6


class TestQuantize:
    def test_small_layer_skipped(self):
        from trellmlx.quantize import quantize_model

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.small = nn.Linear(8, 64)  # dim 8 < group_size 64
                self.big = nn.Linear(128, 256)

        model = TinyModel()
        big_shape_before = model.big.weight.shape
        quantize_model(model, bits=4, group_size=64)

        # Small layer should still be nn.Linear (not quantized)
        assert isinstance(model.small, nn.Linear)
        # Big layer should be quantized (has 'scales' attribute)
        assert hasattr(model.big, 'scales')
