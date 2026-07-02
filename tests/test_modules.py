"""Tests for trellmlx core modules."""

import math
import pytest
import mlx.core as mx
import mlx.nn as nn


class TestLayerNorm32:
    def test_output_shape(self):
        from trellmlx.modules.norm import LayerNorm32
        ln = LayerNorm32(128)
        x = mx.random.normal((4, 128))
        out = ln(x)
        assert out.shape == (4, 128)

    def test_normalized_stats(self):
        from trellmlx.modules.norm import LayerNorm32
        ln = LayerNorm32(256)
        x = mx.random.normal((8, 256)) * 5 + 3  # non-zero mean, large std
        out = ln(x)
        mx.eval(out)
        # After LN, each row should have ~0 mean and ~1 std
        mean = mx.mean(out, axis=-1)
        var = mx.var(out, axis=-1)
        mx.eval(mean, var)
        assert mx.all(mx.abs(mean) < 1e-5).item(), f"Mean not ~0: {mean}"
        assert mx.all(mx.abs(var - 1.0) < 1e-4).item(), f"Var not ~1: {var}"

    def test_fp32_accumulation(self):
        """LN should accumulate in fp32 even with fp16 input."""
        from trellmlx.modules.norm import LayerNorm32
        ln = LayerNorm32(64)
        x = mx.random.normal((4, 64)).astype(mx.float16)
        out = ln(x)
        # Output should be back in fp16
        assert out.dtype == mx.float16

    def test_affine_mode(self):
        from trellmlx.modules.norm import LayerNorm32
        ln = LayerNorm32(32, affine=True)
        x = mx.random.normal((2, 32))
        out = ln(x)
        assert out.shape == (2, 32)


class TestMultiHeadRMSNorm:
    def test_output_shape(self):
        from trellmlx.modules.attention import MultiHeadRMSNorm
        rn = MultiHeadRMSNorm(head_dim=128, num_heads=12)
        x = mx.random.normal((4, 12, 128))  # [T, H, D]
        out = rn(x)
        assert out.shape == (4, 12, 128)

    def test_gamma_shape(self):
        from trellmlx.modules.attention import MultiHeadRMSNorm
        rn = MultiHeadRMSNorm(head_dim=128, num_heads=12)
        assert rn.gamma.shape == (12, 128), f"Expected [12, 128], got {rn.gamma.shape}"

    def test_unit_l2_norm(self):
        """After normalization with gamma=1, each vector should have L2 norm = sqrt(dim)."""
        from trellmlx.modules.attention import MultiHeadRMSNorm
        rn = MultiHeadRMSNorm(head_dim=64, num_heads=4)
        x = mx.random.normal((8, 4, 64)) * 10
        out = rn(x)
        mx.eval(out)
        # normalize to unit L2, then multiply by gamma(=1) * scale(=sqrt(64))
        # Output L2 norm = sqrt(64) ≈ 8.0
        l2 = mx.sqrt(mx.sum(out * out, axis=-1))
        mx.eval(l2)
        expected = 64 ** 0.5  # sqrt(dim) = 8.0
        assert mx.all(mx.abs(l2 - expected) < 0.5).item(), f"L2 not ~{expected}: {l2}"


class TestScaledDotProductAttention:
    def test_self_attention_shape(self):
        from trellmlx.modules.attention import scaled_dot_product_attention
        B, H, T, D = 1, 4, 16, 32
        q = mx.random.normal((B, H, T, D))
        k = mx.random.normal((B, H, T, D))
        v = mx.random.normal((B, H, T, D))
        out = scaled_dot_product_attention(q, k, v)
        assert out.shape == (B, H, T, D)

    def test_cross_attention_shape(self):
        from trellmlx.modules.attention import scaled_dot_product_attention
        B, H, T_q, T_kv, D = 1, 4, 8, 16, 32
        q = mx.random.normal((B, H, T_q, D))
        k = mx.random.normal((B, H, T_kv, D))
        v = mx.random.normal((B, H, T_kv, D))
        out = scaled_dot_product_attention(q, k, v)
        assert out.shape == (B, H, T_q, D)

    def test_attention_concentrates_on_matching_key(self):
        """When one key matches the query exactly, output should be close to its value."""
        from trellmlx.modules.attention import scaled_dot_product_attention
        B, H, D = 1, 1, 64
        q = mx.zeros((B, H, 1, D))
        q = q.at[:, :, :, 0].add(10.0)  # strong signal in dim 0
        k = mx.zeros((B, H, 3, D))
        k = k.at[:, :, 0, 0].add(10.0)  # first key matches
        v = mx.zeros((B, H, 3, D))
        v = v.at[:, :, 0, :].add(1.0)   # first value is all 1s
        out = scaled_dot_product_attention(q, k, v)
        mx.eval(out)
        # Output should be close to v[0] = all 1s
        assert mx.all(out[0, 0, 0] > 0.9).item()

    def test_masking(self):
        from trellmlx.modules.attention import scaled_dot_product_attention
        B, H, T, D = 1, 1, 4, 8
        q = mx.random.normal((B, H, T, D))
        k = mx.random.normal((B, H, T, D))
        v = mx.random.normal((B, H, T, D))
        # Mask out everything except first key
        mask = mx.where(
            mx.arange(T)[None, None, None, :] == 0,
            mx.array(0.0),
            mx.array(float("-inf")),
        )
        mask = mx.broadcast_to(mask, (B, 1, T, T))
        out = scaled_dot_product_attention(q, k, v, mask=mask)
        mx.eval(out)
        # All queries should attend only to first key → output = v[:,:,0,:]
        for ti in range(T):
            diff = mx.abs(out[0, 0, ti] - v[0, 0, 0])
            mx.eval(diff)
            assert mx.all(diff < 1e-5).item(), f"Query {ti} didn't attend to key 0"


class TestVarLenAttention:
    def test_single_sequence(self):
        from trellmlx.modules.attention import varlen_attention_padded
        T, H, D = 16, 4, 32
        q = mx.random.normal((T, H, D))
        k = mx.random.normal((T, H, D))
        v = mx.random.normal((T, H, D))
        out = varlen_attention_padded(q, k, v, [T], [T])
        assert out.shape == (T, H, D)

    def test_two_sequences(self):
        from trellmlx.modules.attention import varlen_attention_padded
        H, D = 4, 32
        T1, T2 = 8, 12
        T_total = T1 + T2
        q = mx.random.normal((T_total, H, D))
        k = mx.random.normal((T_total, H, D))
        v = mx.random.normal((T_total, H, D))
        out = varlen_attention_padded(q, k, v, [T1, T2], [T1, T2])
        assert out.shape == (T_total, H, D)


class TestTimestepEmbedder:
    def test_output_shape(self):
        from trellmlx.models.sparse_structure_flow import TimestepEmbedder
        te = TimestepEmbedder(1536)
        t = mx.array([0.0, 0.5, 1.0])
        out = te(t)
        assert out.shape == (3, 1536)

    def test_different_timesteps_different_embeddings(self):
        from trellmlx.models.sparse_structure_flow import TimestepEmbedder
        te = TimestepEmbedder(256)
        t = mx.array([0.0, 500.0, 999.0])
        out = te(t)
        mx.eval(out)
        # Different timesteps should produce different embeddings
        d01 = mx.sum(mx.abs(out[0] - out[1])).item()
        d02 = mx.sum(mx.abs(out[0] - out[2])).item()
        assert d01 > 0.1, "t=0 and t=500 produced same embedding"
        assert d02 > 0.1, "t=0 and t=999 produced same embedding"


class TestSparseStructureFlowModel:
    def test_output_shape(self):
        from trellmlx.models.sparse_structure_flow import SparseStructureFlowModel
        model = SparseStructureFlowModel(
            in_channels=8, out_channels=8, model_channels=64,
            num_heads=4, num_blocks=2, mlp_hidden=128,
            context_channels=32, resolution=4,
        )
        x = mx.random.normal((1, 8, 4, 4, 4))
        t = mx.array([500.0])
        cond = mx.random.normal((1, 5, 32))
        out = model(x, t, cond)
        assert out.shape == (1, 8, 4, 4, 4)

    def test_final_layernorm_uses_reference_epsilon(self, monkeypatch):
        import trellmlx.models.sparse_structure_flow as sparse_flow

        seen_eps = []
        original_layernorm = sparse_flow._layernorm_noaffine

        def capture_layernorm(x, eps=1e-6):
            seen_eps.append(eps)
            return original_layernorm(x, eps=eps)

        monkeypatch.setattr(sparse_flow, "_layernorm_noaffine", capture_layernorm)
        model = sparse_flow.SparseStructureFlowModel(
            in_channels=8, out_channels=8, model_channels=16,
            num_heads=4, num_blocks=0, mlp_hidden=32,
            context_channels=8, resolution=2,
        )

        x = mx.random.normal((1, 8, 2, 2, 2))
        t = mx.array([1000.0])
        cond = mx.random.normal((1, 5, 8))
        out = model(x, t, cond)
        mx.eval(out)

        assert seen_eps == [1e-5]

    def test_parameter_count_small(self):
        """Small model parameter count should be predictable."""
        import mlx.utils
        from trellmlx.models.sparse_structure_flow import SparseStructureFlowModel
        model = SparseStructureFlowModel(
            in_channels=8, out_channels=8, model_channels=64,
            num_heads=4, num_blocks=2, mlp_hidden=128,
            context_channels=32, resolution=4,
        )
        params = sum(p.size for _, p in mlx.utils.tree_flatten(model.parameters()))
        # Should be > 0 and reasonable for 2 blocks
        assert params > 10000
        assert params < 1000000

    def test_full_size_parameter_count(self):
        """Full model should have ~1.29B parameters (excluding position embeddings)."""
        import mlx.utils
        from trellmlx.models.sparse_structure_flow import SparseStructureFlowModel
        model = SparseStructureFlowModel()  # default = full size
        params = sum(p.size for _, p in mlx.utils.tree_flatten(model.parameters()))
        # RMSNorm gamma is now [H, D] instead of [D], adding (12*128 - 128) * 2 * 2 * 30
        # = 118,080 extra params. Updated target accordingly.
        assert params > 1_290_000_000 and params < 1_300_000_000, f"Expected ~1.29B params, got {params:,}"

    def test_deterministic_with_seed(self):
        """Same seed should produce same output."""
        from trellmlx.models.sparse_structure_flow import SparseStructureFlowModel
        model = SparseStructureFlowModel(
            in_channels=8, out_channels=8, model_channels=64,
            num_heads=4, num_blocks=2, mlp_hidden=128,
            context_channels=32, resolution=4,
        )
        mx.random.seed(42)
        x = mx.random.normal((1, 8, 4, 4, 4))
        t = mx.array([500.0])
        cond = mx.random.normal((1, 5, 32))
        out1 = model(x, t, cond)
        mx.eval(out1)

        mx.random.seed(42)
        x = mx.random.normal((1, 8, 4, 4, 4))
        t = mx.array([500.0])
        cond = mx.random.normal((1, 5, 32))
        out2 = model(x, t, cond)
        mx.eval(out2)

        diff = mx.max(mx.abs(out1 - out2)).item()
        assert diff < 1e-5, f"Non-deterministic: max diff = {diff}"
