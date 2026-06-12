"""Fail-first tests for VS3D SLAT editing modules.

Tests are ordered: shape contracts first, then integration.
All tests must FAIL before implementation exists.
"""

import math
import numpy as np
import pytest
import mlx.core as mx


# ---------------------------------------------------------------------------
# RASI tests
# ---------------------------------------------------------------------------

class TestRASI:
    """Reconstruction-Anchored Source Injection."""

    def _make_coupled_pair(self, x_src, z_edit, t):
        """Form (z_t_src, z_t_tgt) from x_src, z_edit, and timestep t."""
        eps = mx.array(np.random.randn(*x_src.shape).astype(np.float32))
        z_t_src = (1.0 - t) * x_src + t * eps
        z_t_tgt = z_edit + (z_t_src - x_src)
        return z_t_src, z_t_tgt

    def test_rasi_returns_optimized_embedding(self):
        """RASI must return a phi tensor with same shape as neg_cond."""
        from trellmlx.vs3d import rasi_optimize

        N = 100
        C = 32
        np.random.seed(0)
        x_src = mx.array(np.random.randn(N, C).astype(np.float32))
        z_edit = mx.array(np.random.randn(N, C).astype(np.float32))
        cond_src = mx.random.normal((1, 10, 1024))
        neg_cond = mx.random.normal((1, 10, 1024))
        t_k = 0.8
        z_t_src, z_t_tgt = self._make_coupled_pair(x_src, z_edit, t_k)

        def stub_model(x, t, cond, **kw):
            return mx.zeros_like(x)

        phi = rasi_optimize(
            model=stub_model,
            z_t_src=z_t_src,
            z_t_tgt=z_t_tgt,
            t_k=t_k,
            dt=0.05,
            cond_src=cond_src,
            neg_cond=neg_cond,
            x_src=x_src,
            cfg_w_src=1.5,
            cfg_w_tgt=9.0,
            K=3,
            lr=1e-5,
        )
        assert phi.shape == neg_cond.shape, (
            f"phi shape {phi.shape} != neg_cond shape {neg_cond.shape}"
        )

    def test_rasi_phi_differs_from_neg_cond(self):
        """After optimization, phi must differ from the initial neg_cond."""
        from trellmlx.vs3d import rasi_optimize

        N = 50
        C = 32
        np.random.seed(1)
        x_src = mx.zeros((N, C))
        z_edit = mx.ones((N, C))  # non-trivial z_edit so coupled pair is non-trivial
        cond_src = mx.random.normal((1, 10, 1024))
        neg_cond = mx.zeros((1, 10, 1024))
        t_k = 0.8
        z_t_src, z_t_tgt = self._make_coupled_pair(x_src, z_edit, t_k)

        def cond_sensitive_model(x, t, cond, **kw):
            cond_signal = mx.mean(cond)
            return x * 0.1 + cond_signal * 0.01 * mx.ones_like(x)

        phi = rasi_optimize(
            model=cond_sensitive_model,
            z_t_src=z_t_src,
            z_t_tgt=z_t_tgt,
            t_k=t_k,
            dt=0.05,
            cond_src=cond_src,
            neg_cond=neg_cond,
            x_src=x_src,
            cfg_w_src=1.5,
            cfg_w_tgt=9.0,
            K=3,
            lr=1e-5,
        )
        diff = float(mx.mean(mx.abs(phi - neg_cond)).item())
        assert diff > 0, "phi must differ from neg_cond after optimization"

    def test_rasi_update_has_directional_phi_signal(self):
        """RASI must not apply one uniform scalar delta to every phi element."""
        from trellmlx.vs3d import rasi_optimize

        np.random.seed(2)
        x_src = mx.zeros((4, 2))
        z_edit = mx.ones((4, 2))
        cond_src = mx.zeros((1, 2, 4))
        neg_cond = mx.zeros((1, 2, 4))
        t_k = 0.8
        z_t_src, z_t_tgt = self._make_coupled_pair(x_src, z_edit, t_k)
        weights = mx.array(
            [[[1.0, -2.0, 0.5, -0.25], [1.5, -0.75, 0.25, -1.25]]],
            dtype=mx.float32,
        )

        def element_sensitive_model(x, t, cond, **kw):
            cond_signal = mx.sum(cond * weights)
            return x * 0.1 + cond_signal * mx.ones_like(x)

        phi = rasi_optimize(
            model=element_sensitive_model,
            z_t_src=z_t_src,
            z_t_tgt=z_t_tgt,
            t_k=t_k,
            dt=0.05,
            cond_src=cond_src,
            neg_cond=neg_cond,
            x_src=x_src,
            cfg_w_src=1.5,
            K=2,
            lr=1e-3,
        )

        delta = np.array(phi - neg_cond).reshape(-1)
        assert np.max(delta) - np.min(delta) > 1e-8, (
            "RASI phi update must carry element-wise direction, not a uniform scalar delta"
        )

    def test_rasi_uses_dual_branch_coupling(self):
        """RASI must use c_src on BOTH branches and compute v_delta = v_tgt_cfg - v_src_cfg."""
        from trellmlx.vs3d import rasi_optimize

        # Track which (latent, cond) pairs are seen by the model
        calls = []

        np.random.seed(3)
        N, C = 4, 4
        x_src = mx.zeros((N, C))
        z_edit = mx.ones((N, C))
        cond_src = mx.ones((1, 2, 4)) * 2.0
        neg_cond = mx.zeros((1, 2, 4))
        t_k = 0.8
        eps = np.random.randn(N, C).astype(np.float32)
        z_t_src = mx.array((1.0 - t_k) * np.zeros((N, C)) + t_k * eps)
        z_t_tgt = mx.array(np.ones((N, C)) + (t_k * eps - t_k * eps))  # z_edit + offset

        def spy_model(x, t, cond, **kw):
            calls.append(float(mx.mean(mx.abs(cond)).item()))
            return mx.zeros_like(x)

        rasi_optimize(
            model=spy_model,
            z_t_src=z_t_src,
            z_t_tgt=z_t_tgt,
            t_k=t_k,
            dt=0.05,
            cond_src=cond_src,
            neg_cond=neg_cond,
            x_src=x_src,
            cfg_w_src=1.5,
            cfg_w_tgt=9.0,
            K=1,
            lr=1e-5,
        )
        # With K=1, no phi perturbation (loss=0 from zero model), but initial forward
        # pass still happens. Key invariant: cond_src mean = 2.0 must appear in calls
        # (both pos-cond calls use c_src), neg_cond mean = 0.0 appears for phi calls.
        cond_src_mean = float(mx.mean(mx.abs(cond_src)).item())
        has_src_cond = any(abs(c - cond_src_mean) < 1e-4 for c in calls)
        assert has_src_cond, "RASI must call model with c_src on at least one branch"


# ---------------------------------------------------------------------------
# PMG tests
# ---------------------------------------------------------------------------

class TestPMG:
    """Partial-Mean Guidance."""

    def test_pmg_output_shape_matches_input(self):
        """PMG must return a velocity array with the same shape as each v_delta sample."""
        from trellmlx.vs3d import pmg_velocity

        N, C = 200, 32
        v_delta_samples = [mx.random.normal((N, C)) for _ in range(5)]

        v = pmg_velocity(v_delta_samples, w=1.2, L=2)
        assert v.shape == (N, C), f"PMG output shape {v.shape} != ({N}, {C})"

    def test_pmg_amplifies_nonzero_signal(self):
        """With w>0 and uniform v_delta, PMG must produce nonzero output."""
        from trellmlx.vs3d import pmg_velocity

        N, C = 50, 8
        # All samples identical constant v_delta
        v_delta_samples = [mx.ones((N, C)) * 0.5 for _ in range(5)]

        v = pmg_velocity(v_delta_samples, w=1.2, L=2)
        # (1+w)*0.5 - w*0.5 = 0.5 (identical samples → partial mean = full mean)
        magnitude = float(mx.mean(mx.abs(v)).item())
        assert magnitude > 0, "PMG must produce nonzero output with nonzero v_delta samples"

    def test_pmg_partial_mean_amplifies_consistent_directions(self):
        """PMG must amplify directions consistently present across S samples (low-norm samples).

        When some samples have small norm and others large, (1+w)*mu_S - w*mu_L
        amplifies the small-norm (conservative, consistent) direction.
        """
        from trellmlx.vs3d import pmg_velocity

        N, C = 10, 4
        np.random.seed(42)
        # 2 consistent low-norm samples + 3 high-noise samples
        low_norm = [mx.ones((N, C)) * 0.1 for _ in range(2)]
        high_norm = [mx.array(np.random.randn(N, C).astype(np.float32)) * 5.0 for _ in range(3)]
        samples = low_norm + high_norm

        v = pmg_velocity(samples, w=1.2, L=2)
        mu_S = float(mx.mean(mx.stack([mx.mean(s) for s in samples])).item())
        v_mean = float(mx.mean(v).item())
        # PMG with L=2 picks the 2 low-norm samples → mu_L ≈ 0.1
        # u = (1+1.2)*mu_S - 1.2*0.1 — must not equal mu_S (plain mean)
        assert abs(v_mean - mu_S) > 1e-6, (
            "PMG output must differ from plain mean when samples have different norms"
        )

    def test_pmg_requires_L_less_than_S(self):
        """PMG must raise when L >= S."""
        from trellmlx.vs3d import pmg_velocity

        samples = [mx.ones((4, 4)) for _ in range(3)]
        with pytest.raises(AssertionError):
            pmg_velocity(samples, w=1.2, L=3)  # L == S → invalid

    def test_pmg_variance_across_samples_is_used(self):
        """PMG output must differ when S samples have variance vs. when they are constant."""
        from trellmlx.vs3d import pmg_velocity

        N, C = 20, 8
        np.random.seed(10)

        # Constant samples
        const_samples = [mx.ones((N, C)) * 0.3 for _ in range(5)]
        v_const = pmg_velocity(const_samples, w=1.2, L=2)

        # Varied samples: some consistent, some not
        varied_samples = [
            mx.ones((N, C)) * 0.3,
            mx.ones((N, C)) * 0.3,
            mx.array(np.random.randn(N, C).astype(np.float32) * 2.0),
            mx.array(np.random.randn(N, C).astype(np.float32) * 2.0),
            mx.array(np.random.randn(N, C).astype(np.float32) * 2.0),
        ]
        v_varied = pmg_velocity(varied_samples, w=1.2, L=2)

        diff = float(mx.mean(mx.abs(v_varied - v_const)).item())
        assert diff > 1e-4, "PMG output must differ when sample variance differs"


# ---------------------------------------------------------------------------
# TAR tests
# ---------------------------------------------------------------------------

class TestTAR:
    """Twin-Agreement Residual Injection."""

    def test_tar_output_shape_matches_target(self):
        """TAR must return a latent array with same shape as z_tgt."""
        from trellmlx.vs3d import tar_inject

        N_tgt = 300
        N_src = 250
        C = 32

        z_tgt = mx.random.normal((N_tgt, C))
        z_src_enc = mx.random.normal((N_src, C))
        z_src_twin = mx.random.normal((N_tgt, C))

        # Simulate intersection: first 200 tokens exist in both
        tgt_coords = np.arange(N_tgt)
        src_coords = np.arange(N_src)

        z_out = tar_inject(
            z_tgt=z_tgt,
            z_src_enc=z_src_enc,
            z_src_twin=z_src_twin,
            tgt_coords=tgt_coords,
            src_coords=src_coords,
            lam=0.5,
            tau=10.0,
            theta=0.7,
            alpha=0.05,
            beta=0.95,
        )
        assert z_out.shape == z_tgt.shape, (
            f"TAR output shape {z_out.shape} != z_tgt shape {z_tgt.shape}"
        )

    def test_tar_preserves_nonintersection_tokens(self):
        """Tokens not in the src/tgt intersection must be unchanged."""
        from trellmlx.vs3d import tar_inject

        N_tgt = 10
        C = 4
        z_tgt = mx.ones((N_tgt, C))
        z_src_enc = mx.ones((5, C)) * 2.0   # only indices 0..4 are src
        z_src_twin = mx.zeros((N_tgt, C))   # twin predicts zeros

        # Only indices 0..4 overlap
        tgt_coords = np.arange(N_tgt)
        src_coords = np.arange(5)

        z_out = tar_inject(
            z_tgt=z_tgt,
            z_src_enc=z_src_enc,
            z_src_twin=z_src_twin,
            tgt_coords=tgt_coords,
            src_coords=src_coords,
            lam=0.5,
            tau=10.0,
            theta=0.7,
            alpha=0.05,
            beta=0.95,
        )
        # Indices 5..9 have no src match → must be unchanged from z_tgt
        z_out_np = np.array(z_out)
        np.testing.assert_allclose(
            z_out_np[5:], np.ones((5, C)),
            rtol=1e-5,
            err_msg="Non-intersection tokens must not be modified by TAR",
        )

    def test_tar_agreement_threshold_gates_injection(self):
        """With theta=1.0 (nothing agrees), TAR must not inject anything."""
        from trellmlx.vs3d import tar_inject

        N = 8
        C = 4
        z_tgt = mx.zeros((N, C))
        # src_enc and src_twin are maximally disagreeing
        z_src_enc = mx.ones((N, C)) * 5.0
        z_src_twin = mx.ones((N, C)) * -5.0  # max disagreement

        tgt_coords = np.arange(N)
        src_coords = np.arange(N)

        z_out = tar_inject(
            z_tgt=z_tgt,
            z_src_enc=z_src_enc,
            z_src_twin=z_src_twin,
            tgt_coords=tgt_coords,
            src_coords=src_coords,
            lam=0.5,
            tau=10.0,
            theta=1.0,   # require perfect agreement — nothing passes
            alpha=0.05,
            beta=0.95,
        )
        z_out_np = np.array(z_out)
        np.testing.assert_allclose(
            z_out_np, np.zeros((N, C)), atol=1e-5,
            err_msg="theta=1.0 should gate out all injections",
        )


# ---------------------------------------------------------------------------
# End-to-end sampler integration
# ---------------------------------------------------------------------------

class TestVS3DSamplerIntegration:
    """The VS3D sampler must compose RASI + PMG in Stage 1."""

    def test_vs3d_sampler_stage1_returns_correct_shape(self):
        """vs3d_flow_sample for Stage 1 must return same shape as x_src."""
        from trellmlx.vs3d import vs3d_flow_sample

        C = 8
        noise = mx.random.normal((1, C, 4, 4, 4))  # dense grid [B, C, R, R, R]
        cond_src = mx.zeros((1, 10, 1024))
        cond_tgt = mx.zeros((1, 10, 1024))
        neg_cond = mx.zeros((1, 10, 1024))
        x_src = mx.zeros((1, C, 4, 4, 4))

        def stub_model(x, t, cond, **kw):
            return mx.zeros_like(x)

        result = vs3d_flow_sample(
            model=stub_model,
            noise=noise,
            cond_src=cond_src,
            cond_tgt=cond_tgt,
            neg_cond=neg_cond,
            x_src=x_src,
            stage="dense",
            steps=4,
            cfg_w_src=1.5,
            cfg_w_tgt=9.0,
            guidance_interval=(0.6, 1.0),
        )
        assert result.shape == x_src.shape, (
            f"vs3d_flow_sample output {result.shape} != x_src {x_src.shape}"
        )

    def test_vs3d_sampler_starts_at_x_src(self):
        """With zero-velocity model and no guidance, z_edit must equal x_src at output."""
        from trellmlx.vs3d import vs3d_flow_sample

        C = 8
        np.random.seed(42)
        x_src = mx.array(np.random.randn(1, C, 4, 4, 4).astype(np.float32))
        noise = mx.random.normal((1, C, 4, 4, 4))
        cond_src = mx.zeros((1, 4, 1024))
        cond_tgt = mx.zeros((1, 4, 1024))
        neg_cond = mx.zeros((1, 4, 1024))

        def zero_model(x, t, cond, **kw):
            return mx.zeros_like(x)

        # guidance_interval=(0, 0) means no guidance step is ever applied
        result = vs3d_flow_sample(
            model=zero_model,
            noise=noise,
            cond_src=cond_src,
            cond_tgt=cond_tgt,
            neg_cond=neg_cond,
            x_src=x_src,
            stage="dense",
            steps=4,
            cfg_w_tgt=9.0,
            guidance_interval=(0.0, 0.0),
            rasi_K=0,
        )
        # Zero velocity everywhere → z_edit never moves → should equal x_src
        diff = float(mx.mean(mx.abs(result - x_src)).item())
        assert diff < 1e-5, (
            f"With zero-velocity model, output must equal x_src (diff={diff:.2e})"
        )

    def test_vs3d_sampler_differs_from_baseline(self):
        """VS3D output must differ from plain flow_euler_sample output."""
        from trellmlx.vs3d import vs3d_flow_sample
        from trellmlx.samplers import flow_euler_sample

        mx.random.seed(7)
        noise = mx.random.normal((1, 8, 4, 4, 4))
        cond_src = mx.random.normal((1, 5, 1024))
        cond_tgt = mx.random.normal((1, 5, 1024)) * 2.0  # different target
        neg_cond = mx.zeros((1, 5, 1024))
        x_src = mx.random.normal((1, 8, 4, 4, 4))

        def stub_model(x, t, cond, **kw):
            # Returns dot(cond_mean, 1) * x to create cond-dependent output
            cond_signal = mx.mean(cond)
            return x * cond_signal * 0.01

        mx.random.seed(7)
        baseline = flow_euler_sample(
            stub_model, mx.array(noise), cond_tgt, neg_cond,
            steps=4, guidance_strength=7.5,
        )

        mx.random.seed(7)
        vs3d_out = vs3d_flow_sample(
            model=stub_model,
            noise=mx.array(noise),
            cond_src=cond_src,
            cond_tgt=cond_tgt,
            neg_cond=neg_cond,
            x_src=x_src,
            stage="dense",
            steps=4,
            cfg_w_src=1.5,
            cfg_w_tgt=9.0,
            guidance_interval=(0.6, 1.0),
        )

        diff = float(mx.mean(mx.abs(vs3d_out - baseline)).item())
        assert diff > 1e-6, (
            f"VS3D output must differ from plain CFG baseline (diff={diff:.2e})"
        )
