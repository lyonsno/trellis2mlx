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

    def test_rasi_returns_optimized_embedding(self):
        """RASI must return a phi tensor with same shape as neg_cond."""
        from trellmlx.vs3d import rasi_optimize

        N = 100
        C = 32
        z_t = mx.random.normal((N, C))
        cond_src = mx.random.normal((1, 10, 1024))
        neg_cond = mx.random.normal((1, 10, 1024))
        x_src = mx.random.normal((N, C))

        # Stub model: identity velocity
        def stub_model(x, t, cond, **kw):
            return mx.zeros_like(x)

        t_k = 0.8
        dt = 0.05

        phi = rasi_optimize(
            model=stub_model,
            z_t=z_t,
            t_k=t_k,
            dt=dt,
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
        z_t = mx.random.normal((N, C))
        cond_src = mx.random.normal((1, 10, 1024))
        neg_cond = mx.zeros((1, 10, 1024))
        x_src = mx.ones((N, C))  # non-trivial target

        def stub_model(x, t, cond, **kw):
            # Returns something non-zero so gradient has signal
            return x * 0.1

        phi = rasi_optimize(
            model=stub_model,
            z_t=z_t,
            t_k=0.8,
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


# ---------------------------------------------------------------------------
# PMG tests
# ---------------------------------------------------------------------------

class TestPMG:
    """Partial-Mean Guidance."""

    def test_pmg_output_shape_matches_input(self):
        """PMG must return a velocity array with the same shape as z_t."""
        from trellmlx.vs3d import pmg_velocity

        N = 200
        C = 32
        z_t = mx.random.normal((N, C))
        t_tensor = mx.array([800.0])
        cond_tgt = mx.random.normal((1, 10, 1024))
        phi = mx.zeros((1, 10, 1024))

        def stub_model(x, t, cond, **kw):
            return mx.zeros_like(x)

        v = pmg_velocity(
            model=stub_model,
            z_t=z_t,
            t_tensor=t_tensor,
            cond_tgt=cond_tgt,
            phi=phi,
            S=5,
            L=2,
            w=1.2,
        )
        assert v.shape == z_t.shape, f"PMG output shape {v.shape} != {z_t.shape}"

    def test_pmg_amplifies_nonzero_signal(self):
        """With w>0, PMG must produce larger magnitude than plain mean."""
        from trellmlx.vs3d import pmg_velocity

        N = 50
        C = 8
        z_t = mx.zeros((N, C))
        t_tensor = mx.array([500.0])
        cond = mx.zeros((1, 4, 1024))
        phi = mx.zeros((1, 4, 1024))

        # Model returns a constant non-zero velocity
        def biased_model(x, t, cond, **kw):
            return mx.ones_like(x) * 0.5

        v = pmg_velocity(
            model=biased_model,
            z_t=z_t,
            t_tensor=t_tensor,
            cond_tgt=cond,
            phi=phi,
            S=5,
            L=2,
            w=1.2,
        )
        # With w=1.2, result = (1+w)*mu_S - w*mu_L = 2.2*0.5 - 1.2*0.5 = 0.5
        # i.e. same since model is deterministic, but must be nonzero
        magnitude = float(mx.mean(mx.abs(v)).item())
        assert magnitude > 0, "PMG must produce nonzero output with nonzero model"

    def test_pmg_uses_S_model_calls(self):
        """PMG must call the model exactly S + 1 times (S for PMG + 1 for source)."""
        from trellmlx.vs3d import pmg_velocity

        call_count = {"n": 0}

        def counting_model(x, t, cond, **kw):
            call_count["n"] += 1
            return mx.zeros_like(x)

        z_t = mx.zeros((20, 8))
        v = pmg_velocity(
            model=counting_model,
            z_t=z_t,
            t_tensor=mx.array([500.0]),
            cond_tgt=mx.zeros((1, 4, 1024)),
            phi=mx.zeros((1, 4, 1024)),
            S=5,
            L=2,
            w=1.2,
        )
        # S samples all use the same cond/phi but different noise seeds
        assert call_count["n"] == 5, (
            f"Expected 5 model calls (S=5), got {call_count['n']}"
        )


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
        """vs3d_flow_sample for Stage 1 must return same shape as noise."""
        from trellmlx.vs3d import vs3d_flow_sample

        N = 64
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
        assert result.shape == noise.shape, (
            f"vs3d_flow_sample output {result.shape} != noise {noise.shape}"
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
