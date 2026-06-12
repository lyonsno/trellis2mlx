"""VS3D: Velocity-Space 3D Asset Editing for frozen TRELLIS.2 ODE.

Training-free, mask-free, inversion-free local 3D editing.
Three modules: RASI + PMG + TAR.

Reference: arXiv:2605.07385 (VS3D, May 2026)

Architecture: FlowEdit dual-branch coupling.
  - z_edit starts at x_src (not noise)
  - At each step, S noise samples eps_s form COUPLED (z_t_src, z_t_tgt) pairs:
      z_t_src = (1-t)*x_src + t*eps_s
      z_t_tgt = z_edit + (z_t_src - x_src)
  - PMG operates on velocity DIFFERENCES v_delta = v_tgt_cfg - v_src_cfg
  - Step: z_edit += dt * u  (NOTE: += not -=)
  - RASI optimizes phi with c_src on BOTH branches via coupled velocity difference
"""

import numpy as np
import mlx.core as mx


def _cfg(v_pos: mx.array, v_neg: mx.array, omega: float) -> mx.array:
    """Additive CFG: (1+omega)*v_pos - omega*v_neg."""
    return (1.0 + omega) * v_pos - omega * v_neg


# ---------------------------------------------------------------------------
# RASI — Reconstruction-Anchored Source Injection
# ---------------------------------------------------------------------------

def rasi_optimize(
    model,
    z_t_src: mx.array,
    z_t_tgt: mx.array,
    t_k: float,
    dt: float,
    cond_src: mx.array,
    neg_cond: mx.array,
    x_src: mx.array,
    cfg_w_src: float = 1.5,
    cfg_w_tgt: float = 9.0,
    K: int = 3,
    lr: float = 1e-5,
    early_stop: float = 1e-5,
    **model_kwargs,
) -> mx.array:
    """Optimize unconditional embedding φ to anchor source reconstruction.

    Per VS3D Section 3.1: RASI objective is to find φ such that one Euler step
    of the COUPLED dual-branch system (using c_src on BOTH branches) reconstructs x_src.

    Objective:
        minimize ||z_edit + dt*(v_tgt_cfg(z_t_tgt, c_src, φ; ω_tgt)
                               - v_src_cfg(z_t_src, c_src, φ; ω_src)) - x_src||²

    Both branches see c_src (not c_tgt) — this is what drives source reconstruction.

    Args:
        model: Frozen flow model callable (x, t_tensor, cond, **kw) → velocity.
        z_t_src: Coupled source latent (1-t)*x_src + t*eps at this step.
        z_t_tgt: Coupled target latent z_edit + (z_t_src - x_src).
        t_k: Current timestep (float in [0, 1]).
        dt: Step size (positive, same sign as z_edit += dt*u).
        cond_src: Source conditioning, shape [1, L, C_ctx].
        neg_cond: Initial unconditional embedding φ, shape [1, L, C_ctx].
        x_src: Source latent to reconstruct, same shape as z_t_src.
        cfg_w_src: CFG omega for source branch.
        cfg_w_tgt: CFG omega for target branch.
        K: Number of inner optimization steps.
        lr: Learning rate for φ update.
        early_stop: Stop early if loss drops below this value.

    Returns:
        Optimized φ with same shape as neg_cond.
    """
    # z_edit is z_t_tgt when the coupled offset is zero at this noise sample;
    # we need z_edit to compute the step prediction. Recover it:
    #   z_t_tgt = z_edit + (z_t_src - x_src)  =>  z_edit = z_t_tgt - (z_t_src - x_src)
    z_edit = z_t_tgt - (z_t_src - x_src)

    t_tensor = mx.array([1000.0 * t_k], dtype=mx.float32)
    phi = mx.array(neg_cond)  # copy

    for step_idx in range(K):
        # Forward pass: c_src on BOTH branches for both model calls
        v_tgt_pos = model(z_t_tgt, t_tensor, cond_src, **model_kwargs)
        v_tgt_neg = model(z_t_tgt, t_tensor, phi, **model_kwargs)
        v_src_pos = model(z_t_src, t_tensor, cond_src, **model_kwargs)
        v_src_neg = model(z_t_src, t_tensor, phi, **model_kwargs)
        mx.eval(v_tgt_pos, v_tgt_neg, v_src_pos, v_src_neg)

        v_tgt_cfg = _cfg(v_tgt_pos, v_tgt_neg, cfg_w_tgt)
        v_src_cfg = _cfg(v_src_pos, v_src_neg, cfg_w_src)
        v_delta = v_tgt_cfg - v_src_cfg

        # One Euler step prediction: z_hat = z_edit + dt * v_delta
        z_hat = z_edit + dt * v_delta

        diff = z_hat - x_src
        loss = mx.mean(diff * diff)
        mx.eval(loss)

        loss_val = float(loss.item())
        if loss_val < early_stop:
            break

        # Central finite-difference gradient w.r.t. phi
        eps = 1e-2
        # Deterministic probe direction: alternating +1/-1 rolled by step
        total = int(np.prod(phi.shape))
        direction_np = np.ones(total, dtype=np.float32)
        direction_np[3::4] = -1.0
        if total:
            direction_np = np.roll(direction_np, step_idx % total)
        perturb_dir = mx.array(direction_np.reshape(phi.shape), dtype=phi.dtype)

        phi_plus = phi + eps * perturb_dir
        phi_minus = phi - eps * perturb_dir

        # Perturb both branches
        v_tgt_neg_p = model(z_t_tgt, t_tensor, phi_plus, **model_kwargs)
        v_src_neg_p = model(z_t_src, t_tensor, phi_plus, **model_kwargs)
        v_tgt_neg_m = model(z_t_tgt, t_tensor, phi_minus, **model_kwargs)
        v_src_neg_m = model(z_t_src, t_tensor, phi_minus, **model_kwargs)
        mx.eval(v_tgt_neg_p, v_src_neg_p, v_tgt_neg_m, v_src_neg_m)

        v_delta_p = _cfg(v_tgt_pos, v_tgt_neg_p, cfg_w_tgt) - _cfg(v_src_pos, v_src_neg_p, cfg_w_src)
        v_delta_m = _cfg(v_tgt_pos, v_tgt_neg_m, cfg_w_tgt) - _cfg(v_src_pos, v_src_neg_m, cfg_w_src)

        loss_p = float(mx.mean((z_edit + dt * v_delta_p - x_src) ** 2).item())
        loss_m = float(mx.mean((z_edit + dt * v_delta_m - x_src) ** 2).item())
        fd_grad = (loss_p - loss_m) / (2.0 * eps)
        phi = phi - lr * fd_grad * perturb_dir
        mx.eval(phi)

    return phi


# ---------------------------------------------------------------------------
# PMG — Partial-Mean Guidance
# ---------------------------------------------------------------------------

def pmg_velocity(
    v_delta_samples: list,
    w: float = 1.2,
    L: int = 2,
) -> mx.array:
    """Compute Partial-Mean Guidance velocity from a list of v_delta samples.

    PMG operates on velocity DIFFERENCES (v_delta = v_tgt_cfg - v_src_cfg),
    not on raw CFG velocities. The S samples come from S different noise draws
    eps_s forming S different (z_t_src, z_t_tgt) pairs — this is what provides
    the variance that makes PMG meaningful.

    Formula:
        u = (1 + w) * mu_S - w * mu_L

    where mu_S is the mean of all S v_delta samples and mu_L is the mean of
    the L samples with smallest per-token norm (most conservative, highest agreement).

    Args:
        v_delta_samples: List of S v_delta arrays, each shape [N, C] or [B, C, ...].
        w: PMG amplification weight.
        L: Number of partial-mean samples (L < S).

    Returns:
        PMG guidance velocity, same shape as v_delta_samples[0].
    """
    S = len(v_delta_samples)
    assert L < S, f"L={L} must be < S={S}"

    stacked = mx.stack(v_delta_samples, axis=0)  # [S, *shape]
    mu_S = mx.mean(stacked, axis=0)

    # Per-token norms for partial mean selection
    v_shape = v_delta_samples[0].shape
    flat = [mx.reshape(v, (v.shape[0], -1)) if v.ndim > 2 else v for v in v_delta_samples]
    per_token_norms = mx.stack(
        [mx.sqrt(mx.sum(f * f, axis=-1)) for f in flat], axis=0
    )  # [S, N_tokens]
    mx.eval(per_token_norms)

    per_token_norms_np = np.array(per_token_norms)  # [S, N_tokens]
    N_tokens = per_token_norms_np.shape[1]
    flat_np = np.stack([np.array(f) for f in flat], axis=0)  # [S, N_tokens, C]
    partial_mean_flat = np.zeros_like(flat_np[0])  # [N_tokens, C]
    for tok in range(N_tokens):
        best_L = np.argsort(per_token_norms_np[:, tok])[:L]
        partial_mean_flat[tok] = flat_np[best_L, tok, :].mean(axis=0)
    mu_L = mx.reshape(mx.array(partial_mean_flat), v_shape)

    v_pmg = (1.0 + w) * mu_S - w * mu_L
    mx.eval(v_pmg)
    return v_pmg


# ---------------------------------------------------------------------------
# TAR — Twin-Agreement Residual Injection
# ---------------------------------------------------------------------------

def tar_inject(
    z_tgt: mx.array,
    z_src_enc: mx.array,
    z_src_twin: mx.array,
    tgt_coords: np.ndarray,
    src_coords: np.ndarray,
    lam: float = 0.5,
    tau: float = 10.0,
    theta: float = 0.7,
    alpha: float = 0.05,
    beta: float = 0.95,
) -> mx.array:
    """Twin-Agreement Residual Injection.

    For each token in the intersection of src and tgt coordinate sets:
    - Compute twin agreement score between z_src_enc and z_src_twin
    - Only inject if agreement score > theta (cosine similarity threshold)
    - Inject residual: z_tgt[i] ← alpha*z_src_enc[j] + beta*z_tgt[i]

    Tokens not in the intersection are unchanged.

    Args:
        z_tgt: Target latents, shape [N_tgt, C].
        z_src_enc: Source encoded latents, shape [N_src, C].
        z_src_twin: Source twin-forward latents (same shape as z_tgt [N_tgt, C]).
        tgt_coords: Integer token indices for target (array of length N_tgt).
        src_coords: Integer token indices for source (array of length N_src).
        lam: Blend lambda (controls injection strength, unused directly — alpha/beta used).
        tau: Temperature for agreement scoring (higher = sharper threshold).
        theta: Cosine similarity threshold in [0, 1] — tokens with agreement < theta
               are not injected. theta=1.0 means nothing passes.
        alpha: Weight for source residual in injection.
        beta: Weight for target in injection.

    Returns:
        z_out: Modified target latents, same shape as z_tgt.
    """
    z_out_np = np.array(z_tgt)
    z_src_enc_np = np.array(z_src_enc)
    z_src_twin_np = np.array(z_src_twin)

    tgt_set = {int(c): i for i, c in enumerate(tgt_coords)}
    src_set = {int(c): i for i, c in enumerate(src_coords)}
    common_coords = set(tgt_set.keys()) & set(src_set.keys())

    for coord in common_coords:
        tgt_idx = tgt_set[coord]
        src_idx = src_set[coord]

        enc_vec = z_src_enc_np[src_idx]
        twin_vec = z_src_twin_np[tgt_idx]

        enc_norm = np.linalg.norm(enc_vec)
        twin_norm = np.linalg.norm(twin_vec)

        if enc_norm < 1e-10 or twin_norm < 1e-10:
            agreement = 0.0
        else:
            cosine = np.dot(enc_vec, twin_vec) / (enc_norm * twin_norm)
            cosine_01 = (cosine + 1.0) * 0.5
            agreement = float(np.exp(tau * (cosine_01 - 1.0)))

        if agreement >= theta:
            z_out_np[tgt_idx] = alpha * enc_vec + beta * z_out_np[tgt_idx]

    return mx.array(z_out_np)


# ---------------------------------------------------------------------------
# vs3d_flow_sample — Full VS3D sampler (FlowEdit dual-branch coupling)
# ---------------------------------------------------------------------------

def vs3d_flow_sample(
    model,
    noise: mx.array,
    cond_src: mx.array,
    cond_tgt: mx.array,
    neg_cond: mx.array,
    x_src: mx.array,
    stage: str = "dense",
    steps: int = 25,
    cfg_w_src: float = 1.5,
    cfg_w_tgt: float = 9.0,
    guidance_interval: tuple = (0.6, 1.0),
    guidance_rescale: float = 0.7,
    rescale_t: float = 5.0,
    sigma_min: float = 1e-5,
    rasi_K: int = 3,
    rasi_lr: float = 1e-5,
    pmg_S: int = 5,
    pmg_L: int = 2,
    pmg_w: float = 1.2,
    verbose: bool = False,
    concat_cond: mx.array = None,
    **model_kwargs,
) -> mx.array:
    """VS3D-augmented flow sampler using FlowEdit dual-branch coupling.

    z_edit starts at x_src (identity initialization).

    At each guided step:
      For s in range(pmg_S):
        eps_s ~ N(0, I)   [different per sample = variance for PMG]
        z_t_src = (1-t)*x_src + t*eps_s
        z_t_tgt = z_edit + (z_t_src - x_src)   [coupled offset]
        v_tgt_cfg = CFG(z_t_tgt, c_tgt, phi; omega_tgt)
        v_src_cfg = CFG(z_t_src, c_src, phi; omega_src)
        v_delta_s = v_tgt_cfg - v_src_cfg
      u = PMG({v_delta_s}, w, L)
      z_edit = z_edit + dt * u   [NOTE: +=, not -=]

    RASI is applied before PMG using a representative noise sample.

    For Stage != "dense": falls back to base flow sampler (TAR applied externally).

    Args:
        model: Frozen flow model.
        noise: Ignored for dense stage (x_src is the start). Used for sparse fallback.
        cond_src: Source conditioning [1, L, C_ctx].
        cond_tgt: Target conditioning [1, L, C_ctx].
        neg_cond: Negative conditioning (initial φ) [1, L, C_ctx].
        x_src: Source latent — starting point and reconstruction anchor.
        stage: "dense" for Stage 1, "sparse" for Stages 2a/2c/4.
        steps: Number of Euler integration steps (paper: T=25).
        cfg_w_src: CFG omega for source branch (paper: omega_src=1.5).
        cfg_w_tgt: CFG omega for target branch (paper: omega_tgt=9.0).
        guidance_interval: (low, high) fraction of timesteps to apply VS3D guidance.
        guidance_rescale: CFG rescale factor (for sparse fallback).
        rescale_t: Timestep rescaling factor.
        sigma_min: Minimum noise floor.
        rasi_K: RASI inner optimization steps (paper: K=3).
        rasi_lr: RASI learning rate.
        pmg_S: PMG Monte Carlo noise samples (paper: S=5).
        pmg_L: PMG partial-mean count (paper: L=2).
        pmg_w: PMG amplification weight (paper: w=1.2).
        verbose: Print step progress.
        concat_cond: Optional concatenated conditioning (passed to model).

    Returns:
        Edited latent z_edit, same shape as x_src.
    """
    if stage != "dense":
        from trellmlx.samplers import flow_euler_sample
        return flow_euler_sample(
            model, noise, cond_tgt, neg_cond,
            steps=steps,
            guidance_strength=cfg_w_tgt,
            guidance_rescale=guidance_rescale,
            guidance_interval=guidance_interval,
            rescale_t=rescale_t,
            sigma_min=sigma_min,
            verbose=verbose,
            concat_cond=concat_cond,
            **model_kwargs,
        )

    # Dense Stage 1: FlowEdit dual-branch coupling
    z_edit = mx.array(x_src)  # start at source, NOT noise

    # Build timestep schedule (same convention as flow_euler_sample)
    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    t_pairs = [(float(t_seq[i]), float(t_seq[i + 1])) for i in range(steps)]

    phi = mx.array(neg_cond)  # mutable phi, updated by RASI each step

    kw = dict(model_kwargs)
    if concat_cond is not None:
        kw['concat_cond'] = concat_cond

    for step_idx, (t, t_prev) in enumerate(t_pairs):
        if verbose:
            print(f"  VS3D Step {step_idx + 1}/{steps} (t={t:.4f}→{t_prev:.4f})", end="", flush=True)

        t_tensor = mx.array([1000.0 * t], dtype=mx.float32)
        dt = t - t_prev  # positive value (t decreasing)

        apply_guidance = (
            cfg_w_tgt != 1.0
            and guidance_interval[0] <= t <= guidance_interval[1]
        )

        if apply_guidance:
            # Draw S independent noise samples for this step
            # Shape matches x_src
            x_shape = x_src.shape

            eps_list = [
                mx.array(np.random.randn(*x_shape).astype(np.float32))
                for _ in range(pmg_S)
            ]

            # RASI: optimize phi using the first noise sample as representative
            eps_rasi = eps_list[0]
            z_t_src_rasi = (1.0 - t) * x_src + t * eps_rasi
            z_t_tgt_rasi = z_edit + (z_t_src_rasi - x_src)

            if rasi_K > 0:
                phi = rasi_optimize(
                    model=model,
                    z_t_src=z_t_src_rasi,
                    z_t_tgt=z_t_tgt_rasi,
                    t_k=t,
                    dt=dt,
                    cond_src=cond_src,
                    neg_cond=phi,
                    x_src=x_src,
                    cfg_w_src=cfg_w_src,
                    cfg_w_tgt=cfg_w_tgt,
                    K=rasi_K,
                    lr=rasi_lr,
                    **kw,
                )

            # PMG: S noise samples → S v_delta estimates → partial-mean guidance
            v_delta_samples = []
            for eps_s in eps_list:
                z_t_src_s = (1.0 - t) * x_src + t * eps_s
                z_t_tgt_s = z_edit + (z_t_src_s - x_src)

                v_tgt_pos = model(z_t_tgt_s, t_tensor, cond_tgt, **kw)
                v_tgt_neg = model(z_t_tgt_s, t_tensor, phi, **kw)
                v_src_pos = model(z_t_src_s, t_tensor, cond_src, **kw)
                v_src_neg = model(z_t_src_s, t_tensor, phi, **kw)
                mx.eval(v_tgt_pos, v_tgt_neg, v_src_pos, v_src_neg)

                v_tgt_cfg = _cfg(v_tgt_pos, v_tgt_neg, cfg_w_tgt)
                v_src_cfg = _cfg(v_src_pos, v_src_neg, cfg_w_src)
                v_delta_samples.append(v_tgt_cfg - v_src_cfg)

            u = pmg_velocity(v_delta_samples, w=pmg_w, L=pmg_L)

            # FlowEdit Euler step: z_edit += dt * u
            z_edit = z_edit + dt * u
            mx.eval(z_edit)

        else:
            # Outside guidance interval: plain CFG on z_edit directly
            v_pos = model(z_edit, t_tensor, cond_tgt, **kw)
            v_neg = model(z_edit, t_tensor, phi, **kw)
            mx.eval(v_pos, v_neg)
            pred = _cfg(v_pos, v_neg, cfg_w_tgt)
            z_edit = z_edit - dt * pred  # standard flow: -= for non-guided steps
            mx.eval(z_edit)

        if verbose:
            print(" done", flush=True)

    return z_edit
