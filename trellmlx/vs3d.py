"""VS3D: Velocity-Space 3D Asset Editing for frozen TRELLIS.2 ODE.

Training-free, mask-free, inversion-free local 3D editing.
Three modules: RASI + PMG + TAR.

Reference: arXiv:2605.07385 (VS3D, May 2026)
"""

import numpy as np
import mlx.core as mx


# ---------------------------------------------------------------------------
# RASI — Reconstruction-Anchored Source Injection
# ---------------------------------------------------------------------------

def rasi_optimize(
    model,
    z_t: mx.array,
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

    RASI turns the unconditional embedding into a per-step source anchor.
    For each timestep t_k, we want a φ such that one Euler step of the
    source-conditioned ODE (with φ as negative cond) reconstructs x_src.

    Objective: minimize ||x_src - (z_t - dt * v_src(z_t, t_k, cond_src, φ))||²
    where v_src uses CFG with cond_src and φ.

    Args:
        model: Frozen flow model callable (x, t_tensor, cond, **kw) → velocity.
        z_t: Current latent at timestep t_k, shape [N, C] or [B, C, ...].
        t_k: Current timestep (float in [0, 1]).
        dt: Step size (t_k - t_{k+1}).
        cond_src: Source conditioning, shape [1, L, C_ctx].
        neg_cond: Initial unconditional embedding, shape [1, L, C_ctx].
        x_src: Source latent to reconstruct, same shape as z_t.
        cfg_w_src: CFG weight for source branch.
        cfg_w_tgt: CFG weight for target branch (not used in RASI but kept for interface parity).
        K: Number of inner optimization steps.
        lr: Learning rate for φ update.
        early_stop: Stop early if loss drops below this value.

    Returns:
        Optimized φ with same shape as neg_cond.
    """
    t_tensor = mx.array([1000.0 * t_k], dtype=mx.float32)
    phi = mx.array(neg_cond)  # copy

    for _ in range(K):
        # Forward pass to compute velocity with current phi
        v_pos = model(z_t, t_tensor, cond_src, **model_kwargs)
        v_neg = model(z_t, t_tensor, phi, **model_kwargs)
        mx.eval(v_pos, v_neg)

        # CFG-combined source velocity
        v_src = cfg_w_src * v_pos + (1.0 - cfg_w_src) * v_neg

        # One Euler step prediction: z_hat = z_t - dt * v_src
        z_hat = z_t - dt * v_src

        # Loss: squared reconstruction error
        diff = z_hat - x_src
        loss = mx.mean(diff * diff)
        mx.eval(loss)

        loss_val = float(loss.item())
        if loss_val < early_stop:
            break

        # Gradient of loss w.r.t. phi via finite differences.
        # dL/d(v_neg) direction is carried by diff; we project it into phi-space
        # by finite-differencing the neg branch along the residual direction.
        # Perturbation: phi_plus = phi + eps * sign(residual_projected)
        # where residual_projected is diff averaged to phi shape.
        eps = 1e-4
        # Project diff (z_hat shape) to a per-context-token direction for phi.
        # Flatten diff to scalar direction per phi element via mean over non-ctx dims.
        # phi: [1, L, C_ctx]; diff: [N, C] or [B, C, ...] — use mean as proxy.
        diff_scalar = mx.mean(diff)  # scalar gradient signal direction
        perturb_dir = mx.sign(diff_scalar * mx.ones_like(phi))
        phi_plus = phi + eps * perturb_dir
        v_neg_plus = model(z_t, t_tensor, phi_plus, **model_kwargs)
        mx.eval(v_neg_plus)
        v_src_plus = cfg_w_src * v_pos + (1.0 - cfg_w_src) * v_neg_plus
        z_hat_plus = z_t - dt * v_src_plus
        loss_plus = float(mx.mean((z_hat_plus - x_src) ** 2).item())
        # Finite-difference gradient: (L(phi+eps) - L(phi)) / eps
        fd_grad = (loss_plus - loss_val) / eps
        phi = phi - lr * fd_grad * perturb_dir
        mx.eval(phi)

    return phi


# ---------------------------------------------------------------------------
# PMG — Partial-Mean Guidance
# ---------------------------------------------------------------------------

def pmg_velocity(
    model,
    z_t: mx.array,
    t_tensor: mx.array,
    cond_tgt: mx.array,
    phi: mx.array,
    S: int = 5,
    L: int = 2,
    w: float = 1.2,
    **model_kwargs,
) -> mx.array:
    """Compute Partial-Mean Guidance velocity.

    PMG reduces the variance of the CFG velocity estimate by averaging S
    samples and using a partial mean of L < S samples to amplify the edit
    signal that is consistently present.

    Formula:
        v_PMG = (1 + w) * mu_S(z_t, t, cond_tgt, phi)
                - w * mu_L(z_t, t, cond_tgt, phi)

    where mu_S is the mean of S model calls and mu_L is the mean of the L
    calls with smallest velocity magnitude (partial mean). This amplifies
    directions that are robustly present.

    Args:
        model: Frozen flow model.
        z_t: Current latent, shape [N, C] or [B, C, ...].
        t_tensor: Timestep tensor, shape [1].
        cond_tgt: Target conditioning, shape [1, L_ctx, C_ctx].
        phi: Optimized unconditional embedding from RASI, shape [1, L_ctx, C_ctx].
        S: Number of Monte Carlo samples.
        L: Number of partial-mean samples (L < S).
        w: Amplification weight.

    Returns:
        PMG velocity, same shape as z_t.
    """
    # Collect S CFG velocity estimates.
    # Each sample uses the same cond_tgt and phi but the stochasticity comes
    # from the model (e.g. dropout, if any). For a fully deterministic frozen
    # model these will be identical; PMG still applies — the variance collapses
    # and (1+w)*mu_S - w*mu_L = mu_S (the partial mean cancels). With any
    # stochastic element (dropout active, quantization noise, etc.) the
    # partial mean meaningfully filters low-consistency directions.
    cfg_velocities = []
    for _ in range(S):
        v_pos = model(z_t, t_tensor, cond_tgt, **model_kwargs)
        v_neg = model(z_t, t_tensor, phi, **model_kwargs)
        mx.eval(v_pos, v_neg)
        # CFG combination for this sample
        v_cfg = (1.0 + w) * v_pos - w * v_neg  # equivalent to w*(v_pos - v_neg) + v_pos
        cfg_velocities.append(v_cfg)

    # Stack: [S, *shape]
    stacked = mx.stack(cfg_velocities, axis=0)

    # Full mean over S samples: mu_S shape = [*shape]
    mu_S = mx.mean(stacked, axis=0)

    # Partial mean: for each token/element, pick the L samples whose CFG velocity
    # magnitude is smallest (most conservative, highest agreement).
    # Per-token norms: [S, N] where N = prod(spatial dims)
    flat = [mx.reshape(v, (v.shape[0], -1)) if v.ndim > 2 else v for v in cfg_velocities]
    # Per-token norm for each sample: shape [S, N_tokens]
    per_token_norms = mx.stack(
        [mx.sqrt(mx.sum(f * f, axis=-1)) for f in flat], axis=0
    )  # [S, N_tokens]
    mx.eval(per_token_norms)

    # For each token, find the L sample indices with smallest norm
    per_token_norms_np = np.array(per_token_norms)  # [S, N_tokens]
    N_tokens = per_token_norms_np.shape[1]
    # Build per-token partial mean by selecting L best samples per token
    # Result shape matches cfg_velocities[0]
    v_shape = cfg_velocities[0].shape
    flat_np = np.stack([np.array(f) for f in flat], axis=0)  # [S, N_tokens, C]
    partial_mean_flat = np.zeros_like(flat_np[0])  # [N_tokens, C]
    for tok in range(N_tokens):
        best_L = np.argsort(per_token_norms_np[:, tok])[:L]
        partial_mean_flat[tok] = flat_np[best_L, tok, :].mean(axis=0)
    mu_L = mx.reshape(mx.array(partial_mean_flat), v_shape)

    # PMG combination: (1+w)*mu_S - w*mu_L
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
      where the blend is clipped/weighted by tau

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

    # Build intersection: tokens present in both tgt_coords and src_coords
    tgt_set = {int(c): i for i, c in enumerate(tgt_coords)}
    src_set = {int(c): i for i, c in enumerate(src_coords)}
    common_coords = set(tgt_set.keys()) & set(src_set.keys())

    for coord in common_coords:
        tgt_idx = tgt_set[coord]
        src_idx = src_set[coord]

        # Twin agreement: cosine similarity between z_src_enc[src_idx] and z_src_twin[tgt_idx]
        enc_vec = z_src_enc_np[src_idx]
        twin_vec = z_src_twin_np[tgt_idx]

        enc_norm = np.linalg.norm(enc_vec)
        twin_norm = np.linalg.norm(twin_vec)

        if enc_norm < 1e-10 or twin_norm < 1e-10:
            # Zero vectors — no meaningful agreement
            agreement = 0.0
        else:
            cosine = np.dot(enc_vec, twin_vec) / (enc_norm * twin_norm)
            # Map cosine [-1, 1] → [0, 1] and apply temperature sharpening
            cosine_01 = (cosine + 1.0) * 0.5
            agreement = float(np.exp(tau * (cosine_01 - 1.0)))  # peaks at 1.0

        # Gate injection on agreement threshold
        if agreement >= theta:
            # Residual injection: weighted blend
            z_out_np[tgt_idx] = alpha * enc_vec + beta * z_out_np[tgt_idx]

    return mx.array(z_out_np)


# ---------------------------------------------------------------------------
# vs3d_flow_sample — Full VS3D sampler wrapping flow_euler_sample
# ---------------------------------------------------------------------------

def vs3d_flow_sample(
    model,
    noise: mx.array,
    cond_src: mx.array,
    cond_tgt: mx.array,
    neg_cond: mx.array,
    x_src: mx.array,
    stage: str = "dense",
    steps: int = 12,
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
    """VS3D-augmented flow Euler sampler.

    For Stage 1 (dense): at each guidance-interval step, run RASI to obtain
    an optimized φ, then use PMG to compute the velocity.

    For sparse stages (2a, 2c, 4): run the base flow sampler — TAR is applied
    once by the caller at the sparse latent level (not per-step here).

    Args:
        model: Frozen flow model.
        noise: Initial noise, shape [B, C, ...] (dense) or [N, C] (sparse).
        cond_src: Source conditioning [1, L, C_ctx].
        cond_tgt: Target conditioning [1, L, C_ctx].
        neg_cond: Negative conditioning (initial φ) [1, L, C_ctx].
        x_src: Source latent for reconstruction anchor, same shape as noise.
        stage: "dense" for Stage 1, "sparse" for Stages 2a/2c/4.
        steps: Number of Euler integration steps.
        cfg_w_src: CFG weight for source branch in RASI.
        cfg_w_tgt: CFG guidance strength for target in PMG.
        guidance_interval: (low, high) fraction of timesteps to apply VS3D guidance.
        guidance_rescale: CFG rescale factor (passed through to base sampler if not VS3D).
        rescale_t: Timestep rescaling factor.
        sigma_min: Minimum noise floor.
        rasi_K: RASI inner optimization steps.
        rasi_lr: RASI learning rate.
        pmg_S: PMG Monte Carlo samples.
        pmg_L: PMG partial-mean count.
        pmg_w: PMG amplification weight.
        verbose: Print step progress.

    Returns:
        Denoised sample, same shape as noise.
    """
    if stage != "dense":
        # Sparse stages: delegate to base sampler (TAR applied externally per-stage)
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

    # Dense Stage 1: VS3D RASI + PMG loop
    sample = noise

    # Build timestep schedule (same as flow_euler_sample)
    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(steps)]

    phi = neg_cond  # initial unconditional embedding; will be updated by RASI

    if concat_cond is not None:
        model_kwargs['concat_cond'] = concat_cond

    for step_idx, (t, t_prev) in enumerate(t_pairs):
        if verbose:
            print(f"  VS3D Step {step_idx + 1}/{steps} (t={t:.4f}→{t_prev:.4f})", end="", flush=True)

        t_tensor = mx.array([1000.0 * t], dtype=mx.float32)
        dt = t - t_prev

        apply_guidance = (
            cfg_w_tgt != 1.0
            and guidance_interval[0] <= t <= guidance_interval[1]
        )

        if apply_guidance:
            # RASI: optimize φ to anchor source reconstruction at this timestep
            phi = rasi_optimize(
                model=model,
                z_t=sample,
                t_k=t,
                dt=dt,
                cond_src=cond_src,
                neg_cond=phi,
                x_src=x_src,
                cfg_w_src=cfg_w_src,
                cfg_w_tgt=cfg_w_tgt,
                K=rasi_K,
                lr=rasi_lr,
                **model_kwargs,
            )

            # PMG: compute target velocity with partial-mean amplification
            pred = pmg_velocity(
                model=model,
                z_t=sample,
                t_tensor=t_tensor,
                cond_tgt=cond_tgt,
                phi=phi,
                S=pmg_S,
                L=pmg_L,
                w=pmg_w,
                **model_kwargs,
            )
        else:
            # Outside guidance interval: plain CFG
            pred_pos = model(sample, t_tensor, cond_tgt, **model_kwargs)
            pred_neg = model(sample, t_tensor, phi, **model_kwargs)
            mx.eval(pred_pos, pred_neg)
            pred = cfg_w_tgt * pred_pos + (1.0 - cfg_w_tgt) * pred_neg

        # Euler step
        sample = sample - dt * pred
        mx.eval(sample)

        if verbose:
            print(" done", flush=True)

    return sample
