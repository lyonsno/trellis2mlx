"""Flow-matching Euler sampler for TRELLIS.2.

Implements the FlowEulerGuidanceIntervalSampler from the reference:
- Euler integration of the flow ODE
- Classifier-free guidance with interval masking
- Rescaled timestep scheduling
"""

import mlx.core as mx
import numpy as np


def flow_euler_sample(
    model,
    noise: mx.array,
    cond: mx.array,
    neg_cond: mx.array,
    steps: int = 12,
    guidance_strength: float = 7.5,
    guidance_rescale: float = 0.7,
    guidance_interval: tuple = (0.6, 1.0),
    rescale_t: float = 5.0,
    sigma_min: float = 1e-5,
    verbose: bool = True,
    concat_cond: mx.array = None,
    capture_first_step: dict | None = None,
    stop_after_first_step: bool = False,
    **model_kwargs,
):
    """Generate samples using flow-matching Euler sampling with CFG.

    Args:
        model: The flow model (SparseStructureFlowModel or similar).
        noise: Initial noise tensor [B, C, ...].
        cond: Positive conditioning [B, L, C_ctx].
        neg_cond: Negative conditioning (usually zeros) [B, L, C_ctx].
        steps: Number of Euler steps.
        guidance_strength: CFG strength (1.0 = no guidance).
        guidance_rescale: CFG rescale factor (0.0 = no rescale).
        guidance_interval: (low, high) — only apply guidance when t is in this range.
        rescale_t: Rescale factor for timestep schedule.
        sigma_min: Minimum noise scale.
        verbose: Print progress.

    Returns:
        Final denoised sample, same shape as noise.
    """
    sample = noise

    if concat_cond is not None:
        model_kwargs['concat_cond'] = concat_cond

    # Build cross-attention KV caches if the model supports it.
    # The image conditioning doesn't change between steps, so KV projections
    # can be computed once and reused for all 12 steps × 30 blocks.
    pos_kv_cache = None
    neg_kv_cache = None
    if hasattr(model, 'build_cross_kv_cache'):
        pos_kv_cache = model.build_cross_kv_cache(cond)
        if guidance_strength != 1.0:
            neg_kv_cache = model.build_cross_kv_cache(neg_cond)

    # Build timestep schedule
    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(steps)]

    for step_idx, (t, t_prev) in enumerate(t_pairs):
        if verbose:
            print(f"  Step {step_idx + 1}/{steps} (t={t:.4f}→{t_prev:.4f})", end="", flush=True)

        t_tensor = mx.array([1000 * t], dtype=mx.float32)

        # Determine if we apply guidance at this timestep
        apply_guidance = (
            guidance_strength != 1.0
            and guidance_interval[0] <= t <= guidance_interval[1]
        )

        if apply_guidance:
            # Two forward passes: conditioned and unconditioned
            kw = dict(**model_kwargs)
            if pos_kv_cache is not None:
                kw['cross_kv_cache'] = pos_kv_cache
            pred_pos = model(sample, t_tensor, cond, **kw)
            kw_neg = dict(**model_kwargs)
            if neg_kv_cache is not None:
                kw_neg['cross_kv_cache'] = neg_kv_cache
            pred_neg = model(sample, t_tensor, neg_cond, **kw_neg)
            mx.eval(pred_pos, pred_neg)

            # CFG combination
            pred = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg
            pred_cfg = pred

            # CFG rescale (reduces overexposure)
            std_ratio = None
            ratio_raw = None
            x_0_pos = None
            x_0_cfg = None
            x_0_rescaled = None
            x_0 = None
            std_pos = None
            std_cfg = None
            if guidance_rescale > 0:
                x_0_pos = _pred_to_xstart(sample, t, pred_pos, sigma_min)
                x_0_cfg = _pred_to_xstart(sample, t, pred, sigma_min)

                reduce_dims = list(range(1, x_0_pos.ndim))
                # Match PyTorch torch.std() Bessel correction (ddof=1).
                # Clamp the ratio to [0.5, 2.0] to prevent bf16 precision
                # noise from cascading through Euler steps — the std ratio
                # amplifies ~5x per step, producing catastrophic divergence
                # after 4 steps without clamping.
                n = 1
                for d in reduce_dims:
                    n *= x_0_pos.shape[d]
                bessel = n / (n - 1)
                std_pos = mx.sqrt(mx.var(x_0_pos, axis=reduce_dims, keepdims=True) * bessel)
                std_cfg = mx.sqrt(mx.var(x_0_cfg, axis=reduce_dims, keepdims=True) * bessel)

                safe_std_cfg = mx.where(std_cfg > 0, std_cfg, mx.ones_like(std_cfg))
                ratio_raw = std_pos / safe_std_cfg
                ratio_raw = mx.where(std_cfg > 0, ratio_raw, mx.ones_like(ratio_raw))
                std_ratio = mx.clip(ratio_raw, 0.5, 2.0)
                x_0_rescaled = x_0_cfg * std_ratio
                x_0 = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg
                pred = _xstart_to_pred(sample, t, x_0, sigma_min)
        else:
            # Single forward pass with positive conditioning
            kw = dict(**model_kwargs)
            if pos_kv_cache is not None:
                kw['cross_kv_cache'] = pos_kv_cache
            pred = model(sample, t_tensor, cond, **kw)
            mx.eval(pred)
            pred_pos = pred
            pred_neg = None
            pred_cfg = pred
            std_ratio = None
            ratio_raw = None
            x_0_pos = None
            x_0_cfg = None
            x_0_rescaled = None
            x_0 = None
            std_pos = None
            std_cfg = None

        # Euler step
        sample_next = sample - (t - t_prev) * pred
        if capture_first_step is not None and step_idx == 0:
            capture_first_step.update(
                {
                    "pred_pos": pred_pos,
                    "pred_neg": pred_neg,
                    "pred_cfg": pred_cfg,
                    "x0_pos": x_0_pos,
                    "x0_cfg": x_0_cfg,
                    "std_pos": std_pos,
                    "std_cfg": std_cfg,
                    "ratio_raw": ratio_raw,
                    "std_ratio": std_ratio,
                    "ratio_effective": std_ratio,
                    "x0_rescaled": x_0_rescaled,
                    "x0_after_rescale": x_0,
                    "pred_final": pred,
                    "sample_next": sample_next,
                    "t": mx.array(t, dtype=mx.float32),
                    "t_prev": mx.array(t_prev, dtype=mx.float32),
                }
            )
            mx.eval(*[value for value in capture_first_step.values() if value is not None])
        sample = sample_next
        mx.eval(sample)

        if verbose:
            print(f" done", flush=True)
        if stop_after_first_step:
            break

    return sample


def _pred_to_xstart(x_t, t, pred, sigma_min):
    """Convert velocity prediction to x_0 estimate."""
    return (1 - sigma_min) * x_t - (sigma_min + (1 - sigma_min) * t) * pred


def _xstart_to_pred(x_t, t, x_0, sigma_min):
    """Convert x_0 estimate back to velocity prediction."""
    return ((1 - sigma_min) * x_t - x_0) / (sigma_min + (1 - sigma_min) * t)
