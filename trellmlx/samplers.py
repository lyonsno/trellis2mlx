"""Flow-matching Euler sampler for TRELLIS.2.

Implements the FlowEulerGuidanceIntervalSampler from the reference:
- Euler integration of the flow ODE
- Classifier-free guidance with interval masking
- Rescaled timestep scheduling
"""

import mlx.core as mx
import numpy as np


_source_sparse_cfg_std_kernel = None
_source_cuda_dense_cfg_std_kernel = None


def dense_cfg_rescale_std_backend_identity(
    shape: tuple[int, ...],
    *,
    dtype: str,
) -> dict[str, object]:
    geometry = {"shape": list(shape), "dtype": dtype}
    if tuple(shape) == (1, 8, 16, 16, 16) and dtype == "float32":
        return {
            "backend": "source-cuda-t4-welford-metal",
            "algorithm": "pytorch-2.10-cuda-welford-vt2-block512",
            "experimental": True,
            "authenticated_contract": {
                **geometry,
                "thread_count": 512,
                "input_vector_width": 2,
                "online_m2_update": "fp32-fma",
                "shared_tree_offsets": [256, 128, 64, 32],
                "warp_tree_offsets": [16, 8, 4, 2, 1],
                "correction": 1,
                "cuda_device_anchor": "Tesla T4 sm_75",
                "torch_anchor": "2.10.0+cu128",
                "source_checkpoint_sha256": (
                    "c202368155d3f1c28ea7132b2909f4656e79df4a073f2ff04905b5a452ce6520"
                ),
                "authenticated_scalar_count": 12,
            },
        }
    return {
        "backend": "mlx-native-var",
        "algorithm": "mlx-var-bessel-corrected",
        "experimental": True,
        "effective_contract": geometry,
        "excluded_shape": [1, 8, 16, 16, 16],
    }


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
    capture_steps: list[dict] | None = None,
    stop_after_first_step: bool = False,
    start_step_index: int = 0,
    sparse_block_injection=None,
    shape_block_injection=None,
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
    sparse_token_rescale = "coords" in model_kwargs and len(noise.shape) == 2
    if sparse_token_rescale:
        coords = model_kwargs["coords"]
        if coords.ndim != 2 or coords.shape != (noise.shape[0], 3):
            raise ValueError(
                "sparse-token sampling requires spatial coords with shape "
                f"[tokens,3], got noise={noise.shape}, coords={coords.shape}"
            )
        cond_batches = int(cond.shape[0])
        neg_cond_batches = int(neg_cond.shape[0])
        if cond_batches != 1 or neg_cond_batches != 1:
            raise ValueError(
                "sparse-token sampling currently requires one conditioning batch; "
                f"got cond={cond_batches}, neg_cond={neg_cond_batches}"
            )

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
    if start_step_index < 0 or start_step_index >= len(t_pairs):
        raise ValueError(
            f"start_step_index must be in [0, {len(t_pairs) - 1}], got {start_step_index}"
        )

    for step_idx, (t, t_prev) in enumerate(t_pairs[start_step_index:], start=start_step_index):
        if verbose:
            print(f"  Step {step_idx + 1}/{steps} (t={t:.4f}→{t_prev:.4f})", end="", flush=True)

        t_tensor = mx.array([1000 * t], dtype=mx.float32)

        # Determine if we apply guidance at this timestep
        apply_guidance = (
            guidance_strength != 1.0
            and guidance_interval[0] <= t <= guidance_interval[1]
        )
        capture_this_step = (
            (capture_first_step is not None and step_idx == start_step_index)
            or capture_steps is not None
        )

        if apply_guidance:
            # Two forward passes: conditioned and unconditioned
            kw = _branch_model_kwargs(
                model_kwargs,
                sparse_block_injection=sparse_block_injection,
                shape_block_injection=shape_block_injection,
                step_index=step_idx,
                branch="pos",
            )
            if pos_kv_cache is not None:
                kw['cross_kv_cache'] = pos_kv_cache
            pred_pos = model(sample, t_tensor, cond, **kw)
            kw_neg = _branch_model_kwargs(
                model_kwargs,
                sparse_block_injection=sparse_block_injection,
                shape_block_injection=shape_block_injection,
                step_index=step_idx,
                branch="neg",
            )
            if neg_kv_cache is not None:
                kw_neg['cross_kv_cache'] = neg_kv_cache
            pred_neg = model(sample, t_tensor, neg_cond, **kw_neg)
            mx.eval(pred_pos, pred_neg)

            # CFG combination
            pred = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg
            pred_cfg = pred

            # CFG rescale (reduces overexposure)
            x_0_pos = None
            x_0_cfg = None
            std_pos = None
            std_cfg = None
            ratio_raw = None
            std_ratio = None
            x_0_rescaled = None
            x_0 = None
            if guidance_rescale > 0 or capture_this_step:
                x_0_pos = _pred_to_xstart(sample, t, pred_pos, sigma_min)
                x_0_cfg = _pred_to_xstart(sample, t, pred, sigma_min)

                std_pos = _cfg_rescale_std(x_0_pos, sparse_tokens=sparse_token_rescale)
                std_cfg = _cfg_rescale_std(x_0_cfg, sparse_tokens=sparse_token_rescale)

                safe_std_cfg = mx.where(std_cfg > 0, std_cfg, mx.ones_like(std_cfg))
                ratio_raw = std_pos / safe_std_cfg
                ratio_raw = mx.where(std_cfg > 0, ratio_raw, mx.ones_like(ratio_raw))
                std_ratio = ratio_raw
                x_0_rescaled = x_0_cfg * std_ratio
                if guidance_rescale > 0:
                    x_0 = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg
                    pred = _xstart_to_pred(sample, t, x_0, sigma_min)
                else:
                    x_0 = x_0_cfg
        else:
            # Single forward pass with positive conditioning
            kw = _branch_model_kwargs(
                model_kwargs,
                sparse_block_injection=sparse_block_injection,
                shape_block_injection=shape_block_injection,
                step_index=step_idx,
                branch="pos",
            )
            if pos_kv_cache is not None:
                kw['cross_kv_cache'] = pos_kv_cache
            pred = model(sample, t_tensor, cond, **kw)
            mx.eval(pred)
            pred_pos = pred
            pred_neg = pred
            pred_cfg = pred
            x_0_pos = None
            x_0_cfg = None
            std_pos = None
            std_cfg = None
            ratio_raw = None
            std_ratio = None
            x_0_rescaled = None
            x_0 = None
            if capture_this_step:
                x_0 = _pred_to_xstart(sample, t, pred, sigma_min)
                x_0_pos = x_0
                x_0_cfg = x_0
                std_pos = _cfg_rescale_std(x_0, sparse_tokens=sparse_token_rescale)
                std_cfg = std_pos
                ratio_raw = mx.ones_like(std_pos)
                std_ratio = ratio_raw
                x_0_rescaled = x_0

        # Euler step
        euler_dt = np.float32(t - t_prev).item()
        euler_delta = euler_dt * pred
        mx.eval(euler_delta)
        sample_next = sample - euler_delta
        if capture_this_step:
            step_payload = {
                "sample_in": sample,
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
            if capture_first_step is not None and step_idx == start_step_index:
                capture_first_step.update(step_payload)
                mx.eval(*[value for value in capture_first_step.values() if value is not None])
            if capture_steps is not None:
                capture_steps.append(step_payload)
                mx.eval(*[value for value in step_payload.values() if value is not None])
        sample = sample_next
        mx.eval(sample)

        if verbose:
            print(f" done", flush=True)
        if stop_after_first_step:
            break

    return sample


def _branch_model_kwargs(
    model_kwargs: dict,
    *,
    sparse_block_injection,
    shape_block_injection,
    step_index: int,
    branch: str,
) -> dict:
    kw = dict(**model_kwargs)
    if sparse_block_injection is not None:
        if hasattr(sparse_block_injection, "active_for_step_branch"):
            active_injection = sparse_block_injection.active_for_step_branch(
                step_index=step_index,
                branch=branch,
            )
        else:
            active_injection = (
                sparse_block_injection
                if sparse_block_injection.applies(step_index=step_index, branch=branch)
                else None
            )
        kw["sparse_block_injection_branch"] = branch
        kw["sparse_block_injection"] = active_injection
    if shape_block_injection is not None:
        if hasattr(shape_block_injection, "active_for_step_branch"):
            active_injection = shape_block_injection.active_for_step_branch(
                step_index=step_index,
                branch=branch,
            )
        else:
            active_injection = (
                shape_block_injection
                if shape_block_injection.applies(step_index=step_index, branch=branch)
                else None
            )
        kw["shape_block_injection_branch"] = branch
        kw["shape_block_injection"] = active_injection
    return kw


def _pred_to_xstart(x_t, t, pred, sigma_min):
    """Convert velocity prediction to x_0 estimate."""
    one_minus_sigma = mx.array(1 - sigma_min, dtype=x_t.dtype)
    coefficient = mx.array(
        sigma_min + (1 - sigma_min) * t,
        dtype=x_t.dtype,
    )
    return one_minus_sigma * x_t - coefficient * pred


def _xstart_to_pred(x_t, t, x_0, sigma_min):
    """Convert x_0 estimate back to velocity prediction."""
    one_minus_sigma_value = 1 - sigma_min
    coefficient_value = sigma_min + one_minus_sigma_value * t
    one_minus_sigma = mx.array(one_minus_sigma_value, dtype=x_t.dtype)
    inverse_coefficient = mx.array(
        1.0 / coefficient_value,
        dtype=x_t.dtype,
    )
    return (one_minus_sigma * x_t - x_0) * inverse_coefficient


def _cfg_rescale_std(x_0: mx.array, *, sparse_tokens: bool) -> mx.array:
    """Match source CFG-rescale std axes for dense grids versus SparseTensor tokens."""
    if sparse_tokens:
        return _source_sparse_cfg_rescale_std(x_0)

    if x_0.ndim == 5 and x_0.shape == (1, 8, 16, 16, 16):
        return _source_cuda_dense_cfg_rescale_std(x_0)

    reduce_dims = list(range(1, x_0.ndim))
    n = 1
    for d in reduce_dims:
        n *= x_0.shape[d]
    # Match PyTorch torch.std() Bessel correction (correction=1) for dense tensors.
    bessel = n / (n - 1)
    return mx.sqrt(mx.var(x_0, axis=reduce_dims, keepdims=True) * bessel)


def _source_cuda_dense_cfg_rescale_std(x_0: mx.array) -> mx.array:
    """Reproduce PyTorch 2.10 CUDA's dense CFG Welford reduction."""
    global _source_cuda_dense_cfg_std_kernel

    if x_0.ndim != 5 or x_0.shape != (1, 8, 16, 16, 16):
        raise ValueError(
            "source-CUDA dense CFG rescale requires 1x8x16x16x16 input"
        )

    if _source_cuda_dense_cfg_std_kernel is None:
        header = r"""
            struct WelfordDataCfg {
                float mean;
                float m2;
                float count;
            };

            inline WelfordDataCfg cfg_welford_online(
                float value,
                WelfordDataCfg current
            ) {
                float new_count = current.count + 1.0f;
                float delta = value - current.mean;
                float new_mean = current.mean + delta / new_count;
                float new_delta = value - new_mean;
                return {
                    new_mean,
                    metal::fma(delta, new_delta, current.m2),
                    new_count
                };
            }

            inline WelfordDataCfg cfg_welford_combine(
                WelfordDataCfg first,
                WelfordDataCfg second
            ) {
                if (first.count == 0.0f) {
                    return second;
                }
                if (second.count == 0.0f) {
                    return first;
                }
                float delta = second.mean - first.mean;
                float count = first.count + second.count;
                float second_fraction = second.count / count;
                return {
                    first.mean + delta * second_fraction,
                    first.m2 + second.m2
                        + delta * delta * first.count * second_fraction,
                    count
                };
            }
        """
        source = r"""
            constexpr uint thread_count = 512;
            constexpr uint vector_count = 16384;

            uint thread_index = thread_position_in_threadgroup.x;
            WelfordDataCfg even = {0.0f, 0.0f, 0.0f};
            WelfordDataCfg odd = {0.0f, 0.0f, 0.0f};
            for (uint vector_index = thread_index;
                 vector_index < vector_count;
                 vector_index += thread_count) {
                uint offset = vector_index * 2;
                even = cfg_welford_online(
                    static_cast<float>(inp[offset]), even);
                odd = cfg_welford_online(
                    static_cast<float>(inp[offset + 1]), odd);
            }
            WelfordDataCfg value = cfg_welford_combine(even, odd);

            threadgroup float shared_mean[thread_count];
            threadgroup float shared_m2[thread_count];
            threadgroup float shared_count[thread_count];
            shared_mean[thread_index] = value.mean;
            shared_m2[thread_index] = value.m2;
            shared_count[thread_index] = value.count;

            for (uint offset = 256; offset >= 32; offset >>= 1) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (thread_index < offset) {
                    WelfordDataCfg upper = {
                        shared_mean[thread_index + offset],
                        shared_m2[thread_index + offset],
                        shared_count[thread_index + offset]
                    };
                    value = cfg_welford_combine(value, upper);
                    shared_mean[thread_index] = value.mean;
                    shared_m2[thread_index] = value.m2;
                    shared_count[thread_index] = value.count;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (thread_index < 32) {
                value = {
                    shared_mean[thread_index],
                    shared_m2[thread_index],
                    shared_count[thread_index]
                };
                for (ushort offset = 16; offset > 0; offset >>= 1) {
                    WelfordDataCfg upper = {
                        simd_shuffle_down(value.mean, offset),
                        simd_shuffle_down(value.m2, offset),
                        simd_shuffle_down(value.count, offset)
                    };
                    value = cfg_welford_combine(value, upper);
                }
                if (thread_index == 0) {
                    float variance = value.m2 / (value.count - 1.0f);
                    out[0] = metal::precise::sqrt(variance);
                }
            }
        """
        _source_cuda_dense_cfg_std_kernel = mx.fast.metal_kernel(
            name="cfg_rescale_source_cuda_t4_welford_fp32",
            input_names=["inp"],
            output_names=["out"],
            header=header,
            source=source,
            ensure_row_contiguous=True,
        )

    return _source_cuda_dense_cfg_std_kernel(
        inputs=[x_0.astype(mx.float32)],
        grid=(512, 1, 1),
        threadgroup=(512, 1, 1),
        output_shapes=[(1,)],
        output_dtypes=[mx.float32],
    )[0].reshape((1, 1, 1, 1, 1))


def _source_sparse_cfg_rescale_std(x_0: mx.array) -> mx.array:
    """Reproduce SparseTensor.std's CUDA row-tree and segment reduction order."""
    global _source_sparse_cfg_std_kernel

    if x_0.ndim != 2:
        raise ValueError(f"sparse CFG rescale expects [tokens, channels], got {x_0.shape}")
    rows, channels = (int(value) for value in x_0.shape)
    if rows <= 0 or channels <= 0:
        raise ValueError(f"sparse CFG rescale requires non-empty input, got {x_0.shape}")
    if channels > 32 or channels & (channels - 1):
        return mx.sqrt(mx.var(x_0))

    if _source_sparse_cfg_std_kernel is None:
        source = r"""
            uint row_count = rows[0];
            uint channel_count = channels[0];
            float mean_sum = 0.0f;
            float mean2_sum = 0.0f;

            for (uint row = 0; row < row_count; ++row) {
                float values[32];
                float squares[32];
                uint row_offset = row * channel_count;
                for (uint channel = 0; channel < 32; ++channel) {
                    float value = channel < channel_count
                        ? static_cast<float>(inp[row_offset + channel])
                        : 0.0f;
                    values[channel] = value;
                    squares[channel] = value * value;
                }
                for (uint offset = 16; offset > 0; offset >>= 1) {
                    for (uint channel = 0; channel < offset; ++channel) {
                        values[channel] =
                            values[channel] + values[channel + offset];
                        squares[channel] =
                            squares[channel] + squares[channel + offset];
                    }
                }
                float inverse_channels = 1.0f / static_cast<float>(channel_count);
                float row_mean = values[0] * inverse_channels;
                float row_mean2 = squares[0] * inverse_channels;
                mean_sum = mean_sum + row_mean;
                mean2_sum = mean2_sum + row_mean2;
            }

            float mean = mean_sum / static_cast<float>(row_count);
            float mean2 = mean2_sum / static_cast<float>(row_count);
            float variance = mean2 - mean * mean;
            out[0] = metal::precise::sqrt(variance);
        """
        _source_sparse_cfg_std_kernel = mx.fast.metal_kernel(
            name="cfg_rescale_source_sparse_row_tree_segment_fp32",
            input_names=["inp", "rows", "channels"],
            output_names=["out"],
            source=source,
        )

    out = _source_sparse_cfg_std_kernel(
        inputs=[
            x_0,
            mx.array([rows], dtype=mx.uint32),
            mx.array([channels], dtype=mx.uint32),
        ],
        template=[("T", x_0.dtype)],
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[(1,)],
        output_dtypes=[mx.float32],
    )[0]
    return out.reshape(())
