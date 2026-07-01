"""Run the real Trellis-Mac sampler on the sparse structure flow model.

Uses the actual FlowEulerGuidanceIntervalSampler, not a reimplementation.
Saves z_s and per-step sample stats for comparison against MLX.

Usage (via Greenroom):
    /Users/noahlyons/dev/trellis-mac/.venv/bin/python -u \
        scripts/pytorch_single_forward.py \
        --shared-noise /path/to/shared_noise.npz \
        --output-dir /path/to/output
"""

import argparse
import json
import os
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-noise", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ['ATTN_BACKEND'] = 'sdpa'
    os.environ['SPARSE_ATTN_BACKEND'] = 'sdpa'
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    import torch

    sys.path.insert(0, '/Users/noahlyons/dev/trellis-mac/TRELLIS.2')

    from trellis2 import models as trellis_models
    from trellis2.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler

    HF_4B = os.path.expanduser(
        "~/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/"
        "snapshots/af44b45f2e35a493886929c6d786e563ec68364d/ckpts/"
    )

    print("Loading model...", flush=True)
    model = trellis_models.from_pretrained(HF_4B + "ss_flow_img_dit_1_3B_64_bf16")
    model = model.to("mps").eval()
    print("Loaded", flush=True)

    # Load shared noise
    noise_data = np.load(args.shared_noise)
    noise = torch.tensor(noise_data['ss_noise'], device='mps')
    cond = torch.zeros((1, 257, 1024), dtype=torch.float32, device='mps')
    neg_cond = torch.zeros_like(cond)

    # Use the real sampler with the real params from pipeline.json
    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)

    # Hook sample_once to capture per-step states
    _step_states = []
    _orig_sample_once = sampler.sample_once

    def _hooked_sample_once(model, x_t, t, t_prev, cond, **kwargs):
        result = _orig_sample_once(model, x_t, t, t_prev, cond, **kwargs)
        s = result.pred_x_prev.cpu().numpy()
        _step_states.append({
            'step': len(_step_states),
            't': float(t),
            'sample_mean': float(s.mean()),
            'sample_std': float(s.std()),
            'sample_min': float(s.min()),
            'sample_max': float(s.max()),
        })
        print(f"  Step {len(_step_states)-1}: t={t:.4f} mean={s.mean():.6f} std={s.std():.6f}",
              flush=True)
        return result

    sampler.sample_once = _hooked_sample_once

    print(f"Sampling ({args.steps} steps)...", flush=True)
    t0 = time.perf_counter()
    result = sampler.sample(
        model, noise, cond, neg_cond,
        steps=args.steps,
        rescale_t=5.0,
        guidance_strength=7.5,
        guidance_interval=(0.6, 1.0),
        guidance_rescale=0.7,
        verbose=False,
    )
    z_s = result.samples
    print(f"Done: {time.perf_counter()-t0:.1f}s", flush=True)

    z_s_np = z_s.cpu().numpy()
    print(f"z_s: mean={z_s_np.mean():.8f} std={z_s_np.std():.8f}")

    np.save(os.path.join(args.output_dir, "z_s_pytorch.npy"), z_s_np)
    with open(os.path.join(args.output_dir, "ss_flow_debug_pytorch.json"), "w") as f:
        json.dump(_step_states, f, indent=2)

    print("Saved", flush=True)


if __name__ == "__main__":
    main()
