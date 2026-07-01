"""Generate shared noise tensors for matched pipeline comparison.

Creates numpy noise arrays that both trellis2mlx (MLX) and Trellis-Mac
(PyTorch) can load to ensure identical initial noise for flow sampling.

Usage:
    python scripts/generate_shared_noise.py --seed 42 --output /tmp/shared_noise.npz

Then pass to pipelines:
    # trellis2mlx
    generate.py --shared-noise /tmp/shared_noise.npz ...
    # Trellis-Mac
    run_official_trellis2.py --shared-noise /tmp/shared_noise.npz ...
"""

import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="/tmp/shared_noise.npz")
    # Noise shape depends on the sparse structure output, which we don't
    # know in advance. Generate the largest shapes we might need.
    parser.add_argument("--max-tokens", type=int, default=100000,
                        help="Max SLat tokens (for noise tensor sizing)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    noise = {
        # Stage 1: sparse structure flow
        "ss_noise": rng.standard_normal((1, 8, 16, 16, 16)).astype(np.float32),
        # Stage 2: shape SLat flow (size depends on sparse structure output)
        # Generate max size, pipelines will slice to actual N_lr
        "slat_noise_pool": rng.standard_normal((args.max_tokens, 32)).astype(np.float32),
        # Stage 4: texture SLat flow (same token count as shape)
        "tex_noise_pool": rng.standard_normal((args.max_tokens, 32)).astype(np.float32),
    }

    np.savez(args.output, **noise)
    print(f"Saved shared noise to {args.output}")
    for k, v in noise.items():
        print(f"  {k}: {v.shape} {v.dtype}")


if __name__ == "__main__":
    main()
