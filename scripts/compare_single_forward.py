"""Compare a single forward pass of the sparse structure flow model.

Loads the same model weights in both MLX and PyTorch, feeds the same
input tensor, and compares the output to find where numerical divergence
starts.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/compare_single_forward.py \
        --shared-noise /tmp/shared_noise_seed42.npz
"""

import argparse
import os
import sys
import time

import numpy as np


def run_mlx_forward(noise_np, cond_np, t_val=1000.0):
    """Run one forward pass of the MLX sparse structure flow model."""
    import mlx.core as mx
    sys.path.insert(0, '/private/tmp/trellis2mlx-molten-remesh-parity-0629')

    from trellmlx.models.sparse_structure_flow import SparseStructureFlowModel
    from trellmlx.weight_loader import load_weights

    HF_4B = os.path.expanduser(
        "~/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/"
        "snapshots/af44b45f2e35a493886929c6d786e563ec68364d/ckpts/"
    )

    model = SparseStructureFlowModel()
    load_weights(model, HF_4B + "ss_flow_img_dit_1_3B_64_bf16.safetensors", verbose=False)

    x = mx.array(noise_np)
    t = mx.array([t_val], dtype=mx.float32)
    c = mx.array(cond_np)

    t0 = time.perf_counter()
    out = model(x, t, c)
    mx.eval(out)
    print(f"MLX forward: {time.perf_counter()-t0:.2f}s")

    return np.array(out)


def run_pytorch_forward(noise_np, cond_np, t_val=1000.0):
    """Run one forward pass of the PyTorch sparse structure flow model."""
    import torch

    trellis_dir = "/Users/noahlyons/dev/trellis-mac/TRELLIS.2"
    sys.path.insert(0, trellis_dir)
    os.environ['ATTN_BACKEND'] = 'sdpa'
    os.environ['SPARSE_ATTN_BACKEND'] = 'sdpa'

    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.to("mps")

    model = pipeline.models['sparse_structure_flow_model']
    model.eval()

    x = torch.tensor(noise_np, device='mps')
    t = torch.tensor([t_val], device='mps')
    c = torch.tensor(cond_np, device='mps')

    with torch.no_grad():
        t0 = time.perf_counter()
        out = model(x, t, c)
        print(f"PyTorch forward: {time.perf_counter()-t0:.2f}s")

    return out.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-noise", required=True)
    parser.add_argument("--mode", choices=["mlx", "pytorch", "both"], default="mlx",
                        help="Which pipeline to run (both requires trellis-mac venv)")
    args = parser.parse_args()

    noise_data = np.load(args.shared_noise)
    noise = noise_data["ss_noise"]  # (1, 8, 16, 16, 16)

    # We need image conditioning too — use a dummy for now
    # Real conditioning would need DINOv3 features
    # For a single forward pass comparison, we can use zeros
    cond = np.zeros((1, 257, 1024), dtype=np.float32)  # typical DINOv3 shape

    print(f"Input noise: {noise.shape}, mean={noise.mean():.6f}, std={noise.std():.6f}")
    print(f"Conditioning: {cond.shape} (zeros)")

    if args.mode in ("mlx", "both"):
        mlx_out = run_mlx_forward(noise, cond)
        print(f"MLX output: {mlx_out.shape}, mean={mlx_out.mean():.6f}, "
              f"std={mlx_out.std():.6f}, min={mlx_out.min():.6f}, max={mlx_out.max():.6f}")
        np.save("/tmp/ss_forward_mlx.npy", mlx_out)

    if args.mode == "pytorch":
        pt_out = run_pytorch_forward(noise, cond)
        print(f"PyTorch output: {pt_out.shape}, mean={pt_out.mean():.6f}, "
              f"std={pt_out.std():.6f}, min={pt_out.min():.6f}, max={pt_out.max():.6f}")
        np.save("/tmp/ss_forward_pytorch.npy", pt_out)

    if args.mode == "both":
        pt_out = run_pytorch_forward(noise, cond)
        diff = np.abs(mlx_out - pt_out)
        print(f"\n=== Forward Pass Comparison ===")
        print(f"Abs diff: mean={diff.mean():.8f}, max={diff.max():.8f}, "
              f"p95={np.percentile(diff, 95):.8f}")
        rel_scale = np.abs(pt_out).mean()
        print(f"Relative diff: {diff.mean() / rel_scale:.6f}")
        print(f"Correlation: {np.corrcoef(mlx_out.ravel(), pt_out.ravel())[0,1]:.8f}")


if __name__ == "__main__":
    main()
