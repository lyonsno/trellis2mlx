# trellis2mlx

MLX-native [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) inference for Apple Silicon.

Run [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) 3D generation on Mac using [MLX](https://github.com/ml-explore/mlx). No NVIDIA GPU required. Work in progress — sparse structure + shape latent stages working, mesh decoder coming next.

## What works now

Stages 1 and 2 of the TRELLIS.2 pipeline run entirely in MLX:

```bash
python smoke_stage2.py --image photo.png
```

**Stage 1 — Sparse Structure:** SparseStructureFlowModel (1.29B params) + decoder (73.7M params) → 64³ occupancy grid → downsampled sparse coordinates.

**Stage 2 — Shape Latent:** SLatFlowModel (1.29B params) → shape latents at each occupied voxel, ready for mesh decoding.

### Performance (M4 Max, 128GB)

| Metric | trellis2mlx (MLX) | trellis-mac (PyTorch MPS) |
|--------|-------------------|---------------------------|
| Stage 1 (sparse structure, 12 steps) | **48s** | ~10-15 min (chunked SDPA) |
| Stage 2 (shape latent, 3.3K tokens, 12 steps) | **~60s** | ~10-15 min (chunked SDPA) |
| Two-stage total | **~2 min** | ~20-30 min |
| Peak memory (stage 2) | **~3 GB** | 40-55 GB (chunked), 128 GB+ (unchunked) |
| Minimum hardware | **8 GB** (any Apple Silicon) | 24 GB+ |

### Numerical parity (12-step sampling, same weights + noise)

| Step | Correlation | Max diff |
|------|-------------|----------|
| 1 | 0.999999 | 0.009 |
| 3 | 0.999991 | 0.020 |
| 6 | 0.999938 | 0.051 |
| 9 | 0.998852 | 0.434 |
| 12 | 0.968466 | 2.128 |

Divergence is monotonic precision accumulation (bf16 → fp16), not architectural — single forward pass correlation is 0.999999.

### Quantization (experimental)

INT4 quantization via MLX reduces model weight memory 6.4×:

| | FP16 | INT4 |
|---|---|---|
| Weight memory | 5.17 GB | 0.81 GB |
| Forward pass | works | works |

### What's next

- [x] SparseStructureFlowModel (1.29B param DiT) — numerically verified
- [x] SparseStructureDecoder (73.7M param Conv U-Net)
- [x] SLatFlowModel (1.29B param sparse token DiT)
- [x] Weight loading (640/640 + 74/74 + 640/640 params)
- [x] Flow Euler sampler with CFG + guidance interval + rescale
- [x] 3D RoPE position embedding (dynamic, computed from input shape)
- [x] Image conditioning via DINOv3
- [x] MLX Flash Attention (`mx.fast.scaled_dot_product_attention`)
- [x] Periodic eval to prevent memory bus starvation
- [x] INT4 quantization utility
- [x] Two-stage smoke: image → sparse structure → shape latent → colored voxel mesh
- [x] 12-step numerical parity verified against PyTorch
- [ ] Shape SLat decoder → full mesh extraction (`flexible_dual_grid_to_mesh`)
- [ ] Texture SLat flow + decoder → PBR textures
- [ ] Full pipeline: image → textured GLB
- [ ] INT4 speed benchmarks
- [ ] `mx.compile` optimization
- [ ] Native macOS/iOS app via mlx-swift

## Why MLX

TRELLIS.2 runs on Mac via [trellis-mac](https://github.com/shivampkumar/trellis-mac) (PyTorch MPS), but:

- **Memory:** MPS SDPA materializes full N×N attention matrices (275 GB for 262K tokens). MLX's SDPA is real Flash Attention — O(N) memory, handles any sequence length at ~3 GB.
- **Quantization:** INT4 drops weights from 5.17 GB to 0.81 GB. Proportional bandwidth reduction on Apple Silicon's unified memory.
- **Accessibility:** Runs on any Apple Silicon device (8 GB+), not just high-end Macs.
- **Bus-friendly:** Periodic eval yields memory bus between GPU bursts, preventing beachballs during generation.

## Quick start

```bash
git clone https://github.com/lyonsno/trellis2mlx.git
cd trellis2mlx
uv venv .venv --python python3.11
source .venv/bin/activate
uv pip install mlx numpy safetensors trimesh scikit-image scipy pillow tqdm huggingface-hub

# For image conditioning (temporary — uses PyTorch for DINOv3):
uv pip install torch torchvision transformers
# Also need trellis-mac checkout for the DINOv3 feature extractor:
git clone --depth 1 https://github.com/shivampkumar/trellis-mac.git ~/dev/trellis-mac

# HuggingFace auth (needed for gated DINOv3 weights):
huggingface-cli login
# Request access: https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m

# Download model weights:
huggingface-cli download microsoft/TRELLIS.2-4B
huggingface-cli download microsoft/TRELLIS-image-large

# Stage 1 only (sparse structure → occupancy mesh):
PYTHONPATH=. python smoke.py --image your_image.png

# Stages 1+2 (sparse structure + shape latent → colored voxel mesh):
PYTHONPATH=. python smoke_stage2.py --image your_image.png

open /tmp/trellis-mlx-stage2.glb
```

Without `--image`, runs with random conditioning (abstract shapes, useful for verifying the pipeline works).

## Tests

```bash
uv pip install pytest
PYTHONPATH=. pytest tests/ -v
```

40 tests covering LayerNorm32, MultiHeadRMSNorm, SDPA, variable-length attention, RoPE, TimestepEmbedder, SparseStructureFlowModel, SLatFlowModel, SparseStructureDecoder, sampler, weight loader, and quantization.

## Architecture

See [docs/architecture-map.md](docs/architecture-map.md) for the full TRELLIS.2-4B architecture reference.

```
trellmlx/
├── models/
│   ├── sparse_structure_flow.py   # 1.29B param DiT (30 blocks, 3D RoPE, adaLN-Zero)
│   ├── sparse_structure_decoder.py # 73.7M param Conv3d U-Net (pixel shuffle upsample)
│   └── slat_flow.py               # 1.29B param sparse token DiT (shape detail)
├── modules/
│   ├── attention.py               # mx.fast.scaled_dot_product_attention + MultiHeadRMSNorm
│   ├── rope.py                    # 3D Rotary Position Embedding
│   └── norm.py                    # LayerNorm32 (fp32 accumulation)
├── samplers.py                    # Flow Euler sampler with CFG + guidance interval
├── weight_loader.py               # Checkpoint loading (key remap, Conv3d permute, bf16/fp16)
└── quantize.py                    # INT4/INT8 quantization utility
```

## Credits

- [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) by Microsoft Research — the model
- [trellis-mac](https://github.com/shivampkumar/trellis-mac) by Shivam Kumar — proved Mac viability
- [trellis2-apple](https://github.com/pedronaugusto/trellis2-apple) by Pedro Naugusto — Metal modules
- [MLX](https://github.com/ml-explore/mlx) by Apple — the framework

## License

MIT (porting code). Upstream model weights are subject to their own licenses — see [trellis-mac](https://github.com/shivampkumar/trellis-mac#license) for details.
