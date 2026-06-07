# trellis2mlx

MLX-native [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) inference for Apple Silicon.

Run [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) 3D generation on Mac using [MLX](https://github.com/ml-explore/mlx). No NVIDIA GPU required. Image → textured 3D mesh in ~9 minutes on M4 Max.

## What works now

Full pipeline: image → textured GLB with PBR materials.

```bash
# Full pipeline (two-pass, high quality):
PYTHONPATH=. python generate.py --image photo.png --output mesh.glb

# Stage 1+2 only (fast preview, colored voxels):
PYTHONPATH=. python smoke_stage2.py --image photo.png
```

The pipeline uses a two-pass architecture matching the TRELLIS.2 reference:
1. **Sparse Structure** — SparseStructureFlowModel (1.29B params) + decoder → 64³ occupancy grid
2. **LR Shape Latent** — SLatFlowModel (1.29B params) at ~1.7K sparse tokens
3. **Upsample** — Decoder subdivision predictions → high-res coordinate structure
4. **HR Shape Latent** — SLatFlowModel again at ~29K dense tokens
5. **Decode + Extract** — Sparse UNet decoder → `flexible_dual_grid_to_mesh` → GLB

### Performance (M4 Max, 128GB)

| Stage | Time | Notes |
|-------|------|-------|
| Sparse structure (12 steps) | ~34s | 1.29B param DiT on 16³ grid |
| LR SLat (1.7K tokens, 12 steps) | ~14s | |
| Upsample → HR coords | ~6s | 463K voxels |
| HR SLat (7.2K tokens, 12 steps) | ~2 min | 1024 cascade model |
| Shape decode (1.9M voxels) | ~73s | 474M param sparse UNet |
| Mesh extraction + simplify | ~3s | 3.7M → 200K faces |
| Texture SLat (7.2K tokens, 12 steps) | ~1.3 min | No CFG (single pass) |
| Texture decode (1.9M voxels) | ~29s | 6-channel PBR |
| UV unwrap + texture bake | ~2.2 min | xatlas + trilinear sample |
| **Total** | **~8.6 min** | |

Peak memory: ~3 GB for SLat flow, ~5 GB during decode. Runs on any Apple Silicon Mac.

For comparison, the PyTorch MPS path (trellis-mac) takes 20-30 min for stages 1-2 alone at 40-55 GB memory, with no mesh decode or texture support on Mac.

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

### Roadmap

- [x] SparseStructureFlowModel (1.29B param DiT) — numerically verified
- [x] SparseStructureDecoder (73.7M param Conv U-Net)
- [x] SLatFlowModel (1.29B param sparse token DiT)
- [x] ShapeSLatDecoder (474M param sparse UNet)
- [x] Two-pass architecture (LR SLat → upsample → HR SLat → decode)
- [x] `flexible_dual_grid_to_mesh` mesh extraction
- [x] SLat denormalization (pipeline.json mean/std)
- [x] Weight loading (640/640 + 74/74 + 640/640 + 292/292 params)
- [x] Flow Euler sampler with CFG + guidance interval + rescale
- [x] 3D RoPE position embedding (dynamic, computed from input shape)
- [x] Image conditioning via DINOv3
- [x] MLX Flash Attention (`mx.fast.scaled_dot_product_attention`)
- [x] Periodic eval to prevent memory bus starvation
- [x] INT4 quantization utility
- [x] 12-step numerical parity verified against PyTorch
- [x] Mesh simplification via fast-simplification (3.7M → 200K faces in ~1s)
- [x] Texture SLat flow + decoder → per-voxel PBR attributes
- [x] UV unwrap (xatlas) + texture baking (trilinear sample + cv2 inpaint)
- [x] Full pipeline: image → textured GLB with PBR materials (~8.6 min)
- [x] 1024 cascade architecture (LR 512 model + HR 1024 model)
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

# Full pipeline (image → mesh):
PYTHONPATH=. python generate.py --image your_image.png

# Quick preview (stages 1+2 only, colored voxels):
PYTHONPATH=. python smoke_stage2.py --image your_image.png

open /tmp/trellis-mlx-mesh.glb
```

Without `--image`, runs with random conditioning (abstract shapes, useful for verifying the pipeline works).

## Tests

```bash
uv pip install pytest
PYTHONPATH=. pytest tests/ -v
```

43 tests covering all modules.

## Architecture

See [docs/architecture-map.md](docs/architecture-map.md) for the full TRELLIS.2-4B architecture reference.

```
trellmlx/
├── models/
│   ├── sparse_structure_flow.py   # 1.29B param DiT (30 blocks, 3D RoPE, adaLN-Zero)
│   ├── sparse_structure_decoder.py # 73.7M param Conv3d U-Net (pixel shuffle upsample)
│   ├── slat_flow.py               # 1.29B param sparse token DiT (shape detail)
│   └── shape_slat_decoder.py      # 474M param sparse UNet (Channel2Spatial upsample)
├── modules/
│   ├── attention.py               # mx.fast.scaled_dot_product_attention + MultiHeadRMSNorm
│   ├── rope.py                    # 3D Rotary Position Embedding
│   ├── norm.py                    # LayerNorm32 (fp32 accumulation)
│   └── sparse_conv.py             # Submanifold sparse 3D convolution (gather-scatter)
├── mesh_extract.py                # flexible_dual_grid_to_mesh (numpy)
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
