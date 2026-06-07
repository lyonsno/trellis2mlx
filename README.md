# trellis2mlx

MLX-native [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) inference for Apple Silicon.

Generate 3D meshes from single images using [MLX](https://github.com/ml-explore/mlx) on Mac. No NVIDIA GPU required.

## What works now

The **sparse structure pipeline** is complete and verified:

```bash
python smoke.py --image photo.png --output mesh.glb
```

This runs TRELLIS.2's first stage entirely in MLX:
- **SparseStructureFlowModel** (1.29B params) — 30-block DiT with 3D RoPE + adaLN-Zero + cross-attention to DINOv3 image features
- **SparseStructureDecoder** (73.7M params) — 3D Conv U-Net with pixel shuffle upsampling
- Flow-matching Euler sampler with classifier-free guidance
- 16³ latent → 64³ occupancy grid → marching cubes mesh

Numerically verified against PyTorch: **0.9993 correlation** on single forward pass with identical weights and inputs. Image conditioning verified: sphere image produces spherical structure.

**Timing on M4 Max:** ~60s sampling + 0.2s decode.

### What's next

- [x] SparseStructureFlowModel (1.29B param DiT)
- [x] SparseStructureDecoder (73.7M param Conv U-Net)
- [x] Weight loading (640/640 + 74/74 params from TRELLIS.2-4B checkpoint)
- [x] Flow Euler sampler with CFG + guidance interval
- [x] 3D RoPE position embedding
- [x] Image conditioning via DINOv3
- [x] Smoke test: image → occupancy mesh
- [ ] `SparseTensor` representation
- [ ] `SLatFlowModel` (sparse token DiT for shape detail)
- [ ] Shape SLat decoder → full mesh extraction
- [ ] Texture SLat flow + decoder → PBR textures
- [ ] Full pipeline: image → textured GLB
- [ ] INT4/INT8 quantization
- [ ] `mx.compile` optimization

## Why MLX

TRELLIS.2 runs on Mac via [trellis-mac](https://github.com/shivampkumar/trellis-mac) (PyTorch MPS), but MPS has limitations:

- **Memory:** MPS SDPA materializes full N×N attention matrices. MLX's SDPA is real Flash Attention (O(N) memory, confirmed in source).
- **Quantization:** PyTorch MPS has limited INT4/INT8 support. MLX has built-in quantization that directly reduces memory bandwidth.
- **Compilation:** `mx.compile` can fuse transformer forward passes.
- **Speed:** No MPS dispatch overhead. Direct Metal compute.

## Quick start

```bash
git clone https://github.com/lyonsno/trellis2mlx.git
cd trellis2mlx
uv venv .venv --python python3.14
source .venv/bin/activate
uv pip install mlx numpy safetensors trimesh scikit-image pillow tqdm huggingface-hub

# For image conditioning (uses PyTorch for DINOv3 feature extraction, temporary):
uv pip install torch torchvision transformers

# HuggingFace auth (needed for gated DINOv3 weights):
hf auth login

# Generate:
PYTHONPATH=. python smoke.py --image your_image.png
open /tmp/trellis-mlx-smoke.glb
```

Weights download automatically on first run (~15GB for the flow model, ~900MB for the decoder).

## Tests

```bash
uv pip install pytest
PYTHONPATH=. pytest tests/ -v
```

19 tests covering LayerNorm32, MultiHeadRMSNorm, SDPA, variable-length attention, TimestepEmbedder, and SparseStructureFlowModel (shapes, parameter counts, determinism).

## Architecture

See [docs/architecture-map.md](docs/architecture-map.md) for the full TRELLIS.2-4B architecture reference.

The port follows the original model structure faithfully:
- `trellmlx/models/sparse_structure_flow.py` — SparseStructureFlowModel (30 DiT blocks)
- `trellmlx/models/sparse_structure_decoder.py` — SparseStructureDecoder (Conv3d U-Net)
- `trellmlx/modules/attention.py` — SDPA + MultiHeadRMSNorm
- `trellmlx/modules/rope.py` — 3D Rotary Position Embedding
- `trellmlx/modules/norm.py` — LayerNorm32 (fp32 accumulation)
- `trellmlx/weight_loader.py` — Checkpoint loading with key remapping
- `trellmlx/samplers.py` — Flow Euler sampler with CFG

## Credits

- [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) by Microsoft Research — the model
- [trellis-mac](https://github.com/shivampkumar/trellis-mac) by Shivam Kumar — proved Mac viability
- [trellis2-apple](https://github.com/pedronaugusto/trellis2-apple) by Pedro Naugusto — Metal modules
- [MLX](https://github.com/ml-explore/mlx) by Apple — the framework

## License

MIT (porting code). Upstream model weights are subject to their own licenses — see [trellis-mac](https://github.com/shivampkumar/trellis-mac#license) for details.
