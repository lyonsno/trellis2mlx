# trellis2mlx

MLX-native [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) inference for Apple Silicon.

Generate textured 3D meshes from single images, running entirely on Mac via [MLX](https://github.com/ml-explore/mlx) and Metal. No NVIDIA GPU required.

## Status

**Work in progress.** The core transformer modules are implemented. Weight conversion, full pipeline integration, and quantization support are in active development.

### What's here

- Weight converter (TRELLIS.2 safetensors -> MLX format)
- `LayerNorm32` with fp32 accumulation
- Scaled dot-product attention (dense + variable-length padded)
- `TimestepEmbedder` (sinusoidal -> MLP)
- `ModulatedTransformerCrossBlock` (adaLN-Zero DiT block with self-attn + cross-attn + FFN)
- Multi-head attention with QK RMSNorm
- Architecture map documenting the full TRELLIS.2-4B model structure

### What's next

- [x] `SparseStructureFlowModel` (dense 3D grid DiT) — 1.29B params, 19 tests passing
- [ ] Weight loading from TRELLIS.2-4B checkpoint
- [ ] `SparseTensor` representation
- [ ] `SLatFlowModel` (sparse token DiT)
- [ ] Sparse 3D convolution (gather-scatter)
- [ ] VAE decoders
- [ ] Diffusion sampling loop
- [ ] Full pipeline: image -> textured mesh
- [ ] INT8 quantization of DiT backbone
- [ ] `mx.compile` optimization

## Why

TRELLIS.2 currently runs on Mac via [trellis-mac](https://github.com/shivampkumar/trellis-mac) (PyTorch MPS). An MLX-native port enables:

- **Quantization**: MLX has built-in INT4/INT8 support. The diffusion backbone is memory-bandwidth-bound on Apple Silicon; quantization directly reduces bytes loaded per forward pass.
- **Compilation**: `mx.compile` can fuse the DiT forward pass in ways MPS cannot.
- **Native Metal**: Direct integration with Metal compute shaders for texture baking, BVH construction, and mesh processing.

## Requirements

- macOS on Apple Silicon (M1 or later)
- Python 3.11+
- 24GB+ unified memory recommended

## Credits

- [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) by Microsoft Research
- [trellis-mac](https://github.com/shivampkumar/trellis-mac) by Shivam Kumar (MPS port that proved Mac viability)
- [trellis2-apple](https://github.com/pedronaugusto/trellis2-apple) by Pedro Naugusto (Metal modules: mtldiffrast, mtlbvh, mtlgemm, cumesh)
- [MLX](https://github.com/ml-explore/mlx) by Apple

## License

MIT (porting code). Upstream model weights are subject to their own licenses; see [trellis-mac](https://github.com/shivampkumar/trellis-mac#license) for details.
