# TRELLIS.2 Architecture Map for MLX Port

## Pipeline Overview

```
Image -> DINOv3 (stays PyTorch) -> image_cond [B, 1024]
  |
  v
SparseStructureFlowModel (DiT over dense 3D grid)
  - Input: [B, C, R, R, R] dense noise
  - 12 DDPM steps, adaLN-Zero conditioning
  - Output: boolean occupancy -> coords [N, 4]
  |
  v
SLatFlowModel (DiT over sparse tokens) x2 (cascade: 512 -> 1024)
  - Input: SparseTensor [T, 384] noise at occupied coords
  - 12 flow steps, adaLN-Zero + cross-attn to image cond
  - Output: shape latent SparseTensor
  |
  v
FlexiDualGridVaeDecoder (sparse UNet)
  - Upsamples sparse latent through residual blocks
  - flexible_dual_grid_to_mesh() -> triangle mesh
  |
  v
Texture SLat Flow (same architecture as shape, different weights)
  - Generates PBR attributes per voxel
  |
  v
Metal BVH texture bake (stays Metal, not MLX)
  -> GLB with PBR textures
```

## Core Modules to Port

### 1. SparseTensor
- `.feats`: [N_active, C] features
- `.coords`: [N_active, 4] as [batch_idx, z, y, x]
- `.layout`: List[slice] mapping rows to batch elements
- Cached: seqlen, cum_seqlen per batch element

### 2. ModulatedSparseTransformerCrossBlock (the main block)
```
h = norm1(x.feats)
h = h * (1 + scale_msa) + shift_msa      # adaLN-Zero
h = self_attn(h)                          # sparse variable-length attention
h = h * gate_msa
x = x + h
h = norm2(x.feats)
h = cross_attn(h, image_cond)             # cross-attn to dense image features
x = x + h
h = norm3(x.feats)
h = h * (1 + scale_mlp) + shift_mlp
h = mlp(h)                                # 4x expansion FFN
h = h * gate_mlp
x = x + h
```

### 3. Attention
- Self-attn: fused QKV [T, 3, H, head_dim] -> variable-length SDPA
- Cross-attn: Q from sparse [T, H, D], KV from dense [B, L, 2, H, D]
- Variable-length via cum_seqlen (like flash_attn_varlen)
- Current MPS path: pad to max seqlen, SDPA, unpad

### 4. Sparse Convolution (submanifold 3x3x3)
- Weight: [out_C, 3, 3, 3, in_C]
- Gather neighbors by coord offset -> matmul -> scatter-add
- conv_none.py reference implementation (pure Python, slow)

### 5. VAE Decoder
- Multi-scale upsample with SparseResBlock3d
- Each block: LayerNorm -> SiLU -> SparseConv3d -> Upsample -> Conv2
- Final: flexible_dual_grid_to_mesh()

## Weight Shapes (model_channels=384, num_heads=8)

| Layer | Shape |
|-------|-------|
| to_qkv (self-attn) | [1152, 384] + [1152] |
| to_out | [384, 384] + [384] |
| to_q (cross-attn) | [384, 384] + [384] |
| to_kv (cross-attn) | [768, cond_C] + [768] |
| FFN linear1 | [1536, 384] + [1536] |
| FFN linear2 | [384, 1536] + [384] |
| adaLN modulation | [2304, 384] + [2304] |
| timestep embed | [384, 256] + [384] -> [384, 384] + [384] |
| sparse conv 3x3x3 | [out_C, 3, 3, 3, in_C] |

## Implementation Plan

### Phase 1: Weight converter + Dense DiT
- Convert safetensors -> MLX format
- Port SparseStructureFlowModel (operates on dense grid, simpler)
- Port timestep embedding, adaLN-Zero, attention, FFN in mx.nn
- Diffusion sampling loop

### Phase 2: Sparse DiT
- SparseTensor in MLX (feats as mx.array, coords as mx.array)
- Variable-length attention (pad+mask or segmented)
- SLatFlowModel

### Phase 3: VAE + Integration
- Sparse UNet decoder
- Sparse convolution (gather-scatter in MLX)
- flexible_dual_grid_to_mesh (keep as numpy/CPU)
- Connect to Metal bake modules

### Phase 4: Optimization
- mx.compile on DiT forward pass
- INT8/INT4 quantization of DiT backbone
- Benchmark vs MPS path
