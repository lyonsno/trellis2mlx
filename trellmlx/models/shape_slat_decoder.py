"""Shape SLat Decoder — sparse UNet that decodes shape latents into mesh geometry.

Takes shape latents [N, 32] at sparse voxel coordinates and produces
per-voxel outputs [N', 7] at higher resolution:
  [0:3] vertex offsets (sigmoid-scaled)
  [3:6] edge intersection flags
  [6:7] quad split weight

These feed into flexible_dual_grid_to_mesh to produce the final triangle mesh.

Architecture (from TRELLIS.2-4B, shape_dec_next_dc_f16c32_fp16):
  from_latent: Linear(32 → 1024)
  Level 0: 1024 channels, 4 SparseConvNeXtBlocks, upsample to 512
  Level 1: 512 channels, 16 SparseConvNeXtBlocks, upsample to 256
  Level 2: 256 channels, 8 SparseConvNeXtBlocks, upsample to 128
  Level 3: 128 channels, 4 SparseConvNeXtBlocks, upsample to 64
  Level 4: 64 channels, 0 blocks (output level)
  output_layer: Linear(64 → 7)
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..modules.sparse_conv import SparseConv3d, build_neighbor_map
from ..modules.norm import LayerNorm32


def _layernorm_noaffine(x: mx.array, eps: float = 1e-6) -> mx.array:
    orig_dtype = x.dtype
    x = x.astype(mx.float32)
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    x = (x - mean) * mx.rsqrt(var + eps)
    return x.astype(orig_dtype)


class SparseConvNeXtBlock3d(nn.Module):
    """ConvNeXt-style sparse block: Conv → Norm → MLP + skip."""

    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.conv = SparseConv3d(channels, channels, kernel_size=3)
        self.norm = LayerNorm32(channels, affine=True)
        self.mlp_0 = nn.Linear(channels, int(channels * mlp_ratio))
        self.mlp_2 = nn.Linear(int(channels * mlp_ratio), channels)

    def __call__(self, feats: mx.array, neighbor_map: tuple) -> mx.array:
        h = self.conv(feats, neighbor_map)
        h = self.norm(h)
        h = self.mlp_2(nn.silu(self.mlp_0(h)))
        return h + feats


class SparseChannel2Spatial:
    """Upsample sparse voxels by converting channels to spatial children.

    Each voxel with C*8 features spawns up to 8 child voxels with C features.
    The subdivision mask determines which children are actually created.
    """

    @staticmethod
    def upsample(feats: mx.array, coords: mx.array, subdiv_mask: mx.array) -> tuple:
        """
        Args:
            feats: [N, C*8] features
            coords: [N, 4] as (batch, z, y, x)
            subdiv_mask: [N, 8] boolean subdivision mask

        Returns:
            new_feats: [N', C] upsampled features
            new_coords: [N', 4] upsampled coordinates
        """
        N = feats.shape[0]
        C = feats.shape[1] // 8

        # Convert to numpy for coordinate manipulation
        coords_np = np.array(coords)
        subdiv_np = np.array(subdiv_mask)
        feats_np = np.array(feats)

        new_feats_list = []
        new_coords_list = []

        # Reshape features: [N, C*8] → [N, 8, C]
        feats_reshaped = feats_np.reshape(N, 8, C)

        for i in range(N):
            for child in range(8):
                if subdiv_np[i, child]:
                    # Compute child coordinate offset (little-endian: z=bit0, y=bit1, x=bit2)
                    dz = child % 2
                    dy = (child // 2) % 2
                    dx = child // 4

                    new_coord = coords_np[i].copy()
                    new_coord[1] = new_coord[1] * 2 + dz
                    new_coord[2] = new_coord[2] * 2 + dy
                    new_coord[3] = new_coord[3] * 2 + dx

                    new_coords_list.append(new_coord)
                    new_feats_list.append(feats_reshaped[i, child])

        if len(new_feats_list) == 0:
            return mx.zeros((0, C)), mx.zeros((0, 4), dtype=mx.int32)

        new_feats = mx.array(np.stack(new_feats_list))
        new_coords = mx.array(np.stack(new_coords_list).astype(np.int32))

        return new_feats, new_coords


class SparseResBlockC2S3d(nn.Module):
    """Upsample block: Norm→SiLU→Conv→Channel2Spatial→Norm→SiLU→Conv + skip."""

    def __init__(self, channels: int, out_channels: int):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels

        self.norm1 = LayerNorm32(channels, affine=True)
        self.norm2 = LayerNorm32(out_channels)  # no affine, like reference
        self.conv1 = SparseConv3d(channels, out_channels * 8, kernel_size=3)
        self.conv2 = SparseConv3d(out_channels, out_channels, kernel_size=3)
        self.to_subdiv = nn.Linear(channels, 8)

    def __call__(self, feats: mx.array, coords: mx.array, neighbor_map: tuple) -> tuple:
        """
        Returns:
            new_feats: [N', out_channels]
            new_coords: [N', 4]
            subdiv_logits: [N, 8]
        """
        # Predict subdivision
        subdiv_logits = self.to_subdiv(feats)  # [N, 8]
        subdiv_mask = subdiv_logits > 0  # binary mask

        # Pre-upsample: norm → silu → conv
        h = nn.silu(self.norm1(feats))
        h = self.conv1(h, neighbor_map)  # [N, out_channels * 8]

        # Channel to spatial upsample (also upsample skip path)
        new_h, new_coords = SparseChannel2Spatial.upsample(h, coords, subdiv_mask)

        if new_h.shape[0] == 0:
            return new_h, new_coords, subdiv_logits

        # Skip connection: upsample input features via repeat-interleave
        # Each parent voxel's features repeat for each child, filtered by subdiv mask
        feats_np = np.array(feats)
        subdiv_np = np.array(subdiv_mask)
        skip_list = []
        for i in range(len(feats_np)):
            for child in range(8):
                if subdiv_np[i, child]:
                    skip_list.append(feats_np[i])
        skip_feats = mx.array(np.stack(skip_list)) if skip_list else mx.zeros((0, self.channels))
        # Repeat-interleave channels to match out_channels
        repeat_factor = self.out_channels // (self.channels // 8)
        if repeat_factor > 1 and skip_feats.shape[1] != self.out_channels:
            skip_feats = mx.repeat(skip_feats, repeat_factor, axis=1)
        # Truncate or pad to match out_channels
        if skip_feats.shape[1] > self.out_channels:
            skip_feats = skip_feats[:, :self.out_channels]
        elif skip_feats.shape[1] < self.out_channels:
            pad = mx.zeros((skip_feats.shape[0], self.out_channels - skip_feats.shape[1]))
            skip_feats = mx.concatenate([skip_feats, pad], axis=1)

        # Post-upsample: norm → silu → conv
        new_nmap = build_neighbor_map(new_coords)
        new_h = nn.silu(self.norm2(new_h))
        new_h = self.conv2(new_h, new_nmap)

        # Add skip
        new_feats = new_h + skip_feats

        return new_feats, new_coords, subdiv_logits


class ShapeSLatDecoder(nn.Module):
    """Decode shape latents into mesh geometry via sparse UNet.

    Config (TRELLIS.2-4B):
        model_channels: [1024, 512, 256, 128, 64]
        latent_channels: 32
        num_blocks: [4, 16, 8, 4, 0]
        out_channels: 7
    """

    def __init__(
        self,
        out_channels: int = 7,
        latent_channels: int = 32,
        model_channels: list = None,
        num_blocks: list = None,
    ):
        super().__init__()
        if model_channels is None:
            model_channels = [1024, 512, 256, 128, 64]
        if num_blocks is None:
            num_blocks = [4, 16, 8, 4, 0]

        self.model_channels = model_channels
        self.num_blocks = num_blocks

        self.from_latent = nn.Linear(latent_channels, model_channels[0])
        self.output_layer = nn.Linear(model_channels[-1], out_channels)

        # Build blocks per level
        # Build nested block structure matching checkpoint naming:
        # blocks.0.0 ... blocks.0.{n-1} = ConvNeXt blocks at level 0
        # blocks.0.{n} = upsample block (to_subdiv lives here)
        # blocks.1.0 ... etc.
        self.blocks = []
        for i, (ch, n_blocks) in enumerate(zip(model_channels, num_blocks)):
            level = []
            for _ in range(n_blocks):
                level.append(SparseConvNeXtBlock3d(ch))
            if i < len(model_channels) - 1:
                level.append(SparseResBlockC2S3d(ch, model_channels[i + 1]))
            self.blocks.append(level)

    def __call__(self, feats: mx.array, coords: mx.array) -> tuple:
        """
        Args:
            feats: [N, latent_channels] shape latent features
            coords: [N, 4] as (batch, z, y, x)

        Returns:
            out_feats: [N', 7] per-voxel outputs at final resolution
            out_coords: [N', 4] coordinates at final resolution
        """
        # Project latent to decoder channels
        feats = self.from_latent(feats)

        for level_idx, level_blocks in enumerate(self.blocks):
            if not level_blocks:
                continue

            # Build neighbor map for this level
            nmap = build_neighbor_map(coords)

            # Run blocks at this level
            for block in level_blocks:
                if isinstance(block, SparseConvNeXtBlock3d):
                    feats = block(feats, nmap)
                    mx.eval(feats)  # periodic eval for bus-friendliness
                elif isinstance(block, SparseResBlockC2S3d):
                    feats, coords, _ = block(feats, coords, nmap)
                    print(f"  Level {level_idx} → {level_idx+1}: "
                          f"{feats.shape[0]} voxels",
                          flush=True)

        # LayerNorm before output (matches reference: F.layer_norm(h.feats, h.feats.shape[-1:]))
        feats = _layernorm_noaffine(feats)

        # Output projection
        out = self.output_layer(feats)

        return out, coords
