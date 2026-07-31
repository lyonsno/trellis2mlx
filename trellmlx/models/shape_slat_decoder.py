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

import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..modules.sparse_conv import SparseConv3d, build_neighbor_map
from ..modules.norm import LayerNorm32
from ..decoder_turing_silu import silu as decoder_silu
from ..turing_fda import turing_fda_linear


DECODER_LINEAR_BACKEND_ENV = "TRELLIS2MLX_DECODER_LINEAR_BACKEND"
DECODER_LINEAR_BACKENDS = frozenset(("native", "turing_fda"))
AUTHENTICATED_DECODER_AFFINE_LAYERNORM_WIDTHS = frozenset((1024, 512, 256))
AUTHENTICATED_DECODER_NOAFFINE_LAYERNORM_TRANSITIONS = frozenset(
    ((1024, 512),)
)
AUTHENTICATED_SHAPE_DECODER_NOAFFINE_LAYERNORM_TRANSITIONS = frozenset(
    ((512, 256),)
)


def _decoder_linear(linear: nn.Linear, value: mx.array) -> mx.array:
    backend = os.environ.get(DECODER_LINEAR_BACKEND_ENV, "native").lower()
    if backend not in DECODER_LINEAR_BACKENDS:
        raise ValueError(
            "decoder linear backend must be one of "
            f"{sorted(DECODER_LINEAR_BACKENDS)}, got {backend!r}"
        )
    if backend == "turing_fda":
        if linear.bias is None:
            raise ValueError("Turing FDA decoder linear requires an affine bias")
        return turing_fda_linear(value, linear.weight.T, linear.bias)
    return linear(value)


def _decoder_silu(value: mx.array) -> mx.array:
    return decoder_silu(value)


def _layernorm_noaffine(x: mx.array, eps: float = 1e-5) -> mx.array:
    orig_dtype = x.dtype
    x = x.astype(mx.float32)
    return mx.fast.layer_norm(x, None, None, eps).astype(orig_dtype)


class SparseConvNeXtBlock3d(nn.Module):
    """ConvNeXt-style sparse block: Conv → Norm → MLP + skip."""

    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.conv = SparseConv3d(channels, channels, kernel_size=3)
        self.norm = LayerNorm32(
            channels,
            affine=True,
            decoder_layernorm=(
                channels in AUTHENTICATED_DECODER_AFFINE_LAYERNORM_WIDTHS
            ),
        )
        self.mlp_0 = nn.Linear(channels, int(channels * mlp_ratio))
        self.mlp_2 = nn.Linear(int(channels * mlp_ratio), channels)

    def __call__(self, feats: mx.array, neighbor_map: tuple) -> mx.array:
        h = self.conv(feats, neighbor_map)
        h = self.norm(h)
        h = _decoder_linear(
            self.mlp_2,
            _decoder_silu(_decoder_linear(self.mlp_0, h)),
        )
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
            return (
                mx.zeros((0, C), dtype=feats.dtype),
                mx.zeros((0, 4), dtype=mx.int32),
            )

        new_feats = mx.array(np.stack(new_feats_list))
        new_coords = mx.array(np.stack(new_coords_list).astype(np.int32))

        return new_feats, new_coords


class SparseResBlockC2S3d(nn.Module):
    """Upsample block: Norm→SiLU→Conv→Channel2Spatial→Norm→SiLU→Conv + skip."""

    def __init__(
        self,
        channels: int,
        out_channels: int,
        pred_subdiv: bool = True,
        decoder_role: str = "generic",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.pred_subdiv = pred_subdiv

        self.norm1 = LayerNorm32(
            channels,
            affine=True,
            decoder_layernorm=(
                channels in AUTHENTICATED_DECODER_AFFINE_LAYERNORM_WIDTHS
            ),
        )
        self.norm2 = LayerNorm32(
            out_channels,
            decoder_layernorm=(
                (channels, out_channels)
                in AUTHENTICATED_DECODER_NOAFFINE_LAYERNORM_TRANSITIONS
                or (
                    decoder_role == "shape"
                    and (channels, out_channels)
                    in AUTHENTICATED_SHAPE_DECODER_NOAFFINE_LAYERNORM_TRANSITIONS
                )
            ),
        )
        self.conv1 = SparseConv3d(channels, out_channels * 8, kernel_size=3)
        self.conv2 = SparseConv3d(out_channels, out_channels, kernel_size=3)
        if pred_subdiv:
            self.to_subdiv = nn.Linear(channels, 8)

    def __call__(self, feats: mx.array, coords: mx.array, neighbor_map: tuple,
                 subdiv: mx.array = None) -> tuple:
        """
        Args:
            subdiv: [N, 8] external subdivision mask (used when pred_subdiv=False).
                    If None and pred_subdiv=False, all children are kept.

        Returns:
            new_feats: [N', out_channels]
            new_coords: [N', 4]
            subdiv_logits: [N, 8] (or None if pred_subdiv=False)
        """
        # Get subdivision mask
        if self.pred_subdiv:
            subdiv_logits = _decoder_linear(self.to_subdiv, feats)  # [N, 8]
            subdiv_mask = subdiv_logits > 0
        else:
            subdiv_logits = subdiv
            if subdiv is not None:
                subdiv_mask = subdiv > 0 if subdiv.dtype != mx.bool_ else subdiv
            else:
                subdiv_mask = mx.ones((feats.shape[0], 8), dtype=mx.bool_)

        # Pre-upsample: norm → silu → conv
        h = _decoder_silu(self.norm1(feats))
        h = self.conv1(h, neighbor_map)  # [N, out_channels * 8]

        # Channel to spatial upsample (also upsample skip path)
        new_h, new_coords = SparseChannel2Spatial.upsample(h, coords, subdiv_mask)

        if new_h.shape[0] == 0:
            return new_h, new_coords, subdiv_logits

        # Skip connection: upsample original features through Channel2Spatial
        # Each child k gets channel slice [k*C//8 : (k+1)*C//8] from the parent,
        # then repeat-interleaved to match out_channels.
        skip_h, _ = SparseChannel2Spatial.upsample(feats, coords, subdiv_mask)
        # skip_h: [N', channels//8] — each child has its own 1/8th slice
        channels_per_child = self.channels // 8
        repeat_factor = self.out_channels // channels_per_child
        if repeat_factor > 1:
            skip_feats = mx.repeat(skip_h, repeat_factor, axis=1)
        else:
            skip_feats = skip_h
        # Ensure exact match
        if skip_feats.shape[1] != self.out_channels:
            skip_feats = skip_feats[:, :self.out_channels]

        # Post-upsample: norm → silu → conv
        new_nmap = build_neighbor_map(new_coords)
        new_h = _decoder_silu(self.norm2(new_h))
        new_h = self.conv2(new_h, new_nmap)

        # Add skip
        new_feats = new_h + skip_feats

        return new_feats, new_coords, subdiv_logits


class SLatDecoder(nn.Module):
    """Decode structured latents via sparse UNet.

    Used for both shape (out_channels=7, pred_subdiv=True) and
    texture (out_channels=6, pred_subdiv=False) decoding.

    Config (TRELLIS.2-4B shape): model_channels=[1024,512,256,128,64],
        latent_channels=32, num_blocks=[4,16,8,4,0], out_channels=7
    Config (TRELLIS.2-4B texture): same architecture, out_channels=6,
        pred_subdiv=False
    """

    def __init__(
        self,
        out_channels: int = 7,
        latent_channels: int = 32,
        model_channels: list = None,
        num_blocks: list = None,
        pred_subdiv: bool = True,
        use_fp16: bool = True,
    ):
        super().__init__()
        if model_channels is None:
            model_channels = [1024, 512, 256, 128, 64]
        if num_blocks is None:
            num_blocks = [4, 16, 8, 4, 0]

        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.pred_subdiv = pred_subdiv
        self.use_fp16 = use_fp16
        if out_channels == 7 and pred_subdiv:
            self.decoder_role = "shape"
        elif out_channels == 6 and not pred_subdiv:
            self.decoder_role = "texture"
        else:
            self.decoder_role = "generic"

        self.from_latent = nn.Linear(latent_channels, model_channels[0])
        self.output_layer = nn.Linear(model_channels[-1], out_channels)

        self.blocks = []
        for i, (ch, n_blocks) in enumerate(zip(model_channels, num_blocks)):
            level = []
            for _ in range(n_blocks):
                level.append(SparseConvNeXtBlock3d(ch))
            if i < len(model_channels) - 1:
                level.append(
                    SparseResBlockC2S3d(
                        ch,
                        model_channels[i + 1],
                        pred_subdiv=pred_subdiv,
                        decoder_role=self.decoder_role,
                    )
                )
            self.blocks.append(level)
        if use_fp16:
            for level in self.blocks:
                for block in level:
                    block.set_dtype(mx.float16)

    def _forward_levels(self, feats: mx.array, coords: mx.array,
                         stop_after: int = None,
                         guide_subs: list = None,
                         return_subs: bool = False) -> tuple:
        """Run the decoder through levels, optionally stopping early.

        Args:
            feats: [N, C] features (already projected from latent)
            coords: [N, 4] as (batch, z, y, x)
            stop_after: if set, return coords after this many upsample levels
            guide_subs: list of subdivision masks from shape decode (for texture decoder)
            return_subs: if True, collect and return subdivision logits

        Returns:
            (feats, coords) or (feats, coords, subs) if return_subs
        """
        upsample_count = 0
        subs = [] if return_subs else None
        for level_idx, level_blocks in enumerate(self.blocks):
            if stop_after is not None and upsample_count >= stop_after:
                if return_subs:
                    return feats, coords, subs
                return feats, coords
            if not level_blocks:
                continue

            nmap = build_neighbor_map(coords)

            for block in level_blocks:
                if isinstance(block, SparseConvNeXtBlock3d):
                    feats = block(feats, nmap)
                    mx.eval(feats)
                elif isinstance(block, SparseResBlockC2S3d):
                    # Pass guide subdivision if available (texture decoder)
                    guide = guide_subs[upsample_count] if guide_subs is not None else None
                    feats, coords, sub_logits = block(feats, coords, nmap, subdiv=guide)
                    if return_subs and sub_logits is not None:
                        subs.append(sub_logits)
                    upsample_count += 1
                    print(f"  Level {level_idx} → {level_idx+1}: "
                          f"{feats.shape[0]} voxels",
                          flush=True)

        if return_subs:
            return feats, coords, subs
        return feats, coords

    def upsample(self, feats: mx.array, coords: mx.array,
                 upsample_times: int = 4) -> mx.array:
        """Run the decoder to get upsampled coordinates (structure only).

        This is the first pass of the two-pass architecture: run the full
        decoder with subdivision predictions to get the high-resolution
        coordinate structure, then discard the features.

        Args:
            feats: [N, latent_channels] shape latent features
            coords: [N, 4] as (batch, z, y, x)
            upsample_times: number of upsample levels to run through

        Returns:
            hr_coords: [N', 4] upsampled coordinates
        """
        feats = self.from_latent(feats)
        if self.use_fp16:
            feats = feats.astype(mx.float16)
        _, hr_coords = self._forward_levels(feats, coords,
                                            stop_after=upsample_times)
        return hr_coords

    def __call__(self, feats: mx.array, coords: mx.array,
                 guide_subs: list = None, return_subs: bool = False) -> tuple:
        """
        Args:
            feats: [N, latent_channels] shape latent features
            coords: [N, 4] as (batch, z, y, x)
            guide_subs: subdivision masks from shape decoder (for texture decoder)
            return_subs: if True, return subdivision logits for use as guide_subs

        Returns:
            (out_feats, out_coords) or (out_feats, out_coords, subs) if return_subs
        """
        input_dtype = feats.dtype
        feats = self.from_latent(feats)
        if self.use_fp16:
            feats = feats.astype(mx.float16)
        result = self._forward_levels(feats, coords, guide_subs=guide_subs,
                                       return_subs=return_subs)
        if return_subs:
            feats, coords, subs = result
        else:
            feats, coords = result

        feats = _layernorm_noaffine(feats.astype(input_dtype))
        out = self.output_layer(feats)

        if return_subs:
            return out, coords, subs
        return out, coords


# Backwards-compatible alias
ShapeSLatDecoder = SLatDecoder
