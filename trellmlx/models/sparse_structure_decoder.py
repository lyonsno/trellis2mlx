"""Sparse Structure Decoder — dense 3D Conv U-Net that decodes the
flow model's latent into a single-channel occupancy logit volume.

Small model (~50M params). Operates entirely on dense 3D grids.
Thresholding the output at 0 gives the binary occupancy mask.

Architecture (from TRELLIS config, channels=[512, 128, 32]):
    input_layer: Conv3d(8 → 512)
    middle_block: ResBlock3d(512) × 2
    blocks: ResBlock3d(512) × 2, Upsample(512→128), ResBlock3d(128) × 2, Upsample(128→32), ResBlock3d(32) × 2
    out_layer: GroupNorm → SiLU → Conv3d(32 → 1)

Upsample uses pixel shuffle: Conv3d(in_ch → out_ch*8) then reshape
channels into 2×2×2 spatial expansion.
"""

import mlx.core as mx
import mlx.nn as nn


def pixel_shuffle_3d(x: mx.array, factor: int = 2) -> mx.array:
    """3D pixel shuffle: rearrange channels into spatial dimensions.

    Input: [B, D, H, W, C * factor³]
    Output: [B, D*factor, H*factor, W*factor, C]
    """
    B, D, H, W, C_total = x.shape
    C = C_total // (factor ** 3)

    x = x.reshape(B, D, H, W, C, factor, factor, factor)
    # Interleave: move factor dims next to their spatial dims
    x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)  # [B, D, f, H, f, W, f, C]
    x = x.reshape(B, D * factor, H * factor, W * factor, C)
    return x


class Conv3d(nn.Module):
    """3D convolution via mx.conv_general. Channels-last layout."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.padding = padding
        k = kernel_size
        scale = (2.0 / (in_channels * k * k * k)) ** 0.5
        self.weight = mx.random.normal((out_channels, k, k, k, in_channels)) * scale
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.conv_general(x, self.weight, padding=self.padding) + self.bias


class ChannelLayerNorm32(nn.Module):
    """Channel-wise LayerNorm with fp32 accumulation.

    Matches TRELLIS.2's ChannelLayerNorm32 — normalizes over the channel
    dimension (last dim in channels-last layout). The checkpoint's default
    norm_type is "layer", NOT "group".
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((channels,))
        self.bias = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        orig_dtype = x.dtype
        x = x.astype(mx.float32)
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - mean) * mx.rsqrt(var + self.eps)
        x = x.astype(orig_dtype)
        return x * self.weight + self.bias


class ResBlock3d(nn.Module):
    """3D residual block: Norm → SiLU → Conv → Norm → SiLU → Conv + skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = ChannelLayerNorm32(channels)
        self.conv1 = Conv3d(channels, channels)
        self.norm2 = ChannelLayerNorm32(channels)
        self.conv2 = Conv3d(channels, channels)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv1(nn.silu(self.norm1(x)))
        h = self.conv2(nn.silu(self.norm2(h)))
        return x + h


class UpsampleBlock3d(nn.Module):
    """Upsample via Conv3d → pixel_shuffle_3d (2x spatial expansion)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = Conv3d(in_channels, out_channels * 8)

    def __call__(self, x: mx.array) -> mx.array:
        return pixel_shuffle_3d(self.conv(x), 2)


class SparseStructureDecoder(nn.Module):
    """Decode the flow model's 8-channel latent into a 1-channel occupancy logit.

    Config (from TRELLIS):
        out_channels: 1
        latent_channels: 8
        num_res_blocks: 2
        num_res_blocks_middle: 2
        channels: [512, 128, 32]
    """

    def __init__(
        self,
        out_channels: int = 1,
        latent_channels: int = 8,
        num_res_blocks: int = 2,
        num_res_blocks_middle: int = 2,
        channels: list = None,
    ):
        super().__init__()
        if channels is None:
            channels = [512, 128, 32]

        self.input_layer = Conv3d(latent_channels, channels[0])

        self.middle_block = [ResBlock3d(channels[0]) for _ in range(num_res_blocks_middle)]

        self.blocks = []
        for i, ch in enumerate(channels):
            for _ in range(num_res_blocks):
                self.blocks.append(ResBlock3d(ch))
            if i < len(channels) - 1:
                self.blocks.append(UpsampleBlock3d(ch, channels[i + 1]))

        self.out_layer_0 = ChannelLayerNorm32(channels[-1])
        self.out_layer_2 = Conv3d(channels[-1], out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [B, latent_C, D, H, W] channels-first (matching PyTorch convention)

        Returns:
            [B, out_C, D', H', W'] occupancy logits, channels-first
        """
        # Convert to channels-last for MLX
        x = x.transpose(0, 2, 3, 4, 1)  # [B, D, H, W, C]

        x = self.input_layer(x)

        for block in self.middle_block:
            x = block(x)

        for block in self.blocks:
            x = block(x)

        x = nn.silu(self.out_layer_0(x))
        x = self.out_layer_2(x)

        # Back to channels-first
        return x.transpose(0, 4, 1, 2, 3)
