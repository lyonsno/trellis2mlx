"""DINOv3 ViT-L/16 feature extractor in MLX.

Produces image features for TRELLIS.2 conditioning without PyTorch.
Architecture: ViT-L/16 with 2D RoPE, register tokens, and layer scaling.

Output: [B, 1029, 1024] — 1024 patch tokens + 1 CLS + 4 register tokens.
"""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image


# ImageNet normalization
IMAGENET_MEAN = mx.array([0.485, 0.456, 0.406])
IMAGENET_STD = mx.array([0.229, 0.224, 0.225])


class DINOv3ViT(nn.Module):
    """DINOv3 ViT-L/16 for image feature extraction.

    Config (facebook/dinov3-vitl16-pretrain-lvd1689m):
        hidden_size: 1024
        num_heads: 16
        num_layers: 24
        patch_size: 16
        num_register_tokens: 4
        intermediate_size: 4096
        rope_theta: 10000.0
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 16,
        num_layers: int = 24,
        patch_size: int = 16,
        num_register_tokens: int = 4,
        intermediate_size: int = 4096,
        rope_theta: float = 100.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.num_prefix_tokens = 1 + num_register_tokens  # CLS + registers
        self.rope_theta = rope_theta

        # Embeddings
        self.cls_token = mx.zeros((1, 1, hidden_size))
        self.register_tokens = mx.zeros((1, num_register_tokens, hidden_size))
        self.patch_embed = nn.Conv2d(
            3, hidden_size, kernel_size=patch_size, stride=patch_size, bias=True,
        )

        # RoPE inverse frequencies
        self.inv_freq = 1.0 / (rope_theta ** np.arange(0, 1, 4 / self.head_dim, dtype=np.float32))

        # Transformer layers
        self.layers = [
            DINOv3Layer(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ]

    def __call__(self, pixel_values: mx.array) -> mx.array:
        """Extract features from pixel values.

        Args:
            pixel_values: [B, H, W, 3] normalized image (MLX channels-last)

        Returns:
            [B, N, D] features where N = num_patches + 1 + num_register_tokens
        """
        B = pixel_values.shape[0]

        # Patch embedding: [B, H, W, 3] → [B, H/P, W/P, D]
        x = self.patch_embed(pixel_values)
        # Flatten spatial: [B, H/P * W/P, D]
        num_h, num_w = x.shape[1], x.shape[2]
        x = x.reshape(B, -1, self.hidden_size)

        # Prepend CLS + register tokens
        cls = mx.broadcast_to(self.cls_token, (B, 1, self.hidden_size))
        regs = mx.broadcast_to(self.register_tokens, (B, self.num_register_tokens, self.hidden_size))
        x = mx.concatenate([cls, regs, x], axis=1)

        # Compute 2D RoPE for patch tokens
        cos, sin = self._compute_rope(num_h, num_w)

        # Transformer
        for layer in self.layers:
            x = layer(x, cos, sin, self.num_prefix_tokens)

        # Final LayerNorm
        x = mx.fast.layer_norm(x, None, None, 1e-6)

        return x

    def _compute_rope(self, num_h: int, num_w: int):
        """Compute 2D RoPE cos/sin for patch positions."""
        # Patch center coordinates in [-1, 1]
        coords_h = (np.arange(0.5, num_h) / num_h) * 2 - 1
        coords_w = (np.arange(0.5, num_w) / num_w) * 2 - 1
        grid_h, grid_w = np.meshgrid(coords_h, coords_w, indexing="ij")
        coords = np.stack([grid_h.ravel(), grid_w.ravel()], axis=-1)  # [N, 2]

        # Compute angles: coords[:, :, None] * inv_freq[None, None, :]
        # → [N, 2, head_dim/4] → flatten → [N, head_dim/2] → tile → [N, head_dim]
        angles = 2 * math.pi * coords[:, :, None] * self.inv_freq[None, None, :]
        angles = angles.reshape(coords.shape[0], -1)  # [N, head_dim/2]
        angles = np.tile(angles, 2)  # [N, head_dim]

        cos = mx.array(np.cos(angles), dtype=mx.float32)
        sin = mx.array(np.sin(angles), dtype=mx.float32)
        return cos, sin


class DINOv3Layer(nn.Module):
    """Single DINOv3 transformer layer."""

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attention = DINOv3Attention(hidden_size, num_heads)
        self.layer_scale1 = mx.zeros((hidden_size,))

        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = DINOv3MLP(hidden_size, intermediate_size)
        self.layer_scale2 = mx.zeros((hidden_size,))

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array, num_prefix: int) -> mx.array:
        # Attention with residual
        h = self.norm1(x)
        h = self.attention(h, cos, sin, num_prefix)
        h = h * self.layer_scale1
        x = x + h

        # MLP with residual
        h = self.norm2(x)
        h = self.mlp(h)
        h = h * self.layer_scale2
        x = x + h

        return x


class DINOv3Attention(nn.Module):
    """Multi-head attention with 2D RoPE (applied only to patch tokens)."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Separate Q, K, V projections (K has no bias in DINOv3)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array, num_prefix: int) -> mx.array:
        B, N, _ = x.shape

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE only to patch tokens (skip prefix = CLS + registers)
        q_prefix, q_patches = q[:, :, :num_prefix], q[:, :, num_prefix:]
        k_prefix, k_patches = k[:, :, :num_prefix], k[:, :, num_prefix:]

        q_patches = _apply_rope(q_patches, cos, sin)
        k_patches = _apply_rope(k_patches, cos, sin)

        q = mx.concatenate([q_prefix, q_patches], axis=2)
        k = mx.concatenate([k_prefix, k_patches], axis=2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0/scale)

        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(B, N, self.hidden_size)
        return self.o_proj(out)


class DINOv3MLP(nn.Module):
    """GELU MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu(self.up_proj(x)))


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply rotary position embedding (rotate_half style).

    Args:
        x: [B, H, N, D] where N = num patch tokens
        cos, sin: [N, D]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = mx.concatenate([-x2, x1], axis=-1)
    return x * cos + rotated * sin


def load_dinov3_weights(model: DINOv3ViT, weights_path: str):
    """Load HuggingFace DINOv3 weights into MLX model.

    Args:
        model: DINOv3ViT instance
        weights_path: Path to safetensors file or directory
    """
    from safetensors import safe_open

    if weights_path.endswith(".safetensors"):
        paths = [weights_path]
    else:
        import glob
        paths = sorted(glob.glob(f"{weights_path}/*.safetensors"))

    # Load all tensors
    tensors = {}
    for p in paths:
        with safe_open(p, framework="numpy") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Map weights
    new_weights = {}

    # Embeddings
    new_weights["cls_token"] = mx.array(tensors["embeddings.cls_token"])
    new_weights["register_tokens"] = mx.array(tensors["embeddings.register_tokens"])

    # Patch embed conv: HF stores as [out_c, in_c, kH, kW], MLX Conv2d wants [out_c, kH, kW, in_c]
    w = tensors["embeddings.patch_embeddings.weight"]  # [1024, 3, 16, 16]
    w = np.transpose(w, (0, 2, 3, 1))  # [1024, 16, 16, 3]
    new_weights["patch_embed.weight"] = mx.array(w)
    new_weights["patch_embed.bias"] = mx.array(tensors["embeddings.patch_embeddings.bias"])

    # Transformer layers
    for i in range(len(model.layers)):
        prefix = f"layer.{i}"

        # Norms
        new_weights[f"layers.{i}.norm1.weight"] = mx.array(tensors[f"{prefix}.norm1.weight"])
        new_weights[f"layers.{i}.norm1.bias"] = mx.array(tensors[f"{prefix}.norm1.bias"])
        new_weights[f"layers.{i}.norm2.weight"] = mx.array(tensors[f"{prefix}.norm2.weight"])
        new_weights[f"layers.{i}.norm2.bias"] = mx.array(tensors[f"{prefix}.norm2.bias"])

        # Attention
        new_weights[f"layers.{i}.attention.q_proj.weight"] = mx.array(tensors[f"{prefix}.attention.q_proj.weight"])
        new_weights[f"layers.{i}.attention.q_proj.bias"] = mx.array(tensors[f"{prefix}.attention.q_proj.bias"])
        new_weights[f"layers.{i}.attention.k_proj.weight"] = mx.array(tensors[f"{prefix}.attention.k_proj.weight"])
        new_weights[f"layers.{i}.attention.v_proj.weight"] = mx.array(tensors[f"{prefix}.attention.v_proj.weight"])
        new_weights[f"layers.{i}.attention.v_proj.bias"] = mx.array(tensors[f"{prefix}.attention.v_proj.bias"])
        new_weights[f"layers.{i}.attention.o_proj.weight"] = mx.array(tensors[f"{prefix}.attention.o_proj.weight"])
        new_weights[f"layers.{i}.attention.o_proj.bias"] = mx.array(tensors[f"{prefix}.attention.o_proj.bias"])

        # Layer scale
        new_weights[f"layers.{i}.layer_scale1"] = mx.array(tensors[f"{prefix}.layer_scale1.lambda1"])
        new_weights[f"layers.{i}.layer_scale2"] = mx.array(tensors[f"{prefix}.layer_scale2.lambda1"])

        # MLP
        new_weights[f"layers.{i}.mlp.up_proj.weight"] = mx.array(tensors[f"{prefix}.mlp.up_proj.weight"])
        new_weights[f"layers.{i}.mlp.up_proj.bias"] = mx.array(tensors[f"{prefix}.mlp.up_proj.bias"])
        new_weights[f"layers.{i}.mlp.down_proj.weight"] = mx.array(tensors[f"{prefix}.mlp.down_proj.weight"])
        new_weights[f"layers.{i}.mlp.down_proj.bias"] = mx.array(tensors[f"{prefix}.mlp.down_proj.bias"])

    model.load_weights(list(new_weights.items()))
    return len(new_weights)


def extract_features(image_path: str, image_size: int = 512, weights_path: str = None) -> mx.array:
    """Extract DINOv3 features from an image file.

    Args:
        image_path: Path to image file.
        image_size: Resize image to this size (default 512 for TRELLIS.2).
        weights_path: Path to HF model weights directory. Auto-detected if None.

    Returns:
        [1, 1029, 1024] feature tensor.
    """
    import os

    if weights_path is None:
        weights_path = os.path.expanduser(
            "~/.cache/huggingface/hub/models--facebook--dinov3-vitl16-pretrain-lvd1689m/"
            "snapshots/"
        )
        # Find the snapshot directory
        if os.path.isdir(weights_path):
            snapshots = [d for d in os.listdir(weights_path) if not d.startswith(".")]
            if snapshots:
                weights_path = os.path.join(weights_path, snapshots[0])

    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size), Image.LANCZOS)
    pixels = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3]

    # Normalize with ImageNet stats
    pixels = (pixels - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    pixel_values = mx.array(pixels[None])  # [1, H, W, 3]

    # Build model and load weights
    model = DINOv3ViT()
    num_loaded = load_dinov3_weights(model, weights_path)
    print(f"  DINOv3: loaded {num_loaded} weight arrays", flush=True)

    # Extract features
    features = model(pixel_values)
    mx.eval(features)

    return features
