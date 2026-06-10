"""DINOv3 feature projection bridge for Pixal3D conditioning."""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..modules.proj_grid import ProjGrid, sample_features


def split_dinov3_features(
    features: mx.array,
    *,
    num_prefix_tokens: int = 5,
    patch_grid: Optional[int] = None,
) -> tuple[mx.array, mx.array]:
    """Split DINOv3 sequence features into global tokens and a patch map.

    Args:
        features: [B, prefix + patches, C] DINO feature sequence.
        num_prefix_tokens: CLS + register token count. DINOv3 ViT-L/16 uses 5.
        patch_grid: Optional spatial patch grid side length. If omitted, it is
            inferred from the patch token count and must be square.

    Returns:
        ``(global_features, patch_map)`` where ``patch_map`` is [B, H, W, C].
    """
    if features.ndim != 3:
        raise ValueError(f"Expected [B, N, C] DINO features, got {features.shape}")
    if features.shape[1] <= num_prefix_tokens:
        raise ValueError(
            f"Feature sequence has {features.shape[1]} tokens, not enough for {num_prefix_tokens} prefix tokens"
        )

    global_features = features[:, :num_prefix_tokens, :]
    patch_tokens = features[:, num_prefix_tokens:, :]
    patch_count = patch_tokens.shape[1]
    if patch_grid is None:
        patch_grid = int(math.sqrt(patch_count))
        if patch_grid * patch_grid != patch_count:
            raise ValueError(f"Patch token count {patch_count} is not a square")
    elif patch_grid * patch_grid != patch_count:
        raise ValueError(
            f"Patch grid {patch_grid}x{patch_grid} does not match {patch_count} patch tokens"
        )

    return global_features, patch_tokens.reshape(features.shape[0], patch_grid, patch_grid, features.shape[2])


def resize_bhwc_features(feature_map: mx.array, target_size: int | tuple[int, int]) -> mx.array:
    """Bilinearly resize a BHWC feature map using the same sampler as projection."""
    if isinstance(target_size, int):
        target_h = target_w = target_size
    else:
        target_h, target_w = target_size
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")

    B = feature_map.shape[0]
    xs = (np.arange(target_w, dtype=np.float32) + 0.5) / target_w * 2.0 - 1.0
    ys = (np.arange(target_h, dtype=np.float32) + 0.5) / target_h * 2.0 - 1.0
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    query = np.stack([grid_x, grid_y], axis=-1).reshape(1, target_h * target_w, 2)
    query_mx = mx.broadcast_to(mx.array(query), (B, target_h * target_w, 2))
    resized = sample_features(feature_map, query_mx, BHWC=True)
    return resized.reshape(B, target_h, target_w, feature_map.shape[-1])


class DINOv3ProjectionAdapter(nn.Module):
    """Convert DINOv3 sequence features into Pixal3D ``{global, proj}`` context."""

    def __init__(
        self,
        *,
        image_size: int = 512,
        patch_size: int = 16,
        grid_resolution: int = 16,
        num_prefix_tokens: int = 5,
        projection_mode: str = "native",
        hr_feature_size: int | tuple[int, int] | None = None,
    ):
        super().__init__()
        if projection_mode not in {"native", "bilinear_hr_concat"}:
            raise ValueError(f"unknown projection_mode: {projection_mode}")
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_grid = image_size // patch_size
        self.num_prefix_tokens = num_prefix_tokens
        self.projection_mode = projection_mode
        self.hr_feature_size = hr_feature_size if hr_feature_size is not None else image_size
        self.proj_grid = ProjGrid(grid_resolution=grid_resolution, image_resolution=image_size)

    def __call__(
        self,
        features: mx.array,
        *,
        camera_angle_x: mx.array,
        distance: mx.array,
        mesh_scale: mx.array,
        transform_matrix: Optional[mx.array] = None,
    ) -> dict[str, mx.array]:
        global_features, patch_map = split_dinov3_features(
            features,
            num_prefix_tokens=self.num_prefix_tokens,
            patch_grid=self.patch_grid,
        )
        projected_lr = self.proj_grid(
            patch_map,
            camera_angle_x=camera_angle_x,
            distance=distance,
            mesh_scale=mesh_scale,
            transform_matrix=transform_matrix,
            BHWC=True,
        )
        if self.projection_mode == "native":
            projected = projected_lr
        else:
            # Diagnostic bridge for Pixal3D SLat configs trained with LR/HR
            # projection concat. This is bilinear HR context, not NAF output.
            hr_patch_map = resize_bhwc_features(patch_map, self.hr_feature_size)
            projected_hr = self.proj_grid(
                hr_patch_map,
                camera_angle_x=camera_angle_x,
                distance=distance,
                mesh_scale=mesh_scale,
                transform_matrix=transform_matrix,
                BHWC=True,
            )
            projected = mx.concatenate([projected_lr, projected_hr], axis=-1)
        return {"global": global_features, "proj": projected}


# Upstream Pixal3D uses this name for the combined extractor. In MLX this class
# is the feature-to-context adapter; DINO extraction remains in ``dinov3.py``.
DinoV3ProjFeatureExtractor = DINOv3ProjectionAdapter
