"""View-aligned projection utilities for the Pixal3D graft."""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def project_points_to_image_batch(
    points_3d: mx.array,
    transform_matrix: mx.array,
    camera_angle_x: mx.array,
    resolution: int = 518,
) -> tuple[mx.array, mx.array, mx.array]:
    """Project 3D points into image pixel coordinates.

    Matches the Pixal3D PyTorch projection contract closely enough for
    inference: Blender-style cameras face negative Z, and returned pixels are
    in the input image coordinate system.
    """
    if points_3d.ndim == 2:
        points_3d = mx.broadcast_to(points_3d[None, :, :], (transform_matrix.shape[0], points_3d.shape[0], 3))

    B, N, _ = points_3d.shape
    ones = mx.ones((B, N, 1), dtype=points_3d.dtype)
    points_h = mx.concatenate([points_3d, ones], axis=-1)

    with mx.stream(mx.cpu):
        world_to_camera = mx.linalg.inv(transform_matrix.astype(mx.float32)).astype(points_3d.dtype)
    points_camera = mx.matmul(points_h, mx.swapaxes(world_to_camera, -1, -2))[..., :3]

    x_cam = points_camera[..., 0]
    y_cam = points_camera[..., 1]
    z_cam = points_camera[..., 2]
    depth = -z_cam

    sensor_width = 32.0
    focal_length = 16.0 / mx.tan(camera_angle_x / 2.0)
    focal_pixels = (focal_length * resolution / sensor_width)[:, None]

    x_ndc = focal_pixels * x_cam / (-z_cam + 1e-8)
    y_ndc = focal_pixels * y_cam / (-z_cam + 1e-8)

    x_pixel = x_ndc + resolution / 2.0
    y_pixel = -y_ndc + resolution / 2.0

    valid = (
        (x_pixel >= 0)
        & (x_pixel < resolution)
        & (y_pixel >= 0)
        & (y_pixel < resolution)
        & (depth > 0)
    )
    return mx.stack([x_pixel, y_pixel], axis=-1), depth, valid


def _gather_bhwc(feature_map: mx.array, y: mx.array, x: mx.array) -> mx.array:
    B, H, W, C = feature_map.shape
    batch_offsets = (mx.arange(B, dtype=mx.int32) * H * W)[:, None]
    flat_index = (batch_offsets + y.astype(mx.int32) * W + x.astype(mx.int32)).reshape(-1)
    gathered = mx.take(feature_map.reshape(B * H * W, C), flat_index, axis=0)
    return gathered.reshape(B, y.shape[1], C)


def sample_features(feature_map: mx.array, queries_ndc: mx.array, BHWC: bool = True) -> mx.array:
    """Bilinearly sample image features at NDC coordinates.

    Args:
        feature_map: [B, H, W, C] if ``BHWC`` else [B, C, H, W].
        queries_ndc: [B, K, 2] coordinates in PyTorch grid_sample convention.
        BHWC: Whether ``feature_map`` is already channels-last.

    Returns:
        [B, K, C] sampled features with align_corners=False and border padding.
    """
    if not BHWC:
        feature_map = feature_map.transpose(0, 2, 3, 1)

    B, H, W, _ = feature_map.shape
    if queries_ndc.shape[0] != B:
        raise ValueError(f"Batch size mismatch: feature_map B={B}, query B={queries_ndc.shape[0]}")

    x = ((queries_ndc[..., 0] + 1.0) * W - 1.0) / 2.0
    y = ((queries_ndc[..., 1] + 1.0) * H - 1.0) / 2.0
    x = mx.clip(x, 0.0, W - 1.0)
    y = mx.clip(y, 0.0, H - 1.0)

    x0 = mx.floor(x)
    y0 = mx.floor(y)
    x1 = mx.clip(x0 + 1.0, 0.0, W - 1.0)
    y1 = mx.clip(y0 + 1.0, 0.0, H - 1.0)

    x0_i = x0.astype(mx.int32)
    y0_i = y0.astype(mx.int32)
    x1_i = x1.astype(mx.int32)
    y1_i = y1.astype(mx.int32)

    top_left = _gather_bhwc(feature_map, y0_i, x0_i)
    top_right = _gather_bhwc(feature_map, y0_i, x1_i)
    bottom_left = _gather_bhwc(feature_map, y1_i, x0_i)
    bottom_right = _gather_bhwc(feature_map, y1_i, x1_i)

    dx = x - x0
    dy = y - y0
    wa = ((1.0 - dx) * (1.0 - dy))[..., None]
    wb = (dx * (1.0 - dy))[..., None]
    wc = ((1.0 - dx) * dy)[..., None]
    wd = (dx * dy)[..., None]
    return top_left * wa + top_right * wb + bottom_left * wc + bottom_right * wd


class ProjGrid(nn.Module):
    """Project a 3D grid into an image feature map and sample local features."""

    def __init__(self, grid_resolution: int = 16, image_resolution: int = 518):
        super().__init__()
        self.grid_resolution = grid_resolution
        self.image_resolution = image_resolution

        one_dim = np.linspace(-1.0, 1.0, grid_resolution, dtype=np.float32)
        x, y, z = np.meshgrid(one_dim, one_dim, one_dim, indexing="ij")
        grid_points = np.stack((x, y, z), axis=-1)
        rotation = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.grid_points = mx.array((grid_points @ rotation.T).reshape(-1, 3))
        self.front_view_transform_matrix = mx.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, -2.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=mx.float32,
        )

    def __call__(
        self,
        features_map: mx.array,
        camera_angle_x: mx.array,
        distance: mx.array,
        mesh_scale: mx.array,
        transform_matrix: Optional[mx.array] = None,
        BHWC: bool = True,
    ) -> mx.array:
        B = features_map.shape[0]
        grid_points = mx.broadcast_to(
            self.grid_points[None, :, :], (B, self.grid_points.shape[0], 3)
        )
        grid_points = grid_points / mesh_scale[:, None, None] / 2.0

        if transform_matrix is None:
            transform_matrix = mx.broadcast_to(self.front_view_transform_matrix[None, :, :], (B, 4, 4))
            distance_delta = mx.zeros((B, 4, 4), dtype=transform_matrix.dtype)
            distance_delta = distance_delta.at[:, 1, 3].add(2.0 - distance)
            transform_matrix = transform_matrix + distance_delta

        image_points, _, _ = project_points_to_image_batch(
            grid_points, transform_matrix, camera_angle_x, self.image_resolution
        )
        image_points_norm = (image_points + 0.5) / self.image_resolution * 2.0 - 1.0
        return sample_features(features_map, image_points_norm, BHWC=BHWC)
