"""Tests for the Pixal3D view-aligned projection graft."""

import numpy as np
import mlx.core as mx


def test_project_origin_to_front_view_image_center():
    from trellmlx.modules.proj_grid import project_points_to_image_batch

    points = mx.array([[0.0, 0.0, 0.0]])
    transform = mx.array(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, -2.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ]
    )
    camera_angle_x = mx.array([np.pi / 2], dtype=mx.float32)

    points_2d, depth, valid = project_points_to_image_batch(
        points, transform, camera_angle_x, resolution=100
    )
    mx.eval(points_2d, depth, valid)

    assert points_2d.shape == (1, 1, 2)
    assert np.allclose(np.array(points_2d[0, 0]), [50.0, 50.0], atol=1e-5)
    assert np.allclose(np.array(depth[0, 0]), 2.0, atol=1e-5)
    assert bool(valid[0, 0].item())


def test_sample_features_uses_align_corners_false_bilinear_center():
    from trellmlx.modules.proj_grid import sample_features

    feature_map = mx.array(
        [[[[0.0], [10.0]], [[20.0], [30.0]]]],
        dtype=mx.float32,
    )
    queries_ndc = mx.array([[[0.0, 0.0]]], dtype=mx.float32)

    sampled = sample_features(feature_map, queries_ndc, BHWC=True)
    mx.eval(sampled)

    assert sampled.shape == (1, 1, 1)
    assert np.allclose(np.array(sampled[0, 0, 0]), 15.0, atol=1e-5)


def test_projgrid_returns_view_aligned_features_for_every_voxel():
    from trellmlx.modules.proj_grid import ProjGrid

    features = mx.ones((1, 4, 4, 3), dtype=mx.float32)
    proj = ProjGrid(grid_resolution=2, image_resolution=100)

    out = proj(
        features,
        camera_angle_x=mx.array([np.pi / 2], dtype=mx.float32),
        distance=mx.array([2.0], dtype=mx.float32),
        mesh_scale=mx.array([1.0], dtype=mx.float32),
    )
    mx.eval(out)

    assert out.shape == (1, 8, 3)
    assert np.allclose(np.array(out), 1.0, atol=1e-5)


def test_project_attention_adds_projected_local_features_to_global_attention():
    from trellmlx.modules.proj_attention import ProjectAttention

    class ConstantCrossAttention:
        def __call__(self, x, context):
            return mx.ones_like(x) * mx.mean(context)

    module = ProjectAttention(ConstantCrossAttention(), channels=2, proj_in_channels=2)
    module.proj_linear.weight = mx.eye(2)
    module.proj_linear.bias = mx.zeros((2,))

    x = mx.zeros((3, 2), dtype=mx.float32)
    context = {
        "global": mx.ones((1, 4, 2), dtype=mx.float32) * 2.0,
        "proj": mx.array([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]], dtype=mx.float32),
    }

    out = module(x, context)
    mx.eval(out)

    expected = np.array([[3.0, 12.0], [4.0, 22.0], [5.0, 32.0]], dtype=np.float32)
    assert np.allclose(np.array(out), expected, atol=1e-5)


def test_pixal3d_sparse_structure_flow_accepts_global_and_projected_context():
    from trellmlx.models.pixal3d_flow import Pixal3DSparseStructureFlowModel

    model = Pixal3DSparseStructureFlowModel(
        in_channels=8,
        out_channels=8,
        model_channels=64,
        num_heads=4,
        num_blocks=1,
        mlp_hidden=128,
        context_channels=32,
        proj_in_channels=16,
        resolution=2,
    )
    x = mx.random.normal((1, 8, 2, 2, 2))
    t = mx.array([500.0])
    cond = {
        "global": mx.random.normal((1, 5, 32)),
        "proj": mx.random.normal((1, 8, 16)),
    }

    out = model(x, t, cond)
    assert out.shape == (1, 8, 2, 2, 2)


def test_dinov3_projection_adapter_builds_pixal3d_context_dict():
    from trellmlx.models.dinov3_proj import DINOv3ProjectionAdapter

    prefix = mx.zeros((1, 5, 3), dtype=mx.float32)
    patches = mx.ones((1, 4, 3), dtype=mx.float32)
    features = mx.concatenate([prefix, patches], axis=1)

    adapter = DINOv3ProjectionAdapter(
        image_size=32,
        patch_size=16,
        grid_resolution=2,
        num_prefix_tokens=5,
    )
    context = adapter(
        features,
        camera_angle_x=mx.array([np.pi / 2], dtype=mx.float32),
        distance=mx.array([2.0], dtype=mx.float32),
        mesh_scale=mx.array([1.0], dtype=mx.float32),
    )
    mx.eval(context["global"], context["proj"])

    assert set(context) == {"global", "proj"}
    assert context["global"].shape == (1, 5, 3)
    assert context["proj"].shape == (1, 8, 3)
    assert np.allclose(np.array(context["proj"]), 1.0, atol=1e-5)


def test_dinov3_projection_adapter_can_emit_bilinear_hr_concat_context():
    from trellmlx.models.dinov3_proj import DINOv3ProjectionAdapter

    prefix = mx.zeros((1, 5, 3), dtype=mx.float32)
    patches = mx.ones((1, 4, 3), dtype=mx.float32)
    features = mx.concatenate([prefix, patches], axis=1)

    adapter = DINOv3ProjectionAdapter(
        image_size=32,
        patch_size=16,
        grid_resolution=2,
        num_prefix_tokens=5,
        projection_mode="bilinear_hr_concat",
        hr_feature_size=4,
    )
    context = adapter(
        features,
        camera_angle_x=mx.array([np.pi / 2], dtype=mx.float32),
        distance=mx.array([2.0], dtype=mx.float32),
        mesh_scale=mx.array([1.0], dtype=mx.float32),
    )
    mx.eval(context["global"], context["proj"])

    assert set(context) == {"global", "proj"}
    assert context["global"].shape == (1, 5, 3)
    assert context["proj"].shape == (1, 8, 6)
    assert np.allclose(np.array(context["proj"][..., :3]), 1.0, atol=1e-5)
    assert np.allclose(np.array(context["proj"][..., 3:]), 1.0, atol=1e-5)


def test_dinov3_projection_adapter_rejects_unknown_projection_mode():
    from trellmlx.models.dinov3_proj import DINOv3ProjectionAdapter

    try:
        DINOv3ProjectionAdapter(projection_mode="mystery")
    except ValueError as exc:
        assert "projection_mode" in str(exc)
    else:
        raise AssertionError("expected unknown projection_mode to fail")
