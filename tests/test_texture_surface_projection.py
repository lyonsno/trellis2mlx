"""The optional bake lookup changes sampled positions, never mesh geometry."""

import numpy as np
import pytest

from trellmlx.texture_bake import bake_texture


def mesh():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    return vertices, faces


def one_pixel(monkeypatch, sampled):
    import trellmlx.texture_bake as tb

    def raster(uvs, faces, texture_size):
        return (np.ones((1, 1), dtype=bool), np.zeros((1, 1), dtype=np.int64),
                np.array([[[0.5, 0.25, 0.25]]], dtype=np.float32))

    def sample(positions, *args):
        sampled.append(positions.copy())
        return np.array([[0.2, 0.4, positions[0, 2], 0.1, 0.7, 1.]])

    monkeypatch.setattr(tb, "rasterize_uv", raster)
    monkeypatch.setattr(tb, "sample_voxel_attrs", sample)
    monkeypatch.setattr(tb, "inpaint_texture", lambda image, *args, **kw: image)


def test_bake_fetches_color_on_original_surface_without_moving_geometry(monkeypatch):
    original, faces = mesh()
    simplified = original + [0., 0., 0.2]
    before = simplified.copy()
    sampled = []
    one_pixel(monkeypatch, sampled)
    color, _, _ = bake_texture(
        simplified, faces, original[:, :2], np.arange(3),
        np.zeros((1, 3), dtype=np.int32), np.zeros((1, 6)), 512,
        texture_size=1, backend="cpu", original_vertices=original,
        original_faces=faces,
    )
    np.testing.assert_allclose(sampled[0], [[0.25, 0.25, 0.]], atol=1e-12)
    assert color[0, 0, 2] == 0
    np.testing.assert_array_equal(simplified, before)
    np.testing.assert_array_equal(faces, [[0, 1, 2]])


def test_default_bake_keeps_simplified_surface_lookup(monkeypatch):
    original, faces = mesh()
    sampled = []
    one_pixel(monkeypatch, sampled)
    bake_texture(
        original + [0., 0., 0.2], faces, original[:, :2], np.arange(3),
        np.zeros((1, 3), dtype=np.int32), np.zeros((1, 6)), 512,
        texture_size=1, backend="cpu",
    )
    np.testing.assert_allclose(sampled[0], [[0.25, 0.25, 0.2]], atol=1e-12)


@pytest.mark.parametrize("faces", [np.array([[0, 1, 2]]), np.array([[2, 1, 0]])])
def test_closest_surface_includes_triangle_edges_and_ignores_winding(faces):
    from trellmlx.texture_bake import project_texture_positions

    original, _ = mesh()
    positions = np.array([[0.25, 0.25, 2.], [2., 0., 1.], [-1., -1., -1.]])
    projected = project_texture_positions(positions, original, faces)
    np.testing.assert_allclose(projected, [[0.25, 0.25, 0.], [1., 0., 0.],
                                           [0., 0., 0.]], atol=1e-12)


def test_partial_original_mesh_is_rejected():
    original, faces = mesh()
    with pytest.raises(ValueError, match="original_vertices and original_faces"):
        bake_texture(original, faces, original[:, :2], np.arange(3),
                     np.zeros((1, 3), dtype=np.int32), np.zeros((1, 6)), 512,
                     backend="cpu", original_vertices=original)
