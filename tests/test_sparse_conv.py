"""Tests for sparse 3D convolution and neighbor map caching."""

import numpy as np
import pytest

import mlx.core as mx
from trellmlx.modules.sparse_conv import (
    build_neighbor_map,
    clear_neighbor_map_cache,
    _build_neighbor_map_impl,
    _neighbor_map_cache,
    SparseConv3d,
)


def _make_small_coords(n=100, grid_size=16):
    """Random sparse coords in a small grid."""
    np.random.seed(42)
    coords = np.column_stack([
        np.zeros(n, dtype=np.int32),
        np.random.randint(0, grid_size, (n, 3), dtype=np.int32),
    ])
    # Deduplicate
    coords = np.unique(coords, axis=0)
    return mx.array(coords)


def _make_line_coords():
    """3 voxels in a line along z. Known neighbor structure."""
    coords = np.array([
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 2, 0, 0],
    ], dtype=np.int32)
    return mx.array(coords)


class TestBuildNeighborMap:
    def test_known_neighbors_line(self):
        coords = _make_line_coords()
        clear_neighbor_map_cache()
        src, tgt, k = build_neighbor_map(coords)
        src_np = np.array(src)
        tgt_np = np.array(tgt)

        # Middle voxel (idx 1) should have both neighbors (idx 0 and 2)
        # End voxels should have 1 neighbor each (plus self)
        # Each voxel is its own neighbor at kernel center (1,1,1) = k=13
        assert len(src_np) > 0
        # Self-connections: each voxel is its own neighbor
        center_k = 13  # (1,1,1) in 3x3x3
        k_np = np.array(k)
        self_mask = k_np == center_k
        assert self_mask.sum() == 3  # all 3 voxels have self-connection

    def test_empty_coords(self):
        coords = mx.array(np.zeros((0, 4), dtype=np.int32))
        clear_neighbor_map_cache()
        src, tgt, k = build_neighbor_map(coords)
        assert len(np.array(src)) == 0
        assert len(np.array(tgt)) == 0
        assert len(np.array(k)) == 0

    def test_single_voxel(self):
        coords = mx.array([[0, 5, 5, 5]], dtype=mx.int32)
        clear_neighbor_map_cache()
        src, tgt, k = build_neighbor_map(coords)
        # Only self-connection at center kernel position
        assert len(np.array(src)) == 1
        assert np.array(k)[0] == 13

    def test_symmetry(self):
        """If voxel A is a neighbor of B, then B is a neighbor of A."""
        coords = _make_small_coords()
        clear_neighbor_map_cache()
        src, tgt, k = build_neighbor_map(coords)
        src_np, tgt_np, k_np = np.array(src), np.array(tgt), np.array(k)

        # For each (src, tgt, k), the reverse (tgt, src, mirror_k) should exist
        edges = set()
        for s, t, ki in zip(src_np, tgt_np, k_np):
            edges.add((s, t, ki))

        for s, t, ki in zip(src_np, tgt_np, k_np):
            # Mirror kernel index: (kz,ky,kx) → (2-kz, 2-ky, 2-kx)
            kz, ky, kx = ki // 9, (ki % 9) // 3, ki % 3
            mirror_k = (2 - kz) * 9 + (2 - ky) * 3 + (2 - kx)
            assert (t, s, mirror_k) in edges, (
                f"Missing reverse edge: ({t},{s},{mirror_k}) for ({s},{t},{ki})"
            )

    def test_no_out_of_bounds_indices(self):
        coords = _make_small_coords(200)
        clear_neighbor_map_cache()
        src, tgt, k = build_neighbor_map(coords)
        N = len(np.array(coords))
        assert np.array(src).max() < N
        assert np.array(tgt).max() < N
        assert np.array(k).max() < 27
        assert np.array(src).min() >= 0
        assert np.array(tgt).min() >= 0
        assert np.array(k).min() >= 0


class TestNeighborMapCache:
    def test_cache_hit(self):
        coords = _make_small_coords()
        clear_neighbor_map_cache()
        r1 = build_neighbor_map(coords)
        assert len(_neighbor_map_cache) == 1
        r2 = build_neighbor_map(coords)
        assert len(_neighbor_map_cache) == 1
        # Same objects on cache hit
        assert r1 is r2

    def test_different_coords_different_cache(self):
        clear_neighbor_map_cache()
        c1 = _make_small_coords(50, grid_size=8)
        c2 = _make_small_coords(100, grid_size=16)
        build_neighbor_map(c1)
        build_neighbor_map(c2)
        assert len(_neighbor_map_cache) == 2

    def test_clear_cache(self):
        coords = _make_small_coords()
        clear_neighbor_map_cache()
        build_neighbor_map(coords)
        assert len(_neighbor_map_cache) == 1
        clear_neighbor_map_cache()
        assert len(_neighbor_map_cache) == 0

    def test_cache_correctness(self):
        """Cached result matches fresh computation."""
        coords = _make_small_coords()
        coords_np = np.array(coords)
        clear_neighbor_map_cache()

        # Fresh (uncached)
        fresh = _build_neighbor_map_impl(coords_np, 3)
        # Cached
        cached = build_neighbor_map(coords)

        np.testing.assert_array_equal(np.array(fresh[0]), np.array(cached[0]))
        np.testing.assert_array_equal(np.array(fresh[1]), np.array(cached[1]))
        np.testing.assert_array_equal(np.array(fresh[2]), np.array(cached[2]))


class TestSparseConv3d:
    def test_output_shape(self):
        coords = _make_small_coords(50)
        clear_neighbor_map_cache()
        nmap = build_neighbor_map(coords)
        N = len(np.array(coords))

        conv = SparseConv3d(in_channels=16, out_channels=32)
        feats = mx.random.normal((N, 16))
        out = conv(feats, nmap)
        mx.eval(out)

        assert out.shape == (N, 32)

    def test_zero_input_zero_output(self):
        coords = _make_small_coords(50)
        clear_neighbor_map_cache()
        nmap = build_neighbor_map(coords)
        N = len(np.array(coords))

        conv = SparseConv3d(in_channels=8, out_channels=8)
        # Zero bias to test pure convolution
        conv.bias = mx.zeros((8,))
        feats = mx.zeros((N, 8))
        out = conv(feats, nmap)
        mx.eval(out)

        np.testing.assert_allclose(np.array(out), 0.0, atol=1e-7)

    def test_deterministic(self):
        coords = _make_small_coords(50)
        clear_neighbor_map_cache()
        nmap = build_neighbor_map(coords)
        N = len(np.array(coords))

        conv = SparseConv3d(in_channels=8, out_channels=8)
        feats = mx.random.normal((N, 8))
        mx.eval(feats)

        out1 = conv(feats, nmap)
        out2 = conv(feats, nmap)
        mx.eval(out1, out2)

        np.testing.assert_array_equal(np.array(out1), np.array(out2))
