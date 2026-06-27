from __future__ import annotations

import numpy as np

import generate


def test_load_stage1_coords_downsamples_and_dedupes_xyz(tmp_path):
    coords = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [63, 63, 63],
            [62, 62, 62],
        ],
        dtype=np.int32,
    )
    path = tmp_path / "coords.npy"
    np.save(path, coords)

    loaded = generate.load_stage1_coords(path, source_resolution=64, target_resolution=32)

    assert loaded.dtype == np.int32
    assert loaded.tolist() == [[0, 0, 0], [1, 1, 1], [31, 31, 31]]


def test_load_stage1_coords_accepts_leading_batch_column(tmp_path):
    coords = np.array(
        [
            [0, 0, 0, 0],
            [0, 63, 63, 63],
        ],
        dtype=np.int32,
    )
    path = tmp_path / "coords_bxyz.npy"
    np.save(path, coords)

    loaded = generate.load_stage1_coords(path, source_resolution=64, target_resolution=32)

    assert loaded.tolist() == [[0, 0, 0], [31, 31, 31]]
