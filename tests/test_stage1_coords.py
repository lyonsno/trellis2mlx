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


def test_coord_stats_records_empty_and_nonempty_bounds():
    empty = generate.coord_stats(np.empty((0, 3), dtype=np.int32))
    assert empty["count"] == 0
    assert empty["min"] is None
    assert empty["max"] is None

    coords = np.array([[1, 2, 3], [5, 7, 11]], dtype=np.int32)
    stats = generate.coord_stats(coords)
    assert stats["count"] == 2
    assert stats["min"] == [1, 2, 3]
    assert stats["max"] == [5, 7, 11]
    assert stats["span"] == [4, 5, 8]


def test_write_probe_report_creates_parent_and_json(tmp_path):
    path = tmp_path / "nested" / "probe.json"
    generate.write_probe_report(path, {"phase": "stage1", "status": "ok"})

    assert path.exists()
    assert '"phase": "stage1"' in path.read_text(encoding="utf-8")
