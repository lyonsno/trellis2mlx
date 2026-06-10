"""Sparse coordinate support component filtering contracts."""

import numpy as np

from trellmlx.coord_components import filter_sparse_coordinate_components


def test_largest_component_filter_keeps_aligned_features_and_reports_counts():
    coords = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [10, 10, 10],
            [10, 10, 11],
        ],
        dtype=np.int32,
    )
    feats = np.arange(10, dtype=np.float32).reshape(5, 2)

    filtered_coords, filtered_feats, report = filter_sparse_coordinate_components(
        coords,
        feats,
        mode="largest",
    )

    np.testing.assert_array_equal(filtered_coords, coords[:3])
    np.testing.assert_array_equal(filtered_feats, feats[:3])
    assert report["mode"] == "largest"
    assert report["applied"] is True
    assert report["connectivity"] == 6
    assert report["coord_width"] == 3
    assert report["input_count"] == 5
    assert report["kept_count"] == 3
    assert report["dropped_count"] == 2
    assert report["component_count"] == 2
    assert report["component_sizes"] == [3, 2]
    assert report["kept_component_ids"] == [0]
    assert report["largest_fraction"] == 0.6


def test_min_ratio_filter_preserves_substantial_components_and_drops_small_islands():
    coords = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [5, 5, 5],
            [5, 5, 6],
            [9, 9, 9],
        ],
        dtype=np.int32,
    )

    filtered_coords, _, report = filter_sparse_coordinate_components(
        coords,
        mode="min_ratio",
        min_component_ratio=0.5,
    )

    np.testing.assert_array_equal(filtered_coords, coords[:5])
    assert report["mode"] == "min_ratio"
    assert report["input_count"] == 6
    assert report["kept_count"] == 5
    assert report["dropped_count"] == 1
    assert report["component_sizes"] == [3, 2, 1]
    assert report["kept_component_ids"] == [0, 1]
    assert report["min_component_ratio"] == 0.5


def test_noop_mode_reports_components_without_dropping_anything():
    coords = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 7, 7, 7],
        ],
        dtype=np.int32,
    )

    filtered_coords, filtered_feats, report = filter_sparse_coordinate_components(
        coords,
        None,
        mode="none",
    )

    np.testing.assert_array_equal(filtered_coords, coords)
    assert filtered_feats is None
    assert report["mode"] == "none"
    assert report["applied"] is False
    assert report["coord_width"] == 4
    assert report["input_count"] == 3
    assert report["kept_count"] == 3
    assert report["dropped_count"] == 0
    assert report["component_count"] == 2
    assert report["component_sizes"] == [2, 1]


def test_filter_rejects_feature_count_mismatch():
    coords = np.array([[0, 0, 0], [0, 0, 1]], dtype=np.int32)
    feats = np.zeros((1, 4), dtype=np.float32)

    try:
        filter_sparse_coordinate_components(coords, feats, mode="largest")
    except ValueError as exc:
        assert "feature row count" in str(exc)
    else:
        raise AssertionError("expected feature row count mismatch to fail")
