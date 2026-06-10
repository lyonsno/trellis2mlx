import numpy as np
import pytest

from generate import _filter_hr_support_components, _requantize_coords


def test_requantize_coords_matches_pixal3d_run_round_gridres_minus_one_formula():
    raw = np.array(
        [
            [0, 216, 141, 140],
            [0, 416, 505, 384],
            [0, 416, 505, 384],
        ],
        dtype=np.int32,
    )

    actual = _requantize_coords(raw, lr_resolution=512, hr_resolution=1024)

    expected = np.array(
        [
            [0, 27, 17, 17],
            [0, 51, 62, 47],
        ],
        dtype=np.int32,
    )
    assert np.array_equal(actual, expected)


def test_filter_hr_support_components_keeps_row_aligned_largest_component():
    hr_coords_3d = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [9, 9, 9],
            [9, 9, 10],
        ],
        dtype=np.int32,
    )
    quant_coords = np.column_stack([np.zeros(len(hr_coords_3d), dtype=np.int32), hr_coords_3d])
    hr_slat = np.arange(20, dtype=np.float32).reshape(5, 4)

    filtered_slat, filtered_quant, filtered_spatial, report = _filter_hr_support_components(
        hr_slat,
        quant_coords,
        hr_coords_3d,
        mode="largest",
        min_component_ratio=1e-5,
    )

    np.testing.assert_array_equal(filtered_slat, hr_slat[:3])
    np.testing.assert_array_equal(filtered_quant, quant_coords[:3])
    np.testing.assert_array_equal(filtered_spatial, hr_coords_3d[:3])
    assert report["route"] == "hr-support-component-filter"
    assert report["mode"] == "largest"
    assert report["input_count"] == 5
    assert report["kept_count"] == 3
    assert report["dropped_count"] == 2
    assert report["component_sizes"] == [3, 2]
    assert "kept_row_indices" not in report


def test_filter_hr_support_components_rejects_row_count_mismatch():
    hr_slat = np.zeros((2, 4), dtype=np.float32)
    quant_coords = np.zeros((3, 4), dtype=np.int32)
    hr_coords_3d = np.zeros((3, 3), dtype=np.int32)

    with pytest.raises(ValueError, match="SLat row count"):
        _filter_hr_support_components(
            hr_slat,
            quant_coords,
            hr_coords_3d,
            mode="largest",
            min_component_ratio=1e-5,
        )
