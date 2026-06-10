import numpy as np

from generate import _requantize_coords


def test_requantize_coords_matches_pixal3d_reference_floor_formula():
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
            [0, 52, 63, 48],
        ],
        dtype=np.int32,
    )
    assert np.array_equal(actual, expected)
