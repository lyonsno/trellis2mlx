import json

import numpy as np


def test_save_sparse_coords_checkpoint_writes_coords_and_metadata(tmp_path):
    from generate import _save_sparse_coords_checkpoint

    lr_coords = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    lr_coords_4d = np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32)

    _save_sparse_coords_checkpoint(str(tmp_path), lr_coords, lr_coords_4d, 32)

    saved = np.load(tmp_path / "sparse_coords.npz")
    np.testing.assert_array_equal(saved["lr_coords"], lr_coords.astype(np.int32))
    np.testing.assert_array_equal(saved["coords"], lr_coords_4d)

    metadata = json.loads((tmp_path / "sparse_coords.json").read_text())
    assert metadata == {
        "lr_resolution": 32,
        "num_coords": 2,
    }


def test_save_shape_slat_checkpoint_writes_feats_coords_and_metadata(tmp_path):
    from generate import _save_shape_slat_checkpoint

    coords = np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int64)
    feats = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)

    _save_shape_slat_checkpoint(str(tmp_path), feats, coords, "no_cascade_lr_slat")

    saved = np.load(tmp_path / "shape_slat.npz")
    np.testing.assert_allclose(saved["feats"], feats.astype(np.float32))
    np.testing.assert_array_equal(saved["coords"], coords.astype(np.int32))

    metadata = json.loads((tmp_path / "shape_slat.json").read_text())
    assert metadata == {
        "stage": "no_cascade_lr_slat",
        "num_tokens": 2,
        "channels": 2,
    }
