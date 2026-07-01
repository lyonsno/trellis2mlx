import ast
import json
from pathlib import Path

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


def test_save_conditioning_checkpoint_writes_cond_neg_cond_and_metadata(tmp_path):
    from generate import _save_conditioning_checkpoint

    cond = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float64)
    neg_cond = np.zeros_like(cond)

    _save_conditioning_checkpoint(str(tmp_path), cond, neg_cond, "mlx_dinov3")

    saved = np.load(tmp_path / "conditioning.npz")
    np.testing.assert_allclose(saved["cond"], cond.astype(np.float32))
    np.testing.assert_allclose(saved["neg_cond"], neg_cond.astype(np.float32))

    metadata = json.loads((tmp_path / "conditioning.json").read_text())
    assert metadata == {
        "source": "mlx_dinov3",
        "shape": [1, 2, 2],
        "tokens": 2,
        "channels": 2,
    }


def test_generate_initializes_sparse_debug_before_vs3d_stage1_branch():
    source = Path("generate.py").read_text()
    module = ast.parse(source)
    main_fn = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main")

    stage1_vs3d_line = None
    ss_debug_assign_line = None
    for node in ast.walk(main_fn):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "vs3d_mode"
            and 620 <= node.lineno <= 660
        ):
            stage1_vs3d_line = node.lineno
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_ss_debug":
                    ss_debug_assign_line = node.lineno

    assert stage1_vs3d_line is not None
    assert ss_debug_assign_line is not None
    assert ss_debug_assign_line < stage1_vs3d_line
