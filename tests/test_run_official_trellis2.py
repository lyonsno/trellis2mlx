from types import SimpleNamespace

import numpy as np
import pytest


def test_build_run_kwargs_records_seed_steps_pipeline_and_preprocess_switch():
    from scripts.run_official_trellis2 import _build_run_kwargs

    args = SimpleNamespace(
        seed=101,
        steps=8,
        pipeline_type="512",
        no_preprocess=True,
    )

    assert _build_run_kwargs(args) == {
        "seed": 101,
        "pipeline_type": "512",
        "preprocess_image": False,
        "sparse_structure_sampler_params": {"steps": 8},
        "shape_slat_sampler_params": {"steps": 8},
        "tex_slat_sampler_params": {"steps": 8},
    }


def test_sparse_capture_hook_saves_coords_and_can_stop(tmp_path):
    from scripts.run_official_trellis2 import _StopAfterSparse, _install_sparse_capture_hook

    class FakePipeline:
        def __init__(self):
            self.calls = 0

        def sample_sparse_structure(self, *args, **kwargs):
            self.calls += 1
            return np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32)

    pipeline = FakePipeline()
    _install_sparse_capture_hook(pipeline, str(tmp_path), stop_after_sparse=True)

    with pytest.raises(_StopAfterSparse):
        pipeline.sample_sparse_structure("cond", resolution=32)

    saved = np.load(tmp_path / "sparse_coords.npz")
    np.testing.assert_array_equal(
        saved["coords"],
        np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32),
    )
    assert pipeline.calls == 1


def test_shape_slat_capture_hook_saves_feats_coords_and_can_stop(tmp_path):
    from scripts.run_official_trellis2 import _StopAfterShapeSLat, _install_shape_slat_capture_hook

    class FakeSLat:
        feats = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        coords = np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32)

    class FakePipeline:
        def __init__(self):
            self.calls = 0

        def sample_shape_slat(self, *args, **kwargs):
            self.calls += 1
            return FakeSLat()

    pipeline = FakePipeline()
    _install_shape_slat_capture_hook(pipeline, str(tmp_path), stop_after_shape_slat=True)

    with pytest.raises(_StopAfterShapeSLat):
        pipeline.sample_shape_slat("cond", "flow", np.array([[0, 1, 2, 3]], dtype=np.int32))

    saved = np.load(tmp_path / "shape_slat.npz")
    np.testing.assert_allclose(saved["feats"], FakeSLat.feats)
    np.testing.assert_array_equal(saved["coords"], FakeSLat.coords)
    assert pipeline.calls == 1


def test_conditioning_capture_hook_saves_cond_neg_cond_and_can_stop(tmp_path):
    from scripts.run_official_trellis2 import _StopAfterConditioning, _install_conditioning_capture_hook

    class FakeTensor:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class FakePipeline:
        def __init__(self):
            self.calls = 0

        def get_cond(self, *args, **kwargs):
            self.calls += 1
            return {
                "cond": FakeTensor(np.array([[[1.0, 2.0]]], dtype=np.float32)),
                "neg_cond": FakeTensor(np.array([[[0.0, 0.0]]], dtype=np.float32)),
            }

    pipeline = FakePipeline()
    _install_conditioning_capture_hook(pipeline, str(tmp_path), stop_after_conditioning=True)

    with pytest.raises(_StopAfterConditioning):
        pipeline.get_cond(["image"], 512)

    saved = np.load(tmp_path / "conditioning.npz")
    np.testing.assert_allclose(saved["cond"], np.array([[[1.0, 2.0]]], dtype=np.float32))
    np.testing.assert_allclose(saved["neg_cond"], np.array([[[0.0, 0.0]]], dtype=np.float32))
    assert pipeline.calls == 1
