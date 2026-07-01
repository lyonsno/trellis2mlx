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
