import numpy as np
import pytest


def test_flow_euler_sample_can_capture_and_stop_after_first_step(tmp_path):
    import mlx.core as mx
    from trellmlx.samplers import StopAfterFirstFlowStep, flow_euler_sample

    class FakeModel:
        def __call__(self, sample, t_tensor, cond, **kwargs):
            return mx.ones_like(sample) * 2

    capture_path = tmp_path / "shape_flow_step0.npz"
    coords = np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32)

    with pytest.raises(StopAfterFirstFlowStep):
        flow_euler_sample(
            FakeModel(),
            mx.array([[0.1, 0.2], [0.3, 0.4]], dtype=mx.float32),
            mx.zeros((1, 2, 2), dtype=mx.float32),
            mx.zeros((1, 2, 2), dtype=mx.float32),
            steps=1,
            guidance_strength=1.0,
            _capture_first_step_path=str(capture_path),
            _capture_first_step_coords=coords,
            _stop_after_first_step=True,
        )

    saved = np.load(capture_path)
    np.testing.assert_allclose(saved["sample_feats"], np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32))
    np.testing.assert_allclose(saved["pred_v_feats"], np.full((2, 2), 2.0, dtype=np.float32))
    np.testing.assert_array_equal(saved["coords"], coords)
    assert saved["t"].item() == 1.0
