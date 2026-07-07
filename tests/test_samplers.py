import mlx.core as mx
import numpy as np


def test_cfg_rescale_matches_reference_without_ratio_clamp():
    from trellmlx.samplers import flow_euler_sample, _pred_to_xstart, _xstart_to_pred

    noise = mx.zeros((1, 4), dtype=mx.float32)
    cond = mx.ones((1, 1), dtype=mx.float32)
    neg_cond = mx.zeros((1, 1), dtype=mx.float32)
    pred_pos = mx.array([[-10.0, 10.0, -10.0, 10.0]], dtype=mx.float32)
    pred_cfg_target = mx.array([[-1.0, 1.0, -1.0, 1.0]], dtype=mx.float32)
    guidance_strength = 2.0
    pred_neg = guidance_strength * pred_pos - pred_cfg_target

    class TwoPassModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, sample, t, conditioning):
            self.calls += 1
            return pred_pos if self.calls == 1 else pred_neg

    out = flow_euler_sample(
        TwoPassModel(),
        noise,
        cond,
        neg_cond,
        steps=1,
        guidance_strength=guidance_strength,
        guidance_rescale=1.0,
        guidance_interval=(0.0, 1.0),
        rescale_t=1.0,
        verbose=False,
    )

    t = 1.0
    sigma_min = 1e-5
    pred_cfg = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg
    x_0_pos = _pred_to_xstart(noise, t, pred_pos, sigma_min)
    x_0_cfg = _pred_to_xstart(noise, t, pred_cfg, sigma_min)
    std_pos = mx.std(x_0_pos, axis=[1], keepdims=True)
    std_cfg = mx.std(x_0_cfg, axis=[1], keepdims=True)
    unclamped_ratio = std_pos / std_cfg

    expected_x0 = x_0_cfg * unclamped_ratio
    expected_pred = _xstart_to_pred(noise, t, expected_x0, sigma_min)
    expected = noise - expected_pred

    assert float(unclamped_ratio.item()) > 2.0
    np.testing.assert_allclose(np.array(out), np.array(expected), rtol=1e-6, atol=1e-6)
