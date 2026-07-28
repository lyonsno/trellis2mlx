import mlx.core as mx
import numpy as np


def _source_sparse_std_reference(values: np.ndarray) -> np.float32:
    values = np.asarray(values, dtype=np.float32)
    rows, channels = values.shape
    assert channels > 0 and channels & (channels - 1) == 0

    def source_mean(matrix: np.ndarray) -> np.float32:
        work = np.asarray(matrix, dtype=np.float32).copy()
        width = channels
        while width > 1:
            half = width // 2
            work[:, :half] = np.asarray(
                work[:, :half] + work[:, half:width],
                dtype=np.float32,
            )
            width = half
        row_means = np.asarray(
            work[:, 0] * np.float32(1.0 / channels),
            dtype=np.float32,
        )
        total = np.float32(0.0)
        for value in row_means:
            total = np.float32(total + value)
        return np.float32(total / np.int64(rows))

    mean = source_mean(values)
    mean2 = source_mean(np.asarray(values * values, dtype=np.float32))
    variance = np.float32(mean2 - np.float32(mean * mean))
    return np.sqrt(variance, dtype=np.float32)


def test_sparse_cfg_rescale_std_matches_source_reduction_topology():
    from trellmlx.samplers import _cfg_rescale_std

    rng = np.random.default_rng(20260727)
    for channels in (8, 32):
        values = rng.normal(size=(257, channels)).astype(np.float32)
        expected = _source_sparse_std_reference(values)

        actual = np.asarray(
            _cfg_rescale_std(mx.array(values), sparse_tokens=True),
            dtype=np.float32,
        ).reshape(())

        assert actual.view(np.uint32) == expected.view(np.uint32)


def test_sparse_cfg_rescale_rejects_multiple_conditioning_batches():
    from trellmlx.samplers import flow_euler_sample

    noise = mx.zeros((2, 8), dtype=mx.float32)
    cond = mx.ones((2, 1, 1), dtype=mx.float32)
    neg_cond = mx.zeros((2, 1, 1), dtype=mx.float32)
    coords = mx.array([[0, 0, 0], [1, 0, 0]], dtype=mx.int32)

    class ZeroModel:
        def __call__(self, sample, t, conditioning, **kwargs):
            return mx.zeros_like(sample)

    try:
        flow_euler_sample(
            ZeroModel(),
            noise,
            cond,
            neg_cond,
            steps=1,
            guidance_strength=1.0,
            guidance_rescale=0.0,
            verbose=False,
            coords=coords,
        )
    except ValueError as exc:
        assert str(exc) == (
            "sparse-token sampling currently requires one conditioning batch; "
            "got cond=2, neg_cond=2"
        )
    else:
        raise AssertionError("multi-batch sparse-token sampling must fail before model execution")


def test_flow_euler_rounds_product_before_source_subtraction():
    from trellmlx.samplers import flow_euler_sample

    sample_value = np.array(0xBCC9E61B, dtype=np.uint32).view(np.float32).item()
    pred_value = np.array(0xBE2E82DF, dtype=np.uint32).view(np.float32).item()
    expected_bits = np.uint32(0xBC8A70B2)
    noise = mx.array([[sample_value]], dtype=mx.float32)
    prediction = mx.array([[pred_value]], dtype=mx.float32)
    cond = mx.ones((1, 1), dtype=mx.float32)
    neg_cond = mx.zeros((1, 1), dtype=mx.float32)

    class FixedModel:
        def __call__(self, sample, t, conditioning, **kwargs):
            return prediction

    actual = flow_euler_sample(
        FixedModel(),
        noise,
        cond,
        neg_cond,
        steps=8,
        guidance_strength=1.0,
        guidance_rescale=0.0,
        rescale_t=3.0,
        verbose=False,
        stop_after_first_step=True,
    )

    actual_bits = np.asarray(actual, dtype=np.float32).reshape(()).view(np.uint32)
    assert actual_bits == expected_bits


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


def test_flow_euler_can_capture_first_sparse_step_and_stop():
    from trellmlx.samplers import flow_euler_sample, _pred_to_xstart, _xstart_to_pred

    noise = mx.zeros((1, 4), dtype=mx.float32)
    cond = mx.ones((1, 1), dtype=mx.float32)
    neg_cond = mx.zeros((1, 1), dtype=mx.float32)
    pred_pos = mx.array([[0.2, -0.4, 0.6, -0.8]], dtype=mx.float32)
    pred_neg = mx.array([[-0.1, 0.3, -0.5, 0.7]], dtype=mx.float32)
    capture = {}

    class TwoPassModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, sample, t, conditioning):
            self.calls += 1
            return pred_pos if self.calls == 1 else pred_neg

    model = TwoPassModel()
    out = flow_euler_sample(
        model,
        noise,
        cond,
        neg_cond,
        steps=4,
        guidance_strength=2.0,
        guidance_rescale=0.7,
        guidance_interval=(0.0, 1.0),
        rescale_t=1.0,
        verbose=False,
        capture_first_step=capture,
        stop_after_first_step=True,
    )

    assert model.calls == 2
    expected_keys = {
        "pred_pos",
        "pred_neg",
        "pred_cfg",
        "x0_pos",
        "x0_cfg",
        "std_pos",
        "std_cfg",
        "ratio_raw",
        "std_ratio",
        "ratio_effective",
        "x0_rescaled",
        "x0_after_rescale",
        "pred_final",
        "sample_in",
        "sample_next",
        "t",
        "t_prev",
    }
    assert set(capture) == expected_keys

    pred_cfg = 2.0 * pred_pos + (1 - 2.0) * pred_neg
    x0_pos = _pred_to_xstart(noise, 1.0, pred_pos, 1e-5)
    x0_cfg = _pred_to_xstart(noise, 1.0, pred_cfg, 1e-5)
    std_pos = mx.std(x0_pos, axis=[1], keepdims=True)
    std_cfg = mx.std(x0_cfg, axis=[1], keepdims=True)
    x0_after = 0.7 * (x0_cfg * (std_pos / std_cfg)) + 0.3 * x0_cfg
    expected_pred = _xstart_to_pred(noise, 1.0, x0_after, 1e-5)
    expected_next = noise - 0.25 * expected_pred

    np.testing.assert_allclose(np.array(capture["sample_next"]), np.array(expected_next), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.array(out), np.array(expected_next), rtol=1e-6, atol=1e-6)
    assert float(capture["t"].item()) == 1.0
    assert float(capture["t_prev"].item()) == 0.75


def test_flow_euler_uses_sequence_level_cfg_rescale_for_sparse_tokens():
    from trellmlx.samplers import flow_euler_sample, _pred_to_xstart, _xstart_to_pred

    noise = mx.zeros((2, 2), dtype=mx.float32)
    cond = mx.ones((1, 1), dtype=mx.float32)
    neg_cond = mx.zeros((1, 1), dtype=mx.float32)
    coords = mx.array([[0, 0, 0], [1, 0, 0]], dtype=mx.int32)
    pred_pos = mx.array([[0.0, 10.0], [0.0, 10.0]], dtype=mx.float32)
    pred_cfg_target = mx.array([[0.0, 1.0], [0.0, 100.0]], dtype=mx.float32)
    guidance_strength = 2.0
    pred_neg = (guidance_strength * pred_pos - pred_cfg_target) / (guidance_strength - 1.0)
    capture = {}

    class TwoPassModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, sample, t, conditioning, **kwargs):
            self.calls += 1
            assert "coords" in kwargs
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
        coords=coords,
        capture_first_step=capture,
    )

    t = 1.0
    sigma_min = 1e-5
    pred_cfg = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg
    x0_pos = _pred_to_xstart(noise, t, pred_pos, sigma_min)
    x0_cfg = _pred_to_xstart(noise, t, pred_cfg, sigma_min)
    sequence_ratio = mx.std(x0_pos) / mx.std(x0_cfg)
    per_row_ratio = mx.std(x0_pos, axis=[1], keepdims=True) / mx.std(x0_cfg, axis=[1], keepdims=True)
    expected_x0 = x0_cfg * sequence_ratio
    expected_pred = _xstart_to_pred(noise, t, expected_x0, sigma_min)
    expected = noise - expected_pred

    assert not np.allclose(np.array(sequence_ratio), np.array(per_row_ratio))
    np.testing.assert_allclose(np.array(capture["std_ratio"]), np.array(sequence_ratio), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.array(out), np.array(expected), rtol=1e-6, atol=1e-6)


def test_flow_euler_can_capture_every_step():
    from trellmlx.samplers import flow_euler_sample

    noise = mx.zeros((1, 2), dtype=mx.float32)
    cond = mx.ones((1, 1), dtype=mx.float32)
    neg_cond = mx.zeros((1, 1), dtype=mx.float32)
    captures = []

    class ConstantModel:
        def __call__(self, sample, t, conditioning):
            return mx.ones_like(sample)

    out = flow_euler_sample(
        ConstantModel(),
        noise,
        cond,
        neg_cond,
        steps=2,
        guidance_strength=1.0,
        guidance_rescale=0.0,
        guidance_interval=(0.0, 1.0),
        rescale_t=1.0,
        verbose=False,
        capture_steps=captures,
    )

    assert len(captures) == 2
    np.testing.assert_allclose(np.array(captures[0]["sample_in"]), np.zeros((1, 2), dtype=np.float32))
    np.testing.assert_allclose(np.array(captures[0]["sample_next"]), -0.5 * np.ones((1, 2), dtype=np.float32))
    np.testing.assert_allclose(np.array(captures[1]["sample_in"]), -0.5 * np.ones((1, 2), dtype=np.float32))
    np.testing.assert_allclose(np.array(captures[1]["sample_next"]), -1.0 * np.ones((1, 2), dtype=np.float32))
    np.testing.assert_allclose(np.array(out), -1.0 * np.ones((1, 2), dtype=np.float32))


def test_flow_euler_can_start_from_selected_step():
    from trellmlx.samplers import flow_euler_sample

    noise = mx.full((1, 2), 10.0, dtype=mx.float32)
    cond = mx.ones((1, 1), dtype=mx.float32)
    neg_cond = mx.zeros((1, 1), dtype=mx.float32)
    capture = {}
    seen_t = []

    class ConstantModel:
        def __call__(self, sample, t, conditioning):
            seen_t.append(float(np.array(t)[0]))
            return mx.ones_like(sample)

    out = flow_euler_sample(
        ConstantModel(),
        noise,
        cond,
        neg_cond,
        steps=4,
        guidance_strength=1.0,
        guidance_rescale=0.0,
        guidance_interval=(0.0, 1.0),
        rescale_t=1.0,
        verbose=False,
        capture_first_step=capture,
        stop_after_first_step=True,
        start_step_index=2,
    )

    assert seen_t == [500.0]
    np.testing.assert_allclose(np.array(capture["t"]), np.array(0.5, dtype=np.float32))
    np.testing.assert_allclose(np.array(capture["t_prev"]), np.array(0.25, dtype=np.float32))
    np.testing.assert_allclose(np.array(capture["sample_in"]), np.full((1, 2), 10.0, dtype=np.float32))
    np.testing.assert_allclose(np.array(out), np.full((1, 2), 9.75, dtype=np.float32))
