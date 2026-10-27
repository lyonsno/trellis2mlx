"""Stage routing from TRELLIS.2@5565d240 run/get_cond, not seed parity."""

import ast
from pathlib import Path

import numpy as np


def test_high_resolution_encoder_preserves_sampling_rng(monkeypatch):
    import mlx.core as mx

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        import generate

        calls = []

        def extract(path, resolution=512):
            calls.append((path, resolution))
            mx.eval(mx.random.normal((7,)))
            return mx.ones((1, 3, 4)) * (1 if path == "front" else 2)

        monkeypatch.setattr(generate, "_extract_image_features", extract)
        mx.random.seed(17)
        expected = np.array(mx.random.normal((5,)))
        mx.random.seed(17)
        positive, negative = generate._extract_high_resolution_conditioning(["front", "back"])
        actual = np.array(mx.random.normal((5,)))

        assert calls == [("front", 1024), ("back", 1024)]
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(np.array(positive), [[[1] * 4] * 3 + [[2] * 4] * 3])
        np.testing.assert_array_equal(np.array(negative), np.zeros((1, 6, 4)))
    finally:
        mx.set_default_device(previous)


def test_encoder_failure_also_preserves_sampling_rng(monkeypatch):
    import mlx.core as mx
    import pytest

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        import generate

        def broken(*args, **kwargs):
            mx.eval(mx.random.normal((7,)))
            raise RuntimeError("encoder failed")

        monkeypatch.setattr(generate, "_extract_image_features", broken)
        mx.random.seed(29)
        expected = np.array(mx.random.normal((5,)))
        mx.random.seed(29)
        with pytest.raises(RuntimeError, match="encoder failed"):
            generate._extract_high_resolution_conditioning(["front"])
        np.testing.assert_array_equal(np.array(mx.random.normal((5,))), expected)
    finally:
        mx.set_default_device(previous)


def test_hr_shape_and_texture_receive_stage_specific_conditioning():
    # Inspect actual sampler wiring; a correct unused helper cannot pass this.
    source = Path(__file__).resolve().parents[1] / "generate.py"
    tree = ast.parse(source.read_text())
    observed = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "flow_euler_sample" and node.args
                and isinstance(node.args[0], ast.Name)):
            continue
        model = node.args[0].id
        if model not in {"hr_slat_flow", "tex_flow", "lr_slat_flow"}:
            continue
        if not (isinstance(node.args[1], ast.Name) and
                node.args[1].id in {"hr_noise", "tex_noise", "lr_noise"}):
            continue
        cond = node.args[2]
        if model == "lr_slat_flow" and isinstance(cond, ast.Name):
            # Diagnostic prefix replay binds shape_cond to the low-res pair.
            assert cond.id == "shape_cond" and node.args[3].id == "shape_neg_cond"
            continue
        expected = "cond_hr" if model != "lr_slat_flow" else "cond"
        assert isinstance(cond, ast.IfExp) and isinstance(cond.orelse, ast.Name)
        assert cond.orelse.id == expected, model
        assert cond.body.id == ("cond_tgt_hr" if model != "lr_slat_flow" else "cond_tgt"), model
        assert node.args[3].id == ("neg_cond_hr" if model != "lr_slat_flow" else "neg_cond"), model
        observed.add(model)
    assert observed == {"hr_slat_flow", "tex_flow", "lr_slat_flow"}


def test_checkpoint_hr_pair_and_legacy_explicit_sample(tmp_path):
    import mlx.core as mx
    import pytest

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        import generate

        low = mx.zeros((1, 2, 4))
        high = np.ones((1, 5, 4), dtype=np.float32)
        for fields, expected in (
            ({}, "legacy"),
            ({"cond_hr": high, "neg_cond_hr": np.zeros_like(high)}, "pair"),
            ({"cond_hr": high}, "missing"),
            ({"cond_hr": high, "neg_cond_hr": high[:, :2]}, "shape"),
        ):
            path = tmp_path / (expected + ".npz")
            np.savez(path, cond=np.zeros((1, 2, 4)), neg_cond=np.zeros((1, 2, 4)), **fields)
            with np.load(path, allow_pickle=False) as archive:
                if expected in {"missing", "shape"}:
                    with pytest.raises(ValueError, match="both|incompatible"):
                        generate._high_resolution_conditioning_from_sample(archive, low, low)
                else:
                    positive, negative, separate = generate._high_resolution_conditioning_from_sample(archive, low, low)
                    assert separate == (expected == "pair")
                    assert positive.shape == ((1, 5, 4) if separate else (1, 2, 4))
                    np.testing.assert_array_equal(np.array(negative), 0)
    finally:
        mx.set_default_device(previous)


def test_image_ingress_persists_both_stage_pairs(tmp_path, monkeypatch):
    """Exercise the actual conditioning block through its stop-after checkpoint."""
    import types
    import mlx.core as mx
    from PIL import Image

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        import generate

        tree = ast.parse(Path(generate.__file__).read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        start = next(i for i, n in enumerate(main.body) if isinstance(n, ast.If)
                     and isinstance(n.test, ast.Attribute) and n.test.attr == "conditioning_sample")
        stop = next(i for i in range(start + 1, len(main.body))
                    if isinstance(main.body[i], ast.Expr)
                    and isinstance(main.body[i].value, ast.Call)
                    and isinstance(main.body[i].value.func, ast.Name)
                    and main.body[i].value.func.id == "cleanup")
        block = ast.FunctionDef(name="conditioning_block", args=ast.arguments(
            posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=main.body[start:stop], decorator_list=[])
        module = ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[]))
        image = tmp_path / "image.png"
        Image.new("RGB", (8, 8), "white").save(image)
        calls = []

        def extract(path, resolution=512):
            calls.append(resolution)
            return mx.ones((1, 2 if resolution == 512 else 7, 4))

        monkeypatch.setattr(generate, "_extract_image_features", extract)
        for resolution, no_cascade in ((512, False), (768, False), (1024, False), (1024, True)):
            calls.clear()
            output = tmp_path / f"{resolution}-{no_cascade}"
            args = types.SimpleNamespace(conditioning_sample=None, image=[str(image)],
                no_rembg=True, resolution=resolution, no_cascade=no_cascade,
                shape_slat_support_mesh_grid_size=None, save_checkpoints=str(output),
                stop_after_stage="conditioning")
            namespace = dict(vars(generate), args=args)
            exec(compile(module, str(image), "exec"), namespace)
            namespace["conditioning_block"]()
            expected_high = resolution > 512 and not no_cascade
            assert calls == ([512, 1024] if expected_high else [512])
            with np.load(output / "conditioning.npz", allow_pickle=False) as archive:
                assert archive["cond"].shape == (1, 2, 4)
                assert archive["cond_hr"].shape == (1, 7 if expected_high else 2, 4)
                np.testing.assert_array_equal(archive["neg_cond_hr"], 0)
    finally:
        mx.set_default_device(previous)
