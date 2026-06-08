"""Onboarding and runtime-polish contracts."""

from pathlib import Path

import pytest


def test_readme_uses_current_huggingface_cli():
    text = Path("README.md").read_text()

    assert "huggingface-cli" not in text
    assert "hf auth login" in text
    assert "hf download microsoft/TRELLIS.2-4B" in text
    assert "hf download microsoft/TRELLIS-image-large" in text
    assert "hf download facebook/dinov3-vitl16-pretrain-lvd1689m" in text


def test_cleanup_prefers_modern_mlx_clear_cache(monkeypatch):
    import trellmlx.cleanup as cleanup

    calls = []

    class Metal:
        def clear_cache(self):
            calls.append("metal.clear_cache")

    class FakeMX:
        metal = Metal()

        def clear_cache(self):
            calls.append("mx.clear_cache")

        def synchronize(self):
            calls.append("synchronize")

    monkeypatch.setattr(cleanup, "mx", FakeMX())

    cleanup.cleanup()

    assert calls == ["mx.clear_cache", "synchronize"]


def test_cleanup_falls_back_to_metal_clear_cache(monkeypatch):
    import trellmlx.cleanup as cleanup

    calls = []

    class Metal:
        def clear_cache(self):
            calls.append("metal.clear_cache")

    class FakeMX:
        metal = Metal()

        def synchronize(self):
            calls.append("synchronize")

    monkeypatch.setattr(cleanup, "mx", FakeMX())

    cleanup.cleanup()

    assert calls == ["metal.clear_cache", "synchronize"]


def test_generate_image_conditioning_failure_is_not_random_fallback(monkeypatch):
    import builtins
    import generate

    real_import = builtins.__import__

    def fail_dino_imports(name, *args, **kwargs):
        if name == "trellmlx.models.dinov3":
            raise RuntimeError("native missing for test")
        if name == "torch":
            raise ImportError("torch missing for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_dino_imports)

    with pytest.raises(RuntimeError, match="Image feature extraction failed") as exc_info:
        generate._extract_image_features("/tmp/does-not-matter.png")

    message = str(exc_info.value)
    assert "native DINOv3 weights" in message
    assert "trellis-mac" not in message
