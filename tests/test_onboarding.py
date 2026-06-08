"""Onboarding and runtime-polish contracts."""

from pathlib import Path


def test_readme_uses_current_huggingface_cli():
    text = Path("README.md").read_text()

    assert "huggingface-cli" not in text
    assert "hf auth login" in text
    assert "hf download microsoft/TRELLIS.2-4B" in text
    assert "hf download microsoft/TRELLIS-image-large" in text


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
