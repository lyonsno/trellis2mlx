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


def test_cleanup_tolerates_headless_no_metal(monkeypatch):
    import trellmlx.cleanup as cleanup

    calls = []

    class FakeMX:
        def clear_cache(self):
            calls.append("mx.clear_cache")
            raise RuntimeError("[metal::load_device] No Metal device available")

        def synchronize(self):
            calls.append("synchronize")

    monkeypatch.setattr(cleanup, "mx", FakeMX())

    cleanup.cleanup()

    assert calls == ["mx.clear_cache", "synchronize"]


def test_cleanup_propagates_unrelated_synchronize_failure(monkeypatch):
    import trellmlx.cleanup as cleanup

    calls = []

    class FakeMX:
        def clear_cache(self):
            calls.append("mx.clear_cache")

        def synchronize(self):
            calls.append("synchronize")
            raise RuntimeError("synchronize failed for unrelated reason")

    monkeypatch.setattr(cleanup, "mx", FakeMX())

    with pytest.raises(RuntimeError, match="synchronize failed for unrelated reason"):
        cleanup.cleanup()

    assert calls == ["mx.clear_cache", "synchronize"]


def test_readme_tests_section_avoids_stale_exact_count():
    text = Path("README.md").read_text()
    tests_section = text.split("## Tests", 1)[1].split("## Architecture", 1)[0]

    assert "47 tests" not in tests_section
    assert "Test suite covers core modules" in tests_section


def test_readme_quantization_section_states_measured_tradeoff():
    text = Path("README.md").read_text()
    quant_section = text.split("### Quantization (experimental)", 1)[1].split("### Roadmap", 1)[0]

    assert "6.4" in quant_section
    assert "M2 Pro" in quant_section
    assert "no speedup" in quant_section.lower()
    assert "memory" in quant_section.lower()
    assert "scripts/bench_quantization.py" in quant_section
    assert "checkpoint files" in quant_section
    assert "recorded before the reusable harness was committed" in quant_section
    assert "sparse flow plus synthetic 512-shape-SLat stress benchmark" in quant_section
    assert "not a full four-flow `generate.py --quantize` rerun" in quant_section


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
