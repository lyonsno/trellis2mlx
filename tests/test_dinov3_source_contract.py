from pathlib import Path


DINOV3_SOURCE = Path(__file__).resolve().parents[1] / "trellmlx" / "models" / "dinov3.py"


def test_dinov3_uses_source_layer_norm_epsilon():
    from trellmlx.models.dinov3 import DINOv3Layer, DINOv3ViT

    model = DINOv3ViT()
    layer = DINOv3Layer(1024, 16, 4096)

    assert model.layer_norm_eps == 1e-5
    assert layer.norm1.eps == 1e-5
    assert layer.norm2.eps == 1e-5


def test_dinov3_final_norm_uses_source_epsilon_literal():
    source = DINOV3_SOURCE.read_text()

    assert "mx.fast.layer_norm(x, None, None, self.layer_norm_eps)" in source
    assert "eps=1e-6" not in source
    assert "None, None, 1e-6" not in source
