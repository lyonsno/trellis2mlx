import hashlib

import numpy as np
from PIL import Image

from trellmlx.preprocess import (
    ALPHA_THRESHOLD,
    describe_unprocessed_image,
    preprocess_image,
    preprocess_image_with_provenance,
)


def test_preprocess_uses_source_float_crop_box_for_rgba_alpha(tmp_path):
    image_path = tmp_path / "alpha-input.png"
    rgba = np.zeros((11, 9, 4), dtype=np.uint8)
    rgba[:, :, :3] = [120, 80, 40]
    rgba[2:8, 3:9, 3] = 255
    Image.fromarray(rgba).save(image_path)

    actual = np.array(preprocess_image(image_path))
    expected = np.array(_source_float_crop_preprocess(Image.open(image_path)))

    assert np.array_equal(actual, expected)


def test_preprocess_records_alpha_route_and_crop(tmp_path):
    image_path = tmp_path / "alpha-input.png"
    rgba = np.zeros((11, 9, 4), dtype=np.uint8)
    rgba[:, :, :3] = [120, 80, 40]
    rgba[2:8, 3:9, 3] = 255
    Image.fromarray(rgba).save(image_path)

    result = preprocess_image_with_provenance(image_path)

    assert result.provenance["route"] == "input-alpha"
    assert result.provenance["source_parity"] == "reference-crop-and-premultiply"
    assert result.provenance["background_model"] is None
    assert result.provenance["alpha_threshold"] == ALPHA_THRESHOLD
    assert result.provenance["foreground_bbox_xyxy"] == [3, 2, 8, 7]
    assert result.provenance["crop"]["bbox_xyxy"] == [3.5, 2.5, 7.5, 6.5]
    assert result.provenance["output"]["size"] == [4, 4]
    assert result.provenance["output"]["pixel_sha256"] == hashlib.sha256(
        result.image.tobytes()
    ).hexdigest()


def test_preprocess_records_background_model_artifact(tmp_path):
    image_path = tmp_path / "rgb-input.png"
    Image.new("RGB", (8, 8), (80, 40, 20)).save(image_path)
    model_path = tmp_path / "test-model.onnx"
    model_path.write_bytes(b"fixed-model-bytes")

    class FakeSession:
        model_name = "u2net"

        def __init__(self, path):
            self.model_path = str(path)

    def fake_remove(image, *, session):
        assert isinstance(session, FakeSession)
        rgba = np.zeros((image.height, image.width, 4), dtype=np.uint8)
        rgba[:, :, :3] = np.array(image)
        rgba[1:7, 1:7, 3] = 255
        return Image.fromarray(rgba)

    session = FakeSession(model_path)
    result = preprocess_image_with_provenance(
        image_path,
        rembg_session=session,
        remove_background=fake_remove,
    )

    assert result.provenance["route"] == "rembg-u2net"
    assert result.provenance["source_parity"] == (
        "known-background-model-divergence-from-source-rmbg2"
    )
    assert result.provenance["background_model"] == {
        "name": "u2net",
        "path": str(model_path.resolve()),
        "sha256": hashlib.sha256(b"fixed-model-bytes").hexdigest(),
    }
    assert result.provenance["runtime"]["rembg"]
    assert result.provenance["runtime"]["onnxruntime"]


def test_describe_unprocessed_image_records_input_bytes(tmp_path):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (5, 7), (12, 34, 56)).save(image_path)

    provenance = describe_unprocessed_image(image_path)

    assert provenance["route"] == "unprocessed"
    assert provenance["input"]["sha256"] == hashlib.sha256(
        image_path.read_bytes()
    ).hexdigest()
    assert provenance["output"]["size"] == [5, 7]


def _source_float_crop_preprocess(input_image: Image.Image) -> Image.Image:
    has_alpha = False
    if input_image.mode == "RGBA":
        alpha = np.array(input_image)[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True
    max_size = max(input_image.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        input_image = input_image.resize(
            (int(input_image.width * scale), int(input_image.height * scale)),
            Image.Resampling.LANCZOS,
        )
    output = input_image if has_alpha else input_image.convert("RGB")
    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    bbox = np.argwhere(alpha > ALPHA_THRESHOLD)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    output = output.crop(bbox)
    output = np.array(output).astype(np.float32) / 255
    output = output[:, :, :3] * output[:, :, 3:4]
    return Image.fromarray((output * 255).astype(np.uint8))
