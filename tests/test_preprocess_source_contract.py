import numpy as np
from PIL import Image

from trellmlx.preprocess import preprocess_image


def test_preprocess_uses_source_float_crop_box_for_rgba_alpha(tmp_path):
    image_path = (
        "/Users/noahlyons/.local/state/gpu-greenroom/outputs/"
        "kaminos-world-tracing-exporters-trellis-20260627T213412Z/"
        "ablate02-01-denser-interior_flood-alpha_trellis_source_rgba.png"
    )

    actual = np.array(preprocess_image(image_path))
    expected = np.array(_source_float_crop_preprocess(Image.open(image_path)))

    assert np.array_equal(actual, expected)


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
    bbox = np.argwhere(alpha > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    output = output.crop(bbox)
    output = np.array(output).astype(np.float32) / 255
    output = output[:, :, :3] * output[:, :, 3:4]
    return Image.fromarray((output * 255).astype(np.uint8))
