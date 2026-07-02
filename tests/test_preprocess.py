import numpy as np
from PIL import Image

from trellmlx.preprocess import preprocess_image


def _reference_trellis_mac_preprocess(image: Image.Image) -> Image.Image:
    output = image
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


def test_preprocess_matches_trellis_mac_half_pixel_crop(tmp_path):
    pixels = np.zeros((12, 12, 4), dtype=np.uint8)
    for y in range(2, 8):
        for x in range(3, 10):
            pixels[y, x] = [x * 20, y * 20, (x + y) * 10, 255]
    source = Image.fromarray(pixels, mode="RGBA")
    source_path = tmp_path / "source.png"
    source.save(source_path)

    expected = np.array(_reference_trellis_mac_preprocess(source))
    actual = np.array(preprocess_image(str(source_path)))

    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)
