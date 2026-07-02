"""Image preprocessing: background removal + object-centered crop.

Matches the TRELLIS.2 reference pipeline's preprocess_image:
1. Remove background via rembg → alpha mask
2. Find tight bounding box around object (alpha > 0.8)
3. Square crop centered on object
4. Premultiply alpha (background → black)
"""

import numpy as np
from PIL import Image


def preprocess_image(image_path: str, max_size: int = 1024) -> Image.Image:
    """Remove background and center-crop the object.

    Args:
        image_path: Path to input image.
        max_size: Downscale if larger than this.

    Returns:
        RGB PIL Image with background removed (black) and object centered.
    """
    img = Image.open(image_path)

    # Check if image already has meaningful alpha
    has_alpha = False
    if img.mode == "RGBA":
        alpha = np.array(img)[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True

    # Downscale if needed
    scale = min(1, max_size / max(img.size))
    if scale < 1:
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)),
            Image.LANCZOS,
        )

    if has_alpha:
        output = img
    else:
        img = img.convert("RGB")
        from rembg import remove
        output = remove(img)
        print("  Background removed", flush=True)

    output_np = np.array(output).astype(np.float32)
    alpha = output_np[:, :, 3]

    # Find tight bounding box around object
    foreground = np.argwhere(alpha > 0.8 * 255)
    if len(foreground) == 0:
        print("  WARNING: rembg found no foreground, using original image", flush=True)
        return img.convert("RGB")

    bbox = (
        np.min(foreground[:, 1]),
        np.min(foreground[:, 0]),
        np.max(foreground[:, 1]),
        np.max(foreground[:, 0]),
    )

    # Keep the half-pixel crop semantics of the Trellis-Mac reference route.
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1)
    bbox = (
        center[0] - size // 2,
        center[1] - size // 2,
        center[0] + size // 2,
        center[1] + size // 2,
    )
    output = output.crop(bbox)

    # Premultiply alpha: RGB * alpha, background → black
    out_np = np.array(output).astype(np.float32) / 255.0
    rgb = out_np[:, :, :3] * out_np[:, :, 3:4]
    result = Image.fromarray((rgb * 255).astype(np.uint8))

    return result
