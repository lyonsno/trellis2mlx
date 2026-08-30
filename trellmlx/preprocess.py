"""Image preprocessing and provenance for TRELLIS conditioning.

The foreground crop and alpha-premultiplication math matches the TRELLIS.2
reference pipeline. The default background-removal model does not: this port
uses rembg's U2Net model while the source pipeline uses a pinned RMBG-2.0
model. That distinction is recorded explicitly instead of being presented as
full source parity.
"""

from dataclasses import dataclass
import hashlib
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image


PREPROCESS_SCHEMA = "trellis2mlx-preprocess-v1"
DEFAULT_BACKGROUND_MODEL = "u2net"
ALPHA_THRESHOLD = 0.8 * 255


@dataclass(frozen=True)
class PreprocessResult:
    image: Image.Image
    provenance: dict[str, Any]


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_pixels(image: Image.Image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def _package_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _image_identity(image_path: str | Path, image: Image.Image) -> dict[str, Any]:
    path = Path(image_path).expanduser().resolve()
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "mode": image.mode,
        "size": list(image.size),
    }


def describe_unprocessed_image(image_path: str | Path) -> dict[str, Any]:
    """Describe an input used with ``--no-rembg`` without changing it."""
    with Image.open(image_path) as image:
        input_identity = _image_identity(image_path, image)
        pixel_sha256 = _sha256_pixels(image)
    return {
        "schema": PREPROCESS_SCHEMA,
        "route": "unprocessed",
        "source_parity": "no-background-removal-or-source-crop",
        "input": input_identity,
        "output": {
            "mode": input_identity["mode"],
            "size": input_identity["size"],
            "pixel_sha256": pixel_sha256,
        },
    }


def _rembg_model_path(session: Any) -> Path | None:
    explicit_path = getattr(session, "model_path", None)
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()

    model_name = getattr(session, "model_name", None)
    home_method = getattr(type(session), "u2net_home", None)
    if model_name and callable(home_method):
        return (Path(home_method()) / f"{model_name}.onnx").resolve()
    return None


def _background_model_identity(session: Any) -> dict[str, Any]:
    model_path = _rembg_model_path(session)
    identity = {
        "name": getattr(session, "model_name", DEFAULT_BACKGROUND_MODEL),
        "path": str(model_path) if model_path else None,
        "sha256": None,
    }
    if model_path and model_path.is_file():
        identity["sha256"] = _sha256_file(model_path)
    return identity


def preprocess_image_with_provenance(
    image_path: str | Path,
    max_size: int = 1024,
    *,
    rembg_session: Any = None,
    remove_background: Callable[..., Image.Image] | None = None,
) -> PreprocessResult:
    """Remove the background, center-crop, and record the exact route."""
    img = Image.open(image_path)
    input_identity = _image_identity(image_path, img)

    has_alpha = False
    if img.mode == "RGBA":
        alpha = np.array(img)[:, :, 3]
        has_alpha = not np.all(alpha == 255)

    original_size = img.size
    scale = min(1, max_size / max(img.size))
    if scale < 1:
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)),
            Image.Resampling.LANCZOS,
        )

    if has_alpha:
        output = img
        route = "input-alpha"
        source_parity = "reference-crop-and-premultiply"
        background_model = None
    else:
        img = img.convert("RGB")
        if remove_background is None:
            from rembg import new_session, remove

            remove_background = remove
            if rembg_session is None:
                rembg_session = new_session(DEFAULT_BACKGROUND_MODEL)
        elif rembg_session is None:
            raise ValueError("an injected remove_background callable requires rembg_session")

        output = remove_background(img, session=rembg_session)
        print("  Background removed (rembg/u2net)", flush=True)
        route = "rembg-u2net"
        source_parity = "known-background-model-divergence-from-source-rmbg2"
        background_model = _background_model_identity(rembg_session)

    output_np = np.array(output).astype(np.float32)
    if output_np.ndim != 3 or output_np.shape[2] != 4:
        raise ValueError(
            "background-removal output must be RGBA; "
            f"received shape {output_np.shape}"
        )
    alpha = output_np[:, :, 3]

    foreground = np.argwhere(alpha > ALPHA_THRESHOLD)
    provenance: dict[str, Any] = {
        "schema": PREPROCESS_SCHEMA,
        "route": route,
        "source_parity": source_parity,
        "input": input_identity,
        "resize": {
            "max_size": int(max_size),
            "scale": float(scale),
            "input_size": list(original_size),
            "output_size": list(img.size),
            "resampling": "PIL.Image.Resampling.LANCZOS",
        },
        "background_model": background_model,
        "runtime": {
            "rembg": _package_version("rembg") if background_model else None,
            "onnxruntime": _package_version("onnxruntime") if background_model else None,
            "pillow": _package_version("pillow"),
            "numpy": _package_version("numpy"),
        },
        "alpha_threshold": float(ALPHA_THRESHOLD),
        "foreground_pixels": int(len(foreground)),
    }

    if len(foreground) == 0:
        print("  WARNING: rembg found no foreground, using original image", flush=True)
        result = img.convert("RGB")
        provenance["fallback"] = "no-foreground-use-resized-rgb"
        provenance["output"] = {
            "mode": result.mode,
            "size": list(result.size),
            "pixel_sha256": _sha256_pixels(result),
        }
        return PreprocessResult(result, provenance)

    y_min, x_min = foreground.min(axis=0)
    y_max, x_max = foreground.max(axis=0)
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    size = max(x_max - x_min, y_max - y_min)
    half = size // 2
    bbox = (
        center_x - half,
        center_y - half,
        center_x + half,
        center_y + half,
    )
    output = output.crop(bbox)

    out_np = np.array(output).astype(np.float32) / 255.0
    rgb = out_np[:, :, :3] * out_np[:, :, 3:4]
    result = Image.fromarray((rgb * 255).astype(np.uint8))

    provenance["foreground_bbox_xyxy"] = [
        int(x_min),
        int(y_min),
        int(x_max),
        int(y_max),
    ]
    provenance["crop"] = {
        "center_xy": [float(center_x), float(center_y)],
        "source_extent": int(size),
        "bbox_xyxy": [float(value) for value in bbox],
    }
    provenance["output"] = {
        "mode": result.mode,
        "size": list(result.size),
        "pixel_sha256": _sha256_pixels(result),
    }
    return PreprocessResult(result, provenance)


def preprocess_image(image_path: str | Path, max_size: int = 1024) -> Image.Image:
    """Compatibility wrapper returning only the processed RGB image."""
    return preprocess_image_with_provenance(image_path, max_size).image
