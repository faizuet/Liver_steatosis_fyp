"""Reject images that are not plausible liver ultrasounds before CNN inference.

A binary Diseased/Normal model will always pick one of those two labels for any
input. This gate uses appearance checks (dark field, limited colour, scan
texture) so typical photographs are not given a disease label.

Clinical PNGs with machine overlays (text, calipers) are allowed. The CNN is
still trained on a small grayscale set, so confidence can be lower on those
scans — that is shown as a caution, not a hard rejection.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from config import (
    IMAGE_MAX_BYTES,
    IMAGE_MAX_COLORFULNESS,
    IMAGE_MAX_COLORFULNESS_HARD,
    IMAGE_MAX_MEAN_LUMA,
    IMAGE_MIN_CONTRAST,
    IMAGE_MIN_DARK_RATIO,
    IMAGE_MIN_SIDE,
)

REJECT_NOT_ULTRASOUND = (
    "This does not appear to be a liver ultrasound. "
    "Please upload a liver ultrasound scan (JPG or PNG)."
)


class UnsupportedImageError(ValueError):
    """Raised when prediction is withheld for an unsupported image."""


def _luma_stats(image: Image.Image) -> dict:
    sample = image.convert("RGB")
    sample.thumbnail((256, 256))
    arr = np.asarray(sample, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    colorfulness = float((np.abs(r - g).mean() + np.abs(r - b).mean() + np.abs(g - b).mean()) / 3.0)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return {
        "width": image.width,
        "height": image.height,
        "colorfulness": colorfulness,
        "dark_ratio": float((gray < 25).mean()),
        "mean_luma": float(gray.mean()),
        "contrast": float(gray.std()),
    }


def assess_ultrasound_image(image: Image.Image) -> tuple[bool, str]:
    """Return (accepted, message). Message is empty when accepted."""
    w, h = image.size
    if min(w, h) < IMAGE_MIN_SIDE:
        return False, "Image resolution is too low. Please upload a clearer ultrasound scan."

    stats = _luma_stats(image)

    # Colour photos: high chroma and little black field.
    if stats["colorfulness"] > IMAGE_MAX_COLORFULNESS_HARD:
        return False, REJECT_NOT_ULTRASOUND
    if stats["colorfulness"] > IMAGE_MAX_COLORFULNESS and stats["dark_ratio"] < 0.25:
        return False, REJECT_NOT_ULTRASOUND
    if stats["dark_ratio"] < IMAGE_MIN_DARK_RATIO and stats["mean_luma"] > 90:
        return False, REJECT_NOT_ULTRASOUND
    if stats["mean_luma"] > IMAGE_MAX_MEAN_LUMA:
        return False, REJECT_NOT_ULTRASOUND
    if stats["contrast"] < IMAGE_MIN_CONTRAST:
        return False, "This image does not contain enough scan texture to analyse."

    return True, ""


def to_model_image(image: Image.Image) -> Image.Image:
    """Match training: grayscale scan expanded to 3 RGB channels."""
    return image.convert("L").convert("RGB")


def validate_image_file(path: str | Path) -> Image.Image:
    """Open and technically validate an image file. Raises UnsupportedImageError."""
    file_path = Path(path)
    if not file_path.is_file():
        raise UnsupportedImageError("The selected file could not be found.")
    if file_path.stat().st_size > IMAGE_MAX_BYTES:
        raise UnsupportedImageError("The file is too large. Please use an image under 25 MB.")

    try:
        with Image.open(file_path) as probe:
            probe.verify()
        image = Image.open(file_path).convert("RGB")
    except Exception as exc:
        raise UnsupportedImageError(
            "The selected file could not be opened as an image."
        ) from exc

    ok, reason = assess_ultrasound_image(image)
    if not ok:
        raise UnsupportedImageError(reason)
    return image
