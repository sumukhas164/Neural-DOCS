"""Shared image conventions for the preprocessing pipeline.

Inside the pipeline an image is a float32 array scaled to [0, 1]:
grayscale pages are ``(H, W)`` and colour pages are ``(H, W, 3)`` in RGB order.
At the boundaries (PDF renderer in, storage out) images are uint8.
"""

from __future__ import annotations

import cv2
import numpy as np

COLOR_MODE_GRAY = "grayscale"
COLOR_MODE_COLOR = "color"


def to_float(image: np.ndarray) -> np.ndarray:
    """Convert a uint8 (or float) image to float32 in [0, 1]."""
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return np.clip(image.astype(np.float32), 0.0, 1.0)


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a float image in [0, 1] to uint8."""
    if image.dtype == np.uint8:
        return image
    return np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)


def to_gray(image: np.ndarray) -> np.ndarray:
    """Return a single-channel version of ``image`` with the same dtype."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def color_mode_of(image: np.ndarray) -> str:
    return COLOR_MODE_GRAY if image.ndim == 2 else COLOR_MODE_COLOR


def validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")
    if image.ndim == 2:
        return
    if image.ndim == 3 and image.shape[2] == 3:
        return
    raise ValueError(
        f"image must have shape (H, W) or (H, W, 3), got {image.shape}"
    )
