"""Mild local contrast enhancement for pages flagged ``low_contrast``.

Not in the paper (it lists low contrast as future work). CLAHE equalises
luminance locally with a clip limit, so it lifts faint print without the hard
cut-off of binarisation that would erase thin strokes such as decimal points.
"""

from __future__ import annotations

import cv2
import numpy as np

from .imaging import to_float, to_uint8


def enhance_contrast(image: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    """Return a float [0, 1] image with CLAHE applied to its luminance only."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    data = to_uint8(image)
    if data.ndim == 2:
        return to_float(clahe.apply(data))
    # Equalise lightness only so colour stamps and highlights keep their hue.
    lab = cv2.cvtColor(data, cv2.COLOR_RGB2LAB)
    lab[..., 0] = clahe.apply(lab[..., 0])
    return to_float(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
