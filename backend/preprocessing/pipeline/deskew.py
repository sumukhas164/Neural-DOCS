"""Skew detection and correction (paper, Section 4.3.1, eqs. 17-20).

The four steps follow the paper exactly:

1. Otsu binarisation isolates text pixels from the background (eq. 17).
2. A minimum-area bounding rectangle fitted to the text pixels gives the skew
   angle (eq. 18).
3. The page is rotated by the inverse angle with an affine transform (eq. 19).
4. The result is cropped to the bounding box of the content (eq. 20).

Angle convention: ``angle`` is how far the content is rotated
counter-clockwise, in degrees as the page is displayed. Correction rotates the
page clockwise by the same amount.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .imaging import to_gray, to_uint8

# Fewer text pixels than this cannot define an orientation reliably.
_MIN_TEXT_PIXELS = 50
_WHITE = 1.0


@dataclass(frozen=True)
class DeskewResult:
    image: np.ndarray
    angle: float  # detected skew in degrees, counter-clockwise positive
    rotated: bool  # False when the angle was below the threshold
    crop_box: tuple[int, int, int, int] | None  # (x, y, w, h) in rotated coords
    # 2x3 affine matrix mapping original (x, y) pixels to output pixels, so
    # later layers can map OCR coordinates back onto the original page.
    transform: np.ndarray


def binarize_text(image: np.ndarray) -> np.ndarray:
    """Step 1 (eq. 17): Otsu threshold; text pixels are 255, background is 0."""
    gray = to_uint8(to_gray(image))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binary


def estimate_skew_angle(binary: np.ndarray) -> float:
    """Step 2 (eq. 18): skew angle from the minimum-area rectangle of the text.

    Returns 0.0 when there is too little text to measure.
    """
    points = cv2.findNonZero(binary)
    if points is None or len(points) < _MIN_TEXT_PIXELS:
        return 0.0

    # minAreaRect measures its angle clockwise on screen, but the range it
    # reports has changed between OpenCV releases ([-90, 0), (0, 90], ...) and
    # may refer to either side of the rectangle. Folding modulo 90 into
    # [-45, 45) gives the tilt of the text lines under every convention.
    clockwise = (cv2.minAreaRect(points)[-1] + 45.0) % 90.0 - 45.0
    # Report in the module's counter-clockwise-positive convention. Adding 0.0
    # normalises a negative zero.
    return float(-clockwise) + 0.0


def _rotation_matrix(width: int, height: int, angle_ccw: float) -> tuple[np.ndarray, int, int]:
    """Rotation about the centre on a canvas grown to keep the whole page."""
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle_ccw, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = int(np.ceil(width * cos + height * sin))
    new_height = int(np.ceil(width * sin + height * cos))
    matrix[0, 2] += new_width / 2.0 - width / 2.0
    matrix[1, 2] += new_height / 2.0 - height / 2.0
    return matrix, new_width, new_height


def deskew(
    image: np.ndarray,
    min_angle: float = 0.1,
    crop_to_content: bool = True,
    crop_padding: int = 16,
) -> DeskewResult:
    """Deskew a float [0, 1] image of shape (H, W) or (H, W, 3)."""
    height, width = image.shape[:2]
    binary = binarize_text(image)
    angle = estimate_skew_angle(binary)

    rotate = abs(angle) >= min_angle
    if rotate:
        # Step 3 (eq. 19): undo the skew by rotating the opposite way.
        matrix, out_width, out_height = _rotation_matrix(width, height, -angle)
        border = _WHITE if image.ndim == 2 else (_WHITE, _WHITE, _WHITE)
        output = cv2.warpAffine(
            image,
            matrix,
            (out_width, out_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border,
        )
        output = np.clip(output, 0.0, 1.0)
        # The text mask is moved with the same transform so the crop is not
        # confused by the white corners the rotation introduces.
        mask = cv2.warpAffine(
            binary, matrix, (out_width, out_height), flags=cv2.INTER_NEAREST
        )
    else:
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        output, mask = image, binary
        out_width, out_height = width, height

    crop_box = None
    if crop_to_content:
        # Step 4 (eq. 20): crop to the bounding box of the actual content.
        points = cv2.findNonZero(mask)
        if points is not None and len(points) >= _MIN_TEXT_PIXELS:
            x, y, w, h = cv2.boundingRect(points)
            x0 = max(x - crop_padding, 0)
            y0 = max(y - crop_padding, 0)
            x1 = min(x + w + crop_padding, out_width)
            y1 = min(y + h + crop_padding, out_height)
            crop_box = (x0, y0, x1 - x0, y1 - y0)
            output = output[y0:y1, x0:x1]
            matrix = matrix.copy()
            matrix[0, 2] -= x0
            matrix[1, 2] -= y0

    return DeskewResult(
        image=np.ascontiguousarray(output),
        angle=angle,
        rotated=rotate,
        crop_box=crop_box,
        transform=matrix,
    )
