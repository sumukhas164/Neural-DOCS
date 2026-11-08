"""Table-region detection and overlapping crop bands for targeted OCR.

Not in the paper. Used by the invoice workflow to re-read problem pages at
full resolution: a vision model shrinks a whole page to its pixel budget, so a
crop gives small table text more pixels. Adaptive thresholding is used here
only to find ruling lines; the crops themselves are cut from the unmodified
processed page.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .imaging import to_gray, to_uint8


@dataclass(frozen=True)
class Region:
    x: int
    y: int
    width: int
    height: int

    def as_list(self) -> list[int]:
        return [self.x, self.y, self.width, self.height]


def _line_mask(binary: np.ndarray, length: int, horizontal: bool) -> np.ndarray:
    size = (length, 1) if horizontal else (1, length)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    return cv2.dilate(cv2.erode(binary, kernel), kernel)


def detect_table_regions(
    image: np.ndarray,
    min_width_fraction: float = 0.3,
    min_height_px: int = 60,
    padding: int = 12,
) -> list[Region]:
    """Find ruled tables: areas enclosed by long horizontal and vertical lines.

    Returns regions top to bottom. Borderless tables are not detected; callers
    fall back to full-width page bands for those.
    """
    gray = to_gray(to_uint8(image))
    height, width = gray.shape
    binary = cv2.adaptiveThreshold(255 - gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 15, -2)
    horizontal = _line_mask(binary, max(width // 30, 10), horizontal=True)
    vertical = _line_mask(binary, max(height // 60, 10), horizontal=False)
    grid = cv2.dilate(cv2.bitwise_or(horizontal, vertical), np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Many invoices rule only rows; require horizontal lines inside the box.
        rows = int(np.count_nonzero(horizontal[y:y + h, x:x + w].max(axis=1)))
        if w < width * min_width_fraction or h < min_height_px or rows < 2:
            continue
        x0, y0 = max(x - padding, 0), max(y - padding, 0)
        x1, y1 = min(x + w + padding, width), min(y + h + padding, height)
        regions.append(Region(x0, y0, x1 - x0, y1 - y0))
    return sorted(regions, key=lambda region: (region.y, region.x))


def bands(region: Region, max_height: int, overlap: int) -> list[Region]:
    """Split a tall region into horizontal bands that overlap by ``overlap`` px.

    The overlap keeps a row cut by one band edge whole in the neighbouring band.
    """
    if max_height <= overlap:
        raise ValueError("max_height must exceed overlap")
    if region.height <= max_height:
        return [region]
    result = []
    top = region.y
    bottom = region.y + region.height
    while True:
        end = min(top + max_height, bottom)
        result.append(Region(region.x, top, region.width, end - top))
        if end >= bottom:
            return result
        top = end - overlap


def split_at_gaps(image: np.ndarray, max_height: int, search: float = 0.35,
                  ink_threshold: int = 160) -> list[Region]:
    """Cut a page into full-width strips no taller than ``max_height``.

    Each cut is placed on the emptiest pixel row (blank space or a ruling
    line) in the lower ``search`` fraction of the strip, so text lines and
    table rows are not split between strips and no overlap is needed.
    """
    gray = to_gray(to_uint8(image))
    height, width = gray.shape
    ink = np.count_nonzero(gray < ink_threshold, axis=1).astype(np.float32)
    # A ruling line is dark across most of the width: a safe place to cut.
    cost = np.where(ink > 0.6 * width, 0.0, ink)
    cost = np.convolve(cost, np.ones(7, np.float32), mode="same")  # avoid cutting just above descenders
    regions, top = [], 0
    while height - top > max_height:
        low, high = top + int(max_height * (1 - search)), top + max_height
        window = cost[low:high]
        cut = low + len(window) - 1 - int(np.argmin(window[::-1]))  # latest emptiest row keeps strips large
        regions.append(Region(0, top, width, cut - top))
        top = cut
    regions.append(Region(0, top, width, height - top))
    return regions


def ink_fraction(image: np.ndarray, ink_threshold: int = 160) -> float:
    gray = to_gray(to_uint8(image))
    return float(np.count_nonzero(gray < ink_threshold)) / max(gray.size, 1)


def crop(image: np.ndarray, region: Region) -> np.ndarray:
    return image[region.y:region.y + region.height, region.x:region.x + region.width]
