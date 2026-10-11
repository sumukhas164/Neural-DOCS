"""Synthetic documents for tests: no real (personal) scans are kept in the repo."""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

PAPER_TONE = 0.92
INK = 0.08


def make_page(height: int = 640, width: int = 480, color: bool = False) -> np.ndarray:
    """A clean, upright invoice-like page as float [0, 1]."""
    page = np.full((height, width), PAPER_TONE, np.float32)
    left, right = 50, width - 50
    # Full-width rules give the content a rectangular outline, like a table.
    for y in (70, height - 70):
        cv2.line(page, (left, y), (right, y), INK, 2)
    y, row = 100, 0
    while y < height - 80:
        cv2.putText(
            page, f"Item {row:02d}  qty 2  1,234.50", (left + 4, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, INK, 1, cv2.LINE_AA,
        )
        y += 24
        row += 1
    if not color:
        return page
    rgb = np.stack([page] * 3, axis=-1)
    rgb[76:92, left + 4 : left + 120] = (0.80, 0.15, 0.15)  # a coloured logo block
    return rgb


def rotate_ccw(image: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate about the centre (counter-clockwise positive), white fill."""
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    border = 1.0 if image.ndim == 2 else (1.0, 1.0, 1.0)
    return cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=border,
    ).clip(0.0, 1.0)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(10 * np.log10(1.0 / np.mean((a - b) ** 2)))


def make_pdf_bytes(pages: list[np.ndarray], dpi: int = 100) -> bytes:
    """Build a PDF whose pages are the given float images."""
    frames = [
        Image.fromarray(np.rint(p * 255).astype(np.uint8)).convert("RGB") for p in pages
    ]
    buffer = io.BytesIO()
    frames[0].save(
        buffer, format="PDF", resolution=dpi, save_all=True, append_images=frames[1:]
    )
    return buffer.getvalue()
