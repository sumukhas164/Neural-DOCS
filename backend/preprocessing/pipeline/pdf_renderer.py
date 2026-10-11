"""PDF rendering: turns each page of an uploaded PDF into an RGB raster.

Pages are yielded one at a time so memory use is bounded by a single page
regardless of document length.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pypdfium2 as pdfium

# PDF user space is defined in points, 72 per inch.
_POINTS_PER_INCH = 72.0


class PdfRenderError(Exception):
    """Raised when a PDF cannot be opened or rendered."""


@dataclass(frozen=True)
class RenderedPage:
    page_number: int  # 1-based
    image: np.ndarray  # uint8, (H, W, 3), RGB
    dpi: float  # effective DPI after any max_page_pixels downscale


def _open(source: str | Path | bytes, password: str | None = None) -> pdfium.PdfDocument:
    """Open a PDF from a filesystem path or from its raw bytes."""
    if isinstance(source, Path):
        source = str(source)
    try:
        return pdfium.PdfDocument(source, password=password)
    except (pdfium.PdfiumError, FileNotFoundError) as exc:
        raise PdfRenderError(f"cannot open PDF: {exc}") from exc


def count_pages(source: str | Path | bytes, password: str | None = None) -> int:
    pdf = _open(source, password)
    try:
        return len(pdf)
    finally:
        pdf.close()


def render_pdf(
    path: str | Path | bytes,
    dpi: int = 300,
    max_page_pixels: int = 40_000_000,
    max_pages: int | None = None,
    password: str | None = None,
) -> Iterator[RenderedPage]:
    """Yield every page of ``path`` rendered at ``dpi`` on a white background."""
    pdf = _open(path, password)
    try:
        page_count = len(pdf)
        if page_count == 0:
            raise PdfRenderError("PDF has no pages")
        if max_pages is not None:
            page_count = min(page_count, max_pages)

        for index in range(page_count):
            page = pdf[index]
            try:
                width_pt, height_pt = page.get_size()
                scale = dpi / _POINTS_PER_INCH
                pixels = (width_pt * scale) * (height_pt * scale)
                if pixels > max_page_pixels:
                    scale *= math.sqrt(max_page_pixels / pixels)

                bitmap = page.render(
                    scale=scale,
                    fill_color=(255, 255, 255, 255),
                    rev_byteorder=True,  # RGB(A) instead of pdfium's native BGR(A)
                )
                try:
                    array = bitmap.to_numpy()
                    # to_numpy() is a view on the bitmap buffer; copy before closing.
                    image = np.ascontiguousarray(array[:, :, :3]).copy()
                finally:
                    bitmap.close()
            except pdfium.PdfiumError as exc:
                raise PdfRenderError(f"cannot render page {index + 1}: {exc}") from exc
            finally:
                page.close()

            yield RenderedPage(
                page_number=index + 1,
                image=image,
                dpi=scale * _POINTS_PER_INCH,
            )
    finally:
        pdf.close()
