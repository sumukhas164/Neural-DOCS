"""Layer 1 orchestration: render -> quality check -> denoise -> deskew.

Denoising runs before deskewing, as the paper prescribes (Section 5): the skew
angle is measured from Otsu-binarised text pixels, and noise specks would
otherwise be mistaken for text and distort the measurement.

This module has no Django dependency so it can be reused from a worker, a
management command, a notebook, or a test.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .config import DENOISER_NONE, SIGMA_MODE_FIXED, PreprocessingConfig
from .deskew import deskew
from .denoising import denoise
from .imaging import color_mode_of, to_float, to_gray, to_uint8, validate_image
from .pdf_renderer import render_pdf
from .quality import QualityReport, assess_quality, is_effectively_grayscale

logger = logging.getLogger(__name__)

SKIP_DENOISER_DISABLED = "denoiser_disabled"
SKIP_BLANK_PAGE = "blank_page"
SKIP_BELOW_MIN_SIGMA = "noise_below_min_sigma"
SKIP_DESKEW_DISABLED = "deskew_disabled"
SKIP_ANGLE_BELOW_THRESHOLD = "angle_below_threshold"


@dataclass
class PageResult:
    page_number: int
    original: np.ndarray  # uint8 RGB, exactly as rendered
    processed: np.ndarray  # uint8, (H, W) for grayscale pages or (H, W, 3) RGB
    color_mode: str
    dpi: float

    denoiser: str
    denoise_applied: bool
    denoise_skip_reason: str | None
    sigma_estimated: float
    sigma_used: float | None

    skew_angle: float
    deskew_applied: bool
    deskew_skip_reason: str | None
    crop_box: tuple[int, int, int, int] | None
    transform: list[list[float]]  # 2x3 affine, original -> processed pixels

    quality_before: QualityReport
    quality_after: QualityReport
    timings_ms: dict[str, float] = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        """Everything except the pixel data, as JSON-serialisable values."""
        return {
            "page_number": self.page_number,
            "color_mode": self.color_mode,
            "dpi": self.dpi,
            "width": int(self.processed.shape[1]),
            "height": int(self.processed.shape[0]),
            "denoiser": self.denoiser,
            "denoise_applied": self.denoise_applied,
            "denoise_skip_reason": self.denoise_skip_reason,
            "sigma_estimated": self.sigma_estimated,
            "sigma_used": self.sigma_used,
            "skew_angle": self.skew_angle,
            "deskew_applied": self.deskew_applied,
            "deskew_skip_reason": self.deskew_skip_reason,
            "crop_box": list(self.crop_box) if self.crop_box else None,
            "transform": self.transform,
            "quality_before": self.quality_before.to_dict(),
            "quality_after": self.quality_after.to_dict(),
            "timings_ms": self.timings_ms,
        }


def _choose_sigma(
    estimated: float, is_blank: bool, config: PreprocessingConfig
) -> tuple[float | None, str | None]:
    """Return ``(sigma, skip_reason)``; exactly one of the two is ``None``."""
    if config.denoiser == DENOISER_NONE:
        return None, SKIP_DENOISER_DISABLED
    if config.sigma_mode == SIGMA_MODE_FIXED:
        # The paper's setting: the noise level is known and always applied.
        return config.fixed_sigma, None
    if is_blank:
        return None, SKIP_BLANK_PAGE
    if estimated < config.min_sigma:
        return None, SKIP_BELOW_MIN_SIGMA
    # Stay inside the range the paper validated (sigma <= 0.20).
    return min(estimated, config.max_sigma), None


def preprocess_image(
    image: np.ndarray,
    config: PreprocessingConfig | None = None,
    dpi: float = 300.0,
    page_number: int = 1,
) -> PageResult:
    """Run Layer 1 on one page image (uint8 or float; (H, W) or (H, W, 3) RGB)."""
    config = config or PreprocessingConfig()
    validate_image(image)
    timings: dict[str, float] = {}

    def timed(name: str, start: float) -> None:
        timings[name] = round((time.perf_counter() - start) * 1000.0, 1)

    original = to_uint8(image)
    if original.ndim == 2:
        original = np.stack([original] * 3, axis=-1)

    work = to_float(image)
    # Grayscale and colour documents use different pretrained models (paper,
    # Section 4.2), so a colourless page is collapsed to one channel up front.
    if work.ndim == 3 and is_effectively_grayscale(work, config.grayscale_chroma_threshold):
        work = to_gray(work)

    start = time.perf_counter()
    quality_before = assess_quality(work, dpi, config)
    timed("quality_before", start)

    sigma_estimated = quality_before.noise_sigma
    sigma_used, denoise_skip = _choose_sigma(
        sigma_estimated, quality_before.is_blank, config
    )
    start = time.perf_counter()
    if sigma_used is not None:
        work = denoise(work, sigma_used, config)
    timed("denoise", start)

    start = time.perf_counter()
    skew_angle, deskew_applied, crop_box = 0.0, False, None
    transform = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    if not config.deskew_enabled:
        deskew_skip = SKIP_DESKEW_DISABLED
    elif quality_before.is_blank:
        deskew_skip = SKIP_BLANK_PAGE
    else:
        result = deskew(
            work,
            min_angle=config.min_skew_angle,
            crop_to_content=config.crop_to_content,
            crop_padding=config.crop_padding,
        )
        work = result.image
        skew_angle, deskew_applied = result.angle, result.rotated
        crop_box, transform = result.crop_box, result.transform
        deskew_skip = None if result.rotated else SKIP_ANGLE_BELOW_THRESHOLD
    timed("deskew", start)

    start = time.perf_counter()
    quality_after = assess_quality(work, dpi, config)
    timed("quality_after", start)

    return PageResult(
        page_number=page_number,
        original=original,
        processed=to_uint8(work),
        color_mode=color_mode_of(work),
        dpi=float(dpi),
        denoiser=config.denoiser,
        denoise_applied=sigma_used is not None,
        denoise_skip_reason=denoise_skip,
        sigma_estimated=float(sigma_estimated),
        sigma_used=None if sigma_used is None else float(sigma_used),
        skew_angle=float(skew_angle),
        deskew_applied=deskew_applied,
        deskew_skip_reason=deskew_skip,
        crop_box=crop_box,
        transform=[[float(v) for v in row] for row in transform],
        quality_before=quality_before,
        quality_after=quality_after,
        timings_ms=timings,
    )


def preprocess_pdf(
    source: str | Path | bytes,
    config: PreprocessingConfig | None = None,
) -> Iterator[PageResult]:
    """Run Layer 1 on every page of a PDF, yielding one result per page."""
    config = config or PreprocessingConfig()
    pages = render_pdf(
        source,
        dpi=config.render_dpi,
        max_page_pixels=config.max_page_pixels,
        max_pages=config.max_pages,
    )
    for page in pages:
        start = time.perf_counter()
        result = preprocess_image(
            page.image, config, dpi=page.dpi, page_number=page.page_number
        )
        logger.info(
            "page %d: sigma=%.4f denoised=%s skew=%.2f flags=%s (%.1fs)",
            result.page_number,
            result.sigma_estimated,
            result.denoise_applied,
            result.skew_angle,
            result.quality_before.flags,
            time.perf_counter() - start,
        )
        yield result
