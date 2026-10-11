"""No-reference image quality checks for a rendered page.

The architecture lists "image quality checks" as a Layer 1 responsibility but
the paper does not define them: its metrics (MSE, PSNR, SSIM, ...) all need a
clean reference image, which does not exist for a real upload. These checks
are therefore reference-free. They only raise flags; deciding what to do with
a flagged page belongs to Layer 3 (validation and routing).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np

from .config import PreprocessingConfig
from .imaging import to_gray
from .noise import estimate_noise_sigma

FLAG_LOW_RESOLUTION = "low_resolution"
FLAG_BLURRY = "blurry"
FLAG_LOW_CONTRAST = "low_contrast"
FLAG_TOO_DARK = "too_dark"
FLAG_OVEREXPOSED = "overexposed"
FLAG_BLANK_PAGE = "blank_page"
FLAG_HIGH_NOISE = "high_noise"

# Statistics that should not react to pixel noise are measured on a reduced
# copy: area-averaging by a factor k divides the noise std by k.
_ANALYSIS_MAX_SIDE = 512
# A pixel counts as ink when it is this much darker than the page background.
_INK_DARKNESS = 0.15
_MAX_INK_DARKNESS = 0.5


@dataclass(frozen=True)
class QualityReport:
    width: int
    height: int
    dpi: float
    brightness: float  # mean intensity, 0 (black) to 1 (white)
    contrast: float  # spread between the 1st and 99th intensity percentiles
    sharpness: float  # variance of the Laplacian on the 0-255 scale
    ink_ratio: float  # fraction of the page covered by content
    chroma: float  # mean colour saturation; ~0 for a grayscale page
    noise_sigma: float  # estimated Gaussian noise std on the [0, 1] scale
    flags: list[str] = field(default_factory=list)

    @property
    def is_blank(self) -> bool:
        return FLAG_BLANK_PAGE in self.flags

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _reduce(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = _ANALYSIS_MAX_SIDE / max(height, width)
    if scale >= 1.0:
        return image
    size = (max(int(round(width * scale)), 1), max(int(round(height * scale)), 1))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def measure_chroma(image: np.ndarray) -> float:
    """Mean per-pixel spread between the RGB channels of a float image."""
    if image.ndim == 2:
        return 0.0
    small = _reduce(image)
    return float((small.max(axis=2) - small.min(axis=2)).mean())


def is_effectively_grayscale(image: np.ndarray, threshold: float) -> bool:
    """True when a page carries no meaningful colour.

    The paper treats grayscale and colour documents separately and each uses a
    different pretrained model, so the pipeline has to decide which a page is.
    """
    return measure_chroma(image) < threshold


def assess_quality(
    image: np.ndarray, dpi: float, config: PreprocessingConfig
) -> QualityReport:
    """Measure a float [0, 1] page image and flag anything that may hurt OCR."""
    height, width = image.shape[:2]
    gray = to_gray(image).astype(np.float32)
    small = _reduce(gray)

    brightness = float(small.mean())
    low, high = np.percentile(small, [1, 99])
    contrast = float(high - low)
    noise_sigma = estimate_noise_sigma(image)
    background = float(np.median(small))
    # Noise that survived the reduction must not be counted as ink, so the
    # darkness margin is at least four of its standard deviations. The cap
    # keeps real text detectable under extreme noise; there a blank page may
    # be reported as non-blank, which only costs an unnecessary denoise.
    reduction = small.shape[0] / gray.shape[0]
    ink_margin = min(
        max(_INK_DARKNESS, 4.0 * noise_sigma * reduction), _MAX_INK_DARKNESS
    )
    ink_ratio = float((small < background - ink_margin).mean())
    sharpness = float(cv2.Laplacian(gray * 255.0, cv2.CV_32F).var())

    flags: list[str] = []
    if dpi < config.min_effective_dpi:
        flags.append(FLAG_LOW_RESOLUTION)
    if ink_ratio < config.blank_ink_ratio:
        # A blank page has no edges or contrast by nature, so the remaining
        # content checks would only add noise to the report.
        flags.append(FLAG_BLANK_PAGE)
    else:
        if sharpness < config.min_sharpness:
            flags.append(FLAG_BLURRY)
        if contrast < config.min_contrast:
            flags.append(FLAG_LOW_CONTRAST)
        if brightness < config.min_brightness:
            flags.append(FLAG_TOO_DARK)
        elif brightness > config.max_brightness:
            flags.append(FLAG_OVEREXPOSED)
    if noise_sigma > config.max_sigma:
        flags.append(FLAG_HIGH_NOISE)

    return QualityReport(
        width=width,
        height=height,
        dpi=float(dpi),
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
        ink_ratio=ink_ratio,
        chroma=measure_chroma(image),
        noise_sigma=noise_sigma,
        flags=flags,
    )
