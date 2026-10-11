"""Configuration for the Layer 1 preprocessing pipeline.

Every tunable lives here so a run is fully described by one immutable object
that can be snapshotted into the database alongside its results.

Values marked "paper" come from Nather & Ahmed (2026). Values marked
"engineering default" are not specified by the paper and must be calibrated on
real documents; see ai-rules/08-limitations-and-pending.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Mapping

DENOISER_DRUNET = "drunet"
DENOISER_DNCNN = "dncnn"
DENOISER_TV = "tv"
DENOISER_NONE = "none"
DENOISER_CHOICES = (DENOISER_DRUNET, DENOISER_DNCNN, DENOISER_TV, DENOISER_NONE)

SIGMA_MODE_ESTIMATE = "estimate"
SIGMA_MODE_FIXED = "fixed"
SIGMA_MODE_CHOICES = (SIGMA_MODE_ESTIMATE, SIGMA_MODE_FIXED)

# Paper, Section 3: synthetic Gaussian noise levels used for evaluation.
PAPER_NOISE_LEVELS = (0.05, 0.10, 0.20)


class ConfigError(ValueError):
    """Raised when a preprocessing configuration is invalid."""


@dataclass(frozen=True)
class PreprocessingConfig:
    # --- PDF rendering -----------------------------------------------------
    # Paper, Section 3: the dataset was scanned at 300 dpi.
    render_dpi: int = 300
    # Engineering default: pages above this pixel count are rendered at a
    # proportionally lower DPI so one oversized page cannot exhaust memory.
    max_page_pixels: int = 40_000_000
    max_pages: int | None = None

    # --- Denoising (paper, Section 4.2) ------------------------------------
    # Paper, Sections 6-9: DRUNet was the best performer at every noise level.
    denoiser: str = DENOISER_DRUNET
    # The paper only evaluates synthetic noise of known sigma. Real scans have
    # unknown sigma, so by default it is estimated per page (engineering
    # extension); "fixed" reproduces the paper's known-sigma setting.
    sigma_mode: str = SIGMA_MODE_ESTIMATE
    fixed_sigma: float = 0.05
    # Engineering default: below this estimated sigma the page is treated as
    # clean and denoising is skipped (it could only remove detail).
    min_sigma: float = 0.01
    # Paper, Section 3: 0.20 is the highest noise level that was validated.
    max_sigma: float = 0.20
    # TV regularisation weight (lambda in paper eq. 1). The paper does not
    # report the value used. If tv_lambda is None it is derived as
    # tv_lambda_per_sigma * sigma. 0.5 gave the best PSNR at all three paper
    # noise levels on a synthetic text page (engineering default).
    tv_lambda: float | None = None
    tv_lambda_per_sigma: float = 0.5
    tv_max_iterations: int = 300
    # Tiled inference keeps memory bounded on full-resolution pages. The paper
    # instead resized every image to 512x512, which is not usable for OCR.
    tile_size: int = 512
    tile_overlap: int = 32
    # "auto" picks CUDA when available, otherwise CPU.
    device: str = "auto"

    # --- Deskewing (paper, Section 4.3.1) ----------------------------------
    deskew_enabled: bool = True
    # Engineering default: rotations smaller than this are skipped to avoid
    # resampling blur for no visible gain.
    min_skew_angle: float = 0.1
    # Paper, Step 4: crop to the bounding box of the content after rotation.
    crop_to_content: bool = True
    # Engineering default: margin (pixels) kept around the content box so
    # glyphs never touch the image border, which hurts OCR.
    crop_padding: int = 16

    # --- Image quality checks (architecture, Layer 1) ----------------------
    # None of these thresholds come from the paper; all are engineering
    # defaults that only raise flags and never reject a page.
    min_effective_dpi: int = 150
    min_sharpness: float = 50.0
    min_contrast: float = 0.20
    min_brightness: float = 0.25
    max_brightness: float = 0.98
    blank_ink_ratio: float = 0.0005
    grayscale_chroma_threshold: float = 0.02

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.denoiser not in DENOISER_CHOICES:
            raise ConfigError(
                f"denoiser must be one of {DENOISER_CHOICES}, got {self.denoiser!r}"
            )
        if self.sigma_mode not in SIGMA_MODE_CHOICES:
            raise ConfigError(
                f"sigma_mode must be one of {SIGMA_MODE_CHOICES}, got {self.sigma_mode!r}"
            )
        if not 36 <= self.render_dpi <= 1200:
            raise ConfigError("render_dpi must be between 36 and 1200")
        if self.max_page_pixels < 1_000_000:
            raise ConfigError("max_page_pixels must be at least 1,000,000")
        if self.max_pages is not None and self.max_pages < 1:
            raise ConfigError("max_pages must be positive or None")
        if not 0.0 < self.fixed_sigma <= 1.0:
            raise ConfigError("fixed_sigma must be in (0, 1]")
        if not 0.0 <= self.min_sigma < self.max_sigma <= 1.0:
            raise ConfigError("require 0 <= min_sigma < max_sigma <= 1")
        if self.tv_lambda is not None and self.tv_lambda <= 0:
            raise ConfigError("tv_lambda must be positive or None")
        if self.tv_lambda_per_sigma <= 0:
            raise ConfigError("tv_lambda_per_sigma must be positive")
        if self.tv_max_iterations < 1:
            raise ConfigError("tv_max_iterations must be positive")
        # DRUNet downsamples three times, so tiles must be a multiple of 8.
        if self.tile_size < 64 or self.tile_size % 8 != 0:
            raise ConfigError("tile_size must be a multiple of 8 and at least 64")
        if not 0 <= self.tile_overlap < self.tile_size // 2:
            raise ConfigError("tile_overlap must be in [0, tile_size / 2)")
        if self.min_skew_angle < 0:
            raise ConfigError("min_skew_angle must be non-negative")
        if self.crop_padding < 0:
            raise ConfigError("crop_padding must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(f.name for f in fields(cls))

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None) -> "PreprocessingConfig":
        """Build a config from a mapping, rejecting unknown keys."""
        values = dict(values or {})
        unknown = set(values) - set(cls.field_names())
        if unknown:
            raise ConfigError(f"unknown preprocessing options: {sorted(unknown)}")
        try:
            return cls(**values)
        except TypeError as exc:
            raise ConfigError(str(exc)) from exc

    def with_overrides(self, overrides: Mapping[str, Any] | None) -> "PreprocessingConfig":
        overrides = dict(overrides or {})
        unknown = set(overrides) - set(self.field_names())
        if unknown:
            raise ConfigError(f"unknown preprocessing options: {sorted(unknown)}")
        return replace(self, **overrides)
