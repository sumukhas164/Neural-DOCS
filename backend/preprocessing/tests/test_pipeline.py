import json

import numpy as np
import pytest

from preprocessing.pipeline import PreprocessingConfig, preprocess_image, preprocess_pdf
from preprocessing.pipeline.deskew import binarize_text, estimate_skew_angle
from preprocessing.pipeline.imaging import to_float, to_uint8
from preprocessing.pipeline.noise import add_gaussian_noise
from preprocessing.pipeline.pipeline import (
    SKIP_ANGLE_BELOW_THRESHOLD,
    SKIP_BELOW_MIN_SIGMA,
    SKIP_BLANK_PAGE,
    SKIP_DENOISER_DISABLED,
    SKIP_DESKEW_DISABLED,
)

from .helpers import make_page, make_pdf_bytes, psnr, rotate_ccw

# TV needs no pretrained weights, so the full pipeline can be tested offline.
TV = PreprocessingConfig(denoiser="tv", tile_size=256)


def test_clean_upright_page_is_passed_through():
    page = make_page()
    result = preprocess_image(page, TV, dpi=300)
    assert not result.denoise_applied
    assert result.denoise_skip_reason == SKIP_BELOW_MIN_SIGMA
    assert result.sigma_used is None
    assert not result.deskew_applied
    assert result.deskew_skip_reason == SKIP_ANGLE_BELOW_THRESHOLD
    assert result.color_mode == "grayscale"
    assert result.processed.dtype == np.uint8 and result.processed.ndim == 2
    assert result.original.shape == (640, 480, 3)
    assert result.quality_before.flags == []


def test_noisy_skewed_page_is_denoised_then_deskewed():
    clean = make_page()
    degraded = add_gaussian_noise(rotate_ccw(clean, 4.0), 0.1, np.random.default_rng(0))
    result = preprocess_image(degraded, TV, dpi=300)

    assert result.denoise_applied and result.denoise_skip_reason is None
    assert result.sigma_estimated == pytest.approx(0.1, rel=0.15)
    assert result.sigma_used == pytest.approx(result.sigma_estimated)

    assert result.deskew_applied
    assert result.skew_angle == pytest.approx(4.0, abs=0.4)
    residual = estimate_skew_angle(binarize_text(to_float(result.processed)))
    assert abs(residual) < 0.4

    assert result.quality_after.noise_sigma < result.quality_before.noise_sigma / 2
    assert set(result.timings_ms) == {"quality_before", "denoise", "deskew", "quality_after"}


def test_denoising_runs_before_deskewing():
    # With the stages in the paper's order the angle is measured on a cleaned
    # image; measuring it on the raw noisy image is far less accurate.
    degraded = add_gaussian_noise(
        rotate_ccw(make_page(), 4.0), 0.2, np.random.default_rng(3)
    )
    raw_angle = estimate_skew_angle(binarize_text(degraded))
    result = preprocess_image(degraded, TV, dpi=300)
    assert abs(result.skew_angle - 4.0) < abs(raw_angle - 4.0)


def test_fixed_sigma_mode_always_denoises():
    config = TV.with_overrides({"sigma_mode": "fixed", "fixed_sigma": 0.05})
    result = preprocess_image(make_page(), config)
    assert result.denoise_applied
    assert result.sigma_used == 0.05


def test_estimated_sigma_is_capped_at_the_validated_maximum():
    noisy = add_gaussian_noise(make_page(), 0.35, np.random.default_rng(0))
    result = preprocess_image(noisy, TV.with_overrides({"deskew_enabled": False}))
    assert result.sigma_estimated > 0.25
    assert result.sigma_used == TV.max_sigma
    assert "high_noise" in result.quality_before.flags


def test_stages_can_be_disabled():
    config = PreprocessingConfig(denoiser="none", deskew_enabled=False)
    noisy = add_gaussian_noise(rotate_ccw(make_page(), 3.0), 0.1, np.random.default_rng(0))
    result = preprocess_image(noisy, config)
    assert result.denoise_skip_reason == SKIP_DENOISER_DISABLED
    assert result.deskew_skip_reason == SKIP_DESKEW_DISABLED
    assert np.array_equal(result.processed, to_uint8(noisy))


def test_blank_page_skips_both_stages():
    blank = add_gaussian_noise(
        np.full((320, 240), 0.92, np.float32), 0.05, np.random.default_rng(0)
    )
    result = preprocess_image(blank, TV)
    assert result.quality_before.is_blank
    assert result.denoise_skip_reason == SKIP_BLANK_PAGE
    assert result.deskew_skip_reason == SKIP_BLANK_PAGE
    assert result.processed.shape == (320, 240)


def test_colour_page_stays_colour_and_gray_rgb_collapses():
    colourful = make_page(color=True)
    colourful[200:500, 60:400] = (0.2, 0.4, 0.9)
    result = preprocess_image(colourful, TV)
    assert result.color_mode == "color" and result.processed.shape[2] == 3

    gray_rgb = np.stack([make_page()] * 3, axis=-1)
    result = preprocess_image(gray_rgb, TV)
    assert result.color_mode == "grayscale" and result.processed.ndim == 2


def test_accepts_uint8_input():
    result = preprocess_image(to_uint8(make_page(color=True)), TV)
    assert result.original.dtype == np.uint8


def test_metadata_is_json_serialisable_and_complete():
    result = preprocess_image(rotate_ccw(make_page(), 3.0), TV, dpi=200, page_number=7)
    meta = json.loads(json.dumps(result.metadata()))
    assert meta["page_number"] == 7 and meta["dpi"] == 200
    assert (meta["width"], meta["height"]) == result.processed.shape[::-1]
    assert len(meta["transform"]) == 2 and len(meta["transform"][0]) == 3
    assert meta["crop_box"] is not None and len(meta["crop_box"]) == 4


def test_preprocess_pdf_yields_one_result_per_page():
    pdf = make_pdf_bytes([make_page(), rotate_ccw(make_page(), 3.0)], dpi=100)
    config = PreprocessingConfig(denoiser="none", render_dpi=100)
    results = list(preprocess_pdf(pdf, config))
    assert [r.page_number for r in results] == [1, 2]
    assert not results[0].deskew_applied
    assert results[1].deskew_applied
    assert results[1].skew_angle == pytest.approx(3.0, abs=0.4)
    assert all(r.dpi == pytest.approx(100) for r in results)


def test_preprocess_pdf_respects_max_pages():
    pdf = make_pdf_bytes([make_page()] * 3, dpi=100)
    config = PreprocessingConfig(denoiser="none", render_dpi=72, max_pages=2)
    assert len(list(preprocess_pdf(pdf, config))) == 2


@pytest.mark.models
def test_default_drunet_pipeline_end_to_end():
    clean = make_page(height=320, width=240)
    degraded = add_gaussian_noise(rotate_ccw(clean, 3.0), 0.1, np.random.default_rng(0))
    config = PreprocessingConfig(tile_size=256, crop_to_content=False)
    assert config.denoiser == "drunet"
    result = preprocess_image(degraded, config)
    assert result.denoise_applied and result.deskew_applied
    assert result.skew_angle == pytest.approx(3.0, abs=0.4)
    assert result.quality_after.noise_sigma < 0.03

    # Undo nothing but the noise: compare denoise-only output with the truth.
    denoised = preprocess_image(degraded, config.with_overrides({"deskew_enabled": False}))
    reference = rotate_ccw(clean, 3.0)
    assert psnr(to_float(denoised.processed), reference) > psnr(degraded, reference) + 5
