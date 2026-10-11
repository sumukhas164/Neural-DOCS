import cv2
import numpy as np

from preprocessing.pipeline.config import PreprocessingConfig
from preprocessing.pipeline.noise import add_gaussian_noise
from preprocessing.pipeline.quality import (
    FLAG_BLANK_PAGE,
    FLAG_BLURRY,
    FLAG_HIGH_NOISE,
    FLAG_LOW_CONTRAST,
    FLAG_LOW_RESOLUTION,
    FLAG_OVEREXPOSED,
    FLAG_TOO_DARK,
    assess_quality,
    is_effectively_grayscale,
)

from .helpers import make_page

CONFIG = PreprocessingConfig()


def test_clean_page_has_no_flags():
    report = assess_quality(make_page(), 300, CONFIG)
    assert report.flags == []
    assert (report.width, report.height) == (480, 640)
    assert report.noise_sigma < 0.002
    assert 0.01 < report.ink_ratio < 0.3
    assert not report.is_blank


def test_low_resolution_flag():
    assert FLAG_LOW_RESOLUTION in assess_quality(make_page(), 96, CONFIG).flags


def test_blank_page_flag_even_when_noisy():
    blank = np.full((640, 480), 0.92, np.float32)
    assert assess_quality(blank, 300, CONFIG).flags == [FLAG_BLANK_PAGE]
    noisy = add_gaussian_noise(blank, 0.1, np.random.default_rng(0))
    assert FLAG_BLANK_PAGE in assess_quality(noisy, 300, CONFIG).flags


def test_blurry_flag():
    blurred = cv2.GaussianBlur(make_page(), (0, 0), 4)
    assert FLAG_BLURRY in assess_quality(blurred, 300, CONFIG).flags


def test_low_contrast_flag():
    faded = 0.85 + (make_page() - 0.92) * 0.2  # ink barely darker than paper
    assert FLAG_LOW_CONTRAST in assess_quality(faded, 300, CONFIG).flags


def test_exposure_flags():
    dark = make_page() * 0.2
    assert FLAG_TOO_DARK in assess_quality(dark, 300, CONFIG).flags
    washed = np.clip(make_page(height=640, width=480) + 0.5, 0, 1)
    washed[100:110, 100:300] = 0.5  # keep some content so it is not "blank"
    assert FLAG_OVEREXPOSED in assess_quality(washed, 300, CONFIG).flags


def test_high_noise_flag_beyond_validated_range():
    noisy = add_gaussian_noise(make_page(), 0.3, np.random.default_rng(0))
    report = assess_quality(noisy, 300, CONFIG)
    assert FLAG_HIGH_NOISE in report.flags
    moderate = add_gaussian_noise(make_page(), 0.1, np.random.default_rng(0))
    assert FLAG_HIGH_NOISE not in assess_quality(moderate, 300, CONFIG).flags


def test_grayscale_detection():
    gray_as_rgb = np.stack([make_page()] * 3, axis=-1)
    assert is_effectively_grayscale(gray_as_rgb, CONFIG.grayscale_chroma_threshold)
    assert is_effectively_grayscale(make_page(), CONFIG.grayscale_chroma_threshold)
    colourful = make_page(color=True)
    colourful[200:500, 60:400] = (0.2, 0.4, 0.9)
    assert not is_effectively_grayscale(colourful, CONFIG.grayscale_chroma_threshold)


def test_report_is_json_serialisable():
    import json

    json.dumps(assess_quality(make_page(color=True), 300, CONFIG).to_dict())
