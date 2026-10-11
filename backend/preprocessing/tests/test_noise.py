import numpy as np
import pytest

from preprocessing.pipeline.noise import add_gaussian_noise, estimate_noise_sigma

from .helpers import make_page


def test_add_noise_matches_requested_sigma():
    flat = np.full((256, 256), 0.5, np.float32)
    noisy = add_gaussian_noise(flat, 0.1, np.random.default_rng(0))
    assert noisy.dtype == np.float32
    assert abs(float((noisy - flat).std()) - 0.1) < 0.005
    assert abs(float((noisy - flat).mean())) < 0.005


def test_add_noise_clips_by_default():
    white = np.ones((64, 64), np.float32)
    rng = np.random.default_rng(0)
    assert add_gaussian_noise(white, 0.2, rng).max() <= 1.0
    assert add_gaussian_noise(white, 0.2, rng, clip=False).max() > 1.0


def test_add_noise_is_reproducible_and_validates():
    page = make_page()
    a = add_gaussian_noise(page, 0.05, np.random.default_rng(7))
    b = add_gaussian_noise(page, 0.05, np.random.default_rng(7))
    assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        add_gaussian_noise(page, -0.1)


def test_clean_page_estimates_zero_noise():
    assert estimate_noise_sigma(make_page()) < 0.002
    assert estimate_noise_sigma(make_page(color=True)) < 0.002


@pytest.mark.parametrize("color", [False, True])
@pytest.mark.parametrize("sigma", [0.02, 0.05, 0.10, 0.20])
def test_estimate_recovers_injected_sigma(sigma, color):
    noisy = add_gaussian_noise(make_page(color=color), sigma, np.random.default_rng(1))
    assert estimate_noise_sigma(noisy) == pytest.approx(sigma, rel=0.12)


def test_estimate_survives_clipping_on_pure_white_background():
    # The paper's scans have white backgrounds, where half the noise is clipped.
    page = np.ones((512, 512), np.float32)
    page[200:230, 60:450] = 0.1
    noisy = add_gaussian_noise(page, 0.10, np.random.default_rng(2))
    assert estimate_noise_sigma(noisy) == pytest.approx(0.10, rel=0.12)


def test_estimate_handles_tiny_images():
    assert estimate_noise_sigma(np.full((5, 5), 0.5, np.float32)) == 0.0
