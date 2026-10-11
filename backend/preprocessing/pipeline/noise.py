"""Gaussian noise simulation and blind noise-level estimation.

``add_gaussian_noise`` implements the paper's degradation model (eq. 6),
``y = x + n, n ~ N(0, sigma^2)``, used to build the evaluation benchmark.

``estimate_noise_sigma`` is an engineering extension. The paper always knows
sigma because it injects the noise itself; a production scan has no such
label, yet DRUNet needs a noise level as input. The estimate is what makes the
paper's denoisers usable on real pages.
"""

from __future__ import annotations

import numpy as np

_BLOCK_SIZE = 32
# Text strokes make most blocks look "noisy". Background blocks are the
# quietest ones, so the estimate is read from the low end of the distribution.
_FLAT_BLOCK_PERCENTILE = 10.0


def add_gaussian_noise(
    image: np.ndarray,
    sigma: float,
    rng: np.random.Generator | None = None,
    clip: bool = True,
) -> np.ndarray:
    """Degrade a float [0, 1] image with additive white Gaussian noise (eq. 6).

    ``clip`` keeps the result a valid 8-bit-storable image, matching the paper's
    workflow of saving the noisy intermediates to disk.
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    rng = rng or np.random.default_rng()
    noise = rng.normal(0.0, sigma, size=image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0.0, 1.0) if clip else noisy


def _estimate_channel(channel: np.ndarray) -> float:
    height, width = channel.shape
    rows, cols = height // _BLOCK_SIZE, width // _BLOCK_SIZE
    if rows == 0 or cols == 0:
        blocks = channel.reshape(1, -1)
    else:
        blocks = (
            channel[: rows * _BLOCK_SIZE, : cols * _BLOCK_SIZE]
            .reshape(rows, _BLOCK_SIZE, cols, _BLOCK_SIZE)
            .transpose(0, 2, 1, 3)
            .reshape(rows * cols, -1)
        )

    # Paper backgrounds sit near white, where noise is clipped at 1.0 and a
    # plain standard deviation reads low. Pixels darker than the block median
    # are untouched by that clipping, and for Gaussian noise their spread
    # around the median is a half-normal with the same sigma.
    median = np.median(blocks, axis=1, keepdims=True)
    below = np.minimum(blocks - median, 0.0)
    block_sigma = np.sqrt(2.0 * np.mean(below * below, axis=1))
    return float(np.percentile(block_sigma, _FLAT_BLOCK_PERCENTILE))


def estimate_noise_sigma(image: np.ndarray) -> float:
    """Estimate the std of additive Gaussian noise in a float [0, 1] image.

    The page is split into blocks and the noise level is read from the
    quietest ones (flat background), so text does not inflate the estimate.
    Colour pages are measured per channel: averaging channels first would
    cancel independent noise and under-report it.
    """
    image = image.astype(np.float32)
    if image.ndim == 2:
        return _estimate_channel(image)
    return float(np.mean([_estimate_channel(image[:, :, c]) for c in range(3)]))
