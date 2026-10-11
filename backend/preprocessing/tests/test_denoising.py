import numpy as np
import pytest
import torch

from preprocessing.pipeline.config import PreprocessingConfig
from preprocessing.pipeline.denoising import (
    _tile_starts,
    denoise,
    resolve_device,
    run_tiled,
    tv_lambda_for,
)
from preprocessing.pipeline.noise import add_gaussian_noise

from .helpers import make_page, psnr


# --- tiling ----------------------------------------------------------------

@pytest.mark.parametrize("length,tile,overlap", [(100, 64, 8), (64, 64, 8), (513, 128, 16), (1000, 512, 32)])
def test_tile_starts_cover_the_whole_axis(length, tile, overlap):
    starts = _tile_starts(length, tile, overlap)
    covered = np.zeros(length, bool)
    for start in starts:
        assert 0 <= start <= max(length - tile, 0)
        covered[start : start + tile] = True
    assert covered.all()
    # Neighbouring tiles overlap by at least the requested amount.
    assert all(b - a <= tile - overlap for a, b in zip(starts, starts[1:]))


@pytest.mark.parametrize("shape", [(1, 1, 50, 70), (1, 3, 200, 131), (1, 1, 64, 64), (1, 3, 129, 300)])
def test_run_tiled_identity_reproduces_input(shape):
    batch = torch.rand(shape)
    output = run_tiled(batch, lambda tile: tile, tile_size=64, overlap=8)
    assert output.shape == batch.shape
    assert torch.allclose(output, batch, atol=1e-6)


def test_run_tiled_hands_out_sizes_divisible_by_8():
    seen = []

    def record(tile):
        seen.append(tuple(tile.shape[-2:]))
        return tile

    run_tiled(torch.rand(1, 1, 50, 301), record, tile_size=128, overlap=16)
    assert seen and all(h % 8 == 0 and w % 8 == 0 for h, w in seen)
    assert all(h <= 128 and w <= 128 for h, w in seen)


# --- configuration helpers -------------------------------------------------

def test_tv_lambda_scales_with_sigma_unless_fixed():
    assert tv_lambda_for(0.1, PreprocessingConfig()) == pytest.approx(0.05)
    assert tv_lambda_for(0.1, PreprocessingConfig(tv_lambda=0.3)) == 0.3


def test_resolve_device():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cpu", "cuda"}


def test_none_denoiser_returns_input_untouched():
    page = make_page()
    assert denoise(page, 0.1, PreprocessingConfig(denoiser="none")) is page


def test_rejects_bad_shapes():
    with pytest.raises(ValueError):
        denoise(np.zeros((10, 10, 4), np.float32), 0.1, PreprocessingConfig(denoiser="tv"))


# --- the three algorithms of the paper -------------------------------------

def _small_config(**overrides) -> PreprocessingConfig:
    # Small tiles so the tiled path is exercised without a long CPU run.
    return PreprocessingConfig(tile_size=128, tile_overlap=16, **overrides)


@pytest.mark.parametrize("color", [False, True])
def test_tv_reduces_noise(color):
    clean = make_page(height=160, width=200, color=color)
    noisy = add_gaussian_noise(clean, 0.1, np.random.default_rng(0))
    restored = denoise(noisy, 0.1, _small_config(denoiser="tv"))
    assert restored.shape == clean.shape and restored.dtype == np.float32
    assert 0.0 <= restored.min() and restored.max() <= 1.0
    assert psnr(restored, clean) > psnr(noisy, clean) + 2.0


@pytest.mark.models
@pytest.mark.parametrize("color", [False, True])
@pytest.mark.parametrize("sigma", [0.05, 0.20])
def test_drunet_reduces_noise(sigma, color):
    clean = make_page(height=160, width=200, color=color)
    noisy = add_gaussian_noise(clean, sigma, np.random.default_rng(0))
    restored = denoise(noisy, sigma, _small_config(denoiser="drunet"))
    assert restored.shape == clean.shape and restored.dtype == np.float32
    assert psnr(restored, clean) > psnr(noisy, clean) + 5.0


@pytest.mark.models
def test_drunet_beats_tv_as_in_the_paper():
    clean = make_page(height=160, width=200)
    noisy = add_gaussian_noise(clean, 0.1, np.random.default_rng(0))
    drunet = denoise(noisy, 0.1, _small_config(denoiser="drunet"))
    tv = denoise(noisy, 0.1, _small_config(denoiser="tv"))
    assert psnr(drunet, clean) > psnr(tv, clean)


@pytest.mark.models
@pytest.mark.parametrize("color", [False, True])
def test_dncnn_runs_and_does_not_degrade(color):
    # DeepInv's pretrained DnCNN targets sigma = 2/255, so at the paper's
    # noise levels it changes little; it must at least not make things worse.
    clean = make_page(height=160, width=200, color=color)
    noisy = add_gaussian_noise(clean, 0.05, np.random.default_rng(0))
    restored = denoise(noisy, 0.05, _small_config(denoiser="dncnn"))
    assert restored.shape == clean.shape
    assert psnr(restored, clean) >= psnr(noisy, clean) - 0.1
