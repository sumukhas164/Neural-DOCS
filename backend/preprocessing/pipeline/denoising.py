"""Denoising stage (paper, Section 4.2): TV, DnCNN and DRUNet.

As in the paper, the two deep models are the pretrained networks shipped by
the DeepInv library, used without retraining, and TV is DeepInv's solver for
the classical total-variation problem (eq. 1).

Two things differ from the paper's experimental setup, both forced by
production use:

* The paper resized every image to 512x512. Here pages keep their full
  resolution (OCR needs it) and are processed in overlapping tiles so memory
  stays bounded.
* Weights are downloaded by DeepInv on first use and cached by torch.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .config import (
    DENOISER_DNCNN,
    DENOISER_DRUNET,
    DENOISER_NONE,
    DENOISER_TV,
    PreprocessingConfig,
)
from .imaging import validate_image

logger = logging.getLogger(__name__)

# DRUNet halves the resolution three times, so its input sides must divide by 8.
_SIZE_MULTIPLE = 8

TileFn = Callable[[torch.Tensor], torch.Tensor]


class DenoiserUnavailableError(RuntimeError):
    """Raised when a denoising model cannot be loaded (e.g. weights offline)."""


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


@lru_cache(maxsize=None)
def _load_model(name: str, channels: int, device: str) -> torch.nn.Module:
    """Load a pretrained DeepInv model once per process."""
    import deepinv

    logger.info("Loading %s (%d channel) on %s", name, channels, device)
    try:
        if name == DENOISER_DRUNET:
            model = deepinv.models.DRUNet(
                in_channels=channels,
                out_channels=channels,
                pretrained="download",
                device=device,
            )
        elif name == DENOISER_DNCNN:
            model = deepinv.models.DnCNN(
                in_channels=channels,
                out_channels=channels,
                pretrained="download",
                device=device,
            )
        else:
            raise ValueError(f"no pretrained model named {name!r}")
    except ValueError:
        raise
    except Exception as exc:
        raise DenoiserUnavailableError(
            f"could not load pretrained {name} weights: {exc}"
        ) from exc
    return model.eval()


def _blend_window(height: int, width: int, overlap: int, device: str) -> torch.Tensor:
    """Weights that ramp down towards tile edges so seams average smoothly."""

    def ramp(length: int) -> torch.Tensor:
        index = torch.arange(length, dtype=torch.float32, device=device)
        distance = torch.minimum(index + 1, length - index)
        return torch.clamp(distance / (overlap + 1), max=1.0)

    return ramp(height)[:, None] * ramp(width)[None, :]


def _tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    starts = list(range(0, length - tile, stride))
    starts.append(length - tile)
    return starts


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def run_tiled(
    batch: torch.Tensor, fn: TileFn, tile_size: int, overlap: int
) -> torch.Tensor:
    """Apply ``fn`` to a ``(1, C, H, W)`` tensor tile by tile and stitch it.

    Tiles overlap by ``overlap`` pixels and are blended with a linear ramp.
    Images smaller than one tile are padded up to a multiple of 8.
    """
    _, _, height, width = batch.shape
    tile_h = min(tile_size, _ceil_to_multiple(height, _SIZE_MULTIPLE))
    tile_w = min(tile_size, _ceil_to_multiple(width, _SIZE_MULTIPLE))
    pad_h, pad_w = max(tile_h - height, 0), max(tile_w - width, 0)
    if pad_h or pad_w:
        batch = F.pad(batch, (0, pad_w, 0, pad_h), mode="replicate")
    padded_h, padded_w = batch.shape[-2:]

    rows = _tile_starts(padded_h, tile_h, overlap)
    cols = _tile_starts(padded_w, tile_w, overlap)
    if len(rows) == 1 and len(cols) == 1:
        return fn(batch)[..., :height, :width]

    output = torch.zeros_like(batch)
    weight = torch.zeros((1, 1, padded_h, padded_w), device=batch.device)
    window = _blend_window(tile_h, tile_w, overlap, batch.device)
    for top in rows:
        for left in cols:
            tile = batch[..., top : top + tile_h, left : left + tile_w]
            output[..., top : top + tile_h, left : left + tile_w] += fn(tile) * window
            weight[..., top : top + tile_h, left : left + tile_w] += window
    return (output / weight)[..., :height, :width]


def tv_lambda_for(sigma: float, config: PreprocessingConfig) -> float:
    """Regularisation weight lambda of eq. 1 for a given noise level."""
    if config.tv_lambda is not None:
        return config.tv_lambda
    return config.tv_lambda_per_sigma * sigma


def _tile_fn(
    name: str, channels: int, sigma: float, device: str, config: PreprocessingConfig
) -> TileFn:
    if name == DENOISER_DRUNET:
        # DRUNet takes the noise level as an extra input map (paper, Section 6.1),
        # which is why one set of weights covers every sigma.
        model = _load_model(name, channels, device)
        return lambda tile: model(tile, sigma)
    if name == DENOISER_DNCNN:
        # DeepInv's pretrained DnCNN is non-blind and was trained for a single
        # low noise level (sigma = 2/255); it ignores the sigma argument.
        model = _load_model(name, channels, device)
        return lambda tile: model(tile)
    if name == DENOISER_TV:
        import deepinv

        ths = tv_lambda_for(sigma, config)

        def tv(tile: torch.Tensor) -> torch.Tensor:
            # A fresh solver per tile: TVDenoiser keeps warm-start state that
            # must not leak between tiles or pages.
            solver = deepinv.models.TVDenoiser(n_it_max=config.tv_max_iterations)
            return solver(tile, ths=ths)

        return tv
    raise ValueError(f"unknown denoiser {name!r}")


def denoise(
    image: np.ndarray, sigma: float, config: PreprocessingConfig
) -> np.ndarray:
    """Denoise a float [0, 1] image of shape (H, W) or (H, W, 3).

    ``sigma`` is the noise standard deviation on the [0, 1] intensity scale.
    The grayscale or colour variant of the model is chosen from the image shape.
    """
    validate_image(image)
    if config.denoiser == DENOISER_NONE:
        return image

    device = resolve_device(config.device)
    channels = 1 if image.ndim == 2 else 3
    fn = _tile_fn(config.denoiser, channels, float(sigma), device, config)

    array = image[None, :, :] if image.ndim == 2 else image.transpose(2, 0, 1)
    batch = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))[None].to(device)
    with torch.inference_mode():
        restored = run_tiled(batch, fn, config.tile_size, config.tile_overlap)
        restored = restored.clamp(0.0, 1.0)[0].cpu().numpy()

    return restored[0] if image.ndim == 2 else restored.transpose(1, 2, 0)
