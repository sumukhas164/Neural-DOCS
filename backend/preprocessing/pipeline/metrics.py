"""Full-reference image quality metrics (paper, Section 7, eqs. 21-28).

These six metrics compare a restored image against its clean original, so
they are only usable for evaluation, where the original is known. They are
computed with DeepInv's implementations, the library the paper used.

Direction: higher is better for PSNR, SSIM and MS-SSIM; lower is better for
MSE, LPIPS and GMSD.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

METRIC_NAMES = ("mse", "psnr", "ssim", "ms_ssim", "lpips", "gmsd")

# MS-SSIM uses five dyadic scales with an 11-pixel window, which needs images
# larger than (11 - 1) * 2**4 = 160 pixels on the shorter side.
_MS_SSIM_MIN_SIDE = 161


@lru_cache(maxsize=1)
def _metric_functions() -> dict:
    from deepinv.loss import metric

    return {
        "mse": metric.MSE(),
        "psnr": metric.PSNR(max_pixel=1.0),
        "ssim": metric.SSIM(),
        "ms_ssim": metric.SSIM(multiscale=True),
        "gmsd": metric.GMSD(),
    }


@lru_cache(maxsize=1)
def _lpips_function():
    from deepinv.loss import metric

    return metric.LPIPS(device="cpu")


def _to_batch(image: np.ndarray) -> torch.Tensor:
    array = image[None] if image.ndim == 2 else image.transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))[None]


def compute_metrics(
    restored: np.ndarray, reference: np.ndarray, include_lpips: bool = True
) -> dict[str, float | None]:
    """Compute the paper's six metrics for float [0, 1] images of equal shape.

    A metric that cannot be computed for this image (MS-SSIM on a small image,
    or LPIPS when disabled) is reported as ``None``.
    """
    if restored.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: restored {restored.shape} vs reference {reference.shape}"
        )
    x_net = _to_batch(np.clip(restored, 0.0, 1.0))
    x = _to_batch(np.clip(reference, 0.0, 1.0))

    results: dict[str, float | None] = {}
    with torch.inference_mode():
        for name, fn in _metric_functions().items():
            if name == "ms_ssim" and min(x.shape[-2:]) < _MS_SSIM_MIN_SIDE:
                results[name] = None
                continue
            results[name] = float(fn(x_net, x).mean())

        if include_lpips:
            # LPIPS runs a pretrained AlexNet that expects three channels.
            if x.shape[1] == 1:
                x_net, x = x_net.repeat(1, 3, 1, 1), x.repeat(1, 3, 1, 1)
            results["lpips"] = float(_lpips_function()(x_net, x).mean())
        else:
            results["lpips"] = None

    return {name: results[name] for name in METRIC_NAMES}
