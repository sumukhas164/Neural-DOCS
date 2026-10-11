"""Reproduces the paper's denoising benchmark (Sections 3, 5 and 7.2).

Protocol, as described in the paper: each clean scan is the ground truth; it
is resized to 512x512, degraded with synthetic Gaussian noise at
sigma = 0.05, 0.10 and 0.20, restored by each denoiser using the known sigma,
and scored with six full-reference metrics. Scores are then averaged over the
image set for each (algorithm, sigma) pair, giving the layout of Tables 1-2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from .config import (
    DENOISER_DNCNN,
    DENOISER_DRUNET,
    DENOISER_TV,
    PAPER_NOISE_LEVELS,
    PreprocessingConfig,
)
from .denoising import denoise
from .imaging import to_float, to_gray
from .metrics import METRIC_NAMES, compute_metrics
from .noise import add_gaussian_noise
from .quality import is_effectively_grayscale

PAPER_ALGORITHMS = (DENOISER_DRUNET, DENOISER_DNCNN, DENOISER_TV)
PAPER_EVAL_SIZE = 512
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class EvaluationRow:
    image: str
    color_mode: str
    algorithm: str
    sigma: float
    metrics: dict[str, float | None]


def load_image(path: str | Path) -> np.ndarray:
    """Read an image file as float [0, 1]; (H, W) if colourless, else RGB."""
    data = np.fromfile(str(path), dtype=np.uint8)  # tolerates non-ASCII paths
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot read image: {path}")
    image = to_float(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if is_effectively_grayscale(image, PreprocessingConfig().grayscale_chroma_threshold):
        return to_gray(image)
    return image


def list_images(directory: str | Path) -> list[Path]:
    return sorted(
        p for p in Path(directory).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def evaluate_image(
    clean: np.ndarray,
    name: str = "image",
    algorithms: Sequence[str] = PAPER_ALGORITHMS,
    noise_levels: Sequence[float] = PAPER_NOISE_LEVELS,
    resize: int | None = PAPER_EVAL_SIZE,
    config: PreprocessingConfig | None = None,
    include_lpips: bool = True,
    rng: np.random.Generator | None = None,
) -> list[EvaluationRow]:
    """Score every algorithm at every noise level on one clean image."""
    config = config or PreprocessingConfig()
    rng = rng or np.random.default_rng(0)
    if resize:
        clean = cv2.resize(clean, (resize, resize), interpolation=cv2.INTER_AREA)
    color_mode = "grayscale" if clean.ndim == 2 else "color"

    rows = []
    for sigma in noise_levels:
        # Every algorithm sees the same noisy image, so scores are comparable.
        noisy = add_gaussian_noise(clean, sigma, rng)
        for algorithm in algorithms:
            restored = denoise(noisy, sigma, config.with_overrides({"denoiser": algorithm}))
            rows.append(
                EvaluationRow(
                    image=name,
                    color_mode=color_mode,
                    algorithm=algorithm,
                    sigma=float(sigma),
                    metrics=compute_metrics(restored, clean, include_lpips),
                )
            )
    return rows


def summarize(rows: Iterable[EvaluationRow]) -> list[dict]:
    """Mean of each metric per (color_mode, algorithm, sigma), as in Tables 1-2."""
    groups: dict[tuple[str, str, float], list[EvaluationRow]] = {}
    for row in rows:
        groups.setdefault((row.color_mode, row.algorithm, row.sigma), []).append(row)

    summary = []
    for (color_mode, algorithm, sigma), members in sorted(groups.items()):
        entry = {
            "color_mode": color_mode,
            "algorithm": algorithm,
            "sigma": sigma,
            "images": len(members),
        }
        for metric in METRIC_NAMES:
            values = [m.metrics[metric] for m in members if m.metrics[metric] is not None]
            entry[metric] = float(np.mean(values)) if values else None
        summary.append(entry)
    return summary
