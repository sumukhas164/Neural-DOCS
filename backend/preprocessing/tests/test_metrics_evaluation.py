import cv2
import numpy as np
import pytest

from preprocessing.pipeline.config import PreprocessingConfig
from preprocessing.pipeline.evaluation import (
    evaluate_image,
    list_images,
    load_image,
    summarize,
)
from preprocessing.pipeline.metrics import METRIC_NAMES, compute_metrics
from preprocessing.pipeline.noise import add_gaussian_noise

from .helpers import make_page


def test_metric_names_match_the_paper():
    assert METRIC_NAMES == ("mse", "psnr", "ssim", "ms_ssim", "lpips", "gmsd")


@pytest.mark.parametrize("color", [False, True])
def test_metrics_order_images_by_quality(color):
    clean = make_page(height=256, width=256, color=color)
    rng = np.random.default_rng(0)
    mild = compute_metrics(add_gaussian_noise(clean, 0.05, rng), clean, include_lpips=False)
    heavy = compute_metrics(add_gaussian_noise(clean, 0.20, rng), clean, include_lpips=False)

    assert mild["lpips"] is None
    for higher_is_better in ("psnr", "ssim", "ms_ssim"):
        assert mild[higher_is_better] > heavy[higher_is_better]
    for lower_is_better in ("mse", "gmsd"):
        assert mild[lower_is_better] < heavy[lower_is_better]


def test_metric_values_against_definitions():
    clean = np.full((200, 200), 0.5, np.float32)
    shifted = clean + 0.1
    metrics = compute_metrics(shifted, clean, include_lpips=False)
    assert metrics["mse"] == pytest.approx(0.01, rel=1e-3)  # eq. 21
    assert metrics["psnr"] == pytest.approx(20.0, abs=0.01)  # eq. 22: 10*log10(1/0.01)

    same = compute_metrics(make_page(256, 256), make_page(256, 256), include_lpips=False)
    assert same["mse"] == 0.0
    assert same["ssim"] == pytest.approx(1.0, abs=1e-4)
    assert same["gmsd"] == pytest.approx(0.0, abs=1e-6)


def test_ms_ssim_is_skipped_on_small_images():
    small = make_page(height=96, width=96)
    assert compute_metrics(small, small, include_lpips=False)["ms_ssim"] is None


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compute_metrics(make_page(64, 64), make_page(64, 80), include_lpips=False)


@pytest.mark.models
@pytest.mark.parametrize("color", [False, True])
def test_lpips(color):
    clean = make_page(height=256, width=256, color=color)
    noisy = add_gaussian_noise(clean, 0.2, np.random.default_rng(0))
    assert compute_metrics(clean, clean)["lpips"] == pytest.approx(0.0, abs=1e-4)
    assert compute_metrics(noisy, clean)["lpips"] > 0.05


def test_evaluation_protocol_rows_and_summary():
    config = PreprocessingConfig(tile_size=256)
    rows = []
    for name in ("a.png", "b.png"):
        rows += evaluate_image(
            make_page(height=300, width=260),
            name=name,
            algorithms=("tv",),
            noise_levels=(0.05, 0.20),
            resize=192,
            config=config,
            include_lpips=False,
        )
    assert len(rows) == 4
    assert {r.sigma for r in rows} == {0.05, 0.20}
    assert all(r.algorithm == "tv" and r.color_mode == "grayscale" for r in rows)

    summary = summarize(rows)
    assert [(s["algorithm"], s["sigma"], s["images"]) for s in summary] == [
        ("tv", 0.05, 2),
        ("tv", 0.20, 2),
    ]
    # More noise -> worse restoration, the trend of the paper's Tables 1-2.
    assert summary[0]["psnr"] > summary[1]["psnr"]
    assert summary[0]["lpips"] is None


def test_load_and_list_images(tmp_path):
    gray = (make_page(120, 100) * 255).astype(np.uint8)
    cv2.imwrite(str(tmp_path / "gray.png"), gray)
    colour = (make_page(120, 100, color=True) * 255).astype(np.uint8)
    colour[20:100, 10:90] = (30, 90, 220)
    cv2.imwrite(str(tmp_path / "colour.jpg"), colour)
    (tmp_path / "notes.txt").write_text("ignored")

    assert [p.name for p in list_images(tmp_path)] == ["colour.jpg", "gray.png"]
    assert load_image(tmp_path / "gray.png").shape == (120, 100)
    loaded = load_image(tmp_path / "colour.jpg")
    assert loaded.shape == (120, 100, 3) and loaded.dtype == np.float32
    with pytest.raises(ValueError):
        load_image(tmp_path / "notes.txt")
