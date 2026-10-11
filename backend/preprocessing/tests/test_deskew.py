import cv2
import numpy as np
import pytest

from preprocessing.pipeline.deskew import (
    binarize_text,
    deskew,
    estimate_skew_angle,
)

from .helpers import make_page, rotate_ccw

ANGLE_TOLERANCE = 0.3  # degrees


def test_binarize_marks_text_as_foreground():
    binary = binarize_text(make_page())
    assert set(np.unique(binary)) == {0, 255}
    # Text is the minority class on a page.
    assert 0.01 < (binary == 255).mean() < 0.3


@pytest.mark.parametrize("angle", [-10.0, -5.0, -2.0, -0.5, 0.5, 2.0, 5.0, 10.0])
@pytest.mark.parametrize("color", [False, True])
def test_detects_known_skew_with_correct_sign(angle, color):
    skewed = rotate_ccw(make_page(color=color), angle)
    detected = estimate_skew_angle(binarize_text(skewed))
    assert detected == pytest.approx(angle, abs=ANGLE_TOLERANCE)


@pytest.mark.parametrize("angle", [-7.0, -1.5, 3.0, 8.0])
def test_deskew_straightens_the_page(angle):
    result = deskew(rotate_ccw(make_page(), angle))
    assert result.rotated
    assert result.angle == pytest.approx(angle, abs=ANGLE_TOLERANCE)
    residual = estimate_skew_angle(binarize_text(result.image))
    assert abs(residual) < ANGLE_TOLERANCE


def test_upright_page_is_not_rotated():
    page = make_page()
    result = deskew(page, crop_to_content=False)
    assert abs(result.angle) < 0.1
    assert not result.rotated
    assert result.crop_box is None
    assert np.array_equal(result.image, page)
    assert np.allclose(result.transform, [[1, 0, 0], [0, 1, 0]])


def test_min_angle_threshold_skips_small_rotations():
    skewed = rotate_ccw(make_page(), 1.0)
    assert not deskew(skewed, min_angle=2.0).rotated
    assert deskew(skewed, min_angle=0.5).rotated


def test_crop_keeps_content_plus_padding():
    page = make_page(height=640, width=480)
    result = deskew(page, crop_to_content=True, crop_padding=10)
    x, y, w, h = result.crop_box
    assert result.image.shape == (h, w)
    # Content spans x in [50, 430] and y in [70, 570]; the crop hugs it.
    assert 35 <= x <= 45 and 55 <= y <= 65
    assert w < 480 and h < 640
    # Nothing was cut off: every text pixel survives the crop.
    assert (binarize_text(result.image) > 0).sum() == (binarize_text(page) > 0).sum()


def test_color_page_keeps_three_channels():
    result = deskew(rotate_ccw(make_page(color=True), 4.0))
    assert result.image.ndim == 3 and result.image.shape[2] == 3
    assert result.image.dtype == np.float32
    assert 0.0 <= result.image.min() and result.image.max() <= 1.0


def test_transform_maps_original_points_onto_output():
    def marker_centre(image):
        # Only the interior of the solid block stays dark under a box filter;
        # one-pixel text strokes never do.
        solid = cv2.blur(image, (9, 9)) < 0.05
        assert solid.any()
        return np.argwhere(solid).mean(axis=0)[::-1]  # (x, y)

    page = make_page()
    page[292:312, 380:400] = 0.0  # a solid block in the right-hand margin area
    skewed = rotate_ccw(page, 5.0)
    x, y = marker_centre(skewed)

    result = deskew(skewed)
    mapped = result.transform @ np.array([x, y, 1.0])
    assert np.allclose(mapped, marker_centre(result.image), atol=1.0)


def test_blank_page_is_left_alone():
    blank = np.full((200, 200), 0.95, np.float32)
    result = deskew(blank)
    assert result.angle == 0.0
    assert not result.rotated
    assert result.image.shape == blank.shape
