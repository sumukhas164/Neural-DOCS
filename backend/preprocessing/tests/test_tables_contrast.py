import cv2
import numpy as np
import pytest

from preprocessing.pipeline import PreprocessingConfig, preprocess_image
from preprocessing.pipeline.tables import Region, bands, crop, detect_table_regions

from .helpers import make_page, rotate_ccw


def ruled_table(height=1100, width=850, top=300, rows=12, row_height=40):
    page = np.full((height, width), 0.95, np.float32)
    cv2.putText(page, "City Hospital  Bill No 42", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0.1, 2)
    left, right = 60, width - 60
    bottom = top + rows * row_height
    for y in range(top, bottom + 1, row_height):
        cv2.line(page, (left, y), (right, y), 0.1, 2)
    for x in (left, left + 60, left + 420, left + 520, right):
        cv2.line(page, (x, top), (x, bottom), 0.1, 2)
    for row in range(rows):
        cv2.putText(page, f"{row + 1}  Item {row}   1   10.00", (left + 8, top + row * row_height + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0.1, 1)
    return page, (left, top, right, bottom)


def covers(region, box, slack=20):
    left, top, right, bottom = box
    return (region.x <= left + slack and region.y <= top + slack and
            region.x + region.width >= right - slack and region.y + region.height >= bottom - slack)


def test_detects_ruled_table_but_not_header_text():
    page, box = ruled_table()
    regions = detect_table_regions(page)
    assert len(regions) == 1 and covers(regions[0], box)
    assert regions[0].y > 150  # the hospital name above the table is not included


def test_detects_table_on_skewed_and_low_resolution_pages():
    page, box = ruled_table()
    skewed = detect_table_regions(rotate_ccw(page, 2.0))
    assert len(skewed) == 1 and covers(skewed[0], box, slack=40)
    small = cv2.resize(page, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    low = detect_table_regions(small, padding=6)
    assert len(low) == 1 and covers(low[0], tuple(v // 2 for v in box), slack=12)


def test_page_without_ruled_table_returns_no_regions():
    assert detect_table_regions(make_page()) == []


def test_bands_overlap_and_cover_the_region():
    region = Region(10, 100, 500, 3000)
    parts = bands(region, max_height=1400, overlap=160)
    assert parts[0].y == 100 and parts[-1].y + parts[-1].height == 3100
    for before, after in zip(parts, parts[1:]):
        assert after.y == before.y + before.height - 160
    assert bands(Region(0, 0, 10, 500), 1400, 160) == [Region(0, 0, 10, 500)]
    with pytest.raises(ValueError):
        bands(region, 100, 100)
    assert crop(np.zeros((50, 40)), Region(5, 10, 20, 30)).shape == (30, 20)


def test_low_contrast_page_is_enhanced_without_losing_a_decimal_point():
    page = np.full((400, 400), 0.62, np.float32)
    for y in (80, 140, 200, 260, 320):
        cv2.putText(page, "Amount 1234", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0.44, 2)
    page[196:199, 330:333] = 0.44  # a 3-pixel decimal point
    config = PreprocessingConfig(denoiser="none", deskew_enabled=False)
    result = preprocess_image(page, config, dpi=300)
    assert "low_contrast" in result.quality_before.flags and result.contrast_enhanced
    assert result.quality_after.contrast > result.quality_before.contrast
    processed = result.processed.astype(np.float32) / 255
    assert processed[196:199, 330:333].mean() < processed[160:170, 330:340].mean() - 0.05
    assert "contrast" in result.timings_ms


def test_clean_page_and_disabled_option_are_left_untouched():
    clean = preprocess_image(make_page(), PreprocessingConfig(denoiser="none", deskew_enabled=False), dpi=300)
    assert not clean.contrast_enhanced
    flat = np.full((300, 300), 0.62, np.float32)
    cv2.putText(flat, "Rs 50.00", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0.44, 2)
    off = PreprocessingConfig(denoiser="none", deskew_enabled=False, contrast_enhancement=False)
    assert not preprocess_image(flat, off, dpi=300).contrast_enhanced
