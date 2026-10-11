import numpy as np
import pytest

from preprocessing.pipeline.pdf_renderer import PdfRenderError, count_pages, render_pdf

from .helpers import make_page, make_pdf_bytes


@pytest.fixture
def pdf_bytes():
    # 480x640 px at 100 dpi -> a 4.8 x 6.4 inch page.
    return make_pdf_bytes([make_page(), make_page(color=True), make_page()], dpi=100)


def test_count_pages(pdf_bytes, tmp_path):
    assert count_pages(pdf_bytes) == 3
    path = tmp_path / "doc.pdf"
    path.write_bytes(pdf_bytes)
    assert count_pages(path) == 3
    assert count_pages(str(path)) == 3


def test_renders_rgb_pages_at_requested_dpi(pdf_bytes):
    pages = list(render_pdf(pdf_bytes, dpi=200))
    assert [p.page_number for p in pages] == [1, 2, 3]
    for page in pages:
        assert page.image.dtype == np.uint8
        height, width, channels = page.image.shape
        # pdfium rounds page sizes up, so a side may come out one pixel larger.
        assert height in (1280, 1281) and width in (960, 961) and channels == 3
        assert page.dpi == pytest.approx(200)


def test_rendered_content_and_channel_order(pdf_bytes):
    colour_page = list(render_pdf(pdf_bytes, dpi=100))[1].image
    # The synthetic logo block is red: R must dominate (i.e. not BGR).
    r, g, b = colour_page[84, 100].astype(int)
    assert r > 150 and g < 90 and b < 90
    # Paper-tone background is preserved.
    assert abs(int(colour_page[5, 5, 0]) - round(0.92 * 255)) <= 6


def test_max_pages(pdf_bytes):
    assert len(list(render_pdf(pdf_bytes, dpi=72, max_pages=2))) == 2


def test_max_page_pixels_lowers_the_dpi(pdf_bytes):
    page = next(render_pdf(pdf_bytes, dpi=300, max_page_pixels=1_000_000))
    height, width = page.image.shape[:2]
    assert height * width <= 1_010_000
    assert page.dpi < 300
    assert width / height == pytest.approx(480 / 640, rel=0.01)


def test_invalid_input_raises():
    with pytest.raises(PdfRenderError):
        list(render_pdf(b"this is not a pdf"))
    with pytest.raises(PdfRenderError):
        count_pages("does-not-exist.pdf")
