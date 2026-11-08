"""Strip-by-strip page OCR: cuts at blank rows, carries table columns, splits truncated strips."""

import json
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.test import override_settings

from documents.models import Document
from ocr.client import STRIP_PROMPT, strip_prompt
from ocr.models import OcrJob
from ocr.services import _header_cells, recognize_strips, run_job
from preprocessing.models import PreprocessedPage, PreprocessingJob
from preprocessing.pipeline.tables import ink_fraction, split_at_gaps

GATEWAY = dict(NANONETS_BASE_URL="http://127.0.0.1:8090/v1", NANONETS_API_KEY="", OCR_STRIP_HEIGHT=1200)


def page_image(height=3000, width=900, lines=60, blank_from=None):
    """White page with evenly spaced text lines; optionally blank below ``blank_from``."""
    image = np.full((height, width), 255, np.uint8)
    step = (blank_from or height) // (lines + 1)
    for line in range(lines):
        y = step * (line + 1)
        cv2.putText(image, f"Item {line:02d}   1   120.00   120.00", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    return image


def png(image):
    return cv2.imencode(".png", image)[1].tobytes()


def reply(text, finish="stop"):
    return {"choices": [{"message": {"content": text}, "finish_reason": finish}]}


def test_split_at_gaps_cuts_only_through_blank_rows_and_covers_the_page():
    image = page_image()
    strips = split_at_gaps(image, 1200)
    assert len(strips) == 3 and all(strip.height <= 1200 for strip in strips)
    assert strips[0].y == 0 and strips[-1].y + strips[-1].height == image.shape[0]
    for before, after in zip(strips, strips[1:]):
        assert after.y == before.y + before.height  # contiguous, no overlap, no gap
        assert np.count_nonzero(image[after.y] < 160) == 0  # the cut row holds no ink


def test_split_at_gaps_prefers_ruling_lines_and_keeps_short_pages_whole():
    image = np.full((1000, 600), 255, np.uint8)
    for y in range(20, 1000, 40):
        cv2.putText(image, "row text 12.00", (10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
        cv2.line(image, (0, y), (599, y), 0, 1)
    strips = split_at_gaps(image, 700)
    assert (strips[1].y - 20) % 40 in (0, 1, 2, 38, 39)  # cut on or right next to a ruling line
    assert split_at_gaps(image, 1000) == split_at_gaps(image, 5000)[:1]
    assert ink_fraction(np.full((10, 10), 255, np.uint8)) == 0.0


def test_header_cells_and_column_prompt():
    text = ("<table><thead><tr><th>Date</th><th>Service Name</th><th>Rate</th></tr></thead>"
            "<tr><td>1</td></tr></table><table><tr><td>a</td><td>b</td></tr></table>")
    assert _header_cells(text) == ["Date", "Service Name", "Rate"]
    assert _header_cells("<table><tr><th></th><th>12</th></tr></table>") is None
    prompt = strip_prompt(["Date", "Service Name", "Rate"])
    assert prompt.startswith(STRIP_PROMPT) and "Date | Service Name | Rate" in prompt and "exactly 3" in prompt
    assert strip_prompt(None) == STRIP_PROMPT


@override_settings(**GATEWAY)
def test_recognize_strips_keeps_order_passes_columns_and_splits_truncated_strips():
    image = page_image(blank_from=2500)  # the bottom of the page is empty
    header = "<table><thead><tr><th>Item</th><th>Qty</th><th>Rate</th></tr></thead><tr><td>A</td></tr></table>"
    calls = []

    def fake_ask(data, prompt, **options):
        calls.append(prompt)
        if len(calls) == 1:
            return reply(header)
        if len(calls) == 2:
            return reply("cut off", "length")  # second strip is too dense: split it
        return reply(f"part {len(calls)}")

    with patch("ocr.services.ask", side_effect=fake_ask):
        result, artifact, columns = recognize_strips(png(image), page_number=4, model="ocr2")
    text = result["choices"][0]["message"]["content"]
    assert text.startswith(header) and "cut off" not in text
    assert text.index("part 3") < text.index("part 4")
    assert result["choices"][0]["finish_reason"] == "stop"
    assert "Item | Qty | Rate" in calls[1] and "Item | Qty | Rate" in calls[2]
    assert columns == ["Item", "Qty", "Rate"]
    attempts = artifact["_ocr_strips"]["attempts"]
    assert any(attempt.get("depth") == 1 for attempt in attempts)
    assert any(attempt.get("response") == reply("cut off", "length") for attempt in attempts)  # nothing discarded


@override_settings(**{**GATEWAY, "OCR_STRIP_HEIGHT": 2400})
def test_strip_still_truncated_at_the_split_limit_is_reported():
    with patch("ocr.services.ask", return_value=reply("dense", "length")) as ask_call:
        result, artifact, _ = recognize_strips(png(page_image(height=2400, lines=40)), page_number=1, model="ocr2")
    assert result["choices"][0]["finish_reason"] == "length"
    assert ask_call.call_count == 1 + 2 + 4  # whole strip, two halves, four quarters, then stop
    attempts = artifact["_ocr_strips"]["attempts"]
    assert len([attempt for attempt in attempts if "response" in attempt]) == 7
    assert all(attempt.get("skipped") == "blank" for attempt in attempts if "response" not in attempt)


@override_settings(**{**GATEWAY, "OCR_STRIP_HEIGHT": 1200})
def test_short_truncated_strip_is_not_split_below_the_minimum_height():
    with patch("ocr.services.ask", return_value=reply("dense", "length")) as ask_call:
        recognize_strips(png(page_image(height=1100, lines=20)), page_number=1, model="ocr2")
    assert ask_call.call_count == 3  # halves of 1100 px are below 2 × MIN_STRIP_HEIGHT


@override_settings(**GATEWAY, OCR_PAGE_MODE="strips")
@pytest.mark.django_db
def test_ocr_job_reads_pages_as_strips_and_carries_columns_across_pages():
    user = get_user_model().objects.create_user("strips", password="pw")
    document = Document.objects.create(uploaded_by=user, original_filename="bill.pdf", size_bytes=10,
                                       sha256="2" * 64, page_count=2, file=ContentFile(b"%PDF-fake", name="bill.pdf"))
    source = PreprocessingJob.objects.create(document=document, status="completed", config={}, pages_total=2)
    for number in (1, 2):
        page = PreprocessedPage(job=source, page_number=number, width=900, height=1000, dpi=300,
                                color_mode="grayscale", denoiser="none", denoise_applied=False, sigma_estimated=0,
                                skew_angle=0, deskew_applied=False, transform=[[1, 0, 0], [0, 1, 0]],
                                quality_before={}, quality_after={})
        page.original_image.save("o.png", ContentFile(png(page_image(height=1000, lines=10))), save=False)
        page.processed_image.save("p.png", ContentFile(png(page_image(height=1000, lines=10))), save=False)
        page.save()
    prompts = []
    answers = iter([reply("<table><tr><th>Service</th><th>Amount</th></tr><tr><td>X</td><td>1.00</td></tr></table>"),
                    reply("<table><tr><td>Y</td><td>2.00</td></tr></table>")])

    def fake_ask(data, prompt, **options):
        prompts.append(prompt)
        return next(answers)

    with patch("ocr.services.ask", side_effect=fake_ask):
        job = run_job(OcrJob.objects.create(preprocessing_job=source, model_name="ocr2").pk)
    assert job.status == "completed" and job.pages_done == 2
    assert "Service | Amount" in prompts[1]  # page 2 continues page 1's table
    second = job.pages.get(page_number=2)
    assert "Y" in second.text and second.finish_reason == "stop"
    with second.raw_response.open("rb") as handle:
        assert json.load(handle)["_ocr_strips"]["attempts"][0]["columns_hint"] == ["Service", "Amount"]
