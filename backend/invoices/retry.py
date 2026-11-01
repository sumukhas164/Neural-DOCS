"""Targeted table-region OCR for pages the completeness check questions.

Each problem page is cut into ruled-table regions (or full-width bands when no
ruled table is found), split into overlapping bands, and read at full
resolution with a table-only prompt. Every call is returned for audit,
including failures. The rows read here are only compared with page OCR; they
never replace it.
"""

import logging

import cv2
import numpy as np
from django.conf import settings

from ocr.client import TABLE_PROMPT, ask
from preprocessing.pipeline.tables import Region, bands, crop, detect_table_regions

from .completeness import normalise
from .extraction import extract

logger = logging.getLogger(__name__)


def plan_crops(image):
    """Return (coverage, regions): ruled tables if found, else the whole page."""
    regions = detect_table_regions(image)
    coverage = "tables" if regions else "page"
    if not regions:
        regions = [Region(0, 0, image.shape[1], image.shape[0])]
    height, overlap = settings.INVOICE_RETRY_BAND_HEIGHT, settings.INVOICE_RETRY_BAND_OVERLAP
    return coverage, [band for region in regions for band in bands(region, height, overlap)]


def merge_bands(band_rows):
    """Concatenate rows from overlapping bands, dropping the rows read twice in an overlap."""
    merged = []
    for rows in band_rows:
        merged_keys = [normalise(row["source_text"]) for row in merged]
        keys = [normalise(row["source_text"]) for row in rows]
        overlap = next((size for size in range(min(len(merged), len(rows)), 0, -1)
                        if merged_keys[-size:] == keys[:size]), 0)
        merged.extend(rows[overlap:])
    return merged


def _split_by_band(items):
    """extract() restarts row_order for every (page, text) pair; use that to regroup rows by band."""
    groups, last = [], None
    for row in items:
        if last is None or row["row_order"] <= last:
            groups.append([])
        groups[-1].append(row)
        last = row["row_order"]
    return groups


def read_page(page, model, reasons, max_calls):
    """OCR the table regions of one page. Returns (crop_result, attempts)."""
    image = cv2.imdecode(np.frombuffer(page.source_page.processed_image.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Processed image for page {page.page_number} could not be decoded")
    coverage, regions = plan_crops(image)
    limited = len(regions) > max_calls
    if limited:
        logger.warning("Page %s needs %s crops; only %s OCR calls remain", page.page_number, len(regions), max_calls)
        regions = regions[:max_calls]
    attempts, texts, errors = [], [], []
    for region in regions:
        attempt = {"page_number": page.page_number, "region": region.as_list(), "coverage": coverage,
                   "reasons": reasons, "status": "failed", "text": "", "raw_response": {}, "error": ""}
        ok, png = cv2.imencode(".png", crop(image, region))
        try:
            if not ok:
                raise ValueError("PNG encoding of crop failed")
            raw = ask(png.tobytes(), TABLE_PROMPT, url=settings.NANONETS_BASE_URL, model=model,
                      api_key=settings.NANONETS_API_KEY, timeout=settings.NANONETS_TIMEOUT_SECONDS,
                      max_tokens=settings.NANONETS_MAX_TOKENS)
            choice = raw["choices"][0]
            attempt.update(status="truncated" if choice.get("finish_reason") == "length" else "completed",
                           text=choice["message"]["content"], raw_response=raw)
            texts.append(attempt["text"])
        except Exception as exc:  # recorded and reported; one bad crop must not stop the run
            logger.warning("Targeted OCR failed on page %s region %s: %s", page.page_number, region.as_list(), exc)
            attempt["error"] = f"{type(exc).__name__}: {exc}"[:1000]
            errors.append(attempt["error"])
        attempts.append(attempt)
    rows = extract([(page.page_number, text) for text in texts], method="table_crop_ocr")["items"] if texts else []
    result = {"coverage": coverage, "items": merge_bands(_split_by_band(rows)), "errors": errors,
              "truncated": any(attempt["status"] == "truncated" for attempt in attempts), "limited": limited}
    logger.info("Targeted OCR read %s rows from %s crop(s) on page %s (%s coverage, %s failed)",
                len(result["items"]), len(attempts), page.page_number, coverage, len(errors))
    return result, attempts
