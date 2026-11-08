"""Persist page-level OCR without discarding the server's raw response."""

import html
import json
import logging
import re
from io import BytesIO
from time import monotonic

from django.conf import settings
from django.core.files.base import ContentFile
from django.db import transaction
from django.utils import timezone
import cv2
import numpy as np
from PIL import Image

from preprocessing.models import PreprocessingJob
from preprocessing.pipeline.tables import Region, crop, ink_fraction, split_at_gaps

from .client import OcrError, ask, recognize, strip_prompt
from .models import OcrJob, OcrPage

logger = logging.getLogger(__name__)
OCR_RETRY_MAX_EDGES = (2600, 2200)


def _downscaled_png(image: bytes, max_edge: int) -> bytes | None:
    """Reduce image tokens while keeping the complete page and its row order."""
    with Image.open(BytesIO(image)) as source:
        if max(source.size) <= max_edge:
            return None
        reduced = source.copy()
    reduced.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    output = BytesIO()
    reduced.save(output, format="PNG")
    return output.getvalue()


def _recognize_complete(image: bytes, *, page_number: int, model: str) -> tuple[dict, dict]:
    """Retry a context-truncated page at lower resolution; retain every response.

    If every attempt is truncated, the longest output is kept with
    finish_reason "length" instead of failing the job: the invoice workflow's
    completeness check then re-reads the page's table regions at full resolution.
    """
    options = dict(url=settings.NANONETS_BASE_URL, model=model,
                   api_key=settings.NANONETS_API_KEY,
                   timeout=settings.NANONETS_TIMEOUT_SECONDS,
                   max_tokens=settings.NANONETS_MAX_TOKENS)
    result = recognize(image, **options)
    if result["choices"][0].get("finish_reason") != "length":
        return result, result

    earlier = [{"image_max_edge": None, "response": result}]
    for max_edge in OCR_RETRY_MAX_EDGES:
        reduced = _downscaled_png(image, max_edge)
        if reduced is None:
            continue
        logger.warning("OCR page %s was truncated; retrying at max edge %s pixels",
                       page_number, max_edge)
        result = recognize(reduced, **options)
        if result["choices"][0].get("finish_reason") != "length":
            artifact = {**result, "_ocr_retry": {
                "selected_image_max_edge": max_edge,
                "earlier_attempts": earlier,
            }}
            return result, artifact
        earlier.append({"image_max_edge": max_edge, "response": result})
    logger.error("OCR page %s is still truncated after lower-resolution retries; keeping the "
                 "longest output for targeted table OCR and human review", page_number)
    best = max(earlier, key=lambda attempt: len(attempt["response"]["choices"][0]["message"]["content"]))
    artifact = {**best["response"], "_ocr_retry": {
        "selected_image_max_edge": best["image_max_edge"],
        "all_truncated": True,
        "earlier_attempts": earlier,
    }}
    return best["response"], artifact


MIN_STRIP_HEIGHT = 300  # px; below this a truncated strip is not split further
MAX_STRIP_SPLITS = 2
BLANK_STRIP_INK = 0.0005  # fraction of dark pixels; emptier strips are not sent


def _header_cells(text):
    """Headings of the last table in ``text`` whose first row is a printed <th> header."""
    columns = None
    for table in re.findall(r"<table\b.*?(?:</table>|$)", text, flags=re.I | re.S):
        first = re.search(r"<tr\b.*?</tr>", table, flags=re.I | re.S)
        if not first or "<th" not in first.group().lower():
            continue
        cells = [" ".join(html.unescape(re.sub(r"<[^>]+>", " ", cell)).split())
                 for cell in re.findall(r"<t[hd]\b[^>]*>(.*?)</t[hd]>", first.group(), flags=re.I | re.S)]
        if sum(bool(re.search(r"[A-Za-z]{2}", cell)) for cell in cells) >= 2:
            columns = cells
    return columns


def _encode_png(image):
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise OcrError("PNG encoding of page strip failed")
    return buffer.tobytes()


def recognize_strips(image: bytes, *, page_number: int, model: str, columns=None):
    """OCR a page as full-resolution horizontal strips cut at blank rows.

    A dense page read whole exceeds the model's output budget and was being
    downscaled, which loses rows. Strips keep full resolution and stay well
    under the budget. A strip that is still truncated is split again, at most
    MAX_STRIP_SPLITS times. Returns (combined response, raw artifact, columns
    of the table that may continue onto the next page).
    """
    pixels = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_UNCHANGED)
    if pixels is None:
        raise OcrError(f"Processed image for page {page_number} could not be decoded")
    options = dict(url=settings.NANONETS_BASE_URL, model=model, api_key=settings.NANONETS_API_KEY,
                   timeout=settings.NANONETS_TIMEOUT_SECONDS, max_tokens=settings.NANONETS_MAX_TOKENS)
    texts, attempts, truncated = [], [], False

    def read(region: Region, depth: int):
        nonlocal columns, truncated
        part = crop(pixels, region)
        if ink_fraction(part) < BLANK_STRIP_INK:
            attempts.append({"region": region.as_list(), "skipped": "blank"})
            return
        result = ask(_encode_png(part), strip_prompt(columns), **options)
        attempts.append({"region": region.as_list(), "depth": depth, "columns_hint": columns, "response": result})
        choice = result["choices"][0]
        if choice.get("finish_reason") == "length" and depth < MAX_STRIP_SPLITS \
                and region.height >= 2 * MIN_STRIP_HEIGHT:
            logger.warning("OCR page %s strip %s was truncated; splitting it", page_number, region.as_list())
            for sub in split_at_gaps(part, region.height // 2 + 1):
                read(Region(region.x + sub.x, region.y + sub.y, sub.width, sub.height), depth + 1)
            return
        if choice.get("finish_reason") == "length":
            truncated = True
            logger.error("OCR page %s strip %s is still truncated; keeping its output for review",
                         page_number, region.as_list())
        texts.append(choice["message"]["content"])
        columns = _header_cells(choice["message"]["content"]) or columns

    for region in split_at_gaps(pixels, settings.OCR_STRIP_HEIGHT):
        read(region, 0)
    combined = {"model": model, "choices": [{"index": 0, "message": {"role": "assistant", "content": "\n\n".join(texts)},
                                             "finish_reason": "length" if truncated else "stop"}]}
    artifact = {**combined, "_ocr_strips": {"strip_height": settings.OCR_STRIP_HEIGHT, "attempts": attempts}}
    return combined, artifact, columns


def submit_job(preprocessing_job):
    if preprocessing_job.status != PreprocessingJob.Status.COMPLETED:
        raise ValueError("Preprocessing must complete before OCR")
    job = OcrJob.objects.create(preprocessing_job=preprocessing_job, model_name=settings.NANONETS_MODEL)
    if settings.CELERY_TASK_ALWAYS_EAGER and settings.LOCAL_DEMO_MODE:
        from config.local_jobs import launch
        transaction.on_commit(lambda: launch("ocr", job.pk))
    else:
        from .tasks import run_ocr_task
        transaction.on_commit(lambda: run_ocr_task.delay(job.pk))
    return job


def run_job(job_id):
    claimed = OcrJob.objects.filter(pk=job_id, status=OcrJob.Status.PENDING).update(
        status=OcrJob.Status.RUNNING, started_at=timezone.now())
    job = OcrJob.objects.select_related("preprocessing_job").get(pk=job_id)
    if not claimed:
        return job
    logger.info("OCR job %s started with model %s", job_id, job.model_name)
    try:
        if job.preprocessing_job.status != PreprocessingJob.Status.COMPLETED:
            raise OcrError("Source preprocessing job is not completed")
        columns = None  # headings of a table that may continue onto the next page
        for page in job.preprocessing_job.pages.order_by("page_number"):
            page_started = monotonic()
            logger.info("OCR job %s processing page %s of %s", job_id, page.page_number,
                        job.preprocessing_job.pages_total)
            if "blank_page" in page.quality_flags:
                OcrPage.objects.create(job=job, source_page=page, page_number=page.page_number)
            else:
                with page.processed_image.open("rb") as handle:
                    image = handle.read()
                if settings.OCR_PAGE_MODE == "strips":
                    result, artifact, columns = recognize_strips(
                        image, page_number=page.page_number, model=job.model_name, columns=columns)
                else:
                    result, artifact = _recognize_complete(image, page_number=page.page_number,
                                                           model=job.model_name)
                choice = result["choices"][0]
                ocr_page = OcrPage(job=job, source_page=page, page_number=page.page_number,
                                   text=choice["message"]["content"],
                                   finish_reason=choice.get("finish_reason") or "")
                ocr_page.raw_response.save(f"page_{page.page_number:04d}.json",
                                           ContentFile(json.dumps(artifact, ensure_ascii=False).encode("utf-8")), save=False)
                ocr_page.save()
            job.pages_done += 1
            job.save(update_fields=["pages_done"])
            logger.info("OCR job %s completed page %s in %.1f seconds", job_id, page.page_number,
                        monotonic() - page_started)
        job.status = OcrJob.Status.COMPLETED
    except Exception as exc:
        logger.exception("OCR job %s failed", job_id)
        job.status = OcrJob.Status.FAILED
        job.error_message = f"{type(exc).__name__}: {exc}"[:2000]
    job.finished_at = timezone.now()
    job.save(update_fields=["status", "error_message", "finished_at"])
    logger.info("OCR job %s finished with status %s (%s pages)", job_id, job.status, job.pages_done)
    return job
