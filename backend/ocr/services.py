"""Persist page-level OCR without discarding the server's raw response."""

import json
import logging
from io import BytesIO
from time import monotonic

from django.conf import settings
from django.core.files.base import ContentFile
from django.db import transaction
from django.utils import timezone
from PIL import Image

from preprocessing.models import PreprocessingJob

from .client import OcrError, recognize
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
    """Retry a context-truncated page at lower resolution; retain every response."""
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
    raise OcrError(f"Page {page_number} OCR was truncated after lower-resolution retries")


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
        for page in job.preprocessing_job.pages.order_by("page_number"):
            page_started = monotonic()
            logger.info("OCR job %s processing page %s of %s", job_id, page.page_number,
                        job.preprocessing_job.pages_total)
            if "blank_page" in page.quality_flags:
                OcrPage.objects.create(job=job, source_page=page, page_number=page.page_number)
            else:
                with page.processed_image.open("rb") as handle:
                    result, artifact = _recognize_complete(
                        handle.read(), page_number=page.page_number, model=job.model_name)
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
