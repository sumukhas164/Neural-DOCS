"""Persist page-level OCR without discarding the server's raw response."""

import json
import logging
from time import monotonic

from django.conf import settings
from django.core.files.base import ContentFile
from django.db import transaction
from django.utils import timezone

from preprocessing.models import PreprocessingJob

from .client import OcrError, recognize
from .models import OcrJob, OcrPage

logger = logging.getLogger(__name__)


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
                    result = recognize(handle.read(), url=settings.NANONETS_BASE_URL,
                                       model=job.model_name, api_key=settings.NANONETS_API_KEY,
                                       timeout=settings.NANONETS_TIMEOUT_SECONDS,
                                       max_tokens=settings.NANONETS_MAX_TOKENS)
                choice = result["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise OcrError(f"Page {page.page_number} OCR was truncated")
                ocr_page = OcrPage(job=job, source_page=page, page_number=page.page_number,
                                   text=choice["message"]["content"],
                                   finish_reason=choice.get("finish_reason") or "")
                ocr_page.raw_response.save(f"page_{page.page_number:04d}.json",
                                           ContentFile(json.dumps(result, ensure_ascii=False).encode("utf-8")), save=False)
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
