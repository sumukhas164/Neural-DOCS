"""Bridges the framework-free pipeline to Django: jobs, storage and status."""

from __future__ import annotations

import logging
from typing import Any, Mapping

import cv2
import numpy as np
from django.conf import settings
from django.core.files.base import ContentFile
from django.db import transaction
from django.utils import timezone

from documents.models import Document

from .models import PreprocessedPage, PreprocessingJob
from .pipeline import PageResult, PreprocessingConfig, preprocess_pdf

logger = logging.getLogger(__name__)

_ERROR_MESSAGE_LIMIT = 2000


def get_default_config() -> PreprocessingConfig:
    """Pipeline defaults with the project's settings.PREPROCESSING applied."""
    return PreprocessingConfig.from_dict(getattr(settings, "PREPROCESSING", {}))


def create_job(
    document: Document, overrides: Mapping[str, Any] | None = None
) -> PreprocessingJob:
    """Create a pending job. Raises ConfigError if ``overrides`` are invalid."""
    config = get_default_config().with_overrides(overrides)
    return PreprocessingJob.objects.create(
        document=document,
        config=config.to_dict(),
        pages_total=document.page_count,
    )


def submit_job(
    document: Document, overrides: Mapping[str, Any] | None = None
) -> PreprocessingJob:
    """Create a job and hand it to Celery once the transaction has committed."""
    from .tasks import preprocess_document_task

    job = create_job(document, overrides)
    transaction.on_commit(lambda: preprocess_document_task.delay(job.pk))
    return job


def _encode_png(image: np.ndarray) -> bytes:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buffer.tobytes()


def _save_page(job: PreprocessingJob, result: PageResult) -> PreprocessedPage:
    stem = f"page_{result.page_number:04d}"
    page = PreprocessedPage(
        job=job,
        page_number=result.page_number,
        width=result.processed.shape[1],
        height=result.processed.shape[0],
        dpi=result.dpi,
        color_mode=result.color_mode,
        denoiser=result.denoiser,
        denoise_applied=result.denoise_applied,
        denoise_skip_reason=result.denoise_skip_reason or "",
        sigma_estimated=result.sigma_estimated,
        sigma_used=result.sigma_used,
        skew_angle=result.skew_angle,
        deskew_applied=result.deskew_applied,
        deskew_skip_reason=result.deskew_skip_reason or "",
        crop_box=list(result.crop_box) if result.crop_box else None,
        transform=result.transform,
        quality_before=result.quality_before.to_dict(),
        quality_after=result.quality_after.to_dict(),
        quality_flags=list(result.quality_before.flags),
        timings_ms=result.timings_ms,
    )
    page.original_image.save(
        f"{stem}_original.png", ContentFile(_encode_png(result.original)), save=False
    )
    page.processed_image.save(
        f"{stem}_processed.png", ContentFile(_encode_png(result.processed)), save=False
    )
    page.save()
    return page


def _delete_pages(job: PreprocessingJob) -> None:
    for page in job.pages.all():
        page.original_image.delete(save=False)
        page.processed_image.delete(save=False)
    job.pages.all().delete()


def run_job(job_id: int) -> PreprocessingJob:
    """Execute a pending job. Safe to call twice: only one caller proceeds."""
    claimed = PreprocessingJob.objects.filter(
        pk=job_id, status=PreprocessingJob.Status.PENDING
    ).update(status=PreprocessingJob.Status.RUNNING, started_at=timezone.now())
    job = PreprocessingJob.objects.select_related("document").get(pk=job_id)
    if not claimed:
        logger.warning("job %s is %s, not pending; skipping", job_id, job.status)
        return job

    try:
        config = PreprocessingConfig.from_dict(job.config)
        _delete_pages(job)
        # Reading through the storage API keeps this working when uploads live
        # in object storage rather than on the local disk.
        with job.document.file.open("rb") as handle:
            pdf_bytes = handle.read()

        for result in preprocess_pdf(pdf_bytes, config):
            _save_page(job, result)
            job.pages_done = result.page_number
            job.save(update_fields=["pages_done"])

        job.status = PreprocessingJob.Status.COMPLETED
    except Exception as exc:  # the job row must record any failure
        logger.exception("preprocessing job %s failed", job_id)
        job.status = PreprocessingJob.Status.FAILED
        job.error_message = f"{type(exc).__name__}: {exc}"[:_ERROR_MESSAGE_LIMIT]

    job.finished_at = timezone.now()
    job.save(update_fields=["status", "error_message", "finished_at", "pages_done"])
    return job
