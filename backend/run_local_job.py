"""Process one existing job in a separate local process, without Celery."""

import logging
import os
import sys

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django  # noqa: E402

django.setup()

logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in {"ocr", "invoice"}:
        raise SystemExit("Usage: run_local_job.py ocr|invoice <job_id>")
    kind, job_id = sys.argv[1], int(sys.argv[2])
    try:
        if kind == "ocr":
            from ocr.services import run_job
        else:
            from invoices.services import run_job
        job = run_job(job_id)
        logger.info("Local %s job %s finished: %s", kind, job_id, job.status)
        if job.status == "failed":
            raise SystemExit(1)
    except Exception as exc:
        logger.exception("Local %s job %s crashed", kind, job_id)
        from django.utils import timezone
        if kind == "ocr":
            from ocr.models import OcrJob
            model = OcrJob
        else:
            from invoices.models import InvoiceRun
            model = InvoiceRun
        model.objects.filter(pk=job_id).update(
            status="failed", error_message=f"{type(exc).__name__}: {exc}"[:2000],
            finished_at=timezone.now(),
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
