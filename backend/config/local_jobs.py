"""Run long OCR and invoice jobs outside the development web process."""

import logging
import subprocess
import sys

from django.conf import settings
from django.utils import timezone


logger = logging.getLogger(__name__)


def launch(kind, job_id):
    if kind not in {"ocr", "invoice"}:
        raise ValueError("Unsupported local job type")
    log_path = settings.LOCAL_LOG_DIR / f"{kind}-{job_id}.log"
    command = [sys.executable, str(settings.BASE_DIR / "run_local_job.py"), kind, str(job_id)]
    try:
        with log_path.open("ab") as output:
            process = subprocess.Popen(
                command, cwd=settings.BASE_DIR, stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        logger.info("Started local %s job %s as process %s; log: %s", kind, job_id, process.pid, log_path)
    except OSError as exc:
        logger.exception("Could not start local %s job %s", kind, job_id)
        if kind == "ocr":
            from ocr.models import OcrJob
            model = OcrJob
        else:
            from invoices.models import InvoiceRun
            model = InvoiceRun
        model.objects.filter(pk=job_id).update(
            status="failed", error_message=f"Local job could not start: {exc}"[:2000],
            finished_at=timezone.now(),
        )
