from django.core.management.base import BaseCommand, CommandError

from invoices.models import InvoiceRun
from invoices.services import run_job
from ocr.models import OcrJob


class Command(BaseCommand):
    help = "Extract and validate a completed Nanonets OCR job (foreground)."

    def add_arguments(self, parser):
        parser.add_argument("ocr_job_id", type=int)

    def handle(self, *args, **options):
        ocr = OcrJob.objects.filter(pk=options["ocr_job_id"]).first()
        if ocr is None or ocr.status != OcrJob.Status.COMPLETED:
            raise CommandError("A completed OCR job is required")
        run = run_job(InvoiceRun.objects.create(ocr_job=ocr).pk)
        self.stdout.write(f"Invoice run {run.pk}: {run.status}")
        if run.status == InvoiceRun.Status.FAILED:
            raise CommandError(run.error_message)
