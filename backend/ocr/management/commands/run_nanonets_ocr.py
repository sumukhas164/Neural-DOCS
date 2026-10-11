from django.core.management.base import BaseCommand, CommandError

from ocr.models import OcrJob
from ocr.services import run_job
from preprocessing.models import PreprocessingJob


class Command(BaseCommand):
    help = "Run Nanonets OCR on a completed preprocessing job (foreground)."

    def add_arguments(self, parser):
        parser.add_argument("preprocessing_job_id", type=int)

    def handle(self, *args, **options):
        source = PreprocessingJob.objects.filter(pk=options["preprocessing_job_id"]).first()
        if source is None or source.status != PreprocessingJob.Status.COMPLETED:
            raise CommandError("A completed preprocessing job is required")
        from django.conf import settings
        job = OcrJob.objects.create(preprocessing_job=source, model_name=settings.NANONETS_MODEL)
        job = run_job(job.pk)
        self.stdout.write(f"OCR job {job.pk}: {job.status}, {job.pages_done} pages")
        if job.status != OcrJob.Status.COMPLETED:
            raise CommandError(job.error_message)
