from django.db import models

from preprocessing.models import PreprocessedPage, PreprocessingJob


class OcrJob(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    preprocessing_job = models.ForeignKey(PreprocessingJob, on_delete=models.CASCADE, related_name="ocr_jobs")
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.PENDING, db_index=True)
    model_name = models.CharField(max_length=128)
    pages_done = models.PositiveIntegerField(default=0)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)


def artifact_path(instance, filename):
    return f"ocr/job_{instance.job_id}/{filename}"


class OcrPage(models.Model):
    job = models.ForeignKey(OcrJob, on_delete=models.CASCADE, related_name="pages")
    source_page = models.ForeignKey(PreprocessedPage, on_delete=models.CASCADE)
    page_number = models.PositiveIntegerField()
    text = models.TextField(blank=True)
    raw_response = models.FileField(upload_to=artifact_path, max_length=255, blank=True)
    finish_reason = models.CharField(max_length=64, blank=True)

    class Meta:
        ordering = ["page_number"]
        constraints = [models.UniqueConstraint(fields=["job", "page_number"], name="unique_ocr_page_per_job")]
