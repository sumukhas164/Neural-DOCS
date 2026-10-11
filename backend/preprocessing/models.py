from django.db import models

from documents.models import Document


class PreprocessingJob(models.Model):
    """One Layer 1 run over a document, with the exact configuration it used."""

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    document = models.ForeignKey(
        Document, on_delete=models.CASCADE, related_name="preprocessing_jobs"
    )
    status = models.CharField(
        max_length=16, choices=Status.choices, default=Status.PENDING, db_index=True
    )
    # Snapshot of PreprocessingConfig, so results stay reproducible after the
    # defaults change.
    config = models.JSONField()
    pages_total = models.PositiveIntegerField(default=0)
    pages_done = models.PositiveIntegerField(default=0)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"PreprocessingJob #{self.pk} ({self.status})"


def page_image_path(instance: "PreprocessedPage", filename: str) -> str:
    return f"preprocessing/job_{instance.job_id}/{filename}"


class PreprocessedPage(models.Model):
    """The Layer 1 output for one page: the cleaned image plus how it was made.

    This is the hand-off to Layer 2 (OCR), which reads ``processed_image``.
    """

    job = models.ForeignKey(
        PreprocessingJob, on_delete=models.CASCADE, related_name="pages"
    )
    page_number = models.PositiveIntegerField()  # 1-based

    original_image = models.FileField(upload_to=page_image_path, max_length=255)
    processed_image = models.FileField(upload_to=page_image_path, max_length=255)
    width = models.PositiveIntegerField()  # of processed_image, in pixels
    height = models.PositiveIntegerField()
    dpi = models.FloatField()
    color_mode = models.CharField(max_length=16)  # "grayscale" | "color"

    denoiser = models.CharField(max_length=16)
    denoise_applied = models.BooleanField()
    denoise_skip_reason = models.CharField(max_length=64, blank=True)
    sigma_estimated = models.FloatField()
    sigma_used = models.FloatField(null=True, blank=True)

    skew_angle = models.FloatField()  # degrees, counter-clockwise positive
    deskew_applied = models.BooleanField()
    deskew_skip_reason = models.CharField(max_length=64, blank=True)
    crop_box = models.JSONField(null=True, blank=True)  # [x, y, w, h]
    # 2x3 affine matrix mapping original-image pixels to processed-image
    # pixels; invert it to project OCR boxes back onto the original page.
    transform = models.JSONField()

    quality_before = models.JSONField()
    quality_after = models.JSONField()
    # Copy of quality_before["flags"], denormalised for filtering.
    quality_flags = models.JSONField(default=list)
    timings_ms = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["job_id", "page_number"]
        constraints = [
            models.UniqueConstraint(
                fields=["job", "page_number"], name="unique_page_per_job"
            )
        ]

    def __str__(self) -> str:
        return f"Job #{self.job_id} page {self.page_number}"
