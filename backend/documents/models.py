import uuid

from django.conf import settings
from django.db import models


def document_upload_path(instance: "Document", filename: str) -> str:
    # The client's filename is never used on disk; it is kept only as metadata.
    return f"documents/{uuid.uuid4().hex}.pdf"


class Document(models.Model):
    """An uploaded PDF. Every processing layer hangs its results off this row."""

    uploaded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="documents",
    )
    original_filename = models.CharField(max_length=255)
    file = models.FileField(upload_to=document_upload_path, max_length=255)
    size_bytes = models.PositiveBigIntegerField()
    sha256 = models.CharField(max_length=64, db_index=True)
    page_count = models.PositiveIntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.original_filename} (#{self.pk})"
