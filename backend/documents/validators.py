"""Upload validation: a file must really be a readable PDF within the limits."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from django.conf import settings
from rest_framework import serializers

from preprocessing.pipeline.pdf_renderer import PdfRenderError, count_pages

_PDF_MAGIC = b"%PDF-"
# The PDF specification allows the header anywhere in the first 1024 bytes.
_HEADER_SEARCH_BYTES = 1024


@dataclass(frozen=True)
class PdfInfo:
    size_bytes: int
    sha256: str
    page_count: int


def validate_pdf_upload(upload) -> PdfInfo:
    """Check an uploaded file and return its facts, or raise ValidationError."""
    if not upload.name.lower().endswith(".pdf"):
        raise serializers.ValidationError("Only .pdf files are accepted.")
    if upload.size == 0:
        raise serializers.ValidationError("The file is empty.")
    if upload.size > settings.MAX_UPLOAD_SIZE_BYTES:
        limit_mb = settings.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
        raise serializers.ValidationError(f"The file exceeds the {limit_mb} MB limit.")

    upload.seek(0)
    data = upload.read()
    upload.seek(0)

    if _PDF_MAGIC not in data[:_HEADER_SEARCH_BYTES]:
        raise serializers.ValidationError("The file is not a PDF.")
    try:
        page_count = count_pages(data)
    except PdfRenderError:
        raise serializers.ValidationError(
            "The PDF could not be opened. It may be corrupted or password-protected."
        )
    if page_count == 0:
        raise serializers.ValidationError("The PDF has no pages.")
    if page_count > settings.MAX_DOCUMENT_PAGES:
        raise serializers.ValidationError(
            f"The PDF has {page_count} pages; the limit is {settings.MAX_DOCUMENT_PAGES}."
        )

    return PdfInfo(
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        page_count=page_count,
    )
