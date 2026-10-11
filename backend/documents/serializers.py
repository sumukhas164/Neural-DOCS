import os

from rest_framework import serializers

from .models import Document
from .validators import validate_pdf_upload


class DocumentSerializer(serializers.ModelSerializer):
    file = serializers.FileField(write_only=True)

    class Meta:
        model = Document
        fields = [
            "id",
            "file",
            "original_filename",
            "size_bytes",
            "sha256",
            "page_count",
            "created_at",
        ]
        read_only_fields = [
            "original_filename",
            "size_bytes",
            "sha256",
            "page_count",
            "created_at",
        ]

    def validate_file(self, upload):
        self._pdf_info = validate_pdf_upload(upload)
        return upload

    def create(self, validated_data):
        upload = validated_data["file"]
        info = self._pdf_info
        return Document.objects.create(
            uploaded_by=validated_data["uploaded_by"],
            original_filename=os.path.basename(upload.name)[:255],
            file=upload,
            size_bytes=info.size_bytes,
            sha256=info.sha256,
            page_count=info.page_count,
        )
