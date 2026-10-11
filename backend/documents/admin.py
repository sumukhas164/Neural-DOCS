from django.contrib import admin

from .models import Document


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ["id", "original_filename", "uploaded_by", "page_count", "size_bytes", "created_at"]
    search_fields = ["original_filename", "sha256"]
    readonly_fields = ["size_bytes", "sha256", "page_count", "created_at"]
