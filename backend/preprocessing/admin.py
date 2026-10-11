from django.contrib import admin

from .models import PreprocessedPage, PreprocessingJob


class PreprocessedPageInline(admin.TabularInline):
    model = PreprocessedPage
    extra = 0
    can_delete = False
    fields = [
        "page_number",
        "color_mode",
        "denoise_applied",
        "sigma_estimated",
        "skew_angle",
        "quality_flags",
        "processed_image",
    ]
    readonly_fields = fields


@admin.register(PreprocessingJob)
class PreprocessingJobAdmin(admin.ModelAdmin):
    list_display = ["id", "document", "status", "pages_done", "pages_total", "created_at", "finished_at"]
    list_filter = ["status"]
    readonly_fields = ["config", "created_at", "started_at", "finished_at"]
    inlines = [PreprocessedPageInline]
