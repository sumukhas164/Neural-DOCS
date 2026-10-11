from rest_framework import serializers

from .models import PreprocessedPage, PreprocessingJob
from .pipeline import ConfigError
from .services import get_default_config


class PreprocessedPageSerializer(serializers.ModelSerializer):
    original_image = serializers.SerializerMethodField()
    processed_image = serializers.SerializerMethodField()

    def get_original_image(self, page):
        return self.context["request"].build_absolute_uri(f"/api/preprocessing/pages/{page.pk}/original.png")

    def get_processed_image(self, page):
        return self.context["request"].build_absolute_uri(f"/api/preprocessing/pages/{page.pk}/processed.png")
    class Meta:
        model = PreprocessedPage
        fields = [
            "id",
            "page_number",
            "original_image",
            "processed_image",
            "width",
            "height",
            "dpi",
            "color_mode",
            "denoiser",
            "denoise_applied",
            "denoise_skip_reason",
            "sigma_estimated",
            "sigma_used",
            "skew_angle",
            "deskew_applied",
            "deskew_skip_reason",
            "crop_box",
            "transform",
            "quality_before",
            "quality_after",
            "quality_flags",
            "timings_ms",
        ]


class PreprocessingJobSerializer(serializers.ModelSerializer):
    pages = PreprocessedPageSerializer(many=True, read_only=True)

    class Meta:
        model = PreprocessingJob
        fields = [
            "id",
            "document",
            "status",
            "config",
            "pages_total",
            "pages_done",
            "error_message",
            "created_at",
            "started_at",
            "finished_at",
            "pages",
        ]


class PreprocessingOptionsSerializer(serializers.Serializer):
    """Per-run overrides a client may send when re-running Layer 1."""

    # Only options that are safe to expose; resource limits such as tile size,
    # device and page caps stay under server control.
    ALLOWED = (
        "denoiser",
        "sigma_mode",
        "fixed_sigma",
        "deskew_enabled",
        "crop_to_content",
    )

    options = serializers.DictField(required=False, default=dict)

    def validate_options(self, options):
        unknown = set(options) - set(self.ALLOWED)
        if unknown:
            raise serializers.ValidationError(
                f"Unsupported options: {sorted(unknown)}. Allowed: {list(self.ALLOWED)}."
            )
        try:
            get_default_config().with_overrides(options)
        except (ConfigError, TypeError) as exc:
            raise serializers.ValidationError(str(exc))
        return options
