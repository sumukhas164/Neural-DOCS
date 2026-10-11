from rest_framework import serializers

from .models import OcrJob, OcrPage


class OcrPageSerializer(serializers.ModelSerializer):
    class Meta:
        model = OcrPage
        fields = ["page_number", "text", "finish_reason"]


class OcrJobSerializer(serializers.ModelSerializer):
    pages = OcrPageSerializer(many=True, read_only=True)

    class Meta:
        model = OcrJob
        fields = ["id", "preprocessing_job", "status", "model_name", "pages_done",
                  "error_message", "created_at", "started_at", "finished_at", "pages"]
