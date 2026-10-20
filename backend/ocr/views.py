from django.shortcuts import get_object_or_404
from django.conf import settings
from rest_framework import mixins, status, viewsets
from rest_framework.response import Response
from rest_framework.views import APIView

from preprocessing.models import PreprocessingJob

from .models import OcrJob
from .serializers import OcrJobSerializer
from .services import submit_job


class OcrJobViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    serializer_class = OcrJobSerializer

    def get_queryset(self):
        return OcrJob.objects.filter(preprocessing_job__document__uploaded_by=self.request.user).prefetch_related("pages")


class StartOcrView(APIView):
    def post(self, request, preprocessing_job_id):
        source = get_object_or_404(PreprocessingJob, pk=preprocessing_job_id,
                                   document__uploaded_by=request.user)
        if source.status != PreprocessingJob.Status.COMPLETED:
            return Response({"detail": "Preprocessing must complete before OCR"}, status=409)
        if settings.CELERY_TASK_ALWAYS_EAGER and not settings.LOCAL_DEMO_MODE:
            return Response({"detail": "Configure a Celery broker and worker for API OCR; use the run_nanonets_ocr management command for local runs."}, status=503)
        job = submit_job(source)
        return Response(OcrJobSerializer(job, context={"request": request}).data,
                        status=status.HTTP_202_ACCEPTED)
