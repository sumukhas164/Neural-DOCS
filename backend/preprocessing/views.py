from django.shortcuts import get_object_or_404
from django.http import FileResponse, Http404
from rest_framework import mixins, status, viewsets
from rest_framework.response import Response
from rest_framework.views import APIView

from documents.models import Document

from . import services
from .models import PreprocessingJob, PreprocessedPage
from .serializers import PreprocessingJobSerializer, PreprocessingOptionsSerializer


class PreprocessingJobViewSet(
    mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet
):
    """Status and per-page results of Layer 1 runs. Poll this for progress."""

    serializer_class = PreprocessingJobSerializer

    def get_queryset(self):
        queryset = PreprocessingJob.objects.filter(
            document__uploaded_by=self.request.user
        ).prefetch_related("pages")
        document_id = self.request.query_params.get("document")
        if document_id and document_id.isdigit():
            queryset = queryset.filter(document_id=document_id)
        return queryset


class DocumentPreprocessView(APIView):
    """Re-run Layer 1 on an existing document, optionally with other options."""

    def post(self, request, document_id: int):
        document = get_object_or_404(
            Document, pk=document_id, uploaded_by=request.user
        )
        serializer = PreprocessingOptionsSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        job = services.submit_job(document, serializer.validated_data["options"])
        return Response(
            PreprocessingJobSerializer(job, context={"request": request}).data,
            status=status.HTTP_202_ACCEPTED,
        )


class PreprocessedPageImageView(APIView):
    def get(self, request, page_id: int, variant: str):
        if variant not in ("original", "processed"):
            raise Http404
        page = get_object_or_404(PreprocessedPage, pk=page_id,
                                 job__document__uploaded_by=request.user)
        file = page.original_image if variant == "original" else page.processed_image
        response = FileResponse(file.open("rb"), content_type="image/png")
        response["Cache-Control"] = "private, no-store"
        return response
