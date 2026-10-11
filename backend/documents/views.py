from rest_framework import mixins, status, viewsets
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response

from preprocessing import services as preprocessing_services
from preprocessing.serializers import PreprocessingJobSerializer

from .models import Document
from .serializers import DocumentSerializer


class DocumentViewSet(
    mixins.CreateModelMixin,
    mixins.RetrieveModelMixin,
    mixins.ListModelMixin,
    viewsets.GenericViewSet,
):
    """Upload a PDF and read back documents. Users only see their own."""

    serializer_class = DocumentSerializer
    parser_classes = [MultiPartParser, FormParser]

    def get_queryset(self):
        return Document.objects.filter(uploaded_by=self.request.user)

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        document = serializer.save(uploaded_by=request.user)
        # An upload starts Layer 1 straight away. When the orchestration layer
        # exists it takes over this hand-off and chains the later layers.
        job = preprocessing_services.submit_job(document)
        body = self.get_serializer(document).data
        body["preprocessing_job"] = PreprocessingJobSerializer(
            job, context=self.get_serializer_context()
        ).data
        return Response(body, status=status.HTTP_201_CREATED)
