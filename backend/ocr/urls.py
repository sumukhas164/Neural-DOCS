from django.urls import path
from rest_framework.routers import DefaultRouter

from .views import OcrJobViewSet, StartOcrView

router = DefaultRouter()
router.register("ocr/jobs", OcrJobViewSet, basename="ocr-job")

urlpatterns = [path("preprocessing/jobs/<int:preprocessing_job_id>/ocr/", StartOcrView.as_view()), *router.urls]
