from django.urls import path
from rest_framework.routers import DefaultRouter

from .views import DocumentPreprocessView, PreprocessingJobViewSet, PreprocessedPageImageView

router = DefaultRouter()
router.register("preprocessing/jobs", PreprocessingJobViewSet, basename="preprocessing-job")

urlpatterns = [
    path("preprocessing/pages/<int:page_id>/<str:variant>.png", PreprocessedPageImageView.as_view()),
    path(
        "documents/<int:document_id>/preprocess/",
        DocumentPreprocessView.as_view(),
        name="document-preprocess",
    ),
    *router.urls,
]
