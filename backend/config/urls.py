from django.contrib import admin
from django.urls import include, path

from .views import RuntimeConfigView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api-auth/", include("rest_framework.urls")),
    path("api/runtime/", RuntimeConfigView.as_view()),
    path("api/", include("documents.urls")),
    path("api/", include("preprocessing.urls")),
    path("api/", include("ocr.urls")),
    path("api/", include("invoices.urls")),
]
