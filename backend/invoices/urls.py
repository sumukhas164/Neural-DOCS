from django.urls import path
from rest_framework.routers import DefaultRouter

from .views import (CandidateRowView, CorrectItemView, DecideInvoiceView, InvoicePageImageView,
                    InvoiceRunViewSet, ResolveFindingView, StartInvoiceView)

router = DefaultRouter()
router.register("invoices/runs", InvoiceRunViewSet, basename="invoice-run")

urlpatterns = [
    path("ocr/jobs/<int:ocr_job_id>/extract/", StartInvoiceView.as_view()),
    path("invoices/runs/<int:run_id>/items/<str:stable_id>/", CorrectItemView.as_view()),
    path("invoices/runs/<int:run_id>/decision/", DecideInvoiceView.as_view()),
    path("invoices/runs/<int:run_id>/candidates/<str:candidate_id>/", CandidateRowView.as_view()),
    path("invoices/runs/<int:run_id>/findings/<int:finding_id>/resolve/", ResolveFindingView.as_view()),
    path("invoices/runs/<int:run_id>/pages/<int:page_number>/image/", InvoicePageImageView.as_view()),
    *router.urls,
]
