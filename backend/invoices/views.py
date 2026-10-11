from django.db import transaction
from django.conf import settings
from django.http import FileResponse
from django.shortcuts import get_object_or_404
from rest_framework import mixins, status, viewsets
from rest_framework.response import Response
from rest_framework.views import APIView

from ocr.models import OcrJob

from .models import BillingGroup, InvoiceRun, LineItem, ReviewEvent, ValidationFinding
from .serializers import InvoiceRunSerializer, ItemCorrectionSerializer
from .services import submit_run
from .validation import validate


def owned_runs(user):
    return InvoiceRun.objects.filter(ocr_job__preprocessing_job__document__uploaded_by=user)


class InvoiceRunViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    serializer_class = InvoiceRunSerializer

    def get_queryset(self):
        return owned_runs(self.request.user).prefetch_related("groups", "items", "findings",
                                                             "judge_findings", "review_events", "stage_events")


class StartInvoiceView(APIView):
    def post(self, request, ocr_job_id):
        job = get_object_or_404(OcrJob, pk=ocr_job_id,
                                preprocessing_job__document__uploaded_by=request.user)
        if job.status != OcrJob.Status.COMPLETED:
            return Response({"detail": "OCR must complete before extraction"}, status=409)
        if settings.CELERY_TASK_ALWAYS_EAGER:
            return Response({"detail": "Configure a Celery broker and worker for API extraction; use the process_invoice management command locally."}, status=503)
        run = submit_run(job)
        return Response(InvoiceRunSerializer(run).data, status=status.HTTP_202_ACCEPTED)


class CorrectItemView(APIView):
    @transaction.atomic
    def patch(self, request, run_id, stable_id):
        run = get_object_or_404(owned_runs(request.user).select_for_update(), pk=run_id)
        if run.status not in (InvoiceRun.Status.NEEDS_REVIEW, InvoiceRun.Status.APPROVED):
            return Response({"detail": "Run is not ready for review"}, status=409)
        item = get_object_or_404(LineItem, run=run, stable_id=stable_id)
        serializer = ItemCorrectionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        values = serializer.validated_data
        before = {key: str(getattr(item, key)) if getattr(item, key) is not None else None
                  for key in ("name", "quantity", "unit_rate", "gross_amount", "discount", "net_amount")}
        before["group_id"] = item.group.stable_id if item.group else None
        for key, value in values.items():
            if key in ("note", "group_id"):
                continue
            setattr(item, key, value)
        if "group_id" in values:
            group_id = values["group_id"]
            item.group = get_object_or_404(BillingGroup, run=run, stable_id=group_id) if group_id else None
            item.relationship = "human_corrected"
        item.save()
        after = {key: str(getattr(item, key)) if getattr(item, key) is not None else None
                 for key in ("name", "quantity", "unit_rate", "gross_amount", "discount", "net_amount")}
        after["group_id"] = item.group.stable_id if item.group else None
        invoice = run.extraction
        for row in invoice.get("items", []):
            if row["id"] == stable_id:
                row.update(after)
                row["relationship"] = item.relationship
                break
        run.extraction = invoice
        run.status = InvoiceRun.Status.NEEDS_REVIEW
        run.save(update_fields=["extraction", "status"])
        run.findings.all().delete()
        run.judge_findings.all().delete()  # prior verdicts concern the old values
        for finding in validate(invoice):
            ValidationFinding.objects.create(run=run, code=finding["code"], severity=finding["severity"],
                                             target_id=finding["target_id"], message=finding["message"],
                                             evidence=finding.get("evidence") or "")
        ReviewEvent.objects.create(run=run, actor=request.user, action="correct_item", target_id=stable_id,
                                   before=before, after=after, note=values["note"])
        return Response(InvoiceRunSerializer(run).data)


class DecideInvoiceView(APIView):
    @transaction.atomic
    def post(self, request, run_id):
        run = get_object_or_404(owned_runs(request.user).select_for_update(), pk=run_id)
        if run.status != InvoiceRun.Status.NEEDS_REVIEW:
            return Response({"detail": "Run is not awaiting review"}, status=409)
        decision = request.data.get("decision")
        note = str(request.data.get("note", "")).strip()
        if decision not in ("approve", "reject") or not note:
            return Response({"detail": "decision (approve/reject) and note are required"}, status=400)
        if decision == "approve" and (run.findings.filter(severity="error").exists()
                                       or not run.items.filter(kind=LineItem.Kind.ITEM).exists()):
            return Response({"detail": "Resolve validation errors and confirm item rows before approval"}, status=409)
        before = run.status
        run.status = InvoiceRun.Status.APPROVED if decision == "approve" else InvoiceRun.Status.NEEDS_REVIEW
        run.save(update_fields=["status"])
        ReviewEvent.objects.create(run=run, actor=request.user, action=decision,
                                   before={"status": before}, after={"status": run.status}, note=note)
        return Response(InvoiceRunSerializer(run).data)


class InvoicePageImageView(APIView):
    def get(self, request, run_id, page_number):
        run = get_object_or_404(owned_runs(request.user), pk=run_id)
        page = get_object_or_404(run.ocr_job.preprocessing_job.pages, page_number=page_number)
        response = FileResponse(page.processed_image.open("rb"), content_type="image/png")
        response["Cache-Control"] = "private, no-store"
        return response
