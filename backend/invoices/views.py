from django.db import transaction
from django.conf import settings
from django.http import FileResponse
from django.shortcuts import get_object_or_404
from rest_framework import mixins, status, viewsets
from rest_framework.response import Response
from rest_framework.views import APIView

from ocr.models import OcrJob

from .confidence import score
from .models import BillingGroup, InvoiceRun, LineItem, ReviewEvent, ValidationFinding
from .serializers import InvoiceRunSerializer, ItemCorrectionSerializer
from .services import submit_run
from .validation import money, validate


def owned_runs(user):
    return InvoiceRun.objects.filter(ocr_job__preprocessing_job__document__uploaded_by=user)


class InvoiceRunViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    serializer_class = InvoiceRunSerializer

    def get_queryset(self):
        return owned_runs(self.request.user).prefetch_related("groups", "items", "findings", "judge_findings",
                                                             "review_events", "stage_events", "retry_attempts")


class StartInvoiceView(APIView):
    def post(self, request, ocr_job_id):
        job = get_object_or_404(OcrJob, pk=ocr_job_id,
                                preprocessing_job__document__uploaded_by=request.user)
        if job.status != OcrJob.Status.COMPLETED:
            return Response({"detail": "OCR must complete before extraction"}, status=409)
        if settings.CELERY_TASK_ALWAYS_EAGER and not settings.LOCAL_DEMO_MODE:
            return Response({"detail": "Configure a Celery broker and worker for API extraction; use the process_invoice management command locally."}, status=503)
        run = submit_run(job)
        return Response(InvoiceRunSerializer(run).data, status=status.HTTP_202_ACCEPTED)


FIELDS = ("name", "quantity", "unit_rate", "gross_amount", "discount", "net_amount")
REVIEWABLE = (InvoiceRun.Status.NEEDS_REVIEW, InvoiceRun.Status.AUTO_APPROVED, InvoiceRun.Status.APPROVED)


def _snapshot(item):
    values = {key: str(getattr(item, key)) if getattr(item, key) is not None else None for key in FIELDS}
    values["group_id"] = item.group.stable_id if item.group else None
    return values


def _rescore(run):
    findings = list(run.findings.values("code", "severity", "target_id", "category", "resolved"))
    confidence = {**score(run.extraction, findings, run.confidence.get("retried_pages", ())),
                  "threshold": settings.INVOICE_CONFIDENCE_THRESHOLD}
    run.confidence = confidence
    run.confidence_score = confidence["score"]


def _reassess(run, invoice):
    """Re-run financial checks after a reviewer change.

    Completeness findings describe what OCR read, so they stay until a reviewer
    resolves them. A human is now involved, so the run stays in review.
    """
    financial = run.findings.filter(category=ValidationFinding.Category.FINANCIAL)
    resolved = {(finding.code, finding.target_id, finding.message): finding.resolution_note
                for finding in financial.filter(resolved=True)}
    financial.delete()
    run.judge_findings.all().delete()  # prior verdicts concern the old values
    summary = {}
    for finding in validate(invoice, summary):
        key = (finding["code"], finding["target_id"], finding["message"])
        ValidationFinding.objects.create(run=run, category=ValidationFinding.Category.FINANCIAL,
                                         code=finding["code"], severity=finding["severity"],
                                         target_id=finding["target_id"], message=finding["message"],
                                         evidence=finding.get("evidence") or "",
                                         resolved=key in resolved, resolution_note=resolved.get(key, ""))
    run.extraction = invoice
    run.validation_summary = summary
    _rescore(run)
    run.status = InvoiceRun.Status.NEEDS_REVIEW
    run.stage = "human_review"
    run.save(update_fields=["extraction", "validation_summary", "confidence", "confidence_score", "status", "stage"])


def _required_note(request):
    return str(request.data.get("note", "")).strip()[:2000]


class CorrectItemView(APIView):
    @transaction.atomic
    def patch(self, request, run_id, stable_id):
        run = get_object_or_404(owned_runs(request.user).select_for_update(), pk=run_id)
        if run.status not in REVIEWABLE:
            return Response({"detail": "Run is not ready for review"}, status=409)
        item = get_object_or_404(LineItem, run=run, stable_id=stable_id)
        serializer = ItemCorrectionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        values = serializer.validated_data
        before = _snapshot(item)
        for key, value in values.items():
            if key in ("note", "group_id"):
                continue
            setattr(item, key, value)
        if "group_id" in values:
            group_id = values["group_id"]
            item.group = get_object_or_404(BillingGroup, run=run, stable_id=group_id) if group_id else None
            item.relationship = "human_corrected"
        if item.review_status != LineItem.ReviewStatus.HUMAN_ADDED:
            item.review_status = LineItem.ReviewStatus.HUMAN_CORRECTED
        item.save()
        after = _snapshot(item)
        invoice = run.extraction
        for row in invoice.get("items", []):
            if row["id"] == stable_id:
                row.update(after)
                row["relationship"] = item.relationship
                row["review_status"] = item.review_status
                break
        # The reviewer has looked at this row against the source.
        run.findings.filter(category=ValidationFinding.Category.COMPLETENESS, target_id=stable_id,
                            resolved=False).update(resolved=True, resolution_note=f"Row corrected: {values['note']}")
        _reassess(run, invoice)
        ReviewEvent.objects.create(run=run, actor=request.user, action="correct_item", target_id=stable_id,
                                   before=before, after=after, note=values["note"])
        return Response(InvoiceRunSerializer(run).data)


class CandidateRowView(APIView):
    """Accept or dismiss a row that only targeted table OCR found."""

    @transaction.atomic
    def post(self, request, run_id, candidate_id):
        run = get_object_or_404(owned_runs(request.user).select_for_update(), pk=run_id)
        if run.status not in REVIEWABLE:
            return Response({"detail": "Run is not ready for review"}, status=409)
        action, note = request.data.get("action"), _required_note(request)
        if action not in ("accept", "dismiss") or not note:
            return Response({"detail": "action (accept/dismiss) and note are required"}, status=400)
        invoice = run.extraction
        candidate = next((row for row in invoice.get("candidate_rows", [])
                          if row["id"] == candidate_id and row.get("status") == "pending"), None)
        if candidate is None:
            return Response({"detail": "No pending candidate row with this id"}, status=404)
        if action == "accept":
            group_id = request.data.get("group_id") or None
            group = get_object_or_404(BillingGroup, run=run, stable_id=group_id) if group_id else None
            page_rows = run.items.filter(page_number=candidate["page_number"])
            row_order = max(page_rows.values_list("row_order", flat=True), default=0) + 1
            item = LineItem.objects.create(
                run=run, stable_id=candidate_id, group=group, page_number=candidate["page_number"],
                row_order=row_order, serial=candidate.get("serial") or "", kind=LineItem.Kind.ITEM,
                name=(candidate.get("name") or "")[:500], quantity=money(candidate.get("quantity")),
                unit_rate=money(candidate.get("unit_rate")), gross_amount=money(candidate.get("gross_amount")),
                discount=money(candidate.get("discount")), net_amount=money(candidate.get("net_amount")),
                relationship="human_added", source_text=candidate.get("source_text") or "",
                extraction_method=candidate.get("extraction_method", "table_crop_ocr"),
                review_status=LineItem.ReviewStatus.HUMAN_ADDED)
            invoice["items"].append({
                "id": candidate_id, "page_number": item.page_number, "table_number": None,
                "row_order": row_order, "kind": "item", "name": item.name, "serial": item.serial or None,
                "group_id": group_id, "relationship": "human_added", "source_text": item.source_text,
                "extraction_method": item.extraction_method, "flags": [], "review_status": item.review_status,
                "tax_amounts": {}, **{key: candidate.get(key) for key in FIELDS[1:]}})
        candidate["status"] = "accepted" if action == "accept" else "dismissed"
        if not any(row.get("status") == "pending" and row["page_number"] == candidate["page_number"]
                   for row in invoice.get("candidate_rows", [])):
            run.findings.filter(code="rows_only_in_targeted_ocr", target_id=f"page-{candidate['page_number']}",
                                resolved=False).update(resolved=True,
                                                       resolution_note="All candidate rows reviewed")
        _reassess(run, invoice)
        ReviewEvent.objects.create(run=run, actor=request.user, action=f"{action}_candidate", target_id=candidate_id,
                                   before=None, after=candidate, note=note)
        return Response(InvoiceRunSerializer(run).data)


class ResolveFindingView(APIView):
    """Record that a reviewer checked a warning against the source invoice."""

    @transaction.atomic
    def post(self, request, run_id, finding_id):
        run = get_object_or_404(owned_runs(request.user).select_for_update(), pk=run_id)
        if run.status not in REVIEWABLE:
            return Response({"detail": "Run is not ready for review"}, status=409)
        note = _required_note(request)
        if not note:
            return Response({"detail": "A note describing what was checked is required"}, status=400)
        finding = get_object_or_404(ValidationFinding, run=run, pk=finding_id)
        finding.resolved, finding.resolution_note = True, note
        finding.save(update_fields=["resolved", "resolution_note"])
        _rescore(run)
        run.save(update_fields=["confidence", "confidence_score"])
        ReviewEvent.objects.create(run=run, actor=request.user, action="resolve_finding",
                                   target_id=finding.target_id,
                                   before={"code": finding.code, "resolved": False},
                                   after={"code": finding.code, "resolved": True}, note=note)
        return Response(InvoiceRunSerializer(run).data)


class DecideInvoiceView(APIView):
    @transaction.atomic
    def post(self, request, run_id):
        run = get_object_or_404(owned_runs(request.user).select_for_update(), pk=run_id)
        if run.status not in (InvoiceRun.Status.NEEDS_REVIEW, InvoiceRun.Status.AUTO_APPROVED):
            return Response({"detail": "Run is not awaiting review"}, status=409)
        decision = request.data.get("decision")
        note = _required_note(request)
        if decision not in ("approve", "reject") or not note:
            return Response({"detail": "decision (approve/reject) and note are required"}, status=400)
        if decision == "approve" and (run.findings.filter(severity="error", resolved=False).exists()
                                       or not run.items.filter(kind=LineItem.Kind.ITEM).exists()):
            return Response({"detail": "Correct or resolve every error finding and confirm item rows before approval"},
                            status=409)
        before = run.status
        run.status = InvoiceRun.Status.APPROVED if decision == "approve" else InvoiceRun.Status.NEEDS_REVIEW
        run.stage = "approved" if decision == "approve" else "human_review"
        run.save(update_fields=["status", "stage"])
        ReviewEvent.objects.create(run=run, actor=request.user, action=decision,
                                   before={"status": before}, after={"status": run.status}, note=note)
        return Response(InvoiceRunSerializer(run).data)


class InvoicePageImageView(APIView):
    def get(self, request, run_id, page_number):
        run = get_object_or_404(owned_runs(request.user), pk=run_id)
        page = get_object_or_404(run.ocr_job.preprocessing_job.pages, page_number=page_number)
        image = page.original_image if request.query_params.get("variant") == "original" else page.processed_image
        response = FileResponse(image.open("rb"), content_type="image/png")
        response["Cache-Control"] = "private, no-store"
        return response
