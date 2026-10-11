import logging

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from ocr.models import OcrJob

from .models import BillingGroup, InvoiceRun, JudgeFinding, LineItem, StageEvent, ValidationFinding
from .validation import money
from .workflow import build_graph

logger = logging.getLogger(__name__)


def submit_run(ocr_job):
    if ocr_job.status != OcrJob.Status.COMPLETED:
        raise ValueError("OCR must complete before invoice extraction")
    run = InvoiceRun.objects.create(ocr_job=ocr_job)
    from .tasks import process_invoice_task
    transaction.on_commit(lambda: process_invoice_task.delay(run.pk))
    return run


def _persist(run, state):
    invoice = state["invoice"]
    with transaction.atomic():
        run.extraction = invoice
        run.original_extraction = invoice
        run.stage = "human_review"
        run.status = InvoiceRun.Status.NEEDS_REVIEW
        run.finished_at = timezone.now()
        run.save(update_fields=["extraction", "original_extraction", "stage", "status", "finished_at"])
        groups = {}
        for group in invoice["groups"]:
            groups[group["id"]] = BillingGroup.objects.create(
                run=run, stable_id=group["id"], name=group["name"],
                page_number=group["page_number"], source_text=group["source_text"])
        for row in invoice["items"]:
            LineItem.objects.create(
                run=run, stable_id=row["id"], group=groups.get(row["group_id"]),
                page_number=row["page_number"], row_order=row["row_order"], kind=row["kind"],
                name=row["name"], quantity=money(row["quantity"]), unit_rate=money(row["unit_rate"]),
                gross_amount=money(row["gross_amount"]), discount=money(row["discount"]),
                net_amount=money(row["net_amount"]), relationship=row["relationship"],
                source_text=row["source_text"])
        for finding in state["findings"]:
            ValidationFinding.objects.create(run=run, code=finding["code"],
                                             severity=finding["severity"], target_id=finding["target_id"],
                                             message=finding["message"], evidence=finding.get("evidence") or "")
        for verdict in state["judge_findings"]:
            JudgeFinding.objects.create(run=run, **verdict)


def run_job(run_id):
    claimed = InvoiceRun.objects.filter(pk=run_id, status=InvoiceRun.Status.PENDING).update(
        status=InvoiceRun.Status.RUNNING, stage="reconstruct_structure", started_at=timezone.now())
    run = InvoiceRun.objects.select_related("ocr_job").get(pk=run_id)
    if not claimed:
        return run
    try:
        state = {"ocr_job_id": run.ocr_job_id}
        for update in build_graph().stream(state, stream_mode="updates"):
            for stage, values in update.items():
                state.update(values)
                run.stage = stage
                run.checkpoint_state = state
                run.save(update_fields=["stage", "checkpoint_state"])
                StageEvent.objects.create(run=run, name=stage)
        _persist(run, state)
    except Exception as exc:
        logger.exception("invoice run %s failed", run_id)
        run.status = InvoiceRun.Status.FAILED
        run.stage = "failed"
        run.error_message = f"{type(exc).__name__}: {exc}"[:2000]
        run.finished_at = timezone.now()
        run.save(update_fields=["status", "stage", "error_message", "finished_at"])
    return run
