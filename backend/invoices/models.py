from django.conf import settings
from django.db import models

from ocr.models import OcrJob


class InvoiceRun(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        NEEDS_REVIEW = "needs_review", "Needs review"
        APPROVED = "approved", "Approved"
        FAILED = "failed", "Failed"

    ocr_job = models.ForeignKey(OcrJob, on_delete=models.CASCADE, related_name="invoice_runs")
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.PENDING, db_index=True)
    stage = models.CharField(max_length=64, default="pending")
    schema_version = models.CharField(max_length=16, default="1.0")
    original_extraction = models.JSONField(default=dict)
    extraction = models.JSONField(default=dict)
    checkpoint_state = models.JSONField(default=dict)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)


class BillingGroup(models.Model):
    run = models.ForeignKey(InvoiceRun, on_delete=models.CASCADE, related_name="groups")
    stable_id = models.CharField(max_length=64)
    name = models.CharField(max_length=255)
    page_number = models.PositiveIntegerField()
    source_text = models.TextField(blank=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["run", "stable_id"], name="unique_invoice_group_id")]


class LineItem(models.Model):
    class Kind(models.TextChoices):
        ITEM = "item", "Item"
        SUBTOTAL = "subtotal", "Subtotal"
        TOTAL = "total", "Total"

    run = models.ForeignKey(InvoiceRun, on_delete=models.CASCADE, related_name="items")
    stable_id = models.CharField(max_length=64)
    group = models.ForeignKey(BillingGroup, null=True, blank=True, on_delete=models.SET_NULL, related_name="items")
    page_number = models.PositiveIntegerField()
    row_order = models.PositiveIntegerField()
    kind = models.CharField(max_length=16, choices=Kind.choices)
    name = models.CharField(max_length=500)
    quantity = models.DecimalField(max_digits=16, decimal_places=3, null=True, blank=True)
    unit_rate = models.DecimalField(max_digits=16, decimal_places=2, null=True, blank=True)
    gross_amount = models.DecimalField(max_digits=16, decimal_places=2, null=True, blank=True)
    discount = models.DecimalField(max_digits=16, decimal_places=2, null=True, blank=True)
    net_amount = models.DecimalField(max_digits=16, decimal_places=2, null=True, blank=True)
    relationship = models.CharField(max_length=32)
    source_text = models.TextField()

    class Meta:
        ordering = ["page_number", "row_order"]
        constraints = [models.UniqueConstraint(fields=["run", "stable_id"], name="unique_invoice_row_id")]


class ValidationFinding(models.Model):
    run = models.ForeignKey(InvoiceRun, on_delete=models.CASCADE, related_name="findings")
    code = models.CharField(max_length=64)
    severity = models.CharField(max_length=16)
    target_id = models.CharField(max_length=64)
    message = models.TextField()
    evidence = models.TextField(blank=True)


class JudgeFinding(models.Model):
    run = models.ForeignKey(InvoiceRun, on_delete=models.CASCADE, related_name="judge_findings")
    target_id = models.CharField(max_length=64)
    verdict = models.CharField(max_length=32)
    reason = models.TextField()
    evidence = models.TextField(blank=True)
    model_name = models.CharField(max_length=128)
    raw_response = models.JSONField(default=dict)


class ReviewEvent(models.Model):
    run = models.ForeignKey(InvoiceRun, on_delete=models.CASCADE, related_name="review_events")
    actor = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT)
    action = models.CharField(max_length=32)
    target_id = models.CharField(max_length=64, blank=True)
    before = models.JSONField(null=True, blank=True)
    after = models.JSONField(null=True, blank=True)
    note = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


class StageEvent(models.Model):
    run = models.ForeignKey(InvoiceRun, on_delete=models.CASCADE, related_name="stage_events")
    name = models.CharField(max_length=64)
    created_at = models.DateTimeField(auto_now_add=True)
