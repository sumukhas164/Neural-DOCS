from django.conf import settings
from django.db import models

from ocr.models import OcrJob


class InvoiceRun(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        NEEDS_REVIEW = "needs_review", "Needs review"
        AUTO_APPROVED = "auto_approved", "Auto-approved"
        APPROVED = "approved", "Approved"
        FAILED = "failed", "Failed"

    ocr_job = models.ForeignKey(OcrJob, on_delete=models.CASCADE, related_name="invoice_runs")
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.PENDING, db_index=True)
    stage = models.CharField(max_length=64, default="pending")
    schema_version = models.CharField(max_length=16, default="1.0")
    original_extraction = models.JSONField(default=dict)
    extraction = models.JSONField(default=dict)
    checkpoint_state = models.JSONField(default=dict)
    confidence_score = models.PositiveSmallIntegerField(null=True, blank=True)
    confidence = models.JSONField(default=dict)
    validation_summary = models.JSONField(default=dict)
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
        TAX = "tax", "Tax"
        DISCOUNT = "discount", "Discount"
        PAYMENT = "payment", "Payment"
        ADJUSTMENT = "adjustment", "Adjustment"

    class ReviewStatus(models.TextChoices):
        AUTO_EXTRACTED = "auto_extracted", "Automatically extracted"
        NEEDS_REVIEW = "needs_review", "Needs review"
        HUMAN_CORRECTED = "human_corrected", "Human corrected"
        HUMAN_ADDED = "human_added", "Added by reviewer"

    run = models.ForeignKey(InvoiceRun, on_delete=models.CASCADE, related_name="items")
    stable_id = models.CharField(max_length=64)
    group = models.ForeignKey(BillingGroup, null=True, blank=True, on_delete=models.SET_NULL, related_name="items")
    page_number = models.PositiveIntegerField()
    table_number = models.PositiveIntegerField(null=True, blank=True)
    row_order = models.PositiveIntegerField()
    serial = models.CharField(max_length=32, blank=True)
    kind = models.CharField(max_length=16, choices=Kind.choices)
    name = models.CharField(max_length=500)
    quantity = models.DecimalField(max_digits=16, decimal_places=3, null=True, blank=True)
    unit_rate = models.DecimalField(max_digits=16, decimal_places=2, null=True, blank=True)
    gross_amount = models.DecimalField(max_digits=16, decimal_places=2, null=True, blank=True)
    discount = models.DecimalField(max_digits=16, decimal_places=2, null=True, blank=True)
    net_amount = models.DecimalField(max_digits=16, decimal_places=2, null=True, blank=True)
    relationship = models.CharField(max_length=32)
    source_text = models.TextField()
    extraction_method = models.CharField(max_length=64, default="page_ocr")
    review_status = models.CharField(max_length=24, choices=ReviewStatus.choices,
                                     default=ReviewStatus.AUTO_EXTRACTED)
    flags = models.JSONField(default=list)
    tax_amounts = models.JSONField(default=dict)

    class Meta:
        ordering = ["page_number", "row_order"]
        constraints = [models.UniqueConstraint(fields=["run", "stable_id"], name="unique_invoice_row_id")]


class ValidationFinding(models.Model):
    class Category(models.TextChoices):
        FINANCIAL = "financial", "Financial"
        COMPLETENESS = "completeness", "Completeness"

    run = models.ForeignKey(InvoiceRun, on_delete=models.CASCADE, related_name="findings")
    category = models.CharField(max_length=16, choices=Category.choices, default=Category.FINANCIAL)
    code = models.CharField(max_length=64)
    severity = models.CharField(max_length=16)
    target_id = models.CharField(max_length=64)
    message = models.TextField()
    evidence = models.TextField(blank=True)
    resolved = models.BooleanField(default=False)
    resolution_note = models.TextField(blank=True)


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


class OcrRetryAttempt(models.Model):
    """One targeted OCR call on a crop of a problem page, kept for audit."""

    run = models.ForeignKey(InvoiceRun, on_delete=models.CASCADE, related_name="retry_attempts")
    page_number = models.PositiveIntegerField()
    region = models.JSONField(default=list)  # [x, y, width, height] in processed-page pixels
    coverage = models.CharField(max_length=16)  # "tables" or "page"
    reasons = models.JSONField(default=list)
    status = models.CharField(max_length=16)  # completed | truncated | failed
    text = models.TextField(blank=True)
    raw_response = models.JSONField(default=dict)
    error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["page_number", "id"]
