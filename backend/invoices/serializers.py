from rest_framework import serializers

from .models import (BillingGroup, InvoiceRun, JudgeFinding, LineItem, OcrRetryAttempt,
                     ReviewEvent, StageEvent, ValidationFinding)


class GroupSerializer(serializers.ModelSerializer):
    class Meta:
        model = BillingGroup
        fields = ["stable_id", "name", "page_number", "source_text"]


class ItemSerializer(serializers.ModelSerializer):
    group_id = serializers.CharField(source="group.stable_id", read_only=True, allow_null=True)

    class Meta:
        model = LineItem
        fields = ["stable_id", "group_id", "page_number", "table_number", "row_order", "serial", "kind", "name",
                  "quantity", "unit_rate", "gross_amount", "discount", "net_amount", "tax_amounts",
                  "relationship", "source_text", "extraction_method", "review_status", "flags"]


class FindingSerializer(serializers.ModelSerializer):
    class Meta:
        model = ValidationFinding
        fields = ["id", "category", "code", "severity", "target_id", "message", "evidence",
                  "resolved", "resolution_note"]


class JudgeSerializer(serializers.ModelSerializer):
    class Meta:
        model = JudgeFinding
        fields = ["target_id", "verdict", "reason", "evidence", "model_name"]


class ReviewEventSerializer(serializers.ModelSerializer):
    actor = serializers.CharField(source="actor.username", read_only=True)

    class Meta:
        model = ReviewEvent
        fields = ["actor", "action", "target_id", "before", "after", "note", "created_at"]


class RetryAttemptSerializer(serializers.ModelSerializer):
    """Raw responses stay in the database for audit; the API returns the readable text."""

    class Meta:
        model = OcrRetryAttempt
        fields = ["page_number", "region", "coverage", "reasons", "status", "text", "error", "created_at"]


class StageEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = StageEvent
        fields = ["name", "created_at"]


class InvoiceRunSerializer(serializers.ModelSerializer):
    groups = GroupSerializer(many=True, read_only=True)
    items = ItemSerializer(many=True, read_only=True)
    structured_items = serializers.SerializerMethodField()
    findings = FindingSerializer(many=True, read_only=True)
    judge_findings = JudgeSerializer(many=True, read_only=True)
    review_events = ReviewEventSerializer(many=True, read_only=True)
    stage_events = StageEventSerializer(many=True, read_only=True)
    retry_attempts = RetryAttemptSerializer(many=True, read_only=True)
    ocr_job_pages = serializers.IntegerField(source="ocr_job.preprocessing_job.pages_total", read_only=True)

    def get_structured_items(self, run):
        group_names = {group.pk: group.name for group in run.groups.all()}
        rows = []
        for item in run.items.all():
            if item.kind != LineItem.Kind.ITEM:
                continue
            amount = item.net_amount if item.net_amount is not None else item.gross_amount
            rows.append({
                "item_name": item.name,
                "billing_group": group_names.get(item.group_id),
                "qty": str(item.quantity) if item.quantity is not None else None,
                "rate": str(item.unit_rate) if item.unit_rate is not None else None,
                "amount": str(amount) if amount is not None else None,
                "page_number": item.page_number,
                "row_order": item.row_order,
                "source_text": item.source_text,
                "extraction_method": item.extraction_method,
                "review_status": item.review_status,
            })
        return rows

    class Meta:
        model = InvoiceRun
        fields = ["id", "ocr_job", "status", "stage", "schema_version", "extraction", "original_extraction",
                  "confidence_score", "confidence", "validation_summary", "ocr_job_pages",
                  "error_message", "created_at", "started_at", "finished_at",
                  "groups", "items", "structured_items", "findings", "judge_findings", "review_events", "stage_events",
                  "retry_attempts"]


class ItemCorrectionSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=500, required=False)
    group_id = serializers.CharField(max_length=64, allow_null=True, required=False)
    quantity = serializers.DecimalField(max_digits=16, decimal_places=3, allow_null=True, required=False)
    unit_rate = serializers.DecimalField(max_digits=16, decimal_places=2, allow_null=True, required=False)
    gross_amount = serializers.DecimalField(max_digits=16, decimal_places=2, allow_null=True, required=False)
    discount = serializers.DecimalField(max_digits=16, decimal_places=2, allow_null=True, required=False)
    net_amount = serializers.DecimalField(max_digits=16, decimal_places=2, allow_null=True, required=False)
    note = serializers.CharField(required=True, allow_blank=False, max_length=2000)
