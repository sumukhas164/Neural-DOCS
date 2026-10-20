from rest_framework import serializers

from .models import (BillingGroup, InvoiceRun, JudgeFinding, LineItem,
                     ReviewEvent, StageEvent, ValidationFinding)


class GroupSerializer(serializers.ModelSerializer):
    class Meta:
        model = BillingGroup
        fields = ["stable_id", "name", "page_number", "source_text"]


class ItemSerializer(serializers.ModelSerializer):
    group_id = serializers.CharField(source="group.stable_id", read_only=True, allow_null=True)

    class Meta:
        model = LineItem
        fields = ["stable_id", "group_id", "page_number", "row_order", "kind", "name",
                  "quantity", "unit_rate", "gross_amount", "discount", "net_amount",
                  "relationship", "source_text"]


class FindingSerializer(serializers.ModelSerializer):
    class Meta:
        model = ValidationFinding
        fields = ["code", "severity", "target_id", "message", "evidence"]


class JudgeSerializer(serializers.ModelSerializer):
    class Meta:
        model = JudgeFinding
        fields = ["target_id", "verdict", "reason", "evidence", "model_name"]


class ReviewEventSerializer(serializers.ModelSerializer):
    actor = serializers.CharField(source="actor.username", read_only=True)

    class Meta:
        model = ReviewEvent
        fields = ["actor", "action", "target_id", "before", "after", "note", "created_at"]


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
            })
        return rows

    class Meta:
        model = InvoiceRun
        fields = ["id", "ocr_job", "status", "stage", "schema_version", "extraction", "original_extraction",
                  "error_message", "created_at", "started_at", "finished_at",
                  "groups", "items", "structured_items", "findings", "judge_findings", "review_events", "stage_events"]


class ItemCorrectionSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=500, required=False)
    group_id = serializers.CharField(max_length=64, allow_null=True, required=False)
    quantity = serializers.DecimalField(max_digits=16, decimal_places=3, allow_null=True, required=False)
    unit_rate = serializers.DecimalField(max_digits=16, decimal_places=2, allow_null=True, required=False)
    gross_amount = serializers.DecimalField(max_digits=16, decimal_places=2, allow_null=True, required=False)
    discount = serializers.DecimalField(max_digits=16, decimal_places=2, allow_null=True, required=False)
    net_amount = serializers.DecimalField(max_digits=16, decimal_places=2, allow_null=True, required=False)
    note = serializers.CharField(required=True, allow_blank=False, max_length=2000)
