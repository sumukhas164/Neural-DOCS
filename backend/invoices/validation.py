"""Deterministic financial and relationship checks; never changes source values."""

from collections import defaultdict
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP


TOLERANCE = Decimal("0.02")


def money(value):
    if value in (None, ""):
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() else None


def validate(invoice):
    findings = []

    def add(code, severity, target, message, evidence=None):
        findings.append({"code": code, "severity": severity, "target_id": target,
                         "message": message, "evidence": evidence})

    if invoice.get("schema_version") != "1.0":
        add("schema_version", "error", "invoice", "Unexpected extraction schema version")
    groups = {group["id"] for group in invoice.get("groups", [])}
    seen = set()
    group_rows = defaultdict(list)
    group_subtotals = defaultdict(list)
    for row in invoice.get("items", []):
        rid = row.get("id") or "unknown"
        gid = row.get("group_id")
        if gid and gid not in groups:
            add("unknown_group", "error", rid, "Row references a missing billing group", row.get("source_text"))
        if row.get("kind") == "item" and not gid:
            add("unassigned_group", "review", rid, "Billing group is uncertain", row.get("source_text"))
        if row.get("relationship") == "inferred_cross_page":
            add("cross_page_group", "review", rid, "Group assignment continues from an earlier page", row.get("source_text"))
        if row.get("kind") == "item":
            fingerprint = (row.get("name", "").casefold().strip(), row.get("quantity"),
                           row.get("unit_rate"), row.get("net_amount"), gid)
            if fingerprint in seen:
                add("possible_duplicate", "review", rid, "Possible duplicate item row", row.get("source_text"))
            seen.add(fingerprint)
            group_rows[gid].append(row)
            qty = money(row.get("quantity"))
            rate = money(row.get("unit_rate"))
            gross = money(row.get("gross_amount"))
            discount = money(row.get("discount"))
            net = money(row.get("net_amount"))
            if qty is not None and rate is not None and gross is not None:
                expected = (qty * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                if abs(expected - gross) > TOLERANCE:
                    add("line_gross_mismatch", "error", rid, f"Quantity × rate is {expected}; OCR gross is {gross}", row.get("source_text"))
            if qty is not None and rate is not None and net is not None and gross is None and discount is None:
                expected = (qty * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                if abs(expected - net) > TOLERANCE:
                    add("line_net_from_rate_mismatch", "review", rid,
                        f"Quantity × rate is {expected}; OCR bill amount is {net}", row.get("source_text"))
            if gross is not None and discount is not None and net is not None:
                expected = gross - discount
                if abs(expected - net) > TOLERANCE:
                    add("line_net_mismatch", "error", rid, f"Gross − discount is {expected}; OCR net is {net}", row.get("source_text"))
            for key in ("quantity", "unit_rate", "gross_amount", "discount", "net_amount"):
                if row.get(key) is not None and money(row[key]) is None:
                    add("invalid_number", "error", rid, f"Invalid {key}", row.get("source_text"))
        elif row.get("kind") == "subtotal" and gid:
            group_subtotals[gid].append(row)
    for gid, subtotals in group_subtotals.items():
        values = [money(row.get("net_amount")) or money(row.get("gross_amount")) for row in group_rows[gid]]
        if values and all(value is not None for value in values):
            for subtotal in subtotals:
                amount = money(subtotal.get("net_amount")) or money(subtotal.get("gross_amount"))
                if amount is not None and abs(sum(values) - amount) > TOLERANCE:
                    add("group_total_mismatch", "error", subtotal["id"],
                        f"Item sum is {sum(values)}; group subtotal is {amount}", subtotal.get("source_text"))
    total = money(invoice.get("grand_total"))
    item_values = [money(row.get("net_amount")) or money(row.get("gross_amount"))
                   for row in invoice.get("items", []) if row.get("kind") == "item"]
    if total is not None and item_values and all(value is not None for value in item_values):
        if abs(sum(item_values) - total) > TOLERANCE:
            add("grand_total_mismatch", "review", "invoice",
                f"Item sum is {sum(item_values)}; invoice total is {total}. Other charges may exist")
    if not invoice.get("items"):
        add("no_items", "review", "invoice", "No invoice item rows were reconstructed")
    return findings
