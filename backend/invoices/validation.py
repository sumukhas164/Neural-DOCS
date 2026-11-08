"""Deterministic financial and relationship checks; never changes source values.

A check whose operands are not all printed is counted as unavailable; a
missing value is never treated as zero.
"""

from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP


TOLERANCE = Decimal("0.02")
ROUNDING_TOLERANCE = Decimal("1.00")  # Indian bills commonly round the payable amount to the rupee
CHECKS = ("line_quantity_rate", "line_discount", "line_tax", "group_subtotal", "grand_total")


def money(value):
    if value in (None, ""):
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() else None


def amount(row):
    net = money(row.get("net_amount"))
    return net if net is not None else money(row.get("gross_amount"))


def validate(invoice, summary=None):
    """Return findings; if ``summary`` is a dict, fill it with per-check counts."""
    findings = []
    counts = {check: Counter() for check in CHECKS}

    def add(code, severity, target, message, evidence=None):
        findings.append({"code": code, "severity": severity, "target_id": target,
                         "message": message, "evidence": evidence, "category": "financial"})

    def count(check, outcome):
        counts[check][outcome] += 1

    if invoice.get("schema_version") not in ("1.0", "1.1"):
        add("schema_version", "error", "invoice", "Unexpected extraction schema version")
    groups = {group["id"] for group in invoice.get("groups", [])}
    seen = set()
    continued = set()
    group_rows = defaultdict(list)
    group_subtotals = defaultdict(list)
    for row in invoice.get("items", []):
        rid = row.get("id") or "unknown"
        gid = row.get("group_id")
        if gid and gid not in groups:
            add("unknown_group", "error", rid, "Row references a missing billing group", row.get("source_text"))
        if row.get("kind") == "item" and not gid:
            add("unassigned_group", "review", rid, "Billing group is uncertain", row.get("source_text"))
        if row.get("relationship") in ("inferred_cross_page", "inferred_from_section_total") \
                and (gid, row.get("page_number")) not in continued:
            # Continuing a section across a page break is normal; noted once per section and page.
            continued.add((gid, row.get("page_number")))
            add("cross_page_group", "info", rid, "Group assignment continues from an earlier page", row.get("source_text"))
        if row.get("kind") == "item":
            # Repeated descriptions are normal (e.g. daily room rent); only an
            # identical row, including its printed source text, is suspicious.
            fingerprint = (row.get("name", "").casefold().strip(), row.get("quantity"),
                           row.get("unit_rate"), row.get("net_amount"), gid,
                           " ".join((row.get("source_text") or "").split()).casefold())
            if fingerprint in seen:
                add("possible_duplicate", "review", rid, "Possible duplicate item row", row.get("source_text"))
            seen.add(fingerprint)
            group_rows[gid].append(row)
            _line_checks(row, rid, add, count)
        elif row.get("kind") == "subtotal" and gid:
            group_subtotals[gid].append(row)
    for gid, subtotals in group_subtotals.items():
        values = [amount(row) for row in group_rows[gid]]
        for subtotal in subtotals:
            printed = amount(subtotal)
            if not values or printed is None or any(value is None for value in values):
                count("group_subtotal", "unavailable")
                continue
            if abs(sum(values) - printed) > TOLERANCE:
                count("group_subtotal", "failed")
                add("group_total_mismatch", "error", subtotal["id"],
                    f"Item sum is {sum(values)}; group subtotal is {printed}", subtotal.get("source_text"))
            else:
                count("group_subtotal", "passed")
    _grand_total(invoice, add, count)
    if not invoice.get("items"):
        add("no_items", "review", "invoice", "No invoice item rows were reconstructed")
    if summary is not None:
        summary.update({check: {outcome: counts[check][outcome] for outcome in ("passed", "failed", "unavailable")}
                        for check in CHECKS})
    return findings


def _line_checks(row, rid, add, count):
    qty = money(row.get("quantity"))
    rate = money(row.get("unit_rate"))
    gross = money(row.get("gross_amount"))
    discount = money(row.get("discount"))
    net = money(row.get("net_amount"))
    taxes = [money(value) for value in (row.get("tax_amounts") or {}).values()]
    taxes = [value for value in taxes if value is not None]
    for key in ("quantity", "unit_rate", "gross_amount", "discount", "net_amount"):
        if row.get(key) is not None and money(row[key]) is None:
            add("invalid_number", "error", rid, f"Invalid {key}", row.get("source_text"))
    if qty is not None and rate is not None and (gross is not None or (net is not None and discount is None and not taxes)):
        expected = (qty * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        printed = gross if gross is not None else net
        if expected > 0 and printed < 0:
            expected = -expected  # a return prints negative quantity and rate with a negative amount
        if gross is not None and abs(expected - gross) > TOLERANCE:
            count("line_quantity_rate", "failed")
            add("line_gross_mismatch", "error", rid, f"Quantity × rate is {expected}; OCR gross is {gross}", row.get("source_text"))
        elif gross is None and abs(expected - net) > TOLERANCE:
            count("line_quantity_rate", "failed")
            add("line_net_from_rate_mismatch", "review", rid,
                f"Quantity × rate is {expected}; OCR bill amount is {net}", row.get("source_text"))
        else:
            count("line_quantity_rate", "passed")
    else:
        count("line_quantity_rate", "unavailable")
    if taxes:
        if gross is None or net is None:
            count("line_tax", "unavailable")
            return
        base = gross - discount if discount is not None else gross
        # Some bills print tax-inclusive net amounts, others add tax on top.
        if min(abs(base + sum(taxes) - net), abs(base - net)) > TOLERANCE:
            count("line_tax", "failed")
            add("line_tax_mismatch", "review", rid,
                f"Gross {gross}, discount {discount if discount is not None else 'not printed'} and tax "
                f"{sum(taxes)} do not reconcile with net {net}", row.get("source_text"))
        else:
            count("line_tax", "passed")
        return
    if gross is not None and discount is not None and net is not None:
        expected = gross - discount
        if abs(expected - net) > TOLERANCE:
            count("line_discount", "failed")
            add("line_net_mismatch", "error", rid, f"Gross − discount is {expected}; OCR net is {net}", row.get("source_text"))
        else:
            count("line_discount", "passed")
    elif discount is not None:
        count("line_discount", "unavailable")


def _grand_total(invoice, add, count):
    total = money(invoice.get("grand_total"))
    rows = invoice.get("items", [])
    item_values = [amount(row) for row in rows if row.get("kind") == "item"]
    if total is None or not item_values or any(value is None for value in item_values):
        count("grand_total", "unavailable")
        return
    items_sum = sum(item_values)

    def summed(kind):
        values = [amount(row) for row in rows if row.get("kind") == kind]
        return None if any(value is None for value in values) else sum((abs(value) for value in values), Decimal(0))

    taxes, discounts = summed("tax"), summed("discount")
    adjustments = [amount(row) for row in rows if row.get("kind") == "adjustment"]
    variants = [items_sum]
    if taxes is not None and discounts is not None:
        variants.append(items_sum + taxes - discounts)
        if adjustments and all(value is not None for value in adjustments):
            variants.append(items_sum + taxes - discounts + sum(adjustments))
    if any(abs(variant - total) <= TOLERANCE for variant in variants):
        count("grand_total", "passed")
        return
    if not adjustments and total == total.to_integral_value() and \
            any(abs(variant - total) <= ROUNDING_TOLERANCE for variant in variants):
        count("grand_total", "passed")
        return
    count("grand_total", "failed")
    detail = f"Item sum is {items_sum}"
    if taxes or discounts:
        detail += f" (tax rows {taxes if taxes is not None else 'unreadable'}, discount rows " \
                  f"{discounts if discounts is not None else 'unreadable'})"
    add("grand_total_mismatch", "review", "invoice",
        f"{detail}; invoice total is {total}. Other charges or missing rows may explain the difference")
