"""Rule-based extraction confidence that decides whether a run needs human review.

Engineering default: Nanonets returns no field confidence, so the score is
derived from deterministic validation and completeness findings and OCR
retries. It is not calibrated against labelled invoices. Unresolved
completeness warnings block automatic approval whatever the score.
"""

from decimal import Decimal, ROUND_HALF_UP

from .validation import money

INVOICE_FINDING_PENALTY = 15  # page, grand/group total or schema findings affect the whole invoice
RETRIED_PAGE_PENALTY = 10  # downscaled OCR retries can lose small text


def _active(findings):
    return [finding for finding in findings
            if finding.get("severity") != "info" and not finding.get("resolved")]


def score(invoice, findings, retried_pages=()):
    """Return a 0-100 score, the reasons behind it, and any approval blockers."""
    findings = _active(findings)
    items = [row for row in invoice.get("items", []) if row.get("kind") == "item"]
    item_ids = {row["id"] for row in items}
    flagged = {finding["target_id"] for finding in findings}
    reasons = []
    blockers = []
    if any(finding["severity"] == "error" for finding in findings):
        blockers.append("Validation errors block automatic approval")
    completeness = [finding for finding in findings if finding.get("category") == "completeness"]
    if completeness:
        blockers.append(f"{len(completeness)} unresolved completeness warning(s): rows may be missing or incomplete")
    if not items:
        blockers.append("No invoice item rows were reconstructed")
        return {"score": 0, "reasons": ["No item rows"], "blockers": blockers,
                "retried_pages": sorted(retried_pages)}

    clean = [row for row in items if row["id"] not in flagged
             and (money(row.get("net_amount")) is not None or money(row.get("gross_amount")) is not None)]
    value = Decimal(100) * len(clean) / len(items)
    if len(clean) < len(items):
        reasons.append(f"{len(items) - len(clean)} of {len(items)} item rows are flagged or have no amount")
    invoice_findings = [finding for finding in findings if finding["target_id"] not in item_ids]
    if invoice_findings:
        value -= INVOICE_FINDING_PENALTY * len(invoice_findings)
        reasons.append(f"{len(invoice_findings)} invoice, page or subtotal finding(s): "
                       + ", ".join(sorted({finding['code'] for finding in invoice_findings})))
    if retried_pages:
        value -= RETRIED_PAGE_PENALTY * len(retried_pages)
        reasons.append("OCR used a downscaled retry on page(s) "
                       + ", ".join(str(page) for page in sorted(retried_pages)))
    value = max(Decimal(0), min(Decimal(100), value))
    return {"score": int(value.quantize(Decimal(1), rounding=ROUND_HALF_UP)), "reasons": reasons,
            "blockers": blockers, "retried_pages": sorted(retried_pages)}


def needs_review(confidence, threshold):
    return bool(confidence["blockers"]) or confidence["score"] < threshold
