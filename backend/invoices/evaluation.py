"""Measure line-item extraction against manually verified invoices.

Ground truth must be checked by a person against the source invoice. Metrics:

* recall    = correctly extracted actual items / actual items
* precision = correctly extracted actual items / extracted items
* field accuracy for description, quantity, rate and amount over matched rows
* duplicate rate = extra extracted rows that repeat a matched actual item / extracted items
* missing rate   = actual items not extracted / actual items

A row matches when it is on the same page (if the truth gives one) and its
description is similar (difflib ratio >= 0.8 after normalisation). Matching
prefers rows whose amount also agrees, so a misread amount still counts as a
found row with a wrong amount field.
"""

import re
from difflib import SequenceMatcher

from .validation import money

MATCH_RATIO = 0.8
FIELDS = (("description", "name"), ("quantity", "quantity"), ("rate", "unit_rate"), ("amount", None))


def _text(value):
    return " ".join(str(value or "").split()).casefold()


def _key(value):
    return re.sub(r"[^0-9a-z]", "", _text(value))


def _amount(row):
    net = money(row.get("net_amount"))
    return net if net is not None else money(row.get("gross_amount"))


def _similarity(truth, row):
    if truth.get("page_number") and row.get("page_number") != truth["page_number"]:
        return 0.0
    ratio = SequenceMatcher(None, _key(truth.get("description")), _key(row.get("name"))).ratio()
    if ratio < MATCH_RATIO:
        return 0.0
    return ratio + (1.0 if money(truth.get("amount")) == _amount(row) else 0.0)


def evaluate(extracted_items, truth_items):
    extracted = [row for row in extracted_items if row.get("kind", "item") == "item"]
    scored = sorted(((_similarity(truth, row), t, e) for t, truth in enumerate(truth_items)
                     for e, row in enumerate(extracted)), reverse=True)
    truth_used, extracted_used, pairs = set(), set(), []
    for similarity, t, e in scored:
        if similarity and t not in truth_used and e not in extracted_used:
            truth_used.add(t)
            extracted_used.add(e)
            pairs.append((truth_items[t], extracted[e]))
    duplicates = sum(1 for e, row in enumerate(extracted) if e not in extracted_used and any(
        _similarity(truth_items[t], row) for t in truth_used))
    correct = {name: 0 for name, _ in FIELDS}
    compared = {name: 0 for name, _ in FIELDS}
    for truth, row in pairs:
        for name, field in FIELDS:
            if truth.get(name) in (None, ""):
                continue
            compared[name] += 1
            if name == "description":
                correct[name] += _text(truth[name]) == _text(row.get(field))
            elif name == "amount":
                correct[name] += money(truth[name]) == _amount(row)
            else:
                correct[name] += money(truth[name]) == money(row.get(field))
    actual, found = len(truth_items), len(pairs)
    return {
        "actual_items": actual, "extracted_items": len(extracted), "matched_items": found,
        "recall": found / actual if actual else None,
        "precision": found / len(extracted) if extracted else None,
        "missing_rate": (actual - found) / actual if actual else None,
        "duplicate_rate": duplicates / len(extracted) if extracted else None,
        "duplicate_rows": duplicates,
        "field_accuracy": {name: correct[name] / compared[name] if compared[name] else None for name, _ in FIELDS},
        "field_counts": compared, "field_correct": correct,
        "missing": [truth.get("description") for t, truth in enumerate(truth_items) if t not in truth_used],
        "unmatched_extracted": [row.get("name") for e, row in enumerate(extracted) if e not in extracted_used],
    }


def aggregate(results):
    """Micro-average over invoices: every actual and extracted row counts once."""
    actual = sum(result["actual_items"] for result in results)
    extracted = sum(result["extracted_items"] for result in results)
    matched = sum(result["matched_items"] for result in results)
    duplicates = sum(result["duplicate_rows"] for result in results)
    accuracy = {}
    for name, _ in FIELDS:
        count = sum(result["field_counts"][name] for result in results)
        hits = sum(result["field_correct"][name] for result in results)
        accuracy[name] = hits / count if count else None
    return {"invoices": len(results), "actual_items": actual, "extracted_items": extracted,
            "matched_items": matched, "recall": matched / actual if actual else None,
            "precision": matched / extracted if extracted else None,
            "missing_rate": (actual - matched) / actual if actual else None,
            "duplicate_rate": duplicates / extracted if extracted else None, "field_accuracy": accuracy}
