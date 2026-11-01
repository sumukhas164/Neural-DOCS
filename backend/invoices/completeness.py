"""Line-item completeness checks: was every printed row read?

Runs after table reconstruction. It never adds, removes or changes extracted
rows. It reports evidence that rows may be missing, merged, duplicated or
incomplete, names the pages worth re-reading with targeted table OCR, and,
after such a re-read, compares the two methods. Rows that only the targeted
read found become candidate rows for a reviewer; neither method wins
automatically. Subtotal differences are never used to invent rows or amounts.
"""

import re
from collections import Counter
from difflib import SequenceMatcher

from .validation import money

CATEGORY = "completeness"
ROW_FLAG_SEVERITY = {
    "missing_amount": "review", "missing_description": "review", "unparsed": "review",
    "ambiguous_separator": "review", "ocr_char_confusion": "review", "too_many_decimals": "review",
    "split_from_merged_cell": "review", "merged_values": "review", "columns_unknown": "review",
    "wrapped_description": "info",
}
RETRY_CODES = {"ocr_truncated", "ocr_downscaled", "empty_ocr_page", "unclosed_table", "untabled_amount_lines",
               "no_rows_on_page", "serial_gap", "merged_values", "split_from_merged_cell", "columns_unknown",
               "missing_grand_total"}
NAME_MATCH_RATIO = 0.85  # engineering default for pairing the same row read twice


def normalise(name):
    return re.sub(r"[^0-9a-z]", "", (name or "").casefold())


def row_amount(row):
    return money(row.get("net_amount")) if money(row.get("net_amount")) is not None else money(row.get("gross_amount"))


class _Findings(list):
    def add(self, code, severity, target, message, evidence=""):
        self.append({"code": code, "severity": severity, "target_id": target, "message": message,
                     "evidence": (evidence or "")[:2000], "category": CATEGORY})


def check(invoice, pages, expected_pages, crops=None):
    """Return (findings, retry_requests, candidate_rows).

    pages: [{page_number, text_chars, finish_reason, retried, all_truncated, blank}]
    crops: {page_number: {"coverage": "tables"|"page", "items": [...], "errors": [...], "truncated": bool}}
    """
    findings = _Findings()
    retry = {}

    def request(page_number, code):
        if code in RETRY_CODES and page_number:
            retry.setdefault(page_number, set()).add(code)

    def add(code, severity, target, message, evidence="", page_number=None):
        findings.add(code, severity, target, message, evidence)
        request(page_number, code)

    seen = {page["page_number"] for page in pages}
    for number in range(1, (expected_pages or 0) + 1):
        if number not in seen:
            add("missing_ocr_page", "error", f"page-{number}", f"Page {number} has no OCR result")
    for page in pages:
        number = page["page_number"]
        if page.get("finish_reason") == "length":
            add("ocr_truncated", "error", f"page-{number}",
                f"OCR output for page {number} was cut off; rows at the end of the page may be missing",
                page_number=number)
        elif page.get("retried"):
            add("ocr_downscaled", "review", f"page-{number}",
                f"Page {number} was read from a downscaled image; small text may be misread",
                page_number=number)
        if not page.get("blank") and not page.get("text_chars"):
            add("empty_ocr_page", "error", f"page-{number}", f"Page {number} returned no OCR text",
                page_number=number)

    items = [row for row in invoice.get("items", []) if row.get("kind") == "item"]
    for info in invoice.get("pages", []):
        number = info["page_number"]
        if info.get("unclosed_table"):
            add("unclosed_table", "error", f"page-{number}",
                f"A table on page {number} stops mid-table in the OCR output", page_number=number)
        lines = info.get("untabled_amount_lines") or []
        if lines:
            add("untabled_amount_lines", "review", f"page-{number}",
                f"Page {number} has {len(lines)} line(s) with an amount outside any table; they were not "
                "extracted as items", "\n".join(line["text"] for line in lines[:10]), page_number=number)
        page_rows = [row for row in invoice.get("items", []) if row["page_number"] == number and row.get("table_number")]
        if info.get("tables") and not page_rows:
            add("no_rows_on_page", "review", f"page-{number}",
                f"Page {number} has a table but no rows were reconstructed from it", page_number=number)

    for table in invoice.get("tables", []):
        accounted = sum(table.get("dispositions", {}).values())
        if accounted < table.get("rows_seen", 0):
            add("rows_unaccounted", "error", f"page-{table['page_number']}",
                f"Table {table['table_number']} on page {table['page_number']} has "
                f"{table['rows_seen'] - accounted} row(s) that were neither extracted nor classified",
                page_number=table["page_number"])
        if table.get("columns_inherited"):
            add("table_continues", "info", f"page-{table['page_number']}",
                f"Table {table['table_number']} on page {table['page_number']} has no header; columns were "
                "taken from the previous table, assuming it continues")

    if items and not any(row.get("kind") == "total" for row in invoice.get("items", [])):
        last_page = max(row["page_number"] for row in items)
        add("missing_grand_total", "review", "invoice",
            "No grand total was found; the table may continue beyond what was read", page_number=last_page)

    _serial_checks(items, add)

    for row in invoice.get("items", []):
        for flag in row.get("flags", []):
            severity = ROW_FLAG_SEVERITY.get(flag["code"], "review")
            add(flag["code"], severity, row["id"], flag["message"], flag.get("raw") or row.get("source_text", ""),
                page_number=row["page_number"])

    by_page = {}
    for row in items:
        by_page.setdefault(row["page_number"], []).append(row)
    numbers = sorted(by_page)
    for before, after in zip(numbers, numbers[1:]):
        last, first = by_page[before][-1], by_page[after][0]
        if after == before + 1 and normalise(last["source_text"]) == normalise(first["source_text"]):
            add("boundary_duplicate", "review", first["id"],
                f"The first row on page {after} repeats the last row on page {before}; it may be read twice",
                first["source_text"])

    candidates = _compare(by_page, crops or {}, add)
    requests = [{"page_number": number, "reasons": sorted(codes)} for number, codes in sorted(retry.items())]
    return list(findings), requests, candidates


def _serial_checks(items, add):
    previous = None
    for row in items:
        serial = (row.get("serial") or "").strip()
        if not serial.isdigit():
            continue
        number = int(serial)
        if previous is not None:
            last_number, last_row = previous
            if number == last_number:
                add("duplicate_serial", "review", row["id"],
                    f"Serial number {number} appears twice in a row; a row may be duplicated", row["source_text"])
            elif number > last_number + 1:
                missing = f"{last_number + 1}" if number == last_number + 2 else f"{last_number + 1}-{number - 1}"
                add("serial_gap", "review", row["id"],
                    f"Serial numbers jump from {last_number} to {number}; row(s) {missing} may be missing",
                    row["source_text"], page_number=row["page_number"])
                if last_row["page_number"] != row["page_number"]:
                    add("serial_gap", "review", last_row["id"],
                        f"Rows after serial {last_number} may be missing at the end of page {last_row['page_number']}",
                        last_row["source_text"], page_number=last_row["page_number"])
            elif number < last_number and number != 1:
                add("serial_order", "review", row["id"],
                    f"Serial number {number} follows {last_number}; row order may be wrong", row["source_text"])
        previous = (number, row)


def _pair(page_rows, crop_rows):
    """Pair rows read by both methods; return (unmatched page rows, unmatched crop rows, value conflicts)."""
    page_left, crop_left = list(page_rows), list(crop_rows)
    conflicts = []
    for same_amount in (True, False):
        for crop in list(crop_left):
            name = normalise(crop["name"])
            best = None
            for row in page_left:
                if same_amount != (row_amount(row) == row_amount(crop)):
                    continue
                ratio = 1.0 if normalise(row["name"]) == name else SequenceMatcher(None, normalise(row["name"]), name).ratio()
                if ratio >= NAME_MATCH_RATIO and (best is None or ratio > best[0]):
                    best = (ratio, row)
            if best:
                page_left.remove(best[1])
                crop_left.remove(crop)
                if not same_amount:
                    conflicts.append((best[1], crop))
    return page_left, crop_left, conflicts


def _compare(by_page, crops, add):
    candidates = []
    for number, crop in sorted(crops.items()):
        if crop.get("errors") and not crop.get("items"):
            add("targeted_ocr_failed", "review", f"page-{number}",
                f"Targeted table OCR for page {number} failed; the page could not be cross-checked",
                "; ".join(crop["errors"])[:2000])
            continue
        if crop.get("truncated"):
            add("targeted_ocr_truncated", "review", f"page-{number}",
                f"Targeted table OCR output for page {number} was cut off in at least one crop")
        crop_items = [row for row in crop.get("items", []) if row.get("kind") == "item"]
        page_left, crop_left, conflicts = _pair(by_page.get(number, []), crop_items)
        for row, other in conflicts:
            add("method_value_disagreement", "review", row["id"],
                f"Targeted table OCR read the amount for '{row['name']}' as {row_amount(other)}; page OCR read "
                f"{row_amount(row)}. Check the source image", other["source_text"])
        for index, row in enumerate(crop_left, start=1):
            candidate = {key: row.get(key) for key in ("name", "quantity", "unit_rate", "gross_amount",
                                                       "discount", "net_amount", "source_text", "serial")}
            candidates.append({**candidate, "id": f"c-{number}-{index}", "page_number": number,
                               "extraction_method": "table_crop_ocr", "status": "pending",
                               "reason": "Found by targeted table OCR but not by full-page OCR"})
        if crop_left:
            add("rows_only_in_targeted_ocr", "review", f"page-{number}",
                f"Targeted table OCR found {len(crop_left)} row(s) on page {number} that page OCR did not; "
                "they are listed as candidate rows", "\n".join(row["source_text"] for row in crop_left[:10]))
        severity = "review" if crop.get("coverage") == "page" else "info"
        for row in page_left:
            add("not_in_targeted_ocr", severity, row["id"],
                f"Targeted table OCR of page {number} did not read '{row['name']}'", row["source_text"])
    return candidates
