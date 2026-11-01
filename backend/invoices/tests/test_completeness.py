"""Line-item extraction, completeness checks, number parsing and evaluation.

All invoices here are synthetic OCR text in the shapes Nanonets returns.
"""

from invoices.completeness import check
from invoices.evaluation import aggregate, evaluate
from invoices.extraction import extract
from invoices.numbers import parse_number, parse_quantity
from invoices.retry import merge_bands
from invoices.validation import validate

HEADER = "<tr><th>S.No</th><th>Particulars</th><th>Qty</th><th>Rate</th><th>Amount</th></tr>"


def table(*rows, header=True):
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return "<table>" + (HEADER if header else "") + body + "</table>"


def items(invoice):
    return [row for row in invoice["items"] if row["kind"] == "item"]


def codes(findings):
    return {finding["code"] for finding in findings}


def page(number, text, **extra):
    return {"page_number": number, "text_chars": len(text), "finish_reason": "stop", "retried": False,
            "all_truncated": False, "blank": False, **extra}


def test_indian_number_formats_and_ocr_confusions_are_never_guessed():
    assert parse_number("₹1,23,456.00").value == "123456.00"
    assert parse_number("Rs. 1,250/-").value == "1250"
    assert parse_number("123,456.50").value == "123456.50"
    assert parse_number("(100.00)").value == "-100.00"
    assert parse_number("-").value is None and parse_number("-").issue is None
    confused = parse_number("1O0.00")
    assert confused.value is None and confused.issue == "ocr_char_confusion" and confused.suggestion == "100.00"
    comma = parse_number("100,50")
    assert comma.value is None and comma.issue == "ambiguous_separator" and comma.suggestion == "100.50"
    assert parse_number("12,34,5.00").issue == "unparsed"
    assert parse_number("10.505").issue == "too_many_decimals"
    assert parse_quantity("2 Nos").value == "2"


def test_multi_page_table_continuation_keeps_columns_and_group():
    first = "Pharmacy Charges\n" + table(("1", "Paracetamol 500mg", "10", "2.00", "20.00"),
                                         ("2", "Pantoprazole 40mg", "5", "8.00", "40.00"))
    second = table(("3", "Ondansetron Inj", "2", "25.00", "50.00"),
                   ("4", "IV Set", "1", "120.00", "120.00"),
                   ("", "Grand Total", "", "", "230.00"), header=False)
    invoice = extract([(1, first), (2, second)])
    rows = items(invoice)
    assert [row["name"] for row in rows] == ["Paracetamol 500mg", "Pantoprazole 40mg", "Ondansetron Inj", "IV Set"]
    assert rows[2]["quantity"] == "2" and rows[2]["unit_rate"] == "25.00" and rows[2]["serial"] == "3"
    assert rows[2]["relationship"] == "inferred_cross_page" and rows[2]["group_id"] == rows[0]["group_id"]
    assert rows[2]["extraction_method"] == "page_ocr+inherited_columns"
    assert [row["table_number"] for row in rows] == [1, 1, 1, 1]
    assert invoice["grand_total"] == "230.00"
    findings, requests, _ = check(invoice, [page(1, first), page(2, second)], 2)
    assert "table_continues" in codes(findings) and "serial_gap" not in codes(findings)
    assert requests == []
    assert not [f for f in validate(invoice) if f["code"] == "grand_total_mismatch"]


def test_dense_table_keeps_every_row_in_order():
    rows = [(str(n), f"Lab test {n}", "1", f"{n}.00", f"{n}.00") for n in range(1, 61)]
    invoice = extract([(1, "Laboratory Charges\n" + table(*rows))])
    assert [row["serial"] for row in items(invoice)] == [str(n) for n in range(1, 61)]
    assert invoice["tables"][0]["rows_seen"] == 60 + 1
    assert invoice["tables"][0]["dispositions"] == {"header": 1, "item": 60}


def test_wrapped_descriptions_are_joined_and_flagged_not_dropped():
    text = table(("1", "Inj. Ceftriaxone 1g -", "2", "45.00", "90.00"),
                 ("", "vial with diluent", "", "", ""),
                 ("2", "Normal Saline<br>500 ml bottle", "1", "30.00", "30.00"))
    invoice = extract([(1, text)])
    rows = items(invoice)
    assert [row["name"] for row in rows] == ["Inj. Ceftriaxone 1g - vial with diluent", "Normal Saline 500 ml bottle"]
    assert "vial with diluent" in rows[0]["source_text"]
    assert {flag["code"] for flag in rows[0]["flags"]} == {"wrapped_description"}
    assert invoice["tables"][0]["dispositions"]["continuation"] == 1


def test_rows_with_missing_amounts_are_kept_and_flagged():
    text = table(("1", "Dressing", "1", "", ""), ("2", "Consultation", "1", "500.00", "500.00"))
    invoice = extract([(1, text)])
    rows = items(invoice)
    assert [row["name"] for row in rows] == ["Dressing", "Consultation"]
    assert rows[0]["net_amount"] is None and rows[0]["gross_amount"] is None
    assert rows[0]["review_status"] == "needs_review"
    findings, _, _ = check(invoice, [page(1, text)], 1)
    missing = [f for f in findings if f["code"] == "missing_amount"]
    assert missing and missing[0]["target_id"] == rows[0]["id"] and missing[0]["severity"] == "review"


def test_repeated_descriptions_are_separate_charges_unless_identical():
    header = "<tr><th>Date</th><th>Particulars</th><th>Qty</th><th>Rate</th><th>Amount</th></tr>"
    body = "".join(f"<tr><td>0{day}-03-2026</td><td>Room Rent</td><td>1</td><td>1500.00</td><td>1500.00</td></tr>"
                   for day in (1, 2, 3))
    invoice = extract([(1, f"<table>{header}{body}</table>")])
    assert len(items(invoice)) == 3
    assert "possible_duplicate" not in codes(validate(invoice))
    twice = extract([(1, table(("", "Room Rent", "1", "1500.00", "1500.00"), ("", "Room Rent", "1", "1500.00", "1500.00")))])
    assert len(items(twice)) == 2 and "possible_duplicate" in codes(validate(twice))


def test_billing_groups_headings_subtotals_taxes_and_totals_are_distinguished():
    text = ("<table>" + HEADER +
            "<tr><td colspan='5'>Laboratory Charges</td></tr>"
            "<tr><td>1</td><td>CBC</td><td>1</td><td>350.00</td><td>350.00</td></tr>"
            "<tr><td>2</td><td>LFT</td><td>1</td><td>650.00</td><td>650.00</td></tr>"
            "<tr><td></td><td>Total</td><td></td><td></td><td>1000.00</td></tr>"
            "<tr><td colspan='5'>Radiology</td></tr>"
            "<tr><td>3</td><td>X-Ray Chest PA</td><td>1</td><td>400.00</td><td>400.00</td></tr>"
            "<tr><td></td><td>Radiology Total</td><td></td><td></td><td>400.00</td></tr>"
            "<tr><td></td><td>Less: Discount</td><td></td><td></td><td>140.00</td></tr>"
            "<tr><td></td><td>CGST 2.5%</td><td></td><td></td><td>10.00</td></tr>"
            "<tr><td></td><td>Round Off</td><td></td><td></td><td>0.40</td></tr>"
            "<tr><td></td><td>Net Payable</td><td></td><td></td><td>1270.40</td></tr>"
            "<tr><td></td><td>Advance Paid</td><td></td><td></td><td>500.00</td></tr></table>")
    invoice = extract([(1, text)])
    kinds = [(row["name"], row["kind"]) for row in invoice["items"]]
    assert kinds == [("CBC", "item"), ("LFT", "item"), ("Total", "subtotal"), ("X-Ray Chest PA", "item"),
                     ("Radiology Total", "subtotal"), ("Less: Discount", "discount"), ("CGST 2.5%", "tax"),
                     ("Round Off", "adjustment"), ("Net Payable", "total"), ("Advance Paid", "payment")]
    assert [group["name"] for group in invoice["groups"]] == ["Laboratory Charges", "Radiology"]
    assert invoice["items"][3]["group_id"] == invoice["groups"][1]["id"]
    assert invoice["grand_total"] == "1270.40"
    summary = {}
    assert validate(invoice, summary) == []
    assert summary["grand_total"] == {"passed": 1, "failed": 0, "unavailable": 0}
    assert summary["group_subtotal"]["passed"] == 2


def test_merged_cells_are_split_only_when_lines_align():
    aligned = table(("1<br>2", "Blood Test<br>Urine Test", "1<br>1", "100.00<br>40.00", "100.00<br>40.00"))
    invoice = extract([(1, aligned)])
    rows = items(invoice)
    assert [(row["name"], row["gross_amount"]) for row in rows] == [("Blood Test", "100.00"), ("Urine Test", "40.00")]
    assert all(row["extraction_method"] == "page_ocr+split" for row in rows)
    assert all("split_from_merged_cell" in {flag["code"] for flag in row["flags"]} for row in rows)
    misaligned = extract([(1, table(("1", "Blood Test<br>Urine Test", "1", "100.00<br>40.00<br>5.00", "145.00")))])
    row = items(misaligned)[0]
    assert row["unit_rate"] is None and row["gross_amount"] == "145.00"
    assert "merged_values" in {flag["code"] for flag in row["flags"]}


def test_unreadable_numbers_keep_suggestions_for_review():
    invoice = extract([(1, table(("1", "Syringe 5ml", "2", "1O.00", "2O.00"), ("2", "Gauze", "1", "100,50", "100,50")))])
    first, second = items(invoice)
    assert first["unit_rate"] is None and first["gross_amount"] is None
    flag = next(flag for flag in first["flags"] if flag.get("field") == "gross_amount")
    assert flag["code"] == "ocr_char_confusion" and flag["suggestion"] == "20.00"
    assert {flag["code"] for flag in second["flags"]} >= {"ambiguous_separator", "missing_amount"}


def test_truncated_page_keeps_rows_and_requests_targeted_retry():
    text = "<table>" + HEADER + "<tr><td>1</td><td>CBC</td><td>1</td><td>350.00</td><td>350.00</td></tr><tr><td>2</td><td>ES"
    invoice = extract([(1, text)])
    assert [row["name"] for row in items(invoice)] == ["CBC", "ES"]
    assert invoice["pages"][0]["unclosed_table"] is True
    findings, requests, _ = check(invoice, [page(1, text, finish_reason="length")], 1)
    assert {"ocr_truncated", "unclosed_table", "missing_grand_total"} <= codes(findings)
    assert requests[0]["page_number"] == 1 and "ocr_truncated" in requests[0]["reasons"]


def test_missing_pages_serial_gaps_and_untabled_lines_are_reported():
    first = table(("1", "CBC", "1", "350.00", "350.00"), ("2", "LFT", "1", "650.00", "650.00"))
    second = ("Consultation Fee Dr. Rao 500.00\nBill No: 77\n" +
              table(("5", "ECG", "1", "300.00", "300.00"), ("", "Grand Total", "", "", "1800.00"), header=False))
    invoice = extract([(1, first), (2, second)])
    findings, requests, _ = check(invoice, [page(1, first), page(2, second)], 3)
    found = codes(findings)
    assert {"missing_ocr_page", "serial_gap", "untabled_amount_lines"} <= found
    gap = next(f for f in findings if f["code"] == "serial_gap" and f["target_id"] == "r-2-1")
    assert "3-4" in gap["message"]
    assert {request["page_number"] for request in requests} == {1, 2}
    lines = invoice["pages"][1]["untabled_amount_lines"]
    assert [line["amount"] for line in lines] == ["500.00"]


def test_targeted_ocr_disagreement_creates_candidates_without_choosing():
    page_text = table(("1", "CBC", "1", "350.00", "350.00"), ("3", "ECG", "1", "300.00", "300.00"),
                      ("", "Grand Total", "", "", "650.00"))
    invoice = extract([(1, page_text)])
    crop_rows = items(extract([(1, table(("1", "CBC", "1", "350.00", "350.00"), ("2", "LFT", "1", "650.00", "650.00"),
                                         ("3", "ECG", "1", "380.00", "380.00")))], method="table_crop_ocr"))
    crops = {1: {"coverage": "tables", "items": crop_rows, "errors": [], "truncated": False}}
    findings, _, candidates = check(invoice, [page(1, page_text)], 1, crops)
    assert [(c["name"], c["gross_amount"], c["status"]) for c in candidates] == [("LFT", "650.00", "pending")]
    assert {"rows_only_in_targeted_ocr", "method_value_disagreement"} <= codes(findings)
    assert items(invoice)[1]["gross_amount"] == "300.00"  # page OCR value is untouched


def test_overlapping_bands_are_merged_without_duplicates():
    rows = [{"source_text": f"{n} | Item {n}"} for n in range(1, 8)]
    assert merge_bands([rows[:4], rows[2:6], rows[5:]]) == rows


def test_evaluation_metrics_against_verified_rows():
    extracted = items(extract([(1, table(("1", "CBC", "1", "350.00", "350.00"), ("2", "LFT", "1", "600.00", "600.00"),
                                         ("3", "LFT", "1", "600.00", "600.00"), ("4", "Urea", "1", "90.00", "90.00")))]))
    truth = [{"page_number": 1, "description": "CBC", "quantity": "1", "rate": "350.00", "amount": "350.00"},
             {"page_number": 1, "description": "LFT", "quantity": "1", "rate": "650.00", "amount": "650.00"},
             {"page_number": 1, "description": "Creatinine", "amount": "120.00"},
             {"page_number": 1, "description": "Urea", "amount": "90.00"}]
    result = evaluate(extracted, truth)
    assert result["recall"] == 0.75 and result["precision"] == 0.75
    assert result["missing"] == ["Creatinine"] and result["duplicate_rows"] == 1
    assert result["field_accuracy"]["amount"] == 2 / 3 and result["field_accuracy"]["description"] == 1.0
    assert aggregate([result, result])["recall"] == 0.75
