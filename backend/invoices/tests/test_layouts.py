"""Hospital bill layouts seen in practice, rebuilt as synthetic OCR text.

Summary + detail pages, section totals printed in colspan cells, totals blocks,
receipt tables, OCR-invented headers on continuation pages, returns with
negative quantities, descriptions split across pages, and totals whose label
gives no hint. All names and numbers are invented.
"""

import pytest
from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from rest_framework.test import APIClient

from documents.models import Document
from invoices.completeness import check
from invoices.extraction import _infer_columns, extract
from invoices.models import InvoiceRun
from invoices.services import run_job
from invoices.validation import validate
from ocr.models import OcrJob, OcrPage
from preprocessing.models import PreprocessedPage, PreprocessingJob

DETAIL_HEADER = ("<tr><th>Service Date</th><th>Service Name</th><th>Rate</th><th>Qty</th><th>Gross Amount</th>"
                 "<th>Discount</th><th>Net Amount</th><th>Patient Payable</th><th>Payor Payable</th></tr>")


def detail(date, name, rate, qty, amount):
    return (f"<tr><td>{date}</td><td>{name}</td><td>{rate}</td><td>{qty}</td><td>{amount}</td><td></td>"
            f"<td>{amount}</td><td>0.00</td><td>{amount}</td></tr>")


def section(name):
    return f'<tr><td colspan="9">{name} (SAC- 999311)</td></tr>'


def section_total(name, amount):
    return (f'<tr><td colspan="4">{name} (SAC- 999311) Total</td><td>{amount}</td><td></td><td>{amount}</td>'
            f"<td>0.00</td><td>{amount}</td></tr>")


SUMMARY_PAGE = (
    "## Tax Invoice (Summarized)\nBill No. : TST-1\n"
    "<table><thead><tr><th>S.No.</th><th>Department Name</th><th>HSN/SAC</th><th>Gross Amount</th>"
    "<th>Discount</th><th>Net Amount</th><th>Patient Payable</th><th>Payor Payable</th></tr></thead><tbody>"
    "<tr><td>1</td><td>CONSULTATION VISIT</td><td>999311</td><td>460.00</td><td></td><td>460.00</td><td>0.00</td><td>460.00</td></tr>"
    "<tr><td>2</td><td>LABORATORY SERVICES</td><td>999311</td><td>800.00</td><td></td><td>800.00</td><td>0.00</td><td>800.00</td></tr>"
    '<tr><td colspan="4">Total</td><td>1,260.00</td><td></td><td>0.00</td><td>1,260.00</td></tr></tbody></table>'
    "<table><thead><tr><th>Receipt No.</th><th>Receipt Date</th><th>Amount</th><th>Payment Mode</th></tr></thead>"
    "<tbody><tr><td>RC-77</td><td>4 Apr, 2024</td><td>1.00</td><td>Cash</td></tr></tbody></table>"
    "| Net Amount | 1,260.00 |\n|---|---|\n| Taxable Amount | 0.00 |\n| CGST | 0.00 |\n| SGST | 0.00 |\n"
    "| Total Amount (With GST) | 1,260.00 |\n| Net Payable by Patient | 0.60 |\n| Paid by Patient | 1.00 |\n"
    "| Round Off Amount | 0.40 |\n| Payor Payable | 1,259.00 |\n| Payor Balance | 1,259.00 |\n"
)
DETAIL_PAGE_1 = (
    "Tax Invoice\n<table><thead>" + DETAIL_HEADER + "</thead><tbody>" + section("CONSULTATION VISIT")
    + detail("01 Apr 2024", "VISIT-SPECIALITY || Dr. A", "230.00", "1", "230.00")
    + detail("02 Apr 2024", "VISIT-SPECIALITY || Dr. A", "230.00", "1", "230.00")
    + section_total("CONSULTATION VISIT", "460.00") + section("LABORATORY SERVICES")
    + detail("01 Apr 2024", "COMPLETE BLOOD COUNT || WARD TWO SECOND", "450.00", "1", "450.00")
    + "</tbody></table>")
# Page 2 continues the table: OCR returns empty headings and an extra empty column.
DETAIL_PAGE_2 = (
    "<table><thead><tr>" + "<th></th>" * 10 + "</tr></thead><tbody>"
    "<tr><td>01 Apr 2024</td><td>FLOOR</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>"
    "<tr><td>01 Apr 2024</td><td>IRON PROFILE || WARD</td><td>175.00</td><td>2</td><td>350.00</td><td></td><td></td>"
    "<td>350.00</td><td>0.00</td><td>350.00</td></tr>"
    "<tr><td>03 Apr 2024</td><td>GLOVES STERILE - Batch: B1</td><td>-50.00</td><td>-3</td><td>-150.00</td><td></td><td></td>"
    "<td>-150.00</td><td>0.00</td><td>-150.00</td></tr>"
    "<tr><td>03 Apr 2024</td><td>GLOVES STERILE - Batch: B1</td><td>50.00</td><td>3</td><td>150.00</td><td></td><td></td>"
    "<td>150.00</td><td>0.00</td><td>150.00</td></tr>"
    '<tr><td colspan="4">LABORATORY SERVICES (SAC- 999311) Total</td><td>800.00</td><td></td><td></td><td>800.00</td>'
    "<td>0.00</td><td>800.00</td></tr>"
    '<tr><td colspan="4">Total</td><td>1,260.00</td><td></td><td></td><td>1,260.00</td><td>0.00</td><td>1,260.00</td></tr>'
    "</tbody></table>")


def bill():
    return extract([(1, SUMMARY_PAGE), (2, DETAIL_PAGE_1), (3, DETAIL_PAGE_2)])


def rows(invoice, kind):
    return [row for row in invoice["items"] if row["kind"] == kind]


def test_summary_page_totals_block_and_receipts_are_not_items():
    invoice = bill()
    names = [row["name"] for row in rows(invoice, "item")]
    assert names == ["VISIT-SPECIALITY || Dr. A", "VISIT-SPECIALITY || Dr. A",
                     "COMPLETE BLOOD COUNT || WARD TWO SECOND FLOOR", "IRON PROFILE || WARD",
                     "GLOVES STERILE - Batch: B1", "GLOVES STERILE - Batch: B1"]
    summary = rows(invoice, "summary")
    assert {row["name"] for row in summary} == {"CONSULTATION VISIT", "LABORATORY SERVICES"}
    assert all(any(flag["code"] in ("summary_of_detail", "footer_line") for flag in row["flags"]) for row in summary)
    assert [row["name"] for row in rows(invoice, "payment")] == [
        "RC-77", "Net Payable by Patient", "Paid by Patient", "Payor Payable", "Payor Balance"]
    assert {row["name"] for row in rows(invoice, "tax")} == {"Taxable Amount", "CGST", "SGST"}
    # The amount payable by the patient is not the invoice total.
    assert invoice["grand_total"] == "1260.00"


def test_colspan_section_totals_group_items_and_reconcile():
    invoice = bill()
    subtotals = rows(invoice, "subtotal")
    assert [row["name"] for row in subtotals if "SAC" in row["name"]] == [
        "CONSULTATION VISIT (SAC- 999311) Total", "LABORATORY SERVICES (SAC- 999311) Total"]
    groups = {group["id"]: group["name"] for group in invoice["groups"]}
    lab = [row for row in rows(invoice, "item") if groups.get(row["group_id"], "").startswith("LABORATORY")]
    assert len(lab) == 4
    summary = {}
    findings = validate(invoice, summary)
    assert not [f for f in findings if f["severity"] == "error"]
    assert summary["group_subtotal"]["failed"] == 0 and summary["grand_total"]["passed"] == 1
    completeness, _, _ = check(invoice, [], 0)
    assert "section_items_short" not in {f["code"] for f in completeness}


def test_continuation_page_with_shifted_columns_returns_and_split_description():
    invoice = bill()
    by_name = {row["name"]: row for row in rows(invoice, "item")}
    iron = by_name["IRON PROFILE || WARD"]
    assert (iron["quantity"], iron["unit_rate"], iron["gross_amount"]) == ("2", "175.00", "350.00")
    assert "columns_by_arithmetic" in {flag["code"] for flag in iron["flags"]}
    returned = [row for row in rows(invoice, "item") if row["gross_amount"] == "-150.00"][0]
    assert (returned["quantity"], returned["unit_rate"]) == ("-3", "-50.00")
    blood = by_name["COMPLETE BLOOD COUNT || WARD TWO SECOND FLOOR"]
    assert "FLOOR" in blood["source_text"] and blood["page_number"] == 2
    assert not any(row["name"].startswith(("01 Apr", "03 Apr")) for row in invoice["items"])
    assert "line_gross_mismatch" not in {f["code"] for f in validate(invoice)}


def test_invented_header_on_continuation_page_is_overruled_by_arithmetic():
    page = ("<table><thead><tr><th>DATE</th><th>DESCRIPTION</th><th>AMOUNT</th><th>QUANTITY</th><th>UNIT PRICE</th>"
            "<th>NET AMOUNT</th></tr></thead><tbody>"
            "<tr><td>10 May 2025</td><td>NEBULIZATION</td><td>100.00</td><td>2</td><td>200.00</td><td>200.00</td></tr>"
            "<tr><td>11 May 2025</td><td>GRBS</td><td>90.00</td><td>1</td><td>90.00</td><td>90.00</td></tr>"
            "<tr><td>12 May 2025</td><td>GRBS</td><td>90.00</td><td>3</td><td>270.00</td><td>270.00</td></tr>"
            "</tbody></table>")
    items = rows(extract([(1, page)]), "item")
    assert [(row["quantity"], row["unit_rate"], row["gross_amount"]) for row in items] == [
        ("2", "100.00", "200.00"), ("1", "90.00", "90.00"), ("3", "90.00", "270.00")]
    assert all(row["extraction_method"].endswith("+columns_by_arithmetic") for row in items)


def test_correct_header_is_kept_when_it_fits_the_numbers():
    page = ("<table><tr><th>Item</th><th>Qty</th><th>Rate</th><th>Amount</th></tr>"
            "<tr><td>Syringe</td><td>4</td><td>12.50</td><td>50.00</td></tr>"
            "<tr><td>Gauze</td><td>2</td><td>30.00</td><td>60.00</td></tr></table>")
    items = rows(extract([(1, page)]), "item")
    assert [(row["quantity"], row["unit_rate"]) for row in items] == [("4", "12.50"), ("2", "30.00")]
    assert all(row["extraction_method"] == "page_ocr" for row in items)


def test_infer_columns_needs_consistent_evidence():
    def row(*cells):
        return {"cells": list(cells), "spans": [1] * len(cells)}
    found = _infer_columns([row("A", "3", "20.00", "60.00"), row("B", "2", "15.00", "30.00")])
    assert (found["quantity"], found["unit_rate"], found["gross_amount"]) == (1, 2, 3)
    assert _infer_columns([row("A", "3", "20.00", "61.00"), row("B", "2", "15.00", "33.00")]) is None
    assert _infer_columns([row("A", "3", "20.00", "60.00")], min_hits=2) is not None  # one row, one hit
    assert _infer_columns([row("A", "01 Apr 2024", "2", "2.00")]) is None  # dates are not numbers


def test_totals_without_any_total_wording_are_found_by_arithmetic():
    # A layout whose section and invoice totals carry no "total" label at all.
    page = ("<table><tr><th>Particulars</th><th>Qty</th><th>Rate</th><th>Amount</th></tr>"
            "<tr><td colspan='4'>Pharmacy</td></tr>"
            "<tr><td>Tab A</td><td>2</td><td>10.00</td><td>20.00</td></tr>"
            "<tr><td>Tab B</td><td>1</td><td>30.00</td><td>30.00</td></tr>"
            "<tr><td>Pharmacy</td><td></td><td></td><td>50.00</td></tr>"
            "<tr><td colspan='4'>Ward</td></tr>"
            "<tr><td>Bed day</td><td>2</td><td>100.00</td><td>200.00</td></tr>"
            "<tr><td>Diet</td><td>2</td><td>25.00</td><td>50.00</td></tr>"
            "<tr><td>Ward</td><td></td><td></td><td>250.00</td></tr>"
            "<tr><td>Amount</td><td></td><td></td><td>300.00</td></tr></table>")
    invoice = extract([(1, page)])
    kinds = [(row["name"], row["kind"]) for row in invoice["items"]]
    assert kinds == [("Tab A", "item"), ("Tab B", "item"), ("Pharmacy", "subtotal"), ("Bed day", "item"),
                     ("Diet", "item"), ("Ward", "subtotal"), ("Amount", "total")]
    assert invoice["grand_total"] == "300.00"
    assert all("total_by_arithmetic" in {f["code"] for f in row["flags"]}
               for row in invoice["items"] if row["kind"] != "item")
    assert not [f for f in validate(invoice) if f["severity"] == "error"]


def test_item_equal_to_earlier_sum_stays_an_item_when_it_has_quantity_and_rate():
    page = ("<table><tr><th>Particulars</th><th>Qty</th><th>Rate</th><th>Amount</th></tr>"
            "<tr><td>Tab A</td><td>1</td><td>20.00</td><td>20.00</td></tr>"
            "<tr><td>Tab B</td><td>1</td><td>30.00</td><td>30.00</td></tr>"
            "<tr><td>Tab C</td><td>5</td><td>10.00</td><td>50.00</td></tr></table>")
    assert [row["kind"] for row in extract([(1, page)])["items"]] == ["item", "item", "item"]


def test_sections_come_from_totals_when_headings_are_missing():
    page = ("<table><tr><th>Date</th><th>Description</th><th>Rate</th><th>Qty</th><th>Amount</th></tr>"
            "<tr><td colspan='5'>Physiotherapy</td></tr>"
            "<tr><td>9 May 2025</td><td>Exercise</td><td>100.00</td><td>2</td><td>200.00</td></tr>"
            "<tr><td colspan='4'>Physiotherapy Total</td><td>200.00</td></tr>"
            "<tr><td>9 May 2025</td><td>Spinal block</td><td>500.00</td><td>1</td><td>500.00</td></tr>"
            "<tr><td>9 May 2025</td><td>Sedation</td><td>300.00</td><td>1</td><td>300.00</td></tr>"
            "<tr><td colspan='4'>ANESTHESIOLOGY Total</td><td>800.00</td></tr></table>")
    invoice = extract([(1, page)])
    groups = {group["id"]: group["name"] for group in invoice["groups"]}
    assert [groups[row["group_id"]] for row in rows(invoice, "item")] == [
        "Physiotherapy", "ANESTHESIOLOGY", "ANESTHESIOLOGY"]
    assert not [f for f in validate(invoice) if f["code"] == "group_total_mismatch"]


def test_unlabelled_total_closes_the_open_section_without_creating_one():
    page = ("<table><tr><th>Particulars</th><th>Qty</th><th>Rate</th><th>Amount</th></tr>"
            "<tr><td colspan='4'>Laboratory</td></tr>"
            "<tr><td>CBC</td><td>1</td><td>300.00</td><td>300.00</td></tr>"
            "<tr><td>LFT</td><td>1</td><td>500.00</td><td>500.00</td></tr>"
            "<tr><td>Sub Total</td><td></td><td></td><td>800.00</td></tr></table>")
    invoice = extract([(1, page)])
    assert [group["name"] for group in invoice["groups"]] == ["Laboratory"]
    assert rows(invoice, "subtotal")[0]["group_id"] == invoice["groups"][0]["id"]


def test_section_short_of_its_printed_total_requests_a_retry_without_inventing_rows():
    pages = [(1, DETAIL_PAGE_1.replace(detail("02 Apr 2024", "VISIT-SPECIALITY || Dr. A", "230.00", "1", "230.00"), ""))]
    invoice = extract(pages)
    findings, requests, _ = check(invoice, [], 0)
    short = [f for f in findings if f["code"] == "section_items_short"]
    assert short and "230.00" in short[0]["message"] and short[0]["severity"] == "review"
    assert requests and "section_items_short" in requests[0]["reasons"]
    assert len(rows(invoice, "item")) == 2  # nothing was added to make the sums agree


@pytest.mark.django_db
def test_summary_and_detail_bill_end_to_end(settings):
    settings.NANONETS_BASE_URL = ""
    user = get_user_model().objects.create_user("layouts", password="pw")
    doc = Document.objects.create(uploaded_by=user, original_filename="bill.pdf", size_bytes=10, sha256="1" * 64,
                                  page_count=3, file=ContentFile(b"%PDF-fake", name="bill.pdf"))
    prep = PreprocessingJob.objects.create(document=doc, status="completed", config={}, pages_total=3)
    ocr = OcrJob.objects.create(preprocessing_job=prep, status="completed", model_name="nanonets/Nanonets-OCR2-3B")
    for number, text in enumerate((SUMMARY_PAGE, DETAIL_PAGE_1, DETAIL_PAGE_2), start=1):
        page = PreprocessedPage(job=prep, page_number=number, width=1, height=1, dpi=100, color_mode="grayscale",
                                denoiser="none", denoise_applied=False, sigma_estimated=0, skew_angle=0,
                                deskew_applied=False, transform=[[1, 0, 0], [0, 1, 0]], quality_before={},
                                quality_after={})
        page.original_image.save("o.png", ContentFile(b"\x89PNG\r\n\x1a\n"), save=False)
        page.processed_image.save("p.png", ContentFile(b"\x89PNG\r\n\x1a\n"), save=False)
        page.save()
        OcrPage.objects.create(job=ocr, source_page=page, page_number=number, text=text, finish_reason="stop")
    run = run_job(InvoiceRun.objects.create(ocr_job=ocr).pk)
    assert run.status in ("auto_approved", "needs_review")
    assert run.items.filter(kind="item").count() == 6
    assert run.items.filter(kind="summary").exists() and run.items.filter(kind="tax").count() == 3
    client = APIClient()
    client.force_authenticate(user=user)
    data = client.get(f"/api/invoices/runs/{run.pk}/").data
    names = [row["item_name"] for row in data["structured_items"]]
    assert len(names) == 6 and not {"CGST", "SGST", "Taxable Amount", "Payor Balance"} & set(names)
    assert data["validation_summary"]["grand_total"]["passed"] == 1
    assert not [f for f in data["findings"] if f["severity"] == "error"]


def test_unknown_label_inside_a_totals_block_is_a_summary_line_not_a_charge():
    block = ("| Net Amount | 500.00 |\n|---|---|\n| CGST | 0.00 |\n| Paid by Patient | 10.00 |\n"
             "| Hospital Share | 12.00 |\n")
    invoice = extract([(1, "<table><tr><th>Item</th><th>Amount</th></tr><tr><td>Bed</td><td>500.00</td></tr></table>"),
                       (2, block)])
    share = next(row for row in invoice["items"] if row["name"] == "Hospital Share")
    assert share["kind"] == "summary" and "footer_line" in {flag["code"] for flag in share["flags"]}
    assert [row["name"] for row in rows(invoice, "item")] == ["Bed"]
