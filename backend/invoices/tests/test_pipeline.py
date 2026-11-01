from io import BytesIO
from unittest.mock import patch

import pytest
from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.test import override_settings
from PIL import Image
from rest_framework.test import APIClient

from documents.models import Document
from invoices.confidence import needs_review, score
from invoices.extraction import extract
from invoices.models import InvoiceRun, LineItem, ReviewEvent
from invoices.services import run_job
from invoices.validation import validate
from invoices.workflow import _amount_verdict, _group_verdict
from ocr.models import OcrJob, OcrPage
from preprocessing.models import PreprocessedPage, PreprocessingJob


OCR_TEXT = """Patient Name: Anu
Bill No: B-42
Laboratory Charges
<table><tr><th>Particulars</th><th>Qty</th><th>Rate</th><th>Gross Amount</th><th>Discount</th><th>Net Amount</th></tr>
<tr><td>Blood Test</td><td>2</td><td>50.00</td><td>100.00</td><td>10.00</td><td>90.00</td></tr>
<tr><td>Sub Total</td><td></td><td></td><td></td><td></td><td>90.00</td></tr>
<tr><td>Grand Total</td><td></td><td></td><td></td><td></td><td>90.00</td></tr></table>"""

CLEAN_OCR_TEXT = """Bill No: B-43
Laboratory Charges
<table><tr><th>Particulars</th><th>Qty</th><th>Rate</th><th>Amount</th></tr>
<tr><td>Blood Test</td><td>2</td><td>50.00</td><td>100.00</td></tr>
<tr><td>Urine Test</td><td>1</td><td>40.00</td><td>40.00</td></tr>
<tr><td>Grand Total</td><td></td><td></td><td>140.00</td></tr></table>"""


def test_extract_hierarchy_and_validate():
    invoice = extract([(1, OCR_TEXT)])
    assert invoice["invoice_number"] == "B-42"
    assert invoice["patient_name"] == "Anu"
    assert invoice["field_evidence"]["invoice_number"]["page_number"] == 1
    assert len(invoice["groups"]) == 1
    assert [row["kind"] for row in invoice["items"]] == ["item", "subtotal", "total"]
    assert invoice["items"][0]["group_id"] == invoice["groups"][0]["id"]
    assert invoice["grand_total"] == "90.00"
    assert validate(invoice) == []


def test_arithmetic_mismatch_never_overwrites_source():
    invoice = extract([(1, OCR_TEXT.replace("100.00", "120.00"))])
    issues = validate(invoice)
    assert {issue["code"] for issue in issues} == {"line_gross_mismatch", "line_net_mismatch"}
    assert invoice["items"][0]["gross_amount"] == "120.00"


def test_markdown_table_continuation_requires_review():
    first = "Pharmacy Charges\n| Particulars | Qty | Rate | Net Amount |\n| --- | --- | --- | --- |\n| Tablet | 1 | 10.00 | 10.00 |"
    second = "| Particulars | Qty | Rate | Net Amount |\n| --- | --- | --- | --- |\n| Syringe | 1 | 5.00 | 5.00 |"
    invoice = extract([(1, first), (2, second)])
    assert len(invoice["items"]) == 2
    assert invoice["items"][1]["group_id"] == invoice["groups"][0]["id"]
    assert invoice["items"][1]["relationship"] == "inferred_cross_page"
    assert any(issue["code"] == "cross_page_group" for issue in validate(invoice))


def test_judge_disagreement_stays_unresolved():
    assert _amount_verdict("50.00", "50.00")["verdict"] == "supported"
    assert _amount_verdict("60.00", "50.00")["verdict"] == "unresolved"
    assert _amount_verdict("Blood Test 50", "50.00")["verdict"] == "unresolved"
    invoice = extract([(1, OCR_TEXT)])
    assert _group_verdict("Laboratory Charges", "Laboratory Charges", invoice)["verdict"] == "supported"
    assert _group_verdict("Laboratory Charges", None, invoice)["verdict"] == "unresolved"


def test_actual_nanonets_html_shape_with_blank_colspan_row():
    text = ("Kauvery Hospital\nBill No: B-42\nLaboratory Charges\n<table><tr><th>Particulars</th><th>Qty</th>"
            "<th>Rate</th><th>Amount</th></tr><tr><td colspan='4'></td></tr>"
            "<tr><td>Blood Test</td><td>1</td><td>50.00</td><td>50.00</td></tr>"
            "<tr><td>Grand Total</td><td></td><td></td><td>50.00</td></tr></table>")
    invoice = extract([(1, text)])
    assert len(invoice["items"]) == 2
    assert invoice["provider_name"] == "Kauvery Hospital"
    assert invoice["grand_total"] == "50.00"
    assert validate(invoice) == []


def test_numeric_rate_header_catches_real_bill_amount_confusion():
    text = ("<table><tr><th>Service</th><th>0.00</th><th>Qty</th><th>Code</th><th>Bill Amount</th></tr>"
            "<tr><td colspan='5'><strong>Diet</strong></td></tr>"
            "<tr><td>Diet</td><td>740.00</td><td>2</td><td></td><td>19838.00</td></tr></table>")
    invoice = extract([(1, text)])
    row = invoice["items"][0]
    assert row["unit_rate"] == "740.00"
    assert row["net_amount"] == "19838.00"
    assert validate(invoice)[0]["code"] == "line_net_from_rate_mismatch"


def test_confidence_score_routes_low_scores_to_review():
    clean = extract([(1, CLEAN_OCR_TEXT)])
    confidence = score(clean, validate(clean))
    assert confidence["score"] == 100 and not needs_review(confidence, 80)
    retried = score(clean, validate(clean), [1, 2, 3])
    assert retried["score"] == 70 and needs_review(retried, 80)
    clean["items"][0]["quantity"] = "3"
    flagged = score(clean, validate(clean))
    assert flagged["score"] == 50 and needs_review(flagged, 80)
    error = [{"code": "line_gross_mismatch", "severity": "error", "target_id": "r-0"}]
    many = {"items": [{"id": f"r-{n}", "kind": "item", "net_amount": "1.00"} for n in range(10)]}
    blocked = score(many, error)
    assert blocked["score"] == 90 and blocked["blockers"] and needs_review(blocked, 80)
    empty = score({"items": []}, [])
    assert empty["score"] == 0 and needs_review(empty, 0)


def _png(width=1, height=1):
    if width == 1:
        return b"\x89PNG\r\n\x1a\n"
    buffer = BytesIO()
    Image.new("L", (width, height), 255).save(buffer, format="PNG")
    return buffer.getvalue()


def _completed_ocr(user, image=None):
    doc = Document.objects.create(uploaded_by=user, original_filename="bill.pdf", size_bytes=10,
                                  sha256="0" * 64, page_count=1,
                                  file=ContentFile(b"%PDF-fake", name="bill.pdf"))
    prep = PreprocessingJob.objects.create(document=doc, status="completed", config={}, pages_total=1)
    page = PreprocessedPage(job=prep, page_number=1, width=1, height=1, dpi=100, color_mode="grayscale",
                            denoiser="none", denoise_applied=False, sigma_estimated=0, skew_angle=0,
                            deskew_applied=False, transform=[[1, 0, 0], [0, 1, 0]], quality_before={}, quality_after={})
    page.original_image.save("orig.png", ContentFile(b"\x89PNG\r\n\x1a\n"), save=False)
    page.processed_image.save("processed.png", ContentFile(image or _png()), save=False)
    page.save()
    ocr = OcrJob.objects.create(preprocessing_job=prep, status="completed", model_name="nanonets/Nanonets-OCR2-3B")
    return ocr, page


@pytest.mark.django_db
def test_high_confidence_run_skips_review_and_retry_lowers_score(settings):
    user = get_user_model().objects.create_user("owner", password="pw")
    ocr, page = _completed_ocr(user)
    ocr_page = OcrPage.objects.create(job=ocr, source_page=page, page_number=1, text=CLEAN_OCR_TEXT)
    run = run_job(InvoiceRun.objects.create(ocr_job=ocr).pk)
    assert run.status == "auto_approved" and run.stage == "auto_approved"
    assert run.confidence_score == 100 and run.confidence["threshold"] == 80
    client = APIClient()
    client.force_authenticate(user=user)
    rejected = client.post(f"/api/invoices/runs/{run.pk}/decision/", {"decision": "reject", "note": "Spot check"}, format="json")
    assert rejected.status_code == 200 and rejected.data["status"] == "needs_review"

    ocr_page.raw_response.save("page_0001.json", ContentFile(b'{"_ocr_retry": {"selected_image_max_edge": 2600}}'))
    settings.NANONETS_BASE_URL = ""
    retried = run_job(InvoiceRun.objects.create(ocr_job=ocr).pk)
    # A downscaled page is a completeness warning; with no gateway it cannot be re-read.
    assert retried.status == "needs_review" and retried.confidence["retried_pages"] == [1]
    assert retried.confidence_score == 60
    assert any("completeness" in blocker for blocker in retried.confidence["blockers"])
    assert set(retried.findings.filter(category="completeness").values_list("code", flat=True)) == {
        "ocr_downscaled", "targeted_ocr_skipped"}


TRUNCATED_PAGE = ("Bill No: B-90\nLaboratory Charges\n<table><tr><th>S.No</th><th>Particulars</th><th>Qty</th>"
                  "<th>Rate</th><th>Amount</th></tr><tr><td>1</td><td>CBC</td><td>1</td><td>350.00</td>"
                  "<td>350.00</td></tr><tr><td>2</td><td>LF")
CROP_TEXT = ("<table><tr><th>S.No</th><th>Particulars</th><th>Qty</th><th>Rate</th><th>Amount</th></tr>"
             "<tr><td>1</td><td>CBC</td><td>1</td><td>350.00</td><td>350.00</td></tr>"
             "<tr><td>2</td><td>LFT</td><td>1</td><td>650.00</td><td>650.00</td></tr>"
             "<tr><td>3</td><td>Urea</td><td>1</td><td>90.00</td><td>90.00</td></tr>"
             "<tr><td></td><td>Grand Total</td><td></td><td></td><td>1090.00</td></tr></table>")


@pytest.mark.django_db
@override_settings(NANONETS_BASE_URL="http://127.0.0.1:8090/v1", NANONETS_API_KEY="")
def test_truncated_page_triggers_targeted_retry_and_reviewer_resolution():
    user = get_user_model().objects.create_user("owner", password="pw")
    ocr, page = _completed_ocr(user, _png(900, 1200))
    OcrPage.objects.create(job=ocr, source_page=page, page_number=1, text=TRUNCATED_PAGE, finish_reason="length")
    reply = {"choices": [{"message": {"content": CROP_TEXT}, "finish_reason": "stop"}]}
    with patch("invoices.retry.ask", return_value=reply) as ask_call:
        run = run_job(InvoiceRun.objects.create(ocr_job=ocr).pk)
    assert ask_call.call_count == 1  # one page band, retried once
    assert list(run.stage_events.values_list("name", flat=True)) == [
        "reconstruct_structure", "check_completeness", "targeted_ocr_retry", "check_completeness",
        "validate_invoice", "judge_discrepancies", "score_confidence"]
    attempt = run.retry_attempts.get()
    assert attempt.status == "completed" and attempt.coverage == "page" and "ocr_truncated" in attempt.reasons
    assert attempt.raw_response == reply
    assert run.status == "needs_review"
    # Page OCR rows are kept as read; the truncated "LF" row is not replaced by the crop's "LFT".
    assert list(run.items.filter(kind="item").values_list("name", flat=True)) == ["CBC", "LF"]
    candidates = run.extraction["candidate_rows"]
    assert {(row["name"], row["gross_amount"]) for row in candidates} == {("LFT", "650.00"), ("Urea", "90.00")}
    codes = set(run.findings.values_list("code", flat=True))
    assert {"ocr_truncated", "unclosed_table", "rows_only_in_targeted_ocr", "missing_amount"} <= codes

    client = APIClient()
    client.force_authenticate(user=user)
    urea = next(row["id"] for row in candidates if row["name"] == "Urea")
    lft = next(row["id"] for row in candidates if row["name"] == "LFT")
    assert client.post(f"/api/invoices/runs/{run.pk}/candidates/{urea}/", {"action": "accept"}, format="json").status_code == 400
    accepted = client.post(f"/api/invoices/runs/{run.pk}/candidates/{urea}/",
                           {"action": "accept", "note": "Row 3 is on the scan", "group_id": run.groups.get().stable_id},
                           format="json")
    assert accepted.status_code == 200
    added = run.items.get(stable_id=urea)
    assert added.review_status == LineItem.ReviewStatus.HUMAN_ADDED and added.gross_amount == 90
    assert added.extraction_method == "table_crop_ocr"
    assert client.post(f"/api/invoices/runs/{run.pk}/candidates/{lft}/",
                       {"action": "dismiss", "note": "Same as truncated row 2"}, format="json").status_code == 200
    run.refresh_from_db()
    assert run.findings.get(code="rows_only_in_targeted_ocr").resolved
    corrected = client.patch(f"/api/invoices/runs/{run.pk}/items/r-1-2/",
                             {"name": "LFT", "gross_amount": "650.00", "note": "Checked scan"}, format="json")
    assert corrected.status_code == 200
    row = next(item for item in corrected.data["items"] if item["stable_id"] == "r-1-2")
    assert row["review_status"] == "human_corrected"
    assert all(f["resolved"] for f in corrected.data["findings"] if f["target_id"] == "r-1-2" and f["category"] == "completeness")
    assert client.post(f"/api/invoices/runs/{run.pk}/decision/",
                       {"decision": "approve", "note": "Verified"}, format="json").status_code == 409
    for finding in run.findings.filter(severity="error", resolved=False):
        response = client.post(f"/api/invoices/runs/{run.pk}/findings/{finding.pk}/resolve/",
                               {"note": "Compared every row on the original page"}, format="json")
        assert response.status_code == 200
    approved = client.post(f"/api/invoices/runs/{run.pk}/decision/", {"decision": "approve", "note": "Verified"},
                           format="json")
    assert approved.status_code == 200 and approved.data["status"] == "approved"
    assert ReviewEvent.objects.filter(run=run, action="resolve_finding").exists()
    assert client.get(f"/api/invoices/runs/{run.pk}/pages/1/image/?variant=original").status_code == 200


@pytest.mark.django_db
@override_settings(NANONETS_BASE_URL="http://127.0.0.1:8090/v1", NANONETS_API_KEY="", INVOICE_RETRY_MAX_PAGES=0)
def test_retry_limit_sends_unresolved_pages_to_review():
    user = get_user_model().objects.create_user("owner", password="pw")
    ocr, page = _completed_ocr(user, _png(900, 1200))
    OcrPage.objects.create(job=ocr, source_page=page, page_number=1, text=TRUNCATED_PAGE, finish_reason="length")
    with patch("invoices.retry.ask") as ask_call:
        run = run_job(InvoiceRun.objects.create(ocr_job=ocr).pk)
    ask_call.assert_not_called()
    skipped = run.findings.get(code="targeted_ocr_skipped")
    assert "limit" in skipped.message and run.status == "needs_review"


@pytest.mark.django_db
def test_run_and_human_correction(settings, django_capture_on_commit_callbacks):
    user = get_user_model().objects.create_user("reviewer", password="pw")
    doc = Document.objects.create(uploaded_by=user, original_filename="bill.pdf", size_bytes=10,
                                  sha256="0" * 64, page_count=1,
                                  file=ContentFile(b"%PDF-fake", name="bill.pdf"))
    prep = PreprocessingJob.objects.create(document=doc, status="completed", config={}, pages_total=1)
    page = PreprocessedPage(job=prep, page_number=1, width=1, height=1, dpi=100, color_mode="grayscale",
                            denoiser="none", denoise_applied=False, sigma_estimated=0, skew_angle=0,
                            deskew_applied=False, transform=[[1, 0, 0], [0, 1, 0]], quality_before={}, quality_after={})
    page.original_image.save("orig.png", ContentFile(b"\x89PNG\r\n\x1a\n"), save=False)
    page.processed_image.save("processed.png", ContentFile(b"\x89PNG\r\n\x1a\n"), save=False)
    page.save()
    ocr = OcrJob.objects.create(preprocessing_job=prep, status="completed", model_name="nanonets/Nanonets-OCR2-3B")
    OcrPage.objects.create(job=ocr, source_page=page, page_number=1, text=OCR_TEXT)
    run = run_job(InvoiceRun.objects.create(ocr_job=ocr).pk)
    assert run.status == "auto_approved" and run.confidence_score == 100
    assert run.groups.count() == 1 and run.items.count() == 3
    assert list(run.stage_events.values_list("name", flat=True)) == [
        "reconstruct_structure", "check_completeness", "validate_invoice", "judge_discrepancies",
        "score_confidence"]
    client = APIClient()
    client.force_authenticate(user=user)
    structured = client.get(f"/api/invoices/runs/{run.pk}/").data["structured_items"]
    assert structured == [{"item_name": "Blood Test", "billing_group": "Laboratory Charges",
                           "qty": "2.000", "rate": "50.00", "amount": "90.00", "page_number": 1,
                           "row_order": 1, "source_text": "Blood Test | 2 | 50.00 | 100.00 | 10.00 | 90.00",
                           "extraction_method": "page_ocr", "review_status": "auto_extracted"}]
    assert client.post(f"/api/ocr/jobs/{ocr.pk}/extract/").status_code == 503
    with override_settings(LOCAL_DEMO_MODE=True), patch("config.local_jobs.launch") as launch:
        with django_capture_on_commit_callbacks(execute=True):
            queued = client.post(f"/api/ocr/jobs/{ocr.pk}/extract/")
        assert queued.status_code == 202
        launch.assert_called_once_with("invoice", queued.data["id"])
    assert client.get(f"/api/preprocessing/pages/{page.pk}/processed.png").status_code == 200
    assert client.get(f"/api/invoices/runs/{run.pk}/pages/1/image/").status_code == 200
    response = client.patch(f"/api/invoices/runs/{run.pk}/items/r-1-1/", {"net_amount": "89.00", "note": "Checked scan"}, format="json")
    assert response.status_code == 200
    assert response.data["structured_items"][0]["amount"] == "89.00"
    assert run.items.get(stable_id="r-1-1").net_amount == 89
    assert response.data["status"] == "needs_review" and response.data["confidence_score"] == 0
    assert ReviewEvent.objects.filter(run=run, action="correct_item").count() == 1
    assert client.post(f"/api/invoices/runs/{run.pk}/decision/", {"decision": "approve", "note": "Verified"}, format="json").status_code == 409
    other = get_user_model().objects.create_user("not_owner", password="pw")
    client.force_authenticate(user=other)
    assert client.get(f"/api/invoices/runs/{run.pk}/").status_code == 404
    assert client.get(f"/api/preprocessing/pages/{page.pk}/processed.png").status_code == 404
