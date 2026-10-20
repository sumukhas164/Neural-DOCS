from unittest.mock import patch

import pytest
from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.test import override_settings
from rest_framework.test import APIClient

from documents.models import Document
from invoices.extraction import extract
from invoices.models import InvoiceRun, ReviewEvent
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
    assert run.status == "needs_review"
    assert run.groups.count() == 1 and run.items.count() == 3
    assert list(run.stage_events.values_list("name", flat=True)) == [
        "reconstruct_structure", "validate_invoice", "judge_discrepancies"]
    client = APIClient()
    client.force_authenticate(user=user)
    structured = client.get(f"/api/invoices/runs/{run.pk}/").data["structured_items"]
    assert structured == [{"item_name": "Blood Test", "billing_group": "Laboratory Charges",
                           "qty": "2.000", "rate": "50.00", "amount": "90.00"}]
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
    assert ReviewEvent.objects.filter(run=run, action="correct_item").count() == 1
    assert client.post(f"/api/invoices/runs/{run.pk}/decision/", {"decision": "approve", "note": "Verified"}, format="json").status_code == 409
    other = get_user_model().objects.create_user("not_owner", password="pw")
    client.force_authenticate(user=other)
    assert client.get(f"/api/invoices/runs/{run.pk}/").status_code == 404
    assert client.get(f"/api/preprocessing/pages/{page.pk}/processed.png").status_code == 404
