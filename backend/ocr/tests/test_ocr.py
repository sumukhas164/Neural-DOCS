import json
from unittest.mock import patch

import pytest
from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from rest_framework.test import APIClient

from documents.models import Document
from ocr.client import OcrError, recognize
from ocr.models import OcrJob
from ocr.services import run_job
from preprocessing.models import PreprocessedPage, PreprocessingJob


def test_client_sends_png_and_retains_response():
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return b'{"choices":[{"message":{"content":"<table>bill</table>"},"finish_reason":"stop"}]}'

    with patch("ocr.client.request.urlopen", return_value=Response()) as call:
        result = recognize(b"\x89PNG\r\n\x1a\nimage", url="http://127.0.0.1:8090/v1",
                           model="nanonets/Nanonets-OCR2-1.5B-exp", timeout=3, max_tokens=100)
    sent = json.loads(call.call_args.args[0].data)
    assert sent["model"] == "nanonets/Nanonets-OCR2-1.5B-exp"
    assert sent["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert result["choices"][0]["message"]["content"] == "<table>bill</table>"
    with pytest.raises(OcrError):
        recognize(b"wrong", url="http://127.0.0.1:8090/v1", model="nanonets", timeout=3, max_tokens=100)


@pytest.mark.django_db
def test_ocr_job_keeps_raw_response_and_page_order():
    user = get_user_model().objects.create_user("ocr_user", password="pw")
    document = Document.objects.create(uploaded_by=user, original_filename="scan.pdf",
                                       size_bytes=10, sha256="0" * 64, page_count=2,
                                       file=ContentFile(b"%PDF-fake", name="scan.pdf"))
    source = PreprocessingJob.objects.create(document=document, status="completed", config={}, pages_total=2)
    for number in (1, 2):
        page = PreprocessedPage(job=source, page_number=number, width=1, height=1, dpi=100,
                                color_mode="grayscale", denoiser="none", denoise_applied=False,
                                sigma_estimated=0, skew_angle=0, deskew_applied=False,
                                transform=[[1, 0, 0], [0, 1, 0]], quality_before={}, quality_after={})
        page.original_image.save(f"original_{number}.png", ContentFile(b"\x89PNG\r\n\x1a\n"), save=False)
        page.processed_image.save(f"processed_{number}.png", ContentFile(b"\x89PNG\r\n\x1a\n"), save=False)
        page.save()
    with patch("ocr.services.recognize", side_effect=[
        {"choices": [{"message": {"content": "page one"}, "finish_reason": "stop"}]},
        {"choices": [{"message": {"content": "page two"}, "finish_reason": "stop"}]},
    ]):
        job = run_job(OcrJob.objects.create(preprocessing_job=source, model_name="nanonets/Nanonets-OCR2-1.5B-exp").pk)
    assert job.status == "completed" and job.pages_done == 2
    assert list(job.pages.values_list("text", flat=True)) == ["page one", "page two"]
    first = job.pages.first()
    with first.raw_response.open("rb") as artifact:
        assert json.load(artifact)["choices"][0]["message"]["content"] == "page one"
    assert run_job(job.pk).pages_done == 2
    client = APIClient()
    client.force_authenticate(user=user)
    assert client.get(f"/api/ocr/jobs/{job.pk}/").status_code == 200
    assert client.post(f"/api/preprocessing/jobs/{source.pk}/ocr/").status_code == 503
    other = get_user_model().objects.create_user("other", password="pw")
    client.force_authenticate(user=other)
    assert client.get(f"/api/ocr/jobs/{job.pk}/").status_code == 404
