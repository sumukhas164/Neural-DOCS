import json
from io import BytesIO
from unittest.mock import patch

import pytest
from PIL import Image
from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.test import override_settings
from rest_framework.test import APIClient

from documents.models import Document
from ocr.client import OcrError, recognize, validate_connection
from ocr.models import OcrJob
from ocr.services import _recognize_complete, run_job
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
                           model="nanonets/Nanonets-OCR2-3B", timeout=3, max_tokens=100)
    sent = json.loads(call.call_args.args[0].data)
    assert sent["model"] == "nanonets/Nanonets-OCR2-3B"
    assert sent["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert result["choices"][0]["message"]["content"] == "<table>bill</table>"
    with pytest.raises(OcrError):
        recognize(b"wrong", url="http://127.0.0.1:8090/v1", model="nanonets", timeout=3, max_tokens=100)


def test_remote_gateway_requires_https_key_and_polls_with_auth():
    for url, key in [("http://example.com/v1", "secret"), ("https://example.com/v1", "")]:
        with pytest.raises(OcrError):
            validate_connection(url, key)

    class Response:
        def __init__(self, status, body):
            self.status, self.body = status, body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(self.body).encode()

    job_id = "e02e903e-b4a8-49d5-aa37-d6663ce3eb57"
    responses = [
        Response(202, {"id": job_id}),
        Response(202, {"id": job_id, "status": "running"}),
        Response(200, {"choices": [{"message": {"content": "<table>done</table>"}}]}),
    ]
    with patch("ocr.client.request.urlopen", side_effect=responses) as open_url, \
         patch("ocr.client.time.sleep"):
        result = recognize(b"\x89PNG\r\n\x1a\nimage", url="https://example.com/v1/",
                           model="nanonets/Nanonets-OCR2-3B", api_key="secret",
                           timeout=30, max_tokens=100)
    assert result["choices"][0]["message"]["content"] == "<table>done</table>"
    assert open_url.call_count == 3
    for call in open_url.call_args_list:
        assert call.args[0].get_header("Authorization") == "Bearer secret"
    assert open_url.call_args_list[0].args[0].full_url == "https://example.com/v1/chat/completions"
    assert open_url.call_args_list[1].args[0].full_url.endswith("/v1/jobs/" + job_id)


def test_poll_timeout_does_not_cancel_colab_job():
    from ocr.client import ask

    job_id = "e02e903e-b4a8-49d5-aa37-d6663ce3eb57"
    completed = {"choices": [{"message": {"content": "complete"}, "finish_reason": "stop"}]}
    timeout_error = OcrError("Nanonets server request failed: timed out")
    timeout_error.__cause__ = TimeoutError("timed out")
    with patch("ocr.client._read_response", side_effect=[
        (202, {"id": job_id}), timeout_error, (200, completed),
    ]) as read_response, patch("ocr.client.time.sleep"):
        result = ask(b"\x89PNG\r\n\x1a\nimage", "OCR", url="https://example.com/v1",
                     model="nanonets/Nanonets-OCR2-3B", timeout=60, max_tokens=100,
                     api_key="secret")
    assert result == completed
    assert read_response.call_count == 3


def test_truncated_page_retries_at_lower_resolution_and_keeps_raw_attempts():
    image = Image.new("RGB", (800, 3000), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    truncated = {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]}
    complete = {"choices": [{"message": {"content": "all rows"}, "finish_reason": "stop"}]}

    with patch("ocr.services.recognize", side_effect=[truncated, complete]) as recognize_call:
        result, artifact = _recognize_complete(buffer.getvalue(), page_number=3, model="ocr2")

    assert result == complete
    assert recognize_call.call_count == 2
    with Image.open(BytesIO(recognize_call.call_args_list[1].args[0])) as resized:
        assert max(resized.size) == 2600
    assert artifact["choices"] == complete["choices"]
    assert artifact["_ocr_retry"]["earlier_attempts"][0]["response"] == truncated
    assert artifact["_ocr_retry"]["selected_image_max_edge"] == 2600


def test_truncated_page_fails_if_all_resolutions_are_truncated():
    image = Image.new("RGB", (800, 3000), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    truncated = {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]}

    with patch("ocr.services.recognize", return_value=truncated) as recognize_call:
        with pytest.raises(OcrError, match="Page 3 OCR was truncated"):
            _recognize_complete(buffer.getvalue(), page_number=3, model="ocr2")
    assert recognize_call.call_count == 3


@override_settings(NANONETS_BASE_URL="http://127.0.0.1:8090/v1", NANONETS_API_KEY="")
@pytest.mark.django_db
def test_ocr_job_keeps_raw_response_and_page_order(django_capture_on_commit_callbacks):
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
        job = run_job(OcrJob.objects.create(preprocessing_job=source, model_name="nanonets/Nanonets-OCR2-3B").pk)
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
    with override_settings(NANONETS_BASE_URL="", LOCAL_DEMO_MODE=True):
        response = client.post(f"/api/preprocessing/jobs/{source.pk}/ocr/")
        assert response.status_code == 503 and "NANONETS_BASE_URL" in response.data["detail"]
    with override_settings(LOCAL_DEMO_MODE=True), patch("config.local_jobs.launch") as launch:
        with django_capture_on_commit_callbacks(execute=True):
            response = client.post(f"/api/preprocessing/jobs/{source.pk}/ocr/")
        assert response.status_code == 202
        launch.assert_called_once_with("ocr", response.data["id"])
    other = get_user_model().objects.create_user("other", password="pw")
    client.force_authenticate(user=other)
    assert client.get(f"/api/ocr/jobs/{job.pk}/").status_code == 404
