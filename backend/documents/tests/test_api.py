import hashlib

import pytest
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.test import APIClient

from documents.models import Document
from preprocessing.models import PreprocessingJob
from preprocessing.tests.helpers import make_page, make_pdf_bytes, rotate_ccw

pytestmark = pytest.mark.django_db


@pytest.fixture
def pdf_bytes():
    return make_pdf_bytes([make_page(), rotate_ccw(make_page(), 3.0)], dpi=100)


@pytest.fixture
def alice():
    return get_user_model().objects.create_user("alice", password="pw")


@pytest.fixture
def client(alice):
    client = APIClient()
    client.force_authenticate(alice)
    return client


def upload(client, data: bytes, name: str = "scan.pdf"):
    file = SimpleUploadedFile(name, data, content_type="application/pdf")
    return client.post("/api/documents/", {"file": file}, format="multipart")


def test_requires_authentication(pdf_bytes):
    anonymous = APIClient()
    assert upload(anonymous, pdf_bytes).status_code in (401, 403)
    assert anonymous.get("/api/documents/").status_code in (401, 403)
    assert anonymous.get("/api/preprocessing/jobs/").status_code in (401, 403)
    assert Document.objects.count() == 0


def test_upload_creates_document_and_runs_layer_1(
    client, alice, pdf_bytes, django_capture_on_commit_callbacks
):
    with django_capture_on_commit_callbacks(execute=True):
        response = upload(client, pdf_bytes, name="My Invoice.pdf")
    assert response.status_code == 201, response.data

    body = response.data
    assert body["original_filename"] == "My Invoice.pdf"
    assert body["page_count"] == 2
    assert body["size_bytes"] == len(pdf_bytes)
    assert body["sha256"] == hashlib.sha256(pdf_bytes).hexdigest()
    assert "file" not in body
    assert body["preprocessing_job"]["status"] == "pending"

    document = Document.objects.get(pk=body["id"])
    assert document.uploaded_by == alice
    # The client-supplied name is never used as a storage path.
    assert "My Invoice" not in document.file.name
    assert document.file.name.startswith("documents/") and document.file.name.endswith(".pdf")

    job = client.get(f"/api/preprocessing/jobs/{body['preprocessing_job']['id']}/").data
    assert job["status"] == "completed"
    assert job["pages_done"] == job["pages_total"] == 2
    assert [p["page_number"] for p in job["pages"]] == [1, 2]
    second = job["pages"][1]
    assert second["deskew_applied"] and second["processed_image"].endswith(".png")
    assert second["skew_angle"] == pytest.approx(3.0, abs=0.4)


@pytest.mark.parametrize(
    "name,data,message",
    [
        ("notes.txt", b"%PDF-1.4 whatever", "Only .pdf"),
        ("fake.pdf", b"just some text, not a pdf at all", "not a PDF"),
        ("broken.pdf", b"%PDF-1.7\nnot really a pdf body", "could not be opened"),
        ("empty.pdf", b"", ""),
    ],
)
def test_rejects_invalid_uploads(client, name, data, message):
    response = upload(client, data, name=name)
    assert response.status_code == 400
    assert message in str(response.data["file"])
    assert Document.objects.count() == 0
    assert PreprocessingJob.objects.count() == 0


def test_rejects_oversized_and_overlong_documents(client, pdf_bytes, settings):
    settings.MAX_UPLOAD_SIZE_BYTES = 1000
    response = upload(client, pdf_bytes)
    assert response.status_code == 400 and "limit" in str(response.data["file"])

    settings.MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024
    settings.MAX_DOCUMENT_PAGES = 1
    response = upload(client, pdf_bytes)
    assert response.status_code == 400 and "2 pages" in str(response.data["file"])


def test_missing_file_field(client):
    assert client.post("/api/documents/", {}, format="multipart").status_code == 400


def test_users_only_see_their_own_documents_and_jobs(
    client, pdf_bytes, django_capture_on_commit_callbacks
):
    with django_capture_on_commit_callbacks(execute=True):
        body = upload(client, pdf_bytes).data
    job_id = body["preprocessing_job"]["id"]

    bob = get_user_model().objects.create_user("bob", password="pw")
    other = APIClient()
    other.force_authenticate(bob)
    assert other.get("/api/documents/").data == []
    assert other.get(f"/api/documents/{body['id']}/").status_code == 404
    assert other.get(f"/api/preprocessing/jobs/{job_id}/").status_code == 404
    assert other.post(f"/api/documents/{body['id']}/preprocess/", {}, format="json").status_code == 404

    assert len(client.get("/api/documents/").data) == 1
    assert client.get(f"/api/documents/{body['id']}/").status_code == 200


def test_rerun_with_options(client, pdf_bytes, django_capture_on_commit_callbacks):
    with django_capture_on_commit_callbacks(execute=True):
        document_id = upload(client, pdf_bytes).data["id"]
        response = client.post(
            f"/api/documents/{document_id}/preprocess/",
            {"options": {"deskew_enabled": False}},
            format="json",
        )
    assert response.status_code == 202
    assert response.data["config"]["deskew_enabled"] is False

    jobs = client.get(f"/api/preprocessing/jobs/?document={document_id}").data
    assert len(jobs) == 2
    rerun = client.get(f"/api/preprocessing/jobs/{response.data['id']}/").data
    assert rerun["status"] == "completed"
    assert all(p["deskew_skip_reason"] == "deskew_disabled" for p in rerun["pages"])


@pytest.mark.parametrize(
    "options",
    [
        {"denoiser": "bm3d"},  # invalid value
        {"tile_size": 64},  # server-controlled, not client-settable
        {"device": "cuda"},
        {"unknown": 1},
        {"fixed_sigma": "abc"},
    ],
)
def test_rerun_rejects_bad_options(client, pdf_bytes, options):
    document_id = upload(client, pdf_bytes).data["id"]
    before = PreprocessingJob.objects.count()
    response = client.post(
        f"/api/documents/{document_id}/preprocess/", {"options": options}, format="json"
    )
    assert response.status_code == 400
    assert PreprocessingJob.objects.count() == before
