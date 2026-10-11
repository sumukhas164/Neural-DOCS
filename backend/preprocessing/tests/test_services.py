import json

import cv2
import numpy as np
import pytest
from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.core.management import call_command
from django.core.management.base import CommandError

from documents.models import Document
from preprocessing import services
from preprocessing.models import PreprocessedPage, PreprocessingJob
from preprocessing.pipeline import ConfigError
from preprocessing.tasks import preprocess_document_task

from .helpers import make_page, make_pdf_bytes, rotate_ccw

pytestmark = pytest.mark.django_db


@pytest.fixture
def user():
    return get_user_model().objects.create_user("alice", password="pw")


def make_document(user, pdf_bytes: bytes, page_count: int) -> Document:
    document = Document(
        uploaded_by=user,
        original_filename="scan.pdf",
        size_bytes=len(pdf_bytes),
        sha256="0" * 64,
        page_count=page_count,
    )
    document.file.save("scan.pdf", ContentFile(pdf_bytes), save=True)
    return document


@pytest.fixture
def document(user):
    pdf = make_pdf_bytes([make_page(), rotate_ccw(make_page(color=True), 3.0)], dpi=100)
    return make_document(user, pdf, 2)


def test_default_config_comes_from_settings(settings):
    assert services.get_default_config().denoiser == "none"
    settings.PREPROCESSING = {"denoiser": "tv"}
    assert services.get_default_config().denoiser == "tv"
    settings.PREPROCESSING = {"bogus": 1}
    with pytest.raises(ConfigError):
        services.get_default_config()


def test_create_job_snapshots_config(document):
    job = services.create_job(document, {"deskew_enabled": False})
    assert job.status == PreprocessingJob.Status.PENDING
    assert job.pages_total == 2
    assert job.config["deskew_enabled"] is False
    assert job.config["denoiser"] == "none"
    with pytest.raises(ConfigError):
        services.create_job(document, {"denoiser": "bm3d"})


def test_run_job_stores_pages_and_metadata(document):
    job = services.run_job(services.create_job(document).pk)

    assert job.status == PreprocessingJob.Status.COMPLETED
    assert job.error_message == ""
    assert job.pages_done == 2
    assert job.started_at and job.finished_at and job.finished_at >= job.started_at

    first, second = job.pages.order_by("page_number")
    assert (first.page_number, second.page_number) == (1, 2)
    assert first.color_mode == "grayscale" and not first.deskew_applied
    assert second.deskew_applied and second.skew_angle == pytest.approx(3.0, abs=0.4)
    assert second.crop_box and len(second.transform) == 2
    assert first.denoise_skip_reason == "denoiser_disabled"
    assert first.quality_flags == first.quality_before["flags"]
    assert first.dpi == pytest.approx(100)

    # The stored PNG is exactly the processed page the metadata describes.
    with first.processed_image.open("rb") as handle:
        stored = cv2.imdecode(np.frombuffer(handle.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    assert stored.shape == (first.height, first.width)
    with first.original_image.open("rb") as handle:
        original = cv2.imdecode(np.frombuffer(handle.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    # pdfium rounds page sizes up, so a side may come out one pixel larger.
    assert original.shape[2] == 3
    assert original.shape[0] in (640, 641) and original.shape[1] in (480, 481)
    assert f"job_{job.pk}/page_0001_processed" in first.processed_image.name


def test_run_job_only_runs_pending_jobs(document):
    job = services.create_job(document)
    services.run_job(job.pk)
    page_ids = set(job.pages.values_list("pk", flat=True))
    again = services.run_job(job.pk)
    assert again.status == PreprocessingJob.Status.COMPLETED
    assert set(again.pages.values_list("pk", flat=True)) == page_ids


def test_run_job_records_failure(user):
    broken = make_document(user, b"%PDF-1.4 truncated garbage", 1)
    job = services.run_job(services.create_job(broken).pk)
    assert job.status == PreprocessingJob.Status.FAILED
    assert "PdfRenderError" in job.error_message
    assert job.finished_at is not None
    assert job.pages.count() == 0


def test_celery_task_runs_the_job(document):
    job = services.create_job(document)
    result = preprocess_document_task.delay(job.pk).get()
    assert result == {"job_id": job.pk, "status": "completed", "pages_done": 2}
    assert PreprocessedPage.objects.filter(job=job).count() == 2


def test_submit_job_dispatches_after_commit(document, django_capture_on_commit_callbacks):
    with django_capture_on_commit_callbacks(execute=True) as callbacks:
        job = services.submit_job(document)
    assert len(callbacks) == 1
    job.refresh_from_db()
    assert job.status == PreprocessingJob.Status.COMPLETED


def test_preprocess_pdf_command(tmp_path, capsys):
    pdf = tmp_path / "in.pdf"
    pdf.write_bytes(make_pdf_bytes([rotate_ccw(make_page(), 2.0)], dpi=100))
    out = tmp_path / "out"
    call_command("preprocess_pdf", str(pdf), "--output-dir", str(out), "--save-original")

    assert (out / "page_0001_processed.png").is_file()
    assert (out / "page_0001_original.png").is_file()
    report = json.loads((out / "report.json").read_text())
    assert report["config"]["denoiser"] == "none"
    assert report["pages"][0]["skew_angle"] == pytest.approx(2.0, abs=0.4)
    assert "1 page(s)" in capsys.readouterr().out

    with pytest.raises(CommandError):
        call_command("preprocess_pdf", str(tmp_path / "missing.pdf"), "--output-dir", str(out))


def test_evaluate_denoisers_command(tmp_path, capsys):
    images = tmp_path / "clean"
    images.mkdir()
    cv2.imwrite(str(images / "doc.png"), (make_page(200, 200) * 255).astype(np.uint8))
    csv_path = tmp_path / "results.csv"
    call_command(
        "evaluate_denoisers", str(images), "--algorithms", "tv", "--sigmas", "0.05", "0.1",
        "--size", "176", "--no-lpips", "--output", str(csv_path),
    )
    lines = csv_path.read_text().strip().splitlines()
    assert lines[0].startswith("image,color_mode,algorithm,sigma,mse,psnr")
    assert len(lines) == 3
    assert "grayscale" in capsys.readouterr().out

    with pytest.raises(CommandError):
        call_command("evaluate_denoisers", str(tmp_path / "nowhere"))
