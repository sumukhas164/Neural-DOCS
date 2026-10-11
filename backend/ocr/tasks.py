from celery import shared_task

from . import services


@shared_task(name="ocr.run_nanonets")
def run_ocr_task(job_id):
    job = services.run_job(job_id)
    return {"job_id": job.pk, "status": job.status, "pages_done": job.pages_done}
