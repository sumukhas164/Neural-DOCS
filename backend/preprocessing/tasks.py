from celery import shared_task

from . import services


@shared_task(name="preprocessing.preprocess_document")
def preprocess_document_task(job_id: int) -> dict:
    """Run Layer 1 for one job. Failures are recorded on the job, not raised."""
    job = services.run_job(job_id)
    return {"job_id": job.pk, "status": job.status, "pages_done": job.pages_done}
