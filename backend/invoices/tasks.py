from celery import shared_task

from .services import run_job


@shared_task(name="invoices.process_invoice")
def process_invoice_task(run_id):
    run = run_job(run_id)
    return {"run_id": run.pk, "status": run.status}
