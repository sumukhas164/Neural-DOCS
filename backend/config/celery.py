"""Celery application: the hook the orchestration layer uses to run pipelines."""

import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("neural_docs")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
