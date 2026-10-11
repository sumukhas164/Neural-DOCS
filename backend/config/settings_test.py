"""Settings for the test suite.

Tests run on SQLite by default so they need no credentials. Set
TEST_DB_ENGINE=mysql to run them against the MySQL server configured in .env
(the database user then needs permission to create ``test_<DB_NAME>``).
"""

import os
import tempfile

os.environ.setdefault("DJANGO_DEBUG", "true")

from .settings import *  # noqa: E402,F401,F403
from .settings import DATABASES  # noqa: E402

if os.environ.get("TEST_DB_ENGINE", "sqlite").lower() != "mysql":
    DATABASES = {
        "default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}
    }

MEDIA_ROOT = tempfile.mkdtemp(prefix="neural_docs_test_media_")
CELERY_TASK_ALWAYS_EAGER = True
MAX_DOCUMENT_PAGES = 5
# Deep models are exercised by dedicated pipeline tests; everything else uses
# the pipeline without a denoiser so the suite stays fast.
PREPROCESSING = {"denoiser": "none", "render_dpi": 100, "max_pages": MAX_DOCUMENT_PAGES}
PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]
