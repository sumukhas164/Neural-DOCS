"""Django settings for Neural-DOCS.

All environment-specific values are read from the process environment, which
is populated from ``backend/.env`` when that file exists (see ``.env.example``).
"""

import os
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_list(name: str, default: str = "") -> list[str]:
    return [item.strip() for item in os.environ.get(name, default).split(",") if item.strip()]


DEBUG = env_bool("DJANGO_DEBUG", False)
LOCAL_DEMO_MODE = DEBUG and env_bool("LOCAL_DEMO_MODE", False)

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "")
if not SECRET_KEY:
    if not DEBUG:
        raise ImproperlyConfigured("DJANGO_SECRET_KEY must be set when DJANGO_DEBUG is off")
    SECRET_KEY = "insecure-development-key-do-not-use-in-production"

ALLOWED_HOSTS = env_list("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "documents",
    "preprocessing",
    "ocr",
    "invoices",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

# --- Database --------------------------------------------------------------
# MySQL is the project database. DB_ENGINE=sqlite exists only so the test
# suite and a first local run work on a machine without MySQL credentials.
DB_ENGINE = os.environ.get("DB_ENGINE", "mysql").lower()
if DB_ENGINE == "mysql":
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.mysql",
            "NAME": os.environ.get("DB_NAME", "neural_docs"),
            "USER": os.environ.get("DB_USER", "root"),
            "PASSWORD": os.environ.get("DB_PASSWORD", ""),
            "HOST": os.environ.get("DB_HOST", "127.0.0.1"),
            "PORT": os.environ.get("DB_PORT", "3306"),
            "OPTIONS": {
                "charset": "utf8mb4",
                "init_command": "SET sql_mode='STRICT_TRANS_TABLES'",
            },
        }
    }
elif DB_ENGINE == "sqlite":
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }
else:
    raise ImproperlyConfigured("DB_ENGINE must be 'mysql' or 'sqlite'")

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"

# --- File / object storage -------------------------------------------------
# Uploaded PDFs and page images go through Django's storage API, so swapping
# the local filesystem for object storage later is a settings-only change.
MEDIA_URL = "/media/"
MEDIA_ROOT = Path(os.environ.get("MEDIA_ROOT", BASE_DIR / "media"))

# --- REST API --------------------------------------------------------------
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "config.local_auth.LocalDemoAuthentication",
        "rest_framework.authentication.SessionAuthentication",
        "rest_framework.authentication.BasicAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.IsAuthenticated"],
}

# --- Upload validation -----------------------------------------------------
MAX_UPLOAD_SIZE_BYTES = int(os.environ.get("MAX_UPLOAD_SIZE_MB", "25")) * 1024 * 1024
DATA_UPLOAD_MAX_MEMORY_SIZE = MAX_UPLOAD_SIZE_BYTES
MAX_DOCUMENT_PAGES = int(os.environ.get("MAX_DOCUMENT_PAGES", "50"))

# --- Celery ----------------------------------------------------------------
# Without a broker configured, tasks run inline in the calling process so the
# pipeline is usable before the orchestration layer (workers + Redis) exists.
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "") or None
CELERY_TASK_ALWAYS_EAGER = env_bool("CELERY_TASK_ALWAYS_EAGER", not CELERY_BROKER_URL)
CELERY_TASK_EAGER_PROPAGATES = False
CELERY_TASK_ACKS_LATE = True
CELERY_WORKER_PREFETCH_MULTIPLIER = 1

# --- Layer 1: preprocessing ------------------------------------------------
# Overrides for preprocessing.pipeline.config.PreprocessingConfig defaults.
# Unknown keys are rejected at startup by the preprocessing app.
PREPROCESSING = {
    key: caster(os.environ[env_name])
    for key, env_name, caster in (
        ("denoiser", "PREPROCESSING_DENOISER", str),
        ("sigma_mode", "PREPROCESSING_SIGMA_MODE", str),
        ("fixed_sigma", "PREPROCESSING_FIXED_SIGMA", float),
        ("render_dpi", "PREPROCESSING_RENDER_DPI", int),
        ("tile_size", "PREPROCESSING_TILE_SIZE", int),
        ("device", "PREPROCESSING_DEVICE", str),
    )
    if os.environ.get(env_name)
}
PREPROCESSING.setdefault("max_pages", MAX_DOCUMENT_PAGES)

# Local OpenAI-compatible server loaded with Nanonets-OCR2-1.5B-exp.
NANONETS_BASE_URL = os.environ.get("NANONETS_BASE_URL", "http://127.0.0.1:8090/v1")
NANONETS_MODEL = os.environ.get("NANONETS_MODEL", "nanonets/Nanonets-OCR2-1.5B-exp")
if "ocr2-3b" in NANONETS_MODEL.lower():
    raise ImproperlyConfigured("OCR2-3B is no longer configured; use Nanonets OCR2-1.5B-exp")
NANONETS_TIMEOUT_SECONDS = int(os.environ.get("NANONETS_TIMEOUT_SECONDS", "1800"))
NANONETS_MAX_TOKENS = int(os.environ.get("NANONETS_MAX_TOKENS", "8192"))

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "root": {"handlers": ["console"], "level": os.environ.get("LOG_LEVEL", "INFO")},
}
