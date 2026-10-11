# 05 - Project structure

```
Neural-DOCS/
├── README.md                     Quick start
├── .gitignore
├── ai-rules/                     This documentation
└── backend/
    ├── manage.py
    ├── requirements.txt
    ├── pytest.ini                Test settings module, "models" marker
    ├── .env.example              Template for backend/.env (git-ignored)
    ├── config/                   Django project
    │   ├── settings.py           All settings, read from the environment
    │   ├── settings_test.py      Test overrides (SQLite, temp media, no denoiser)
    │   ├── celery.py             Celery app
    │   ├── urls.py               Mounts admin, /api-auth/, /api/
    │   ├── wsgi.py, asgi.py
    ├── documents/                Uploaded PDFs (thin slice of the REST API layer)
    │   ├── models.py             Document
    │   ├── validators.py         PDF upload validation
    │   ├── serializers.py        DocumentSerializer
    │   ├── views.py              DocumentViewSet (upload, list, retrieve)
    │   ├── urls.py, admin.py, apps.py
    │   ├── migrations/0001_initial.py
    │   └── tests/test_api.py     API tests for upload, jobs, permissions
    └── preprocessing/            Layer 1
        ├── pipeline/             Framework-free image pipeline (no Django imports)
        │   ├── __init__.py       Public API
        │   ├── config.py         PreprocessingConfig
        │   ├── imaging.py        Image conventions and conversions
        │   ├── pdf_renderer.py   PDF -> page images
        │   ├── noise.py          Gaussian noise simulation, blind sigma estimation
        │   ├── quality.py        Reference-free quality checks and flags
        │   ├── denoising.py      TV / DnCNN / DRUNet with tiled inference
        │   ├── deskew.py         Otsu + min-area rectangle + rotate + crop
        │   ├── pipeline.py       Stage orchestration, PageResult
        │   ├── metrics.py        The paper's six full-reference metrics
        │   └── evaluation.py     The paper's benchmark protocol
        ├── models.py             PreprocessingJob, PreprocessedPage
        ├── services.py           Jobs, storage, status: bridge between Django and pipeline
        ├── tasks.py              Celery task
        ├── serializers.py        Job, page and option serializers
        ├── views.py              Job status and re-run endpoints
        ├── urls.py, admin.py, apps.py
        ├── management/commands/
        │   ├── preprocess_pdf.py     Run Layer 1 on a file from the shell
        │   └── evaluate_denoisers.py Run the paper's benchmark on a folder
        ├── migrations/0001_initial.py
        └── tests/
            ├── helpers.py            Synthetic pages and PDFs
            ├── test_config.py
            ├── test_noise.py
            ├── test_quality.py
            ├── test_denoising.py
            ├── test_deskew.py
            ├── test_pdf_renderer.py
            ├── test_pipeline.py
            ├── test_metrics_evaluation.py
            └── test_services.py
```

Not in git: `backend/.venv/`, `backend/.env`, `backend/media/`,
`backend/db.sqlite3`, `backend/eval_output/`, `*.pth`.

## Module responsibilities

### `preprocessing/pipeline/` (the core)

| Module | Responsibility | Main names |
| --- | --- | --- |
| `config.py` | The one place for tunables; validation; dict round-trip for DB snapshots | `PreprocessingConfig`, `ConfigError`, `PAPER_NOISE_LEVELS` |
| `imaging.py` | Defines the internal image format and converts at the edges | `to_float`, `to_uint8`, `to_gray`, `validate_image` |
| `pdf_renderer.py` | Opens a PDF and yields pages lazily | `render_pdf`, `count_pages`, `RenderedPage`, `PdfRenderError` |
| `noise.py` | Paper eq. 6 noise model; blind estimator | `add_gaussian_noise`, `estimate_noise_sigma` |
| `quality.py` | Measurements and flags; grayscale detection | `assess_quality`, `QualityReport`, `is_effectively_grayscale`, `FLAG_*` |
| `denoising.py` | Loads and caches models; tiling; dispatch by algorithm | `denoise`, `run_tiled`, `tv_lambda_for`, `DenoiserUnavailableError` |
| `deskew.py` | The paper's four deskew steps | `deskew`, `binarize_text`, `estimate_skew_angle`, `DeskewResult` |
| `pipeline.py` | Runs the stages in order; decides what to skip; collects metadata | `preprocess_image`, `preprocess_pdf`, `PageResult`, `SKIP_*` |
| `metrics.py` | MSE, PSNR, SSIM, MS-SSIM, LPIPS, GMSD via DeepInv | `compute_metrics`, `METRIC_NAMES` |
| `evaluation.py` | Noise -> denoise -> score, and averaging | `evaluate_image`, `summarize`, `load_image` |

Internal image format (defined in `imaging.py`): float32 in [0, 1];
grayscale is `(H, W)`, colour is `(H, W, 3)` in RGB order. Images are uint8
only at the boundaries (renderer output, stored PNGs).

### `preprocessing/` (Django side)

| Module | Responsibility |
| --- | --- |
| `models.py` | Database schema for jobs and pages |
| `services.py` | `get_default_config`, `create_job`, `submit_job`, `run_job`; PNG encoding; writing through the storage API |
| `tasks.py` | `preprocess_document_task`: a three-line wrapper so the queue has something to call |
| `serializers.py` | Output shape of jobs and pages; whitelist of client-settable options |
| `views.py` | `PreprocessingJobViewSet` (read-only), `DocumentPreprocessView` (re-run) |
| `apps.py` | Validates `settings.PREPROCESSING` at startup |

### `documents/`

| Module | Responsibility |
| --- | --- |
| `models.py` | `Document`: owner, original filename (metadata only), stored file, size, SHA-256, page count |
| `validators.py` | Extension, size, `%PDF-` header, opens with pdfium, page limit |
| `views.py` | Upload creates the document and submits a Layer 1 job |

## Dependency direction

```
documents  <--  preprocessing (models, services)  -->  preprocessing.pipeline
    ^                                                         (no Django)
    └── documents.views calls preprocessing.services.submit_job on upload
```

`documents.views` importing from `preprocessing` is the one upward reference.
It is a temporary stand-in for the orchestration layer and should move there
when that layer is built.

## Conventions

- New layers get their own Django app next to `preprocessing/`, with the same
  split: framework-free logic in a subpackage, `services.py` as the bridge.
- Skip reasons and flags are string constants defined next to the code that
  produces them (`pipeline.SKIP_*`, `quality.FLAG_*`); use the constants.
- Comments explain why, and cite the paper section or equation where a step
  comes from it.
- Tests live beside the app they test, in `tests/`, and use synthetic data.
