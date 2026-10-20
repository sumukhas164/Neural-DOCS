# 07 - Configuration and testing

## Environment variables (`backend/.env`, template in `.env.example`)

| Variable | Default | Purpose |
| --- | --- | --- |
| `DJANGO_DEBUG` | `false` | Debug mode; also enables serving media files |
| `DJANGO_SECRET_KEY` | none | Required unless debug is on |
| `DJANGO_ALLOWED_HOSTS` | `localhost,127.0.0.1` | Comma-separated |
| `DB_ENGINE` | `mysql` | `sqlite` is allowed for a first local run without MySQL credentials |
| `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT` | `neural_docs`, `root`, empty, `127.0.0.1`, `3306` | MySQL connection |
| `MEDIA_ROOT` | `backend/media` | Where uploads and page images are stored |
| `MAX_UPLOAD_SIZE_MB` | `25` | Upload size limit |
| `MAX_DOCUMENT_PAGES` | `50` | Page limit per document |
| `CELERY_BROKER_URL` | empty | Empty means tasks run inline |
| `CELERY_TASK_ALWAYS_EAGER` | true when no broker | Force inline or queued execution |
| `PREPROCESSING_DENOISER` | `drunet` | `drunet`, `dncnn`, `tv`, `none` |
| `PREPROCESSING_SIGMA_MODE` | `estimate` | `estimate` or `fixed` |
| `PREPROCESSING_FIXED_SIGMA` | `0.05` | Sigma for `fixed` mode |
| `PREPROCESSING_RENDER_DPI` | `300` | Render resolution |
| `PREPROCESSING_TILE_SIZE` | `512` | Tile side, multiple of 8 |
| `PREPROCESSING_DEVICE` | `auto` | `auto`, `cpu`, `cuda` |
| `LOG_LEVEL` | `INFO` | Root log level |
| `NANONETS_BASE_URL` | empty | Current Colab Quick Tunnel HTTPS URL ending in `/v1` |
| `NANONETS_API_KEY` | empty | Secret shared with the Colab gateway; not a vendor API key |
| `NANONETS_MODEL` | `nanonets/Nanonets-OCR2-3B` | vLLM served model ID |
| `NANONETS_TIMEOUT_SECONDS` | `1800` | Overall page OCR timeout, including gateway polling |
| `NANONETS_MAX_TOKENS` | `8192` | Maximum model response tokens |

`settings.PREPROCESSING` is a dict of overrides for `PreprocessingConfig`.
Any field can be set there in code; only the six above have environment
variables. An unknown or invalid key stops Django at startup.

## `PreprocessingConfig` reference

Source column: **paper** means the value is taken from the paper; **eng**
means an engineering default that the paper does not specify.

| Field | Default | Source | Meaning |
| --- | --- | --- | --- |
| `render_dpi` | 300 | paper | PDF render resolution |
| `max_page_pixels` | 40,000,000 | eng | Larger pages render at a lower DPI |
| `max_pages` | None | eng | Stop after this many pages |
| `denoiser` | `drunet` | paper | Best performer in the paper |
| `sigma_mode` | `estimate` | eng | `fixed` reproduces the paper's known-sigma setting |
| `fixed_sigma` | 0.05 | paper | Lowest of the paper's three levels |
| `min_sigma` | 0.01 | eng | Below this, the page is treated as clean |
| `max_sigma` | 0.20 | paper | Highest level the paper validated |
| `tv_lambda` | None | eng | Fixed TV weight; overrides the next field |
| `tv_lambda_per_sigma` | 0.5 | eng | TV weight as a multiple of sigma (calibrated) |
| `tv_max_iterations` | 300 | eng | TV solver iteration cap |
| `tile_size` | 512 | eng | Matches the paper's 512x512 working size |
| `tile_overlap` | 32 | eng | Blended overlap between tiles |
| `device` | `auto` | eng | CUDA if present, else CPU |
| `deskew_enabled` | True | paper | |
| `min_skew_angle` | 0.1 | eng | Smaller rotations are skipped |
| `crop_to_content` | True | paper | Step 4 of the deskew algorithm |
| `crop_padding` | 16 | eng | Margin kept around the content, pixels |
| `min_effective_dpi` | 150 | eng | `low_resolution` flag |
| `min_sharpness` | 50.0 | eng | `blurry` flag |
| `min_contrast` | 0.20 | eng | `low_contrast` flag |
| `min_brightness`, `max_brightness` | 0.25, 0.98 | eng | `too_dark`, `overexposed` flags |
| `blank_ink_ratio` | 0.0005 | eng | `blank_page` flag |
| `grayscale_chroma_threshold` | 0.02 | eng | Below this a page is treated as grayscale |

## Running things

All commands run from `backend/` with the virtual environment's Python.
On Windows, use `.\.venv\Scripts\python.exe` explicitly so a system Python
without Celery does not launch Django.

```powershell
# Layer 1 on a file, no database needed
.\.venv\Scripts\python.exe manage.py preprocess_pdf scan.pdf --output-dir out [--denoiser tv] [--sigma-mode fixed --fixed-sigma 0.1] [--dpi 200] [--max-pages 3] [--no-deskew] [--save-original]

# The paper's benchmark on a folder of clean scans
.\.venv\Scripts\python.exe manage.py evaluate_denoisers clean_scans/ --output eval_output/results.csv [--algorithms drunet tv] [--sigmas 0.05 0.1 0.2] [--size 512] [--no-lpips]

# Server
.\.venv\Scripts\python.exe manage.py migrate
.\.venv\Scripts\python.exe manage.py runserver

# Worker (only when CELERY_BROKER_URL is set)
celery -A config worker --pool=solo -l info
```

`--pool=solo` is needed on Windows, where Celery's default pool is not
supported.

For the local demo, Django can run without a superuser. If account login is
enabled, create a regular user as shown in the root README. Run the Colab
notebook first, then set `NANONETS_BASE_URL` and `NANONETS_API_KEY` in the
ignored `backend/.env`, restart Django, and start OCR from the React app.
The Colab notebook's generated key is printed only after the tunnel is ready.
The tunnel uses HTTP/2 and retries startup; its URL changes after a new
Colab session. Do not include the key in logs, commits, or screenshots.

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest                 # full backend suite
.\.venv\Scripts\python.exe -m pytest -m "not models" # skip pretrained-weight tests
.\.venv\Scripts\python.exe -m pytest -m models       # deep denoiser model tests
```

Tests use `config/settings_test.py`: in-memory SQLite, a temporary media
folder, inline Celery, `denoiser="none"` and 100 dpi rendering so that
database and API tests are fast. Set `TEST_DB_ENGINE=mysql` to run them
against MySQL instead.

### What is covered

| File | Tests | Covers |
| --- | --- | --- |
| `test_config.py` | 13 | Paper defaults, validation of every bounded field, unknown keys, dict round-trip |
| `test_noise.py` | 14 | Noise statistics, clipping, reproducibility; estimator on gray and colour pages at four sigmas, on a pure-white background, on clean and tiny images |
| `test_quality.py` | 9 | Every flag, grayscale detection, JSON serialisation |
| `test_denoising.py` | 22 | Tile coverage and overlap, exact reconstruction with an identity function, tile sizes divisible by 8, TV on gray and colour, DRUNet at two sigmas on gray and colour, DRUNet beating TV, DnCNN not degrading |
| `test_deskew.py` | 27 | Detection of eight known angles on gray and colour with the correct sign, full correction, upright pages untouched, threshold, crop keeps all text, coordinate transform, blank page |
| `test_pdf_renderer.py` | 6 | Page count, size at a requested DPI, RGB channel order, page and pixel limits, invalid input |
| `test_pipeline.py` | 13 | Clean pass-through, noisy and skewed page end to end, stage order, fixed-sigma mode, sigma cap, disabling stages, blank pages, colour handling, metadata, multi-page PDFs, default DRUNet pipeline |
| `test_metrics_evaluation.py` | 10 | The six metrics by name, ordering with noise, MSE and PSNR against their formulas, MS-SSIM on small images, LPIPS, the benchmark protocol and summary, image loading |
| `test_services.py` | 9 | Settings-driven config, job creation, stored files and metadata, idempotent start, failure recording, Celery task, on-commit dispatch, both management commands |
| `documents/tests/test_api.py` | 15 | Authentication required, successful upload through to completed job, four kinds of invalid upload, size and page limits, per-user isolation, re-run with options, rejected options |
| `ocr/tests/test_ocr.py` | varies | Gateway URL and key validation, OCR request/polling, OCR jobs |
| `invoices/tests/test_pipeline.py` | varies | Extraction, validation, judge workflow, review API |
| `colab/test_gateway.py` | 1 | Gateway authentication, job polling, and model response flow |

### Latest verification

The current backend suite passed with `129 passed, 10 deselected` using
`-m "not models"` on Windows and SQLite. The gateway test passed separately.
The ten pretrained-weight tests and a live Colab-to-Django invoice run were
not rerun for this change. The earlier 138-test full run predates the OCR2-3B
integration and does not verify it.

### Not covered by automated tests

- **MySQL.** No credentials were available, so migrations have not been
  applied to MySQL and the suite has only run on SQLite. The models use
  standard field types (including `JSONField`, supported by MySQL 8).
- **A real Celery worker with Redis.** Only inline execution is tested.
- **GPU inference.**
- **Full-resolution deep denoising of a whole page** (too slow on the dev
  CPU for the suite); tiling is tested at small sizes and was run manually on
  a 704x1000 crop.

## Manual verification performed

On the owner's five-page sample bill (300 dpi, grayscale), using the
`preprocess_pdf` command with default settings: 5 pages in 13.7 s; sigma
estimated at 0.0000 on every page so denoising was skipped; skew of -0.30 to
-1.07 degrees detected and corrected; page 2 (nearly empty) flagged
`low_contrast` and `overexposed`. Denoising on the same document with
injected noise is reported in `03-layer1-implementation.md`.

## Reference measurements (synthetic 512x512 gray page, CPU)

| Algorithm | PSNR at sigma 0.05 | 0.10 | 0.20 | Time per tile |
| --- | --- | --- | --- | --- |
| Noisy input | 26.2 | 21.2 | 16.3 | - |
| DRUNet | 36.4 | 32.5 | 24.5 | 10-16 s |
| TV (factor 0.5) | 32.3 | 27.0 | 20.9 | 3-4 s |
| DnCNN | 26.4 | 21.3 | 16.3 | 3-5 s |

Colour, DRUNet: 39.2 / 35.8 / 26.3 dB.

These are from a single synthetic page and are a sanity check, not a
reproduction of the paper's tables, which need its 40-image dataset.
