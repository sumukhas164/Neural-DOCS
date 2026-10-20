# 06 - Data flow, formats and integration points

## End-to-end flow (as built)

```
Client                    Django                          Worker (or inline)             Storage
  |  POST /api/documents/   |                                   |                           |
  |------------------------>| validate PDF                      |                           |
  |                         | save Document  ------------------------------------------->  documents/<uuid>.pdf
  |                         | create PreprocessingJob (pending) |                           |
  |                         | on commit: enqueue task --------->|                           |
  |  201 {document, job}    |                                   | claim job (running)       |
  |<------------------------|                                   | for each page:            |
  |                         |                                   |   render -> quality ->    |
  |  GET /api/preprocessing/jobs/<id>/                          |   denoise -> deskew       |
  |------------------------>|                                   |   save PreprocessedPage -->  preprocessing/job_<id>/page_NNNN_*.png
  |  200 {status, pages...} |                                   |   pages_done += 1         |
  |<------------------------|                                   | mark completed / failed   |
```

When `CELERY_BROKER_URL` is empty (the default), the task runs inline in the
web process straight after the upload transaction commits. With a broker
configured it runs on a Celery worker.

## Current OCR and invoice continuation

After preprocessing completes, `POST /api/preprocessing/jobs/<id>/ocr/`
creates an OCR job. In local demo mode, a separate local process reads each
processed PNG, sends it with a Bearer gateway key through the current
Cloudflare Quick Tunnel, and polls the Colab gateway until vLLM returns OCR2-3B
text. The full model response is stored per page. The API exposes progress at
`GET /api/ocr/jobs/<id>/`.

The invoice workflow reconstructs billing groups and item rows from OCR text,
checks arithmetic with Decimal, asks targeted questions of the same model for
uncertain rows, and routes every run to human review. `backend/invoices/`
stores stage events and reviewer edits. Its judge is not independent of the
OCR model. See `09-invoice-workflow.md` for the current limits.

## Input

| Property | Requirement | Enforced in |
| --- | --- | --- |
| Type | PDF only: `.pdf` extension **and** a `%PDF-` header in the first 1024 bytes **and** opens with pdfium | `documents/validators.py` |
| Size | At most `MAX_UPLOAD_SIZE_MB` (25) | same |
| Pages | 1 to `MAX_DOCUMENT_PAGES` (50) | same, and `max_pages` in the pipeline |
| Protection | Password-protected PDFs are rejected | same |
| Filename | Kept as metadata only; the stored name is a random UUID | `documents/models.py` |

The framework-free pipeline also accepts a single image array through
`preprocess_image`, which is how the tests and the evaluation harness use it.

## Output per page

### Files (PNG, lossless)

| File | Content |
| --- | --- |
| `page_NNNN_original.png` | The page exactly as rendered, RGB |
| `page_NNNN_processed.png` | The cleaned page: 8-bit grayscale for colourless pages, RGB otherwise. **This is what Layer 2 reads.** |

### Metadata (`PreprocessedPage` row, also `PageResult.metadata()`)

| Field | Type | Meaning |
| --- | --- | --- |
| `page_number` | int | 1-based |
| `width`, `height` | int | Size of the processed image in pixels |
| `dpi` | float | Effective render DPI |
| `color_mode` | str | `grayscale` or `color` |
| `denoiser` | str | Algorithm configured for the run |
| `denoise_applied` | bool | Whether denoising actually ran |
| `denoise_skip_reason` | str | `denoiser_disabled`, `blank_page`, `noise_below_min_sigma`, or empty |
| `sigma_estimated` | float | Blind noise estimate on the [0, 1] scale |
| `sigma_used` | float or null | Sigma given to the denoiser |
| `skew_angle` | float | Detected skew, degrees, counter-clockwise positive |
| `deskew_applied` | bool | Whether the page was rotated |
| `deskew_skip_reason` | str | `deskew_disabled`, `blank_page`, `angle_below_threshold`, or empty |
| `crop_box` | `[x, y, w, h]` or null | Crop applied after rotation, in rotated-image pixels |
| `transform` | 2x3 matrix | Maps original-image pixel `(x, y, 1)` to processed-image pixel |
| `quality_before`, `quality_after` | object | `width, height, dpi, brightness, contrast, sharpness, ink_ratio, chroma, noise_sigma, flags` |
| `quality_flags` | list of str | Copy of `quality_before.flags` |
| `timings_ms` | object | `quality_before, denoise, deskew, quality_after` |

Quality flags: `low_resolution`, `blurry`, `low_contrast`, `too_dark`,
`overexposed`, `blank_page`, `high_noise`.

## Database schema (MySQL)

```
auth_user 1---* documents_document 1---* preprocessing_preprocessingjob 1---* preprocessing_preprocessedpage
```

**`documents_document`**: `id`, `uploaded_by_id`, `original_filename`,
`file`, `size_bytes`, `sha256` (indexed), `page_count`, `created_at`.

**`preprocessing_preprocessingjob`**: `id`, `document_id`, `status`
(indexed), `config` (JSON), `pages_total`, `pages_done`, `error_message`,
`created_at`, `started_at`, `finished_at`.

**`preprocessing_preprocessedpage`**: `id`, `job_id`, `page_number`, the two
file paths, and the metadata fields above. Unique on `(job_id, page_number)`.

A document can have several jobs (one per run). Nothing is overwritten:
re-running creates a new job with its own pages.

## REST API

All endpoints require an authenticated user and only expose that user's data.

| Method and path | Purpose | Success |
| --- | --- | --- |
| `POST /api/documents/` (multipart, field `file`) | Upload a PDF; starts Layer 1 | `201` with the document and `preprocessing_job` |
| `GET /api/documents/` | List own documents | `200` |
| `GET /api/documents/<id>/` | One document | `200` |
| `POST /api/documents/<id>/preprocess/` | Re-run Layer 1, optional `{"options": {...}}` | `202` with the new job |
| `GET /api/preprocessing/jobs/` (`?document=<id>`) | List own jobs | `200` |
| `GET /api/preprocessing/jobs/<id>/` | Status, progress and per-page results | `200` |

Options a client may set on a re-run: `denoiser`, `sigma_mode`,
`fixed_sigma`, `deskew_enabled`, `crop_to_content`. Resource-related options
(tile size, device, DPI, page limits) are server-side only.

Errors: `400` validation (message under the field name), `401/403`
unauthenticated, `404` not found or not owned.

## Layer 1 hand-off and remaining integration work

### Layer 2 (OCR) consumes

```python
job = PreprocessingJob.objects.get(pk=..., status=PreprocessingJob.Status.COMPLETED)
for page in job.pages.all():          # ordered by page_number
    with page.processed_image.open("rb") as f:
        png_bytes = f.read()
    page.dpi, page.width, page.height, page.color_mode, page.quality_flags
```

- Skip OCR for pages flagged `blank_page`.
- OCR coordinates are in **processed-image** pixels. To draw them on the
  original scan, invert `page.transform`:
  `cv2.invertAffineTransform(np.array(page.transform))`.

### Layer 3 (validation and routing) consumes

`quality_flags`, `quality_before`, `quality_after`, `sigma_estimated`,
`skew_angle`. Layer 1 has already measured these; Layer 3 decides what they
mean (re-scan request, human review, proceed).

### Orchestration layer takes over

- The call `preprocessing_services.submit_job(document)` in
  `documents/views.py`. Replace it with a workflow that runs
  `preprocessing.preprocess_document`, then the Layer 2 task, and so on.
- Retries. `run_job` deliberately does not retry and does not raise: it
  records `failed` with `error_message`. A failed job is not re-runnable
  (only `pending` jobs start); create a new job to retry.
- Unified status and automatic recovery across layers. Preprocessing, OCR,
  and invoice runs each have status, but there is no durable cross-layer
  coordinator or automatic resume after a crash.

### Frontend consumes

Upload endpoint, job polling endpoint, and the two image URLs per page for a
before/after view. Page images now use authenticated API routes; Django does
not expose the media directory directly.

## Calling the pipeline directly

```python
from preprocessing.pipeline import PreprocessingConfig, preprocess_pdf, preprocess_image

config = PreprocessingConfig(denoiser="drunet")          # all defaults follow the paper
for page in preprocess_pdf("scan.pdf", config):          # path or bytes
    page.processed          # uint8 array
    page.metadata()         # JSON-serialisable dict
```
