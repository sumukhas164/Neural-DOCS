# 01 - System architecture

## Purpose

Neural-DOCS turns uploaded medical invoice PDFs into reviewable structured
data. A user uploads a PDF, tracks its status, reviews extracted fields, and
corrects them. GST/IGST tax invoices are out of scope.

Technology stack fixed by the project owner: **Django** backend, **React**
frontend, **MySQL** database.

## The architecture (transcribed from the owner's diagram)

```
React frontend
  Upload PDF, track status, review extracted fields, correct results
        |
Django REST API
  Authentication, upload validation, job creation
        |
Document orchestration and queue
  Celery workers + Redis broker, status tracking
        |
Shared document-processing stages
  Layer 1  Preprocessing        PDF rendering, DRUNet/DnCNN/TV, deskewing, image quality checks
  Layer 2  OCR                  Nanonets OCR2-3B via Colab gateway: page-level text and HTML tables
  Layer 3  Validation + routing OCR quality, layout checks, review routing
  Layer 4  Structured extraction  Conservative parser of OCR tables and headings
  Final validation and reconciliation
                                Arithmetic, schema validation, evidence, human review
        |
  Medical billing and item extraction only
        |
MySQL + file/object storage
  Invoice metadata, extracted JSON, validation results, corrections, audit history
```

## Layer responsibilities

| Component | Owns | Does not own |
| --- | --- | --- |
| React frontend | Upload UI, status display, field review and correction | Any processing |
| Django REST API | Authentication, upload validation, creating jobs | Running the stages |
| Orchestration and queue | Running stages on Celery workers, chaining layers, retries, status | Stage logic |
| **Layer 1 Preprocessing** | Rendering PDF pages to images, denoising, deskewing, measuring image quality | Deciding what to do with a poor page |
| Layer 2 OCR | Page-level text, HTML tables when returned, raw response | Image clean-up |
| Layer 3 Validation and routing | Judging OCR quality, checking layout, routing to review | Extraction |
| Layer 4 structured extraction | Hierarchy and typed output, subject to benchmark | Arithmetic checks |
| Final validation | Arithmetic, schema checks, evidence, human-review flagging | - |
| Storage | Metadata, extracted JSON, validation results, corrections, audit history | - |

A consequence worth remembering: Layer 1 **flags** quality problems
(`blurry`, `low_contrast`, ...) but never rejects a page. Acting on flags is
Layer 3's job.

## Build status

| Component | Status | Notes |
| --- | --- | --- |
| Layer 1 Preprocessing | **Implemented, tested, documented** | See files 02-08 |
| Django project skeleton | Implemented | `backend/config/` |
| `Document` model and upload API | Minimal version implemented | Only what Layer 1 needs to receive a PDF |
| Celery task for Layer 1 | Implemented as an integration point | Runs inline when no broker is configured |
| MySQL | Configured | Apply new OCR and invoice migrations to the target database |
| LangGraph invoice workflow | Implemented | OCR-to-review graph with persisted stage snapshots; no automatic resume yet |
| Layer 2 OCR | Implemented | Nanonets OCR2-3B via Colab gateway and Cloudflare Tunnel; requires live invoice verification |
| Extraction, validation, judge, review | Implemented baseline | Real invoice evaluation and hardening pending |
| React frontend | Implemented | Local Vite review interface |

## What was built outside Layer 1, and why

Layer 1 cannot be exercised inside the system without a way to receive a PDF
and a place to store results, so three thin pieces of neighbouring components
exist. Each is deliberately minimal and is expected to be extended, not
replaced, by its real owner:

- **`documents` app**: the `Document` model and `POST /api/documents/`
  with upload validation. Belongs to the "Django REST API" box.
- **Celery task `preprocessing.preprocess_document`**: the function the
  orchestration layer will schedule. Belongs to the "orchestration" box only
  in the sense that it is what that layer calls.
- **Authentication default**: DRF is set to require an authenticated user
  (session or basic auth). No login UI, tokens or roles exist yet.

## Design principles used so far

- **One Django app per layer** (`preprocessing` now; later `ocr`, `routing`,
  `extraction`, ...), each depending only on `documents` and on the output
  models of the layer before it.
- **Stage logic is plain Python** in a `pipeline/` package with no Django
  imports; a `services.py` bridges it to models and storage.
- **Storage goes through Django's storage API**, so moving from the local
  disk to object storage is a settings change.
- **Every run stores the exact configuration it used**, so results stay
  reproducible when defaults change.
