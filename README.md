# Neural-DOCS

Medical invoice processing system: upload a PDF, preprocess its pages, run
Nanonets OCR2-1.5B-exp, reconstruct billing groups and rows, validate amounts,
and send the result to human review.

**Status:** The preprocessing, OCR integration, conservative HTML/Markdown
table extraction, decimal validation, targeted Nanonets judge calls, LangGraph
workflow, review API, and React UI are implemented. A real invoice OCR trace,
accuracy evaluation, Redis worker, and production storage/auth hardening remain
pending. No run is automatically approved. Design notes live in
[`ai-rules/`](ai-rules/README.md).

## Nanonets OCR

Run a local OpenAI-compatible vision server loaded with
`nanonets/Nanonets-OCR2-1.5B-exp`. A launch script is provided at
`scripts/start_nanonets_ocr2_1_5b.ps1`; it expects the quantized model and
vision projector in `models/` and the existing llama.cpp binary in
`D:\nanonets ocr\llama.cpp`.
The local files come from the
[OCR2-1.5B-exp GGUF conversion](https://huggingface.co/Mungert/Nanonets-OCR2-1.5B-exp-GGUF/tree/main):
`Nanonets-OCR2-1.5B-exp-q8_0.gguf` and its matching `q8_0.mmproj`.
They are excluded from Git. Their SHA256 hashes were checked against the
repository metadata before the server was started.
Set `NANONETS_BASE_URL` and, if your server uses an alias, `NANONETS_MODEL` in
`backend/.env`. The default URL is `http://127.0.0.1:8090/v1`. After a
preprocessing job completes, authenticated owners can start OCR with
`POST /api/preprocessing/jobs/<id>/ocr/` and poll `GET /api/ocr/jobs/<id>/`.
The API requires a Celery broker and worker so a long OCR run cannot block the
upload/API process. For a local foreground run without Redis, use
`python manage.py run_nanonets_ocr <preprocessing_job_id>`.
Each page keeps the recognized text and the full server JSON response in storage.
The raw OCR response is retained. Parsed values are checked with Decimal
arithmetic and routed to review; the judge cannot overwrite them.

## Extraction and review

With a Celery broker and worker, `POST /api/ocr/jobs/<id>/extract/` creates an
invoice run. Poll `GET /api/invoices/runs/<id>/` for groups, items, validation
findings, and judge verdicts. The authenticated document owner can correct a
row with `PATCH /api/invoices/runs/<id>/items/<stable_id>/` (a `note` is required)
and make a recorded decision with `POST /api/invoices/runs/<id>/decision/`.
Processed source page images are served through an authenticated endpoint.

Without Redis, use `python manage.py process_invoice <ocr_job_id>` in the
`backend` directory. API OCR and extraction requests return 503 while Celery
runs eagerly, since model calls can take minutes and would block the web process.
The workflow graph starts after OCR, reconstructs structure, validates, runs at
most three targeted judge calls, and persists a reviewable result. It does not
infer absent values or auto-approve a run. Stage snapshots are stored in MySQL;
automatic resume after a worker crash is not implemented.

## Review interface

In another terminal, run `cd frontend`, `npm install`, then `npm run dev`.
Start Django on port 8000 first. Open the Vite URL, use **Django sign in**, then
upload and review invoices. For actual background OCR, configure Redis in
`backend/.env` and run `celery -A config worker -l info --pool=solo` from
`backend` on Windows.

The 1.5B model is experimental; evaluate it on manually checked medical
invoices before treating results as
reliable. The current parser does not support every invoice layout, and the
same Nanonets model supplies OCR and judge verdicts, so the judge is not an
independent model.

A local synthetic invoice returned text plus an HTML table. A billing crop
from a scanned invoice yielded eight groups and thirteen item rows; Decimal
checks flagged three amounts that OCR had mixed with section totals. The crop
took about 5.7 minutes on CPU. These observations are not a full invoice
accuracy benchmark. See [the workflow notes](ai-rules/09-invoice-workflow.md).

## Quick start (backend)

```powershell
cd backend
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env        # then fill in DJANGO_SECRET_KEY and the MySQL password
```

Create the MySQL schema once, then migrate:

```sql
CREATE DATABASE neural_docs CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

```powershell
.venv\Scripts\python manage.py migrate
.venv\Scripts\python manage.py createsuperuser
.venv\Scripts\python manage.py runserver
```

## Try Layer 1 without the database

```powershell
.venv\Scripts\python manage.py preprocess_pdf path\to\scan.pdf --output-dir out --save-original
```

Writes one cleaned PNG per page plus `report.json` (noise level, skew angle,
quality flags, timings). The first run with a deep denoiser downloads the
pretrained weights.

## Tests

```powershell
.venv\Scripts\python -m pytest                  # full suite
.venv\Scripts\python -m pytest -m "not models"  # skip tests that need pretrained weights
```
