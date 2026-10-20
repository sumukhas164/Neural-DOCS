# Neural-DOCS

Medical invoice processing system: upload a PDF, preprocess its pages, run
Nanonets OCR2-3B, reconstruct billing groups and rows, validate amounts,
and send the result to human review.

**Status:** The preprocessing, OCR integration, conservative HTML/Markdown
table extraction, decimal validation, targeted Nanonets judge calls, LangGraph
workflow, review API, and React UI are implemented. A real invoice OCR trace,
accuracy evaluation, Redis worker, and production storage/auth hardening remain
pending. No run is automatically approved. Design notes live in
[`ai-rules/`](ai-rules/README.md).

## Nanonets OCR

Open [`colab/nanonets_ocr2_3b.ipynb`](colab/nanonets_ocr2_3b.ipynb) in Google
Colab using **File → Upload notebook**; do not paste its JSON into a code cell.
To paste the setup into one cell in an existing notebook, copy
[`colab/PASTE_IN_ONE_CELL.py`](colab/PASTE_IN_ONE_CELL.py). For separate cells,
use [`colab/COPY_PASTE.md`](colab/COPY_PASTE.md). Select a GPU runtime and run
the cells. You can add a `NANONETS_API_KEY` secret in Colab, or let the notebook
generate one after tunnel setup. This is a shared gateway secret, not a
Nanonets vendor API key. The notebook installs vLLM in an isolated Python 3.12
environment using uv, loads the official
[`nanonets/Nanonets-OCR2-3B`](https://huggingface.co/nanonets/Nanonets-OCR2-3B)
model, runs an authenticated job gateway, and opens a temporary Cloudflare Quick
Tunnel. The notebook includes an optional synthetic-image check. It prints a
`NANONETS_BASE_URL` ending in `/v1`; copy that URL and the same API key into
`backend/.env`, then restart Django. The tunnel setup uses HTTP/2 and retries
transient connection failures. Keep the Colab runtime and tunnel running
while processing invoices. Each new Colab session produces a new tunnel URL.
Do not commit the API key or put real invoice data in notebook cells.

The backend rejects missing URLs, unauthenticated remote connections, and
non-HTTPS remote URLs before starting OCR. It posts pages to the gateway and
polls for results, which avoids a long single HTTP request through Cloudflare.
The vLLM server listens only on Colab loopback; only the key-protected gateway
is tunnelled. On a shared or regulated deployment, use a durable GPU service
and an access-controlled tunnel. Free Colab and Quick Tunnels can stop without
warning, so they are suitable for temporary development runs.

After a preprocessing job completes, authenticated owners can start OCR with
`POST /api/preprocessing/jobs/<id>/ocr/` and poll `GET /api/ocr/jobs/<id>/`.
In local demo mode, the API starts OCR in a separate local process, so Redis is
not needed and the web request returns immediately. Keep the Colab notebook
running before clicking **Start OCR**. For a foreground run, use
`python manage.py run_nanonets_ocr <preprocessing_job_id>`.
Each page keeps the recognized text and the full server JSON response in storage.
The raw OCR response is retained. Parsed values are checked with Decimal
arithmetic and routed to review; the judge cannot overwrite them.

If a dense page reaches the model's context limit, OCR retries that page at
smaller image sizes while keeping the complete page in view. The stored raw
artifact includes the truncated attempts and the selected retry response.
Every extracted value still requires human review.

## Extraction and review

`POST /api/ocr/jobs/<id>/extract/` creates an
invoice run. Poll `GET /api/invoices/runs/<id>/` for groups, items, validation
findings, and judge verdicts. The authenticated document owner can correct a
row with `PATCH /api/invoices/runs/<id>/items/<stable_id>/` (a `note` is required)
and make a recorded decision with `POST /api/invoices/runs/<id>/decision/`.
Processed source page images are served through an authenticated endpoint.

The review page shows a structured table with item name, billing group, quantity,
rate, and amount. The same rows are returned as `structured_items` by
`GET /api/invoices/runs/<id>/` and can be downloaded as JSON from the page:

```json
[{"item_name":"Blood Test","billing_group":"Laboratory Charges","qty":"2.000","rate":"50.00","amount":"90.00"}]
```

`amount` is the net amount when present, otherwise the gross amount. Missing
values stay `null`; subtotal and grand-total rows are excluded from this item
list. A regular Django user can upload and review their own invoices. A
superuser is not required for extraction or review.

Local demo mode starts extraction in a separate local process without Redis.
For a foreground run, use `python manage.py process_invoice <ocr_job_id>` in the
`backend` directory. Outside local demo mode, API OCR and extraction need a
configured Celery broker and worker.
The workflow graph starts after OCR, reconstructs structure, validates, runs at
most three targeted judge calls, and persists a reviewable result. It does not
infer absent values or auto-approve a run. Stage snapshots are stored in MySQL;
automatic resume after a worker crash is not implemented.

## Review interface

In another terminal, run `cd frontend`, `npm install`, then `npm run dev`.
Start Django on `127.0.0.1:8000` first. With `DJANGO_DEBUG=true` and
`LOCAL_DEMO_MODE=true` in `backend/.env`, open the Vite URL and upload invoices
directly; no sign-in is needed. A dedicated non-admin local identity owns these
documents. Local demo access is limited to loopback requests and disabled when
debug mode is off. In local demo mode, **Start OCR** and **Extract invoice** run
in separate processes. The page shows page progress, failures, and retry
buttons. Logs are written to `backend/logs/app.log`, `backend/logs/ocr-<id>.log`,
and `backend/logs/invoice-<id>.log`; this directory is excluded from Git.
For Celery deployment, configure Redis in `backend/.env` and run
`celery -A config worker -l info --pool=solo` from `backend` on Windows.

Evaluate the 3B model on manually checked medical invoices before treating results as
reliable. The current parser does not support every invoice layout, and the
same Nanonets model supplies OCR and judge verdicts, so the judge is not an
independent model.

No full-invoice accuracy benchmark has been run with the Colab-hosted 3B model.
See [the workflow notes](ai-rules/09-invoice-workflow.md).

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
.venv\Scripts\python manage.py runserver
```

If local demo mode is disabled, create a regular reviewer account with
`.venv\Scripts\python manage.py shell` and enter:

```python
from django.contrib.auth import get_user_model
from getpass import getpass
get_user_model().objects.create_user(username="reviewer", password=getpass("Password: "))
```

Then sign in through the React app. This account has no Django admin privileges.

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
