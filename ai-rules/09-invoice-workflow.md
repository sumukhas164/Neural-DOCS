# 09 - Nanonets invoice workflow

## What is implemented

- OCR uses `nanonets/Nanonets-OCR2-3B` on a Colab GPU through vLLM, an
  authenticated job gateway, and a temporary Cloudflare Quick Tunnel.
  `NANONETS_API_KEY` is a shared gateway secret, not a Nanonets vendor key.
  Django posts each page and polls the gateway's job endpoint. No OCR model
  weights are required on the local machine.
- OCR stores one response per processed page, including the complete server
  JSON artifact. The model's message content is text with HTML tables where
  it can identify them. Coordinates and confidence values must not be assumed.
- A `finish_reason=length` response is not accepted as complete. The backend
  retries the entire page with a 2600-pixel maximum edge, then 2200 pixels if
  needed. It stores earlier truncated responses under `_ocr_retry` in the raw
  artifact. If every attempt truncates, the OCR job fails for review.
- `invoices.extraction` converts HTML or Markdown tables into a versioned
  `1.0` structure with page numbers, row order, billing groups, item kinds,
  source row text, and nullable amounts. It preserves cross-page group
  continuation as an inferred relationship that needs review.
- `invoices.validation` checks arithmetic and relationships with Decimal.
  Missing operands skip a calculation. Findings never alter extracted values.
- LangGraph runs reconstruction, validation, and at most three targeted judge
  questions. It stores stage snapshots and stage events in MySQL. The Nanonets
  judge is a second prompt to the same model, not an independent model; its
  output never overwrites extraction. Ambiguous answers stay unresolved.
- Every run routes to human review. The document owner can correct item values
  or group assignments with a required note. The original extraction,
  validation findings, decisions, and before/after changes remain stored.
  Approval is blocked on validation errors or when no item rows exist.
- React displays progress, groups, items, source page images, findings, and
  audit history. Images use authenticated API routes.

## Earlier local 1.5B observations (not a 3B benchmark)

- A synthetic invoice produced plain text and an HTML table inside the
  OpenAI-compatible response's `choices[0].message.content`. It did not
  expose boxes, coordinates, or field confidence. The parser found one group,
  one item, and a grand total.
- A cropped billing region from an existing scanned medical invoice produced
  eight group headings and thirteen item rows. Nanonets merged three nearby
  section totals into item amounts. The rate column header was transcribed as
  `0.00`; parsing it as the rate column made Decimal validation flag the three
  affected rows for review. This is a crop, not a full invoice accuracy score.
- That crop took about 344 seconds on the local CPU: roughly 275 seconds of
  prompt/image processing and 69 seconds generating 897 tokens. Full pages
  may take longer.
- A broad JSON judge request returned another transcription. Narrow questions
  about one row's amount or billing section returned short answers. The judge
  therefore uses targeted amount and group questions and treats disagreements
  as unresolved.

## Known gaps

- There is no manually verified invoice dataset or calibrated accuracy result.
- No full invoice was processed end to end after the model switch.
- The Colab runtime and Quick Tunnel URL are temporary. A new session needs a
  new URL in `backend/.env`; Django must restart after configuration changes.
- The gateway key must match in Colab and Django. It protects the public
  tunnel but is not a Nanonets vendor API credential. Never commit or share it.
- The conservative parser will miss rows in unsupported layouts; missing rows
  cannot be reliably detected without source annotations.
- Downscaled OCR retries leave the whole page in view but can lose small text.
  Check amounts against the source image during human review.
- The graph stores stage snapshots but cannot automatically resume a crashed
  worker at its previous node. A fresh run can be created.
- The local React interface relies on Django login. Production auth, object
  storage policy, Redis deployment, and reviewer roles need hardening.
