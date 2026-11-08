# 09 - Nanonets invoice workflow

## What is implemented

### Flow

```
PDF -> preprocessing (render, quality flags, denoise, deskew, CLAHE on low-contrast pages)
    -> page OCR (Nanonets OCR2-3B; truncation retries at 2600/2200 px)
    -> reconstruct_structure -> check_completeness -> [targeted_ocr_retry -> check_completeness]
    -> validate_invoice -> judge_discrepancies -> score_confidence -> human review / auto-approval
```

### OCR

- OCR uses `nanonets/Nanonets-OCR2-3B` on a Colab GPU through vLLM, an
  authenticated job gateway, and a temporary Cloudflare Quick Tunnel.
  `NANONETS_API_KEY` is a shared gateway secret, not a Nanonets vendor key.
  Django posts each page and polls the gateway's job endpoint. No OCR model
  weights are required on the local machine.
- The page prompt asks for one `<tr>` per printed row, empty cells kept,
  wrapped descriptions in one cell, and rows without amounts transcribed.
  Prompt wording is an engineering default; its effect is unmeasured.
- OCR stores one response per processed page, including the complete server
  JSON artifact. Coordinates and confidence values must not be assumed.
- **Strip OCR (default, `OCR_PAGE_MODE=strips`).** Dense hospital bill pages
  read in one call exceeded the model's output budget on every page of a
  13-page bill, and the downscaled retries then lost rows (a section printed
  3,760.00 but only 380.00 of rows survived). Each page is now cut into
  full-width strips of at most `OCR_STRIP_HEIGHT` pixels
  (`preprocessing.pipeline.tables.split_at_gaps`). Each cut lies on the
  emptiest pixel row (blank space or a ruling line) in the lower third of the
  strip, so no text row is split and no overlap or de-duplication is needed.
  Strips are read at full resolution with a strip prompt; blank strips are
  skipped. When a strip shows a table header, its column names are passed to
  the next strip and the next page so continuing rows keep the same columns
  and the model is told not to invent headings. A strip that still
  truncates is split again (at most twice, not below 300 px); one still
  truncated after that keeps `finish_reason=length` for the completeness
  check. Every strip request, response, region and column hint is kept under
  `_ocr_strips` in the page artifact. The page text is the strips' text in
  order.
- In legacy `OCR_PAGE_MODE=page`, a `finish_reason=length` response is not
  accepted as complete. The backend
  retries the entire page with a 2600-pixel maximum edge, then 2200 pixels.
  Earlier responses are kept under `_ocr_retry` in the raw artifact. If every
  attempt truncates, the longest output is kept with `finish_reason=length`
  and `_ocr_retry.all_truncated=true`; the job no longer fails. The invoice
  workflow then reports the page as truncated and re-reads its tables.

### Extraction (`invoices/extraction.py`, schema `1.1`)

- HTML tables (colspan/rowspan expanded, `<br>` kept as in-cell line
  breaks, unclosed tables and rows from truncated output kept) and pipe
  tables with or without outer pipes are read row by row.
- Every table row gets a disposition: `header`, `heading` (billing group),
  `continuation` (wrapped description joined to the previous item), or an
  emitted row of kind `item`, `subtotal`, `total`, `tax`, `discount`,
  `payment` or `adjustment` (round-off). Per-table counts are stored in
  `extraction.tables`; nothing is skipped silently. Kinds are decided by the
  row label, never by whether a number is present.
- Rows with blank or unreadable amounts are kept and flagged
  (`missing_amount`). Repeated descriptions are kept as separate rows.
- **Columns are checked against the numbers.** A table without a header
  inherits the previous header when the column count is within two
  (`+inherited_columns`). The quantity, rate and amount columns of a printed,
  inherited or missing header are then tested on rows with quantity ≠ 1
  (where quantity × rate = amount distinguishes columns). If another column
  triple fits the rows better, it is used (`+columns_by_arithmetic`, info
  flag). This catches headings the OCR model invents on continuation pages
  (seen: `AMOUNT | QUANTITY | UNIT PRICE` over rate, qty, gross) and extra
  empty columns. Returns with negative quantity and rate compare magnitudes.
  Values are never changed, only which column they are read from.
- Date cells (`01 Apr 2024`) are never item names or numbers; code/date
  columns from the header are context, not values. A label held in a colspan
  cell that covers the description column (e.g. `<td colspan=4>X Total`) is
  read as the row's label.
- Descriptions split across a page or strip: a first row with only a date and
  text at the top of a continuing table is joined to the previous item; a
  short first line in the first row's description cell is moved back to the
  previous page's item (`fragment_from_next_page`, review).
- **Totals and sections by arithmetic, not wording.** Labels differ between
  hospitals and the model may reword them. After reading, a row with no
  quantity or rate whose amount equals the sum of at least two preceding
  items since the last section boundary becomes a section total; one equal
  to the sum of all items so far becomes the invoice total
  (`total_by_arithmetic`). Label patterns are only a first hint, and every
  total is still checked by validation. Items between two section totals
  belong to the later one: a total closes the open section when its items
  already have one, and names a section (existing or new, from its label)
  only when the heading was missed. No word list decides this.
- **Restated amounts are not charges** (kind `summary`, info flag, row kept):
  a summary table (e.g. a department-wise first page) whose row names match
  sections itemised elsewhere (`summary_of_detail`), and a label/amount
  totals block whose other lines are totals, taxes or payments
  (`footer_line`). Tables headed with receipt/payment-mode/refund columns are
  payments. Payable-by-patient/payor, balances, taxable amount and amount in
  words are payment or tax lines, so the grand total comes from the bill
  total, not the patient's share.
- A row whose cells hold the same number of `<br>` lines in every value
  column is split into one row per line (`split_from_merged_cell`);
  misaligned multi-value cells are not used (`merged_values`).
- Numbers (`invoices/numbers.py`): Indian and international grouping, ₹/Rs/
  INR, `/-`, bracketed negatives. O/0, I/1, S/5, decimal-comma and stray-space
  confusions produce a suggestion in the row's flags; the field stays empty
  until a reviewer confirms it.
- Each row keeps `page_number`, `table_number`, `row_order`, `serial`,
  `source_text`, `name`, numeric fields, `tax_amounts`, `group_id`,
  `relationship`, `extraction_method`, `review_status` and `flags`.
  `extraction.pages` records per-page table counts, unclosed tables and
  amount-bearing lines found outside tables (not extracted as items).

### Completeness check (`invoices/completeness.py`)

Runs before financial validation. It never edits rows. Findings use
category `completeness`:

| Code | Severity | Meaning |
| --- | --- | --- |
| `missing_ocr_page`, `empty_ocr_page` | error | A preprocessed page has no OCR text |
| `ocr_truncated`, `unclosed_table` | error | Output was cut off; rows at the end may be missing |
| `rows_unaccounted` | error | Parser self-check: a table row has no disposition |
| `ocr_downscaled` | review | Page read from a downscaled retry |
| `untabled_amount_lines` | review | Lines with amounts outside any table |
| `no_rows_on_page`, `missing_grand_total` | review | A table produced no rows; no total found |
| `serial_gap`, `duplicate_serial`, `serial_order` | review | S.No sequence jumps, repeats or goes backwards |
| `boundary_duplicate` | review | First row of a page repeats the last row of the previous page |
| row flags (`missing_amount`, `ocr_char_confusion`, `merged_values`, ...) | review | Incomplete or uncertain fields |
| `rows_only_in_targeted_ocr`, `method_value_disagreement`, `not_in_targeted_ocr` | review | Page OCR and table-crop OCR disagree |
| `targeted_ocr_failed`, `targeted_ocr_truncated`, `targeted_ocr_skipped` | review | The cross-check could not run or the retry limit was reached |
| `section_items_short` | review | A section's items add up to less than its printed total or summary line; its pages are re-read |
| `fragment_from_next_page` | review | A description fragment was moved across a page break |
| `table_continues` | info | Context only; not a blocker. Row-level info flags (`wrapped_description`, `columns_by_arithmetic`, `total_by_arithmetic`, `summary_of_detail`, `footer_line`) stay on the row and are not emitted as findings |

Subtotal differences are never used to invent rows or amounts.

### Targeted OCR retry (`invoices/retry.py`)

- Pages with retry-worthy findings (truncation, downscaled read, unclosed
  table, untabled amounts, empty table, serial gaps, merged cells, missing
  total) are re-read once each, up to `INVOICE_RETRY_MAX_PAGES` pages and
  `INVOICE_RETRY_MAX_CALLS` gateway calls per run.
- `preprocessing/pipeline/tables.py` finds ruled tables (adaptive threshold +
  morphological line detection, used only for detection). Without a ruled
  table the whole page width is used. Regions are cut into
  `INVOICE_RETRY_BAND_HEIGHT`-pixel bands overlapping by
  `INVOICE_RETRY_BAND_OVERLAP` pixels and sent at full resolution with a
  table-only prompt. Rows repeated in an overlap are dropped when merging.
- Every call is stored as an `OcrRetryAttempt` (region, coverage, reasons,
  status, text, full raw response, error).
- The completeness check then pairs page rows with crop rows (same page,
  description similarity ≥ 0.85, amount). Crop-only rows become
  `extraction.candidate_rows` with status `pending`; amount conflicts and
  page-only rows are flagged. Page OCR values are never replaced.

### Financial validation (`invoices/validation.py`)

Decimal checks of quantity × rate, gross − discount = net, line tax
(tax-inclusive or added), group subtotals, and grand total (items, plus tax
rows, minus discount rows, plus round-off rows; whole-rupee rounding within
1.00 accepted when no round-off row exists). Payments are ignored. A check
with any missing operand is counted as `unavailable` in
`run.validation_summary`; missing values are never treated as zero.
Findings never alter extracted values.

### Judge, confidence and review

- The judge asks at most three targeted questions about financial findings
  (a second prompt to the same model, not an independent one). Its output
  never overwrites extraction; ambiguous answers stay unresolved.
- `score_confidence` (`invoices/confidence.py`, engineering default,
  uncalibrated) starts from the percentage of item rows with no active
  finding and an amount, subtracts 15 per page/invoice/subtotal-level
  finding and 10 per downscaled page. Info and resolved findings are
  ignored.
- A run is `auto_approved` only when the score is at least
  `INVOICE_CONFIDENCE_THRESHOLD` (default 80) **and** it has item rows, no
  error finding and no unresolved completeness warning. Otherwise it goes to
  `needs_review`.
- Reviewers can correct rows (`review_status=human_corrected`; completeness
  findings on that row are marked resolved with the note), accept or dismiss
  candidate rows (`human_added` rows record `extraction_method`
  `table_crop_ocr`), and mark any finding checked with a required note.
  Approval needs item rows and no unresolved error finding. Every action is a
  `ReviewEvent`; the original extraction is kept in `original_extraction`.
- React shows completeness warnings first, candidate rows, per-row page/
  table/row references, extraction method and review status, row warnings,
  check counts, targeted OCR attempts, and the processed or original page
  image beside the selected row's source text.

### Evaluation

`invoices/evaluation.py` and `manage.py evaluate_extraction <files>` measure
line-item recall, precision, field accuracy (description, quantity, rate,
amount), duplicate rate and missing rate against manually verified JSON
ground truth (stored run or saved OCR text). No verified invoice set exists
yet, so no accuracy figure is claimed.

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

## Observed on two locally stored hospital bills (not in the repository)

Re-running the parser on stored OCR text: a 5-page summary + detail bill
went from 22 page-1 summary/footer lines counted as items, names taken from
the HSN or S.No column, and a grand total of 0.72 (the patient's share), to
58 items whose sum, every section total and the grand total (13,817.72)
reconcile, with 9 summary lines, 6 tax and 11 payment lines kept aside. On
the first 6 pages of a 13-page bill (OCR was still running), arithmetic
column selection overruled invented headers and every section total
reconciled. These are two bills, not a benchmark; strip OCR itself has not
yet been run against the live gateway.

## Known gaps

- There is no manually verified invoice dataset or calibrated accuracy result.
  The confidence weights and the 80 threshold are therefore unvalidated; a
  wrong but internally consistent extraction (for example a misread amount
  repeated in the total) can score 100 and be auto-approved.
- No full invoice was processed end to end after the model switch.
- The Colab runtime and Quick Tunnel URL are temporary. A new session needs a
  new URL in `backend/.env`; Django must restart after configuration changes.
- The gateway key must match in Colab and Django. It protects the public
  tunnel but is not a Nanonets vendor API credential. Never commit or share it.
- The parser can still miss rows in unsupported layouts (borderless free-text
  bills, multi-column item lists). Completeness checks catch many but not all
  such losses: an invoice without serial numbers, totals or a second read
  gives no evidence of a missing row.
- Targeted OCR uses the same model as page OCR, so both can make the same
  mistake. Agreement is not proof. Crops cut at band edges can produce
  partial rows that appear as spurious candidates.
- Description matching between methods is fuzzy (ratio 0.85); OCR variants of
  one row can appear as one candidate plus one `not_in_targeted_ocr` flag.
- Medicine names are preserved exactly; no terminology normalisation or
  look-alike medicine detection is implemented.
- Tax, payment, discount and round-off lines are still recognised from their
  labels (English patterns). A bill that labels them differently gets them
  as items, where arithmetic total checks and reconciliation flag the
  mismatch rather than hide it.
- Strip OCR costs one call per strip (about three per A4 page at 300 dpi)
  and sends no image of the rest of the page; headings far above a strip are
  carried only through the column hint.
- Downscaled OCR retries leave the whole page in view but can lose small text.
  Check amounts against the source image during human review.
- The graph stores stage snapshots but cannot automatically resume a crashed
  worker at its previous node. A fresh run can be created.
- The local React interface relies on Django login. Production auth, object
  storage policy, Redis deployment, and reviewer roles need hardening.
