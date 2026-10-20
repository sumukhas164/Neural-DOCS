# 08 - Limitations, assumptions and pending work

## Limitations of the method (inherited from the paper)

1. **The denoisers target additive Gaussian noise only.** The paper evaluates
   nothing else. Real scan defects such as dust specks, salt-and-pepper
   dots, JPEG blocking, bleed-through, stains, shadows and uneven lighting are
   not Gaussian. Measured on the owner's sample: the isolated specks survive
   DRUNet. The paper itself lists blur and low contrast as future work.
2. **The paper validated sigma up to 0.20.** Higher estimates are capped at
   0.20 and the page is flagged `high_noise`.
3. **Skew estimation by minimum-area rectangle is fragile on real pages.**
   The rectangle is fitted to every dark pixel, so a logo, QR code, stamp,
   speck or scanner edge shadow pulls the angle. Measured on the sample bill
   against an independent projection-profile measurement:

   | Page | True skew | Detected | Residual after deskew |
   | --- | --- | --- | --- |
   | 1 | -0.35 | -0.41 | +0.05 |
   | 2 | -0.35 | -0.37 | +0.05 |
   | 3 | -0.40 | -0.87 | +0.45 |
   | 4 | -0.45 | -0.30 | -0.15 |
   | 5 | -0.65 | -1.07 | +0.45 |

   Mean absolute skew improved from 0.44 to 0.23 degree, but pages 3 and 5
   were over-corrected. On clean synthetic pages the same code is accurate to
   0.3 degree across -10 to +10.
4. **Skew is only resolvable within +/-45 degrees.** A page scanned sideways
   or upside down is not detected or fixed. Orientation detection is not in
   the paper.
5. **Otsu assumes dark text on a light background.** Inverted pages, or
   pages that are mostly dark, produce a meaningless text mask; the angle
   then comes out near zero, so the failure is harmless but silent.
6. **No evidence yet that this improves OCR.** The paper asserts it without
   measuring it. On an already-clean scan, denoising can only lose detail,
   which is why the pipeline skips it.

## Limitations of this implementation

7. **CPU inference is slow.** About 11 s per 512x512 DRUNet tile on the dev
   laptop: roughly 9 minutes for a noisy A4 page at 300 dpi. Clean pages are
   unaffected because denoising is skipped. A GPU is required for production
   throughput on noisy documents. With no broker configured, this work runs
   inside the upload request.
8. **DnCNN is ineffective at the paper's noise levels** with DeepInv's
   pretrained weights (trained for sigma = 2/255). The paper's DnCNN results
   could not be reproduced. Kept as an option; not recommended.
9. **The noise estimator assumes Gaussian, spatially uniform, white noise**
   and needs some flat background. It reads low on pages with almost no plain
   background (full-page photos, dense halftones) and does not model JPEG
   artefacts. It also cannot see noise that exists only on the ink when the
   background is saturated to pure white.
10. **Quality thresholds are uncalibrated.** Sharpness in particular depends
    on resolution and content, and is inflated by noise, so a noisy page is
    rarely flagged `blurry` before denoising.
11. **"Effective DPI" is the render DPI,** not the resolution of the scan
    embedded in the PDF. A 100 dpi scan rendered at 300 dpi is reported as
    300 and is not flagged `low_resolution`.
12. **Every page is rasterised,** including born-digital PDFs that already
    contain a text layer. That text layer is ignored.
13. **Cropping changes page geometry.** Processed pages differ in size from
    the originals and from each other; use the stored `transform` to relate
    coordinates.
14. **Originals are stored as full-resolution PNGs** alongside processed
    pages: several megabytes per page. There is no retention or clean-up
    policy, and deleting a job row does not delete its files.
15. **A failed job cannot be restarted;** a new job must be created. There
    are no automatic retries, and a worker killed mid-run leaves the job in
    `running` forever.

## Assumptions made

| Assumption | Basis |
| --- | --- |
| Denoise before deskew | Paper Section 5 and Figure 1 (Section 3.4 is ambiguous) |
| Skip denoising on clean pages | Engineering judgement; the paper has no clean inputs |
| Estimate sigma instead of fixing it | Needed because real uploads have no known sigma |
| Process at full resolution in tiles | 512x512 is unusable for OCR |
| White fill and an enlarged canvas on rotation | Paper says "white margins"; canvas size unspecified |
| PNG as the hand-off format to Layer 2 | Lossless; Layer 2's needs are not yet known |
| Uploads are PDFs only | Architecture diagram says "Upload PDF" |
| One user owns a document | No sharing or roles are specified |
| Layer 1 flags, Layer 3 decides | Architecture places routing in Layer 3 |

## Not verified

- **MySQL**: migrations exist but have not been applied to MySQL; tests ran
  on SQLite. Needs credentials in `backend/.env`.
- **Celery with a real Redis broker.**
- **GPU execution.**
- **The paper's Tables 1-2**: reproducing them needs its 40-image dataset,
  which is not available. The `evaluate_denoisers` command is ready for any
  folder of clean scans.

## Pending work

### To finish hardening Layer 1

- [x] Put MySQL credentials in `backend/.env`, create the schema, and apply
      migrations. Running the full suite against MySQL remains pending.
- [ ] Decide whether to replace or supplement the minimum-area-rectangle skew
      estimator with a more robust one (projection profile or Hough lines).
      This departs from the paper, so it needs the owner's decision.
- [ ] Collect a labelled set of real bills and invoices to calibrate the
      quality thresholds and to run `evaluate_denoisers`.
- [ ] Decide whether non-Gaussian defects (specks, shadows) need handling;
      the paper does not cover them.
- [ ] Provision a GPU, or decide on a lower DPI or a lighter default, for
      noisy documents.
- [ ] Storage policy: whether to keep originals, retention, file clean-up on
      delete.
- [ ] Once Layer 2 exists, measure OCR accuracy with and without Layer 1 to
      test the paper's central claim.

### Remaining work

- [ ] Orchestration: Redis broker, layer chaining, retries, stuck-job
      recovery, cross-layer status. Move the upload-to-Layer-1 hand-off out
      of `documents/views.py`.
- [x] Layer 2: OCR integration with Nanonets OCR2-3B via a Colab gateway.
      Model and gateway startup were observed; a tunnel-connected 3B invoice
      run and full invoice verification remain pending. Earlier synthetic and
      billing-crop observations came from the local 1.5B model.
- [x] Baseline invoice validation and human-review routing.
- [ ] Benchmark Nanonets OCR2-3B and the downstream parser on real medical invoices.
- [ ] Evaluate the legibility and amount accuracy of 2600/2200-pixel retries
      on dense pages that exceed the Colab model's 12,288-token context window.
- [ ] Calibrate final validation and reconciliation on labelled medical invoices.
- [ ] REST API: real authentication (tokens or sessions for the SPA), CORS,
      pagination, rate limits.
- [x] Local React review interface; browser QA and production auth remain.
- [ ] Production concerns: object storage backend, serving media, HTTPS
      settings, logging and monitoring.
