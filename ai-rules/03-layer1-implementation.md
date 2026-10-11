# 03 - Layer 1 implementation

What Layer 1 does and why. For where the code lives see
`05-project-structure.md`; for settings see `07-configuration-and-testing.md`.

## Pipeline at a glance

```
PDF --render--> page image --quality check + noise estimate--> denoise --> deskew --> quality check --> output
     (300 dpi)   (RGB)        (flags, sigma)                  (DRUNet)    (Otsu +     (after)          PNG + metadata
                                                                           min-area rect)
```

Order of the two paper stages: **denoise, then deskew** (paper Section 5).

Each decision below is tagged:

- **[paper]** taken directly from the paper;
- **[arch]** required by the architecture diagram;
- **[eng]** an engineering decision the paper does not cover.

## Features

### 1. PDF rendering [arch]

- Renders each page with pypdfium2 on a white background, RGB, at
  `render_dpi` (default 300, the paper's scan resolution **[paper]**).
- Pages are produced one at a time by a generator, so memory is bounded by a
  single page whatever the document length **[eng]**.
- A page whose pixel count would exceed `max_page_pixels` (default 40 MP) is
  rendered at a proportionally lower DPI; the effective DPI is recorded
  **[eng]**.
- Accepts a path or raw bytes, so it works with object storage.
- Corrupt, empty or password-protected PDFs raise `PdfRenderError`.

### 2. Grayscale vs colour detection [paper, eng]

The paper treats the two kinds separately, and DeepInv ships different
pretrained weights for 1-channel and 3-channel images. A page is treated as
grayscale when its mean colour saturation is below
`grayscale_chroma_threshold` (0.02); it is then collapsed to one channel and
handled by the grayscale model. The threshold itself is **[eng]**.

### 3. Image quality checks [arch]

Reference-free measurements, taken before and after processing:

| Measurement | How | Flag raised when |
| --- | --- | --- |
| Effective DPI | From the renderer | `low_resolution`: below `min_effective_dpi` (150) |
| Brightness | Mean intensity | `too_dark` below 0.25, `overexposed` above 0.98 |
| Contrast | 99th minus 1st intensity percentile | `low_contrast` below 0.20 |
| Sharpness | Variance of the Laplacian | `blurry` below 50 |
| Ink ratio | Share of pixels clearly darker than the background | `blank_page` below 0.0005 |
| Chroma | Mean spread between RGB channels | (used for grayscale detection) |
| Noise sigma | Blind estimator, see below | `high_noise` above `max_sigma` (0.20) |

All thresholds are **[eng]** defaults and have not been calibrated on a
labelled set. Flags never reject a page; they are stored for Layer 3.

Blank pages skip denoising and deskewing: there is nothing to restore, and
the skew of an empty page is undefined.

### 4. Blind noise estimation [eng]

The single largest addition to the paper. DRUNet needs the noise level; the
paper supplies it because the noise is synthetic. For real uploads it is
estimated per page:

1. Split the page into 32x32 blocks.
2. In each block take the pixels darker than the block median and compute
   their spread around the median (a half-normal, whose scale equals sigma for
   Gaussian noise).
3. Take the 10th percentile across blocks.

Why this design: text makes most blocks look noisy, so the estimate is read
from the quietest blocks (plain background). Paper backgrounds are near
white, where noise is clipped at 1.0 and an ordinary standard deviation reads
low; pixels below the median are unaffected by that clipping. Colour pages
are measured per channel and averaged, because converting to gray first
averages away independent channel noise.

Measured accuracy on synthetic pages (injected 0.01 / 0.02 / 0.05 / 0.10 /
0.20): estimates were within 3% in every case, on gray and colour pages, with
a paper-tone and with a pure-white background. A clean page estimates 0.

### 5. Denoising [paper]

- Algorithms: `drunet` (default, the paper's winner), `dncnn`, `tv`, or
  `none`. All three come from DeepInv, as in the paper. Deep models use
  DeepInv's pretrained weights with no retraining.
- **Sigma selection [eng]**, set by `sigma_mode`:
  - `estimate` (default): use the blind estimate, capped at `max_sigma`
    (0.20, the highest level the paper validated). If the estimate is below
    `min_sigma` (0.01) the page is considered clean and **denoising is
    skipped**.
  - `fixed`: always denoise with `fixed_sigma`. This is the paper's
    known-sigma setting.
- **Tiled inference [eng]**: the page is processed in `tile_size` (512)
  tiles overlapping by `tile_overlap` (32) pixels, blended with a linear
  ramp. This replaces the paper's resize to 512x512, which would make text
  unreadable for OCR, and keeps memory bounded. Tile sides are multiples of 8
  because DRUNet downsamples three times.
- **TV weight [eng]**: `lambda = tv_lambda_per_sigma * sigma` with a default
  factor of 0.5, or a fixed `tv_lambda`. The factor was calibrated on a
  synthetic text page, where 0.5 gave the best PSNR at all three paper noise
  levels among 0.15-2.0. 300 iterations gave the same PSNR as 1000.
- Models are loaded once per process and cached. Device is CUDA when
  available, otherwise CPU.

### 6. Deskewing [paper]

The four steps of Section 4.3.1 are implemented as written:

1. Otsu binarisation, text as foreground.
2. `cv2.minAreaRect` on all text-pixel coordinates gives the angle.
3. Affine rotation by the inverse angle.
4. Crop to the bounding box of the content.

Details the paper leaves open **[eng]**:

- The rectangle angle is folded modulo 90 degrees into [-45, 45). OpenCV has
  changed the range it reports between releases; folding makes the result
  independent of the installed version. (A first version assumed one
  convention and rotated real pages by 90 degrees; a test now pins this.)
- The canvas grows to fit the rotated page so nothing is cut off; corners are
  filled with white ("white margins", as the paper describes).
- Cubic interpolation.
- The crop box is computed from the rotated text mask, not by re-thresholding,
  so the white corners cannot confuse it. `crop_padding` (16 px) is kept
  around the content so glyphs do not touch the border.
- Rotations below `min_skew_angle` (0.1 degree) are skipped to avoid
  resampling blur for no gain.
- A 2x3 affine matrix mapping original pixels to output pixels is stored with
  every page, so OCR boxes can be projected back onto the original scan.

Angle convention everywhere in the code and database: **degrees,
counter-clockwise positive, as the page is displayed**.

### 7. Evaluation harness [paper]

`pipeline/evaluation.py` and the `evaluate_denoisers` command reproduce the
protocol of Section 7.2: resize to 512x512, add Gaussian noise at 0.05 / 0.10
/ 0.20, restore with each algorithm using the known sigma, compute the six
metrics, average per algorithm and noise level. Metrics come from DeepInv
(`MSE`, `PSNR`, `SSIM`, `SSIM(multiscale=True)`, `LPIPS`, `GMSD`).

### 8. Persistence, jobs and API [arch]

- `PreprocessingJob`: one run over a document, with status
  (`pending`, `running`, `completed`, `failed`), progress counters, error
  message and a JSON snapshot of the configuration used.
- `PreprocessedPage`: one row per page with the original and processed PNGs
  and all metadata.
- `services.run_job` claims a job atomically (only a `pending` job can
  start), stores pages as they finish so progress is visible, and records any
  failure on the job instead of raising.
- Celery task `preprocessing.preprocess_document` wraps `run_job`.
- REST endpoints for upload, job status and re-running with options (see
  `06-data-flow-and-integration.md`).
- Command-line entry points: `preprocess_pdf` and `evaluate_denoisers`.

## Findings from building it

These were measured, not assumed, and they affect how Layer 1 should be used.

1. **DRUNet behaves as the paper says.** On a synthetic gray page at 512x512:
   36.4 / 32.5 / 24.5 dB at sigma 0.05 / 0.10 / 0.20 (paper: 37.8 / 32.3 /
   26.9 on its own images).
2. **DeepInv's pretrained DnCNN does almost nothing at the paper's noise
   levels.** Measured: 26.4 dB against a noisy input of 26.2 dB at sigma
   0.05, and the same pattern at 0.10 and 0.20. DeepInv's DnCNN weights are
   trained for a single low noise level (sigma = 2/255, about 0.008) and the
   model ignores the sigma argument. The paper reports much larger DnCNN
   gains and does not say which weights it used, so its DnCNN numbers could
   not be reproduced. DnCNN is kept as an option because the paper lists it,
   but it is not recommended.
3. **Real scans may contain no Gaussian noise at all.** The owner's sample
   bill (5 pages, 300 dpi, grayscale) has a saturated white background and an
   estimated sigma of 0.0000 on every page, so denoising is skipped. Forcing
   DRUNet at sigma 0.05 on it left the text intact but lowered sharpness by
   about 13% and did not remove the isolated dust specks, which are not
   Gaussian noise.
4. **When Gaussian noise is present, it is removed.** Injecting noise into a
   full-resolution crop of the same sample and running the default pipeline:

   | Injected sigma | Estimated | PSNR noisy -> restored | SSIM noisy -> restored |
   | --- | --- | --- | --- |
   | 0.05 | 0.049 | 28.6 -> 32.8 dB | 0.656 -> 0.995 |
   | 0.10 | 0.097 | 22.7 -> 27.0 dB | 0.410 -> 0.987 |
   | 0.20 | 0.194 | 16.8 -> 21.0 dB | 0.242 -> 0.960 |

5. **The paper's skew estimator is accurate on clean synthetic pages and
   only roughly accurate on real ones.** Synthetic: within 0.3 degree for
   skews from -10 to +10. Real sample, checked against an independent
   projection-profile measurement: errors up to about 0.5 degree; mean
   absolute skew went from 0.44 to 0.23 degree, but two of five pages were
   over-corrected. See `08-limitations-and-pending.md`.
6. **CPU inference is slow.** About 11 s per 512x512 tile on a 4-core laptop
   CPU, which is roughly 9 minutes for an A4 page at 300 dpi (48 tiles). A
   GPU is needed for production throughput whenever pages are actually noisy.
