# 02 - Preprocessing methodology (from the research paper)

Source: Nather and Ahmed (2026), full citation in `README.md`. Section and
equation numbers below refer to that paper.

## What the paper proposes

A modular pipeline that improves scanned grayscale and colour documents so
that OCR and archiving work better. It has two processing stages:

1. **Denoising** with one of three algorithms: Total Variation (TV), DnCNN,
   DRUNet.
2. **Deskewing** with Otsu thresholding and a minimum-area bounding rectangle.

The paper's conclusion: **DRUNet is the best denoiser at every noise level
tested**, for both grayscale and colour images.

## Stage order

**Denoise first, then deskew.** Section 5 states it explicitly: the cascade
"allows us to perform the denoising first and then deskewing, which makes the
skew angle computation more precise and prevents noise influence when angle
is computed." Figure 1 shows the same order.

> Inconsistency in the paper: Section 3.4 says deskewing prepares images "to
> undergo other denoising algorithms", which reads as deskew-first. Section 5
> and Figure 1 are explicit and agree with each other, so they are followed.

## Step-by-step requirements derived from the paper

### Step 0 - Input preparation (Sections 3, 5)

| Paper | Detail |
| --- | --- |
| Scans at 300 dpi | Dataset of 40 images: 20 colour, 20 grayscale |
| Grayscale and colour handled separately | "converted to gray-scale if necessary" |
| Resize to 512x512 | Done for the experiments only |
| Add Gaussian noise, sigma = 0.05, 0.10, 0.20 | Experimental degradation, eq. 6: `y = x + n, n ~ N(0, sigma^2)` |

### Step 1 - Denoising (Section 4.2)

All three run on both grayscale and colour images. Deep models are
**pretrained, taken from the DeepInv library, and not retrained**.

**Total Variation (4.2.1, eqs. 1-5).** Solve
`min_x 0.5 * ||x - y||^2 + lambda * TV(x)`, iteratively. `lambda` controls
smoothing: large removes detail, small leaves noise. The paper does not
report the `lambda` it used.

**DRUNet (4.2.2, eqs. 6-9).** A U-Net with residual blocks and skip
connections. It takes the noise level as an additional input map (Section
6.1), which is why one set of weights handles every sigma. Used with "the
standard configuration proposed by the original authors".

**DnCNN (4.2.3, eqs. 10-16).** A plain deep CNN: convolution + ReLU, then
convolution + batch-norm + ReLU layers, then a final convolution. Predicts
the noise residual, which is subtracted from the input (eq. 14).

### Step 2 - Deskewing (Section 4.3.1, eqs. 17-20)

1. **Binarise with Otsu's method** (eq. 17): text pixels are 1 where the gray
   value is below the automatic threshold, 0 elsewhere.
2. **Estimate the angle** (eq. 18): fit a minimum-area bounding rectangle to
   the coordinates of all text pixels; its angle is the skew.
3. **Rotate** (eq. 19): apply an affine rotation by the inverse angle.
4. **Crop** (eq. 20): rotation leaves white margins at the corners; crop to
   the bounding box of the actual content.

### Step 3 - Evaluation (Section 7)

Six full-reference metrics, each comparing the restored image with the clean
original:

| Metric | Eq. | Better when | Measures |
| --- | --- | --- | --- |
| MSE | 21 | lower | Pixel error |
| PSNR (dB) | 22 | higher | Pixel error on a log scale |
| SSIM | 23-24 | higher | Luminance, contrast, structure |
| MS-SSIM | 25 | higher | SSIM across several scales |
| LPIPS | 26 | lower | Distance between deep-network features |
| GMSD | 27-28 | lower | Deviation of gradient-magnitude similarity (edges) |

Protocol (Section 7.2): for each image, each sigma and each algorithm,
compute the six metrics, then average over the 20 images of each set.

## The paper's reported results (for comparison)

Mean PSNR in dB, from Tables 1 and 2:

| Set | Algorithm | sigma 0.05 | sigma 0.10 | sigma 0.20 |
| --- | --- | --- | --- | --- |
| Grayscale | DRUNet | 37.81 | 32.30 | 26.91 |
| Grayscale | DnCNN | 34.05 | 28.45 | 23.17 |
| Grayscale | TV | 27.39 | 22.66 | 19.17 |
| Colour | DRUNet | 39.99 | 36.06 | 32.08 |
| Colour | DnCNN | 35.74 | 30.67 | 25.80 |
| Colour | TV | 32.17 | 27.47 | 23.40 |

## Implementation environment in the paper (Section 5)

Python 3.10, Google Colab with GPU, OpenCV, NumPy, PIL, DeepInv, and a Gradio
interface. This project replaces Gradio with the Django/React stack; the
image-processing libraries are the same.

## What the paper does not specify

These gaps had to be filled to build a working system. How each was filled is
in `03-layer1-implementation.md`; the risks are in
`08-limitations-and-pending.md`.

| Gap | Why it matters |
| --- | --- |
| **Noise level of a real scan** | The paper always knows sigma because it adds the noise itself. DRUNet needs sigma as input; a real upload has none. |
| **When not to denoise** | Every image in the paper is noisy. A clean scan gains nothing from denoising and can lose sharpness. |
| **TV `lambda` and iteration count** | Not reported. |
| **Which DnCNN weights** | Only "pre-trained through DeepInv". |
| **Full-resolution processing** | 512x512 is too small for OCR; nothing is said about memory or tiling on real pages. |
| **Rotation details** | Canvas size, interpolation and the fill colour are not given; the minimum-area-rectangle angle range is library-dependent. |
| **Image quality checks** | Named in the architecture, absent from the paper. All paper metrics need a clean reference that production does not have. |
| **PDF input** | The paper starts from images; the system starts from PDFs. |

## Things in the paper that were not adopted

- **Resizing to 512x512** in production (kept only in the evaluation harness).
- **BM3D**: appears in Table 3 as a comparison baseline but is not part of the
  described methodology, so it is not implemented.
- **Gradio interface**: superseded by the project's stack.
- **Table 3 numbers**: they do not match Tables 1-2 and the paper does not say
  how they were produced, so they are not used as a reference.
- **The claim that OCR accuracy improved** (Section 9): the paper reports no
  OCR measurement, so this is treated as a hypothesis to test once Layer 2
  exists, not as an established result.
