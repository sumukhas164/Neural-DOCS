# 04 - Technology stack

## Fixed by the project owner

| Area | Technology |
| --- | --- |
| Backend | Django |
| Frontend | React and Vite (local review interface implemented) |
| Database | MySQL |

## Runtime

| Item | Version | Notes |
| --- | --- | --- |
| Python | 3.11 | The paper used 3.10; 3.11 was chosen for wheel availability. 3.10 and 3.13 are also installed on the dev machine but untested. |
| MySQL | 8.0 | Local Windows service `MySQL80` |
| Redis | not installed | Only needed once Celery runs with a real broker |
| GPU | none on the dev machine | Everything was developed and tested on CPU |

## Python dependencies (`backend/requirements.txt`)

Versions are the ones installed and tested.

| Package | Version | Used for | Licence |
| --- | --- | --- | --- |
| Django | 5.2.18 (LTS) | Web framework, ORM, admin | BSD-3 |
| djangorestframework | 3.18.3 | REST API | BSD-3 |
| python-dotenv | 1.2.4 | Loading `backend/.env` | BSD-3 |
| mysqlclient | 2.3.0 | MySQL driver for Django | GPL-2 |
| celery | 5.6.3 | Task integration point for the orchestration layer | BSD-3 |
| redis | 6.4.0 | Celery broker client | MIT |
| numpy | 2.4.6 | Array maths | BSD-3 |
| opencv-python-headless | 4.14.0 | Otsu, min-area rectangle, affine rotation, PNG I/O | Apache-2 |
| pypdfium2 | 5.14.0 | PDF rendering | Apache-2 / BSD-3 |
| torch | 2.14.1 (CPU) | Runs the deep denoisers | BSD-3 |
| deepinv | 0.4.2 | DRUNet, DnCNN, TV denoiser, and the six metrics | BSD-3 |
| pytest, pytest-django | 9.1.1, 4.14.0 | Tests | MIT / BSD-3 |
| Pillow | 12.3.0 | Building synthetic PDFs in tests | HPND |

Installed indirectly by `deepinv` and relied on: `torchvision` 0.29.1 and
`torchmetrics` 1.9.0 (SSIM, MS-SSIM, LPIPS).

## Why these choices

- **DeepInv** is the library the paper used for all three denoisers, and it
  also provides all six evaluation metrics, so no second metrics library is
  needed.
- **pypdfium2** renders PDFs with no system dependency (no Poppler or
  Ghostscript install) and has a permissive licence. PyMuPDF was avoided
  because of its AGPL licence.
- **opencv-python-headless** rather than `opencv-python`: no GUI libraries
  are needed on a server.
- **Celery + Redis** are named in the architecture diagram.
- **pytest** with **pytest-django**: the pipeline tests need no Django at
  all, and one runner covers both kinds.

## Pretrained model weights

Downloaded by DeepInv on first use and cached in the torch hub cache
(`%USERPROFILE%\.cache\torch\hub\checkpoints`). The first run of a deep
denoiser therefore needs internet access; later runs do not.

| Model | Variant | Notes |
| --- | --- | --- |
| DRUNet | grayscale (1 channel) and colour (3 channels) | Accepts any sigma |
| DnCNN | grayscale and colour | Trained for sigma = 2/255 only |
| AlexNet (for LPIPS) | - | Used by the evaluation harness only |

No weights are stored in the repository (`*.pth` is git-ignored).

## Development environment notes

- Windows 11, PowerShell. Paths with spaces need quoting.
- The virtual environment is `backend/.venv`. Point the IDE's Python
  interpreter at it, otherwise the editor reports the packages as missing.
- OpenCV's `imread`/`imwrite` fail on non-ASCII Windows paths; the code uses
  `imdecode`/`imencode` with explicit file I/O where a user path is involved.
