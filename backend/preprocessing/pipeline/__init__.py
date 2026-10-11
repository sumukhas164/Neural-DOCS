"""Layer 1 preprocessing pipeline (framework-independent).

Public entry points:

* :func:`preprocess_pdf` / :func:`preprocess_image` run the pipeline.
* :class:`PreprocessingConfig` holds every tunable.
"""

from .config import ConfigError, PreprocessingConfig
from .pdf_renderer import PdfRenderError
from .pipeline import PageResult, preprocess_image, preprocess_pdf

__all__ = [
    "ConfigError",
    "PageResult",
    "PdfRenderError",
    "PreprocessingConfig",
    "preprocess_image",
    "preprocess_pdf",
]
