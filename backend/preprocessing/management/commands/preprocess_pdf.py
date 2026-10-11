"""Run Layer 1 on a PDF from the command line, without the database.

Example:
    python manage.py preprocess_pdf scan.pdf --output-dir out --denoiser drunet
"""

import json
from pathlib import Path

import cv2
from django.core.management.base import BaseCommand, CommandError

from preprocessing.pipeline import ConfigError, PdfRenderError, preprocess_pdf
from preprocessing.pipeline.config import DENOISER_CHOICES, SIGMA_MODE_CHOICES
from preprocessing.services import get_default_config


class Command(BaseCommand):
    help = "Preprocess a PDF (render, quality check, denoise, deskew) into PNGs + a JSON report."

    def add_arguments(self, parser):
        parser.add_argument("pdf", type=Path)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--denoiser", choices=DENOISER_CHOICES)
        parser.add_argument("--sigma-mode", choices=SIGMA_MODE_CHOICES)
        parser.add_argument("--fixed-sigma", type=float)
        parser.add_argument("--dpi", type=int, dest="render_dpi")
        parser.add_argument("--max-pages", type=int)
        parser.add_argument("--no-deskew", action="store_true")
        parser.add_argument(
            "--save-original", action="store_true", help="also write the rendered page"
        )

    def handle(self, *args, **options):
        pdf: Path = options["pdf"]
        if not pdf.is_file():
            raise CommandError(f"file not found: {pdf}")

        overrides = {
            key: options[key]
            for key in ("denoiser", "sigma_mode", "fixed_sigma", "render_dpi", "max_pages")
            if options[key] is not None
        }
        if options["no_deskew"]:
            overrides["deskew_enabled"] = False
        try:
            config = get_default_config().with_overrides(overrides)
        except ConfigError as exc:
            raise CommandError(str(exc))

        output_dir: Path = options["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)

        report = []
        try:
            for result in preprocess_pdf(pdf, config):
                stem = f"page_{result.page_number:04d}"
                self._write(output_dir / f"{stem}_processed.png", result.processed)
                if options["save_original"]:
                    self._write(output_dir / f"{stem}_original.png", result.original)
                report.append(result.metadata())
                self.stdout.write(
                    f"page {result.page_number}: {result.color_mode}, "
                    f"sigma={result.sigma_estimated:.4f}, "
                    f"denoised={result.denoise_applied}"
                    f"{'' if result.denoise_applied else f' ({result.denoise_skip_reason})'}, "
                    f"skew={result.skew_angle:+.2f} deg, "
                    f"flags={result.quality_before.flags or 'none'}"
                )
        except PdfRenderError as exc:
            raise CommandError(str(exc))

        report_path = output_dir / "report.json"
        report_path.write_text(
            json.dumps({"config": config.to_dict(), "pages": report}, indent=2)
        )
        self.stdout.write(self.style.SUCCESS(f"{len(report)} page(s) -> {output_dir}"))

    @staticmethod
    def _write(path: Path, image) -> None:
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, buffer = cv2.imencode(".png", image)
        if not ok:
            raise CommandError(f"could not encode {path.name}")
        path.write_bytes(buffer.tobytes())  # imwrite cannot handle non-ASCII paths
