"""Reproduce the paper's denoising benchmark on a folder of clean scans.

Example:
    python manage.py evaluate_denoisers clean_scans/ --output eval_output/results.csv
"""

import csv
from pathlib import Path

import numpy as np
from django.core.management.base import BaseCommand, CommandError

from preprocessing.pipeline.config import PAPER_NOISE_LEVELS
from preprocessing.pipeline.evaluation import (
    PAPER_ALGORITHMS,
    PAPER_EVAL_SIZE,
    evaluate_image,
    list_images,
    load_image,
    summarize,
)
from preprocessing.pipeline.metrics import METRIC_NAMES
from preprocessing.services import get_default_config


class Command(BaseCommand):
    help = "Add Gaussian noise to clean images, denoise with each algorithm, report the paper's six metrics."

    def add_arguments(self, parser):
        parser.add_argument("image_dir", type=Path, help="folder of clean reference images")
        parser.add_argument("--output", type=Path, help="CSV file for per-image rows")
        parser.add_argument("--algorithms", nargs="+", default=list(PAPER_ALGORITHMS), choices=PAPER_ALGORITHMS)
        parser.add_argument("--sigmas", nargs="+", type=float, default=list(PAPER_NOISE_LEVELS))
        parser.add_argument(
            "--size", type=int, default=PAPER_EVAL_SIZE,
            help="resize images to SIZE x SIZE as the paper does; 0 keeps the original size",
        )
        parser.add_argument("--no-lpips", action="store_true", help="skip LPIPS (needs AlexNet weights)")
        parser.add_argument("--seed", type=int, default=0)

    def handle(self, *args, **options):
        image_dir: Path = options["image_dir"]
        if not image_dir.is_dir():
            raise CommandError(f"not a directory: {image_dir}")
        paths = list_images(image_dir)
        if not paths:
            raise CommandError(f"no images found in {image_dir}")

        config = get_default_config()
        rng = np.random.default_rng(options["seed"])
        rows = []
        for path in paths:
            self.stdout.write(f"evaluating {path.name} ...")
            rows += evaluate_image(
                load_image(path),
                name=path.name,
                algorithms=options["algorithms"],
                noise_levels=options["sigmas"],
                resize=options["size"] or None,
                config=config,
                include_lpips=not options["no_lpips"],
                rng=rng,
            )

        if options["output"]:
            output: Path = options["output"]
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["image", "color_mode", "algorithm", "sigma", *METRIC_NAMES])
                for row in rows:
                    writer.writerow(
                        [row.image, row.color_mode, row.algorithm, row.sigma]
                        + [row.metrics[m] for m in METRIC_NAMES]
                    )
            self.stdout.write(f"per-image rows -> {output}")

        header = f"{'mode':<10}{'algorithm':<10}{'sigma':>6}{'n':>4}" + "".join(
            f"{m:>10}" for m in METRIC_NAMES
        )
        self.stdout.write("\n" + header)
        for entry in summarize(rows):
            values = "".join(
                f"{entry[m]:>10.4f}" if entry[m] is not None else f"{'-':>10}"
                for m in METRIC_NAMES
            )
            self.stdout.write(
                f"{entry['color_mode']:<10}{entry['algorithm']:<10}"
                f"{entry['sigma']:>6.2f}{entry['images']:>4}{values}"
            )
