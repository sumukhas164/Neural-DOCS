"""Score line-item extraction against manually verified ground-truth files.

Each JSON file describes one invoice and its verified rows:

    {"name": "bill-07",
     "run_id": 12,                                   # score a stored invoice run, or
     "pages": [{"page_number": 1, "text": "..."}],   # re-extract saved OCR text
     "items": [{"page_number": 1, "description": "CBC", "quantity": "1",
                "rate": "350.00", "amount": "350.00"}]}

Keep these files out of the repository: they hold real patient data.
"""

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from invoices.evaluation import aggregate, evaluate
from invoices.extraction import extract
from invoices.models import InvoiceRun


def _percent(value):
    return "n/a" if value is None else f"{value * 100:.1f}%"


class Command(BaseCommand):
    help = "Measure line-item recall, precision and field accuracy against verified invoices."

    def add_arguments(self, parser):
        parser.add_argument("paths", nargs="+", help="Ground-truth JSON files or folders of them")
        parser.add_argument("--output", help="Write the full report as JSON to this path")

    def handle(self, *args, paths, output=None, **options):
        files = []
        for raw in paths:
            path = Path(raw)
            files.extend(sorted(path.glob("*.json")) if path.is_dir() else [path])
        if not files:
            raise CommandError("No ground-truth JSON files found")
        results = []
        for path in files:
            try:
                truth = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise CommandError(f"{path}: {exc}") from exc
            if "run_id" in truth:
                run = InvoiceRun.objects.filter(pk=truth["run_id"]).first()
                if run is None:
                    raise CommandError(f"{path}: invoice run {truth['run_id']} does not exist")
                items = run.extraction.get("items", [])
            elif "pages" in truth:
                items = extract((page["page_number"], page["text"]) for page in truth["pages"])["items"]
            else:
                raise CommandError(f"{path}: give either run_id or pages")
            result = {"name": truth.get("name", path.stem), **evaluate(items, truth.get("items", []))}
            results.append(result)
            self.stdout.write(f"{result['name']}: recall {_percent(result['recall'])}, precision "
                              f"{_percent(result['precision'])}, missing {len(result['missing'])}, "
                              f"duplicates {result['duplicate_rows']}")
        summary = aggregate(results)
        self.stdout.write(self.style.SUCCESS(
            f"{summary['invoices']} invoice(s), {summary['actual_items']} verified rows: "
            f"recall {_percent(summary['recall'])}, precision {_percent(summary['precision'])}, "
            f"missing rate {_percent(summary['missing_rate'])}, duplicate rate {_percent(summary['duplicate_rate'])}"))
        self.stdout.write("Field accuracy: " + ", ".join(
            f"{name} {_percent(value)}" for name, value in summary["field_accuracy"].items()))
        if output:
            Path(output).write_text(json.dumps({"summary": summary, "invoices": results}, indent=2), encoding="utf-8")
