"""Conservative structure reconstruction from Nanonets page text/HTML tables.

Only values present in an OCR row are emitted; nothing is inferred from totals.
Every table row is accounted for (item, heading, header, continuation, total,
...), so a row that cannot be classified is kept and flagged instead of being
dropped. Source text, page, table and row order accompany every row, and the
per-table row accounting feeds the completeness check.
"""

import re
from collections import Counter, defaultdict
from decimal import Decimal
from html import escape
from html.parser import HTMLParser

from .numbers import looks_numeric, parse_number, parse_quantity

SCHEMA_VERSION = "1.1"
NUMERIC_FIELDS = ("quantity", "unit_rate", "gross_amount", "discount", "net_amount")
ROW_KINDS = ("item", "subtotal", "total", "tax", "discount", "payment", "adjustment", "summary")

TOTAL = re.compile(r"\b(grand\s+total|net\s+(?:amount\s+)?payable|amount\s+payable|total\s+amount|bill\s+total|"
                   r"total\s+bill(?:\s+amount)?|invoice\s+total|total\s+payable|net\s+total)\b", re.I)
SUBTOTAL = re.compile(r"\b(sub\s*-?\s*total|section\s+total|group\s+total)\b|\S\s+total\s*[:.]?\s*$", re.I)
BARE_TOTAL = re.compile(r"^\s*total(?:\s+charges)?\s*(?:\(?\s*(?:rs\.?|inr|₹)\s*\)?)?\s*[:.]?\s*$", re.I)
TAX = re.compile(r"^\s*(?:add\s*[:\-]?\s*)?(?:total\s+)?(?:[csiu]\s*gst|gst|vat|cess|service\s+tax|tax)\b|"
                 r"^\s*taxable\s+(?:amount|value)\b", re.I)
NET_AMOUNT_LINE = re.compile(r"^\s*net\s+amount\s*(?:\(.*\))?\s*[:.]?\s*$", re.I)
DISCOUNT_ROW = re.compile(r"^\s*(?:less\s*[:\-]?\s*)?(?:total\s+)?(?:discount|concession|rebate|waiver|deduction)\b", re.I)
PAYMENT = re.compile(r"^\s*(?:less\s*[:\-]?\s*)?(?:advance|deposit|amount\s+(?:paid|received)|paid|received|"
                     r"refund|balance|due\s+amount|amount\s+due|(?:net\s+)?payable\s+by|bill\s+status|"
                     r"amount\s+in\s+words|(?:net\s+)?(?:patient|payor|payer|sponsor|insurance|insurer|tpa|"
                     r"company|corporate)\s+(?:payable|balance|share|amount|paid))\b", re.I)
PAYMENT_HEADER = re.compile(r"\b(?:receipt|payment\s+mode|refund|cheque|transaction|utr)\b", re.I)
DATE = re.compile(r"^\s*\d{1,2}[\s\-/.]+(?:[A-Za-z]{3,9}|\d{1,2})[\s\-/.,]+\d{2,4}\b")
ADJUSTMENT = re.compile(r"\bround(?:ed|ing)?\s*-?\s*off\b|^\s*rounding\b", re.I)
GROUP_HINT = re.compile(r"(?i)charges|pharmacy|laboratory|radiology|room|procedure|consultation|others|nursing|"
                        r"consumables|medicine|surgery|doctor|investigation|services|implants|ward|icu|ot\b")
HEADER_FIELD_LINE = re.compile(r"(?i)^\s*(bill|invoice|receipt|ip|op|uhid|mrn|reg(?:istration)?|gstin|pan|phone|"
                               r"mobile|tel|date|age|admission|discharge|policy|claim|patient|doctor|consultant)\b")
UNTABLED_AMOUNT = re.compile(r"^(?P<label>.*?[A-Za-z]{3,}.*?)[\s:.\-]+(?:₹|Rs\.?|INR)?\s*"
                             r"(?P<amount>\d[\d,]*\.\d{2})\s*(?:/-)?\s*$")
BREAK = "\n"  # marks a <br> inside a table cell; other whitespace is collapsed
INFO_FLAGS = {"wrapped_description", "columns_by_arithmetic", "summary_of_detail", "footer_line",
              "total_by_arithmetic"}

HEADER_KEYS = (
    ("serial", re.compile(r"^(s\.?\s*no\.?|sl\.?\s*no\.?|sr\.?\s*no\.?|#|no\.?|sno)$", re.I)),
    ("tax", re.compile(r"\b[csiu]?gst\b|\btax\b|\bvat\b", re.I)),
    ("discount", re.compile(r"\bdisc(?:ount)?\b|\bconcession\b", re.I)),
    ("net_amount", re.compile(r"\bnet\b|\bpayable\b|\bbill\s+amount\b|\bfinal\b", re.I)),
    ("unit_rate", re.compile(r"\brate\b|\bunit\s*(?:price|cost)\b|\bprice\b|\bmrp\b|\btariff\b", re.I)),
    ("quantity", re.compile(r"^(?:qty|quantity|units?|nos?\.?|no\.?\s+of\s+\w+|days)\b", re.I)),
    ("name", re.compile(r"\bname\b|\bdescription\b|\bparticulars?\b", re.I)),
    ("other", re.compile(r"\b(?:code|hsn|sac|batch|exp(?:iry)?|date|dept|department|mfr|schedule|pack|uom)\b", re.I)),
    ("name", re.compile(r"\b(?:description|particulars?|services?|item(?:\s+name)?|tests?(?:\s+name)?|medicines?|"
                        r"drugs?|procedures?|details|narration)\b", re.I)),
    ("gross_amount", re.compile(r"\b(?:amount|gross|total|value|charges?)\b", re.I)),
)


class TableReader(HTMLParser):
    """Collect table rows, expanding colspan/rowspan so columns stay aligned.

    Rows and cells left open by a truncated response are still returned.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []  # each: {"cells": [str], "spans": [int]}
        self.row = None
        self.spans = None
        self.cell = None
        self.colspan = 1
        self.rowspan = 1
        self.depth = 0
        self.pending = {}  # column index -> rows still covered by an earlier rowspan

    @staticmethod
    def _span(attrs, name):
        try:
            return min(max(int(dict(attrs).get(name) or 1), 1), 50)
        except ValueError:
            return 1

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.depth += 1
        elif self.depth != 1:
            if self.cell is not None and tag in ("br", "p", "div", "li"):
                self.cell.append(BREAK)
        elif tag == "tr":
            self._flush_row()
            self.row, self.spans = [], []
        elif tag in ("td", "th"):
            self._close_cell()
            if self.row is None:
                self.row, self.spans = [], []
            self.cell = []
            self.colspan, self.rowspan = self._span(attrs, "colspan"), self._span(attrs, "rowspan")
        elif self.cell is not None and tag in ("br", "p", "div", "li"):
            self.cell.append(BREAK)

    def handle_startendtag(self, tag, attrs):
        if self.cell is not None and tag == "br":
            self.cell.append(BREAK)

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data.replace("\n", " ").replace("\r", " "))

    def handle_endtag(self, tag):
        if self.depth != 1:
            if tag == "table" and self.depth:
                self.depth -= 1
            return
        if tag in ("td", "th"):
            self._close_cell()
        elif tag == "tr":
            self._flush_row()
        elif tag == "table":
            self._flush_row()
            self.depth -= 1

    def _fill_rowspans(self):
        while len(self.row) in self.pending:
            column = len(self.row)
            self.row.append("")
            self.spans.append(0)
            self.pending[column] -= 1
            if not self.pending[column]:
                del self.pending[column]

    def _close_cell(self):
        if self.cell is None:
            return
        lines = (" ".join(line.split()) for line in "".join(self.cell).split(BREAK))
        text = BREAK.join(line for line in lines if line)
        self._fill_rowspans()
        start = len(self.row)
        self.row.append(text)
        self.spans.append(self.colspan)
        for _ in range(self.colspan - 1):
            self.row.append("")
            self.spans.append(0)
        if self.rowspan > 1:
            for column in range(start, start + self.colspan):
                self.pending[column] = self.rowspan - 1
        self.cell = None

    def _flush_row(self):
        self._close_cell()
        if self.row is None:
            return
        self._fill_rowspans()
        for column in sorted(self.pending):
            if column >= len(self.row):
                while len(self.row) < column:
                    self.row.append("")
                    self.spans.append(1)
                self._fill_rowspans()
        if any(cell.strip() for cell in self.row):
            self.rows.append({"cells": self.row, "spans": self.spans})
        self.row = self.spans = None

    def close(self):
        super().close()
        self._flush_row()


def _field(text, labels):
    for label in labels:
        match = re.search(r"(?im)^\s*" + label + r"\s*[:#-]\s*([^\n|<]+)", text)
        if match:
            return match.group(1).strip()[:255]
    return None


def _provider(text):
    for line in text.splitlines()[:10]:
        candidate = re.sub(r"<[^>]+>", "", line).strip()
        if re.search(r"(?i)\b(hospital|clinic|medical centre|medical center)\b", candidate) and not re.search(r"(?i)charges|bill amount", candidate):
            return candidate[:255]
    return None


def _column_key(cell):
    for key, pattern in HEADER_KEYS:
        if pattern.search(cell):
            return key
    return None


def _header_map(cells):
    """Return a column map when a row is a header, else None.

    A header names at least two columns, one of them numeric, and is mostly
    text; a data row such as "Consultation Charges | 1 | 500 | 500" is not.
    """
    filled = [cell for cell in cells if cell.strip()]
    numeric = [cell for cell in filled if parse_number(cell).value is not None]
    if not filled or len(numeric) * 2 > len(filled):
        return None
    fields, taxes, others, recognised = {}, [], [], 0
    for index, cell in enumerate(cells):
        if not cell.strip() or parse_number(cell).value is not None:
            continue
        key = _column_key(cell.replace(BREAK, " "))
        if not key:
            continue
        recognised += 1
        if key == "tax":
            taxes.append(index)
        elif key == "other":
            others.append(index)
            fields.setdefault("other", index)
        elif key not in fields:
            fields[key] = index
    if recognised < 2 or not (set(fields) & set(NUMERIC_FIELDS) or taxes):
        return None
    if "name" not in fields:
        used = set(fields.values()) | set(taxes)
        fields["name"] = next((index for index in range(len(cells)) if index not in used), 0)
    if "unit_rate" not in fields and "quantity" in fields:
        # Some Nanonets table headers transcribe the rate heading as "0.00".
        # Only use a numeric column located between description and quantity.
        for index in range(fields["name"] + 1, fields["quantity"]):
            if parse_number(cells[index]).value is not None:
                fields["unit_rate"] = index
                break
    return {"fields": fields, "taxes": taxes, "others": others, "width": len(cells),
            "labels": {str(index): cells[index] for index in taxes},
            "payment": any(PAYMENT_HEADER.search(cell) for cell in cells)}


def _heading(text):
    lines = [re.sub(r"<[^>]+>", "", line).strip(" #*: \t") for line in text.splitlines()]
    for line in reversed(lines):
        if not line or len(line) > 80 or ":" in line or looks_numeric(line):
            continue
        if line.startswith("|") or line.lower().startswith(("invoice", "patient", "date", "bill")):
            continue
        if GROUP_HINT.search(line):
            return line
    return None


def _render_markdown_tables(text):
    """Convert pipe tables (with or without outer pipes) to HTML tables."""
    separator = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$")
    lines = text.split("\n")
    output, block = [], []

    def flush():
        if len(block) >= 2 and (any(separator.match(line) for line in block)
                                or all(line.strip().startswith("|") for line in block)):
            rows = []
            for line in block:
                if separator.match(line):
                    continue
                cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
                cells = [re.sub(r"&lt;br\s*/?&gt;", "<br>", escape(cell), flags=re.I) for cell in cells]
                rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
            output.append("<table>" + "".join(rows) + "</table>")
        else:
            output.extend(block)
        block.clear()

    for line in lines:
        if "|" in line and "<t" not in line.lower():
            block.append(line)
        else:
            flush()
            output.append(line)
    flush()
    return "\n".join(output)


def classify(label, has_group_items):
    if not label:
        return "item"
    if ADJUSTMENT.search(label):
        return "adjustment"
    if PAYMENT.search(label):
        return "payment"
    if TAX.search(label):
        return "tax"
    if DISCOUNT_ROW.search(label):
        return "discount"
    if TOTAL.search(label) or NET_AMOUNT_LINE.match(label):
        return "total"
    if BARE_TOTAL.match(label):
        # A bare "Total" closes the current billing group when it has items.
        return "subtotal" if has_group_items else "total"
    if SUBTOTAL.search(label):
        return "subtotal"
    return "item"


def _flag(code, message, **details):
    return {"code": code, "message": message, **{key: value for key, value in details.items() if value is not None}}


def _parse_field(field, raw, flags):
    parsed = parse_quantity(raw) if field == "quantity" else parse_number(raw)
    if parsed.issue:
        text = raw.replace(BREAK, " / ")
        message = f"{field.replace('_', ' ')} '{text}' could not be read as a number"
        if parsed.suggestion:
            message += f"; it may be {parsed.suggestion} (needs confirmation)"
        flags.append(_flag(parsed.issue, message, field=field, raw=text, suggestion=parsed.suggestion))
    return parsed.value


def _values(cells, columns, label_index, flags):
    """Map a row's cells to numeric fields without inventing any value."""
    values = dict.fromkeys(NUMERIC_FIELDS)
    taxes = {}
    serial = None
    if columns:
        fields = columns["fields"]
        for field in NUMERIC_FIELDS:
            index = fields.get(field)
            if index is not None and index < len(cells) and cells[index].strip():
                values[field] = _parse_field(field, cells[index], flags)
        for index in columns["taxes"]:
            if index < len(cells) and cells[index].strip():
                value = _parse_field("tax_amount", cells[index], flags)
                if value is not None:
                    taxes[columns["labels"][str(index)]] = value
        index = fields.get("serial")
        if index is not None and index < len(cells):
            serial = cells[index].strip().rstrip(".") or None
        return values, taxes, serial
    # No header: read only what the layout makes unambiguous.
    if label_index:
        lead = cells[label_index - 1].strip().rstrip(".")
        if lead.isdigit():
            serial = lead
    start = 0 if label_index is None else label_index + 1
    numbers = [(index, cell) for index, cell in enumerate(cells) if index >= start and _numberish(cell)]
    parsed = [_parse_field("net_amount", cell, flags) for _, cell in numbers]
    present = [value for value in parsed if value is not None]
    if not present:
        return values, taxes, serial
    values["net_amount"] = parsed[-1]
    if len(parsed) >= 3 and all(value is not None for value in parsed[-3:]):
        quantity, rate, amount = (Decimal(value) for value in parsed[-3:])
        if abs(quantity * rate - amount) <= Decimal("0.02"):
            values["quantity"], values["unit_rate"] = parsed[-3], parsed[-2]
            return values, taxes, serial
    if len(parsed) > 1:
        flags.append(_flag("columns_unknown", "Table has no header; only the last number was used as the amount",
                           raw=" | ".join(cell for _, cell in numbers)))
    return values, taxes, serial


def _is_date(cell):
    return bool(DATE.match(cell))


def _numberish(cell):
    """A printed number, possibly garbled by OCR (a few letters such as O or I)."""
    return (any(ch.isdigit() for ch in cell) and sum(ch.isalpha() for ch in cell) <= 3
            and not _is_date(cell))


def _label_index(cells, columns):
    if columns:
        return columns["fields"].get("name", 0)
    for index, cell in enumerate(cells):
        if re.search(r"[A-Za-z]{2}", cell) and not looks_numeric(cell) and not _is_date(cell):
            return index
    return None


def _covering(cells, spans, index):
    """Index of the cell that a colspan stretches over ``index`` (itself if not covered)."""
    if index is None or index >= len(cells) or cells[index].strip():
        return index
    for start in range(index - 1, -1, -1):
        if spans[start] > 0:
            return start if start + spans[start] > index and cells[start].strip() else index
    return index


def _row_numbers(cells):
    numbers = {}
    for index, cell in enumerate(cells):
        if cell.strip() and BREAK not in cell and not _is_date(cell):
            value = parse_number(cell, decimals=3).value
            if value is not None:
                numbers[index] = Decimal(value)
    return numbers


def _arithmetic_share(rows, quantity, rate, amount):
    """Share of rows with all three values where quantity × rate = amount, or None.

    Rows with a quantity of 1 are skipped: there rate and amount are equal, so
    they cannot tell a correct column map from a swapped one.
    """
    checked = matched = 0
    for numbers in rows:
        if all(index in numbers for index in (quantity, rate, amount))                 and abs(numbers[quantity]) != 1 and abs(numbers[rate]) != 1:
            checked += 1
            # Returns print negative quantities and rates; compare magnitudes.
            matched += abs(abs(numbers[quantity] * numbers[rate]) - abs(numbers[amount])) <= Decimal("0.05")
    return matched / checked if checked else None


def _infer_columns(rows, min_hits=2):
    """Find the quantity, rate and amount columns from rows where quantity × rate = amount.

    Used when a table has no header, or when its header (sometimes invented by
    the OCR model on continuation pages) does not fit the printed numbers.
    """
    numbers = [_row_numbers(row["cells"]) for row in rows]
    votes = Counter()
    for values in numbers:
        columns = sorted(values)
        for first in columns:
            for second in columns:
                if second <= first:
                    continue
                for amount in columns:
                    if amount <= second:
                        continue
                    a, b, c = values[first], values[second], values[amount]
                    if a and b and c and abs(abs(a * b) - abs(c)) <= Decimal("0.05"):
                        votes[(first, second, amount)] += 1
    if not votes:
        return None
    (first, second, amount), hits = max(votes.items(), key=lambda vote: (vote[1], -vote[0][2]))
    share = _arithmetic_share(numbers, first, second, amount)
    if hits < min(min_hits, len(rows)) or (share is not None and share < 0.6):
        return None

    def whole(index):
        # Quantities are usually printed without decimals ("2"), rates with them ("114.00").
        return sum(1 for row in rows if index < len(row["cells"]) and "." not in row["cells"][index]
                   and _row_numbers(row["cells"]).get(index) is not None)

    quantity, rate = (first, second) if whole(first) >= whole(second) else (second, first)
    return {"quantity": quantity, "unit_rate": rate, "gross_amount": amount, "share": share}


def _resolve_columns(rows, header, widest, inherited_from):
    """Pick the column map for a table and say how it was chosen.

    Returns (columns or None, method suffix). Arithmetic decides between a
    printed header, an inherited header and inferred columns; it never
    changes a value, only which column a value is read from.
    """
    numbers = [_row_numbers(row["cells"]) for row in rows]
    base, suffix = header, ""
    if base is None and inherited_from and abs(inherited_from["width"] - widest) <= 2:
        base, suffix = inherited_from, "+inherited_columns"
    inferred = _infer_columns(rows)
    if base:
        fields = base["fields"]
        amount = fields.get("gross_amount", fields.get("net_amount"))
        share = (_arithmetic_share(numbers, fields["quantity"], fields["unit_rate"], amount)
                 if None not in (fields.get("quantity"), fields.get("unit_rate"), amount) else None)
        if share == 0 and not inferred:
            # The printed header contradicts every row; one row that fits is enough evidence.
            inferred = _infer_columns(rows, min_hits=1)
        inferred_share = inferred.pop("share") if inferred else None
        if base["width"] == widest and (not inferred or share is None and inferred_share is None
                                        or (share or 0) >= (inferred_share or 0)):
            return base, suffix
        if not inferred:
            return None, ""
        kept = {key: index for key, index in fields.items()
                if key not in NUMERIC_FIELDS and (base["width"] == widest or key == "name")}
        taxes = base["taxes"] if base["width"] == widest else []
        columns = {"fields": {**kept, **inferred}, "taxes": taxes,
                   "others": base.get("others", []) if base["width"] == widest else [],
                   "width": widest, "labels": base["labels"] if taxes else {}, "payment": False}
        if "name" not in columns["fields"] or base["width"] != widest:
            labels = Counter(_label_index(row["cells"], None) for row in rows)
            labels.pop(None, None)
            columns["fields"]["name"] = labels.most_common(1)[0][0] if labels else 0
        return columns, suffix + "+columns_by_arithmetic"
    if inferred:
        inferred.pop("share")
        labels = Counter(_label_index(row["cells"], None) for row in rows)
        labels.pop(None, None)
        name = labels.most_common(1)[0][0] if labels else 0
        return {"fields": {"name": name, **inferred}, "taxes": [], "others": [], "width": widest,
                "labels": {}, "payment": False}, "+columns_by_arithmetic"
    return None, ""


def _split_merged(cells, columns, label_index):
    """Split one OCR row into several when its cells hold aligned <br> lines.

    Returns a list of cell lists, and whether a split happened. Only splits
    when every non-empty value column has the same line count as the label.
    """
    if label_index is None or label_index >= len(cells):
        return [cells], False
    names = cells[label_index].split(BREAK)
    others = [index for index, cell in enumerate(cells) if index != label_index and cell.strip()]
    if len(names) < 2 or not others:
        return [cells], False
    counts = {len(cells[index].split(BREAK)) for index in others}
    if counts != {len(names)}:
        return [cells], False
    if columns:
        value_columns = set(columns["fields"].values()) | set(columns["taxes"])
        if not any(index in value_columns for index in others):
            return [cells], False
    split = []
    for line in range(len(names)):
        split.append([cell.split(BREAK)[line] if cell.strip() else "" for cell in cells])
    return split, True


def _untabled_lines(fragment):
    found = []
    for line in fragment.splitlines():
        clean = re.sub(r"<[^>]+>", " ", line).strip(" #*|\t")
        match = UNTABLED_AMOUNT.match(" ".join(clean.split()))
        if not match:
            continue
        label = match.group("label").strip()
        if HEADER_FIELD_LINE.search(label) or classify(label, False) != "item":
            continue
        found.append({"text": clean[:300], "label": label[:200], "amount": match.group("amount")})
    return found


def _text_totals(fragment):
    for line in fragment.splitlines():
        clean = " ".join(re.sub(r"<[^>]+>", " ", line).split())
        match = UNTABLED_AMOUNT.match(clean)
        if match and TOTAL.search(match.group("label")):
            value = parse_number(match.group("amount")).value
            if value is not None:
                yield clean, match.group("label").strip(), value


class _Builder:
    def __init__(self, result, method):
        self.result = result
        self.method = method
        self.group_id = self.group_name = self.group_page = None
        self.group_has_items = False
        self.last_columns = None
        self.last_item = None

    def new_group(self, name, page_number, source):
        self.group_name, self.group_page = name, page_number
        self.group_id = f"g-{page_number}-{len(self.result['groups']) + 1}"
        self.group_has_items = False
        self.result["groups"].append({"id": self.group_id, "name": name[:255],
                                      "page_number": page_number, "source_text": source[:500]})

    def relationship(self, kind, page_number):
        if not self.group_id or kind in ("total", "payment", "adjustment"):
            return "unassigned"
        return "observed" if self.group_page == page_number else "inferred_cross_page"


def extract(pages, method="page_ocr"):
    """pages: iterable of (page_number, OCR text), in original order."""
    pages = list(pages)
    all_text = "\n".join(text for _, text in pages)
    result = {
        "schema_version": SCHEMA_VERSION,
        "invoice_number": _field(all_text, (r"invoice\s*(?:no\.?|number)", r"bill\s*(?:no\.?|number)")),
        "invoice_date": _field(all_text, (r"invoice\s*date", r"bill\s*date", r"date")),
        "patient_name": _field(all_text, (r"patient\s*name", r"name")),
        "provider_name": _provider(all_text),
        "grand_total": None,
        "field_evidence": {}, "groups": [], "items": [], "tables": [], "pages": [],
        "candidate_rows": [],
    }
    for field in ("invoice_number", "invoice_date", "patient_name", "provider_name"):
        value = result[field]
        if value:
            for number, page_text in pages:
                line = next((line.strip() for line in page_text.splitlines() if value in line), None)
                if line:
                    result["field_evidence"][field] = {"page_number": number, "source_text": line[:500]}
                    break
    builder = _Builder(result, method)
    for page_number, text in pages:
        text = _render_markdown_tables(text or "")
        unclosed = len(re.findall(r"<table\b", text, re.I)) > len(re.findall(r"</table>", text, re.I))
        if unclosed:
            text += "</table>"  # keep the rows of a table cut off by truncation
        fragments = re.split(r"(<table\b[^>]*>.*?</table>)", text, flags=re.I | re.S)
        page_info = {"page_number": page_number, "text_chars": len(text), "tables": 0,
                     "unclosed_table": unclosed, "untabled_amount_lines": []}
        result["pages"].append(page_info)
        row_order = 0
        for fragment in fragments:
            if not fragment.lower().lstrip().startswith("<table"):
                candidate = _heading(fragment)
                if candidate and candidate != builder.group_name:
                    builder.new_group(candidate, page_number, candidate)
                page_info["untabled_amount_lines"] += _untabled_lines(fragment)
                for source, label, value in _text_totals(fragment):
                    row_order += 1
                    result["items"].append(_row(result, builder, page_number, None, row_order, "total", label,
                                                {"net_amount": value}, {}, None, source, "text_line", []))
                continue
            page_info["tables"] += 1
            row_order = _read_table(result, builder, page_number, page_info["tables"], fragment,
                                    row_order, unclosed and fragment is fragments[-2])
    _mark_summaries(result)
    _totals_by_arithmetic(result)
    _groups_from_subtotals(result)
    totals = [row for row in result["items"] if row["kind"] == "total"
              and (row["net_amount"] or row["gross_amount"])]
    if totals:
        result["grand_total"] = totals[-1]["net_amount"] or totals[-1]["gross_amount"]
    return result


def _row(result, builder, page_number, table_number, row_order, kind, name, values, taxes, serial,
         source, method, flags):
    row = {"id": f"r-{page_number}-{row_order}", "page_number": page_number, "table_number": table_number,
           "row_order": row_order, "kind": kind, "name": (name or "")[:500], "serial": serial,
           "group_id": builder.group_id if builder.relationship(kind, page_number) != "unassigned" else None,
           "relationship": builder.relationship(kind, page_number),
           "source_text": source[:2000], "extraction_method": method, "flags": flags,
           "review_status": "auto_extracted", "tax_amounts": taxes,
           **{field: values.get(field) for field in NUMERIC_FIELDS}}
    if kind == "item":
        builder.group_has_items = True
        builder.last_item = row
    elif kind == "subtotal":
        builder.group_has_items = False
    return row


def _read_table(result, builder, page_number, table_number, fragment, row_order, cut_off):
    parser = TableReader()
    parser.feed(fragment)
    parser.close()
    widest = max((len(row["cells"]) for row in parser.rows), default=0)
    header_rows = [index for index, row in enumerate(parser.rows) if _header_map(row["cells"])]
    first_header = header_rows[0] if header_rows and header_rows[0] <= 2 else None
    data_rows = [row for index, row in enumerate(parser.rows) if index not in header_rows]
    printed = _header_map(parser.rows[first_header]["cells"]) if first_header is not None else None
    resolved, suffix = _resolve_columns(data_rows, printed, widest,
                                        None if printed else builder.last_columns)
    columns = None if printed else resolved
    inherited = "+inherited_columns" in suffix and not printed
    table = {"page_number": page_number, "table_number": table_number,
             "method": builder.method, "rows_seen": len(parser.rows), "columns_inherited": inherited,
             "columns_by_arithmetic": "+columns_by_arithmetic" in suffix,
             "has_header": False, "cut_off": cut_off, "dispositions": Counter(), "row_kinds": []}
    result["tables"].append(table)
    # A table without its own header may continue the previous page or strip.
    previous = None if printed else builder.last_item
    at_start = True
    for position, raw in enumerate(parser.rows):
        cells, spans = raw["cells"], raw["spans"]
        source = " | ".join(cell.replace(BREAK, " / ") for cell in cells)
        header = _header_map(cells)
        if header:
            columns = resolved if position == first_header and resolved else header
            builder.last_columns = columns
            table["has_header"] = True
            table["dispositions"]["header"] += 1
            previous = None
            continue
        label_index = _covering(cells, spans, _label_index(cells, columns))
        filled = [index for index, cell in enumerate(cells) if cell.strip()]
        if len(filled) == 1 and re.search(r"[A-Za-z]{2}", cells[filled[0]]):
            label_index = filled[0]  # a lone text cell (often a colspan heading) is the row's label
        label = cells[label_index].replace(BREAK, " ").strip() if label_index is not None and label_index < len(cells) else ""
        others = set(columns.get("others", [])) if columns else set()
        context = [index for index in filled if index != label_index and (index in others or _is_date(cells[index]))]
        value_cells = [index for index in filled if index != label_index and index not in context]
        if columns and columns["fields"].get("serial") in value_cells and len(value_cells) == 1:
            value_cells = []  # a serial number alone is not a value
        serial_index = columns["fields"].get("serial") if columns else (
            label_index - 1 if label_index and cells[label_index - 1].strip().rstrip(".").isdigit() else None)
        has_serial = (serial_index is not None and serial_index != label_index and serial_index < len(cells)
                      and cells[serial_index].strip().rstrip(".").isdigit())
        if not value_cells and not has_serial and label:
            # Text-only row: billing heading, wrapped description, or an item whose amount is blank.
            kind = classify(label, builder.group_has_items)
            spanned = len(filled) == 1 and (len(cells) == 1 or spans[filled[0]] > 1)
            tail = previous["name"].rstrip()[-1:] if previous else ""
            continues = previous is not None and not spanned and (
                label[:1].islower() or (tail != "" and tail in "-,&(/") or (at_start and bool(context)))
            if kind == "item" and continues:
                previous["name"] = (previous["name"] + " " + label)[:500]
                previous["source_text"] = (previous["source_text"] + " / " + source)[:2000]
                if not any(flag["code"] == "wrapped_description" for flag in previous["flags"]):
                    previous["flags"].append(_flag("wrapped_description",
                                                   "Description continues on the next OCR row and was joined"))
                table["dispositions"]["continuation"] += 1
                at_start = False
                continue
            if kind == "item" and not context and (spanned or GROUP_HINT.search(label)
                                                   or (label.isupper() and len(label) <= 60)):
                builder.new_group(label, page_number, source)
                table["dispositions"]["heading"] += 1
                previous = None
                at_start = False
                continue
        prior = previous if previous is not None else builder.last_item
        if at_start and prior is not None and label_index is not None and prior["page_number"] != page_number:
            lines = cells[label_index].split(BREAK)
            if len(lines) > 1 and len(lines[0]) <= 25 and not re.search(r"\d", lines[0]):
                # e.g. "Wing_MSR<br>CHEST PHYSIOTHERAPY ...": the first line ends the previous page's row.
                prior["name"] = (prior["name"] + " " + lines[0])[:500]
                prior["source_text"] = (prior["source_text"] + " / " + lines[0])[:2000]
                prior["flags"].append(_flag("fragment_from_next_page", f"'{lines[0]}' printed at the top of page "
                                               f"{page_number} was joined to this description; check it"))
                prior["review_status"] = "needs_review"
                cells = list(cells)
                cells[label_index] = BREAK.join(lines[1:])
        at_start = False
        pieces, was_split = _split_merged(cells, columns, label_index)
        for piece in pieces:
            flags = []
            if was_split:
                flags.append(_flag("split_from_merged_cell",
                                   "Several charges were in one OCR row and were split line by line"))
            piece_label = piece[label_index].replace(BREAK, " ").strip() if label_index is not None and label_index < len(piece) else ""
            if not was_split and any(BREAK in piece[index] for index in range(len(piece)) if index != label_index):
                flags.append(_flag("merged_values", "A cell holds several values that do not line up with the "
                                   "description; those values were not used",
                                   raw=" | ".join(cell.replace(BREAK, " / ") for cell in piece)))
                piece = [cell if BREAK not in cell or index == label_index else "" for index, cell in enumerate(piece)]
            values, taxes, serial = _values([cell if index not in context else "" for index, cell in enumerate(piece)],
                                            columns, label_index, flags)
            kind = "payment" if columns and columns.get("payment") else classify(piece_label, builder.group_has_items)
            if kind != "item":
                values["quantity"] = values["unit_rate"] = None  # totals have no quantity or rate
            if kind == "item":
                if not piece_label:
                    flags.append(_flag("missing_description", "Row has values but no description"))
                if values["net_amount"] is None and values["gross_amount"] is None:
                    flags.append(_flag("missing_amount", "No amount was printed or read for this row"))
            elif kind != "payment" and values["net_amount"] is None and values["gross_amount"] is None                     and not taxes:
                printed = [parse_number(piece[index]).value for index in value_cells if index < len(piece)]
                printed = [value for value in printed if value is not None]
                if printed:
                    # A colspan shifted this total's cells off the header columns.
                    values = dict.fromkeys(NUMERIC_FIELDS)
                    values["net_amount"] = printed[-1]
                    flags.append(_flag("columns_unknown", f"{kind.title()} row cells do not line up with the "
                                       "header; the last printed number was used", raw=source))
                else:
                    flags.append(_flag("missing_amount", f"{kind.title()} row has no readable amount"))
            if "+columns_by_arithmetic" in suffix and kind == "item":
                flags.append(_flag("columns_by_arithmetic", "Quantity, rate and amount columns were chosen where "
                                   "quantity × rate = amount, because the header was missing or did not fit"))
            row_order += 1
            source_text = " | ".join(cell.replace(BREAK, " / ") for cell in piece) if was_split else source
            method = builder.method + suffix + ("+split" if was_split else "")
            row = _row(result, builder, page_number, table_number, row_order, kind, piece_label, values,
                       taxes, serial, source_text, method, flags)
            if any(flag["code"] not in INFO_FLAGS for flag in flags):
                row["review_status"] = "needs_review"
            result["items"].append(row)
            table["dispositions"][kind] += 1
            table["row_kinds"].append(kind)
            previous = row if kind == "item" else None
    table["dispositions"] = dict(table["dispositions"])
    return row_order


def _base_name(name):
    """Section name without codes, brackets or the word total, for matching summaries to details."""
    name = re.sub(r"\(.*?\)", " ", name.casefold())
    name = re.sub(r"\b(?:sac|hsn)\b.*$|\btotal\b", " ", name)
    return re.sub(r"[^a-z]", "", name)


def _mark_summaries(result):
    """Re-label rows that restate other rows, so amounts are not counted twice.

    * A summary table (e.g. a department-wise first page) whose rows name the
      billing sections itemised elsewhere in the invoice.
    * A label/amount footer block (Net Amount, Payor Balance, ...) whose other
      rows are totals, taxes or payments.
    Rows are kept, with kind "summary" and an info flag; nothing is deleted.
    """
    sections = {_base_name(group["name"]) for group in result["groups"]}
    sections |= {_base_name(row["name"]) for row in result["items"] if row["kind"] == "subtotal"}
    sections.discard("")
    tables = defaultdict(list)
    for row in result["items"]:
        if row.get("table_number"):
            tables[(row["page_number"], row["table_number"])].append(row)
    accounting = {(table["page_number"], table["table_number"]): table for table in result["tables"]}
    for key, rows in tables.items():
        items = [row for row in rows if row["kind"] == "item"]
        if not items:
            continue
        detail_elsewhere = any(row["kind"] == "item" and (row["page_number"], row.get("table_number")) != key
                               for row in result["items"])
        named = [row for row in items if _base_name(row["name"]) in sections]
        if detail_elsewhere and len(items) >= 2 and len(named) >= 0.6 * len(items):
            code, message = "summary_of_detail", "Section total from a summary table; its charges are itemised elsewhere"
        elif len(rows) >= 3 and len(rows) - len(items) >= 2 and all(
                len([cell for cell in row["source_text"].split(" | ") if cell.strip()]) <= 2 for row in rows):
            code, message = "footer_line", "Label and amount from the bill's totals block, not a charge"
        else:
            continue
        for row in items:
            row["kind"] = "summary"
            row["group_id"], row["relationship"] = None, "unassigned"
            row["flags"].append(_flag(code, message))
            row["review_status"] = "auto_extracted" if all(
                flag["code"] in INFO_FLAGS for flag in row["flags"]) else row["review_status"]
        dispositions = accounting[key]["dispositions"]
        dispositions["item"] -= len(items)
        dispositions["summary"] = dispositions.get("summary", 0) + len(items)
        if not dispositions["item"]:
            del dispositions["item"]


def _amount(row):
    for field in ("net_amount", "gross_amount"):
        if row.get(field) not in (None, ""):
            return Decimal(row[field])
    return None


def _totals_by_arithmetic(result):
    """Recognise section and grand totals by their numbers, not their wording.

    Labels differ between hospitals and the OCR model may reword them, so a
    row with no quantity or rate whose amount equals the sum of at least two
    item rows before it is a total whatever it is called: the sum since the
    last section boundary makes it a section total, the sum of every item so
    far a grand total. Rows already classified by their label are only used
    as boundaries; their arithmetic is checked later by validation.
    """
    tolerance = Decimal("0.02")
    section, everything = [], []
    for row in result["items"]:
        kind = row["kind"]
        if kind == "subtotal":
            section = []
            continue
        if kind != "item":
            continue
        amount = _amount(row)
        printed_alone = row.get("quantity") is None and row.get("unit_rate") is None
        if amount is not None and printed_alone:
            section_sum = sum((_amount(item) for item in section), Decimal(0))
            all_sum = sum((_amount(item) for item in everything), Decimal(0))
            if len(section) >= 2 and abs(section_sum - amount) <= tolerance:
                row["kind"] = "subtotal"
                row["flags"].append(_flag("total_by_arithmetic", "Amount equals the sum of the items above it, "
                                          "so it is treated as a section total"))
                section = []
                continue
            if len(everything) >= 2 and abs(all_sum - amount) <= tolerance and len(everything) > len(section):
                row["kind"] = "total"
                row["group_id"], row["relationship"] = None, "unassigned"
                row["flags"].append(_flag("total_by_arithmetic", "Amount equals the sum of all items above it, "
                                          "so it is treated as the invoice total"))
                section = []
                continue
        if amount is None:
            section = []  # an unreadable amount breaks the running sums; do not guess past it
            everything = []
            continue
        section.append(row)
        everything.append(row)


def _groups_from_subtotals(result):
    """Use section totals to place items in sections without relying on wording.

    Items between two section totals belong to the later total. If those
    items already sit in a section whose heading was read and not yet closed,
    the total simply closes it. If their section was already closed by an
    earlier total (its heading was missed), the total's own label names the
    section: an existing section with the same name, or a new one.
    """
    groups = {_base_name(group["name"]): group for group in result["groups"]}
    closed, pending = set(), []
    for row in result["items"]:
        if row["kind"] == "item":
            pending.append(row)
            continue
        if row["kind"] != "subtotal":
            continue
        current = {item["group_id"] for item in pending}
        base = _base_name(row["name"])
        group = groups.get(base) if base else None
        if group is None and len(current) == 1 and None not in current and not current & closed:
            group_id = current.pop()  # the open section's heading was read: this total closes it
            group = next(group for group in result["groups"] if group["id"] == group_id)
        if group is None and base and pending:
            name = re.sub(r"\s*\btotal\b\s*[:.]?\s*$", "", row["name"], flags=re.I).strip() or row["name"]
            group = {"id": f"g-{row['page_number']}-{len(result['groups']) + 1}", "name": name[:255],
                     "page_number": row["page_number"], "source_text": row["source_text"][:500]}
            result["groups"].append(group)
            groups[base] = group
        if group is not None:
            for item in pending:
                if item["group_id"] != group["id"]:
                    item["group_id"], item["relationship"] = group["id"], "inferred_from_section_total"
            row["group_id"] = group["id"]
            if row["relationship"] == "unassigned":
                row["relationship"] = "observed"
            closed.add(group["id"])
        pending = []
