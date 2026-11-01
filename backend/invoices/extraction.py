"""Conservative structure reconstruction from Nanonets page text/HTML tables.

Only values present in an OCR row are emitted; nothing is inferred from totals.
Every table row is accounted for (item, heading, header, continuation, total,
...), so a row that cannot be classified is kept and flagged instead of being
dropped. Source text, page, table and row order accompany every row, and the
per-table row accounting feeds the completeness check.
"""

import re
from collections import Counter
from decimal import Decimal
from html import escape
from html.parser import HTMLParser

from .numbers import looks_numeric, parse_number, parse_quantity

SCHEMA_VERSION = "1.1"
NUMERIC_FIELDS = ("quantity", "unit_rate", "gross_amount", "discount", "net_amount")
ROW_KINDS = ("item", "subtotal", "total", "tax", "discount", "payment", "adjustment")

TOTAL = re.compile(r"\b(grand\s+total|net\s+(?:amount\s+)?payable|amount\s+payable|total\s+amount|bill\s+total|"
                   r"total\s+bill(?:\s+amount)?|invoice\s+total|total\s+payable|net\s+total)\b", re.I)
SUBTOTAL = re.compile(r"\b(sub\s*-?\s*total|section\s+total|group\s+total)\b|\w\s+total\s*[:.]?\s*$", re.I)
BARE_TOTAL = re.compile(r"^\s*total(?:\s+charges)?\s*(?:\(?\s*(?:rs\.?|inr|₹)\s*\)?)?\s*[:.]?\s*$", re.I)
TAX = re.compile(r"^\s*(?:add\s*[:\-]?\s*)?(?:total\s+)?(?:[csiu]\s*gst|gst|vat|cess|service\s+tax|tax)\b", re.I)
DISCOUNT_ROW = re.compile(r"^\s*(?:less\s*[:\-]?\s*)?(?:total\s+)?(?:discount|concession|rebate|waiver|deduction)\b", re.I)
PAYMENT = re.compile(r"^\s*(?:less\s*[:\-]?\s*)?(?:advance|deposit|amount\s+(?:paid|received)|paid|received|"
                     r"refund|balance|due\s+amount|amount\s+due)\b", re.I)
ADJUSTMENT = re.compile(r"\bround(?:ed|ing)?\s*-?\s*off\b|^\s*rounding\b", re.I)
GROUP_HINT = re.compile(r"(?i)charges|pharmacy|laboratory|radiology|room|procedure|consultation|others|nursing|"
                        r"consumables|medicine|surgery|doctor|investigation|services|implants|ward|icu|ot\b")
HEADER_FIELD_LINE = re.compile(r"(?i)^\s*(bill|invoice|receipt|ip|op|uhid|mrn|reg(?:istration)?|gstin|pan|phone|"
                               r"mobile|tel|date|age|admission|discharge|policy|claim|patient|doctor|consultant)\b")
UNTABLED_AMOUNT = re.compile(r"^(?P<label>.*?[A-Za-z]{3,}.*?)[\s:.\-]+(?:₹|Rs\.?|INR)?\s*"
                             r"(?P<amount>\d[\d,]*\.\d{2})\s*(?:/-)?\s*$")
BREAK = "\n"  # marks a <br> inside a table cell; other whitespace is collapsed

HEADER_KEYS = (
    ("serial", re.compile(r"^(s\.?\s*no\.?|sl\.?\s*no\.?|sr\.?\s*no\.?|#|no\.?|sno)$", re.I)),
    ("tax", re.compile(r"\b[csiu]?gst\b|\btax\b|\bvat\b", re.I)),
    ("discount", re.compile(r"\bdisc(?:ount)?\b|\bconcession\b", re.I)),
    ("net_amount", re.compile(r"\bnet\b|\bpayable\b|\bbill\s+amount\b|\bfinal\b", re.I)),
    ("unit_rate", re.compile(r"\brate\b|\bunit\s*(?:price|cost)\b|\bprice\b|\bmrp\b|\btariff\b", re.I)),
    ("quantity", re.compile(r"^(?:qty|quantity|units?|nos?\.?|no\.?\s+of\s+\w+|days)\b", re.I)),
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
    fields, taxes, recognised = {}, [], 0
    for index, cell in enumerate(cells):
        if not cell.strip() or parse_number(cell).value is not None:
            continue
        key = _column_key(cell.replace(BREAK, " "))
        if not key:
            continue
        recognised += 1
        if key == "tax":
            taxes.append(index)
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
    return {"fields": fields, "taxes": taxes, "width": len(cells),
            "labels": {str(index): cells[index] for index in taxes}}


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
    if TOTAL.search(label):
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


def _numberish(cell):
    """A printed number, possibly garbled by OCR (a few letters such as O or I)."""
    return any(ch.isdigit() for ch in cell) and sum(ch.isalpha() for ch in cell) <= 3


def _label_index(cells, columns):
    if columns:
        return columns["fields"].get("name", 0)
    for index, cell in enumerate(cells):
        if re.search(r"[A-Za-z]{2}", cell) and not looks_numeric(cell):
            return index
    return None


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
    elif kind == "subtotal":
        builder.group_has_items = False
    return row


def _read_table(result, builder, page_number, table_number, fragment, row_order, cut_off):
    parser = TableReader()
    parser.feed(fragment)
    parser.close()
    widest = max((len(row["cells"]) for row in parser.rows), default=0)
    columns = None
    inherited = False
    if parser.rows and _header_map(parser.rows[0]["cells"]) is None and builder.last_columns \
            and builder.last_columns["width"] == widest:
        columns, inherited = builder.last_columns, True
    table = {"page_number": page_number, "table_number": table_number,
             "method": builder.method, "rows_seen": len(parser.rows), "columns_inherited": inherited,
             "has_header": False, "cut_off": cut_off, "dispositions": Counter(), "row_kinds": []}
    result["tables"].append(table)
    previous = None  # last emitted item row in this table, for wrapped descriptions
    for raw in parser.rows:
        cells, spans = raw["cells"], raw["spans"]
        source = " | ".join(cell.replace(BREAK, " / ") for cell in cells)
        header = _header_map(cells)
        if header:
            columns, inherited = header, False
            builder.last_columns = header
            table["has_header"] = True
            table["dispositions"]["header"] += 1
            previous = None
            continue
        label_index = _label_index(cells, columns)
        filled = [index for index, cell in enumerate(cells) if cell.strip()]
        if len(filled) == 1 and re.search(r"[A-Za-z]{2}", cells[filled[0]]):
            label_index = filled[0]  # a lone text cell (often a colspan heading) is the row's label
        label = cells[label_index].replace(BREAK, " ").strip() if label_index is not None and label_index < len(cells) else ""
        value_cells = [index for index in filled if index != label_index]
        if columns and columns["fields"].get("serial") in value_cells and len(value_cells) == 1:
            value_cells = []  # a serial number alone is not a value
        serial_index = columns["fields"].get("serial") if columns else (
            label_index - 1 if label_index and cells[label_index - 1].strip().rstrip(".").isdigit() else None)
        has_serial = (serial_index is not None and serial_index != label_index and serial_index < len(cells)
                      and cells[serial_index].strip().rstrip(".").isdigit())
        if not value_cells and not has_serial:
            # Text-only row: billing heading, wrapped description, or an item whose amount is blank.
            kind = classify(label, builder.group_has_items)
            spanned = len(filled) == 1 and (len(cells) == 1 or spans[filled[0]] > 1)
            continues = previous is not None and (label[:1].islower() or
                                                  previous["name"].rstrip()[-1:] in "-,&(/")
            if kind == "item" and continues:
                previous["name"] = (previous["name"] + " " + label)[:500]
                previous["source_text"] = (previous["source_text"] + " / " + source)[:2000]
                if not any(flag["code"] == "wrapped_description" for flag in previous["flags"]):
                    previous["flags"].append(_flag("wrapped_description",
                                                   "Description continues on the next OCR row and was joined"))
                table["dispositions"]["continuation"] += 1
                continue
            if kind == "item" and (spanned or GROUP_HINT.search(label) or (label.isupper() and len(label) <= 60)):
                builder.new_group(label, page_number, source)
                table["dispositions"]["heading"] += 1
                previous = None
                continue
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
            values, taxes, serial = _values(piece, columns, label_index, flags)
            kind = classify(piece_label, builder.group_has_items)
            if kind == "item":
                if not piece_label:
                    flags.append(_flag("missing_description", "Row has values but no description"))
                if values["net_amount"] is None and values["gross_amount"] is None:
                    flags.append(_flag("missing_amount", "No amount was printed or read for this row"))
            elif not any(values[field] for field in NUMERIC_FIELDS) and not taxes:
                flags.append(_flag("missing_amount", f"{kind.title()} row has no readable amount"))
            if kind == "tax" and values["net_amount"] is None and values["gross_amount"] is None and taxes:
                values["net_amount"] = None  # keep tax columns separate; never sum them into an amount
            row_order += 1
            source_text = " | ".join(cell.replace(BREAK, " / ") for cell in piece) if was_split else source
            method = builder.method + ("+inherited_columns" if inherited else "") + ("+split" if was_split else "")
            row = _row(result, builder, page_number, table_number, row_order, kind, piece_label, values,
                       taxes, serial, source_text, method, flags)
            if any(flag["code"] != "wrapped_description" for flag in flags):
                row["review_status"] = "needs_review"
            result["items"].append(row)
            table["dispositions"][kind] += 1
            table["row_kinds"].append(kind)
            previous = row if kind == "item" else None
    table["dispositions"] = dict(table["dispositions"])
    return row_order
