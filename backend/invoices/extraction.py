"""Conservative structure reconstruction from Nanonets page text/HTML tables.

Only values present in an OCR row are emitted. Unclear group relationships stay
unassigned. Source text and the original page number accompany every row.
"""

import re
from html import escape
from html.parser import HTMLParser


SCHEMA_VERSION = "1.0"
AMOUNT = re.compile(r"^\s*(?:₹|Rs\.?\s*)?[-+]?\d[\d,]*(?:\.\d{1,2})?\s*$", re.I)
TOTAL = re.compile(r"\b(grand\s+total|net\s+payable|amount\s+payable|total\s+amount)\b", re.I)
SUBTOTAL = re.compile(r"\b(sub\s*total|section\s+total|group\s+total)\b", re.I)
HEADER_WORDS = ("description", "particular", "service", "item", "qty", "quantity", "rate", "amount", "discount", "net")


class TableReader(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.row = None
        self.cell = None
        self.depth = 0

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.depth += 1
        elif self.depth and tag == "tr":
            self.row = []
        elif self.depth and tag in ("td", "th"):
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None and self.row is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if any(self.row):
                self.rows.append(self.row)
            self.row = None
        elif tag == "table" and self.depth:
            self.depth -= 1


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


def _number(value):
    value = value.strip()
    if not AMOUNT.match(value):
        return None
    return re.sub(r"^(?:₹|Rs\.?\s*)", "", value, flags=re.I).replace(",", "").strip()


def _header_map(row):
    mapping = {}
    for index, cell in enumerate(row):
        value = cell.lower().strip()
        if any(word in value for word in ("description", "particular", "service", "item name")):
            mapping["name"] = index
        elif value in ("qty", "quantity", "units"):
            mapping["quantity"] = index
        elif "rate" in value or "unit price" in value:
            mapping["unit_rate"] = index
        elif "discount" in value or value == "disc":
            mapping["discount"] = index
        elif "net" in value or "payable" in value or "bill amount" in value:
            mapping["net_amount"] = index
        elif "amount" in value or "gross" in value or value == "total":
            mapping["gross_amount"] = index
    if "unit_rate" not in mapping and "name" in mapping and "quantity" in mapping:
        # Some Nanonets table headers transcribe the rate heading as "0.00".
        # Only use a numeric column located between description and quantity.
        for index in range(mapping["name"] + 1, mapping["quantity"]):
            if _number(row[index]) is not None:
                mapping["unit_rate"] = index
                break
    return mapping if "name" in mapping and len(mapping) > 1 else None


def _heading(text):
    lines = [re.sub(r"<[^>]+>", "", line).strip(" #*: \t") for line in text.splitlines()]
    for line in reversed(lines):
        if not line or len(line) > 80 or ":" in line or AMOUNT.match(line):
            continue
        if line.startswith("|") or line.lower().startswith(("invoice", "patient", "date", "bill")):
            continue
        if re.search(r"(?i)charges|pharmacy|laboratory|radiology|room|procedure|consultation|others|nursing|consumables|medicine|surgery|doctor", line):
            return line
    return None


def _render_markdown_tables(text):
    def render(match):
        lines = match.group().splitlines()
        rows = []
        for line in lines:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if cells and all(re.fullmatch(r"[-: ]+", cell or " ") for cell in cells):
                continue
            rows.append("<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in cells) + "</tr>")
        return "<table>" + "".join(rows) + "</table>" if rows else match.group()

    return re.sub(r"(?m)(?:^\|[^\n]*\|\s*$\n?){2,}", render, text)


def extract(pages):
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
        "field_evidence": {}, "groups": [], "items": [],
    }
    for field in ("invoice_number", "invoice_date", "patient_name", "provider_name"):
        value = result[field]
        if value:
            for number, page_text in pages:
                line = next((line.strip() for line in page_text.splitlines() if value in line), None)
                if line:
                    result["field_evidence"][field] = {"page_number": number, "source_text": line[:500]}
                    break
    group_id = None
    group_name = None
    group_page = None
    for page_number, text in pages:
        text = _render_markdown_tables(text)
        fragments = re.split(r"(<table\b[^>]*>.*?</table>)", text, flags=re.I | re.S)
        row_order = 0
        for fragment in fragments:
            if not fragment.lower().lstrip().startswith("<table"):
                candidate = _heading(fragment)
                if candidate and candidate != group_name:
                    group_name = candidate
                    group_page = page_number
                    group_id = f"g-{page_number}-{len(result['groups']) + 1}"
                    result["groups"].append({"id": group_id, "name": candidate,
                                             "page_number": page_number, "source_text": candidate})
                continue
            parser = TableReader()
            parser.feed(fragment)
            columns = None
            for row in parser.rows:
                if len(row) == 1 and row[0] and not AMOUNT.match(row[0]):
                    if not TOTAL.search(row[0]) and not SUBTOTAL.search(row[0]):
                        group_name = row[0]
                        group_page = page_number
                        group_id = f"g-{page_number}-{len(result['groups']) + 1}"
                        result["groups"].append({"id": group_id, "name": group_name,
                                                 "page_number": page_number, "source_text": row[0]})
                    continue
                maybe_header = _header_map(row)
                if maybe_header:
                    columns = maybe_header
                    continue
                if len(row) < 2:
                    continue
                name_index = columns.get("name", 0) if columns else 0
                name = row[name_index].strip() if name_index < len(row) else ""
                if not name or any(name.lower() == word for word in HEADER_WORDS):
                    continue
                kind = "total" if TOTAL.search(name) else "subtotal" if SUBTOTAL.search(name) else "item"
                amounts = {}
                for key in ("quantity", "unit_rate", "gross_amount", "discount", "net_amount"):
                    index = columns.get(key) if columns else None
                    amounts[key] = _number(row[index]) if index is not None and index < len(row) else None
                if not columns:
                    amounts["net_amount"] = _number(row[-1])
                if kind == "item" and not any(amounts.values()):
                    continue
                row_order += 1
                item = {"id": f"r-{page_number}-{row_order}", "page_number": page_number,
                        "row_order": row_order, "kind": kind, "name": name[:500],
                        "group_id": group_id if kind != "total" else None,
                        "relationship": ("observed" if group_page == page_number else "inferred_cross_page")
                        if group_id and kind != "total" else "unassigned",
                        "source_text": " | ".join(row), **amounts}
                result["items"].append(item)
                if kind == "total" and (amounts["net_amount"] or amounts["gross_amount"]):
                    result["grand_total"] = amounts["net_amount"] or amounts["gross_amount"]
    return result
