"""Parse printed invoice numbers without guessing.

Accepts Indian (1,23,456.00) and international (123,456.00) digit grouping,
currency marks (₹, Rs, INR), the "/-" suffix and bracketed or minus-signed
negatives. Anything else is returned unparsed. Where a common OCR confusion
(O/0, I/1, S/5, a comma used as the decimal point, a stray space) explains the
text, a suggestion is attached for a reviewer to confirm; it is never used as
the value.
"""

import re
from dataclasses import dataclass
from decimal import Decimal

CURRENCY_PREFIX = re.compile(r"^(?:₹|rs\.?|inr)\s*", re.I)
CURRENCY_SUFFIX = re.compile(r"\s*(?:₹|rs\.?|inr)$", re.I)
DASH_SUFFIX = re.compile(r"\s*/-$")
INDIAN_GROUPS = re.compile(r"^\d{1,2}(?:,\d{2})*,\d{3}$")
WESTERN_GROUPS = re.compile(r"^\d{1,3}(?:,\d{3})+$")
DECIMAL_COMMA = re.compile(r"^\d+,\d{1,2}$")
QUANTITY_UNIT = re.compile(r"^(\d+(?:\.\d+)?)\s*(?:nos?|tabs?|caps?|units?|ml|strips?|vials?|days?|pcs?)\.?$", re.I)
CONFUSABLE = str.maketrans({"O": "0", "o": "0", "D": "0", "I": "1", "l": "1", "|": "1",
                            "S": "5", "s": "5", "B": "8", "Z": "2"})
BLANK = {"", "-", "--", "—", "–", "nil", "na", "n/a", "x", "*"}


@dataclass(frozen=True)
class Parsed:
    value: str | None  # canonical decimal string, e.g. "123456.00"
    issue: str | None = None  # unparsed | ambiguous_separator | ocr_char_confusion | too_many_decimals
    suggestion: str | None = None


def _canonical(text, decimals):
    text = CURRENCY_SUFFIX.sub("", CURRENCY_PREFIX.sub("", DASH_SUFFIX.sub("", text))).strip()
    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative, text = True, text[1:-1].strip()
    if text[:1] in "-−":
        negative, text = True, text[1:].strip()
    elif text[:1] == "+":
        text = text[1:].strip()
    text = CURRENCY_PREFIX.sub("", text)
    whole, dot, fraction = text.partition(".")
    if dot and not fraction.isdigit():
        return None, "unparsed"
    if len(fraction) > decimals:
        return None, "too_many_decimals"
    if "," in whole:
        if INDIAN_GROUPS.match(whole) or WESTERN_GROUPS.match(whole):
            whole = whole.replace(",", "")
        elif not dot and DECIMAL_COMMA.match(whole):
            return None, "ambiguous_separator"
        else:
            return None, "unparsed"
    if not whole.isdigit() or not whole.isascii():
        return None, "unparsed"
    number = Decimal(whole + ("." + fraction if fraction else ""))
    return str(-number if negative else number), None


def parse_number(raw, decimals=2) -> Parsed:
    text = " ".join(str(raw or "").split())
    if text.casefold() in BLANK:
        return Parsed(None)
    value, issue = _canonical(text, decimals)
    if value is not None:
        return Parsed(value)
    if issue == "ambiguous_separator":
        suggestion, _ = _canonical(text.replace(",", "."), decimals)
        return Parsed(None, issue, suggestion)
    for candidate, reason in ((text.replace(" ", ""), "unparsed"),
                              (text.translate(CONFUSABLE), "ocr_char_confusion"),
                              (text.replace(" ", "").translate(CONFUSABLE), "ocr_char_confusion")):
        if candidate != text:
            suggestion, _ = _canonical(candidate, decimals)
            if suggestion is not None:
                return Parsed(None, reason, suggestion)
    return Parsed(None, issue or "unparsed")


def parse_quantity(raw) -> Parsed:
    text = " ".join(str(raw or "").split())
    unit = QUANTITY_UNIT.match(text)
    return parse_number(unit.group(1) if unit else text, decimals=3)


def looks_numeric(raw) -> bool:
    """True when a cell is printed as a number, even if it does not parse."""
    text = " ".join(str(raw or "").split())
    return bool(text) and bool(re.fullmatch(r"[\d₹RrSsIiNn.,/()+\-−\s]+", text)) and any(ch.isdigit() for ch in text)
