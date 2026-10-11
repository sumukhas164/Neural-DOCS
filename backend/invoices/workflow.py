"""LangGraph stages for OCR-backed medical invoice review."""

import logging
import re
from typing import TypedDict

from django.conf import settings
from langgraph.graph import END, START, StateGraph

from ocr.client import ask
from ocr.models import OcrJob

from .extraction import extract
from .validation import money, validate

logger = logging.getLogger(__name__)
MAX_JUDGE_CALLS = 3


class InvoiceState(TypedDict, total=False):
    ocr_job_id: int
    invoice: dict
    findings: list[dict]
    judge_findings: list[dict]


def reconstruct_structure(state: InvoiceState):
    job = OcrJob.objects.prefetch_related("pages").get(pk=state["ocr_job_id"])
    if job.status != OcrJob.Status.COMPLETED:
        raise ValueError("OCR must complete before invoice extraction")
    return {"invoice": extract((page.page_number, page.text) for page in job.pages.order_by("page_number"))}


def validate_invoice(state: InvoiceState):
    return {"findings": validate(state["invoice"])}


def _question(finding, item, invoice):
    code = finding["code"]
    amount_fields = {"line_gross_mismatch": "gross_amount", "line_net_mismatch": "net_amount",
                     "line_net_from_rate_mismatch": "net_amount",
                     "group_total_mismatch": "net_amount", "grand_total_mismatch": "grand_total"}
    if code in amount_fields:
        field = amount_fields[code]
        expected = invoice.get("grand_total") if field == "grand_total" else (item or {}).get(field)
        label = (item or {}).get("name", "Grand Total")
        return (f"What {field.replace('_', ' ')} is printed for {label} on this medical invoice? "
                "Reply with only the number, or 'unclear' if unreadable.", expected, "amount")
    if code in ("unassigned_group", "cross_page_group") and item:
        groups = {group["id"]: group["name"] for group in invoice["groups"]}
        expected = groups.get(item.get("group_id"))
        return (f"Which billing section contains the row '{item['name']}'? "
                "Answer only with the printed section heading, or 'unclear'.", expected, "group")
    return None, None, None


def _amount_verdict(answer, expected):
    match = re.fullmatch(r"\s*(?:₹|Rs\.?\s*)?([\d,]+(?:\.\d{1,2})?)\s*", answer, flags=re.I)
    if not match or money(expected) is None:
        return {"verdict": "unresolved", "reason": "The model did not give an unambiguous amount", "evidence": answer[:1000]}
    observed = money(match.group(1).replace(",", ""))
    if observed == money(expected):
        return {"verdict": "supported", "reason": "A separate image question returned the extracted amount; arithmetic finding remains", "evidence": answer[:1000]}
    return {"verdict": "unresolved", "reason": f"Image question returned {observed}; extracted value was {expected}", "evidence": answer[:1000]}


def _group_verdict(answer, expected, invoice):
    candidate = answer.strip()
    names = {group["name"].casefold(): group["name"] for group in invoice["groups"]}
    matched = names.get(candidate.casefold())
    if not matched:
        return {"verdict": "unresolved", "reason": "The model did not name a known billing group", "evidence": candidate[:1000]}
    if expected and matched.casefold() == expected.casefold():
        return {"verdict": "supported", "reason": "A targeted image question returned the assigned group", "evidence": matched}
    return {"verdict": "unresolved", "reason": f"Image question suggested {matched}; group assignment still needs review", "evidence": matched}


def judge_discrepancies(state: InvoiceState):
    job = OcrJob.objects.get(pk=state["ocr_job_id"])
    items = {item["id"]: item for item in state["invoice"]["items"]}
    results = []
    for finding in state["findings"][:MAX_JUDGE_CALLS]:
        item = items.get(finding["target_id"])
        page_number = item["page_number"] if item else 1
        page = job.pages.filter(page_number=page_number).select_related("source_page").first()
        verdict = {"verdict": "unresolved", "reason": "Judge unavailable", "evidence": ""}
        raw = {}
        question, expected, answer_kind = _question(finding, item, state["invoice"])
        if page and page.text and question:
            try:
                with page.source_page.processed_image.open("rb") as handle:
                    raw = ask(handle.read(), question, url=settings.NANONETS_BASE_URL,
                              model=job.model_name, timeout=settings.NANONETS_TIMEOUT_SECONDS,
                              max_tokens=min(settings.NANONETS_MAX_TOKENS, 128))
                answer = raw["choices"][0]["message"]["content"]
                verdict = (_amount_verdict(answer, expected) if answer_kind == "amount"
                           else _group_verdict(answer, expected, state["invoice"]))
            except Exception as exc:
                logger.warning("judge failed for %s: %s", finding["target_id"], exc)
                verdict["reason"] = f"Judge failed: {type(exc).__name__}"[:1000]
        elif not question:
            verdict["reason"] = "No reliable targeted question is available for this issue type"
        results.append({"target_id": finding["target_id"], "model_name": job.model_name,
                        "raw_response": raw, **verdict})
    return {"judge_findings": results}


def build_graph():
    graph = StateGraph(InvoiceState)
    graph.add_node("reconstruct_structure", reconstruct_structure)
    graph.add_node("validate_invoice", validate_invoice)
    graph.add_node("judge_discrepancies", judge_discrepancies)
    graph.add_edge(START, "reconstruct_structure")
    graph.add_edge("reconstruct_structure", "validate_invoice")
    graph.add_edge("validate_invoice", "judge_discrepancies")
    graph.add_edge("judge_discrepancies", END)
    return graph.compile()
