"""LangGraph stages for OCR-backed medical invoice review.

reconstruct_structure -> check_completeness -> [targeted_ocr_retry ->
check_completeness] -> validate_invoice -> judge_discrepancies ->
score_confidence. The retry loop re-reads only pages the completeness check
questions, at most INVOICE_RETRY_MAX_PAGES pages and INVOICE_RETRY_MAX_CALLS
OCR calls per run; each page is retried once.
"""

import json
import logging
import re
from typing import TypedDict

from django.conf import settings
from langgraph.graph import END, START, StateGraph

from ocr.client import ask
from ocr.models import OcrJob

from .completeness import CATEGORY, check
from .confidence import score
from .extraction import extract
from .retry import read_page
from .validation import money, validate

logger = logging.getLogger(__name__)
MAX_JUDGE_CALLS = 3


class InvoiceState(TypedDict, total=False):
    ocr_job_id: int
    pages: list[dict]
    expected_pages: int
    invoice: dict
    completeness_findings: list[dict]
    retry_requests: list[dict]
    retried_pages: list[int]
    retry_attempts: list[dict]
    crop_results: dict
    findings: list[dict]
    validation_summary: dict
    judge_findings: list[dict]
    confidence: dict


def _ocr_retry_info(page):
    if not page.raw_response:
        return None
    try:
        with page.raw_response.open("rb") as handle:
            return json.load(handle).get("_ocr_retry")
    except (OSError, ValueError) as exc:
        logger.warning("could not read OCR artifact for page %s: %s", page.page_number, exc)
        return None


def reconstruct_structure(state: InvoiceState):
    job = OcrJob.objects.select_related("preprocessing_job").get(pk=state["ocr_job_id"])
    if job.status != OcrJob.Status.COMPLETED:
        raise ValueError("OCR must complete before invoice extraction")
    ocr_pages = list(job.pages.select_related("source_page").order_by("page_number"))
    pages = []
    for page in ocr_pages:
        retry = _ocr_retry_info(page)
        pages.append({"page_number": page.page_number, "text_chars": len(page.text.strip()),
                      "finish_reason": page.finish_reason, "retried": bool(retry),
                      "all_truncated": bool(retry and retry.get("all_truncated")),
                      "blank": "blank_page" in (page.source_page.quality_flags or [])})
    return {"invoice": extract((page.page_number, page.text) for page in ocr_pages),
            "pages": pages, "expected_pages": job.preprocessing_job.pages_total,
            "retried_pages": [], "retry_attempts": [], "crop_results": {}}


def check_completeness(state: InvoiceState):
    crops = {int(number): result for number, result in state.get("crop_results", {}).items()}
    findings, requests, candidates = check(state["invoice"], state["pages"], state["expected_pages"], crops)
    retried = set(state.get("retried_pages", []))
    pending = [request for request in requests if request["page_number"] not in retried]
    budget = (len(retried) < settings.INVOICE_RETRY_MAX_PAGES
              and len(state.get("retry_attempts", [])) < settings.INVOICE_RETRY_MAX_CALLS)
    available = bool(settings.NANONETS_BASE_URL)
    if pending and not (budget and available):
        reason = ("the targeted OCR limit for this run was reached" if available
                  else "the OCR gateway is not configured")
        for request in pending:
            findings.append({"code": "targeted_ocr_skipped", "severity": "review", "category": CATEGORY,
                             "target_id": f"page-{request['page_number']}",
                             "message": f"Page {request['page_number']} was not re-read because {reason}",
                             "evidence": ", ".join(request["reasons"])})
        pending = []
    logger.info("Completeness check: %s finding(s), %s page(s) queued for targeted OCR",
                len(findings), len(pending))
    return {"invoice": {**state["invoice"], "candidate_rows": candidates},
            "completeness_findings": findings, "retry_requests": pending}


def after_completeness(state: InvoiceState):
    return "targeted_ocr_retry" if state.get("retry_requests") else "validate_invoice"


def targeted_ocr_retry(state: InvoiceState):
    job = OcrJob.objects.get(pk=state["ocr_job_id"])
    retried = list(state.get("retried_pages", []))
    attempts = list(state.get("retry_attempts", []))
    crops = dict(state.get("crop_results", {}))
    for request in state["retry_requests"][:max(settings.INVOICE_RETRY_MAX_PAGES - len(retried), 0)]:
        calls_left = settings.INVOICE_RETRY_MAX_CALLS - len(attempts)
        if calls_left <= 0:
            break
        number = request["page_number"]
        logger.info("Targeted OCR of page %s because: %s", number, ", ".join(request["reasons"]))
        retried.append(number)
        page = job.pages.select_related("source_page").filter(page_number=number).first()
        if page is None:
            crops[str(number)] = {"coverage": "page", "items": [], "errors": ["No OCR page record exists"]}
            continue
        try:
            result, page_attempts = read_page(page, job.model_name, request["reasons"], calls_left)
        except Exception as exc:  # recorded as a finding; the run continues to human review
            logger.warning("Targeted OCR could not start for page %s: %s", number, exc)
            result, page_attempts = {"coverage": "page", "items": [], "errors": [f"{type(exc).__name__}: {exc}"]}, []
        crops[str(number)] = result
        attempts.extend(page_attempts)
    return {"retried_pages": retried, "retry_attempts": attempts, "crop_results": crops, "retry_requests": []}


def validate_invoice(state: InvoiceState):
    summary = {}
    findings = validate(state["invoice"], summary)
    return {"findings": findings, "validation_summary": summary}


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
                              model=job.model_name, api_key=settings.NANONETS_API_KEY,
                              timeout=settings.NANONETS_TIMEOUT_SECONDS,
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


def score_confidence(state: InvoiceState):
    retried = [page["page_number"] for page in state["pages"] if page["retried"]]
    findings = state["findings"] + state.get("completeness_findings", [])
    return {"confidence": score(state["invoice"], findings, retried)}


def build_graph():
    graph = StateGraph(InvoiceState)
    graph.add_node("reconstruct_structure", reconstruct_structure)
    graph.add_node("check_completeness", check_completeness)
    graph.add_node("targeted_ocr_retry", targeted_ocr_retry)
    graph.add_node("validate_invoice", validate_invoice)
    graph.add_node("judge_discrepancies", judge_discrepancies)
    graph.add_node("score_confidence", score_confidence)
    graph.add_edge(START, "reconstruct_structure")
    graph.add_edge("reconstruct_structure", "check_completeness")
    graph.add_conditional_edges("check_completeness", after_completeness,
                                ["targeted_ocr_retry", "validate_invoice"])
    graph.add_edge("targeted_ocr_retry", "check_completeness")
    graph.add_edge("validate_invoice", "judge_discrepancies")
    graph.add_edge("judge_discrepancies", "score_confidence")
    graph.add_edge("score_confidence", END)
    return graph.compile()
