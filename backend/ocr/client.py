"""Authenticated Nanonets OCR2 client for the Colab gateway."""

import base64
import json
import time
from urllib import error, request
from urllib.parse import urlsplit
from uuid import UUID


OCR_PROMPT = (
    "Extract all visible text from this medical invoice in reading order. "
    "Keep section headings and row order. Return tables as HTML tables. "
    "Preserve amounts and labels exactly as shown. Do not infer missing text."
)


class OcrError(RuntimeError):
    pass


def validate_connection(url: str, api_key: str = "") -> str:
    """Reject missing or unprotected remote OCR endpoints before uploading pages."""
    parsed = urlsplit(url)
    if not parsed.hostname or parsed.path.rstrip("/") != "/v1" or parsed.query or parsed.fragment or parsed.username:
        raise OcrError("Set NANONETS_BASE_URL to the Colab tunnel HTTPS URL ending in /v1")
    loopback = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme != "https" and not (loopback and parsed.scheme == "http"):
        raise OcrError("Remote Nanonets endpoint must use HTTPS")
    if not loopback and not api_key:
        raise OcrError("Set NANONETS_API_KEY to the key in Colab before starting OCR")
    return url.rstrip("/")


def _read_response(req: request.Request, timeout: float) -> tuple[int, dict]:
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return getattr(response, "status", 200), json.load(response)
    except error.HTTPError as exc:
        raise OcrError(f"Nanonets server returned HTTP {exc.code}") from exc
    except (error.URLError, TimeoutError, ValueError) as exc:
        raise OcrError(f"Nanonets server request failed: {exc}") from exc


def ask(image: bytes, prompt: str, *, url: str, model: str, timeout: int,
        max_tokens: int, api_key: str = "") -> dict:
    """Submit a page and poll the Colab gateway if inference exceeds tunnel limits."""
    base_url = validate_connection(url, api_key)
    if not image.startswith(b"\x89PNG\r\n\x1a\n"):
        raise OcrError("Processed page is not a PNG")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image).decode("ascii")}},
            {"type": "text", "text": prompt},
        ]}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    deadline = time.monotonic() + timeout
    req = request.Request(base_url + "/chat/completions", data=json.dumps(payload).encode("utf-8"),
                          headers=headers, method="POST")
    status, result = _read_response(req, min(90, max(1, deadline - time.monotonic())))
    if status == 202:
        try:
            job_id = str(UUID(result["id"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise OcrError("Nanonets gateway returned an invalid job ID") from exc
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OcrError(f"Nanonets OCR timed out after {timeout} seconds")
            time.sleep(min(3, remaining))
            poll = request.Request(base_url + "/jobs/" + job_id, headers=headers, method="GET")
            status, result = _read_response(poll, min(30, max(1, deadline - time.monotonic())))
            if status != 202:
                break
    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise OcrError("Nanonets server response has no message content") from exc
    if not isinstance(content, str) or not content.strip():
        raise OcrError("Nanonets returned empty OCR text")
    return result


def recognize(image: bytes, *, url: str, model: str, timeout: int,
              max_tokens: int, api_key: str = "") -> dict:
    """Return OCR text and the full server response for audit."""
    return ask(image, OCR_PROMPT, url=url, model=model, timeout=timeout,
               max_tokens=max_tokens, api_key=api_key)
