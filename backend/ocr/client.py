"""Nanonets OCR2 client for a local OpenAI-compatible vision server."""

import base64
import json
from urllib import error, request


OCR_PROMPT = (
    "Extract all visible text from this medical invoice in reading order. "
    "Keep section headings and row order. Return tables as HTML tables. "
    "Preserve amounts and labels exactly as shown. Do not infer missing text."
)


class OcrError(RuntimeError):
    pass


def ask(image: bytes, prompt: str, *, url: str, model: str, timeout: int, max_tokens: int) -> dict:
    """Send an image question to the local Nanonets vision model."""
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
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url.rstrip("/") + "/chat/completions", data=data,
                          headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            result = json.load(response)
    except (error.URLError, TimeoutError, ValueError) as exc:
        raise OcrError(f"Nanonets server request failed: {exc}") from exc
    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise OcrError("Nanonets server response has no message content") from exc
    if not isinstance(content, str) or not content.strip():
        raise OcrError("Nanonets returned empty OCR text")
    return result


def recognize(image: bytes, *, url: str, model: str, timeout: int, max_tokens: int) -> dict:
    """Return OCR text and the full server response for audit."""
    return ask(image, OCR_PROMPT, url=url, model=model, timeout=timeout, max_tokens=max_tokens)
