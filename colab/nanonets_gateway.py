"""Small authenticated job gateway for Colab-hosted Nanonets OCR2-3B.

Only this gateway is exposed by Cloudflare Tunnel. vLLM stays on loopback.
Jobs and invoice images are held in memory for the life of the Colab runtime.
"""

import hmac
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import request
from uuid import UUID, uuid4


MODEL = "nanonets/Nanonets-OCR2-3B"
UPSTREAM = "http://127.0.0.1:8001/v1/chat/completions"
MAX_BODY = 80 * 1024 * 1024
MAX_JOBS = 4
JOB_TTL = 3600
_jobs = {}
_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=1)


def _infer(job_id, payload):
    with _lock:
        _jobs[job_id]["status"] = "running"
    try:
        req = request.Request(UPSTREAM, data=json.dumps(payload).encode("utf-8"),
                              headers={"Content-Type": "application/json"}, method="POST")
        with request.urlopen(req, timeout=1800) as response:
            result = json.load(response)
        if not result.get("choices"):
            raise ValueError("No model response")
        with _lock:
            _jobs[job_id].update(status="completed", result=result, finished=time.monotonic())
    except Exception as exc:
        # Keep upstream details and invoice text out of public error responses.
        print(f"OCR job {job_id} failed: {type(exc).__name__}", flush=True)
        with _lock:
            _jobs[job_id].update(status="failed", finished=time.monotonic())


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Default logging includes request paths; job IDs are enough for diagnostics.
        pass

    def _send(self, status, body):
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self):
        key = os.environ.get("NANONETS_API_KEY", "")
        supplied = self.headers.get("Authorization", "")
        if not key or not hmac.compare_digest(supplied, "Bearer " + key):
            self._send(401, {"detail": "Unauthorized"})
            return False
        return True

    def do_GET(self):
        if not self._authorized():
            return
        if self.path == "/v1/models":
            self._send(200, {"data": [{"id": MODEL}]})
            return
        if self.path == "/v1/health":
            self._send(200, {"status": "ok"})
            return
        if not self.path.startswith("/v1/jobs/"):
            self._send(404, {"detail": "Not found"})
            return
        try:
            job_id = str(UUID(self.path.removeprefix("/v1/jobs/")))
        except ValueError:
            self._send(404, {"detail": "Not found"})
            return
        with _lock:
            job = _jobs.get(job_id)
            if job:
                status = job["status"]
                result = job.get("result")
            else:
                status = None
        if status is None:
            self._send(404, {"detail": "Job expired or runtime restarted"})
        elif status in {"pending", "running"}:
            self._send(202, {"id": job_id, "status": status})
        elif status == "failed":
            self._send(502, {"detail": "Model inference failed; check Colab output"})
        else:
            self._send(200, result)

    def do_POST(self):
        if not self._authorized():
            return
        if self.path != "/v1/chat/completions":
            self._send(404, {"detail": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length < 1 or length > MAX_BODY:
            self._send(413, {"detail": "Invalid page size"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict) or payload.get("model") != MODEL or not isinstance(payload.get("messages"), list):
                raise ValueError("Invalid model request")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            self._send(400, {"detail": "Invalid model request"})
            return
        with _lock:
            now = time.monotonic()
            for old_id, job in list(_jobs.items()):
                if job.get("finished") and now - job["finished"] > JOB_TTL:
                    del _jobs[old_id]
            active = sum(job["status"] in {"pending", "running"} for job in _jobs.values())
            if active >= MAX_JOBS:
                self._send(429, {"detail": "Colab OCR queue is full"})
                return
            job_id = str(uuid4())
            _jobs[job_id] = {"status": "pending"}
        _pool.submit(_infer, job_id, payload)
        self._send(202, {"id": job_id, "status": "pending"})


def main():
    if not os.environ.get("NANONETS_API_KEY"):
        raise RuntimeError("NANONETS_API_KEY must be set")
    server = ThreadingHTTPServer(("127.0.0.1", 8002), Handler)
    print("Nanonets gateway listening on 127.0.0.1:8002", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
