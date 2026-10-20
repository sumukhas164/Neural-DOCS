# Copy these cells into your existing Google Colab notebook

Paste each **Python code block** below into its own Colab code cell and run them in order. Do not paste the `.ipynb` JSON: keys such as `execution_count` and values such as `null` are notebook metadata, not Python code.

First select **Runtime → Change runtime type → GPU**. Optionally create `NANONETS_API_KEY` in Colab Secrets. If absent, a key is generated and printed after tunnel setup.

## Cell 1

```python
import os, pathlib, subprocess, sys

# Install uv from the Python package index; astral.sh may return HTTP 403 in Colab.
# --target avoids changing Colab's system Python packages.
uv_package_dir = pathlib.Path('/content/nanonets-uv-package')
if not (uv_package_dir / 'uv').is_dir():
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--target', str(uv_package_dir), 'uv'],
        check=True,
    )
uv_cmd = [sys.executable, '-m', 'uv']
uv_env = {**os.environ, 'PYTHONPATH': str(uv_package_dir)}
# Keep vLLM separate from Colab's Python 3.13 runtime.
venv = pathlib.Path('/content/nanonets-vllm-venv')
venv_python = venv / 'bin/python'
if not venv_python.is_file():
    subprocess.run(uv_cmd + ['venv', str(venv), '--python', '3.12', '--seed', '--managed-python'],
                   env=uv_env, check=True)
version = subprocess.check_output(
    [str(venv_python), '-c', 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'],
    text=True,
).strip()
if version != '3.12':
    raise RuntimeError(f'Expected Python 3.12 in {venv}, found {version}.')
vllm_bin = venv / 'bin/vllm'
if not vllm_bin.is_file():
    subprocess.run(
        uv_cmd + ['pip', 'install', '--python', str(venv_python), 'vllm', '--torch-backend=auto'],
        env=uv_env,
        check=True,
    )
print('vLLM is installed in', venv)

```

## Cell 2

```python
import os, secrets, subprocess
from google.colab import userdata

subprocess.run(['nvidia-smi'], check=True)
generated_key = False
try:
    key = userdata.get('NANONETS_API_KEY')
except Exception as exc:
    if type(exc).__name__ == 'SecretNotFoundError':
        # A rerun in the same Colab runtime must keep the running gateway's key.
        key = os.environ.get('NANONETS_API_KEY') or secrets.token_urlsafe(48)
        generated_key = True
    elif type(exc).__name__ == 'NotebookAccessError':
        raise RuntimeError('Allow this notebook to access NANONETS_API_KEY in Colab Secrets.') from exc
    else:
        raise
if not key or len(key) < 32:
    raise ValueError('NANONETS_API_KEY must contain at least 32 characters.')
os.environ['NANONETS_API_KEY'] = key
print('GPU and API key are ready.')

```

## Cell 3

```python
import json, pathlib, subprocess, time, urllib.request

model_name = 'nanonets/Nanonets-OCR2-3B'
model_log = pathlib.Path('/content/nanonets-vllm.log')
model_cmd = [
    str(vllm_bin), 'serve', model_name, '--served-model-name', model_name,
    '--host', '127.0.0.1', '--port', '8001', '--dtype', 'half',
    '--max-model-len', '12288', '--max-num-seqs', '1',
    '--gpu-memory-utilization', '0.90',
]
def model_ready():
    try:
        with urllib.request.urlopen('http://127.0.0.1:8001/v1/models', timeout=3) as response:
            models = json.load(response)
        return any(item['id'] == model_name for item in models['data'])
    except Exception:
        return False

if not model_ready():
    model_process = subprocess.Popen(model_cmd, stdout=model_log.open('w'), stderr=subprocess.STDOUT)
else:
    model_process = None
for _ in range(180):
    if model_ready():
        break
    if model_process is not None and model_process.poll() is not None:
        raise RuntimeError(f'vLLM exited. Check {model_log}:\n' + model_log.read_text()[-3000:])
    time.sleep(5)
else:
    raise TimeoutError(f'vLLM did not start. Check {model_log}')
print('Nanonets OCR2-3B is ready on the Colab GPU.')

```

## Cell 4

```python
GATEWAY_SOURCE = r'''
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

'''
import pathlib, subprocess, time, urllib.error, urllib.request

gateway_source = GATEWAY_SOURCE
gateway_path = pathlib.Path('/content/nanonets_gateway.py')
gateway_path.write_text(gateway_source, encoding='utf-8')
gateway_log = pathlib.Path('/content/nanonets-gateway.log')
health_url = 'http://127.0.0.1:8002/v1/health'
def gateway_ready():
    req = urllib.request.Request(health_url, headers={'Authorization': 'Bearer ' + key})
    try:
        with urllib.request.urlopen(req, timeout=2) as response:
            return response.status == 200
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise RuntimeError('An existing gateway uses a different key. Restart the Colab runtime, then rerun this cell.') from exc
        raise
    except urllib.error.URLError:
        return False

if not gateway_ready():
    gateway_process = subprocess.Popen(
        [sys.executable, str(gateway_path)],
        env=os.environ.copy(), stdout=gateway_log.open('w'), stderr=subprocess.STDOUT,
    )
else:
    gateway_process = None
for _ in range(20):
    if gateway_ready():
        break
    if gateway_process is not None and gateway_process.poll() is not None:
        raise RuntimeError('Gateway exited: ' + gateway_log.read_text()[-2000:])
    time.sleep(1)
else:
    raise TimeoutError('Gateway did not start: ' + gateway_log.read_text()[-2000:])
print('Authenticated job gateway is ready.')

```

## Cell 5

```python
import os, pathlib, re, subprocess, time, urllib.request, json

binary = pathlib.Path('/content/cloudflared')
if not binary.is_file():
    urllib.request.urlretrieve(
        'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64',
        binary,
    )
    binary.chmod(0o755)
tunnel_log = pathlib.Path('/content/cloudflared.log')
def tunnel_ready(url):
    try:
        req = urllib.request.Request(url + '/models',
                                     headers={'Authorization': 'Bearer ' + key})
        with urllib.request.urlopen(req, timeout=5) as response:
            return any(item['id'] == model_name for item in json.load(response)['data'])
    except Exception:
        return False

old_tunnel = globals().get('tunnel_process')
old_url = globals().get('base_url')
if old_tunnel is not None and old_tunnel.poll() is None and old_url and tunnel_ready(old_url):
    base_url = old_url
else:
    for attempt in range(1, 4):
        tunnel_process = subprocess.Popen(
            [str(binary), 'tunnel', '--protocol', 'http2', '--url', 'http://127.0.0.1:8002'],
            stdout=tunnel_log.open('w'), stderr=subprocess.STDOUT,
        )
        connected = False
        for _ in range(60):
            if tunnel_process.poll() is not None:
                break
            match = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', tunnel_log.read_text())
            if match:
                base_url = match.group(0) + '/v1'
                if tunnel_ready(base_url):
                    connected = True
                    break
            time.sleep(2)
        if connected:
            break
        if tunnel_process.poll() is None:
            tunnel_process.terminate()
            tunnel_process.wait(timeout=10)
        if attempt < 3:
            time.sleep(3)
    else:
        raise RuntimeError('Cloudflare Tunnel failed after three HTTP/2 attempts: ' + tunnel_log.read_text()[-2000:])
print('NANONETS_BASE_URL=' + base_url)
if generated_key:
    print('NANONETS_API_KEY=' + key)
    print('Copy this key to backend/.env, then clear this cell output after saving it.')
else:
    print('Set NANONETS_API_KEY in backend/.env to the same Colab secret.')
print('Restart Django after setting both values.')

```

## Cell 6

```python
import base64, io, json, time, urllib.request
from PIL import Image, ImageDraw

image = Image.new('RGB', (700, 180), 'white')
draw = ImageDraw.Draw(image)
draw.text((20, 20), 'INVOICE', fill='black')
draw.text((20, 70), 'Blood Test  1  50.00  50.00', fill='black')
buffer = io.BytesIO()
image.save(buffer, format='PNG')
payload = {
    'model': model_name,
    'messages': [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()}},
        {'type': 'text', 'text': 'Extract all visible text in reading order.'},
    ]}],
    'max_tokens': 256,
}
headers = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}
req = urllib.request.Request(base_url + '/chat/completions', data=json.dumps(payload).encode(),
                             headers=headers, method='POST')
with urllib.request.urlopen(req, timeout=30) as response:
    job_id = json.load(response)['id']
for _ in range(120):
    req = urllib.request.Request(base_url + '/jobs/' + job_id, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response:
        result = json.load(response)
        if response.status == 200:
            print(result['choices'][0]['message']['content'])
            break
    time.sleep(3)
else:
    raise TimeoutError('Synthetic OCR did not finish; check the Colab vLLM log.')

```

After Cell 5 prints `NANONETS_BASE_URL`, put that URL and the same secret in `backend/.env` on your PC, then restart Django. Cell 6 is an optional synthetic test.
