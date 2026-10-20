"""Regenerate the self-contained Colab notebook from the gateway source."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GATEWAY = (ROOT / "nanonets_gateway.py").read_text(encoding="utf-8")


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


cells = [
    cell("markdown", """# Nanonets OCR2-3B on Colab for Neural-DOCS

**Open this `.ipynb` with Colab's File → Upload notebook. Do not paste the JSON text into a Python cell.** If you want to paste one code block into your existing notebook, use `PASTE_IN_ONE_CELL.py` from this folder instead.

1. Select **Runtime → Change runtime type → GPU**. GPU availability and memory vary; the notebook will report a model-load error if the assigned GPU cannot run it.
2. Optional: In Colab **Secrets** (key icon), create `NANONETS_API_KEY` with a random value of at least 32 characters and allow this notebook to access it. If absent, the setup generates a new key and prints it after the tunnel is ready.
3. Run the cells in order. Copy the printed `NANONETS_BASE_URL` to `backend/.env` on your PC, and put the **same secret** in `NANONETS_API_KEY` there. Restart Django.

The public tunnel reaches only an authenticated gateway. vLLM stays bound to loopback. Colab sessions and Quick Tunnel URLs are temporary; run the notebook and update the URL after each restart. Do not put invoice images or the key in notebook source. If the notebook prints a generated key, copy it to `backend/.env` and clear the cell output before sharing the notebook.
"""),
    cell("code", """import os, pathlib, subprocess, sys

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
"""),
    cell("code", """import os, secrets, subprocess
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
"""),
    cell("code", """import json, pathlib, subprocess, time, urllib.request

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
        raise RuntimeError(f'vLLM exited. Check {model_log}:\\n' + model_log.read_text()[-3000:])
    time.sleep(5)
else:
    raise TimeoutError(f'vLLM did not start. Check {model_log}')
print('Nanonets OCR2-3B is ready on the Colab GPU.')
"""),
    cell("code", """import pathlib, subprocess, time, urllib.error, urllib.request

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
"""),
    cell("code", """import os, pathlib, re, subprocess, time, urllib.request, json

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
            match = re.search(r'https://[a-z0-9-]+\\.trycloudflare\\.com', tunnel_log.read_text())
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
"""),
    cell("markdown", """## Optional end-to-end check

This sends a tiny generated invoice image through the same tunnel and polls for the OCR result. It sends no real patient data.
"""),
    cell("code", """import base64, io, json, time, urllib.request
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
"""),
]

# Embed readable gateway source so each cell can be copied into another notebook.
gateway_cell = cells[4]
if "'''" in GATEWAY:
    raise ValueError("Gateway source cannot contain triple single quotes")
gateway_cell["source"] = ("GATEWAY_SOURCE = r'''\n" + GATEWAY + "\n'''\n" + "".join(gateway_cell["source"])).splitlines(keepends=True)

notebook = {
    "cells": cells,
    "metadata": {"colab": {"name": "nanonets_ocr2_3b.ipynb"},
                 "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
(ROOT / "nanonets_ocr2_3b.ipynb").write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")

copy_paste = [
    "# Copy these cells into your existing Google Colab notebook\n\n",
    "Paste each **Python code block** below into its own Colab code cell and run them in order. "
    "Do not paste the `.ipynb` JSON: keys such as `execution_count` and values such as `null` are notebook metadata, not Python code.\n\n",
    "First select **Runtime → Change runtime type → GPU**. Optionally create "
    "`NANONETS_API_KEY` in Colab Secrets. If absent, a key is generated and printed after tunnel setup.\n\n",
]
for number, notebook_cell in enumerate((item for item in cells if item["cell_type"] == "code"), 1):
    copy_paste.extend((f"## Cell {number}\n\n", "```python\n", "".join(notebook_cell["source"]),
                       "\n```\n\n"))
copy_paste.append(
    "After Cell 5 prints `NANONETS_BASE_URL`, put that URL and the same secret in "
    "`backend/.env` on your PC, then restart Django. Cell 6 is an optional synthetic test.\n"
)
(ROOT / "COPY_PASTE.md").write_text("".join(copy_paste), encoding="utf-8")

single_cell = [
    "# Paste this entire Python file into ONE Google Colab code cell and run it.\n",
    "# First select a GPU runtime. A NANONETS_API_KEY secret is optional.\n",
    "# This file contains Python only; do not paste the .ipynb JSON into a code cell.\n\n",
]
for number, notebook_cell in enumerate((item for item in cells if item["cell_type"] == "code"), 1):
    if number == 6:  # Synthetic smoke test is optional and remains a separate cell.
        break
    single_cell.extend((f"# Step {number}\n", "".join(notebook_cell["source"]), "\n"))
(ROOT / "PASTE_IN_ONE_CELL.py").write_text("".join(single_cell).rstrip("\n") + "\n", encoding="utf-8")
