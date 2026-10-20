"""Exercise the temporary gateway without downloading the model."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error, request

import pytest

from colab import nanonets_gateway as gateway


@pytest.fixture
def servers(monkeypatch):
    class FakeModel(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert payload["model"] == gateway.MODEL
            body = json.dumps({"choices": [{"message": {"content": "<table>invoice</table>"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    model = ThreadingHTTPServer(("127.0.0.1", 0), FakeModel)
    gateway_server = ThreadingHTTPServer(("127.0.0.1", 0), gateway.Handler)
    monkeypatch.setattr(gateway, "UPSTREAM", f"http://127.0.0.1:{model.server_port}/v1/chat/completions")
    monkeypatch.setenv("NANONETS_API_KEY", "unit-test-secret")
    with gateway._lock:
        gateway._jobs.clear()
    threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in (model, gateway_server)]
    for thread in threads:
        thread.start()
    yield f"http://127.0.0.1:{gateway_server.server_port}/v1"
    for server in (gateway_server, model):
        server.shutdown()
        server.server_close()
    for thread in threads:
        thread.join(timeout=2)


def test_authenticated_gateway_accepts_and_returns_model_result(servers):
    with pytest.raises(error.HTTPError) as unauthorized:
        request.urlopen(servers + "/models")
    assert unauthorized.value.code == 401

    headers = {"Authorization": "Bearer unit-test-secret", "Content-Type": "application/json"}
    payload = {"model": gateway.MODEL, "messages": [{"role": "user", "content": "invoice"}]}
    req = request.Request(servers + "/chat/completions", data=json.dumps(payload).encode(),
                          headers=headers, method="POST")
    with request.urlopen(req) as response:
        assert response.status == 202
        job_id = json.load(response)["id"]
    for _ in range(20):
        poll = request.Request(servers + "/jobs/" + job_id, headers=headers)
        with request.urlopen(poll) as response:
            result = json.load(response)
            if response.status == 200:
                break
        threading.Event().wait(0.05)
    else:
        pytest.fail("Fake model did not return")
    assert result["choices"][0]["message"]["content"] == "<table>invoice</table>"
