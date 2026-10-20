"""End-to-end: the demo fleet Q&A asserts the core invariant — answers are
grounded in live state, and spoken commands actually execute."""
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from tesla_fleet_chat.config import Settings, VehicleConfig
from tesla_fleet_chat.server import make_handler
from tesla_fleet_chat.service import FleetChatService
from tesla_fleet_chat.simulator import FakeTeslaClient, demo_vehicles, run_demo


def _settings(**kw):
    return Settings(vehicles=[VehicleConfig("Model Y", "VINY"),
                              VehicleConfig("Model 3", "VIN3")],
                    ai_mode="never", **kw)


def test_demo_qa_and_command_execution():
    results = run_demo(_settings())
    answers = {q: a for q, a, _ in results}
    assert "Model Y" in answers["Which car has more battery?"]
    assert "Charging" in answers["Is the Model 3 charging?"]
    final_log = results[-1][2]
    assert ("charge_start", "VINY") in final_log


def test_http_chat_roundtrip():
    fake = FakeTeslaClient(demo_vehicles())
    service = FleetChatService(_settings(), client_factory=lambda: fake)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        body = json.dumps({"model": "tesla-fleet-chat", "messages": [
            {"role": "user", "content": "Which car has more battery?"}]}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        assert "Model Y" in data["choices"][0]["message"]["content"]

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models",
                                    timeout=10) as resp:
            models = json.loads(resp.read())
        assert models["data"][0]["id"] == "tesla-fleet-chat"

        body = json.dumps({"input": "hello fleet"}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/embeddings",
                                     data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            emb = json.loads(resp.read())
        assert len(emb["data"][0]["embedding"]) == 384
    finally:
        httpd.shutdown()


def test_http_auth_required_when_key_set():
    fake = FakeTeslaClient(demo_vehicles())
    service = FleetChatService(_settings(server_api_key="secret"),
                               client_factory=lambda: fake)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10)
            raise AssertionError("expected 401")
        except urllib.error.HTTPError as e:
            assert e.code == 401
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/models",
            headers={"Authorization": "Bearer secret"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200
    finally:
        httpd.shutdown()
