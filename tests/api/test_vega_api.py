"""Vega decision service and HTTP API, driven by the test double (no inference; results say live=false)."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from action_sentinel.api import create_app
from action_sentinel.vegalab.service import VegaDecisionService
from action_sentinel.vegalab.testing import FakeVegaRunner
from tests.conftest import make_sentinel

AGENT = {"X-API-Key": "dev-agent-key"}
AUDITOR = {"X-API-Key": "dev-auditor-key"}
SENIOR = {"X-API-Key": "dev-senior-key"}  # has the evaluator role
OPERATOR = {"X-API-Key": "dev-operator-key"}

REQUEST = {"state": "Invoice overdue, third notice", "questions": [
    {"question_id": "team", "type": "choice", "instructions": "Which team?",
     "options": [{"label": "billing", "description": "payments"}, {"label": "technical", "description": "faults"}]}]}


def client_for(runner, **kw):
    s = make_sentinel()
    svc = VegaDecisionService(runner, **kw)
    return TestClient(create_app(s.settings, service=s, vega_service=svc)), svc


def test_service_lifecycle_and_accounting():
    svc = VegaDecisionService(FakeVegaRunner(), lazy_load=False)
    with pytest.raises(Exception) as e:
        svc.decide(REQUEST)
    assert e.value.error.code == "MODEL_NOT_READY"
    assert svc.load() == "ready" and svc.ready()["ready"] is True
    res = svc.decide(REQUEST)
    assert res.decisions["team"].selected_label == "billing" and res.live_inference is False
    info = svc.info()
    assert info["requests"] == 2 and info["errors"] == {"MODEL_NOT_READY": 1} and info["latency_ms"]["n"] == 1


def test_load_failure_is_reported():
    svc = VegaDecisionService(FakeVegaRunner(fail_load="weights not found"))
    with pytest.raises(Exception) as e:
        svc.decide(REQUEST)  # lazy load attempted, fails, typed error
    assert e.value.error.code == "MODEL_LOAD_FAILED" and svc.state == "failed"
    assert svc.ready()["error"]["message"] == "weights not found"


def test_runner_loaded_once():
    runner = FakeVegaRunner()
    loads = []
    orig = runner.load
    runner.load = lambda: (loads.append(1), orig())[1]
    svc = VegaDecisionService(runner)
    for _ in range(3):
        svc.decide(REQUEST)
    assert loads == [1]


def test_api_endpoints():
    c, svc = client_for(FakeVegaRunner())
    assert c.get("/vega/health").json() == {"status": "ok", "model_state": "not_loaded"}
    assert c.get("/vega/ready").status_code == 503
    assert c.get("/vega/model").status_code == 401
    assert c.post("/vega/decide", json={"request": REQUEST}, headers=AUDITOR).status_code == 403
    r = c.post("/vega/decide", json={"request": REQUEST, "adapter_mode": "off"}, headers={**AGENT, "X-Request-ID": "r-1"})
    assert r.status_code == 200 and r.headers["X-Request-ID"] == "r-1"
    body = r.json()["result"]
    assert body["request_id"] == "r-1" and body["live_inference"] is False and body["adapter_mode"] == "off"
    assert body["native"]["answers"]["team"]["choice"] == "billing" and r.json()["server_ms"] >= 0
    assert c.get("/vega/ready").status_code == 200
    assert c.get("/vega/model", headers=AUDITOR).json()["requests"] == 1
    assert c.post("/vega/decide", json={"request": REQUEST}, headers=SENIOR).status_code == 200


@pytest.mark.parametrize("req,status,code", [
    ({"state": "x", "questions": []}, 422, "INVALID_INPUT"),
    ({"state": "x"}, 422, "INVALID_INPUT"),
    ({**REQUEST, "views": 9}, 422, "INVALID_INPUT"),
])
def test_api_validation_errors(req, status, code):
    c, _ = client_for(FakeVegaRunner())
    r = c.post("/vega/decide", json={"request": req}, headers=AGENT)
    assert r.status_code == status and r.json()["error"]["code"] == code


def test_api_inference_errors():
    c, _ = client_for(FakeVegaRunner(raise_on_decide=RuntimeError("cuda died")))
    r = c.post("/vega/decide", json={"request": REQUEST}, headers=AGENT)
    assert r.status_code == 500 and r.json()["error"]["code"] == "INFERENCE_FAILED"

    def slow(state, q, mode):
        time.sleep(1)
        return FakeVegaRunner.default_response(state, q, mode)
    c2, _ = client_for(FakeVegaRunner(slow), timeout_s=0.1)
    r = c2.post("/vega/decide", json={"request": REQUEST}, headers=AGENT)
    assert r.status_code == 504 and r.json()["error"]["retryable"] is True
    c3, _ = client_for(FakeVegaRunner(fail_load="no weights"))
    r = c3.post("/vega/decide", json={"request": REQUEST}, headers=AGENT)
    assert r.status_code == 503 and r.json()["error"]["code"] == "MODEL_LOAD_FAILED"
    c4, _ = client_for(FakeVegaRunner(lambda s, q, m: {"answers": {"team": {"choice": "legal"}}}))
    assert c4.post("/vega/decide", json={"request": REQUEST}, headers=AGENT).status_code == 502


def test_vega_routes_absent_unless_enabled():
    s = make_sentinel()
    c = TestClient(create_app(s.settings, service=s))
    assert c.get("/vega/health").status_code == 404
