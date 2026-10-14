from __future__ import annotations

import socket
import subprocess
import sys
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from action_sentinel.api import create_app
from tests.conftest import ROOT, action_dict, make_sentinel

AGENT = {"X-API-Key": "dev-agent-key"}
OPERATOR = {"X-API-Key": "dev-operator-key"}
SENIOR = {"X-API-Key": "dev-senior-key"}
ADMIN = {"X-API-Key": "dev-admin-key"}
AUDITOR = {"X-API-Key": "dev-auditor-key"}


@pytest.fixture
def client():
    s = make_sentinel()
    with TestClient(create_app(s.settings, service=s)) as c:
        c.sentinel = s
        yield c


def evaluate(client, headers=AGENT, **kw):
    return client.post("/decisions/evaluate", json={"action": action_dict(**kw)}, headers=headers)


def test_health_is_public_everything_else_authenticated(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["policy_version"].startswith("gov-1.0.0")
    assert client.get("/providers").status_code == 401
    assert client.get("/providers", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/providers", headers=AUDITOR).status_code == 200


def test_role_checks(client):
    assert evaluate(client, headers=AUDITOR).status_code == 403
    assert client.get("/reviews/pending", headers=AGENT).status_code == 403
    assert client.get("/decisions", headers=AGENT).status_code == 403
    assert client.post("/evaluations/run", headers=OPERATOR, json={}).status_code == 403
    assert evaluate(client, agent_id="someone-else").status_code == 403


def test_evaluate_approve_and_execute(client):
    r = evaluate(client)
    assert r.status_code == 200
    body = r.json()
    assert body["decision"]["status"] == "APPROVE" and body["authorization_token"]
    ex = client.post("/executions", json={"token": body["authorization_token"], "action": action_dict()},
                     headers=AGENT).json()
    assert ex["status"] == "EXECUTED"
    dup = client.post("/executions", json={"token": body["authorization_token"], "action": action_dict()},
                      headers=AGENT).json()
    assert dup["status"] == "DUPLICATE"
    did = body["decision"]["decision_id"]
    assert client.get("/decisions/%s" % did, headers=AGENT).status_code == 200
    events = client.get("/decisions/%s/events" % did, headers=AUDITOR).json()["items"]
    assert events[0]["event_type"] == "proposal.received" and events[-1]["event_type"] == "execution.duplicate"


def test_invalid_payload_and_conflicts(client):
    assert client.post("/decisions/evaluate", json={"action": {"event_id": "x"}}, headers=AGENT).status_code == 422
    assert evaluate(client, proposed_at="2026-10-14T10:00:00").status_code == 422
    assert evaluate(client).status_code == 200
    again = evaluate(client)
    assert again.status_code == 200 and again.json()["duplicate"] and again.json()["authorization_token"] is None
    assert evaluate(client, action_payload={"service": "other"}).status_code == 409


def test_review_flow_over_api(client):
    body = evaluate(client, target_resource="prod/checkout-api").json()
    assert body["decision"]["status"] == "HUMAN_REVIEW"
    rid = body["review_id"]
    pending = client.get("/reviews/pending", headers=OPERATOR).json()["items"]
    assert [p["review_id"] for p in pending] == [rid]
    r = client.post("/reviews/%s/reject" % rid, json={"reason": ""}, headers=OPERATOR)
    assert r.status_code == 400 and r.json()["error"] == "REASON_REQUIRED"
    ok = client.post("/reviews/%s/approve" % rid, json={"reason": "verified"}, headers=OPERATOR)
    assert ok.status_code == 200 and ok.json()["authorization_token"]
    again = client.post("/reviews/%s/approve" % rid, json={}, headers=OPERATOR)
    assert again.status_code == 409
    assert client.post("/reviews/rev_missing/approve", json={}, headers=OPERATOR).status_code == 404


def test_insufficient_reviewer_level_is_403_and_override_works(client):
    body = evaluate(client, source_system="support-agent", action_type="refund_payment",
                    target_resource="payments/p1", action_payload={"payment_id": "p1", "amount": 900}).json()
    rid = body["review_id"]
    assert client.post("/reviews/%s/approve" % rid, json={}, headers=OPERATOR).status_code == 403
    bad = client.post("/reviews/%s/override" % rid, headers=SENIOR,
                      json={"reason": "too large", "changes": {"action_payload": {"payment_id": "p1", "amount": 99999}}})
    assert bad.status_code == 422 and bad.json()["error"] == "OVERRIDE_REVALIDATION_FAILED"
    good = client.post("/reviews/%s/override" % rid, headers=SENIOR,
                       json={"reason": "partial refund", "changes": {"action_payload": {"payment_id": "p1",
                                                                                        "amount": 200}}})
    assert good.status_code == 200 and good.json()["review"]["state"] == "OVERRIDDEN"


def test_blocked_action_and_revocation(client):
    body = evaluate(client, action_type="disable_audit_logging").json()
    assert body["decision"]["status"] == "BLOCK" and body["authorization_token"] is None
    ok = evaluate(client, event_id="e2", idempotency_key="k2").json()
    gid = ok["decision"]["authorization_id"]
    assert client.post("/authorizations/%s/revoke" % gid, json={"reason": "incident"}, headers=OPERATOR).status_code == 403
    assert client.post("/authorizations/%s/revoke" % gid, json={"reason": "incident"}, headers=ADMIN).status_code == 200
    ex = client.post("/executions", json={"token": ok["authorization_token"],
                                          "action": action_dict(event_id="e2", idempotency_key="k2")}, headers=AGENT)
    assert ex.json()["reason_codes"] == ["AUTHORIZATION_REVOKED"]


def test_pagination_and_not_found(client):
    for i in range(3):
        evaluate(client, event_id="p%d" % i, idempotency_key="pk%d" % i)
    page = client.get("/decisions?limit=2&offset=0", headers=AUDITOR).json()
    assert len(page["items"]) == 2 and page["limit"] == 2
    assert client.get("/decisions?limit=0", headers=AUDITOR).status_code == 422
    assert client.get("/decisions/nope", headers=AUDITOR).status_code == 404
    assert client.get("/evaluations/nope", headers=AUDITOR).status_code == 404


def test_evaluation_endpoints(client):
    cfg = {"name": "api-run", "dataset_path": "examples/evaluation_datasets/sentinel_synthetic_v1.jsonl",
           "providers": ["rules"], "limit": 5}
    r = client.post("/evaluations/run", json=cfg, headers=SENIOR)
    assert r.status_code == 200 and r.json()["n_examples"] == 5
    run_id = r.json()["run_id"]
    assert client.get("/evaluations/%s" % run_id, headers=AUDITOR).json()["config"]["name"] == "api-run"
    md = client.get("/evaluations/%s/report" % run_id, headers=AUDITOR).text
    assert "Governance outcomes" in md
    assert client.get("/evaluations", headers=AUDITOR).json()["items"][0]["run_id"] == run_id
    bad = client.post("/evaluations/run", json={**cfg, "providers": ["nope"]}, headers=SENIOR)
    assert bad.status_code == 400


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_real_server_process(tmp_path):
    """Start the real API with uvicorn in a subprocess and talk to it over HTTP."""
    port = _free_port()
    env = {"SENTINEL_DB_PATH": str(tmp_path / "api.db"), "SENTINEL_ENV": "dev", "PYTHONWARNINGS": "ignore"}
    import os
    proc = subprocess.Popen([sys.executable, "-m", "action_sentinel", "serve", "--port", str(port)], cwd=ROOT,
                            env={**os.environ, **env}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        base = "http://127.0.0.1:%d" % port
        for _ in range(100):
            try:
                if httpx.get(base + "/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        else:
            pytest.fail("server did not start")
        r = httpx.post(base + "/decisions/evaluate", json={"action": action_dict()}, headers=AGENT, timeout=30)
        assert r.status_code == 200 and r.json()["decision"]["status"] == "APPROVE"
        assert httpx.get(base + "/providers", timeout=30).status_code == 401
    finally:
        proc.terminate()
        proc.wait(timeout=10)
