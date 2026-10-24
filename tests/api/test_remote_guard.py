"""Remote mode: a Guard in the agent's process using a Sentinel served over HTTP (RemoteSentinel)."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from action_sentinel.api import create_app
from action_sentinel.guard import ActionBlocked, ActionDenied, ActionHeld, Guard
from action_sentinel.remote import RemoteError, RemoteSentinel
from tests.conftest import action_dict, make_sentinel

AGENT = {"X-API-Key": "dev-agent-key"}
ADMIN = {"X-API-Key": "dev-admin-key"}


@pytest.fixture
def server():
    s = make_sentinel()
    with TestClient(create_app(s.settings, service=s)) as c:
        c.sentinel = s
        yield c


def remote(client, key="dev-agent-key") -> RemoteSentinel:
    def transport(method, path, data, headers):
        r = client.request(method, path, content=data, headers=headers)
        return r.status_code, r.content
    return RemoteSentinel("http://sentinel.test", key, transport=transport)


def tools(guard, calls):
    @guard.tool(target=lambda a: "staging/%s" % a["service"])
    def read_logs(service: str, lines: int = 100) -> str:
        calls.append(("read_logs", service))
        return "%d lines of %s" % (lines, service)

    @guard.tool(target=lambda a: "prod/%s" % a["database"],
                evidence=lambda a: [{"ref_id": "t-1", "kind": "ticket", "stance": "supports"},
                                    {"ref_id": "t-2", "kind": "ticket", "stance": "supports"}])
    def delete_database(database: str) -> str:
        calls.append(("delete_database", database))
        return "dropped"

    @guard.tool(target=lambda a: "staging/%s" % a["deployment"],
                evidence=lambda a: [{"ref_id": "alert-1", "kind": "alert", "stance": "supports"}])
    async def scale_deployment(deployment: str, replicas: int) -> str:
        calls.append(("scale_deployment", deployment, replicas))
        return "scaled to %d" % replicas

    return read_logs, delete_database, scale_deployment


def test_remote_enforce_approve_block_and_review(server):
    guard = Guard(remote(server), agent_id="ops-agent", agent_version="1", source_system="infra-agent",
                  mode="enforce")
    calls = []
    read_logs, delete_database, scale_deployment = tools(guard, calls)

    assert read_logs("search", lines=2) == "2 lines of search"
    with pytest.raises(ActionBlocked) as blocked:
        delete_database("orders")
    assert "HARD_POLICY_VIOLATION" in blocked.value.reason_codes

    with pytest.raises(ActionHeld) as held:
        asyncio.run(scale_deployment("web", 4))
    r = server.post("/reviews/%s/approve" % held.value.review_id, json={"reason": "expected load"}, headers=ADMIN)
    token = r.json()["authorization_token"]
    assert asyncio.run(guard.aresume(held.value.review_id, token)) == "scaled to 4"

    assert calls == [("read_logs", "search"), ("scale_deployment", "web", 4)]
    execs = server.sentinel.store.list_executions()
    assert sorted(e.status for e in execs) == ["EXECUTED", "EXECUTED"]
    assert {e.started_by for e in execs} == {"ops-agent"} and {e.adapter for e in execs} == {"guard"}


def test_remote_shadow_records_on_the_server(server):
    guard = Guard(remote(server), agent_id="ops-agent", agent_version="1", source_system="infra-agent")
    calls = []
    read_logs, delete_database, _ = tools(guard, calls)
    assert delete_database("orders") == "dropped" and read_logs("search") == "100 lines of search"
    assert [r.outcome for r in guard.records] == ["BLOCK", "APPROVE"]
    s = server.sentinel
    assert s.store.list_decisions() == [] and s.store.list_executions() == []
    events = s.store.list_events(correlation_id=guard.records[0].correlation_id)
    types = [e.event_type for e in events]
    assert "validation.completed" in types and "shadow.decision" in types  # the assessment and the record


def test_remote_enforce_fails_closed_when_the_server_is_unreachable():
    def down(*a):
        raise RemoteError(0, "UNREACHABLE", "connection refused")
    guard = Guard(RemoteSentinel("http://nowhere", "k", transport=down), agent_id="ops-agent",
                  agent_version="1", source_system="infra-agent", mode="enforce")
    calls = []
    read_logs, *_ = tools(guard, calls)
    with pytest.raises(ActionDenied) as exc:
        read_logs("search")
    assert exc.value.reason_codes == ["SENTINEL_FAILED"] and calls == []
    guard.mode = "shadow"
    assert read_logs("search") == "100 lines of search"  # shadow never breaks the tool
    assert guard.records[-1].guard_error.startswith("RemoteError")


def test_remote_agent_key_must_match_the_agent_id(server):
    guard = Guard(remote(server), agent_id="someone-else", agent_version="1", source_system="infra-agent",
                  mode="enforce")
    calls = []
    read_logs, *_ = tools(guard, calls)
    with pytest.raises(ActionDenied) as exc:
        read_logs("search")
    assert "403" in str(exc.value) and calls == []


def test_begin_and_finish_endpoints(server):
    ev = server.post("/decisions/evaluate", json={"action": action_dict()}, headers=AGENT).json()
    body = {"token": ev["authorization_token"], "action": action_dict()}
    started = server.post("/executions/begin", json=body, headers=AGENT).json()
    assert started["status"] == "STARTED" and started["started_by"] == "ops-agent"
    eid = started["execution_id"]
    assert server.post("/executions/begin", json=body, headers=AGENT).json()["status"] == "DUPLICATE"
    other = {"X-API-Key": "dev-operator-key"}  # a reviewer, not the agent that started it
    assert server.post("/executions/%s/finish" % eid, json={"ok": True}, headers=other).status_code == 403
    done = server.post("/executions/%s/finish" % eid, json={"ok": True, "output": {"rows": 3}}, headers=AGENT)
    assert done.json()["status"] == "EXECUTED" and done.json()["output"] == {"rows": 3}
    again = server.post("/executions/%s/finish" % eid, json={"ok": False}, headers=AGENT)
    assert again.status_code == 409 and again.json()["error"] == "NOT_STARTED"
    assert server.post("/executions/exe_missing/finish", json={"ok": True}, headers=AGENT).status_code == 404
    types = [e.event_type for e in server.sentinel.store.list_events(correlation_id=ev["decision"]["correlation_id"])]
    assert types.count("execution.started") == 1 and "execution.executed" in types


def test_assess_endpoint_stores_no_decision(server):
    r = server.post("/decisions/assess", json={"action": action_dict()}, headers=AGENT)
    assert r.status_code == 200 and r.json()["decision"]["status"] == "APPROVE"
    assert server.sentinel.store.list_decisions() == []
    assert server.post("/decisions/assess", json={"action": action_dict(agent_id="x")}, headers=AGENT).status_code == 403


def test_shadow_records_are_size_capped(server):
    big = {"record": {"tool": "x" * 30_000}, "action_hash": "0" * 64}
    assert server.post("/shadow-records", json=big, headers=AGENT).status_code == 413
