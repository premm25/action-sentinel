"""The @guard decorator: shadow mode never changes behaviour; enforce mode runs tools only through the gateway."""

from __future__ import annotations

import json

import pytest

from action_sentinel.contracts import Outcome
from action_sentinel.guard import ActionBlocked, ActionDenied, ActionHeld, Guard, GuardAdapter
from tests.conftest import make_sentinel


def build(mode="enforce", **kw):
    sentinel = make_sentinel(adapter=GuardAdapter())
    guard = Guard(sentinel, agent_id="ops-agent", agent_version="1.0.0", source_system="infra-agent", mode=mode, **kw)
    calls = []

    @guard.tool(target=lambda a: "staging/%s" % a["service"])
    def read_logs(service: str, lines: int = 100) -> str:
        calls.append(("read_logs", service, lines))
        return "%d lines of %s" % (lines, service)

    @guard.tool(target=lambda a: "prod/%s" % a["database"],
                evidence=lambda a: [{"ref_id": "t-1", "kind": "ticket", "stance": "supports"},
                                    {"ref_id": "t-2", "kind": "ticket", "stance": "supports"}])
    def delete_database(database: str) -> str:
        calls.append(("delete_database", database))
        return "dropped"

    @guard.tool(target=lambda a: "staging/%s" % a["deployment"],
                evidence=lambda a: [{"ref_id": "alert-1", "kind": "alert", "stance": "supports"}])
    def scale_deployment(deployment: str, replicas: int) -> str:
        calls.append(("scale_deployment", deployment, replicas))
        return "scaled to %d" % replicas

    return sentinel, guard, calls, read_logs, delete_database, scale_deployment


def test_enforce_approved_call_runs_once_through_the_gateway():
    sentinel, guard, calls, read_logs, *_ = build()
    assert read_logs("checkout-api", lines=5) == "5 lines of checkout-api"
    assert calls == [("read_logs", "checkout-api", 5)]
    [d] = sentinel.store.list_decisions()
    assert d.status == Outcome.APPROVE
    assert d.action.action_payload == {"service": "checkout-api", "lines": 5}  # defaults and kwargs bound
    [e] = sentinel.store.list_executions()
    assert e.status == "EXECUTED" and e.adapter == "guard"


def test_enforce_blocked_call_never_runs():
    sentinel, guard, calls, _, delete_database, _ = build()
    with pytest.raises(ActionBlocked) as exc:
        delete_database("orders")
    assert "HARD_POLICY_VIOLATION" in exc.value.reason_codes
    assert calls == [] and sentinel.store.list_executions() == []


def test_enforce_review_holds_the_call_until_a_reviewer_approves():
    sentinel, guard, calls, *_, scale_deployment = build()
    with pytest.raises(ActionHeld) as exc:
        scale_deployment("web", 4)
    assert calls == [] and guard.held() == [exc.value.review_id]
    approved = sentinel.approve_review(exc.value.review_id, "ada", "expected load")
    assert guard.resume(exc.value.review_id, approved.authorization.token) == "scaled to 4"
    assert calls == [("scale_deployment", "web", 4)]
    with pytest.raises(KeyError):
        guard.resume(exc.value.review_id, approved.authorization.token)


def test_enforce_grant_for_one_call_cannot_run_another():
    sentinel, guard, calls, *_, scale_deployment = build()
    with pytest.raises(ActionHeld) as first:
        scale_deployment("web", 4)
    with pytest.raises(ActionHeld) as second:
        scale_deployment("web", 400)
    token = sentinel.approve_review(first.value.review_id, "ada", "ok").authorization.token
    with pytest.raises(ActionBlocked) as exc:
        guard.resume(second.value.review_id, token)
    assert "ACTION_MISMATCH" in exc.value.reason_codes and calls == []


def test_enforce_tool_errors_reach_the_caller_and_are_recorded():
    sentinel = make_sentinel(adapter=GuardAdapter())
    guard = Guard(sentinel, agent_id="ops-agent", agent_version="1", source_system="infra-agent", mode="enforce")

    @guard.tool("read_logs", target="staging/x")
    def broken(service: str):
        raise ConnectionError("log store down")

    with pytest.raises(ConnectionError, match="log store down"):
        broken("x")
    [e] = sentinel.store.list_executions()
    assert e.status == "FAILED"


def test_enforce_fails_closed_when_sentinel_fails(monkeypatch):
    sentinel, guard, calls, read_logs, *_ = build()
    monkeypatch.setattr(sentinel, "evaluate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db gone")))
    with pytest.raises(ActionDenied) as exc:
        read_logs("checkout-api")
    assert exc.value.reason_codes == ["SENTINEL_FAILED"] and calls == []


def test_enforce_needs_the_guard_adapter():
    with pytest.raises(ValueError, match="GuardAdapter"):
        Guard(make_sentinel(), agent_id="a", agent_version="1", source_system="infra-agent", mode="enforce")
    g = Guard(make_sentinel(), agent_id="a", agent_version="1", source_system="infra-agent")  # shadow is fine
    with pytest.raises(ValueError):
        g.mode = "enforce"


def test_shadow_runs_everything_and_records_what_would_have_happened(tmp_path):
    log = tmp_path / "shadow.jsonl"
    sentinel, guard, calls, read_logs, delete_database, _ = build(mode="shadow", shadow_log=log)
    assert delete_database("orders") == "dropped"
    assert read_logs("search") == "100 lines of search"
    assert [c[0] for c in calls] == ["delete_database", "read_logs"]
    blocked, approved = guard.records
    assert (blocked.outcome, blocked.would_run) == ("BLOCK", False)
    assert (approved.outcome, approved.would_run) == ("APPROVE", True)
    # nothing is decided, queued for review or authorized in shadow mode
    assert sentinel.store.list_decisions() == [] and sentinel.store.list_executions() == []
    events = sentinel.store.list_events(correlation_id=blocked.correlation_id)
    assert "shadow.decision" in [e.event_type for e in events]
    lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert [x["outcome"] for x in lines] == ["BLOCK", "APPROVE"]


def test_shadow_survives_sentinel_failure_and_keeps_tool_errors(monkeypatch):
    sentinel, guard, calls, read_logs, *_ = build(mode="shadow")
    monkeypatch.setattr(sentinel, "assess", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("provider down")))
    assert read_logs("search") == "100 lines of search"
    assert guard.records[-1].guard_error.startswith("RuntimeError") and guard.records[-1].outcome is None

    @guard.tool("read_logs", target="staging/x")
    def broken(service: str):
        raise ValueError("bad")

    with pytest.raises(ValueError):
        broken("x")
    assert guard.records[-1].tool_error == "ValueError: bad"


def test_async_tools_are_rejected():
    guard = Guard(make_sentinel(), agent_id="a", agent_version="1", source_system="infra-agent")
    with pytest.raises(TypeError, match="synchronous"):
        @guard.tool("read_logs", target="x")
        async def tool(service: str):
            return service
