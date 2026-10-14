from __future__ import annotations

import json

import pytest

from action_sentinel.audit import REDACTED, export_jsonl, sanitize
from action_sentinel.contracts import Outcome
from action_sentinel.service import DuplicateEventConflict
from action_sentinel.storage import ConflictError, SQLiteStore
from tests.conftest import make_action, make_sentinel


def test_full_lifecycle_reconstructable_from_audit():
    s = make_sentinel()
    ev = s.evaluate(make_action(target_resource="prod/checkout-api"))
    appr = s.approve_review(ev.review.review_id, "olivia", "verified")
    s.execute(appr.authorization.token, make_action(target_resource="prod/checkout-api"))
    events = s.store.list_events(correlation_id=ev.decision.correlation_id)
    types = [e.event_type for e in events]
    assert types == ["proposal.received", "validation.completed", "provider.requested", "provider.result",
                     "uncertainty.assessed", "review.created", "governance.decision", "review.approved",
                     "authorization.issued", "execution.started", "execution.executed"]
    seqs = [e.seq for e in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    assert all(e.policy_version == s.policy_version and e.component_version for e in events)
    assert events[7].actor == "olivia"


def test_persistence_survives_restart(tmp_path):
    db = tmp_path / "s.db"
    s = make_sentinel()
    s.store = SQLiteStore(db)
    s.audit.store = s.store
    s.authorizer.store = s.store
    s.gateway.store = s.store
    d = s.evaluate(make_action()).decision
    s.store.close()
    again = SQLiteStore(db)
    assert again.get_decision(d.decision_id).status == Outcome.APPROVE
    assert again.find_decision_by_event("evt-1").decision_id == d.decision_id
    assert again.count_decisions_by_status() == {"APPROVE": 1}
    assert again.list_events(decision_id=d.decision_id)


def test_duplicate_requests():
    s = make_sentinel()
    first = s.evaluate(make_action())
    again = s.evaluate(make_action())
    assert again.duplicate and again.decision.decision_id == first.decision.decision_id
    assert again.authorization is None  # tokens are only handed out once
    with pytest.raises(DuplicateEventConflict):
        s.evaluate(make_action(action_payload={"service": "other"}))
    assert {e.event_type for e in s.store.list_events()} >= {"proposal.duplicate", "proposal.conflict"}
    with pytest.raises(ConflictError):
        s.store.save_decision(first.decision)


def test_sensitive_values_redacted_in_audit_and_export(tmp_path):
    s = make_sentinel()
    s.evaluate(make_action(action_payload={"service": "a", "password": "hunter2"},
                           context={"api_key": "sk-123", "note": "contact jane@example.com"}))
    blob = json.dumps([e.model_dump(mode="json") for e in s.store.list_events()])
    assert "hunter2" not in blob and "sk-123" not in blob
    out = tmp_path / "audit.jsonl"
    n = export_jsonl(s.store.list_events(), out)
    text = out.read_text()
    assert n == len(text.splitlines()) and "jane@example.com" not in text and "[EMAIL]" in text


def test_sanitize_bounds():
    assert sanitize({"Authorization": "Bearer x"}) == {"Authorization": REDACTED}
    assert sanitize("x" * 5000).endswith("[TRUNCATED]")
    assert len(sanitize(list(range(500)))) == 201
    deep = {}
    cur = deep
    for _ in range(20):
        cur["a"] = {}
        cur = cur["a"]
    assert "TRUNCATED" in json.dumps(sanitize(deep))


def test_review_compare_and_set():
    s = make_sentinel()
    r = s.evaluate(make_action(target_resource="prod/checkout-api")).review
    stale = r.model_copy()
    s.reject_review(r.review_id, "olivia", "not now please")
    with pytest.raises(ConflictError):
        s.store.update_review(stale, stale.state)


def test_pagination_and_filters():
    s = make_sentinel()
    for i in range(5):
        s.evaluate(make_action(event_id="e%d" % i, idempotency_key="k%d" % i))
    assert len(s.store.list_decisions(limit=2)) == 2
    assert len(s.store.list_decisions(limit=10, offset=3)) == 2
    assert len(s.store.list_decisions(status="APPROVE")) == 5
    assert s.store.list_decisions(status="BLOCK") == []
