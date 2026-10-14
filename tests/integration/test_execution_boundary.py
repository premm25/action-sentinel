"""Only authorized, current, matching, unexecuted actions reach the executor."""

from __future__ import annotations

import copy
import threading

import pytest
import yaml

from action_sentinel.contracts import Outcome
from action_sentinel.execution import Authorizer
from action_sentinel.validation import check_pack_data
from tests.conftest import CORE_PACK, Clock, make_action, make_sentinel


def approved(s, **kw):
    out = s.evaluate(make_action(**kw))
    assert out.decision.status == Outcome.APPROVE
    return out


def test_authorized_action_executes_once():
    s = make_sentinel()
    out = approved(s)
    r = s.execute(out.authorization.token, make_action())
    assert r.status == "EXECUTED" and r.output["simulated"] is True
    assert len(s.adapter.executed) == 1
    dup = s.execute(out.authorization.token, make_action())
    assert dup.status == "DUPLICATE" and dup.original_execution_id == r.execution_id
    assert len(s.adapter.executed) == 1


def test_concurrent_duplicate_execution_runs_once():
    s = make_sentinel()
    out = approved(s)
    results = []
    threads = [threading.Thread(target=lambda: results.append(s.execute(out.authorization.token, make_action())))
               for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(r.status for r in results).count("EXECUTED") == 1
    assert len(s.adapter.executed) == 1


def test_regression_concurrent_execution_stress():
    """Regression: a shared SQLite connection read outside its lock once let two threads both execute.

    Found during implementation (about 1 in 8 runs of the test above failed). Repeated here to make a
    recurrence very likely to show up.
    """
    for trial in range(25):
        s = make_sentinel()
        out = approved(s, event_id="s%d" % trial, idempotency_key="sk%d" % trial)
        a = make_action(event_id="s%d" % trial, idempotency_key="sk%d" % trial)
        results = []
        def run(s=s, token=out.authorization.token, a=a, results=results):
            results.append(s.execute(token, a))

        threads = [threading.Thread(target=run) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert [r.status for r in results].count("EXECUTED") == 1, trial
        assert len(s.adapter.executed) == 1, trial
        assert all(r.status in ("EXECUTED", "DUPLICATE") for r in results), [r.reason_codes for r in results]


def _token_for_decision(s, decision):
    """Forge the best token an attacker holding the signing key could make for a non-approved decision."""
    issued = s.authorizer.issue(decision_id=decision.decision_id, correlation_id=decision.correlation_id,
                                action=decision.action, policy_version=s.policy_version, issued_by="attacker")
    return issued.token


@pytest.mark.parametrize("kw,providers,script,status", [
    (dict(action_type="disable_audit_logging"), None, None, Outcome.BLOCK),
    ({}, ["mock-a"], {"mock-a": {"evt-1": {"abstain": True}}}, Outcome.ABSTAIN),
    ({}, ["mock-a"], {"mock-a": {"evt-1": {"fail": "x"}}}, Outcome.ERROR),
    (dict(target_resource="prod/checkout-api"), None, None, Outcome.HUMAN_REVIEW),
])
def test_non_approved_decisions_cannot_execute(kw, providers, script, status):
    s = make_sentinel(script)
    d = s.evaluate(make_action(**kw), providers=providers).decision
    assert d.status == status
    r = s.execute(_token_for_decision(s, d), make_action(**kw))
    assert r.status == "REJECTED" and r.reason_codes == ["DECISION_NOT_EXECUTABLE"]
    assert s.adapter.executed == []


def test_pending_review_cannot_execute_but_approved_review_can():
    s = make_sentinel()
    out = s.evaluate(make_action(target_resource="prod/checkout-api"))
    a = make_action(target_resource="prod/checkout-api")
    appr = s.approve_review(out.review.review_id, "olivia", "ok")
    assert s.execute(appr.authorization.token, a).status == "EXECUTED"


def test_changed_payload_invalidates_authorization():
    s = make_sentinel()
    out = approved(s)
    tampered = make_action(action_payload={"service": "payments-db", "reason": "x"})
    r = s.execute(out.authorization.token, tampered)
    assert r.status == "REJECTED" and r.reason_codes == ["ACTION_MISMATCH"] and not s.adapter.executed


def test_override_requires_execution_of_the_new_payload():
    s = make_sentinel()
    ev = s.evaluate(make_action(source_system="support-agent", action_type="refund_payment",
                                target_resource="payments/p1", action_payload={"payment_id": "p1", "amount": 900}))
    ov = s.override_review(ev.review.review_id, "sam", "partial refund",
                           {"action_payload": {"payment_id": "p1", "amount": 300}})
    original = ev.decision.action
    assert s.execute(ov.authorization.token, original).reason_codes == ["ACTION_MISMATCH"]
    assert s.execute(ov.authorization.token, ov.review.final_action).status == "EXECUTED"
    assert s.adapter.executed[0].action_payload["amount"] == 300


def test_expired_revoked_forged_and_unknown_tokens():
    clock = Clock()
    s = make_sentinel(clock=clock)
    a = make_action()
    out = approved(s)
    grant_id, sig = out.authorization.token.split(".")
    assert s.execute(grant_id + "." + "0" * 64, a).reason_codes == ["INVALID_TOKEN"]
    assert s.execute("garbage", a).reason_codes == ["INVALID_TOKEN"]
    assert s.execute("auth_nope." + sig, a).reason_codes == ["UNKNOWN_GRANT"]
    other_key = Authorizer(s.store, b"another-key-0123456789abcdefgh", clock=clock)
    assert s.execute(other_key.token_for(out.authorization.grant), a).reason_codes == ["INVALID_TOKEN"]
    clock.advance(16 * 60)
    assert s.execute(out.authorization.token, a).reason_codes == ["AUTHORIZATION_EXPIRED"]

    out2 = approved(s, event_id="evt-2", idempotency_key="idem-2")
    s.revoke_authorization(out2.authorization.grant.grant_id, "incident")
    r = s.execute(out2.authorization.token, make_action(event_id="evt-2", idempotency_key="idem-2"))
    assert r.reason_codes == ["AUTHORIZATION_REVOKED"] and not s.adapter.executed


def _pack(version, extra_rule=None):
    raw = yaml.safe_load(CORE_PACK.read_text())
    raw["version"] = version
    if extra_rule:
        raw["rules"].append(extra_rule)
    return check_pack_data(copy.deepcopy(raw))[0]


def test_policy_change_invalidates_stale_authorization():
    s = make_sentinel()
    out = approved(s)
    s.reload_policy([_pack("1.0.1")])
    assert s.execute(out.authorization.token, make_action()).reason_codes == ["POLICY_VERSION_STALE"]


def test_safety_constraints_rechecked_at_execution():
    s = make_sentinel()
    old = s.policy_version
    out = approved(s)
    s.reload_policy([_pack("1.1.0", {"id": "NO_RESTARTS", "description": "freeze", "severity": "high",
                                     "effect": "block", "applies_to": ["restart_service"],
                                     "when": {"field": "action_type", "op": "eq", "value": "restart_service"},
                                     "version": "1", "provenance": "test"})])
    s.settings.accepted_policy_versions.append(old)  # old version still accepted, but the new rule blocks
    r = s.execute(out.authorization.token, make_action())
    assert r.reason_codes == ["SAFETY_CONSTRAINT_FAILED"] and not s.adapter.executed


def test_adapter_failure_recorded_and_not_retried():
    s = make_sentinel()
    s.adapter.fail_actions.add("restart_service")
    out = approved(s)
    assert s.execute(out.authorization.token, make_action()).status == "FAILED"
    assert s.execute(out.authorization.token, make_action()).status == "DUPLICATE"
