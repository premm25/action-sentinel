"""Targeted tests for authorization and governance branches found uncovered by the coverage run."""

from __future__ import annotations

import json

import pytest

from action_sentinel.contracts import Outcome
from action_sentinel.execution import AuthorizationError
from action_sentinel.governance import GovernanceEngine, GovernancePolicy, PipelineFlags
from tests.conftest import make_action, make_sentinel


def test_grant_bound_to_unfinished_review_cannot_execute():
    s = make_sentinel()
    out = s.evaluate(make_action(target_resource="prod/checkout-api"))
    a = out.decision.action
    # A grant that claims a review the reviewer never completed (e.g. forged by someone holding the key).
    forged = s.authorizer.issue(decision_id=out.decision.decision_id, correlation_id=out.decision.correlation_id,
                                action=a, policy_version=s.policy_version, issued_by="attacker",
                                review_id=out.review.review_id)
    r = s.execute(forged.token, a)
    assert r.reason_codes == ["DECISION_NOT_EXECUTABLE"] and s.adapter.executed == []
    # Approving the review issues a *different* grant; the forged one stays useless.
    appr = s.approve_review(out.review.review_id, "olivia", "ok")
    assert s.execute(forged.token, a).reason_codes == ["DECISION_NOT_EXECUTABLE"]
    assert s.execute(appr.authorization.token, a).status == "EXECUTED"


def test_grant_for_missing_decision_and_rejected_review():
    s = make_sentinel()
    a = make_action()
    orphan = s.authorizer.issue(decision_id="dec_missing", correlation_id="c", action=a,
                                policy_version=s.policy_version, issued_by="attacker")
    assert s.execute(orphan.token, a).reason_codes == ["DECISION_NOT_EXECUTABLE"]
    out = s.evaluate(make_action(event_id="e2", idempotency_key="k2", target_resource="prod/checkout-api"))
    s.reject_review(out.review.review_id, "olivia", "not today")
    forged = s.authorizer.issue(decision_id=out.decision.decision_id, correlation_id="c", action=out.decision.action,
                                policy_version=s.policy_version, issued_by="attacker", review_id=out.review.review_id)
    assert s.execute(forged.token, out.decision.action).reason_codes == ["DECISION_NOT_EXECUTABLE"]


def test_revoke_unknown_grant():
    s = make_sentinel()
    with pytest.raises(AuthorizationError):
        s.revoke_authorization("auth_missing", "x")


def test_min_successful_providers_and_ungated_abstention():
    s = make_sentinel({"mock-b": {"evt-1": {"fail": "down"}}})
    s.governance = GovernanceEngine(GovernancePolicy(min_successful_providers=2))
    d = s.assess(make_action(), correlation_id="c", providers=["mock-a", "mock-b"])
    assert d.status == Outcome.HUMAN_REVIEW and "TOO_FEW_PROVIDER_RESULTS" in d.reason_codes
    s2 = make_sentinel({"mock-a": {"evt-1": {"abstain": True}}})
    d2 = s2.assess(make_action(), correlation_id="c", providers=["mock-a"], flags=PipelineFlags(use_uncertainty=False))
    assert d2.status == Outcome.ABSTAIN


def test_naive_abstention():
    s = make_sentinel({"mock-a": {"evt-1": {"abstain": True}}})
    d = s.assess(make_action(), correlation_id="c", providers=["mock-a"], flags=PipelineFlags(mode="naive"))
    assert d.status == Outcome.ABSTAIN


def test_cli_in_process(tmp_path, capsys, monkeypatch):
    from action_sentinel.cli import main
    from tests.conftest import action_dict
    monkeypatch.setenv("SENTINEL_ENV", "dev")
    db = str(tmp_path / "c.db")
    f = tmp_path / "a.json"
    f.write_text(json.dumps([action_dict(), action_dict(event_id="e9", idempotency_key="k9",
                                                        target_resource="prod/checkout-api")]))
    assert main(["--db", db, "evaluate", str(f), "-v"]) == 0
    assert main(["--db", db, "evaluate", str(f)]) == 0  # duplicates reported, not re-evaluated
    out = capsys.readouterr().out
    assert "APPROVE" in out and "(duplicate)" in out
    conflict = tmp_path / "b.json"
    conflict.write_text(json.dumps(action_dict(action_payload={"service": "other"})))
    assert main(["--db", db, "evaluate", str(conflict)]) == 2
    assert main(["--db", db, "validate-rules"]) == 0
    assert main(["--db", db, "providers"]) == 0
    assert main(["--db", db, "reviews", "list"]) == 0
    listed = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("rev_")]
    rid = listed[-1].split()[0]
    assert main(["--db", db, "reviews", "override", rid, "--reviewer", "olivia", "--reason", "staging instead",
                 "--changes", json.dumps({"target_resource": "staging/checkout-api"})]) == 0
    assert "OVERRIDDEN" in capsys.readouterr().out
    assert main(["--db", db, "lab", "run", "--providers", "rules,mock-a", "--limit", "5",
                 "--out", str(tmp_path / "r")]) == 0
    assert main(["--db", db, "lab", "show"]) == 0
    assert main(["--db", db, "lab", "show", "run_missing"]) == 1
    assert main(["--db", db, "audit-export", "--out", str(tmp_path / "a.jsonl")]) == 0
