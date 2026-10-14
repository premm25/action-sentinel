from __future__ import annotations

import copy

import pytest
import yaml

from action_sentinel.contracts import Outcome, ReviewState
from action_sentinel.governance import (
    InvalidTransition,
    RevalidationFailed,
    ReviewError,
    ReviewerNotAuthorized,
    ReviewNotFound,
)
from action_sentinel.validation import check_pack_data
from tests.conftest import CORE_PACK, Clock, make_action, make_sentinel

PROD_RESTART = dict(target_resource="prod/checkout-api")  # medium risk -> operator review
REFUND = dict(source_system="support-agent", action_type="refund_payment", target_resource="payments/p1",
              action_payload={"payment_id": "p1", "amount": 900})  # high risk -> senior review


def pending(s, **kw):
    out = s.evaluate(make_action(**kw))
    assert out.decision.status == Outcome.HUMAN_REVIEW and out.review is not None
    return out.review


def test_review_created_with_required_level():
    s = make_sentinel()
    r = pending(s, **REFUND)
    assert r.state == ReviewState.PENDING and r.required_level.value == "senior"
    assert [p.review_id for p in s.store.list_reviews(state="PENDING")] == [r.review_id]
    assert r.history[0].action == "created"


def test_approve_issues_authorization_and_records_history():
    s = make_sentinel()
    r = pending(s, **PROD_RESTART)
    out = s.approve_review(r.review_id, "olivia", "checked the alert")
    assert out.review.state == ReviewState.APPROVED and out.review.final_disposition == "approved"
    assert out.authorization.grant.review_id == r.review_id
    assert out.review.reviewer_id == "olivia" and out.review.decided_at is not None
    assert [h.action for h in out.review.history] == ["created", "approve"]


def test_unauthorized_reviewers_rejected_and_recorded():
    s = make_sentinel()
    r = pending(s, **REFUND)
    with pytest.raises(ReviewerNotAuthorized) as e:
        s.approve_review(r.review_id, "olivia")  # operator < senior
    assert e.value.code == "INSUFFICIENT_APPROVAL_LEVEL"
    with pytest.raises(ReviewerNotAuthorized) as e:
        s.approve_review(r.review_id, "mallory")
    assert e.value.code == "UNKNOWN_REVIEWER"
    rec = s.store.get_review(r.review_id)
    assert rec.state == ReviewState.PENDING and [h.action for h in rec.history].count("denied") == 2
    assert len(s.store.list_events(event_type="review.denied")) == 2


def test_self_review_and_inactive_reviewer_rejected():
    s = make_sentinel(reviewers=[{"reviewer_id": "ops-agent", "level": "admin"},
                                 {"reviewer_id": "gone", "level": "admin", "active": False}])
    r = pending(s, **PROD_RESTART)
    with pytest.raises(ReviewerNotAuthorized, match="own action"):
        s.approve_review(r.review_id, "ops-agent")
    with pytest.raises(ReviewerNotAuthorized, match="inactive"):
        s.approve_review(r.review_id, "gone")


def test_invalid_transitions_and_duplicate_requests():
    s = make_sentinel()
    r = pending(s, **PROD_RESTART)
    s.approve_review(r.review_id, "olivia")
    with pytest.raises(InvalidTransition):
        s.approve_review(r.review_id, "olivia")  # duplicate approval
    with pytest.raises(InvalidTransition):
        s.reject_review(r.review_id, "sam", "changed my mind")
    with pytest.raises(InvalidTransition):
        s.cancel_review(r.review_id, "ada", "no longer needed")
    with pytest.raises(ReviewNotFound):
        s.approve_review("rev_nope", "olivia")


def test_reject_requires_reason():
    s = make_sentinel()
    r = pending(s, **PROD_RESTART)
    with pytest.raises(ReviewError, match="reason"):
        s.reject_review(r.review_id, "olivia", "")
    out = s.reject_review(r.review_id, "olivia", "not during peak hours")
    assert out.state == ReviewState.REJECTED and out.final_disposition == "rejected"


def test_expiry():
    clock = Clock()
    s = make_sentinel(clock=clock)
    r = pending(s, **PROD_RESTART)
    clock.advance(25 * 3600)
    with pytest.raises(InvalidTransition, match="expired"):
        s.approve_review(r.review_id, "olivia")
    assert s.store.get_review(r.review_id).state == ReviewState.EXPIRED
    r2 = pending(s, **{**PROD_RESTART, "event_id": "evt-2", "idempotency_key": "idem-2"})
    clock.advance(25 * 3600)
    assert s.expire_due_reviews() == [r2.review_id]


def test_cancel():
    s = make_sentinel()
    r = pending(s, **PROD_RESTART)
    assert s.cancel_review(r.review_id, "ada", "duplicate ticket").state == ReviewState.CANCELLED


def test_override_is_revalidated_and_approved():
    s = make_sentinel()
    r = pending(s, **REFUND)
    out = s.override_review(r.review_id, "sam", "partial refund only",
                            {"action_payload": {"payment_id": "p1", "amount": 300}})
    assert out.review.state == ReviewState.OVERRIDDEN
    assert out.decision.derived_from == r.decision_id and out.decision.event_id.endswith(".override1")
    assert out.review.final_action.action_payload["amount"] == 300
    assert out.authorization.grant.action_hash == out.review.final_action.action_hash()
    assert out.authorization.grant.action_hash != r.original_action.action_hash()


def test_override_failing_revalidation_is_rejected():
    s = make_sentinel()
    r = pending(s, **REFUND)
    with pytest.raises(RevalidationFailed):
        s.override_review(r.review_id, "sam", "refund everything",
                          {"action_payload": {"payment_id": "p1", "amount": 50_000}})  # REFUND_HARD_LIMIT
    rec = s.store.get_review(r.review_id)
    assert rec.state == ReviewState.PENDING and rec.history[-1].action == "override_rejected"
    assert s.store.list_events(event_type="review.override_rejected")


def test_override_raising_risk_needs_higher_authority():
    s = make_sentinel()
    r = pending(s, **PROD_RESTART)  # operator review
    with pytest.raises(RevalidationFailed, match="FORCE_IN_PROD"):  # an override into a hard violation
        s.override_review(r.review_id, "olivia", "force it",
                          {"target_resource": "prod/checkout-api", "action_payload": {"service": "x", "force": True}})
    s2 = make_sentinel()
    r2 = pending(s2, source_system="monitoring-agent", action_type="scale_deployment", target_resource="staging/web",
                 action_payload={"deployment": "web", "replicas": 3})
    with pytest.raises(ReviewerNotAuthorized, match="requires senior"):
        s2.override_review(r2.review_id, "olivia", "scale harder", {"action_payload": {"deployment": "web",
                                                                                       "replicas": 80}})


def test_override_field_restrictions():
    s = make_sentinel()
    r = pending(s, **PROD_RESTART)
    with pytest.raises(ReviewError, match="cannot be overridden"):
        s.override_review(r.review_id, "olivia", "swap agent", {"agent_id": "root"})
    with pytest.raises(ReviewError, match="must change"):
        s.override_review(r.review_id, "olivia", "nothing", {})
    with pytest.raises(ReviewError, match="reason"):
        s.override_review(r.review_id, "olivia", "", {"target_resource": "staging/x"})


def test_approval_revalidates_against_policy_changed_since_review():
    s = make_sentinel()
    r = pending(s, **PROD_RESTART)
    raw = yaml.safe_load(CORE_PACK.read_text())
    raw["version"] = "1.1.0"
    raw["rules"].append({"id": "PROD_FREEZE", "description": "prod restarts frozen", "severity": "high",
                         "effect": "block", "applies_to": ["restart_service"],
                         "when": {"field": "derived.target_env", "op": "eq", "value": "prod"},
                         "version": "1", "provenance": "test"})
    s.reload_policy([check_pack_data(copy.deepcopy(raw))[0]])
    with pytest.raises(RevalidationFailed):
        s.approve_review(r.review_id, "ada", "approving anyway")
    assert s.store.get_review(r.review_id).state == ReviewState.PENDING
