"""Governance precedence through the full pipeline (in-memory store, scripted providers)."""

from __future__ import annotations

import pytest

from sentinel.contracts import Outcome
from sentinel.governance import GovernanceEngine, GovernancePolicy, PipelineFlags
from tests.conftest import make_action, make_sentinel

DESTRUCTIVE = dict(source_system="infra-agent", action_type="delete_database", target_resource="prod/orders",
                   action_payload={"database": "orders"},
                   evidence_refs=[{"ref_id": "a", "kind": "t", "stance": "supports"},
                                  {"ref_id": "b", "kind": "t", "stance": "supports"}])


def evaluate(s, providers=None, **kw):
    return s.evaluate(make_action(**kw), providers=providers).decision


def test_rules_only_auto_approves_clean_low_risk_action():
    d = evaluate(make_sentinel())
    assert d.status == Outcome.APPROVE and d.reason_codes == ["ALL_CONDITIONS_MET"]
    assert d.authorization_id and d.policy_version.startswith("gov-1.0.0+core@1.0.0")


def test_hard_violation_blocks_despite_confident_unanimous_models():
    s = make_sentinel()
    d = evaluate(s, providers=["mock-a", "mock-b"], **DESTRUCTIVE)  # both mocks: safe at 0.95 / 0.9
    assert d.status == Outcome.BLOCK
    assert "PROD_DATABASE_DELETE" in d.failed_rules and d.authorization_id is None
    assert any(r.kind == "model_vs_policy" for r in d.disagreement.records)


def test_high_confidence_alone_never_approves_high_risk():
    s = make_sentinel()
    d = evaluate(s, providers=["mock-a"], source_system="infra-agent", action_type="rotate_credentials",
                 target_resource="staging/db", action_payload={"credential_id": "c1"})
    assert d.status == Outcome.HUMAN_REVIEW and "RISK_ABOVE_AUTO_APPROVAL:high" in d.reason_codes


def test_all_providers_failed_is_error_not_approval():
    s = make_sentinel({"mock-a": {"evt-1": {"fail": "down"}}, "mock-b": {"evt-1": {"timeout": True}}})
    d = evaluate(s, providers=["mock-a", "mock-b"])
    assert d.status == Outcome.ERROR and "PROVIDERS_FAILED" in d.reason_codes and d.authorization_id is None


def test_abstention_is_not_approval():
    s = make_sentinel({"mock-a": {"evt-1": {"abstain": True}}})
    d = evaluate(s, providers=["mock-a"])
    assert d.status == Outcome.ABSTAIN and d.authorization_id is None


def test_abstention_with_review_reasons_escalates():
    s = make_sentinel({"mock-a": {"evt-1": {"abstain": True}}})
    d = evaluate(s, providers=["mock-a"], target_resource="prod/checkout-api")
    assert d.status == Outcome.HUMAN_REVIEW and "PROVIDERS_ABSTAINED" in d.reason_codes


def test_disagreement_escalates():
    s = make_sentinel({"mock-b": {"evt-1": {"label": "unsafe", "confidence": 0.9}}})
    d = evaluate(s, providers=["mock-a", "mock-b"])
    assert d.status == Outcome.HUMAN_REVIEW and "DISAGREEMENT:CATEGORY" in d.reason_codes


def test_majority_cannot_outvote_policy():
    s = make_sentinel()
    d = evaluate(s, providers=["rules", "mock-a", "mock-b"], action_type="disable_audit_logging")
    assert d.status == Outcome.BLOCK


@pytest.mark.parametrize("conf,status", [(0.99, Outcome.APPROVE), (0.7, Outcome.HUMAN_REVIEW),
                                         (0.2, Outcome.ABSTAIN)])
def test_confidence_thresholds(conf, status):
    s = make_sentinel({"mock-a": {"evt-1": {"label": "safe", "confidence": conf}}})
    assert evaluate(s, providers=["mock-a"]).status == status


def test_missing_confidence_routes_to_review():
    s = make_sentinel({"mock-a": {"evt-1": {"label": "safe"}}})
    d = evaluate(s, providers=["mock-a"])
    assert d.status == Outcome.HUMAN_REVIEW and "UNCERTAINTY_UNKNOWN" in d.reason_codes


def test_malformed_provider_output_cannot_approve():
    s = make_sentinel({"mock-a": {"evt-1": {"malformed": True}}})
    assert evaluate(s, providers=["mock-a"]).status == Outcome.ERROR


def test_insufficient_evidence_escalates():
    d = evaluate(make_sentinel(), evidence_refs=[])
    assert d.status == Outcome.HUMAN_REVIEW and "EVIDENCE_INSUFFICIENT" in d.reason_codes


def test_model_flagging_unsafe_escalates_rather_than_blocking():
    s = make_sentinel({"mock-a": {"evt-1": {"label": "unsafe", "confidence": 0.99}}})
    d = evaluate(s, providers=["mock-a"])
    assert d.status == Outcome.HUMAN_REVIEW and "PROVIDER_NOT_PERMISSIVE:mock-a=unsafe" in d.reason_codes


def test_naive_and_ablation_modes():
    eng = GovernanceEngine(GovernancePolicy())
    s = make_sentinel()
    d = s.assess(make_action(**DESTRUCTIVE), correlation_id="c", providers=["mock-a"],
                 flags=PipelineFlags(mode="naive"))
    assert d.status == Outcome.APPROVE  # the naive baseline approves what policy blocks: that is the point
    no_review = s.assess(make_action(evidence_refs=[]), correlation_id="c", providers=["mock-a"],
                         flags=PipelineFlags(use_review=False))
    assert no_review.status == Outcome.APPROVE and "REVIEW_DISABLED_FALLBACK_APPROVE" in no_review.reason_codes
    assert eng.decide(None, [], None, None, PipelineFlags(mode="naive")).status == Outcome.ERROR
