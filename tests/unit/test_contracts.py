from __future__ import annotations

import pytest
from pydantic import ValidationError

from sentinel.contracts import (
    DISPOSITION_QUESTION,
    ActionProposal,
    DatasetExample,
    DecisionQuestion,
    GovernanceDecision,
    Outcome,
    ProviderResult,
)
from tests.conftest import action_dict, make_action


def test_valid_action_round_trips_through_json():
    a = make_action()
    again = ActionProposal.model_validate_json(a.model_dump_json())
    assert again == a
    assert again.action_hash() == a.action_hash()


@pytest.mark.parametrize("field,value", [
    ("event_id", ""),
    ("event_id", "has spaces"),
    ("action_type", "Restart-Service"),
    ("target_resource", ""),
    ("proposed_at", "2026-10-14T10:00:00"),  # naive datetime rejected
    ("proposed_at", "not a date"),
])
def test_invalid_fields_rejected(field, value):
    with pytest.raises(ValidationError):
        ActionProposal.model_validate(action_dict(**{field: value}))


def test_unknown_fields_rejected():
    with pytest.raises(ValidationError):
        ActionProposal.model_validate(action_dict(approved=True))


def test_missing_required_field_rejected():
    d = action_dict()
    del d["idempotency_key"]
    with pytest.raises(ValidationError):
        ActionProposal.model_validate(d)


def test_payload_size_and_depth_bounded():
    with pytest.raises(ValidationError):
        make_action(action_payload={"blob": "x" * (70 * 1024)})
    deep: dict = {}
    cur = deep
    for _ in range(15):
        cur["a"] = {}
        cur = cur["a"]
    with pytest.raises(ValidationError):
        make_action(action_payload=deep)


def test_duplicate_evidence_ids_rejected():
    with pytest.raises(ValidationError):
        make_action(evidence_refs=[{"ref_id": "a", "kind": "x"}, {"ref_id": "a", "kind": "y"}])


def test_evidence_stance_defaults_to_unknown():
    a = make_action(evidence_refs=[{"ref_id": "a", "kind": "ticket"}])
    assert a.evidence_refs[0].stance == "unknown"


def test_action_hash_changes_with_payload_but_not_with_evidence():
    a = make_action()
    assert make_action(action_payload={"service": "other"}).action_hash() != a.action_hash()
    assert make_action(target_resource="prod/checkout-api").action_hash() != a.action_hash()
    assert make_action(evidence_refs=[]).action_hash() == a.action_hash()


def _result(**kw):
    base = dict(request_id="r", provider_name="p", provider_version="1", provider_kind="statistical",
                status="ok", label="safe", latency_ms=1.0)
    base.update(kw)
    return ProviderResult(**base)


def test_provider_result_consistency_rules():
    assert _result().succeeded
    with pytest.raises(ValidationError):
        _result(label=None)  # ok without label
    with pytest.raises(ValidationError):
        _result(status="error")  # error without details
    with pytest.raises(ValidationError):
        _result(status="error", error={"code": "X", "message": "m"})  # failed with a label
    with pytest.raises(ValidationError):
        _result(confidence=1.5)
    with pytest.raises(ValidationError):
        _result(confidence_source="self_reported")  # source without a confidence
    with pytest.raises(ValidationError):
        _result(probabilities={"safe": 0.7, "unsafe": 0.7})
    with pytest.raises(ValidationError):
        _result(probabilities={"safe": float("nan"), "unsafe": 0.0})
    r = _result(status="timeout", label=None, error={"code": "TIMEOUT", "message": "slow"})
    assert r.failed and not r.succeeded
    assert r.confidence is None  # unknown, not invented


def test_boolean_question_shape_enforced():
    DecisionQuestion(question_id="q", instructions="x", answer_type="boolean",
                     categories={"true": "y", "false": "n"})
    with pytest.raises(ValidationError):
        DecisionQuestion(question_id="q", instructions="x", answer_type="boolean",
                         categories={"yes": "y", "no": "n"})


def test_governance_decision_only_approve_is_executable():
    a = make_action()
    for status in Outcome:
        d = GovernanceDecision(correlation_id="c", event_id=a.event_id, status=status, reason_codes=[],
                               action=a, action_hash=a.action_hash(), question_id="disposition",
                               policy_version="p")
        assert d.executable is (status == Outcome.APPROVE)


def test_dataset_example_label_must_be_category():
    with pytest.raises(ValidationError):
        DatasetExample(example_id="e", dataset_version="v", synthetic=True, split="test",
                       situation=make_action(), question=DISPOSITION_QUESTION,
                       expected_label="maybe", expected_outcome=Outcome.APPROVE)
