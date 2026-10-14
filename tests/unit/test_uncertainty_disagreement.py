from __future__ import annotations

import pytest

from action_sentinel.contracts import EvidenceAssessment, ProviderResult, Severity
from action_sentinel.disagreement import detect_disagreement
from action_sentinel.uncertainty import UncertaintyThresholds, assess_uncertainty
from action_sentinel.validation import ValidationEngine, load_pack_file
from tests.conftest import CORE_PACK, make_action

EV = EvidenceAssessment(status="sufficient", required=1, supporting=1, contradicting=0, neutral=0, unknown=0)
TH = UncertaintyThresholds()


def res(name="m", kind="statistical", status="ok", label="safe", conf=0.95, **kw):
    if status in ("error", "timeout", "unavailable"):
        label, conf = None, None
        kw.setdefault("error", {"code": status.upper(), "message": "x"})
    return ProviderResult(request_id="r", provider_name=name, provider_version="1", provider_kind=kind,
                          status=status, label=label, confidence=conf,
                          confidence_source="model_probability" if conf is not None else "none",
                          latency_ms=1, **kw)


# ------------------------------------------------------------------ uncertainty

@pytest.mark.parametrize("results,status", [
    ([], "failed"),
    ([res(status="error"), res("n", status="timeout")], "failed"),
    ([res(status="abstained", label=None, conf=None)], "abstained"),
    ([res(conf=0.3)], "abstained"),                                   # below abstain threshold
    ([res(conf=0.7)], "uncertain"),                                   # below approval threshold
    ([res(conf=None)], "unknown"),                                    # missing confidence is not assumed high
    ([res(kind="deterministic", conf=None)], "confident"),           # rules carry no probability
    ([res(conf=0.95), res("n", status="error")], "uncertain"),        # partial failure
    ([res(conf=0.95, prediction_set=["safe", "unsafe"])], "uncertain"),
    ([res(conf=0.99)], "confident"),
])
def test_uncertainty_status(results, status):
    assert assess_uncertainty(results, EV, TH).status == status


def test_unknown_calibration_is_explicit_and_can_be_required():
    a = assess_uncertainty([res(conf=0.99)], EV, TH)
    assert a.providers[0].calibration_status == "unknown"
    assert "m:CALIBRATION_UNKNOWN" in a.reasons
    strict = assess_uncertainty([res(conf=0.99)], EV, TH.model_copy(update={"require_measured_calibration": True}))
    assert strict.status == "uncertain"


def test_measured_calibration_profile_used():
    good = assess_uncertainty([res(conf=0.99)], EV, TH, {"m": {"ece": 0.03}})
    assert good.status == "confident" and good.providers[0].calibration_ece == 0.03
    bad = assess_uncertainty([res(conf=0.99)], EV, TH, {"m": {"ece": 0.3}})
    assert bad.status == "uncertain" and "m:POORLY_CALIBRATED" in bad.reasons


def test_high_confidence_does_not_change_evidence_assessment():
    weak = EvidenceAssessment(status="insufficient", required=1, supporting=0, contradicting=0, neutral=0, unknown=0)
    a = assess_uncertainty([res(conf=0.999)], weak, TH)
    assert a.status == "confident" and a.evidence.status == "insufficient"  # kept separate, not merged


# ------------------------------------------------------------------ disagreement

@pytest.fixture(scope="module")
def engine():
    return ValidationEngine([load_pack_file(CORE_PACK)])


def kinds(rep):
    return [r.kind for r in rep.records]


def test_complete_agreement(engine):
    v = engine.validate(make_action(), "p")
    rep = detect_disagreement([res("a"), res("b", conf=0.9)], v, "parallel")
    assert rep.agreement and rep.records == []


def test_category_disagreement_severity(engine):
    v = engine.validate(make_action(), "p")
    rep = detect_disagreement([res("a"), res("b", label="unsafe")], v, "parallel")
    cat = [r for r in rep.records if r.kind == "category"][0]
    assert not rep.agreement and cat.severity == Severity.HIGH and cat.recommended_handling == "human_review"
    mild = detect_disagreement([res("a"), res("b", label="needs_review")], v, "parallel")
    assert [r for r in mild.records if r.kind == "category"][0].severity == Severity.MEDIUM


def test_confidence_disagreement_is_informational(engine):
    v = engine.validate(make_action(), "p")
    rep = detect_disagreement([res("a", conf=0.99), res("b", conf=0.55)], v, "parallel")
    assert kinds(rep) == ["confidence"] and rep.agreement


def test_model_vs_policy(engine):
    blocked = engine.validate(make_action(action_type="disable_audit_logging"), "p")
    rep = detect_disagreement([res("a", label="safe", conf=0.99)], blocked, "single")
    rec = [r for r in rep.records if r.kind == "model_vs_policy"][0]
    assert rec.recommended_handling == "policy_precedence" and rec.severity == Severity.HIGH
    clean = engine.validate(make_action(), "p")
    rep2 = detect_disagreement([res("a", label="unsafe")], clean, "single")
    assert "model_vs_policy" in kinds(rep2)


def test_evidence_vs_model_and_contradictory_findings(engine):
    contra = engine.validate(make_action(evidence_refs=[{"ref_id": "x", "kind": "m", "stance": "contradicts"}]), "p")
    rep = detect_disagreement([res("a", label="safe")], contra, "single")
    assert "evidence_vs_model" in kinds(rep)
    v = contra.model_copy(update={"findings": contra.findings + [contra.findings[0].model_copy(
        update={"code": "POLICY_CONFLICT"})]})
    assert "contradictory_findings" in kinds(detect_disagreement([res("a", label="needs_review")], v, "single"))


def test_missing_results_recorded(engine):
    v = engine.validate(make_action(), "p")
    rep = detect_disagreement([res("a"), res("b", status="timeout")], v, "parallel")
    assert kinds(rep) == ["missing_result"] and rep.agreement
