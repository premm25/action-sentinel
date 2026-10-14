"""VegaCore adapter and contracts, against the clearly-labelled test double (no inference here)."""

from __future__ import annotations

import time

import pytest
from pydantic import ValidationError

from action_sentinel.vegalab.contracts import (
    CandidateOption,
    DecisionEvidence,
    DecisionQuestion,
    DecisionRequest,
)
from action_sentinel.vegalab.testing import FakeVegaRunner
from action_sentinel.vegalab.vega_core import VegaCore, VegaInferenceError


def choice_q(qid="team", labels=("billing", "technical", "sales")):
    return DecisionQuestion(question_id=qid, type="choice", instructions="Which team?",
                            options=[CandidateOption(label=lab, description="desc " + lab) for lab in labels])


def bool_q(qid="churn"):
    return DecisionQuestion(question_id=qid, type="boolean", instructions="At risk?",
                            options=[CandidateOption(label="false", description="no signal"),
                                     CandidateOption(label="true", description="intent to cancel")])


def score_q(qid="urgency"):
    return DecisionQuestion(question_id=qid, type="score", instructions="How urgent?",
                            options=[CandidateOption(label=str(i), description=d)
                                     for i, d in enumerate(["low", "medium", "high"])])


def req(*qs, **kw):
    return DecisionRequest(state=kw.pop("state", "invoice overdue"), questions=list(qs) or [choice_q()], **kw)


# ------------------------------------------------------------------ contracts

def test_question_shapes_and_native_mapping():
    assert choice_q().to_vega() == {"type": "choice", "instructions": "Which team?",
                                    "criteria": {"billing": "desc billing", "technical": "desc technical",
                                                 "sales": "desc sales"}}
    assert score_q().to_vega()["criteria"] == ["low", "medium", "high"]
    assert bool_q().to_vega()["criteria"] == {"false": "no signal", "true": "intent to cancel"}
    bare = DecisionQuestion(question_id="b", type="boolean", instructions="x",
                            options=[CandidateOption(label="false"), CandidateOption(label="true")])
    assert "criteria" not in bare.to_vega()


@pytest.mark.parametrize("kw", [
    dict(type="choice", options=[{"label": "a"}]),                                   # one option
    dict(type="choice", options=[{"label": "a"}, {"label": "a"}]),                   # duplicate
    dict(type="boolean", options=[{"label": "yes"}, {"label": "no"}]),               # wrong labels
    dict(type="boolean", options=[{"label": "true"}, {"label": "false"}]),           # wrong order
    dict(type="score", options=[{"label": "1", "description": "a"}, {"label": "2", "description": "b"}]),
    dict(type="score", options=[{"label": "0"}, {"label": "1", "description": "b"}]),  # missing description
    dict(type="maybe", options=[{"label": "a"}, {"label": "b"}]),
])
def test_invalid_questions(kw):
    with pytest.raises(ValidationError):
        DecisionQuestion(question_id="q", instructions="x", **kw)


def test_request_validation():
    with pytest.raises(ValidationError):
        DecisionRequest(state="  ", questions=[choice_q()])
    with pytest.raises(ValidationError):
        DecisionRequest(state="x", questions=[choice_q(), choice_q()])        # duplicate ids
    with pytest.raises(ValidationError):
        DecisionRequest(state="x" * 500_000, questions=[choice_q()])
    with pytest.raises(ValidationError):
        DecisionRequest(state={"evidence": []}, questions=[choice_q()],
                        evidence=[DecisionEvidence(ref_id="a", text="b")])
    with pytest.raises(ValidationError):
        DecisionRequest(state="x", questions=[choice_q()], views=4)


def test_evidence_is_merged_into_state():
    r = DecisionRequest(state="email text", questions=[choice_q()],
                        evidence=[DecisionEvidence(ref_id="e1", text="ledger shows two charges")])
    assert r.vega_state() == {"situation": "email text", "evidence": [{"ref": "e1", "text": "ledger shows two charges"}]}


# ------------------------------------------------------------------ adapter

def test_normalization_preserves_native_and_flags_test_double():
    runner = FakeVegaRunner()
    core = VegaCore(runner)
    res = core.decide(req(choice_q(), bool_q(), score_q()), adapter_mode="off")
    assert res.live_inference is False  # a test double can never pass as live inference
    assert runner.mode == "off" and res.adapter_mode == "off"
    d = res.decisions
    assert d["team"].selected_label == "billing" and d["team"].uncertainty.probabilities["billing"] == 0.7
    assert d["churn"].selected_label == "false" and d["churn"].uncertainty.prediction_set == ["false"]
    assert set(d["churn"].uncertainty.probabilities) == {"false", "true"}       # "False"/"True" normalized
    assert d["urgency"].selected_label == "0" and d["urgency"].score == 0
    assert d["urgency"].expected_score == pytest.approx(0.15 * 1 + 0.15 * 2)
    assert res.native["answers"]["churn"]["probs"] == {"False": 0.7, "True": 0.3}  # native kept verbatim
    assert res.decisions["team"].uncertainty.abstain_risk == "0.1"


def _respond_with(answer):
    def respond(state, questions, mode):
        return {"answers": {"team": answer}, "latency_ms": 1.0, "n_tokens": 3, "views": 1}
    return respond


def test_missing_confidence_stays_none_and_abstain_is_not_hidden():
    ans = {"probs": {"billing": 0.4, "technical": 0.35, "sales": 0.25}, "choice": "billing", "abstain": True}
    res = VegaCore(FakeVegaRunner(_respond_with(ans))).decide(req())
    u = res.decisions["team"].uncertainty
    assert u.confidence is None and u.prediction_set is None and u.unbound is None and u.settle_steps is None
    assert res.decisions["team"].abstained is True
    assert res.decisions["team"].selected_label == "billing"   # the native top option, flagged, not dropped


@pytest.mark.parametrize("answer,needle", [
    ({"probs": {"billing": 1.0}, "choice": "billing"}, "cover every option"),
    ({"probs": {"billing": 0.5, "technical": 0.3, "sales": 0.1, "legal": 0.1}, "choice": "billing"}, "unknown option"),
    ({"probs": {"billing": 1.5, "technical": 0.0, "sales": 0.0}, "choice": "billing"}, "invalid probability"),
    ({"probs": {"billing": 0.5, "technical": 0.3, "sales": 0.2}, "choice": "legal"}, "selected option"),
    ({"probs": {"billing": 0.5, "technical": 0.3, "sales": 0.2}}, "selected option"),
    ("not a dict", "no answer"),
])
def test_malformed_native_output_is_an_error(answer, needle):
    with pytest.raises(VegaInferenceError) as e:
        VegaCore(FakeVegaRunner(_respond_with(answer))).decide(req())
    assert e.value.error.code == "MALFORMED_NATIVE_OUTPUT" and needle in e.value.error.message
    assert e.value.error.request_id


def test_runner_exceptions_become_typed_errors():
    boom = VegaCore(FakeVegaRunner(raise_on_decide=RuntimeError("kernel exploded")))
    with pytest.raises(VegaInferenceError) as e:
        boom.decide(req())
    assert e.value.error.code == "INFERENCE_FAILED" and "kernel exploded" in e.value.error.message
    long_ = VegaCore(FakeVegaRunner(raise_on_decide=ValueError("prompt is too long for max_len 73728")))
    with pytest.raises(VegaInferenceError) as e:
        long_.decide(req())
    assert e.value.error.code == "INPUT_TOO_LONG"
    with pytest.raises(VegaInferenceError) as e:
        VegaCore(FakeVegaRunner()).decide({"not": "a request"})  # type: ignore[arg-type]
    assert e.value.error.code == "INVALID_INPUT" and "questions" in e.value.error.message
    with pytest.raises(VegaInferenceError) as e:
        VegaCore(FakeVegaRunner()).decide(req(), adapter_mode="sideways")  # type: ignore[arg-type]
    assert e.value.error.code == "INVALID_INPUT"
    ok = VegaCore(FakeVegaRunner()).decide({"state": "x", "questions": [choice_q().model_dump()]})
    assert ok.decisions["team"].selected_label == "billing"


def test_timeout():
    def slow(state, questions, mode):
        time.sleep(1.0)
        return FakeVegaRunner.default_response(state, questions, mode)
    core = VegaCore(FakeVegaRunner(slow), timeout_s=0.1)
    with pytest.raises(VegaInferenceError) as e:
        core.decide(req())
    assert e.value.error.code == "TIMEOUT" and e.value.error.retryable


def test_no_answer_for_question():
    def respond(state, questions, mode):
        return {"answers": {}, "latency_ms": 1.0}
    with pytest.raises(VegaInferenceError, match="no answer"):
        VegaCore(FakeVegaRunner(respond)).decide(req())
