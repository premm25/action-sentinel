from __future__ import annotations

import json
import math

import httpx
import pytest

from action_sentinel.contracts import DISPOSITION_QUESTION, DecisionQuestion, DecisionRequest
from action_sentinel.providers import (
    DecisionProvider,
    MockDecisionProvider,
    OllamaDecisionProvider,
    ProviderConfigError,
    ProviderRegistry,
    ProviderRunner,
    RawAnswer,
    RuleBasedDecisionProvider,
    VegaMLDecisionProvider,
)
from tests.conftest import CLASSIFIER, ROOT, make_action


def req(**kw) -> DecisionRequest:
    return DecisionRequest(correlation_id="c1", action=make_action(**kw))


# ------------------------------------------------------------------ rule-based

@pytest.fixture(scope="module")
def rules():
    return RuleBasedDecisionProvider.from_file("rules", CLASSIFIER)


def test_rule_based_is_deterministic(rules):
    r1, r2 = rules.decide(req()), rules.decide(req())
    assert r1.label == r2.label == "safe"
    assert r1.provider_kind == "deterministic" and r1.calibration_status == "not_applicable"
    assert r1.confidence is None and r1.confidence_source == "none"  # no invented confidence
    assert r1.metadata["matched_rule"] == "default"


@pytest.mark.parametrize("kw,label,rule", [
    (dict(action_type="delete_database", action_payload={"database": "x"}), "unsafe", "C_DESTRUCTIVE"),
    (dict(target_resource="prod/api", action_payload={"service": "a", "force": True}), "unsafe", "C_FORCE_PROD"),
    (dict(evidence_refs=[{"ref_id": "a", "kind": "m", "stance": "contradicts"}]), "unsafe", "C_CONTRADICTED"),
    (dict(target_resource="prod/api"), "needs_review", "C_PROD"),
    (dict(evidence_refs=[]), "needs_review", "C_NO_SUPPORT"),
])
def test_rule_based_classification(rules, kw, label, rule):
    r = rules.decide(req(**kw))
    assert (r.label, r.metadata["matched_rule"]) == (label, rule)


def test_rule_based_abstains_on_unknown_question(rules):
    q = DecisionQuestion(question_id="other", instructions="x", categories={"a": "a", "b": "b"})
    r = rules.decide(DecisionRequest(correlation_id="c", action=make_action(), question=q))
    assert r.status == "abstained" and r.metadata["abstain_reason"] == "NO_RULES_FOR_QUESTION"


def test_rule_based_label_mismatch_is_error(rules):
    q = DecisionQuestion(question_id="disposition", instructions="x", categories={"yes": "y", "no": "n"})
    r = rules.decide(DecisionRequest(correlation_id="c", action=make_action(), question=q))
    assert r.status == "error" and r.error.code == "CONFIG_MISMATCH"


# ------------------------------------------------------------------ mock

@pytest.mark.parametrize("behaviour,status,code", [
    ({"label": "safe", "confidence": 0.9}, "ok", None),
    ({"abstain": True}, "abstained", None),
    ({"fail": "boom"}, "error", "SIMULATED_FAILURE"),
    ({"timeout": True}, "timeout", "TIMEOUT"),
    ({"malformed": True}, "error", "MALFORMED_OUTPUT"),
    ({"label": "safe", "probabilities": {"safe": 0.5, "unsafe": 0.6}}, "error", "MALFORMED_OUTPUT"),
    ({"label": "safe", "probabilities": {"safe": 0.5, "maybe": 0.5}}, "error", "MALFORMED_OUTPUT"),
])
def test_mock_behaviours(behaviour, status, code):
    m = MockDecisionProvider("mock", script={"evt-1": behaviour})
    r = m.decide(req())
    assert r.status == status and r.is_test_output
    assert (r.error.code if r.error else None) == code


def test_mock_probabilities_give_confidence_and_default_abstains():
    m = MockDecisionProvider("mock", script={"evt-1": {"label": "unsafe", "probabilities":
                                                       {"safe": 0.1, "needs_review": 0.2, "unsafe": 0.7}}})
    r = m.decide(req())
    assert r.confidence == pytest.approx(0.7) and r.confidence_source == "model_probability"
    assert m.decide(req(event_id="other")).status == "abstained"
    assert m.calls == ["evt-1", "other"]


def test_provider_exception_becomes_error():
    class Broken(DecisionProvider):
        provider_type = "broken"

        def _decide(self, request):
            raise ZeroDivisionError("oops")

    r = Broken("broken").decide(req())
    assert r.status == "error" and r.error.code == "PROVIDER_EXCEPTION" and r.label is None


# ------------------------------------------------------------------ ollama (no network: MockTransport)

def _ollama(handler, **kw):
    return OllamaDecisionProvider("llm", transport=httpx.MockTransport(handler), **kw)


def _chat(content, tops=None):
    body = {"message": {"role": "assistant", "content": content}}
    if tops is not None:
        body["logprobs"] = [{"token": content, "logprob": tops[0][1],
                             "top_logprobs": [{"token": t, "logprob": lp} for t, lp in tops]}]
    return httpx.Response(200, json=body)


def test_ollama_reads_option_distribution_from_logprobs():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _chat("C", [("C", math.log(0.6)), (" C", math.log(0.1)), ("B", math.log(0.2)), ("x", math.log(0.05))])

    r = _ollama(handler).decide(req())
    assert r.status == "ok" and r.label == "unsafe"
    assert r.probabilities["unsafe"] == pytest.approx(0.7 / 0.9)
    assert r.probabilities["safe"] == 0.0
    assert r.confidence_source == "model_probability" and r.calibration_status == "unknown"
    assert r.metadata["option_mass"] == pytest.approx(0.9, abs=1e-3)
    b = seen["body"]
    assert b["options"]["temperature"] == 0 and b["logprobs"] is True
    assert "A. safe" in b["messages"][1]["content"]


def test_ollama_without_logprobs_reports_no_confidence():
    r = _ollama(lambda request: _chat("A")).decide(req())
    assert r.label == "safe" and r.confidence is None and r.metadata["logprobs"] == "unavailable"


def test_ollama_malformed_answer_retries_then_errors():
    calls = []

    def handler(request):
        calls.append(1)
        return _chat("Sure! The answer is")

    r = _ollama(handler, max_retries=1).decide(req())
    assert r.status == "error" and r.error.code == "MALFORMED_OUTPUT" and len(calls) == 2


def test_ollama_retry_recovers():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(500) if len(calls) == 1 else _chat("A")

    assert _ollama(handler).decide(req()).label == "safe"


def test_ollama_timeout_and_unreachable():
    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    def down(request):
        raise httpx.ConnectError("refused", request=request)

    assert _ollama(slow).decide(req()).status == "timeout"
    r = _ollama(down).decide(req())
    assert r.status == "unavailable" and r.label is None


def test_ollama_health():
    ok = _ollama(lambda r: httpx.Response(200, json={"models": [{"name": "llama3.1:8b"}]})).health()
    assert ok.available
    missing = _ollama(lambda r: httpx.Response(200, json={"models": []})).health()
    assert not missing.available and "not pulled" in missing.detail

    def down(request):
        raise httpx.ConnectError("refused", request=request)
    assert not _ollama(down).health().available


@pytest.mark.ollama
def test_ollama_live():  # optional: needs a running server with the model pulled
    p = OllamaDecisionProvider("llm", timeout_s=300)
    if not p.health().available:
        pytest.skip("ollama not available")
    r = p.decide(req(action_type="delete_database", source_system="infra-agent", target_resource="prod/orders",
                     action_payload={"database": "orders"}))
    assert r.status == "ok" and r.label in DISPOSITION_QUESTION.categories


# ------------------------------------------------------------------ vegaml (VegaCore + test double; no inference)

def _vega_provider(respond=None, **kw):
    from action_sentinel.vegalab.testing import FakeVegaRunner
    from action_sentinel.vegalab.vega_core import VegaCore
    runner = FakeVegaRunner(respond) if respond else FakeVegaRunner(**kw)
    return VegaMLDecisionProvider("vega", core=VegaCore(runner), adapter_mode="off"), runner


def test_vegaml_provider_maps_native_signals():
    p, runner = _vega_provider()
    r = p.decide(req())
    assert r.status == "ok" and r.label == "safe" and r.confidence == 0.7  # FakeVegaRunner: first option wins
    assert r.confidence_source == "model_probability" and r.calibration_status == "unknown"
    assert r.prediction_set == ["safe"] and abs(sum(r.probabilities.values()) - 1) < 1e-9
    assert r.metadata["test_double"] is True and r.metadata["live_inference"] is False
    state, questions, mode = runner.calls[0]
    assert mode == "off" and state["action_type"] == "restart_service" and state["evidence"]
    assert set(questions["disposition"]["criteria"]) == {"safe", "needs_review", "unsafe"}


@pytest.mark.parametrize("flag,reason", [("abstain", "VEGAML_ABSTAIN"), ("unbound", "VEGAML_UNBOUND")])
def test_vegaml_abstain_and_unbound_become_abstention(flag, reason):
    from action_sentinel.vegalab.testing import FakeVegaRunner

    def respond(state, questions, mode):
        out = FakeVegaRunner.default_response(state, questions, mode)
        out["answers"]["disposition"][flag] = True
        return out
    p, _ = _vega_provider(respond)
    r = p.decide(req())
    assert r.status == "abstained" and r.label == "safe" and r.metadata["abstain_reason"] == reason


def test_vegaml_failures_are_typed():
    p, _ = _vega_provider(raise_on_decide=RuntimeError("kernel died"))
    r = p.decide(req())
    assert r.status == "error" and r.error.code == "INFERENCE_FAILED" and r.label is None


def test_vegaml_unavailable_fails_safe(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    p = VegaMLDecisionProvider("vega")
    r = p.decide(req())
    assert r.status == "unavailable" and r.error.code == "NOT_INSTALLED"
    assert not p.health().available


# ------------------------------------------------------------------ registry and runner

def test_registry_builds_from_config_and_rejects_bad_config():
    reg = ProviderRegistry()
    names = reg.build([
        {"name": "rules", "type": "rule_based", "config": {"classifier_path": "examples/rule_packs/classifier.yaml"}},
        {"name": "mock-a", "type": "mock"},
        {"name": "llm", "type": "ollama", "enabled": False},
    ], ROOT)
    assert names == ["rules", "mock-a"] and reg.get("rules").kind == "deterministic"
    assert {h.name for h in reg.health()} == {"rules", "mock-a"}
    with pytest.raises(ProviderConfigError):
        reg.get("llm")
    for bad in ([{"name": "x", "type": "quantum"}], [{"name": "Bad Name", "type": "mock"}],
                [{"name": "y", "type": "ollama", "config": {"timeout_s": -1}}],
                [{"name": "z", "type": "rule_based", "config": {"classifier_path": "nope.yaml"}}],
                [{"name": "rules", "type": "mock"}]):
        with pytest.raises(ProviderConfigError):
            reg.build(bad, ROOT)


def test_registry_accepts_new_provider_types():
    class Constant(DecisionProvider):
        provider_type = "constant"

        def _decide(self, request):
            return RawAnswer(label="needs_review")

    reg = ProviderRegistry()
    reg.register_type("constant", lambda spec, base: Constant(spec.name))
    reg.build([{"name": "const", "type": "constant"}], ROOT)
    assert reg.get("const").decide(req()).label == "needs_review"


def test_runner_strategies_and_enforced_timeout():
    fast_conf = MockDecisionProvider("a", default={"label": "safe", "confidence": 0.95})
    low_conf = MockDecisionProvider("b", default={"label": "safe", "confidence": 0.4})
    slow = MockDecisionProvider("c", default={"label": "safe", "confidence": 0.9, "sleep_s": 3.0})
    # Generous margin: the fast provider must finish well inside the timeout even on a loaded machine.
    runner = ProviderRunner(timeout_s=1.0, confident_at=0.85)
    try:
        assert [r.provider_name for r in runner.run([fast_conf, low_conf], req(), "single")] == ["a"]
        assert [r.provider_name for r in runner.run([fast_conf, low_conf], req(), "sequential")] == ["a"]
        assert [r.provider_name for r in runner.run([low_conf, fast_conf], req(), "sequential")] == ["b", "a"]
        res = runner.run([fast_conf, slow], req(), "parallel")
        assert [r.status for r in res] == ["ok", "timeout"]
        assert res[1].error.code == "TIMEOUT" and res[1].label is None
        assert runner.run([], req()) == []
    finally:
        runner.close()
