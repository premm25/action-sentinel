"""Scripted test double. Every result it produces is marked `is_test_output=True`.

Behaviour is looked up by the request's event_id and falls back to `default`. Scripts are supplied
by the caller (tests, the demo, Confidence Lab) and never travel inside the proposal, so a real
provider evaluated alongside the mock cannot see them. A behaviour is a mapping:

    {"label": "safe", "confidence": 0.9}          answer
    {"label": "safe", "probabilities": {...}}     answer with a distribution
    {"abstain": true}                             abstain
    {"fail": "message"}                           provider error
    {"timeout": true}                             simulated timeout (no real waiting)
    {"malformed": true}                           returns a label outside the categories
    {"sleep_s": 0.2, ...}                         really sleep first (for timeout-enforcement tests)
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from action_sentinel.contracts import DecisionRequest

from .base import DecisionProvider, ProviderFailure, RawAnswer


class MockDecisionProvider(DecisionProvider):
    provider_type = "mock"
    kind = "test_double"

    def __init__(self, name: str, script: Mapping[str, Mapping[str, Any]] | None = None,
                 default: Mapping[str, Any] | None = None):
        super().__init__(name, "mock-1")
        self.script = dict(script or {})
        self.default = dict(default or {"abstain": True})
        self.calls: list[str] = []

    def behaviour_for(self, request: DecisionRequest) -> Mapping[str, Any]:
        return self.script.get(request.action.event_id, self.default)

    def _decide(self, request: DecisionRequest) -> RawAnswer:
        self.calls.append(request.action.event_id)
        b = self.behaviour_for(request)
        if b.get("sleep_s"):
            time.sleep(float(b["sleep_s"]))
        if b.get("timeout"):
            raise ProviderFailure("TIMEOUT", "simulated timeout", status="timeout")
        if b.get("fail"):
            raise ProviderFailure("SIMULATED_FAILURE", str(b["fail"]))
        if b.get("malformed"):
            return RawAnswer(label="__not_a_category__")
        conf = b.get("confidence")
        probs = b.get("probabilities")
        if probs is not None and conf is None and b.get("label") is not None:
            conf = probs.get(b["label"])
        return RawAnswer(
            label=b.get("label"), confidence=conf,
            confidence_source="model_probability" if conf is not None else "none",
            probabilities=dict(probs) if probs else None,
            prediction_set=list(b["prediction_set"]) if b.get("prediction_set") else None,
            abstain=bool(b.get("abstain")), abstain_reason="SCRIPTED" if b.get("abstain") else None,
            metadata={"scripted": True})
