"""Test double for VegaRunner. NOT inference: it returns scripted dictionaries shaped like vegaml 0.9.0 output.

`live = False` is propagated into every DecisionResult, so its outputs can never be mistaken for a
real model call.
"""

from __future__ import annotations

from typing import Any, Callable

from .contracts import AdapterMode


class FakeVegaRunner:
    live = False

    def __init__(self, respond: Callable[[Any, dict[str, Any], AdapterMode], dict[str, Any]] | None = None,
                 fail_load: str | None = None, raise_on_decide: BaseException | None = None):
        self.respond = respond or self.default_response
        self.fail_load = fail_load
        self.raise_on_decide = raise_on_decide
        self.loaded = False
        self.mode: AdapterMode = "auto"
        self.calls: list[tuple[Any, dict[str, Any], AdapterMode]] = []

    def load(self) -> None:
        from .contracts import InferenceError, VegaInferenceError
        if self.fail_load:
            raise VegaInferenceError(InferenceError(code="MODEL_LOAD_FAILED", message=self.fail_load))
        self.loaded = True

    def info(self) -> dict[str, Any]:
        return {"loaded": self.loaded, "model": "fake", "vegaml": "fake", "default_risk": "0.1",
                "adapter_mode": self.mode, "test_double": True}

    def set_adapter_mode(self, mode: AdapterMode) -> None:
        self.mode = mode

    def decide(self, state, questions, *, alpha, risk, views):
        self.calls.append((state, questions, self.mode))
        if self.raise_on_decide is not None:
            raise self.raise_on_decide
        return self.respond(state, questions, self.mode)

    @staticmethod
    def default_response(state, questions, mode) -> dict[str, Any]:
        """First option wins with p=0.7, the rest share 0.3; shaped like vegaml's native answer."""
        answers = {}
        for qid, q in questions.items():
            t = q["type"]
            if t == "choice":
                labels = list(q["criteria"])
            elif t == "score":
                labels = list(range(len(q["criteria"])))
            else:
                labels = [False, True]
            k = len(labels)
            probs = [0.7] + [0.3 / (k - 1)] * (k - 1)
            a = {"probs": {str(lab): p for lab, p in zip(labels, probs, strict=True)}, "confidence": 0.7,
                 "prediction_set": [labels[0]], "abstain": False, "unbound": False, "settle_steps": 12,
                 "energy": -1.0}
            if t == "choice":
                a["choice"] = labels[0]
            elif t == "score":
                a["score"], a["expected_score"] = 0, float(sum(i * p for i, p in enumerate(probs)))
            else:
                a["boolean"] = probs[1]
            answers[qid] = a
        return {"answers": answers, "latency_ms": 1.0, "n_tokens": 10, "views": 1}
