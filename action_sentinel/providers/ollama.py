"""Optional local LLM provider via an Ollama server.

The model is shown the situation and the categories as lettered options and asked to answer with a
single letter. The answer distribution is read from the token log-probabilities of that one
generated token, renormalized over the option letters. The result is a model probability, but its
calibration is unknown until Confidence Lab measures it. If the server does not return logprobs,
the provider falls back to the generated letter with no confidence, and never invents one.

Failure of any kind (server down, timeout, unparseable answer) becomes an error result.
"""

from __future__ import annotations

import math
import string
from typing import Any

import httpx

from action_sentinel.contracts import DecisionRequest

from .base import DecisionProvider, ProviderFailure, RawAnswer, render_situation

SYSTEM_PROMPT = (
    "You are a careful reviewer of actions proposed by automated agents. You answer with exactly one "
    "capital letter naming the best option, and nothing else."
)


class OllamaDecisionProvider(DecisionProvider):
    provider_type = "ollama"
    kind = "statistical"

    def __init__(self, name: str, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434",
                 timeout_s: float = 120.0, max_retries: int = 1, top_logprobs: int = 10,
                 transport: httpx.BaseTransport | None = None):
        super().__init__(name, "ollama:%s" % model)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max(0, int(max_retries))
        self.top_logprobs = top_logprobs
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout_s, transport=transport)

    # ------------------------------------------------------------------ health
    def health(self):
        h = super().health()
        try:
            r = self._client.get("/api/tags", timeout=min(5.0, self.timeout_s))
            r.raise_for_status()
            names = {m.get("name") for m in r.json().get("models", [])}
        except (httpx.HTTPError, ValueError) as exc:
            return h.model_copy(update={"available": False, "detail": "ollama unreachable: %s" % type(exc).__name__})
        if self.model not in names:
            return h.model_copy(update={"available": False, "detail": "model %s not pulled" % self.model})
        return h.model_copy(update={"detail": "model %s available at %s" % (self.model, self.base_url)})

    # ------------------------------------------------------------------ prompting
    @staticmethod
    def letters(request: DecisionRequest) -> dict[str, str]:
        cats = list(request.question.categories)
        if len(cats) > 26:
            raise ProviderFailure("TOO_MANY_OPTIONS", "at most 26 categories are supported")
        return {string.ascii_uppercase[i]: c for i, c in enumerate(cats)}

    def build_prompt(self, request: DecisionRequest) -> str:
        q = request.question
        opts = "\n".join("%s. %s: %s" % (letter, cat, q.categories[cat])
                         for letter, cat in self.letters(request).items())
        return ("%s\n\nOptions:\n%s\n\nSituation:\n%s\n\nAnswer with the single letter of the best option."
                % (q.instructions, opts, render_situation(request)))

    def _decide(self, request: DecisionRequest) -> RawAnswer:
        letters = self.letters(request)
        body = {
            "model": self.model, "stream": False, "logprobs": True, "top_logprobs": self.top_logprobs,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": self.build_prompt(request)}],
            "options": {"temperature": 0, "num_predict": 1, "seed": 0},
        }
        last: ProviderFailure | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.post("/api/chat", json=body)
                resp.raise_for_status()
                data = resp.json()
            except httpx.TimeoutException as exc:
                # A timeout is not retried: the next attempt would most likely time out as well.
                raise ProviderFailure("TIMEOUT", "no answer within %.0f s" % self.timeout_s, "timeout") from exc
            except httpx.ConnectError as exc:
                raise ProviderFailure("UNAVAILABLE", "cannot reach %s" % self.base_url, "unavailable") from exc
            except (httpx.HTTPError, ValueError) as exc:
                last = ProviderFailure("HTTP_ERROR", "%s: %s" % (type(exc).__name__, exc))
                continue
            try:
                return self._parse(data, letters, attempt)
            except ProviderFailure as exc:
                last = exc  # malformed answer: retry once with the same deterministic settings
        assert last is not None
        raise last

    @staticmethod
    def _parse(data: dict[str, Any], letters: dict[str, str], attempt: int) -> RawAnswer:
        content = str((data.get("message") or {}).get("content", "")).strip()
        meta: dict[str, Any] = {"raw_answer": content[:20], "attempt": attempt + 1}
        lp = data.get("logprobs")
        probs = None
        if isinstance(lp, list) and lp and isinstance(lp[0], dict) and lp[0].get("top_logprobs"):
            mass: dict[str, float] = {}
            for alt in lp[0]["top_logprobs"]:
                tok = str(alt.get("token", "")).strip().rstrip(".:)")
                if tok in letters:
                    mass[letters[tok]] = mass.get(letters[tok], 0.0) + math.exp(float(alt["logprob"]))
            covered = sum(mass.values())
            if covered > 0:
                probs = {cat: mass.get(cat, 0.0) / covered for cat in letters.values()}
                meta["option_mass"] = round(covered, 4)  # probability mass the options received
        answer = content[:1].upper() if content else ""
        if probs is not None:
            label = max(probs, key=probs.get)
            return RawAnswer(label=label, confidence=probs[label], confidence_source="model_probability",
                             probabilities=probs, metadata=meta)
        if answer in letters:
            meta["logprobs"] = "unavailable"
            return RawAnswer(label=letters[answer], metadata=meta)
        raise ProviderFailure("MALFORMED_OUTPUT", "answer %r is not an option letter" % content[:40])
