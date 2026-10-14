"""The DecisionProvider interface and the normalization every provider goes through."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from action_sentinel.contracts import DecisionRequest, ProviderError, ProviderHealth, ProviderResult


@dataclass
class RawAnswer:
    """What a provider implementation returns before normalization. Unknown values stay None."""

    label: str | None = None
    confidence: float | None = None
    confidence_source: Literal["model_probability", "self_reported", "none"] = "none"
    probabilities: dict[str, float] | None = None
    prediction_set: list[str] | None = None
    abstain: bool = False
    abstain_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ProviderFailure(Exception):
    """Raised by implementations for an expected, classified failure."""

    def __init__(self, code: str, message: str, status: Literal["error", "timeout", "unavailable"] = "error"):
        super().__init__(message)
        self.code = code
        self.status = status


class DecisionProvider(ABC):
    """Accepts a DecisionRequest, returns a normalized ProviderResult. Never raises from `decide`."""

    provider_type: str = "abstract"
    kind: Literal["deterministic", "statistical", "test_double"] = "statistical"

    def __init__(self, name: str, version: str = "unknown"):
        self.name = name
        self.version = version
        # Calibration measured for this provider on Sentinel data. None means unknown.
        self.calibration: dict[str, Any] | None = None

    @abstractmethod
    def _decide(self, request: DecisionRequest) -> RawAnswer:
        ...

    def health(self) -> ProviderHealth:
        return ProviderHealth(name=self.name, provider_type=self.provider_type, version=self.version,
                              kind=self.kind, available=True, detail="")

    def decide(self, request: DecisionRequest) -> ProviderResult:
        started = time.perf_counter()
        try:
            raw = self._decide(request)
        except ProviderFailure as exc:
            return self._failure(request, exc.status, exc.code, str(exc), started)
        except Exception as exc:  # an unexpected bug in a provider is a failed result, never an approval
            return self._failure(request, "error", "PROVIDER_EXCEPTION", "%s: %s" % (type(exc).__name__, exc),
                                 started)
        return self.normalize(request, raw, started)

    # ------------------------------------------------------------------ helpers
    def _base(self, request: DecisionRequest, started: float) -> dict[str, Any]:
        return {
            "request_id": request.request_id, "provider_name": self.name, "provider_version": self.version,
            "provider_kind": self.kind, "latency_ms": round(1000 * (time.perf_counter() - started), 3),
            "is_test_output": self.kind == "test_double",
            "calibration_status": "not_applicable" if self.kind == "deterministic"
            else ("measured" if self.calibration else "unknown"),
            "calibration_ref": (self.calibration or {}).get("ref"),
        }

    def _failure(self, request: DecisionRequest, status: str, code: str, message: str,
                 started: float) -> ProviderResult:
        return ProviderResult(status=status, error=ProviderError(code=code, message=message[:2000]),
                              **self._base(request, started))

    def normalize(self, request: DecisionRequest, raw: RawAnswer, started: float) -> ProviderResult:
        """Check the raw answer against the question; anything malformed becomes an error result."""
        cats = request.question.categories
        base = self._base(request, started)
        try:
            if raw.label is not None and raw.label not in cats:
                raise ValueError("label %r is not one of the allowed categories" % raw.label)
            probs = raw.probabilities
            if probs is not None:
                unknown = set(probs) - set(cats)
                if unknown:
                    raise ValueError("probabilities name unknown categories %s" % sorted(unknown))
            pset = raw.prediction_set
            if pset is not None and not set(pset) <= set(cats):
                raise ValueError("prediction set names unknown categories")
            if raw.abstain:
                return ProviderResult(
                    status="abstained", label=raw.label, confidence=raw.confidence,
                    confidence_source=raw.confidence_source if raw.confidence is not None else "none",
                    probabilities=probs, prediction_set=pset,
                    metadata={**raw.metadata, "abstain_reason": raw.abstain_reason}, **base)
            if raw.label is None:
                raise ValueError("no label and no abstention")
            return ProviderResult(
                status="ok", label=raw.label, confidence=raw.confidence,
                confidence_source=raw.confidence_source if raw.confidence is not None else "none",
                probabilities=probs, prediction_set=pset, metadata=raw.metadata, **base)
        except ValueError as exc:  # includes pydantic ValidationError
            return ProviderResult(status="error", error=ProviderError(code="MALFORMED_OUTPUT",
                                                                      message=str(exc)[:2000]), **base)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "type": self.provider_type, "version": self.version, "kind": self.kind,
                "calibration": self.calibration}


def render_situation(request: DecisionRequest) -> str:
    """A plain-text rendering of the proposal for language-model providers."""
    a = request.action
    lines = [
        "Agent: %s (version %s) from source system %s" % (a.agent_id, a.agent_version, a.source_system),
        "Action type: %s" % a.action_type,
        "Target resource: %s" % a.target_resource,
        "Payload: %s" % _compact(a.action_payload),
    ]
    if a.context:
        lines.append("Context: %s" % _compact(a.context))
    if a.evidence_refs:
        lines.append("Evidence:")
        for e in a.evidence_refs:
            lines.append("  - [%s, stance=%s] %s" % (e.kind, e.stance, e.description or "(no description)"))
    else:
        lines.append("Evidence: none provided")
    return "\n".join(lines)


def _compact(value: Any) -> str:
    import json
    text = json.dumps(value, sort_keys=True, default=str)
    return text if len(text) <= 2000 else text[:2000] + "...(truncated)"
