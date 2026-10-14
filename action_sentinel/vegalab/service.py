"""VegaDecisionService: one resident model, readiness states, request accounting.

The model is loaded once (eagerly at startup or lazily on first use) and shared by all requests;
VegaCore serializes calls to it.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from typing import Any, Literal

from .contracts import AdapterMode, DecisionRequest, DecisionResult, InferenceError, VegaInferenceError
from .vega_core import VegaCore, VegaRunner

ReadyState = Literal["not_loaded", "loading", "ready", "failed"]


class VegaDecisionService:
    def __init__(self, runner: VegaRunner, model_id: str = "vegaml-0.8b", default_adapter_mode: AdapterMode = "auto",
                 lazy_load: bool = True, timeout_s: float = 300.0):
        self.runner = runner
        self.core = VegaCore(runner, model_id=model_id, timeout_s=timeout_s)
        self.model_id = model_id
        self.default_adapter_mode = default_adapter_mode
        self.lazy_load = lazy_load
        self.state: ReadyState = "not_loaded"
        self.load_error: InferenceError | None = None
        self._load_lock = threading.Lock()
        self.requests = 0
        self.errors: Counter[str] = Counter()
        self._latencies: deque[float] = deque(maxlen=1000)

    def load(self) -> ReadyState:
        with self._load_lock:
            if self.state == "ready":
                return self.state
            self.state = "loading"
            try:
                self.runner.load()
            except VegaInferenceError as exc:
                self.state, self.load_error = "failed", exc.error
                return self.state
            except Exception as exc:
                self.state = "failed"
                self.load_error = InferenceError(code="MODEL_LOAD_FAILED", message="%s: %s" % (type(exc).__name__, exc))
                return self.state
            self.state, self.load_error = "ready", None
            return self.state

    def ready(self) -> dict[str, Any]:
        return {"state": self.state, "ready": self.state == "ready", "live": self.core.live,
                "error": self.load_error.model_dump() if self.load_error else None}

    def info(self) -> dict[str, Any]:
        lat = sorted(self._latencies)

        def pct(q: float) -> float | None:
            return round(lat[min(len(lat) - 1, int(q * (len(lat) - 1)))], 1) if lat else None

        return {"model_id": self.model_id, "state": self.state, "live_inference": self.core.live,
                "default_adapter_mode": self.default_adapter_mode, "runner": self.runner.info(),
                "requests": self.requests, "errors": dict(self.errors),
                "latency_ms": {"p50": pct(0.5), "p95": pct(0.95), "n": len(lat)}}

    def decide(self, request: DecisionRequest | dict[str, Any],
               adapter_mode: AdapterMode | None = None) -> DecisionResult:
        self.requests += 1
        try:
            if self.state != "ready":
                if self.lazy_load and self.state in ("not_loaded", "failed"):
                    self.load()
                if self.state != "ready":
                    raise VegaInferenceError(self.load_error or InferenceError(
                        code="MODEL_NOT_READY", message="model state is %s" % self.state, retryable=True))
            t0 = time.perf_counter()
            res = self.core.decide(request, adapter_mode or self.default_adapter_mode)
            self._latencies.append(1000 * (time.perf_counter() - t0))
            return res
        except VegaInferenceError as exc:
            self.errors[exc.error.code] += 1
            raise

    def close(self) -> None:
        self.core.close()
