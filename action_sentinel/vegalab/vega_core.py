"""VegaCore: the adapter around the real vegaml package. It does not reimplement any part of VegaML.

    core = VegaCore(LiveVegaRunner("0.8b", device="cpu"))   # real model
    result = core.decide(request, adapter_mode="off")

`VegaRunner` is the seam for dependency injection. `LiveVegaRunner` is the only implementation that
touches vegaml. Test doubles implement the same protocol and must report `live = False`, which is
copied into every result, so a mock result can never be presented as live inference.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any, Protocol

from pydantic import ValidationError

from .contracts import (
    AdapterMode,
    DecisionQuestion,
    DecisionRequest,
    DecisionResult,
    InferenceError,
    QuestionDecision,
    UncertaintyMetadata,
    VegaInferenceError,
)


class VegaRunner(Protocol):
    live: bool

    def load(self) -> None: ...
    def info(self) -> dict[str, Any]: ...
    def set_adapter_mode(self, mode: AdapterMode) -> None: ...
    def decide(self, state: Any, questions: dict[str, Any], *, alpha: str, risk: str | None,
               views: int) -> dict[str, Any]: ...


class LiveVegaRunner:
    """Loads a real VegaML checkpoint once and calls its engine readout.

    Loads with adapters="auto" so both modes are available, then switches with
    `engine.set_adapters(...)`. Per vegaml 0.9.0, mode "off" leaves the base engine exactly unchanged
    (gates are None, so every adapted layer returns its base output) and uses the base calibration.
    `decide` calls `Vega.engine.decide`, which is exactly what `Vega.decide(mode="engine")` calls,
    because only the engine method accepts `risk`.
    """

    live = True

    def __init__(self, model: str = "0.8b", device: str | None = "cpu", revision: str | None = None):
        self.model = model
        self.device = device
        self.revision = revision
        self._vega: Any = None
        self._mode: AdapterMode | None = None
        self.load_seconds: float | None = None

    @staticmethod
    def available() -> tuple[bool, str]:
        if importlib.util.find_spec("vegaml") is None:
            return False, "vegaml is not installed"
        mod = importlib.import_module("vegaml")
        return True, "vegaml %s" % getattr(mod, "__version__", "unknown")

    def load(self) -> None:
        if self._vega is not None:
            return
        ok, detail = self.available()
        if not ok:
            raise VegaInferenceError(InferenceError(code="MODEL_LOAD_FAILED", message=detail))
        vegaml = importlib.import_module("vegaml")
        t0 = time.perf_counter()
        kw: dict[str, Any] = {"mode": "engine", "adapters": "auto"}
        if self.device:
            kw["device"] = self.device
        if self.revision:
            kw["revision"] = self.revision
        try:
            self._vega = vegaml.load(self.model, **kw)
        except Exception as exc:
            raise VegaInferenceError(InferenceError(
                code="MODEL_LOAD_FAILED", message="%s: %s" % (type(exc).__name__, str(exc)[:1000]),
                retryable=True)) from exc
        self.load_seconds = round(time.perf_counter() - t0, 2)
        self._mode = "auto"

    def info(self) -> dict[str, Any]:
        if self._vega is None:
            return {"loaded": False, "model": self.model, "device": self.device}
        import torch
        import transformers
        import vegaml
        eng = self._vega.engine
        return {
            "loaded": True, "model": self.model, "device": str(eng.device), "vegaml": vegaml.__version__,
            "torch": torch.__version__, "transformers": transformers.__version__, "model_dir": str(eng.model_dir),
            "backbone": str(eng.backbone), "feature_layers": eng.cfg.get("feature_layers"),
            "option_layers": eng.cfg.get("option_layers"), "n_steps": eng.cfg.get("n_steps"),
            "max_len": eng.cfg.get("max_len"), "default_risk": eng.risk,
            "adapters": [a["name"] for a in eng.adapters], "adapter_mode": self._mode,
            "engine_sha256": eng.base_sha256(), "load_seconds": self.load_seconds,
        }

    def set_adapter_mode(self, mode: AdapterMode) -> None:
        if self._vega is None:
            raise VegaInferenceError(InferenceError(code="MODEL_NOT_READY", message="model not loaded"))
        if mode != self._mode:
            self._vega.engine.set_adapters(mode)
            self._mode = mode

    def decide(self, state: Any, questions: dict[str, Any], *, alpha: str, risk: str | None,
               views: int) -> dict[str, Any]:
        if self._vega is None:
            raise VegaInferenceError(InferenceError(code="MODEL_NOT_READY", message="model not loaded"))
        out = self._vega.engine.decide(state, questions, alpha=alpha, risk=risk, views=views)
        out["mode"] = "engine"
        return out


class VegaCore:
    """Validates requests, calls the runner under a lock, normalizes the native output without losing it."""

    def __init__(self, runner: VegaRunner, model_id: str = "vegaml-0.8b", timeout_s: float = 300.0):
        self.runner = runner
        self.model_id = model_id
        self.timeout_s = timeout_s
        self._lock = threading.Lock()   # one decision at a time: vegaml keeps per-call state on the model
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vega")

    @property
    def live(self) -> bool:
        return bool(getattr(self.runner, "live", False))

    def decide(self, request: DecisionRequest | dict[str, Any],
               adapter_mode: AdapterMode = "auto") -> DecisionResult:
        request = self.validate(request)
        if adapter_mode not in ("off", "auto", "on"):
            raise VegaInferenceError(InferenceError(request_id=request.request_id, code="INVALID_INPUT",
                                                    message="adapter_mode must be off, auto or on"))
        state, questions = request.vega_state(), request.vega_questions()
        t0 = time.perf_counter()
        with self._lock:
            fut = self._pool.submit(self._call, state, questions, request, adapter_mode)
            try:
                native = fut.result(timeout=self.timeout_s)
            except FutureTimeout:
                raise VegaInferenceError(InferenceError(
                    request_id=request.request_id, code="TIMEOUT", retryable=True,
                    message="no result within %.0f s; the call keeps running in the background and the "
                            "next request waits for it" % self.timeout_s)) from None
            except VegaInferenceError as exc:
                exc.error = exc.error.model_copy(update={"request_id": request.request_id})
                raise
            except Exception as exc:
                msg = "%s: %s" % (type(exc).__name__, str(exc)[:1000])
                code = "INPUT_TOO_LONG" if "too long" in str(exc).lower() or "max_len" in str(exc) else \
                    "INFERENCE_FAILED"
                raise VegaInferenceError(InferenceError(request_id=request.request_id, code=code,
                                                        message=msg)) from exc
        wall = round(1000 * (time.perf_counter() - t0), 2)
        return self.normalize(request, native, adapter_mode, wall)

    @staticmethod
    def validate(request: DecisionRequest | dict[str, Any]) -> DecisionRequest:
        """Validate before inference; invalid input never reaches the model."""
        if isinstance(request, DecisionRequest):
            return request
        try:
            return DecisionRequest.model_validate(request)
        except ValidationError as exc:
            detail = "; ".join("%s: %s" % (".".join(str(x) for x in e["loc"]), e["msg"]) for e in exc.errors())
            raise VegaInferenceError(InferenceError(code="INVALID_INPUT", message=detail[:4000])) from exc

    def _call(self, state, questions, request: DecisionRequest, adapter_mode: AdapterMode) -> dict[str, Any]:
        self.runner.set_adapter_mode(adapter_mode)
        return self.runner.decide(state, questions, alpha=request.alpha, risk=request.risk, views=request.views)

    def normalize(self, request: DecisionRequest, native: dict[str, Any], adapter_mode: AdapterMode,
                  wall_ms: float) -> DecisionResult:
        answers = native.get("answers") if isinstance(native, dict) else None
        if not isinstance(answers, dict):
            raise self._malformed(request, "native output has no 'answers' mapping")
        decisions = {}
        info = {}
        try:
            info = self.runner.info()
        except Exception:
            info = {}
        for q in request.questions:
            a = answers.get(q.question_id)
            if not isinstance(a, dict):
                raise self._malformed(request, "no answer for question %r" % q.question_id)
            decisions[q.question_id] = self._question(request, q, a, info)
        return DecisionResult(
            request_id=request.request_id, engine_version=str(info.get("vegaml", "unknown")),
            model_id=self.model_id, adapter_mode=adapter_mode, live_inference=self.live, decisions=decisions,
            wall_ms=wall_ms, native_latency_ms=_num(native.get("latency_ms")),
            n_tokens=int(native["n_tokens"]) if isinstance(native.get("n_tokens"), int) else None,
            native=native)

    def _question(self, request: DecisionRequest, q: DecisionQuestion, a: dict[str, Any],
                  info: dict[str, Any]) -> QuestionDecision:
        labels = q.labels
        probs_raw = a.get("probs")
        probs = None
        if isinstance(probs_raw, dict):
            probs = {}
            for k, v in probs_raw.items():
                lab = _label(q, k)
                if lab not in labels:
                    raise self._malformed(request, "probability for unknown option %r in %s" % (k, q.question_id))
                p = _num(v)
                if p is None or not (0.0 <= p <= 1.0):
                    raise self._malformed(request, "invalid probability %r" % v)
                probs[lab] = p
            if set(probs) != set(labels):
                raise self._malformed(request, "probabilities do not cover every option of %s" % q.question_id)
        if q.type == "choice":
            sel = a.get("choice")
            selected = str(sel) if sel is not None else None
        elif q.type == "score":
            sel = a.get("score")
            selected = str(int(sel)) if isinstance(sel, (int, float)) else None
        else:
            # vegaml has no "choice" for booleans; its readout is argmax over [false, true] (ties -> false).
            if probs is not None:
                selected = "true" if probs["true"] > probs["false"] else "false"
            else:
                pb = _num(a.get("boolean"))
                selected = None if pb is None else ("true" if pb > 0.5 else "false")
        if selected is None or selected not in labels:
            raise self._malformed(request, "missing or unknown selected option for %s" % q.question_id)
        pset = a.get("prediction_set")
        unc = UncertaintyMetadata(
            confidence=_num(a.get("confidence")), probabilities=probs,
            prediction_set=[_label(q, x) for x in pset] if isinstance(pset, list) else None,
            prediction_set_alpha=request.alpha if isinstance(pset, list) else None,
            abstain=a.get("abstain") if isinstance(a.get("abstain"), bool) else None,
            abstain_risk=request.risk or info.get("default_risk"),
            unbound=a.get("unbound") if isinstance(a.get("unbound"), bool) else None,
            settle_steps=a.get("settle_steps") if isinstance(a.get("settle_steps"), int) else None,
            energy=_num(a.get("energy")), view_agreement=_num(a.get("view_agreement")),
            adapter=a.get("adapter") if isinstance(a.get("adapter"), str) else None,
            gate=_num(a.get("gate")),
            calibration_source="vegaml_checkpoint" if probs is not None else "none")
        return QuestionDecision(
            question_id=q.question_id, type=q.type, selected_label=selected,
            score=int(a["score"]) if q.type == "score" and isinstance(a.get("score"), (int, float)) else None,
            expected_score=_num(a.get("expected_score")),
            boolean_probability=_num(a.get("boolean")) if q.type == "boolean" else None, uncertainty=unc)

    @staticmethod
    def _malformed(request: DecisionRequest, msg: str) -> VegaInferenceError:
        return VegaInferenceError(InferenceError(request_id=request.request_id, code="MALFORMED_NATIVE_OUTPUT",
                                                 message=msg))

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _label(q: DecisionQuestion, raw: Any) -> str:
    """Map a native label to ours. Boolean answers come back as True/False or "True"/"False"."""
    if q.type == "boolean":
        if raw in (True, "True", "true", 1, "1"):
            return "true"
        if raw in (False, "False", "false", 0, "0"):
            return "false"
    return str(raw)
