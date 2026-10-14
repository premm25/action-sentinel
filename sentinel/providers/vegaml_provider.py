"""Optional VegaML provider.

Written against the public API of `vegaml` 0.9.0 as inspected in its source (NandhaKishorM/vegaml,
commit 7803fcf):

    v = vegaml.load(repo_or_path=None, device=..., mode=...)
    out = v.decide(state, {qid: {"type": "choice", "instructions": str, "criteria": {label: desc}}},
                   mode="engine", alpha="0.1")
    out["answers"][qid] -> {"choice", "probs", "confidence", "prediction_set", "abstain", "unbound", ...}

`mode="engine"` is requested so every answer carries VegaML's abstain flag and conformal set. VegaML's
own calibration was fitted by its authors on their data, not on Sentinel's, so calibration is
reported here as `unknown` until measured on Sentinel data.

Importing this module never imports vegaml; the package is loaded on first use.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

from sentinel.contracts import DecisionRequest

from .base import DecisionProvider, ProviderFailure, RawAnswer, render_situation

SUPPORTED_MAJOR_MINOR = ((0, 9),)


class VegaMLDecisionProvider(DecisionProvider):
    provider_type = "vegaml"
    kind = "statistical"

    def __init__(self, name: str, checkpoint: str | None = None, device: str | None = None, alpha: str = "0.1"):
        super().__init__(name, "vegaml:unloaded")
        self.checkpoint = checkpoint
        self.device = device
        self.alpha = alpha
        self._model: Any = None
        self._load_error: str | None = None

    def _module(self):
        if importlib.util.find_spec("vegaml") is None:
            raise ProviderFailure("NOT_INSTALLED", "vegaml is not installed (pip install vegaml)", "unavailable")
        return importlib.import_module("vegaml")

    def health(self):
        h = super().health()
        try:
            mod = self._module()
        except ProviderFailure as exc:
            return h.model_copy(update={"available": False, "detail": str(exc)})
        ver = getattr(mod, "__version__", "unknown")
        parts = tuple(int(x) for x in ver.split(".")[:2] if x.isdigit())
        detail = "vegaml %s importable" % ver
        if parts not in SUPPORTED_MAJOR_MINOR:
            detail += " (untested version; adapter verified against 0.9.x)"
        if self._load_error:
            return h.model_copy(update={"available": False, "detail": "load failed: %s" % self._load_error})
        return h.model_copy(update={"version": "vegaml:%s" % ver, "detail": detail + (
            "; model loaded" if self._model is not None else "; model loads on first decision")})

    def _load(self):
        if self._model is not None:
            return self._model
        mod = self._module()
        kwargs: dict[str, Any] = {"mode": "engine"}
        if self.device:
            kwargs["device"] = self.device
        try:
            self._model = mod.load(self.checkpoint, **kwargs)
        except Exception as exc:
            self._load_error = "%s: %s" % (type(exc).__name__, str(exc)[:200])
            raise ProviderFailure("LOAD_FAILED", self._load_error, "unavailable") from exc
        self.version = "vegaml:%s" % getattr(mod, "__version__", "unknown")
        return self._model

    def _decide(self, request: DecisionRequest) -> RawAnswer:
        model = self._load()
        q = request.question
        qid = q.question_id
        vq = {qid: {"type": "choice", "instructions": q.instructions, "criteria": dict(q.categories)}}
        out = model.decide(render_situation(request), vq, mode="engine", alpha=self.alpha)
        try:
            ans = out["answers"][qid]
        except (KeyError, TypeError) as exc:
            raise ProviderFailure("MALFORMED_OUTPUT", "vegaml returned no answer for %s" % qid) from exc
        probs = ans.get("probs")
        probs = {str(k): float(v) for k, v in probs.items()} if isinstance(probs, dict) else None
        label = ans.get("choice")
        label = str(label) if label is not None else None
        conf = ans.get("confidence")
        conf = float(conf) if conf is not None else None
        pset = ans.get("prediction_set")
        pset = [str(x) for x in pset] if isinstance(pset, list) else None
        meta = {"unbound": bool(ans.get("unbound", False)), "vegaml_abstain": bool(ans.get("abstain", False)),
                "vegaml_latency_ms": out.get("latency_ms"), "adapter": ans.get("adapter")}
        abstain = meta["vegaml_abstain"] or meta["unbound"]
        reason = "VEGAML_UNBOUND" if meta["unbound"] else ("VEGAML_ABSTAIN" if abstain else None)
        return RawAnswer(label=label, confidence=conf,
                         confidence_source="model_probability" if conf is not None else "none",
                         probabilities=probs, prediction_set=pset, abstain=abstain, abstain_reason=reason,
                         metadata={k: v for k, v in meta.items() if v is not None})
