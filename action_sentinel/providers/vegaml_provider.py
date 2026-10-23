"""VegaML as a Sentinel decision provider, built on the lab's VegaCore adapter.

VegaML supplies the decision signal; Sentinel's validation and governance decide what is permitted.
Native VegaML signals map onto Sentinel's provider contract as follows:
  probs -> probabilities, confidence -> confidence (source "model_probability"),
  prediction_set -> prediction_set, abstain or unbound -> status "abstained" (label kept, reason recorded),
  adapter/gate/settle_steps -> metadata. Calibration is reported "unknown": VegaML's calibration was
  fitted by its authors on their data, not on this deployment's actions.

The model loads on first use, or at startup through `warm_up()` (Sentinel.warm_up calls it). With
readout="ttt" the provider serves VegaML's few-shot head: fitted at load from `ttt_fit_dataset`, or loaded
from a file saved by `action-sentinel vega fit` (`ttt_head_path`), which skips the fit.

Pass `core` to inject a VegaCore (tests use one backed by FakeVegaRunner, whose results carry
live_inference=False and are marked as test output here).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from action_sentinel.contracts import DecisionRequest

from .base import DecisionProvider, ProviderFailure, RawAnswer

# One loaded model per (checkpoint, device), shared by every provider instance: the adapter mode is
# passed per call, so "vegaml-off" and "vegaml-auto" do not load the 3.5 GB model twice.
_SHARED_CORES: dict[tuple[str, str | None], Any] = {}


class VegaMLDecisionProvider(DecisionProvider):
    provider_type = "vegaml"
    kind = "statistical"

    def __init__(self, name: str, checkpoint: str = "0.8b", device: str | None = "cpu",
                 adapter_mode: str = "auto", core: Any = None, framing: str = "rich", readout: str = "engine",
                 ttt_fit_dataset: str | None = None, ttt_fit_split: str = "calibration",
                 ttt_head_path: str | None = None):
        super().__init__(name, "vegaml:%s:%s:%s:%s" % (checkpoint, adapter_mode, framing, readout))
        if framing not in ("rich", "compact"):
            raise ValueError("framing must be rich or compact")
        if readout not in ("engine", "ttt"):
            raise ValueError("readout must be engine or ttt")
        if readout == "ttt" and not (ttt_fit_dataset or ttt_head_path):
            raise ValueError("readout=ttt needs ttt_fit_dataset or ttt_head_path")
        if ttt_fit_split == "test":
            raise ValueError("a head must never be fitted on the test split")
        self.checkpoint = checkpoint
        self.device = device
        self.adapter_mode = adapter_mode
        self.framing = framing
        self.readout = readout
        self.ttt_fit_dataset = ttt_fit_dataset
        self.ttt_fit_split = ttt_fit_split
        self.ttt_head_path = ttt_head_path
        self.fit_report: dict[str, Any] | None = None
        self._head_question: tuple[str, str] | None = None  # (question_id, fingerprint) the head serves
        self._fit_question: dict[str, Any] | None = None
        self._core = core
        self._load_error: str | None = None

    def _get_core(self):
        core = self._load_core()
        if self.readout == "ttt" and self.fit_report is None:
            if self.ttt_head_path:
                self._load_saved_head(core)
            else:
                self._fit_head(core)
        return core

    def warm_up(self) -> str:
        """Load the model (and fit or load the head) now, so the first decision is not slowed by it."""
        self._get_core()
        return self.health().detail

    def _load_saved_head(self, core) -> None:
        from action_sentinel.vegalab.heads import HeadFileError, check_head, load_head
        try:
            arrays, meta = load_head(self.ttt_head_path)  # type: ignore[arg-type]
            check_head(meta, framing=self.framing, checkpoint=self.checkpoint,
                       engine_sha256=(core.runner.info() or {}).get("engine_sha256"))
        except HeadFileError as exc:
            raise ProviderFailure("TTT_HEAD_INVALID", str(exc), "unavailable") from exc
        core.install_head(meta["question_id"], meta["labels"], arrays, meta.get("cv_accuracy"))
        self._head_question = (meta["question_id"], meta["question_sha256"])
        self._fit_question = meta.get("question")
        self.fit_report = {"source": "file", "path": str(self.ttt_head_path),
                           **{k: meta.get(k) for k in ("split", "examples", "cv_accuracy", "dataset", "created_at")}}

    def save_head(self, path: str) -> str:
        """Write the fitted head to a file that `ttt_head_path` can load (see action_sentinel.vegalab.heads)."""
        import hashlib

        from action_sentinel.vegalab.heads import save_head
        core = self._get_core()
        if self.readout != "ttt" or self._head_question is None or self.fit_report is None:
            raise ValueError("only a provider with a fitted head (readout=ttt) can save one")
        qid, qsha = self._head_question
        arrays, labels, cv = core.export_head(qid)
        info = core.runner.info() or {}
        ds = self.ttt_fit_dataset
        meta = {"question_id": qid, "question_sha256": qsha, "question": self._fit_question, "labels": labels,
                "framing": self.framing, "checkpoint": self.checkpoint, "engine_sha256": info.get("engine_sha256"),
                "vegaml": info.get("vegaml"), "cv_accuracy": cv, "examples": self.fit_report.get("examples"),
                "split": self.ttt_fit_split, "dataset": Path(ds).name if ds else None,
                "dataset_sha256": hashlib.sha256(Path(ds).read_bytes()).hexdigest() if ds else None}
        return str(save_head(path, arrays, meta))

    def _fit_head(self, core) -> None:
        """Fit VegaML's few-shot head from the configured dataset split (never the test split)."""
        from action_sentinel.contracts import DatasetExample
        from action_sentinel.contracts import DecisionRequest as SReq
        rows = [DatasetExample.model_validate_json(line) for line in
                open(self.ttt_fit_dataset, encoding="utf-8").read().splitlines() if line.strip()]  # type: ignore[arg-type]
        rows = [r for r in rows if r.split == self.ttt_fit_split]
        if len(rows) < 6:
            raise ProviderFailure("TTT_FIT_FAILED", "fewer than 6 %s examples to fit a head" % self.ttt_fit_split)
        q = rows[0].question
        if any(r.question != q for r in rows):
            raise ProviderFailure("TTT_FIT_FAILED", "the %s examples ask different questions; a head serves one"
                                  % self.ttt_fit_split)
        question = {"type": "choice", "instructions": q.instructions, "criteria": dict(q.categories)}
        examples = [(self.state_text(SReq(correlation_id="fit", action=r.situation, question=r.question)),
                     {q.question_id: r.expected_label}) for r in rows]
        report = core.fit_heads(examples, {q.question_id: question})
        rep = report.get(q.question_id, {})
        if not rep.get("fitted"):
            raise ProviderFailure("TTT_FIT_FAILED", "head not kept: %s" % rep.get("reason", rep))
        self.fit_report = {"source": "fitted", "split": self.ttt_fit_split, "examples": len(rows), **rep}
        self._head_question = (q.question_id, _fingerprint(q))
        self._fit_question = {"instructions": q.instructions, "categories": dict(q.categories)}

    def _load_core(self):
        if self._core is not None:
            return self._core
        key = (self.checkpoint, self.device)
        if key in _SHARED_CORES:
            self._core = _SHARED_CORES[key]
            return self._core
        from action_sentinel.vegalab.contracts import VegaInferenceError
        from action_sentinel.vegalab.vega_core import LiveVegaRunner, VegaCore
        ok, detail = LiveVegaRunner.available()
        if not ok:
            raise ProviderFailure("NOT_INSTALLED", detail + " (pip install vegaml)", "unavailable")
        runner = LiveVegaRunner(self.checkpoint, device=self.device)
        try:
            runner.load()
        except VegaInferenceError as exc:
            self._load_error = exc.error.message
            raise ProviderFailure("LOAD_FAILED", exc.error.message, "unavailable") from exc
        self._core = _SHARED_CORES[key] = VegaCore(runner, model_id="vegaml-%s" % self.checkpoint)
        return self._core

    def health(self):
        h = super().health()
        if self._core is not None:
            return h.model_copy(update={"detail": "loaded (live=%s), adapters=%s, framing=%s, readout=%s%s" % (
                self._core.live, self.adapter_mode, self.framing, self.readout,
                ", head cv=%s" % self.fit_report.get("cv_accuracy") if self.fit_report else "")})
        from action_sentinel.vegalab.vega_core import LiveVegaRunner
        ok, detail = LiveVegaRunner.available()
        if not ok or self._load_error:
            return h.model_copy(update={"available": False, "detail": self._load_error or detail})
        return h.model_copy(update={"detail": "%s; model loads on first decision (~3.5 GB for 0.8b on CPU)" % detail})

    @staticmethod
    def state_of(request: DecisionRequest) -> dict[str, Any]:
        a = request.action
        state: dict[str, Any] = {"agent": a.agent_id, "source_system": a.source_system, "action_type": a.action_type,
                                 "target_resource": a.target_resource, "payload": a.action_payload}
        if a.context:
            state["context"] = a.context
        state["evidence"] = [{"kind": e.kind, "stance": e.stance, "description": e.description}
                             for e in a.evidence_refs] or "none provided"
        return state

    @staticmethod
    def compact_text(request: DecisionRequest) -> str:
        """One-sentence framing (see docs/vegalab/AGENT_FRAMING.md)."""
        import json
        a = request.action
        payload = json.dumps(a.action_payload, sort_keys=True)
        ev = "; ".join("%s (%s)" % (e.description or e.kind, e.stance) for e in a.evidence_refs) or "none"
        ctx = (" Context: %s." % json.dumps(a.context, sort_keys=True)) if a.context else ""
        return ("Agent %s (source %s) wants to %s on %s with %s.%s Evidence: %s."
                % (a.agent_id, a.source_system, a.action_type.replace("_", " "), a.target_resource, payload, ctx, ev))

    def state_text(self, request: DecisionRequest) -> Any:
        return self.compact_text(request) if self.framing == "compact" else self.state_of(request)

    def _decide(self, request: DecisionRequest) -> RawAnswer:
        from action_sentinel.vegalab.contracts import CandidateOption, DecisionQuestion, VegaInferenceError
        from action_sentinel.vegalab.contracts import DecisionRequest as VReq
        core = self._get_core()
        q = request.question
        if self.readout == "ttt" and self._head_question != (q.question_id, _fingerprint(q)):
            raise ProviderFailure("TTT_HEAD_QUESTION_MISMATCH",
                                  "the head was fitted for question %r with other wording or options than %r asks; "
                                  "its features would not match" % ((self._head_question or ("?",))[0], q.question_id))
        vq = DecisionQuestion(question_id=q.question_id, type="choice", instructions=q.instructions,
                              options=[CandidateOption(label=k, description=v) for k, v in q.categories.items()])
        try:
            res = core.decide(VReq(request_id=request.request_id[:128], state=self.state_text(request), questions=[vq]),
                              adapter_mode=self.adapter_mode, readout=self.readout)
        except VegaInferenceError as exc:
            status = "timeout" if exc.error.code == "TIMEOUT" else "error"
            raise ProviderFailure(exc.error.code, exc.error.message, status) from exc
        self.version = "vegaml:%s:%s:%s:%s:%s" % (res.engine_version, self.checkpoint, self.adapter_mode,
                                                  self.framing, self.readout)
        d = res.decisions[q.question_id]
        u = d.uncertainty
        abstain = bool(u.abstain) or bool(u.unbound)
        reason = "VEGAML_UNBOUND" if u.unbound else ("VEGAML_ABSTAIN" if u.abstain else None)
        meta = {"live_inference": res.live_inference, "adapter_mode": self.adapter_mode, "framing": self.framing,
                "readout": res.readout, "head_cv_accuracy": (self.fit_report or {}).get("cv_accuracy"),
                "adapter": u.adapter,
                "gate": u.gate, "settle_steps": u.settle_steps, "unbound": u.unbound, "vegaml_abstain": u.abstain}
        if not res.live_inference:
            meta["test_double"] = True
        return RawAnswer(label=d.selected_label, confidence=u.confidence,
                         confidence_source="model_probability" if u.confidence is not None else "none",
                         probabilities=u.probabilities, prediction_set=u.prediction_set, abstain=abstain,
                         abstain_reason=reason, metadata={k: v for k, v in meta.items() if v is not None})


def _fingerprint(q) -> str:
    from action_sentinel.vegalab.heads import question_fingerprint
    return question_fingerprint(q.question_id, q.instructions, dict(q.categories))
