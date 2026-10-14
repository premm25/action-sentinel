"""Project-level contracts for the Vega Decision Intelligence Lab.

These are OUR contracts, not VegaML's native API. Every result keeps VegaML's untouched output in
`DecisionResult.native`, so nothing is lost by normalizing. Fields VegaML did not return are None,
never zero, and an abstention flag is never turned into a confident decision.

Native mapping (vegaml 0.9.0, verified in the source and in a live run):
  question  {"type": "choice"|"score"|"boolean", "instructions": str, "criteria": ...}
  answer    probs, confidence, prediction_set, abstain, unbound, settle_steps, energy,
            choice | score + expected_score | boolean, [adapter, gate], [view_agreement]
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from action_sentinel.contracts.common import StrictModel, new_id, utcnow

QuestionType = Literal["choice", "score", "boolean"]
AdapterMode = Literal["off", "auto", "on"]
MAX_STATE_CHARS = 400_000
MAX_OPTIONS = 200


class CandidateOption(StrictModel):
    """One answer the question allows. For `score` questions the label is the level index as a string."""

    label: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)


class DecisionQuestion(StrictModel):
    question_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_\-]{0,63}$")
    type: QuestionType
    instructions: str = Field(min_length=1, max_length=4000)
    options: list[CandidateOption] = Field(min_length=2, max_length=MAX_OPTIONS)

    @model_validator(mode="after")
    def _shape(self) -> "DecisionQuestion":
        labels = [o.label for o in self.options]
        if len(labels) != len(set(labels)):
            raise ValueError("option labels must be unique")
        if self.type == "boolean":
            if labels != ["false", "true"]:
                raise ValueError("a boolean question has exactly the options ['false', 'true'], in that order")
        if self.type == "score":
            if labels != [str(i) for i in range(len(labels))]:
                raise ValueError("score options must be labelled '0', '1', ... in ascending order")
            if any(not o.description for o in self.options):
                raise ValueError("every score level needs a description")
        return self

    @property
    def labels(self) -> list[str]:
        return [o.label for o in self.options]

    def to_vega(self) -> dict[str, Any]:
        """The native vegaml question dict (documented schema; see module docstring)."""
        if self.type == "choice":
            crit: Any = {o.label: o.description for o in self.options}
        elif self.type == "score":
            crit = [o.description for o in self.options]
        else:
            crit = {o.label: o.description for o in self.options if o.description}
            crit = crit or None
        q: dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if crit is not None:
            q["criteria"] = crit
        return q


class DecisionEvidence(StrictModel):
    """Supporting material. It is added to the state under the key "evidence" (a documented transformation)."""

    ref_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=20_000)
    source: str | None = None


class DecisionRequest(StrictModel):
    request_id: str = Field(default_factory=lambda: new_id("vreq"))
    state: str | dict[str, JsonValue]
    questions: list[DecisionQuestion] = Field(min_length=1, max_length=16)
    evidence: list[DecisionEvidence] = Field(default_factory=list, max_length=50)
    alpha: Literal["0.1", "0.05"] = "0.1"
    risk: Literal["0.05", "0.1", "0.2"] | None = None  # None: the checkpoint's default_risk
    views: int = Field(default=1, ge=1, le=3)

    @field_validator("questions")
    @classmethod
    def _unique_ids(cls, v: list[DecisionQuestion]) -> list[DecisionQuestion]:
        ids = [q.question_id for q in v]
        if len(ids) != len(set(ids)):
            raise ValueError("question ids must be unique")
        return v

    @model_validator(mode="after")
    def _state(self) -> "DecisionRequest":
        if isinstance(self.state, str) and not self.state.strip():
            raise ValueError("state must not be empty")
        if isinstance(self.state, dict) and self.evidence and "evidence" in self.state:
            raise ValueError("state already has an 'evidence' key; pass evidence in one place only")
        st = self.vega_state()
        if len(st if isinstance(st, str) else json.dumps(st)) > MAX_STATE_CHARS:
            raise ValueError("state exceeds %d characters" % MAX_STATE_CHARS)
        return self

    def vega_state(self) -> str | dict[str, Any]:
        if not self.evidence:
            return self.state
        ev = [{"ref": e.ref_id, "text": e.text, **({"source": e.source} if e.source else {})} for e in self.evidence]
        if isinstance(self.state, str):
            return {"situation": self.state, "evidence": ev}
        return {**self.state, "evidence": ev}

    def vega_questions(self) -> dict[str, dict[str, Any]]:
        return {q.question_id: q.to_vega() for q in self.questions}


class UncertaintyMetadata(StrictModel):
    """VegaML's uncertainty signals, unchanged in meaning. Calibration was fitted by VegaML's authors on
    their data; whether it holds for a given task has to be measured (Confidence Lab)."""

    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    prediction_set: list[str] | None = None
    prediction_set_alpha: str | None = None
    abstain: bool | None = None
    abstain_risk: str | None = None
    unbound: bool | None = None
    settle_steps: int | None = None
    energy: float | None = None
    view_agreement: float | None = None
    adapter: str | None = None
    gate: float | None = None
    calibration_source: Literal["vegaml_checkpoint", "none"] = "vegaml_checkpoint"


class QuestionDecision(StrictModel):
    question_id: str
    type: QuestionType
    selected_label: str  # VegaML's top option, reported even when abstain is True
    score: int | None = None
    expected_score: float | None = None
    boolean_probability: float | None = None  # P(true) for boolean questions
    uncertainty: UncertaintyMetadata

    @property
    def abstained(self) -> bool:
        return bool(self.uncertainty.abstain)


class DecisionResult(StrictModel):
    request_id: str
    engine: Literal["vegaml"] = "vegaml"
    engine_version: str
    model_id: str
    adapter_mode: AdapterMode
    live_inference: bool  # False only for test doubles; never set False for a real model call
    decisions: dict[str, QuestionDecision]
    wall_ms: float
    native_latency_ms: float | None = None
    n_tokens: int | None = None
    native: dict[str, Any]  # the exact dict vegaml returned
    created_at: datetime = Field(default_factory=utcnow)


ErrorCode = Literal["INVALID_INPUT", "MODEL_NOT_READY", "MODEL_LOAD_FAILED", "INFERENCE_FAILED", "TIMEOUT",
                    "MALFORMED_NATIVE_OUTPUT", "INPUT_TOO_LONG"]


class InferenceError(StrictModel):
    request_id: str | None = None
    code: ErrorCode
    message: str = Field(max_length=4000)
    retryable: bool = False


class VegaInferenceError(Exception):
    """Raised by the adapter and service; carries a typed InferenceError."""

    def __init__(self, error: InferenceError):
        super().__init__("%s: %s" % (error.code, error.message))
        self.error = error


class EvaluationRecord(StrictModel):
    """One system's prediction for one dataset example. Written to JSONL by the lab runner."""

    run_id: str
    system: str
    task: str
    split: str
    example_id: str
    gold_label: str
    labels: list[str]
    slices: dict[str, str] = Field(default_factory=dict)
    status: Literal["ok", "error"]
    predicted_label: str | None = None
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    abstain: bool | None = None
    prediction_set: list[str] | None = None
    unbound: bool | None = None
    settle_steps: int | None = None
    adapter: str | None = None
    wall_ms: float | None = None
    error: InferenceError | None = None
    native: dict[str, JsonValue] | None = None
