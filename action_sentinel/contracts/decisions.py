"""Decision requests, provider results, validation, uncertainty, disagreement and governance contracts.

Each stage has its own type. A provider result is not a validation result, and a governance
decision is not an authorization.
"""

from __future__ import annotations

import math
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import Field, JsonValue, model_validator

from .actions import ActionProposal
from .common import ApprovalLevel, RiskLevel, Severity, StrictModel, new_id, utcnow

# --------------------------------------------------------------------------- requests


class DecisionQuestion(StrictModel):
    """A question whose answer space is fixed in advance."""

    question_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    instructions: str = Field(min_length=1, max_length=4000)
    answer_type: Literal["choice", "boolean"] = "choice"
    categories: dict[str, str] = Field(min_length=2, max_length=64)

    @model_validator(mode="after")
    def _boolean_shape(self) -> "DecisionQuestion":
        if self.answer_type == "boolean" and set(self.categories) != {"true", "false"}:
            raise ValueError("a boolean question must have exactly the categories 'true' and 'false'")
        return self


DISPOSITION_QUESTION = DecisionQuestion(
    question_id="disposition",
    instructions=(
        "You review actions proposed by an automated agent before they run. Decide how the action "
        "should be handled, judging its potential for harm, reversibility, blast radius and whether the "
        "cited evidence supports it."
    ),
    categories={
        "safe": "routine, low-impact, reversible and supported by the evidence; may run automatically",
        "needs_review": "plausible but impactful, ambiguous or weakly supported; a human should decide",
        "unsafe": "destructive, irreversible, unauthorized or contradicted by evidence; must not run",
    },
)


class DecisionRequest(StrictModel):
    request_id: str = Field(default_factory=lambda: new_id("req"))
    correlation_id: str
    action: ActionProposal
    question: DecisionQuestion = DISPOSITION_QUESTION


# --------------------------------------------------------------------------- providers

ProviderStatus = Literal["ok", "abstained", "error", "timeout", "unavailable"]


class ProviderError(StrictModel):
    code: str
    message: str = Field(max_length=2000)


class ProviderResult(StrictModel):
    """A normalized provider answer. Fields a provider cannot supply are None, never invented."""

    request_id: str
    provider_name: str
    provider_version: str
    provider_kind: Literal["deterministic", "statistical", "test_double"]
    status: ProviderStatus
    label: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_source: Literal["model_probability", "self_reported", "none"] = "none"
    probabilities: dict[str, float] | None = None
    prediction_set: list[str] | None = None
    calibration_status: Literal["measured", "unknown", "not_applicable"] = "unknown"
    calibration_ref: str | None = None
    latency_ms: float = Field(ge=0.0)
    error: ProviderError | None = None
    is_test_output: bool = False
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistent(self) -> "ProviderResult":
        if self.status == "ok" and self.label is None:
            raise ValueError("an ok result must carry a label")
        if self.status in ("error", "timeout", "unavailable") and self.error is None:
            raise ValueError("a failed result must carry error details")
        if self.status in ("error", "timeout", "unavailable") and self.label is not None:
            raise ValueError("a failed result must not carry a label")
        if self.confidence is None and self.confidence_source != "none":
            raise ValueError("confidence_source must be 'none' when confidence is missing")
        if self.probabilities is not None:
            vals = list(self.probabilities.values())
            if any((not math.isfinite(p)) or p < 0 or p > 1 for p in vals):
                raise ValueError("probabilities must be finite and within [0, 1]")
            if abs(sum(vals) - 1.0) > 1e-3:
                raise ValueError("probabilities must sum to 1")
        return self

    @property
    def succeeded(self) -> bool:
        return self.status == "ok"

    @property
    def failed(self) -> bool:
        return self.status in ("error", "timeout", "unavailable")


class ProviderHealth(StrictModel):
    name: str
    provider_type: str
    version: str
    kind: str
    available: bool
    detail: str = ""


# --------------------------------------------------------------------------- validation


class ValidationFinding(StrictModel):
    code: str
    message: str
    severity: Severity
    effect: Literal["block", "require_review", "flag"]
    rule_id: str | None = None
    rule_version: str | None = None
    pack_id: str | None = None
    fields: list[str] = Field(default_factory=list)
    resource: str | None = None
    remediation: str | None = None
    provenance: str | None = None


class EvidenceAssessment(StrictModel):
    status: Literal["sufficient", "insufficient", "unverified", "contradicted", "not_required"]
    required: int = Field(ge=0)
    supporting: int = Field(ge=0)
    contradicting: int = Field(ge=0)
    neutral: int = Field(ge=0)
    unknown: int = Field(ge=0)


class RiskAssessment(StrictModel):
    level: RiskLevel
    base_level: RiskLevel
    raised_by: list[str] = Field(default_factory=list)
    required_approval_level: ApprovalLevel


class ValidationReport(StrictModel):
    findings: list[ValidationFinding] = Field(default_factory=list)
    risk: RiskAssessment
    evidence: EvidenceAssessment
    rule_packs: list[str] = Field(default_factory=list)
    policy_version: str

    @property
    def hard_violations(self) -> list[ValidationFinding]:
        return [f for f in self.findings if f.effect == "block"]

    @property
    def review_findings(self) -> list[ValidationFinding]:
        return [f for f in self.findings if f.effect == "require_review"]

    @property
    def passed_hard_constraints(self) -> bool:
        return not self.hard_violations


# --------------------------------------------------------------------------- uncertainty


class ProviderUncertainty(StrictModel):
    provider_name: str
    status: ProviderStatus
    confidence: float | None
    confidence_source: str
    calibration_status: str
    calibration_ece: float | None = None
    usable: bool
    notes: list[str] = Field(default_factory=list)


class UncertaintyAssessment(StrictModel):
    """Model confidence, calibration, evidence and provider health, kept as separate fields."""

    status: Literal["confident", "uncertain", "abstained", "unknown", "failed"]
    providers: list[ProviderUncertainty] = Field(default_factory=list)
    evidence: EvidenceAssessment
    reasons: list[str] = Field(default_factory=list)
    thresholds: dict[str, float] = Field(default_factory=dict)


# --------------------------------------------------------------------------- disagreement

DisagreementKind = Literal[
    "category", "confidence", "model_vs_policy", "evidence_vs_model", "missing_result", "contradictory_findings"
]


class DisagreementRecord(StrictModel):
    kind: DisagreementKind
    severity: Severity
    providers: list[str] = Field(default_factory=list)
    detail: str
    observed: dict[str, JsonValue] = Field(default_factory=dict)
    recommended_handling: Literal["human_review", "policy_precedence", "informational"]


class DisagreementReport(StrictModel):
    agreement: bool
    records: list[DisagreementRecord] = Field(default_factory=list)
    strategy: Literal["single", "sequential", "parallel"]

    @property
    def max_severity(self) -> Severity | None:
        if not self.records:
            return None
        return max((r.severity for r in self.records), key=lambda s: s.rank)


# --------------------------------------------------------------------------- governance


class Outcome(str, Enum):
    APPROVE = "APPROVE"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    BLOCK = "BLOCK"
    ABSTAIN = "ABSTAIN"
    ERROR = "ERROR"


class GovernanceDecision(StrictModel):
    decision_id: str = Field(default_factory=lambda: new_id("dec"))
    correlation_id: str
    event_id: str
    status: Outcome
    reason_codes: list[str]
    action: ActionProposal
    action_hash: str
    question_id: str
    evidence_refs: list[str] = Field(default_factory=list)
    failed_rules: list[str] = Field(default_factory=list)
    validation: ValidationReport | None = None
    provider_results: list[ProviderResult] = Field(default_factory=list)
    uncertainty: UncertaintyAssessment | None = None
    disagreement: DisagreementReport | None = None
    risk_level: RiskLevel | None = None
    required_approval_level: ApprovalLevel = ApprovalLevel.NONE
    policy_version: str
    review_id: str | None = None
    authorization_id: str | None = None
    derived_from: str | None = None  # decision id this one revalidates (human override)
    created_at: datetime = Field(default_factory=utcnow)
    audit: dict[str, Any] = Field(default_factory=dict)

    @property
    def executable(self) -> bool:
        """Only APPROVE can lead directly to execution; HUMAN_REVIEW needs a completed review first."""
        return self.status == Outcome.APPROVE
