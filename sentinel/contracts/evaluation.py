"""Confidence Lab datasets, run configuration, per-example records and reports."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, JsonValue, model_validator

from .actions import ActionProposal
from .common import RiskLevel, StrictModel, new_id, utcnow
from .decisions import DecisionQuestion, Outcome


class DatasetExample(StrictModel):
    example_id: str
    dataset_version: str
    synthetic: bool
    split: Literal["dev", "calibration", "test"]
    situation: ActionProposal
    question: DecisionQuestion
    expected_label: str
    expected_outcome: Outcome
    expected_risk: RiskLevel | None = None
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None
    # Scripted behaviour for named mock providers. Used only by test doubles; real providers ignore it.
    mock_script: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _label_in_question(self) -> "DatasetExample":
        if self.expected_label not in self.question.categories:
            raise ValueError("expected_label %r is not a category of the question" % self.expected_label)
        return self


class EvaluationRunConfig(StrictModel):
    name: str
    dataset_path: str
    splits: list[Literal["dev", "calibration", "test"]] = Field(default_factory=lambda: ["test"])
    providers: list[str]
    use_validation: bool = True
    use_disagreement: bool = True
    use_uncertainty: bool = True
    use_review: bool = True
    governance_mode: Literal["sentinel", "naive"] = "sentinel"
    strategy: Literal["single", "sequential", "parallel"] = "parallel"
    seed: int = 0
    limit: int | None = None


class EvaluationRecord(StrictModel):
    example_id: str
    split: str
    tags: list[str]
    expected_label: str
    expected_outcome: Outcome
    provider_labels: dict[str, str | None]
    provider_status: dict[str, str]
    provider_confidence: dict[str, float | None]
    provider_probabilities: dict[str, dict[str, float] | None]
    provider_latency_ms: dict[str, float]
    outcome: Outcome
    reason_codes: list[str]
    disagreement: bool
    latency_ms: float


class EvaluationReport(StrictModel):
    run_id: str = Field(default_factory=lambda: new_id("run"))
    created_at: datetime = Field(default_factory=utcnow)
    config: EvaluationRunConfig
    dataset_version: str
    dataset_sha256: str
    n_examples: int
    environment: dict[str, Any] = Field(default_factory=dict)
    provider_metrics: dict[str, dict[str, Any]] = Field(default_factory=dict)
    governance_metrics: dict[str, Any] = Field(default_factory=dict)
    records: list[EvaluationRecord] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
