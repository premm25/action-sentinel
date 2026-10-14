"""Deterministic classification from ordered, declarative rules. No language model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from action_sentinel.contracts import DecisionRequest
from action_sentinel.contracts.common import StrictModel
from action_sentinel.validation.engine import build_context
from action_sentinel.validation.grammar import Condition, evaluate, parse_condition

from .base import DecisionProvider, ProviderFailure, RawAnswer


class ClassifierRule(StrictModel):
    id: str
    label: str
    when: dict[str, Any]


class QuestionRules(StrictModel):
    rules: list[ClassifierRule] = Field(default_factory=list)
    default: str | None = None


class ClassifierConfig(StrictModel):
    version: str
    questions: dict[str, QuestionRules]


class RuleBasedDecisionProvider(DecisionProvider):
    provider_type = "rule_based"
    kind = "deterministic"

    def __init__(self, name: str, config: ClassifierConfig):
        super().__init__(name, "rules-%s" % config.version)
        self.config = config
        self._compiled: dict[str, list[tuple[ClassifierRule, Condition]]] = {
            qid: [(r, parse_condition(r.when)) for r in q.rules] for qid, q in config.questions.items()
        }

    @classmethod
    def from_file(cls, name: str, path: str | Path) -> "RuleBasedDecisionProvider":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(name, ClassifierConfig.model_validate(raw))

    def _decide(self, request: DecisionRequest) -> RawAnswer:
        qid = request.question.question_id
        if qid not in self.config.questions:
            return RawAnswer(abstain=True, abstain_reason="NO_RULES_FOR_QUESTION")
        q = self.config.questions[qid]
        for label in [r.label for r in q.rules] + ([q.default] if q.default else []):
            if label not in request.question.categories:
                raise ProviderFailure("CONFIG_MISMATCH", "classifier label %r is not a category of %s" % (label, qid))
        ctx = build_context(request.action)
        for rule, cond in self._compiled[qid]:
            if evaluate(cond, ctx):
                return RawAnswer(label=rule.label, metadata={"matched_rule": rule.id})
        if q.default is None:
            return RawAnswer(abstain=True, abstain_reason="NO_RULE_MATCHED")
        return RawAnswer(label=q.default, metadata={"matched_rule": "default"})
