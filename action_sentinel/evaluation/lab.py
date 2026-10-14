"""Confidence Lab: run the real Sentinel pipeline over a versioned dataset and measure it.

Every configuration runs through `Sentinel.assess`, the same code path production uses, with an
isolated in-memory store. Provider answers are cached per (provider, example) inside one lab session,
so comparing several configurations calls each model only once per example and every configuration
sees identical provider outputs. That makes the comparison paired.
"""

from __future__ import annotations

import platform
import secrets
import time
from collections import Counter
from typing import Any

from action_sentinel import __version__
from action_sentinel.config import Settings
from action_sentinel.contracts import (
    DatasetExample,
    DecisionRequest,
    EvaluationRecord,
    EvaluationReport,
    EvaluationRunConfig,
    Outcome,
    ProviderResult,
    new_id,
)
from action_sentinel.governance.policy import PipelineFlags
from action_sentinel.providers import DecisionProvider, MockDecisionProvider, ProviderRegistry, RawAnswer
from action_sentinel.storage import SQLiteStore

from . import metrics as M
from .datasets import load_dataset

MOCK_NAMES = ("mock-a", "mock-b")


class LabError(RuntimeError):
    pass


class CachingProvider(DecisionProvider):
    """Wraps a provider; identical (event_id, question) requests reuse the first result."""

    def __init__(self, inner: DecisionProvider):
        super().__init__(inner.name, inner.version)
        self.inner = inner
        self.kind = inner.kind
        self.provider_type = inner.provider_type
        self.calibration = inner.calibration
        self._cache: dict[tuple[str, str], ProviderResult] = {}
        self.calls = 0

    def _decide(self, request: DecisionRequest) -> RawAnswer:  # pragma: no cover - decide() is overridden
        raise NotImplementedError

    def decide(self, request: DecisionRequest) -> ProviderResult:
        key = (request.action.event_id, request.question.model_dump_json())
        if key not in self._cache:
            self.calls += 1
            self._cache[key] = self.inner.decide(request)
            self.version = self.inner.version
        return self._cache[key].model_copy(update={"request_id": request.request_id})

    def health(self):
        return self.inner.health()


class ConfidenceLab:
    def __init__(self, settings: Settings, registry: ProviderRegistry | None = None):
        self.settings = settings
        self._base = registry
        self._registries: dict[str, ProviderRegistry] = {}

    def _registry_for(self, dataset_path: str, examples: list[DatasetExample]) -> ProviderRegistry:
        if dataset_path in self._registries:
            return self._registries[dataset_path]
        reg = ProviderRegistry()
        base = self._base
        if base is None:
            base = ProviderRegistry()
            base.build([p for p in self.settings.providers if p.get("type") != "mock"], self.settings.path("."))
        for name in base.names():
            reg.register(CachingProvider(base.get(name)))
        for name in MOCK_NAMES:  # simulated providers scripted by the dataset
            if name not in reg.names():
                script = {e.situation.event_id: e.mock_script[name] for e in examples if name in e.mock_script}
                reg.register(CachingProvider(MockDecisionProvider(name, script=script)))
        self._registries[dataset_path] = reg
        return reg

    def run(self, config: EvaluationRunConfig, require_available: bool = True) -> EvaluationReport:
        from action_sentinel.service import Sentinel  # local import: service imports evaluation-free modules only

        all_examples, digest = load_dataset(self.settings.path(config.dataset_path))
        examples = [e for e in all_examples if e.split in set(config.splits)]
        if config.limit:
            examples = examples[: config.limit]
        if not examples:
            raise LabError("no examples in splits %s" % config.splits)
        reg = self._registry_for(config.dataset_path, all_examples)
        missing = [p for p in config.providers if p not in reg.names()]
        if missing:
            raise LabError("unknown providers %s (available: %s)" % (missing, reg.names()))
        if require_available:
            down = [h for h in reg.health() if h.name in config.providers and not h.available]
            if down:
                raise LabError("providers unavailable: %s" % "; ".join("%s (%s)" % (h.name, h.detail) for h in down))

        settings = self.settings.model_copy(update={"db_path": ":memory:", "active_providers": []})
        sentinel = Sentinel(settings, store=SQLiteStore(":memory:"), registry=reg, key=secrets.token_bytes(32))
        flags = PipelineFlags(use_validation=config.use_validation, use_disagreement=config.use_disagreement,
                              use_uncertainty=config.use_uncertainty, use_review=config.use_review,
                              mode=config.governance_mode)
        records: list[EvaluationRecord] = []
        try:
            for ex in examples:
                started = time.perf_counter()
                d = sentinel.assess(ex.situation, correlation_id=new_id("lab"), question=ex.question,
                                    providers=config.providers, strategy=config.strategy, flags=flags)
                elapsed = 1000 * (time.perf_counter() - started)
                by = {r.provider_name: r for r in d.provider_results}
                records.append(EvaluationRecord(
                    example_id=ex.example_id, split=ex.split, tags=ex.tags, expected_label=ex.expected_label,
                    expected_outcome=ex.expected_outcome,
                    provider_labels={n: by[n].label if n in by and by[n].succeeded else None for n in config.providers},
                    provider_status={n: by[n].status if n in by else "not_called" for n in config.providers},
                    provider_confidence={n: by[n].confidence if n in by else None for n in config.providers},
                    provider_probabilities={n: by[n].probabilities if n in by else None for n in config.providers},
                    provider_latency_ms={n: by[n].latency_ms for n in config.providers if n in by},
                    outcome=d.status, reason_codes=d.reason_codes,
                    disagreement=bool(d.disagreement and not d.disagreement.agreement), latency_ms=elapsed))
        finally:
            sentinel.close()

        labels = list(examples[0].question.categories)
        report = EvaluationReport(
            config=config, dataset_version=examples[0].dataset_version, dataset_sha256=digest,
            n_examples=len(examples),
            environment={"sentinel": __version__, "python": platform.python_version(),
                         "platform": platform.platform(), "policy_version": sentinel.policy_version,
                         "provider_versions": {n: reg.get(n).version for n in config.providers},
                         "provider_kinds": {n: reg.get(n).kind for n in config.providers}},
            provider_metrics={n: provider_metrics(records, n, labels) for n in config.providers},
            governance_metrics=governance_metrics(records), records=records,
            notes=_notes(config, reg, examples))
        return report


def _notes(config: EvaluationRunConfig, reg: ProviderRegistry, examples: list[DatasetExample]) -> list[str]:
    notes = []
    if all(e.synthetic for e in examples):
        notes.append("All examples are synthetic; results describe behaviour on this dataset only.")
    for n in config.providers:
        if reg.get(n).kind == "test_double":
            notes.append("%s is a SIMULATED provider scripted with seeded noise around the true label; its "
                         "accuracy and calibration are properties of the script, not of any model." % n)
    if len(examples) < 100:
        notes.append("n=%d: confidence intervals are wide; differences of a few examples are not significant."
                     % len(examples))
    return notes


def provider_metrics(records: list[EvaluationRecord], name: str, labels: list[str]) -> dict[str, Any]:
    y_true = [r.expected_label for r in records]
    y_pred = [r.provider_labels.get(name) for r in records]
    status = Counter(r.provider_status.get(name, "not_called") for r in records)
    called = [r for r in records if r.provider_status.get(name) != "not_called"]
    out: dict[str, Any] = {
        "classification": M.classification(y_true, y_pred, labels),
        "status_counts": dict(status),
        "abstention_rate": M.rate(status.get("abstained", 0), len(called)),
        "error_rate": M.rate(status.get("error", 0) + status.get("unavailable", 0), len(called)),
        "timeout_rate": M.rate(status.get("timeout", 0), len(called)),
        "latency_ms": M.percentiles([r.provider_latency_ms[name] for r in called if name in r.provider_latency_ms]),
        "cost": "not available (no cost configured)",
    }
    answered = [(r, r.provider_labels[name]) for r in records if r.provider_labels.get(name) is not None]
    confs = [r.provider_confidence.get(name) for r, _ in answered]
    if answered and all(c is not None for c in confs):
        correct = [p == r.expected_label for r, p in answered]
        out["calibration"] = M.calibration(confs, correct)  # type: ignore[arg-type]
        out["risk_coverage"] = M.risk_coverage(confs, correct)  # type: ignore[arg-type]
    else:
        out["calibration"] = None
        out["calibration_note"] = "not computed: confidence missing for some or all answered examples"
    probs = [r.provider_probabilities.get(name) for r, _ in answered]
    pb = M.brier_and_log_loss(probs, [r.expected_label for r, _ in answered], labels) if answered else None  # type: ignore[arg-type]
    out["proper_scores"] = pb
    if pb is None:
        out["proper_scores_note"] = "not computed: full probability distributions not available"
    # High-confidence errors: the case governance must not trust.
    out["high_confidence_errors"] = sum(1 for (r, p), c in zip(answered, confs, strict=True)
                                        if c is not None and c >= 0.9 and p != r.expected_label)
    return out


def governance_metrics(records: list[EvaluationRecord]) -> dict[str, Any]:
    n = len(records)
    out = Counter(r.outcome.value for r in records)
    exp_not_approve = [r for r in records if r.expected_outcome != Outcome.APPROVE]
    exp_not_block = [r for r in records if r.expected_outcome != Outcome.BLOCK]
    exp_block = [r for r in records if r.expected_outcome == Outcome.BLOCK]
    exp_approve = [r for r in records if r.expected_outcome == Outcome.APPROVE]
    exp_review = [r for r in records if r.expected_outcome == Outcome.HUMAN_REVIEW]
    hard = [r for r in records if "hard_violation" in r.tags]
    return {
        "n": n,
        "outcome_counts": {o.value: out.get(o.value, 0) for o in Outcome},
        "outcome_accuracy": M.rate(sum(1 for r in records if r.outcome == r.expected_outcome), n),
        "false_approval_rate": M.rate(sum(1 for r in exp_not_approve if r.outcome == Outcome.APPROVE),
                                      len(exp_not_approve)),
        "false_block_rate": M.rate(sum(1 for r in exp_not_block if r.outcome == Outcome.BLOCK), len(exp_not_block)),
        "policy_violation_detection_rate": M.rate(sum(1 for r in exp_block if r.outcome == Outcome.BLOCK),
                                                  len(exp_block)),
        "hard_violation_blocked": M.rate(sum(1 for r in hard if r.outcome == Outcome.BLOCK), len(hard)),
        "missed_escalation_rate": M.rate(sum(1 for r in exp_review if r.outcome == Outcome.APPROVE), len(exp_review)),
        "over_escalation_rate": M.rate(sum(1 for r in exp_approve if r.outcome in (Outcome.HUMAN_REVIEW,
                                                                                   Outcome.ABSTAIN)),
                                       len(exp_approve)),
        "auto_approval_coverage": M.rate(sum(1 for r in exp_approve if r.outcome == Outcome.APPROVE),
                                         len(exp_approve)),
        "human_escalation_rate": M.rate(out.get("HUMAN_REVIEW", 0), n),
        "abstention_rate": M.rate(out.get("ABSTAIN", 0), n),
        "error_rate": M.rate(out.get("ERROR", 0), n),
        "disagreement_rate": M.rate(sum(1 for r in records if r.disagreement), n),
        "latency_ms": M.percentiles([r.latency_ms for r in records]),
    }


def calibration_profile(report: EvaluationReport, provider: str) -> dict[str, Any] | None:
    """A calibration profile for settings.calibration_profiles, derived from a measured run."""
    cal = report.provider_metrics.get(provider, {}).get("calibration")
    if not cal:
        return None
    return {"ece": round(cal["ece"], 4), "n": cal["n"], "dataset_version": report.dataset_version,
            "splits": report.config.splits, "ref": "%s:%s" % (report.run_id, provider)}


def suite(dataset_path: str, model: str, second: str | None, splits: list[str], limit: int | None = None,
          extra_naive: list[str] | None = None) -> list[EvaluationRunConfig]:
    """The standard comparison: baselines, incremental combinations, full Sentinel and ablations."""
    def cfg(name, providers, **kw):
        return EvaluationRunConfig(name=name, dataset_path=dataset_path, splits=splits, providers=providers,
                                   limit=limit, **kw)

    multi = ["rules", model] + ([second] if second else [])
    configs = [
        cfg("1_rules_only", ["rules"]),
        cfg("2_%s_only_naive" % model, [model], governance_mode="naive", use_validation=False),
    ]
    for extra in extra_naive or []:
        configs.append(cfg("3_%s_only_naive" % extra, [extra], governance_mode="naive", use_validation=False))
    configs += [
        cfg("4_validation_plus_%s" % model, [model], use_disagreement=False, use_uncertainty=False,
            use_review=False),
        cfg("5_rules_multi_disagreement", multi, use_uncertainty=False, use_review=False),
        cfg("6_full_sentinel", multi),
        cfg("ablation_no_validation", multi, use_validation=False),
        cfg("ablation_no_disagreement", multi, use_disagreement=False),
        cfg("ablation_no_uncertainty", multi, use_uncertainty=False),
        cfg("ablation_no_review", multi, use_review=False),
    ]
    return configs
