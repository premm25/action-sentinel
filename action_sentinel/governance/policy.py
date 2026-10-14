"""Deterministic governance: combine validation, provider results, uncertainty and disagreement into one outcome.

Precedence, highest first:

1. BLOCK         any hard policy violation (validation effect "block"). Nothing overrides this.
2. ERROR         no provider produced a result (all failed / timed out / unavailable).
3. HUMAN_REVIEW  any review reason: review findings, risk above the auto-approval ceiling, evidence not
                 sufficient, material disagreement, uncertainty, or a provider not rating the action
                 permissible.
4. ABSTAIN       providers abstained and nothing else calls for review.
5. APPROVE       only if every condition above is clear AND at least one provider rated the action
                 permissible. Confidence alone is never sufficient.

Ablation switches exist only for Confidence Lab. With review disabled, a case that would be escalated is
resolved by the first provider's label, conservatively: only an undisputed permissive label approves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import Field

from action_sentinel.contracts import (
    ApprovalLevel,
    DisagreementReport,
    Outcome,
    ProviderResult,
    RiskLevel,
    Severity,
    UncertaintyAssessment,
    ValidationReport,
)
from action_sentinel.contracts.common import StrictModel
from action_sentinel.uncertainty import UncertaintyThresholds


class GovernancePolicy(StrictModel):
    version: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$")
    max_auto_approve_risk: RiskLevel = RiskLevel.LOW
    permissive_labels: list[str] = Field(default_factory=lambda: ["safe"])
    restrictive_labels: list[str] = Field(default_factory=lambda: ["unsafe"])
    thresholds: UncertaintyThresholds = Field(default_factory=UncertaintyThresholds)
    review_on_disagreement_severity: Severity = Severity.MEDIUM
    review_ttl_seconds: int = Field(default=24 * 3600, gt=0)
    authorization_ttl_seconds: int = Field(default=15 * 60, gt=0)
    min_successful_providers: int = Field(default=1, ge=1)


@dataclass
class PipelineFlags:
    """Component switches. Production uses the defaults; Confidence Lab varies them for ablations."""

    use_validation: bool = True
    use_disagreement: bool = True
    use_uncertainty: bool = True
    use_review: bool = True
    mode: str = "sentinel"  # or "naive": map the first provider's label straight to an outcome


@dataclass
class Verdict:
    status: Outcome
    reasons: list[str] = field(default_factory=list)
    failed_rules: list[str] = field(default_factory=list)
    required_approval_level: ApprovalLevel = ApprovalLevel.NONE


class GovernanceEngine:
    component = "governance"

    def __init__(self, policy: GovernancePolicy):
        self.policy = policy

    def decide(self, validation: ValidationReport | None, results: list[ProviderResult],
               uncertainty: UncertaintyAssessment | None, disagreement: DisagreementReport | None,
               flags: PipelineFlags | None = None) -> Verdict:
        flags = flags or PipelineFlags()
        if flags.mode == "naive":
            return self._naive(results)
        p = self.policy
        level = validation.risk.required_approval_level if validation else ApprovalLevel.ADMIN

        # 1. hard policy
        if flags.use_validation and validation is not None and validation.hard_violations:
            hv = validation.hard_violations
            return Verdict(Outcome.BLOCK, ["HARD_POLICY_VIOLATION"] + [f.code for f in hv],
                           failed_rules=[f.rule_id or f.code for f in hv], required_approval_level=level)

        # 2. evaluation failure
        ok = [r for r in results if r.succeeded]
        answered = [r for r in results if r.status in ("ok", "abstained")]
        if not answered:
            codes = sorted({r.error.code for r in results if r.error}) or ["NO_PROVIDERS"]
            return Verdict(Outcome.ERROR, ["PROVIDERS_FAILED"] + ["PROVIDER_ERROR:%s" % c for c in codes],
                           required_approval_level=level)

        # 3. review reasons
        review: list[str] = []
        if flags.use_validation and validation is not None:
            review += ["REVIEW_FINDING:%s" % f.code for f in validation.review_findings]
            if validation.risk.level.rank > p.max_auto_approve_risk.rank:
                review.append("RISK_ABOVE_AUTO_APPROVAL:%s" % validation.risk.level.value)
            if validation.evidence.status not in ("sufficient", "not_required"):
                review.append("EVIDENCE_%s" % validation.evidence.status.upper())
        if flags.use_disagreement and disagreement is not None:
            for rec in disagreement.records:
                if rec.recommended_handling == "human_review" and \
                        rec.severity.rank >= p.review_on_disagreement_severity.rank:
                    review.append("DISAGREEMENT:%s" % rec.kind.upper())
        considered = ok if flags.use_disagreement else ok[:1]
        for r in considered:
            if r.label not in p.permissive_labels:
                review.append("PROVIDER_NOT_PERMISSIVE:%s=%s" % (r.provider_name, r.label))
        if 0 < len(ok) < p.min_successful_providers:
            review.append("TOO_FEW_PROVIDER_RESULTS")
        abstained = False
        if flags.use_uncertainty and uncertainty is not None:
            if uncertainty.status == "abstained":
                abstained = True
            elif uncertainty.status in ("uncertain", "unknown"):
                review.append("UNCERTAINTY_%s" % uncertainty.status.upper())
        elif not ok:
            abstained = True  # without uncertainty gating, all-abstained still cannot approve

        review = list(dict.fromkeys(review))  # de-duplicate, keep order
        if review and flags.use_review:
            if abstained:
                review.append("PROVIDERS_ABSTAINED")
            return Verdict(Outcome.HUMAN_REVIEW, review, required_approval_level=level)
        if abstained or not ok:
            return Verdict(Outcome.ABSTAIN, ["PROVIDERS_ABSTAINED"] + review, required_approval_level=level)
        if review:  # review disabled (ablation only): fall back to the first provider, conservatively
            first = ok[0]
            disputed = any(r.label != first.label for r in considered)
            if first.label in p.permissive_labels and not disputed:
                return Verdict(Outcome.APPROVE, ["REVIEW_DISABLED_FALLBACK_APPROVE"] + review)
            return Verdict(Outcome.BLOCK, ["REVIEW_DISABLED_FALLBACK_BLOCK"] + review, required_approval_level=level)

        # 5. approve: every mandatory condition is clear
        return Verdict(Outcome.APPROVE, ["ALL_CONDITIONS_MET"], required_approval_level=ApprovalLevel.NONE)

    def _naive(self, results: list[ProviderResult]) -> Verdict:
        """Baseline for comparison only: trust the first provider's label outright."""
        if not results or results[0].failed:
            return Verdict(Outcome.ERROR, ["NAIVE_PROVIDER_FAILED"])
        r = results[0]
        if r.status == "abstained" or r.label is None:
            return Verdict(Outcome.ABSTAIN, ["NAIVE_PROVIDER_ABSTAINED"])
        if r.label in self.policy.permissive_labels:
            return Verdict(Outcome.APPROVE, ["NAIVE_LABEL:%s" % r.label])
        if r.label in self.policy.restrictive_labels:
            return Verdict(Outcome.BLOCK, ["NAIVE_LABEL:%s" % r.label])
        return Verdict(Outcome.HUMAN_REVIEW, ["NAIVE_LABEL:%s" % r.label])
