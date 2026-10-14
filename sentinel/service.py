"""Sentinel: wires validation, providers, uncertainty, disagreement, governance, review, authorization,
execution and audit into one service. The API, CLI, dashboard, demo and Confidence Lab all use this class.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from sentinel import __version__
from sentinel.audit import AuditLogger
from sentinel.config import Settings, signing_key
from sentinel.contracts import (
    ActionProposal,
    ApprovalLevel,
    DecisionQuestion,
    DecisionRequest,
    ExecutionResult,
    GovernanceDecision,
    IssuedAuthorization,
    Outcome,
    ReviewRecord,
    ReviewState,
    new_id,
    utcnow,
)
from sentinel.disagreement import detect_disagreement
from sentinel.execution import Authorizer, ExecutionAdapter, ExecutionGateway, MockExecutionAdapter
from sentinel.governance import review as rv
from sentinel.governance.policy import GovernanceEngine, GovernancePolicy, PipelineFlags
from sentinel.providers import ProviderRegistry, ProviderRunner
from sentinel.storage import ConflictError, SQLiteStore
from sentinel.uncertainty import assess_uncertainty
from sentinel.validation import RulePack, ValidationEngine, load_pack_file


class DuplicateEventConflict(Exception):
    """The same event_id was submitted with a different action."""


@dataclass
class EvaluationResult:
    decision: GovernanceDecision
    review: ReviewRecord | None = None
    authorization: IssuedAuthorization | None = None
    duplicate: bool = False


class Sentinel:
    def __init__(self, settings: Settings, *, store=None, registry: ProviderRegistry | None = None,
                 adapter: ExecutionAdapter | None = None, clock: Callable[[], datetime] = utcnow,
                 key: bytes | None = None, packs: list[RulePack] | None = None):
        self.settings = settings
        self.clock = clock
        self.store = store or SQLiteStore(settings.path(settings.db_path))
        if registry is None:
            registry = ProviderRegistry()
            registry.build(settings.providers, settings.path("."))
        self.registry = registry
        unknown = [n for n in settings.active_providers if n not in registry.names()]
        if unknown:
            raise ValueError("active_providers are not registered: %s" % unknown)
        self.runner = ProviderRunner(timeout_s=settings.provider_timeout_s,
                                     confident_at=settings.governance.thresholds.approve_min_confidence)
        self._set_policy(packs or [load_pack_file(settings.path(p)) for p in settings.rule_packs],
                         settings.governance)
        self.audit = AuditLogger(self.store, self.policy_version)
        self.authorizer = Authorizer(self.store, key or signing_key(settings),
                                     ttl_seconds=settings.governance.authorization_ttl_seconds, clock=clock)
        self.adapter = adapter or MockExecutionAdapter()
        self.gateway = ExecutionGateway(self.store, self.authorizer, self.adapter, self.revalidate,
                                        self.accepted_policy_versions, self.audit)

    # ------------------------------------------------------------------ policy
    def _set_policy(self, packs: list[RulePack], governance: GovernancePolicy) -> None:
        self.validation = ValidationEngine(packs)
        self.governance = GovernanceEngine(governance)
        self.policy_version = "gov-%s+%s" % (governance.version, "+".join(p.ref for p in packs))

    def reload_policy(self, packs: list[RulePack], governance: GovernancePolicy | None = None,
                      actor: str = "admin") -> str:
        old = self.policy_version
        self._set_policy(packs, governance or self.governance.policy)
        self.audit.policy_version = self.policy_version
        self.audit.emit("policy.changed", new_id("corr"), "policy", actor=actor,
                        data={"from": old, "to": self.policy_version})
        return self.policy_version

    def accepted_policy_versions(self) -> set[str]:
        return {self.policy_version, *self.settings.accepted_policy_versions}

    def revalidate(self, action: ActionProposal):
        return self.validation.validate(action, self.policy_version)

    # ------------------------------------------------------------------ evaluation
    def assess(self, action: ActionProposal, *, correlation_id: str, question: DecisionQuestion | None = None,
               providers: list[str] | None = None, strategy: str | None = None,
               flags: PipelineFlags | None = None, derived_from: str | None = None) -> GovernanceDecision:
        """Run the pipeline and return a decision. Writes audit events, but no decision, review or grant."""
        flags = flags or PipelineFlags()
        strategy = strategy or self.settings.strategy
        corr = correlation_id
        request = DecisionRequest(correlation_id=corr, action=action, **({"question": question} if question else {}))

        validation = self.validation.validate(action, self.policy_version)
        self.audit.emit("validation.completed", corr, "validation", reason_codes=[f.code for f in validation.findings],
                        data={"risk": validation.risk.level.value, "evidence": validation.evidence.status,
                              "hard_violations": len(validation.hard_violations)})

        names = providers if providers is not None else self.settings.active_providers
        provs = [self.registry.get(n) for n in names]
        self.audit.emit("provider.requested", corr, "providers", data={"providers": names, "strategy": strategy,
                                                                     "request_id": request.request_id})
        results = self.runner.run(provs, request, strategy)  # type: ignore[arg-type]
        for r in results:
            self.audit.emit("provider.result", corr, "providers", reason_codes=[r.status] + (
                [r.error.code] if r.error else []), data={
                "provider": r.provider_name, "version": r.provider_version, "status": r.status, "label": r.label,
                "confidence": r.confidence, "latency_ms": r.latency_ms, "test_output": r.is_test_output})

        gp = self.governance.policy
        uncertainty = assess_uncertainty(results, validation.evidence, gp.thresholds,
                                         self.settings.calibration_profiles)
        disagreement = detect_disagreement(results, validation, strategy, gp.permissive_labels,
                                           gp.restrictive_labels)
        self.audit.emit("uncertainty.assessed", corr, "uncertainty", reason_codes=uncertainty.reasons,
                        data={"status": uncertainty.status})
        if disagreement.records:
            self.audit.emit("disagreement.detected", corr, "disagreement",
                            reason_codes=[r.kind for r in disagreement.records],
                            data={"agreement": disagreement.agreement,
                                  "records": [r.model_dump(mode="json") for r in disagreement.records]})

        verdict = self.governance.decide(validation if flags.use_validation else None, results,
                                         uncertainty if flags.use_uncertainty else None,
                                         disagreement if flags.use_disagreement else None, flags)
        return GovernanceDecision(
            correlation_id=corr, event_id=action.event_id, status=verdict.status, reason_codes=verdict.reasons,
            action=action, action_hash=action.action_hash(), question_id=request.question.question_id,
            evidence_refs=[e.ref_id for e in action.evidence_refs], failed_rules=verdict.failed_rules,
            validation=validation, provider_results=results, uncertainty=uncertainty, disagreement=disagreement,
            risk_level=validation.risk.level, required_approval_level=verdict.required_approval_level,
            policy_version=self.policy_version, derived_from=derived_from, created_at=self.clock(),
            audit={"sentinel_version": __version__, "governance_version": gp.version,
                   "rule_packs": self.validation.pack_refs, "strategy": strategy,
                   "flags": {"validation": flags.use_validation, "disagreement": flags.use_disagreement,
                             "uncertainty": flags.use_uncertainty, "review": flags.use_review, "mode": flags.mode}})

    def evaluate(self, action: ActionProposal, *, question: DecisionQuestion | None = None,
                 providers: list[str] | None = None, strategy: str | None = None,
                 flags: PipelineFlags | None = None, actor: str | None = None) -> EvaluationResult:
        existing = self.store.find_decision_by_event(action.event_id)
        if existing is not None:
            if existing.action_hash != action.action_hash():
                self.audit.emit("proposal.conflict", existing.correlation_id, "intake", actor=actor,
                                decision_id=existing.decision_id, reason_codes=["EVENT_ID_REUSED"])
                raise DuplicateEventConflict("event_id %s was already evaluated with a different action"
                                             % action.event_id)
            self.audit.emit("proposal.duplicate", existing.correlation_id, "intake", actor=actor,
                            decision_id=existing.decision_id, reason_codes=["DUPLICATE_REQUEST"])
            review = self.store.get_review(existing.review_id) if existing.review_id else None
            return EvaluationResult(existing, review, None, duplicate=True)

        corr = new_id("corr")
        self.audit.emit("proposal.received", corr, "intake", actor=actor,
                        data={"action": action.model_dump(mode="json")})
        decision = self.assess(action, correlation_id=corr, question=question, providers=providers,
                               strategy=strategy, flags=flags)
        try:
            self.store.save_decision(decision)
        except ConflictError as exc:  # a concurrent submission of the same event won the race
            raise DuplicateEventConflict(str(exc)) from exc

        review = auth = None
        if decision.status == Outcome.HUMAN_REVIEW:
            review = rv.new_review(decision, self.governance.policy.review_ttl_seconds, self.clock())
            self.store.save_review(review)
            decision = decision.model_copy(update={"review_id": review.review_id})
            self.store.update_decision(decision)
            self.audit.emit("review.created", corr, "review", decision_id=decision.decision_id,
                            data={"review_id": review.review_id, "required_level": review.required_level.value})
        elif decision.status == Outcome.APPROVE:
            auth = self.authorizer.issue(decision_id=decision.decision_id, correlation_id=corr, action=action,
                                         policy_version=self.policy_version, issued_by="governance:auto")
            decision = decision.model_copy(update={"authorization_id": auth.grant.grant_id})
            self.store.update_decision(decision)
            self.audit.emit("authorization.issued", corr, "authorization", decision_id=decision.decision_id,
                            data={"grant_id": auth.grant.grant_id, "expires_at": auth.grant.expires_at.isoformat(),
                                  "issued_by": "governance:auto"})
        self.audit.emit("governance.decision", corr, "governance", decision_id=decision.decision_id,
                        reason_codes=decision.reason_codes,
                        data={"status": decision.status.value, "risk": decision.risk_level.value
                              if decision.risk_level else None})
        return EvaluationResult(decision, review, auth)

    # ------------------------------------------------------------------ human review
    def _review(self, review_id: str) -> ReviewRecord:
        r = self.store.get_review(review_id)
        if r is None:
            raise rv.ReviewNotFound("REVIEW_NOT_FOUND", "no review %s" % review_id)
        return r

    def _save_review(self, new: ReviewRecord, expected: ReviewState) -> None:
        try:
            self.store.update_review(new, expected)
        except ConflictError as exc:
            raise rv.InvalidTransition("CONCURRENT_UPDATE", str(exc)) from exc

    def _expire(self, rec: ReviewRecord) -> ReviewRecord:
        new = rv.transition(rec, ReviewState.EXPIRED, "sentinel", "expire", now=self.clock())
        new = new.model_copy(update={"final_disposition": "expired", "decided_at": self.clock()})
        self._save_review(new, rec.state)
        self.audit.emit("review.expired", rec.correlation_id, "review", decision_id=rec.decision_id,
                        data={"review_id": rec.review_id})
        return new

    def _authorize_reviewer(self, rec: ReviewRecord, reviewer_id: str, action: str,
                            required: ApprovalLevel | None = None):
        try:
            return rv.check_reviewer(rec, self.settings.reviewer(reviewer_id), required)
        except rv.ReviewerNotAuthorized as exc:
            self._save_review(rv.note(rec, reviewer_id, "denied", detail={"attempted": action, "code": exc.code},
                                      now=self.clock()), rec.state)
            self.audit.emit("review.denied", rec.correlation_id, "review", decision_id=rec.decision_id,
                            actor=reviewer_id, reason_codes=[exc.code], data={"attempted": action})
            raise

    def approve_review(self, review_id: str, reviewer_id: str, reason: str | None = None) -> EvaluationResult:
        rec = self._review(review_id)
        rv.ensure_open(rec, self.clock(), self._expire)
        self._authorize_reviewer(rec, reviewer_id, "approve")
        # Revalidate against the policy in force now: approval cannot bypass hard constraints.
        report = self.revalidate(rec.original_action)
        if report.hard_violations:
            codes = [f.code for f in report.hard_violations]
            self._save_review(rv.note(rec, reviewer_id, "approve_rejected", reason,
                                      {"hard_violations": codes}, self.clock()), rec.state)
            self.audit.emit("review.approve_rejected", rec.correlation_id, "review", decision_id=rec.decision_id,
                            actor=reviewer_id, reason_codes=codes)
            raise rv.RevalidationFailed("REVALIDATION_FAILED", "current policy blocks this action: %s"
                                        % ", ".join(codes))
        rec_needed = report.risk.required_approval_level
        if rec_needed.rank > rec.required_level.rank:
            self._authorize_reviewer(rec, reviewer_id, "approve", rec_needed)
        auth = self.authorizer.issue(decision_id=rec.decision_id, correlation_id=rec.correlation_id,
                                     action=rec.original_action, policy_version=self.policy_version,
                                     issued_by="reviewer:%s" % reviewer_id, review_id=rec.review_id,
                                     approvals=[reviewer_id])
        new = rv.transition(rec, ReviewState.APPROVED, reviewer_id, "approve", reason, now=self.clock())
        new = new.model_copy(update={"reviewer_id": reviewer_id, "reviewer_reason": reason,
                                     "decided_at": self.clock(), "final_disposition": "approved",
                                     "final_action": rec.original_action, "authorization_id": auth.grant.grant_id})
        try:
            self._save_review(new, rec.state)
        except rv.InvalidTransition:
            self.authorizer.revoke(auth.grant.grant_id, "review changed concurrently")
            raise
        self.audit.emit("review.approved", rec.correlation_id, "review", decision_id=rec.decision_id,
                        actor=reviewer_id, data={"review_id": review_id, "reason": reason})
        self.audit.emit("authorization.issued", rec.correlation_id, "authorization", decision_id=rec.decision_id,
                        actor=reviewer_id, data={"grant_id": auth.grant.grant_id, "issued_by": "reviewer"})
        return EvaluationResult(self.store.get_decision(rec.decision_id), new, auth)

    def reject_review(self, review_id: str, reviewer_id: str, reason: str) -> ReviewRecord:
        rec = self._review(review_id)
        reason = rv.require_reason(reason)
        rv.ensure_open(rec, self.clock(), self._expire)
        self._authorize_reviewer(rec, reviewer_id, "reject")
        new = rv.transition(rec, ReviewState.REJECTED, reviewer_id, "reject", reason, now=self.clock())
        new = new.model_copy(update={"reviewer_id": reviewer_id, "reviewer_reason": reason,
                                     "decided_at": self.clock(), "final_disposition": "rejected"})
        self._save_review(new, rec.state)
        self.audit.emit("review.rejected", rec.correlation_id, "review", decision_id=rec.decision_id,
                        actor=reviewer_id, data={"review_id": review_id, "reason": reason})
        return new

    def override_review(self, review_id: str, reviewer_id: str, reason: str, changes: dict) -> EvaluationResult:
        """Approve a modified action. The modified action is re-evaluated by the full pipeline first."""
        rec = self._review(review_id)
        reason = rv.require_reason(reason)
        rv.ensure_open(rec, self.clock(), self._expire)
        self._authorize_reviewer(rec, reviewer_id, "override")
        n = 1 + sum(1 for h in rec.history if h.action in ("override", "override_rejected"))
        new_action = rv.apply_override(rec.original_action, changes, n)
        self.audit.emit("review.override_proposed", rec.correlation_id, "review", decision_id=rec.decision_id,
                        actor=reviewer_id, data={"changes": changes, "reason": reason,
                                                 "new_event_id": new_action.event_id})
        derived = self.assess(new_action, correlation_id=rec.correlation_id, derived_from=rec.decision_id)
        self.store.save_decision(derived)
        if derived.status in (Outcome.BLOCK, Outcome.ERROR):
            self._save_review(rv.note(rec, reviewer_id, "override_rejected", reason,
                                      {"derived_decision": derived.decision_id, "status": derived.status.value,
                                       "reason_codes": derived.reason_codes}, self.clock()), rec.state)
            self.audit.emit("review.override_rejected", rec.correlation_id, "review", decision_id=rec.decision_id,
                            actor=reviewer_id, reason_codes=derived.reason_codes,
                            data={"derived_decision": derived.decision_id})
            raise rv.RevalidationFailed("OVERRIDE_REVALIDATION_FAILED", "the overridden action is %s: %s"
                                        % (derived.status.value, ", ".join(derived.reason_codes)))
        needed = max(rec.required_level, derived.required_approval_level, key=lambda lv: lv.rank)
        if needed.rank > rec.required_level.rank:
            self._authorize_reviewer(rec, reviewer_id, "override", needed)
        auth = self.authorizer.issue(decision_id=derived.decision_id, correlation_id=rec.correlation_id,
                                     action=new_action, policy_version=self.policy_version,
                                     issued_by="reviewer:%s" % reviewer_id, review_id=rec.review_id,
                                     approvals=[reviewer_id])
        derived = derived.model_copy(update={"authorization_id": auth.grant.grant_id, "review_id": rec.review_id})
        self.store.update_decision(derived)
        new = rv.transition(rec, ReviewState.OVERRIDDEN, reviewer_id, "override", reason,
                            {"derived_decision": derived.decision_id}, self.clock())
        new = new.model_copy(update={"reviewer_id": reviewer_id, "reviewer_reason": reason,
                                     "decided_at": self.clock(), "final_disposition": "overridden",
                                     "final_action": new_action, "override_decision_id": derived.decision_id,
                                     "authorization_id": auth.grant.grant_id})
        try:
            self._save_review(new, rec.state)
        except rv.InvalidTransition:
            self.authorizer.revoke(auth.grant.grant_id, "review changed concurrently")
            raise
        self.audit.emit("review.overridden", rec.correlation_id, "review", decision_id=rec.decision_id,
                        actor=reviewer_id, data={"derived_decision": derived.decision_id,
                                                 "derived_status": derived.status.value})
        self.audit.emit("authorization.issued", rec.correlation_id, "authorization", decision_id=derived.decision_id,
                        actor=reviewer_id, data={"grant_id": auth.grant.grant_id, "issued_by": "reviewer-override"})
        return EvaluationResult(derived, new, auth)

    def cancel_review(self, review_id: str, actor: str, reason: str) -> ReviewRecord:
        rec = self._review(review_id)
        reason = rv.require_reason(reason)
        if rec.state != ReviewState.PENDING:
            raise rv.InvalidTransition("REVIEW_CLOSED", "review is already %s" % rec.state.value)
        new = rv.transition(rec, ReviewState.CANCELLED, actor, "cancel", reason, now=self.clock())
        new = new.model_copy(update={"final_disposition": "cancelled", "decided_at": self.clock()})
        self._save_review(new, rec.state)
        self.audit.emit("review.cancelled", rec.correlation_id, "review", decision_id=rec.decision_id, actor=actor,
                        data={"reason": reason})
        return new

    def expire_due_reviews(self) -> list[str]:
        now = self.clock()
        expired = []
        for rec in self.store.list_reviews(state=ReviewState.PENDING.value, limit=10_000):
            if now >= rec.expires_at:
                self._expire(rec)
                expired.append(rec.review_id)
        return expired

    # ------------------------------------------------------------------ execution
    def execute(self, token: str, action: ActionProposal, actor: str = "agent") -> ExecutionResult:
        return self.gateway.execute(token, action, actor)

    def revoke_authorization(self, grant_id: str, reason: str, actor: str = "admin"):
        g = self.authorizer.revoke(grant_id, reason)
        self.audit.emit("authorization.revoked", g.correlation_id, "authorization", decision_id=g.decision_id,
                        actor=actor, data={"grant_id": grant_id, "reason": reason})
        return g

    # ------------------------------------------------------------------ queries
    def provider_health(self):
        return self.registry.health()

    def close(self) -> None:
        self.runner.close()
