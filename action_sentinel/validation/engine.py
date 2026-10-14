"""Deterministic validation of action proposals against rule packs. No model is involved."""

from __future__ import annotations

import json
from collections.abc import Mapping
from fnmatch import fnmatchcase
from typing import Any

from action_sentinel.contracts import (
    ActionProposal,
    ApprovalLevel,
    EvidenceAssessment,
    RiskAssessment,
    RiskLevel,
    Severity,
    ValidationFinding,
    ValidationReport,
)

from .grammar import MISSING, evaluate, resolve
from .packs import CompiledRule, RulePack, RulePackError, check_pack_set, compile_rules

DEFAULT_APPROVAL_LEVELS: dict[RiskLevel, ApprovalLevel] = {
    RiskLevel.LOW: ApprovalLevel.OPERATOR,
    RiskLevel.MEDIUM: ApprovalLevel.OPERATOR,
    RiskLevel.HIGH: ApprovalLevel.SENIOR,
    RiskLevel.CRITICAL: ApprovalLevel.ADMIN,
}


def build_context(action: ActionProposal) -> dict[str, Any]:
    """The data a rule condition may read: the proposal plus a few derived values."""
    ctx = action.model_dump(mode="json")
    stances = [e.stance for e in action.evidence_refs]
    ctx["derived"] = {
        "evidence_count": len(stances),
        "supporting_evidence": stances.count("supports"),
        "contradicting_evidence": stances.count("contradicts"),
        "payload_bytes": len(json.dumps(action.action_payload, default=str)),
        "target_env": action.target_resource.split("/", 1)[0],
    }
    return ctx


class ValidationEngine:
    component = "validation"

    def __init__(self, packs: list[RulePack], approval_levels: Mapping[RiskLevel, ApprovalLevel] | None = None):
        if not packs:
            raise ValueError("at least one rule pack is required")
        errors = [i for i in check_pack_set(packs) if i.level == "error"]
        if errors:
            raise RulePackError(errors)
        self.packs = packs
        self.rules: list[CompiledRule] = [r for p in packs for r in compile_rules(p)]
        self.approval_levels = dict(approval_levels or DEFAULT_APPROVAL_LEVELS)
        self.action_types = {}
        for p in packs:
            for name, spec in p.action_types.items():
                self.action_types.setdefault(name, (p.ref, spec))
        self.forbidden = {a: p.ref for p in packs for a in p.forbidden_actions}

    @property
    def pack_refs(self) -> list[str]:
        return [p.ref for p in self.packs]

    def validate(self, action: ActionProposal, policy_version: str) -> ValidationReport:
        findings: list[ValidationFinding] = []
        try:
            return self._validate(action, policy_version, findings)
        except Exception as exc:  # fail closed: an internal error is a hard violation, never a pass
            findings.append(ValidationFinding(
                code="VALIDATION_INTERNAL_ERROR", severity=Severity.CRITICAL, effect="block",
                message="validation could not complete: %s" % type(exc).__name__,
                remediation="inspect the rule packs and the proposal; the action was not evaluated"))
            return ValidationReport(
                findings=findings, policy_version=policy_version, rule_packs=self.pack_refs,
                risk=RiskAssessment(level=RiskLevel.CRITICAL, base_level=RiskLevel.CRITICAL,
                                    required_approval_level=ApprovalLevel.ADMIN),
                evidence=EvidenceAssessment(status="insufficient", required=0, supporting=0, contradicting=0,
                                            neutral=0, unknown=0))

    def _validate(self, action: ActionProposal, policy_version: str,
                  findings: list[ValidationFinding]) -> ValidationReport:
        ctx = build_context(action)
        at = action.action_type
        spec_entry = self.action_types.get(at)

        if at in self.forbidden:
            findings.append(ValidationFinding(
                code="ACTION_FORBIDDEN", severity=Severity.CRITICAL, effect="block", pack_id=self.forbidden[at],
                fields=["action_type"], message="action type %r is forbidden by policy" % at,
                remediation="this action cannot be performed by an agent"))
        if spec_entry is None:
            findings.append(ValidationFinding(
                code="UNSUPPORTED_ACTION_TYPE", severity=Severity.HIGH, effect="block", fields=["action_type"],
                message="action type %r is not defined in any active rule pack" % at,
                remediation="use a supported action type or add it to a reviewed rule pack"))
        self._check_source(action, findings)

        base = RiskLevel.CRITICAL if spec_entry is None else spec_entry[1].risk
        required = 0
        if spec_entry is not None:
            pack_ref, spec = spec_entry
            required = spec.min_evidence
            for path in spec.required_payload_fields:
                val = resolve(ctx, tuple(path.split(".")))
                if val is MISSING or val is None or val == "":
                    findings.append(ValidationFinding(
                        code="MISSING_REQUIRED_FIELD", severity=Severity.HIGH, effect="block", pack_id=pack_ref,
                        fields=[path], message="required field %s is missing or empty" % path,
                        remediation="supply %s and resubmit" % path))

        evidence = self._evidence(action, required)
        if spec_entry is not None and evidence.status in ("insufficient", "unverified", "contradicted"):
            pack_ref, spec = spec_entry
            effect = spec.on_insufficient_evidence if evidence.status == "insufficient" else "require_review"
            findings.append(ValidationFinding(
                code="EVIDENCE_%s" % evidence.status.upper(), severity=Severity.MEDIUM, effect=effect,
                pack_id=pack_ref, fields=["evidence_refs"],
                message="evidence is %s: %d supporting of %d required, %d contradicting, %d of unknown stance"
                        % (evidence.status, evidence.supporting, evidence.required, evidence.contradicting,
                           evidence.unknown),
                remediation="attach evidence that supports the action"))

        level = base
        raised_by: list[str] = []
        set_risk_hits: list[tuple[str, RiskLevel]] = []
        for rule in self.rules:
            spec_r = rule.spec
            if spec_r.applies_to and at not in spec_r.applies_to:
                continue
            if not evaluate(rule.condition, ctx):
                continue
            if spec_r.effect == "set_risk":
                assert spec_r.risk is not None
                set_risk_hits.append((spec_r.id, spec_r.risk))
                if spec_r.risk.rank > level.rank:
                    level = spec_r.risk
                    raised_by.append(spec_r.id)
                continue
            findings.append(ValidationFinding(
                code="RULE_%s" % spec_r.id, severity=spec_r.severity, effect=spec_r.effect, rule_id=spec_r.id,
                rule_version=spec_r.version, pack_id=rule.pack, message=spec_r.description,
                resource=action.target_resource, remediation=spec_r.remediation, provenance=spec_r.provenance))

        if len({r for _, r in set_risk_hits}) > 1:
            # Rules disagree about the risk level. The highest is used, and a human should look.
            findings.append(ValidationFinding(
                code="POLICY_CONFLICT", severity=Severity.MEDIUM, effect="require_review",
                message="matched rules assign different risk levels: %s; the highest applies"
                        % ", ".join("%s=%s" % (i, r.value) for i, r in set_risk_hits),
                remediation="resolve the conflicting rules in the rule packs"))

        approval = self.approval_levels[level]
        if spec_entry is not None and spec_entry[1].approval_level is not None:
            override = spec_entry[1].approval_level
            approval = override if override.rank > approval.rank else approval
        return ValidationReport(
            findings=findings, policy_version=policy_version, rule_packs=self.pack_refs, evidence=evidence,
            risk=RiskAssessment(level=level, base_level=base, raised_by=raised_by, required_approval_level=approval))

    def _check_source(self, action: ActionProposal, findings: list[ValidationFinding]) -> None:
        packs_with_sources = [p for p in self.packs if p.sources]
        if not packs_with_sources:
            return
        defining = [p for p in packs_with_sources if action.source_system in p.sources]
        if not defining:
            findings.append(ValidationFinding(
                code="UNKNOWN_SOURCE", severity=Severity.HIGH, effect="block", fields=["source_system"],
                message="source system %r is not registered in any rule pack" % action.source_system,
                remediation="register the source system with its permitted actions"))
            return
        for p in defining:  # every pack that knows this source must permit the action
            src = p.sources[action.source_system]
            if action.action_type not in src.allowed_actions:
                findings.append(ValidationFinding(
                    code="SOURCE_ACTION_FORBIDDEN", severity=Severity.HIGH, effect="block", pack_id=p.ref,
                    fields=["source_system", "action_type"],
                    message="source %r may not perform %r" % (action.source_system, action.action_type),
                    remediation="route this action through a source that is permitted to perform it"))
            if not any(fnmatchcase(action.target_resource, pat) for pat in src.allowed_targets):
                findings.append(ValidationFinding(
                    code="TARGET_FORBIDDEN", severity=Severity.HIGH, effect="block", pack_id=p.ref,
                    fields=["target_resource"], resource=action.target_resource,
                    message="source %r may not act on %r" % (action.source_system, action.target_resource),
                    remediation="target a resource this source is permitted to change"))

    @staticmethod
    def _evidence(action: ActionProposal, required: int) -> EvidenceAssessment:
        stances = [e.stance for e in action.evidence_refs]
        counts = {s: stances.count(s) for s in ("supports", "contradicts", "neutral", "unknown")}
        if counts["contradicts"]:
            status = "contradicted"
        elif required == 0:
            status = "not_required"
        elif counts["supports"] >= required:
            status = "sufficient"
        elif counts["supports"] + counts["unknown"] >= required:
            status = "unverified"  # enough refs exist, but nothing says they support the action
        else:
            status = "insufficient"
        return EvidenceAssessment(status=status, required=required, supporting=counts["supports"],
                                  contradicting=counts["contradicts"], neutral=counts["neutral"],
                                  unknown=counts["unknown"])
