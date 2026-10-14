"""Versioned, declarative YAML rule packs: schema, loading and static validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationError, field_validator, model_validator

from sentinel.contracts.common import ApprovalLevel, RiskLevel, Severity, StrictModel, canonical_json, sha256_hex

from .grammar import Condition, GrammarError, canonical, parse_condition

MAX_PACK_BYTES = 512 * 1024


class ActionTypeSpec(StrictModel):
    description: str = ""
    risk: RiskLevel
    min_evidence: int = Field(default=0, ge=0, le=20)
    on_insufficient_evidence: Literal["block", "require_review"] = "require_review"
    required_payload_fields: list[str] = Field(default_factory=list)
    approval_level: ApprovalLevel | None = None


class SourceSpec(StrictModel):
    description: str = ""
    allowed_actions: list[str]
    allowed_targets: list[str] = Field(default_factory=lambda: ["*"])  # glob patterns


class RuleSpec(StrictModel):
    id: str = Field(pattern=r"^[A-Z][A-Z0-9_\-]{1,63}$")
    description: str = Field(min_length=1)
    severity: Severity
    effect: Literal["block", "require_review", "flag", "set_risk"]
    risk: RiskLevel | None = None
    applies_to: list[str] = Field(default_factory=list)  # action types; empty = all
    when: dict[str, Any]
    remediation: str | None = None
    version: str = Field(min_length=1)
    provenance: str = Field(min_length=1)

    @model_validator(mode="after")
    def _risk_iff_set_risk(self) -> "RuleSpec":
        if (self.effect == "set_risk") != (self.risk is not None):
            raise ValueError("'risk' is required for effect set_risk and only allowed there")
        return self


class RulePack(StrictModel):
    pack_id: str = Field(pattern=r"^[a-z][a-z0-9_\-]{0,63}$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    description: str = ""
    provenance: str = Field(min_length=1)
    action_types: dict[str, ActionTypeSpec] = Field(default_factory=dict)
    sources: dict[str, SourceSpec] = Field(default_factory=dict)
    forbidden_actions: list[str] = Field(default_factory=list)
    rules: list[RuleSpec] = Field(default_factory=list)

    @field_validator("action_types")
    @classmethod
    def _action_names(cls, v: dict) -> dict:
        import re
        for name in v:
            if not re.match(r"^[a-z][a-z0-9_.]{0,63}$", name):
                raise ValueError("invalid action type name %r" % name)
        return v

    @property
    def ref(self) -> str:
        return "%s@%s" % (self.pack_id, self.version)

    def digest(self) -> str:
        return sha256_hex(canonical_json(self.model_dump(mode="json")))[:12]


@dataclass(frozen=True)
class CompiledRule:
    spec: RuleSpec
    pack: str
    condition: Condition


@dataclass
class PackIssue:
    level: Literal["error", "warning"]
    code: str
    message: str
    location: str = ""

    def __str__(self) -> str:
        loc = (" [%s]" % self.location) if self.location else ""
        return "%s %s%s: %s" % (self.level.upper(), self.code, loc, self.message)


class RulePackError(ValueError):
    def __init__(self, issues: list[PackIssue]):
        self.issues = issues
        super().__init__("; ".join(str(i) for i in issues))


def load_pack_file(path: str | Path) -> RulePack:
    """Load and fully validate a pack. Raises RulePackError listing every error found."""
    pack, issues = check_pack_file(path)
    errors = [i for i in issues if i.level == "error"]
    if errors or pack is None:
        raise RulePackError(errors or [PackIssue("error", "PACK_INVALID", "pack could not be loaded")])
    return pack


def check_pack_file(path: str | Path) -> tuple[RulePack | None, list[PackIssue]]:
    p = Path(path)
    if not p.is_file():
        return None, [PackIssue("error", "PACK_NOT_FOUND", "no such file", str(p))]
    if p.stat().st_size > MAX_PACK_BYTES:
        return None, [PackIssue("error", "PACK_TOO_LARGE", "pack exceeds %d bytes" % MAX_PACK_BYTES, str(p))]
    try:
        # safe_load: no Python object tags, so a pack cannot construct arbitrary objects.
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return None, [PackIssue("error", "YAML_SYNTAX", str(exc).splitlines()[0], str(p))]
    return check_pack_data(raw, str(p))


def check_pack_data(raw: Any, location: str = "<data>") -> tuple[RulePack | None, list[PackIssue]]:
    if not isinstance(raw, dict):
        return None, [PackIssue("error", "PACK_NOT_MAPPING", "top level must be a mapping", location)]
    try:
        pack = RulePack.model_validate(raw)
    except ValidationError as exc:
        return None, [
            PackIssue("error", "SCHEMA", "%s: %s" % (".".join(str(x) for x in e["loc"]), e["msg"]), location)
            for e in exc.errors()
        ]
    issues = _semantic_issues(pack, location)
    return pack, issues


def compile_rules(pack: RulePack) -> list[CompiledRule]:
    return [CompiledRule(r, pack.ref, parse_condition(r.when)) for r in pack.rules]


def _semantic_issues(pack: RulePack, location: str) -> list[PackIssue]:
    issues: list[PackIssue] = []
    seen: dict[str, int] = {}
    for i, rule in enumerate(pack.rules):
        loc = "%s rules[%d] %s" % (location, i, rule.id)
        if rule.id in seen:
            issues.append(PackIssue("error", "DUPLICATE_RULE_ID", "rule id %s is defined twice" % rule.id, loc))
        seen[rule.id] = i
        try:
            parse_condition(rule.when)
        except GrammarError as exc:
            issues.append(PackIssue("error", "INVALID_CONDITION", str(exc), loc))
        for at in rule.applies_to:
            if at not in pack.action_types:
                issues.append(PackIssue("warning", "UNKNOWN_ACTION_TYPE", "applies_to names undefined %r" % at, loc))
    for src, spec in pack.sources.items():
        for at in spec.allowed_actions:
            if at not in pack.action_types:
                issues.append(PackIssue("error", "UNKNOWN_ACTION_TYPE",
                                        "source %s allows undefined action type %r" % (src, at), location))
            if at in pack.forbidden_actions:
                issues.append(PackIssue("error", "POLICY_CONFLICT",
                                        "source %s allows %r which the pack forbids" % (src, at), location))
    for at in pack.forbidden_actions:
        if at in pack.action_types:
            issues.append(PackIssue("warning", "FORBIDDEN_BUT_DEFINED",
                                    "%r is both defined and forbidden; it will always be blocked" % at, location))
    issues.extend(_conflicts(pack.rules, location))
    return issues


def _conflicts(rules: list[RuleSpec], location: str) -> list[PackIssue]:
    """Rules with identical conditions and scope but incompatible outcomes."""
    issues: list[PackIssue] = []
    groups: dict[Any, list[RuleSpec]] = {}
    for r in rules:
        try:
            key = (canonical(parse_condition(r.when)), tuple(sorted(r.applies_to)))
        except GrammarError:
            continue
        groups.setdefault(key, []).append(r)
    for group in groups.values():
        if len(group) < 2:
            continue
        ids = ", ".join(r.id for r in group)
        risks = {r.risk for r in group if r.effect == "set_risk"}
        if len(risks) > 1:
            issues.append(PackIssue("error", "RULE_CONFLICT",
                                    "rules %s share a condition but set different risk levels" % ids, location))
        effects = {r.effect for r in group if r.effect != "set_risk"}
        if len(effects) > 1:
            issues.append(PackIssue("warning", "RULE_OVERLAP",
                                    "rules %s share a condition with different effects %s; the strictest applies"
                                    % (ids, sorted(effects)), location))
        if len(group) > len({(r.effect, r.risk) for r in group}):
            issues.append(PackIssue("warning", "RULE_DUPLICATE", "rules %s appear to duplicate each other" % ids,
                                    location))
    return issues


def check_pack_set(packs: list[RulePack]) -> list[PackIssue]:
    """Conflicts between packs loaded together."""
    issues: list[PackIssue] = []
    rule_owner: dict[str, str] = {}
    action_owner: dict[str, tuple[str, ActionTypeSpec]] = {}
    for pack in packs:
        for r in pack.rules:
            if r.id in rule_owner:
                issues.append(PackIssue("error", "DUPLICATE_RULE_ID",
                                        "rule %s defined in %s and %s" % (r.id, rule_owner[r.id], pack.ref)))
            rule_owner[r.id] = pack.ref
        for name, spec in pack.action_types.items():
            if name in action_owner and action_owner[name][1] != spec:
                issues.append(PackIssue("error", "POLICY_CONFLICT", "action type %r defined differently in %s and %s"
                                        % (name, action_owner[name][0], pack.ref)))
            action_owner.setdefault(name, (pack.ref, spec))
    forbidden = {a for p in packs for a in p.forbidden_actions}
    for pack in packs:
        for src, spec in pack.sources.items():
            for at in spec.allowed_actions:
                if at in forbidden:
                    issues.append(PackIssue("error", "POLICY_CONFLICT",
                                            "%s source %s allows %r, forbidden by another pack" % (pack.ref, src, at)))
    return issues
