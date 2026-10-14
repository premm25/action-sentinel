from __future__ import annotations

import copy

import pytest
import yaml

from action_sentinel.contracts import RiskLevel
from action_sentinel.validation import (
    GrammarError,
    RulePackError,
    ValidationEngine,
    check_pack_data,
    check_pack_file,
    check_pack_set,
    evaluate,
    load_pack_file,
    parse_condition,
)
from tests.conftest import CORE_PACK, make_action

PV = "test-policy"


@pytest.fixture(scope="module")
def engine():
    return ValidationEngine([load_pack_file(CORE_PACK)])


def codes(report):
    return {f.code for f in report.findings}


# ------------------------------------------------------------------ grammar

@pytest.mark.parametrize("raw", [
    "action_type == 'x'",                       # strings are not conditions
    {"field": "a", "op": "eval", "value": 1},   # unknown operator
    {"field": "__class__", "op": "exists"},     # dunder path
    {"field": "a.__globals__", "op": "exists"},
    {"field": "a;rm -rf", "op": "exists"},
    {"all": []},
    {"field": "a", "op": "exists", "value": 1},
    {"field": "a", "op": "eq"},
    {"field": "a", "op": "in", "value": "notalist"},
    {"field": "a", "op": "gt", "value": "10"},
    {"field": "a", "op": "matches", "value": "("},
    {"field": "a", "op": "matches", "value": "a" * 500},
    {"field": "a", "op": "eq", "value": {"nested": 1}},
    {"python": "import os"},
])
def test_grammar_rejects_unsafe_or_invalid(raw):
    with pytest.raises(GrammarError):
        parse_condition(raw)


def test_grammar_depth_limit():
    cond: dict = {"field": "a", "op": "exists"}
    for _ in range(12):
        cond = {"not": cond}
    with pytest.raises(GrammarError):
        parse_condition(cond)


@pytest.mark.parametrize("cond,expected", [
    ({"field": "a.b", "op": "eq", "value": 1}, True),
    ({"field": "a.missing", "op": "eq", "value": 1}, False),
    ({"field": "a.missing", "op": "ne", "value": 1}, True),
    ({"field": "a.missing", "op": "missing"}, True),
    ({"field": "a.b", "op": "gt", "value": 0}, True),
    ({"field": "s", "op": "gt", "value": 0}, False),          # non-numeric never compares
    ({"field": "flag", "op": "gt", "value": 0}, False),       # bools are not numbers here
    ({"field": "s", "op": "starts_with", "value": "pro"}, True),
    ({"field": "s", "op": "glob", "value": "prod*"}, True),
    ({"field": "s", "op": "matches", "value": "^prod$"}, True),
    ({"field": "items", "op": "contains", "value": "x"}, True),
    ({"field": "items.1", "op": "eq", "value": "y"}, True),
    ({"field": "items.9", "op": "exists"}, False),
    ({"any": [{"field": "s", "op": "eq", "value": "no"}, {"field": "a.b", "op": "lte", "value": 1}]}, True),
    ({"not": {"field": "s", "op": "in", "value": ["prod"]}}, False),
])
def test_grammar_semantics(cond, expected):
    ctx = {"a": {"b": 1}, "s": "prod", "flag": True, "items": ["x", "y"]}
    assert evaluate(parse_condition(cond), ctx) is expected


# ------------------------------------------------------------------ packs

def _core_raw():
    return yaml.safe_load(CORE_PACK.read_text())


def test_core_pack_is_clean():
    pack, issues = check_pack_file(CORE_PACK)
    assert pack is not None
    assert [i for i in issues if i.level == "error"] == []


def test_pack_rejects_python_yaml_tags(tmp_path):
    p = tmp_path / "evil.yaml"
    p.write_text("pack_id: x\nversion: 1.0.0\nprovenance: t\nrules: !!python/object/apply:os.system ['echo hi']\n")
    pack, issues = check_pack_file(p)
    assert pack is None and issues[0].code == "YAML_SYNTAX"


def test_pack_invalid_yaml_and_missing_file(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("pack_id: [unclosed\n")
    assert check_pack_file(p)[1][0].code == "YAML_SYNTAX"
    assert check_pack_file(tmp_path / "nope.yaml")[1][0].code == "PACK_NOT_FOUND"


def test_pack_schema_errors_reported():
    raw = _core_raw()
    del raw["provenance"]
    raw["rules"][0]["severity"] = "apocalyptic"
    pack, issues = check_pack_data(raw)
    assert pack is None
    assert {i.code for i in issues} == {"SCHEMA"} and len(issues) >= 2


def test_pack_rule_missing_fields():
    raw = _core_raw()
    del raw["rules"][0]["version"]
    assert check_pack_data(raw)[0] is None


def test_duplicate_rule_id_and_bad_condition():
    raw = _core_raw()
    raw["rules"].append(copy.deepcopy(raw["rules"][0]))
    raw["rules"].append({**copy.deepcopy(raw["rules"][1]), "id": "BAD_COND", "when": {"field": "x", "op": "exec"}})
    _, issues = check_pack_data(raw)
    found = {i.code for i in issues if i.level == "error"}
    assert {"DUPLICATE_RULE_ID", "INVALID_CONDITION"} <= found
    with pytest.raises(RulePackError):
        raise RulePackError([i for i in issues if i.level == "error"])


def test_conflicting_set_risk_rules_detected():
    raw = _core_raw()
    raw["rules"].append({**copy.deepcopy(raw["rules"][-1]), "id": "BULK_EMAIL_RISK_LOW", "risk": "low"})
    _, issues = check_pack_data(raw)
    assert "RULE_CONFLICT" in {i.code for i in issues if i.level == "error"}


def test_source_allowing_forbidden_action_is_conflict():
    raw = _core_raw()
    raw["sources"]["monitoring-agent"]["allowed_actions"].append("disable_audit_logging")
    _, issues = check_pack_data(raw)
    assert "POLICY_CONFLICT" in {i.code for i in issues}


def test_set_risk_requires_risk():
    raw = _core_raw()
    raw["rules"][-1].pop("risk")
    assert check_pack_data(raw)[0] is None


def test_cross_pack_conflicts():
    a = load_pack_file(CORE_PACK)
    raw = _core_raw()
    raw["pack_id"] = "other"
    raw["action_types"]["restart_service"]["risk"] = "high"
    b, _ = check_pack_data(raw)
    issues = check_pack_set([a, b])
    found = {i.code for i in issues}
    assert {"DUPLICATE_RULE_ID", "POLICY_CONFLICT"} <= found
    with pytest.raises(RulePackError):
        ValidationEngine([a, b])


# ------------------------------------------------------------------ engine

def test_clean_low_risk_action_passes(engine):
    r = engine.validate(make_action(), PV)
    assert r.passed_hard_constraints and r.findings == []
    assert r.risk.level == RiskLevel.LOW
    assert r.evidence.status == "sufficient"
    assert r.policy_version == PV and r.rule_packs == ["core@1.0.0"]


def test_unsupported_action_blocked(engine):
    r = engine.validate(make_action(action_type="launch_rocket"), PV)
    assert "UNSUPPORTED_ACTION_TYPE" in codes(r) and not r.passed_hard_constraints


def test_forbidden_action_blocked(engine):
    r = engine.validate(make_action(action_type="disable_audit_logging"), PV)
    assert "ACTION_FORBIDDEN" in codes(r)


def test_source_restrictions(engine):
    assert "UNKNOWN_SOURCE" in codes(engine.validate(make_action(source_system="rogue-agent"), PV))
    r = engine.validate(make_action(action_type="refund_payment",
                                    action_payload={"payment_id": "p1", "amount": 5}), PV)
    assert "SOURCE_ACTION_FORBIDDEN" in codes(r)
    r = engine.validate(make_action(target_resource="payments/ledger"), PV)
    assert "TARGET_FORBIDDEN" in codes(r)


def test_missing_required_payload_field(engine):
    r = engine.validate(make_action(action_payload={"reason": "x"}), PV)
    f = [f for f in r.findings if f.code == "MISSING_REQUIRED_FIELD"]
    assert f and f[0].fields == ["action_payload.service"] and f[0].effect == "block"


def test_evidence_states(engine):
    none = engine.validate(make_action(evidence_refs=[]), PV)
    assert none.evidence.status == "insufficient" and "EVIDENCE_INSUFFICIENT" in codes(none)
    unknown = engine.validate(make_action(evidence_refs=[{"ref_id": "a", "kind": "ticket"}]), PV)
    assert unknown.evidence.status == "unverified"  # unknown stance is not treated as support
    contra = engine.validate(make_action(evidence_refs=[
        {"ref_id": "a", "kind": "alert", "stance": "supports"},
        {"ref_id": "b", "kind": "metric", "stance": "contradicts"}]), PV)
    assert contra.evidence.status == "contradicted"
    assert all(f.effect == "require_review" for f in contra.findings)
    logs = engine.validate(make_action(action_type="read_logs", evidence_refs=[]), PV)
    assert logs.evidence.status == "not_required" and logs.findings == []


def test_insufficient_evidence_can_block_by_policy(engine):
    r = engine.validate(make_action(source_system="infra-agent", action_type="delete_database",
                                    target_resource="staging/orders", action_payload={"database": "orders"},
                                    evidence_refs=[]), PV)
    ev = [f for f in r.findings if f.code == "EVIDENCE_INSUFFICIENT"][0]
    assert ev.effect == "block"


def test_hard_rule_blocks_and_carries_provenance(engine):
    r = engine.validate(make_action(source_system="infra-agent", action_type="delete_database",
                                    target_resource="prod/orders", action_payload={"database": "orders"}), PV)
    f = [f for f in r.findings if f.rule_id == "PROD_DATABASE_DELETE"][0]
    assert f.effect == "block" and f.provenance and f.rule_version == "1" and f.remediation
    assert r.risk.level == RiskLevel.CRITICAL


def test_risk_raised_by_rules(engine):
    r = engine.validate(make_action(target_resource="prod/checkout-api"), PV)
    assert r.risk.level == RiskLevel.MEDIUM and r.risk.base_level == RiskLevel.LOW
    assert r.risk.raised_by == ["PROD_TARGET_RISK"]
    big = engine.validate(make_action(source_system="support-agent", action_type="refund_payment",
                                      target_resource="payments/p1",
                                      action_payload={"payment_id": "p1", "amount": 900}), PV)
    assert big.risk.level == RiskLevel.HIGH and big.risk.required_approval_level.value == "senior"


def test_secret_in_payload_blocked(engine):
    r = engine.validate(make_action(action_payload={"service": "a", "password": "hunter2"}), PV)
    assert "RULE_SECRET_IN_PAYLOAD" in codes(r)


def test_runtime_policy_conflict_requires_review():
    raw = _core_raw()
    raw["rules"].append({"id": "PROD_LOW", "description": "conflicting", "severity": "info", "effect": "set_risk",
                         "risk": "low", "when": {"field": "target_resource", "op": "starts_with", "value": "prod/"},
                         "version": "1", "provenance": "test"})
    pack, issues = check_pack_data(raw)
    eng = ValidationEngine([pack])
    r = eng.validate(make_action(target_resource="prod/checkout-api"), PV)
    conflict = [f for f in r.findings if f.code == "POLICY_CONFLICT"]
    assert conflict and conflict[0].effect == "require_review"
    assert r.risk.level == RiskLevel.MEDIUM  # the highest level wins


def test_internal_error_fails_closed(engine, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("bug")
    monkeypatch.setattr(engine, "_check_source", boom)
    r = engine.validate(make_action(), PV)
    assert not r.passed_hard_constraints
    assert codes(r) == {"VALIDATION_INTERNAL_ERROR"}


def test_engine_requires_packs():
    with pytest.raises(ValueError):
        ValidationEngine([])
