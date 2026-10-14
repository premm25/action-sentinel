"""Reproducible, offline end-to-end demonstration.

    sentinel demo                 # temporary database, deleted afterwards
    sentinel demo --db var/demo.db  # keep the records (e.g. to browse them in the dashboard)

Uses the deterministic rule-based provider plus two SCRIPTED mock providers (clearly test doubles),
so no model, network or API key is needed. Every step checks its expected result and the command exits
non-zero if any check fails.
"""

from __future__ import annotations

import secrets
import tempfile
from pathlib import Path

from sentinel.config import load_settings
from sentinel.contracts import ActionProposal, EvaluationRunConfig
from sentinel.evaluation import ConfidenceLab
from sentinel.governance import ReviewError
from sentinel.providers import MockDecisionProvider, ProviderRegistry, RuleBasedDecisionProvider
from sentinel.service import Sentinel
from sentinel.storage import SQLiteStore


def _action(event_id: str, **kw) -> ActionProposal:
    base = dict(event_id=event_id, task_id="demo-task", agent_id="ops-agent", agent_version="1.0.0",
                action_type="restart_service", action_payload={"service": "checkout-api"},
                target_resource="staging/checkout-api", source_system="monitoring-agent",
                evidence_refs=[{"ref_id": "alert-1", "kind": "alert", "stance": "supports",
                                "description": "health check failing for 6 minutes"}],
                proposed_at="2026-10-14T10:00:00+00:00", idempotency_key="idem-" + event_id)
    base.update(kw)
    return ActionProposal.model_validate(base)


# Scripted mock outputs, keyed by event id. These are test doubles, not models.
SCRIPTS = {
    "mock-a": {
        "demo-clean": {"label": "safe", "confidence": 0.97},
        "demo-disagree": {"label": "safe", "confidence": 0.93},
        "demo-prod-db-delete": {"label": "safe", "confidence": 0.98},  # confidently wrong: policy must win
        "demo-abstain": {"abstain": True},
        "demo-prod-restart": {"label": "needs_review", "confidence": 0.8},
        "demo-refund": {"label": "needs_review", "confidence": 0.88},
    },
    "mock-b": {
        "demo-clean": {"label": "safe", "confidence": 0.9},
        "demo-disagree": {"label": "unsafe", "confidence": 0.86},
    },
}


class Demo:
    def __init__(self, quiet: bool):
        self.quiet = quiet
        self.failures: list[str] = []

    def say(self, text: str = "") -> None:
        if not self.quiet:
            print(text)

    def step(self, n: int, title: str) -> None:
        self.say("\n== %d. %s" % (n, title))

    def check(self, ok: bool, what: str) -> None:
        self.say("   [%s] %s" % ("ok" if ok else "FAIL", what))
        if not ok:
            self.failures.append(what)


def _decision_line(d) -> str:
    provs = ", ".join("%s=%s%s" % (r.provider_name, r.label or r.status,
                                   "@%.2f" % r.confidence if r.confidence is not None else "")
                      for r in d.provider_results)
    return "%s -> %s  [%s]  providers: %s" % (d.event_id, d.status.value, ", ".join(d.reason_codes[:4]), provs)


def run_demo(db_path: str | None = None, quiet: bool = False) -> int:
    tmp = None
    if db_path is None:
        tmp = tempfile.TemporaryDirectory()
        db_path = str(Path(tmp.name) / "demo.db")
    demo = Demo(quiet)
    settings = load_settings(None, db_path=db_path, environment="dev",
                             active_providers=["rules", "mock-a"])
    reg = ProviderRegistry()
    reg.register(RuleBasedDecisionProvider.from_file("rules", settings.path("examples/rule_packs/classifier.yaml")))
    for name in ("mock-a", "mock-b"):
        reg.register(MockDecisionProvider(name, script=SCRIPTS[name], default={"abstain": True}))
    s = Sentinel(settings, store=SQLiteStore(db_path), registry=reg, key=secrets.token_bytes(32))
    try:
        demo.say("Sentinel offline demonstration. Database: %s. Policy: %s" % (db_path, s.policy_version))
        demo.say("Providers: rules (deterministic), mock-a and mock-b (SCRIPTED TEST DOUBLES).")

        demo.step(1, "Submit proposed agent actions (rules + mock-a)")
        clean = s.evaluate(_action("demo-clean"))
        prod_restart = s.evaluate(_action("demo-prod-restart", target_resource="prod/checkout-api"))
        refund = s.evaluate(_action("demo-refund", source_system="support-agent", action_type="refund_payment",
                                    target_resource="payments/p-77",
                                    action_payload={"payment_id": "p-77", "amount": 900}))
        for out in (clean, prod_restart, refund):
            demo.say("   " + _decision_line(out.decision))
        demo.check(clean.decision.status.value == "APPROVE", "clean low-risk staging restart auto-approved")

        demo.step(2, "Provider agreement and disagreement (rules + mock-a + mock-b)")
        demo.check(clean.decision.disagreement.agreement, "providers agreed on the clean action")
        dis = s.evaluate(_action("demo-disagree", action_type="clear_cache", action_payload={"cache_name": "c"}),
                         providers=["rules", "mock-a", "mock-b"])
        demo.say("   " + _decision_line(dis.decision))
        demo.check(dis.decision.status.value == "HUMAN_REVIEW" and "DISAGREEMENT:CATEGORY" in dis.decision.reason_codes,
                   "category disagreement escalated to human review (no majority vote)")

        demo.step(3, "Hard policy beats a confident model")
        blocked = s.evaluate(_action("demo-prod-db-delete", source_system="infra-agent", action_type="delete_database",
                                     target_resource="prod/orders", action_payload={"database": "orders"},
                                     evidence_refs=[{"ref_id": "a", "kind": "metric", "stance": "supports"},
                                                    {"ref_id": "b", "kind": "ticket", "stance": "supports"}]),
                             providers=["mock-a"])
        demo.say("   " + _decision_line(blocked.decision))
        demo.check(blocked.decision.status.value == "BLOCK" and "PROD_DATABASE_DELETE" in blocked.decision.failed_rules,
                   "BLOCK despite mock-a rating it safe at 0.98")

        demo.step(4, "Abstention and escalation")
        ab = s.evaluate(_action("demo-abstain", action_type="clear_cache", action_payload={"cache_name": "c"}),
                        providers=["mock-a"])
        demo.say("   " + _decision_line(ab.decision))
        demo.check(ab.decision.status.value == "ABSTAIN" and ab.authorization is None,
                   "abstaining provider produced ABSTAIN, never an approval")
        demo.check(prod_restart.decision.status.value == "HUMAN_REVIEW", "production restart escalated for review")

        demo.step(5, "Approve and reject mock review requests")
        try:
            s.approve_review(refund.review.review_id, "olivia", "looks fine")
            demo.check(False, "operator should not approve a high-risk refund")
        except ReviewError as exc:
            demo.check(exc.code == "INSUFFICIENT_APPROVAL_LEVEL", "operator denied on high-risk review (%s)" % exc.code)
        approved = s.approve_review(prod_restart.review.review_id, "olivia", "latency alert confirmed")
        demo.check(approved.review.state.value == "APPROVED", "olivia approved the production restart")
        rejected = s.reject_review(dis.review.review_id, "olivia", "providers disagree; not worth the risk")
        demo.check(rejected.state.value == "REJECTED", "olivia rejected the disputed cache clear")

        demo.step(6, "Override followed by re-validation")
        try:
            s.override_review(refund.review.review_id, "sam", "refund the whole contract",
                              {"action_payload": {"payment_id": "p-77", "amount": 25000}})
            demo.check(False, "override above the hard limit should fail")
        except ReviewError as exc:
            demo.check(exc.code == "OVERRIDE_REVALIDATION_FAILED", "override to 25,000 rejected by re-validation")
        ov = s.override_review(refund.review.review_id, "sam", "partial refund agreed with customer",
                               {"action_payload": {"payment_id": "p-77", "amount": 300}})
        demo.say("   derived decision %s -> %s" % (ov.decision.event_id, ov.decision.status.value))
        demo.check(ov.review.state.value == "OVERRIDDEN", "sam's override to 300 passed re-validation")

        demo.step(7, "Stale or invalid authorization is rejected")
        r1 = s.execute(ov.authorization.token, refund.decision.action)
        demo.check(r1.reason_codes == ["ACTION_MISMATCH"], "original 900 refund cannot use the 300 authorization")
        tampered = _action("demo-clean", action_payload={"service": "payments-db"})
        r2 = s.execute(clean.authorization.token, tampered)
        demo.check(r2.reason_codes == ["ACTION_MISMATCH"], "tampered payload rejected")
        s.revoke_authorization(approved.authorization.grant.grant_id, "incident freeze", actor="ada")
        r3 = s.execute(approved.authorization.token, prod_restart.decision.action)
        demo.check(r3.reason_codes == ["AUTHORIZATION_REVOKED"], "revoked authorization rejected")
        r4 = s.execute("auth_forged.%s" % ("0" * 64), clean.decision.action)
        demo.check(r4.reason_codes == ["UNKNOWN_GRANT"], "forged token rejected")

        demo.step(8, "Authorized actions reach the mock executor exactly once")
        e1 = s.execute(clean.authorization.token, clean.decision.action)
        e2 = s.execute(ov.authorization.token, ov.review.final_action)
        e3 = s.execute(clean.authorization.token, clean.decision.action)
        demo.check(e1.status == "EXECUTED" and e2.status == "EXECUTED", "approved restart and overridden refund ran")
        demo.check(e3.status == "DUPLICATE", "repeat request returned DUPLICATE")
        demo.check(len(s.adapter.executed) == 2, "mock executor saw exactly 2 actions: %s"
                   % [a.event_id for a in s.adapter.executed])

        demo.step(9, "Confidence Lab on the versioned synthetic dataset (test split)")
        lab = ConfidenceLab(settings.model_copy(update={"db_path": ":memory:"}))
        ds = "examples/evaluation_datasets/sentinel_synthetic_v1.jsonl"
        full = lab.run(EvaluationRunConfig(name="full_sentinel", dataset_path=ds, providers=["rules", "mock-a", "mock-b"]))
        naive = lab.run(EvaluationRunConfig(name="mock-a_naive", dataset_path=ds, providers=["mock-a"],
                                            governance_mode="naive", use_validation=False))
        for rep in (naive, full):
            s.store.save_evaluation(rep)

        demo.step(10, "Metrics computed from those examples")
        for rep in (naive, full):
            g = rep.governance_metrics
            demo.say("   %-14s n=%d  outcome_acc=%.3f  false_approvals=%d/%d  hard_violations_blocked=%d/%d  "
                     "escalation=%.2f" % (rep.config.name, rep.n_examples, g["outcome_accuracy"]["value"],
                                          g["false_approval_rate"]["k"], g["false_approval_rate"]["n"],
                                          g["hard_violation_blocked"]["k"], g["hard_violation_blocked"]["n"],
                                          g["human_escalation_rate"]["value"]))
        demo.say("   (mock-a/mock-b are scripted simulations; these numbers test the pipeline, not a model)")
        demo.check(full.governance_metrics["false_approval_rate"]["k"] == 0, "full Sentinel: no false approvals")
        demo.check(full.governance_metrics["hard_violation_blocked"]["value"] == 1.0,
                   "full Sentinel: every hard violation blocked")

        demo.step(11, "Audit history of the refund workflow")
        events = s.store.list_events(correlation_id=refund.decision.correlation_id)
        for e in events:
            demo.say("   #%-4d %-28s %-13s actor=%-8s %s" % (e.seq, e.event_type, e.component, e.actor or "-",
                                                         ",".join(e.reason_codes[:3])))
        types = [e.event_type for e in events]
        demo.check(types[0] == "proposal.received" and "review.override_rejected" in types
                   and "review.overridden" in types and types[-1] == "execution.executed",
                   "audit trail reconstructs proposal -> review -> override -> execution")

        demo.say("\nResult: %s" % ("ALL CHECKS PASSED" if not demo.failures
                                   else "%d CHECK(S) FAILED: %s" % (len(demo.failures), demo.failures)))
        return 0 if not demo.failures else 1
    finally:
        s.close()
        s.store.close()
        if tmp is not None:
            tmp.cleanup()
