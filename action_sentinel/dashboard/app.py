"""Sentinel dashboard (Streamlit). Reads and writes the same SQLite store as the API and CLI.

    action-sentinel dashboard
    # or: streamlit run action_sentinel/dashboard/app.py -- --config config/sentinel.yaml

Every number shown is computed from stored records; nothing is hard-coded or simulated here.
"""

import argparse
import hmac
import json
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from action_sentinel.audit import export_jsonl, sanitize  # noqa: E402
from action_sentinel.config import hash_key, load_settings  # noqa: E402
from action_sentinel.contracts import EvaluationRunConfig, Outcome  # noqa: E402
from action_sentinel.evaluation import ConfidenceLab, render_comparison  # noqa: E402
from action_sentinel.governance import ReviewError  # noqa: E402

st.set_page_config(page_title="Sentinel", layout="wide")


def _config_arg():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args.config or None


@st.cache_resource
def get_service(config, db_path):  # db_path is part of the cache key: one service per database
    from action_sentinel.service import Sentinel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return Sentinel(load_settings(config))


@st.cache_resource
def get_lab(config, db_path):
    return ConfidenceLab(load_settings(config))


svc = get_service(_config_arg(), os.environ.get("SENTINEL_DB_PATH"))
store = svc.store
settings = svc.settings

st.title("Sentinel")
st.caption("Decision governance for AI agent actions. Policy `%s`. Database `%s`."
           % (svc.policy_version, settings.db_path))

tabs = st.tabs(["Overview", "Decision Explorer", "Confidence Lab", "Human Review", "Audit Explorer",
                "Configuration"])


# ---------------------------------------------------------------- overview
with tabs[0]:
    counts = store.count_decisions_by_status()
    total = sum(counts.values())
    cols = st.columns(6)
    cols[0].metric("Evaluated decisions", total)
    for col, o in zip(cols[1:], Outcome, strict=True):
        col.metric(o.value, counts.get(o.value, 0))
    if total == 0:
        st.info("No decisions recorded yet. Submit actions through the API or run `action-sentinel demo --db %s`."
                % settings.db_path)
    else:
        recent = store.list_decisions(limit=200)
        violations = {}
        for d in recent:
            for r in d.failed_rules:
                violations[r] = violations.get(r, 0) + 1
        st.subheader("Policy violations (last %d decisions)" % len(recent))
        if violations:
            st.dataframe(pd.DataFrame(sorted(violations.items(), key=lambda kv: -kv[1]),
                                      columns=["rule", "blocked decisions"]), hide_index=True)
        else:
            st.write("No hard policy violations recorded.")
    st.subheader("Providers")
    st.dataframe(pd.DataFrame([h.model_dump() for h in svc.provider_health()]), hide_index=True)
    st.subheader("Recent evaluation runs")
    runs = store.list_evaluations(10)
    if runs:
        st.dataframe(pd.DataFrame(runs), hide_index=True)
    else:
        st.write("No Confidence Lab runs stored.")


# ---------------------------------------------------------------- decision explorer
with tabs[1]:
    status = st.selectbox("Status", ["all"] + [o.value for o in Outcome], key="dx_status")
    decisions = store.list_decisions(limit=200, status=None if status == "all" else status)
    if not decisions:
        st.info("No decisions match.")
    else:
        st.dataframe(pd.DataFrame([{
            "created": d.created_at.isoformat(timespec="seconds"), "event_id": d.event_id, "status": d.status.value,
            "action": d.action.action_type, "target": d.action.target_resource,
            "risk": d.risk_level.value if d.risk_level else None, "reasons": ", ".join(d.reason_codes),
            "decision_id": d.decision_id} for d in decisions]), hide_index=True)
        chosen = st.selectbox("Inspect decision", [d.decision_id for d in decisions], key="dx_pick",
                              format_func=lambda i: next("%s · %s" % (d.event_id, d.status.value)
                                                         for d in decisions if d.decision_id == i))
        d = store.get_decision(chosen)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Proposed action**")
            st.json(sanitize(d.action.model_dump(mode="json")))
            st.markdown("**Validation findings**")
            if d.validation and d.validation.findings:
                st.dataframe(pd.DataFrame([{"code": f.code, "effect": f.effect, "severity": f.severity.value,
                                            "message": f.message, "remediation": f.remediation}
                                           for f in d.validation.findings]), hide_index=True)
            else:
                st.write("None.")
        with c2:
            st.markdown("**Final decision**: `%s`" % d.status.value)
            st.write("Reason codes:", d.reason_codes)
            st.write("Risk: %s · required approval: %s · evidence: %s" % (
                d.risk_level.value if d.risk_level else "n/a", d.required_approval_level.value,
                d.validation.evidence.status if d.validation else "n/a"))
            st.markdown("**Provider outputs**")
            st.dataframe(pd.DataFrame([{"provider": r.provider_name, "kind": r.provider_kind, "status": r.status,
                                        "label": r.label, "confidence": r.confidence,
                                        "calibration": r.calibration_status, "test output": r.is_test_output,
                                        "error": r.error.code if r.error else None}
                                       for r in d.provider_results]), hide_index=True)
            if d.uncertainty:
                st.markdown("**Uncertainty**: `%s`" % d.uncertainty.status)
                st.write(d.uncertainty.reasons or "no notes")
            st.markdown("**Disagreements**")
            if d.disagreement and d.disagreement.records:
                st.dataframe(pd.DataFrame([{"kind": r.kind, "severity": r.severity.value,
                                            "handling": r.recommended_handling, "detail": r.detail}
                                           for r in d.disagreement.records]), hide_index=True)
            else:
                st.write("None.")


# ---------------------------------------------------------------- confidence lab
with tabs[2]:
    lab = get_lab(_config_arg(), os.environ.get("SENTINEL_DB_PATH"))
    ds_dir = settings.path("examples/evaluation_datasets")
    datasets = sorted("examples/evaluation_datasets/" + p.name for p in ds_dir.glob("*.jsonl"))
    with st.expander("Run an evaluation", expanded=False):
        if not datasets:
            st.warning("No datasets found under examples/evaluation_datasets.")
        else:
            ds = st.selectbox("Dataset", datasets, key="lab_ds")
            provider_names = sorted(set(svc.registry.names()) | {"mock-a", "mock-b"})
            provs = st.multiselect("Providers", provider_names, default=["rules"], key="lab_provs")
            splits = st.multiselect("Splits", ["dev", "calibration", "test"], default=["test"], key="lab_splits")
            c = st.columns(5)
            use_val = c[0].checkbox("validation", True, key="lab_v")
            use_dis = c[1].checkbox("disagreement", True, key="lab_d")
            use_unc = c[2].checkbox("uncertainty", True, key="lab_u")
            use_rev = c[3].checkbox("review", True, key="lab_r")
            naive = c[4].checkbox("naive mode", False, key="lab_n")
            name = st.text_input("Run name", "dashboard-run", key="lab_name")
            if st.button("Run", key="lab_run", disabled=not provs or not splits):
                try:
                    with st.spinner("Evaluating (local LLM providers can take minutes)..."):
                        rep = lab.run(EvaluationRunConfig(
                            name=name, dataset_path=ds, providers=provs, splits=splits, use_validation=use_val,
                            use_disagreement=use_dis, use_uncertainty=use_unc, use_review=use_rev,
                            governance_mode="naive" if naive else "sentinel"))
                    store.save_evaluation(rep)
                    st.success("Stored run %s (%d examples)" % (rep.run_id, rep.n_examples))
                except Exception as exc:  # shown to the operator, not swallowed
                    st.error("%s: %s" % (type(exc).__name__, exc))
    runs = store.list_evaluations(100)
    if not runs:
        st.info("No evaluation runs stored. Run one above or with `action-sentinel lab suite`.")
    else:
        picked = st.multiselect("Compare runs", [r["run_id"] for r in runs], default=[runs[0]["run_id"]],
                                key="lab_pick", format_func=lambda rid: next(
                                    "%s (%s)" % (r["name"], r["run_id"][-6:]) for r in runs if r["run_id"] == rid))
        reports = [store.get_evaluation(rid) for rid in picked]
        if reports:
            gov_keys = ["outcome_accuracy", "false_approval_rate", "hard_violation_blocked", "false_block_rate",
                        "missed_escalation_rate", "over_escalation_rate", "human_escalation_rate",
                        "abstention_rate", "error_rate", "disagreement_rate"]
            st.subheader("Governance metrics")
            st.dataframe(pd.DataFrame({r.config.name: {k: (r.governance_metrics[k]["value"]) for k in gov_keys}
                                       for r in reports}))
            rep = reports[0]
            st.subheader("Provider detail: %s" % rep.config.name)
            prov = st.selectbox("Provider", list(rep.provider_metrics), key="lab_prov")
            pm = rep.provider_metrics[prov]
            cm = pm["classification"]["confusion_matrix"]
            st.markdown("Confusion matrix (rows: expected, columns: predicted)")
            st.dataframe(pd.DataFrame(cm["rows_true_cols_pred"], index=cm["labels"], columns=cm["labels"]))
            cal = pm.get("calibration")
            if cal:
                st.markdown("Reliability diagram (ECE %.3f, n=%d)" % (cal["ece"], cal["n"]))
                rel = pd.DataFrame([b for b in cal["reliability"] if b["count"]])
                st.scatter_chart(rel, x="mean_confidence", y="accuracy", size="count")
            else:
                st.write("Calibration not computed: %s" % pm.get("calibration_note", "no confidence values"))
            st.markdown("Incorrect predictions and false approvals")
            wrong = [x for x in rep.records if x.provider_labels.get(prov) != x.expected_label
                     or (x.outcome.value == "APPROVE" and x.expected_outcome.value != "APPROVE")]
            st.dataframe(pd.DataFrame([{"example": x.example_id, "tags": ",".join(x.tags),
                                        "expected label": x.expected_label, "predicted": x.provider_labels.get(prov),
                                        "confidence": x.provider_confidence.get(prov),
                                        "expected outcome": x.expected_outcome.value, "outcome": x.outcome.value,
                                        "false approval": x.outcome.value == "APPROVE"
                                        and x.expected_outcome.value != "APPROVE"} for x in wrong]),
                         hide_index=True)
            st.download_button("Markdown report", render_comparison(reports), file_name="confidence_lab.md",
                               key="lab_dl")
            for n in rep.notes:
                st.caption(n)


# ---------------------------------------------------------------- human review
with tabs[3]:
    reviewers = [r.reviewer_id for r in settings.reviewers]
    c1, c2 = st.columns(2)
    who = c1.selectbox("Reviewer", reviewers or ["(none configured)"], key="hr_who")
    key = c2.text_input("Reviewer API key", type="password", key="hr_key")
    principal = next((p for p in settings.principals if p.reviewer_id == who
                      and key and hmac.compare_digest(p.key_sha256, hash_key(key))), None)
    pending = store.list_reviews(state="PENDING", limit=200)
    if not pending:
        st.info("No pending reviews.")
    for rec in pending:
        d = store.get_decision(rec.decision_id)
        with st.expander("%s · %s on %s · requires %s" % (rec.review_id[-8:], rec.original_action.action_type,
                                                         rec.original_action.target_resource,
                                                         rec.required_level.value)):
            st.write("Reasons:", rec.reason_codes)
            st.json(sanitize(rec.original_action.model_dump(mode="json")))
            if d and d.validation:
                st.write("Findings:", [f.code for f in d.validation.findings] or "none")
            st.write("Evidence:", [e.model_dump() for e in rec.original_action.evidence_refs] or "none")
            st.write("History:", [h.model_dump(mode="json") for h in rec.history])
            reason = st.text_input("Reason", key="r_" + rec.review_id)
            changes = st.text_area("Override changes (JSON: action_payload / target_resource / context)", "{}",
                                   key="c_" + rec.review_id)
            b1, b2, b3 = st.columns(3)
            if principal is None:
                st.warning("Enter the selected reviewer's API key to act on reviews.")
            else:
                try:
                    if b1.button("Approve", key="a_" + rec.review_id):
                        svc.approve_review(rec.review_id, who, reason or None)
                        st.success("Approved; authorization issued.")
                    if b2.button("Reject", key="j_" + rec.review_id):
                        svc.reject_review(rec.review_id, who, reason)
                        st.success("Rejected.")
                    if b3.button("Override", key="o_" + rec.review_id):
                        out = svc.override_review(rec.review_id, who, reason, json.loads(changes))
                        st.success("Override revalidated: new decision %s (%s)."
                                   % (out.decision.decision_id, out.decision.status.value))
                except ReviewError as exc:
                    st.error("%s: %s" % (exc.code, exc))
                except json.JSONDecodeError as exc:
                    st.error("Override changes are not valid JSON: %s" % exc)
    closed = [r for r in store.list_reviews(limit=200) if r.state.value != "PENDING"]
    if closed:
        st.subheader("Completed reviews")
        st.dataframe(pd.DataFrame([{"review": r.review_id, "state": r.state.value, "reviewer": r.reviewer_id,
                                    "reason": r.reviewer_reason, "decided": r.decided_at} for r in closed]),
                     hide_index=True)


# ---------------------------------------------------------------- audit explorer
with tabs[4]:
    c1, c2, c3 = st.columns(3)
    corr = c1.text_input("Correlation id", key="au_corr")
    etype = c2.text_input("Event type", key="au_type")
    limit = c3.number_input("Limit", 10, 5000, 500, key="au_limit")
    events = store.list_events(correlation_id=corr or None, event_type=etype or None, limit=int(limit))
    if not events:
        st.info("No audit events match.")
    else:
        st.dataframe(pd.DataFrame([{"seq": e.seq, "time": e.timestamp.isoformat(timespec="seconds"),
                                    "type": e.event_type, "component": e.component, "actor": e.actor,
                                    "correlation": e.correlation_id, "reasons": ", ".join(e.reason_codes)}
                                   for e in events]), hide_index=True)
        out = settings.path("var/audit_export.jsonl")
        if st.button("Export sanitized JSONL", key="au_export"):
            n = export_jsonl(events, out)
            st.success("Exported %d events to %s" % (n, out))


# ---------------------------------------------------------------- configuration
with tabs[5]:
    st.write("Environment: `%s`" % settings.environment)
    st.write("Active providers: %s · strategy: `%s`" % (settings.active_providers, settings.strategy))
    st.json(settings.governance.model_dump(mode="json"))
    st.write("Rule packs:", svc.validation.pack_refs)
    st.write("Calibration profiles:", settings.calibration_profiles or "none (calibration unknown for all providers)")
    st.write("Datasets:", datasets or "none")
    st.write("Reviewers:", [r.model_dump() for r in settings.reviewers])
